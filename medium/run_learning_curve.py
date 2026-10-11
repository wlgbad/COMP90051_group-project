"""
Learning curve: performance vs number of training samples, clean and with failures.

For each outer repeat and training fraction f, draw a class-stratified fraction f
of the outer-train sample, refit pipeline + RBF-SVM with that repeat's tuned
(C, gamma_mult), and score the outer test on every shared failure pattern for
k in LC_KS (raw_mean imputation with that subsample's means).

Outputs
    results/medium_learning_curve.csv
        algorithm, outer_repeat, train_fraction, n_train, n_failed_sensors,
        pattern_id, accuracy, macro_f1

    python -m medium.run_learning_curve [--smoke] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from medium.common import load_bundles, load_data, load_patterns, results_path, stratified_subsample
from medium.model import ALGORITHM, make_model
from shared_features import FeaturePipeline
from shared_metrics import classification_metrics_from_scratch
from shared_protocol import SEED
from shared_robustness import corrupt_raw_with_outer_train_mean

FRACTIONS = (0.1, 0.25, 0.5, 0.75, 1.0)
LC_KS = (0, 4, 8)


def curve_repeat(params: dict, split: dict, X_raw, y, patterns, fractions) -> list[dict]:
    tr_full, te = split["train_indices"], split["test_indices"]
    rows = []
    for f_i, fraction in enumerate(fractions):
        rng = np.random.default_rng([SEED, 2, split["repeat"], f_i])
        tr = tr_full if fraction == 1.0 else stratified_subsample(tr_full, y, fraction, rng)
        pipe = FeaturePipeline()
        Z_tr = pipe.fit_transform(X_raw[tr])
        model = make_model(params, Z_tr).fit(Z_tr, y[tr])
        train_mean = X_raw[tr].mean(axis=0)
        for k in LC_KS:
            for pattern_id, failed in enumerate(patterns[k], start=1):
                X_te = corrupt_raw_with_outer_train_mean(X_raw[te], failed, train_mean)
                metrics, _ = classification_metrics_from_scratch(
                    y[te], model.predict(pipe.transform(X_te)))
                rows.append({"algorithm": ALGORITHM, "outer_repeat": split["repeat"],
                             "train_fraction": fraction, "n_train": len(tr),
                             "n_failed_sensors": k, "pattern_id": pattern_id,
                             "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    _, X_raw, y, _, splits = load_data()
    split_by_repeat = {s["repeat"]: s for s in splits}
    patterns = load_patterns()
    if args.smoke:
        patterns = {k: v[:3] for k, v in patterns.items()}
    fractions = (0.25, 1.0) if args.smoke else FRACTIONS
    bases = load_bundles("base", args.smoke)

    t0 = time.time()
    outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(curve_repeat)(b["params"], split_by_repeat[b["repeat"]], X_raw, y,
                              patterns, fractions)
        for b in bases)
    df = pd.DataFrame([r for out in outputs for r in out])
    df.to_csv(results_path("medium_learning_curve", args.smoke), index=False)

    per_repeat = df.groupby(["n_failed_sensors", "train_fraction", "outer_repeat"]).macro_f1.mean()
    print(per_repeat.groupby(["n_failed_sensors", "train_fraction"])
          .agg(["mean", "std"]).round(3).to_string())
    print(f"\nWall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
