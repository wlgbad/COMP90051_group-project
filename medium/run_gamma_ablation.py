"""
Mechanism check for the RBF-SVM robustness story: does a larger gamma make the
model more brittle under sensor failure?

For each outer repeat, keep that repeat's tuned C, refit on the full outer-train
sample for every gamma_mult in GAMMAS, and evaluate every shared failure pattern.
Besides accuracy / macro-F1 it records
    displacement  mean ||z_failed - z_clean||^2 of the test points in feature space
    max_kernel    mean over test points of max_j k(x, sv_j) = exp(-gamma * min_j ||x - sv_j||^2)
which show how imputation moves points away from the support vectors and how
quickly kernel similarity decays with gamma.

This is a diagnostic ablation, not model selection: the outer test is used only
to describe the behaviour of already-specified models.

Outputs
    results/medium_gamma_ablation.csv

    python -m medium.run_gamma_ablation [--smoke] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from medium.common import load_bundles, load_data, load_patterns, results_path
from medium.model import ALGORITHM, gamma_scale, make_model
from shared_features import FeaturePipeline
from shared_metrics import classification_metrics_from_scratch
from shared_robustness import corrupt_raw_with_outer_train_mean

GAMMAS = (0.03, 0.1, 0.3, 1.0, 3.0)


def sq_dist_min(Z: np.ndarray, SV: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """min_j ||z_i - sv_j||^2 for every row of Z."""
    Z, SV = Z.astype(np.float64), SV.astype(np.float64)
    sv_sq = (SV**2).sum(axis=1)
    out = np.empty(len(Z))
    for s in range(0, len(Z), chunk):
        block = Z[s:s + chunk]
        d = (block**2).sum(axis=1)[:, None] + sv_sq[None, :] - 2 * block @ SV.T
        out[s:s + chunk] = np.maximum(d, 0).min(axis=1)
    return out


def ablate_repeat(C: float, split: dict, X_raw, y, patterns, gammas) -> list[dict]:
    tr, te = split["train_indices"], split["test_indices"]
    pipe = FeaturePipeline()
    Z_tr = pipe.fit_transform(X_raw[tr])
    Z_clean = pipe.transform(X_raw[te])
    train_mean = X_raw[tr].mean(axis=0)
    Z_failed = {(k, pid): pipe.transform(corrupt_raw_with_outer_train_mean(X_raw[te], failed, train_mean))
                for k, pats in patterns.items() for pid, failed in enumerate(pats, start=1)}

    rows = []
    for g in gammas:
        model = make_model({"C": C, "gamma_mult": g}, Z_tr).fit(Z_tr, y[tr])
        gamma = g * gamma_scale(Z_tr)
        SV = model.support_vectors_
        for (k, pattern_id), Z in Z_failed.items():
            metrics, _ = classification_metrics_from_scratch(y[te], model.predict(Z))
            rows.append({"algorithm": ALGORITHM, "outer_repeat": split["repeat"], "C": C,
                         "gamma_mult": g, "gamma": gamma, "n_support_vectors": len(SV),
                         "n_failed_sensors": k, "pattern_id": pattern_id,
                         "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
                         "displacement": float(((Z - Z_clean)**2).sum(axis=1).mean()),
                         "max_kernel": float(np.exp(-gamma * sq_dist_min(Z, SV)).mean())})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    _, X_raw, y, _, splits = load_data()
    split_by_repeat = {s["repeat"]: s for s in splits}
    patterns = load_patterns()
    gammas = GAMMAS
    if args.smoke:
        patterns, gammas = {k: v[:2] for k, v in patterns.items()}, (0.1, 1.0)
    bases = load_bundles("base", args.smoke)

    t0 = time.time()
    outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(ablate_repeat)(b["params"]["C"], split_by_repeat[b["repeat"]], X_raw, y,
                               patterns, gammas)
        for b in bases)
    df = pd.DataFrame([r for out in outputs for r in out])
    df.to_csv(results_path("medium_gamma_ablation", args.smoke), index=False)

    per_repeat = df.groupby(["gamma_mult", "n_failed_sensors", "outer_repeat"])[
        ["macro_f1", "max_kernel"]].mean()
    summary = per_repeat.groupby(["gamma_mult", "n_failed_sensors"]).mean()
    print(summary.unstack("gamma_mult").round(3).to_string())
    print(f"\nWall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
