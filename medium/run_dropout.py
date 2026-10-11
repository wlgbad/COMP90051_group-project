"""
Mitigation: train the RBF-SVM with sensor-dropout augmentation.

For each outer repeat the training set is the clean outer-train sample plus one
augmented copy of it in which every row has k ~ Uniform{1..MAX_DROP} random
sensors replaced by the outer-train raw means (the same imputation used at test
time). The feature pipeline is fitted on the clean outer-train sample only,
exactly as for the base model, and (C, gamma_mult) are reused from the base
model's nested-CV choice for that repeat, so the only difference is the
training data. One model per repeat; no per-k retraining. Dropout masks are
drawn independently of the shared test failure patterns.

Outputs
    cache/medium/dropout/repeat_XX.joblib   (then run run_failure_sweep --variant dropout)

    python -m medium.run_dropout [--smoke] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from joblib import Parallel, delayed

from medium.common import load_bundles, load_data, save_bundle
from medium.model import make_model
from shared_features import FeaturePipeline
from shared_protocol import N_SENSORS, SEED
from shared_robustness import failed_feature_positions

MAX_DROP = 8


def augment(X_raw_train: np.ndarray, train_mean: np.ndarray,
            rng: np.random.Generator) -> np.ndarray:
    """One copy of the training rows with a random sensor subset failed per row."""
    X_aug = X_raw_train.copy()
    for i in range(len(X_aug)):
        k = int(rng.integers(1, MAX_DROP + 1))
        failed = rng.choice(np.arange(1, N_SENSORS + 1), size=k, replace=False)
        cols = failed_feature_positions(tuple(int(s) for s in failed))
        X_aug[i, cols] = train_mean[cols]
    return X_aug


def train_repeat(base: dict, split: dict, X_raw, y) -> dict:
    t0 = time.time()
    tr = split["train_indices"]
    pipe = FeaturePipeline().fit(X_raw[tr])
    train_mean = X_raw[tr].mean(axis=0)
    rng = np.random.default_rng([SEED, 1, split["repeat"]])
    X_train = np.vstack([X_raw[tr], augment(X_raw[tr], train_mean, rng)])
    y_train = np.concatenate([y[tr], y[tr]])
    model = make_model(base["params"], pipe.transform(X_raw[tr])).fit(
        pipe.transform(X_train), y_train)
    print(f"repeat {split['repeat']:2d}: {len(y_train)} rows, "
          f"{int(model.n_support_.sum())} SVs, {time.time() - t0:.1f}s", flush=True)
    return {"repeat": split["repeat"], "params": base["params"], "pipeline": pipe,
            "model": model, "train_mean": train_mean}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    _, X_raw, y, _, splits = load_data()
    split_by_repeat = {s["repeat"]: s for s in splits}
    bases = load_bundles("base", args.smoke)

    t0 = time.time()
    bundles = Parallel(n_jobs=args.n_jobs)(
        delayed(train_repeat)(b, split_by_repeat[b["repeat"]], X_raw, y) for b in bases)
    for bundle in bundles:
        save_bundle(bundle, "dropout", args.smoke)
    print(f"Wall-clock: {time.time() - t0:.1f}s ({len(bundles)} repeats)")


if __name__ == "__main__":
    main()
