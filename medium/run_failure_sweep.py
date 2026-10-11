"""
Inference-only sensor-failure sweep on the cached RBF-SVMs (no retraining).

Uses the shared sensor_failure_patterns.json (k = 0, 1, 2, 4, 8) and
shared_robustness.evaluate_failure_patterns in raw_mean mode: failed sensors'
raw columns are replaced by the outer-train means, then the cached feature
pipeline and model are applied.

Outputs (variant 'base' -> algorithm 'rbf_svm'; 'dropout' -> 'rbf_svm_dropout')
    results/medium[_dropout]_failure_sweep.csv   one row per (repeat, k, pattern, test_batch)
    results/medium[_dropout]_confusion.csv       long-format confusion counts (test_batch = all)
    results/medium[_dropout]_robustness_summary.csv   shared_robustness.summarise_robustness

    python -m medium.run_failure_sweep [--variant base|dropout] [--smoke] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import time

import pandas as pd
from joblib import Parallel, delayed

from medium.common import load_bundles, load_data, load_patterns, results_path, sweep_with_breakdown
from medium.model import ALGORITHM
from shared_robustness import summarise_robustness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["base", "dropout"], default="base")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    _, X_raw, y, batch, splits = load_data()
    test_by_repeat = {s["repeat"]: s["test_indices"] for s in splits}
    patterns = load_patterns()
    algorithm = ALGORITHM if args.variant == "base" else f"{ALGORITHM}_dropout"
    bundles = load_bundles(args.variant, args.smoke)

    t0 = time.time()
    outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(sweep_with_breakdown)(
            b["model"], b["pipeline"], b["train_mean"], X_raw[test_by_repeat[b["repeat"]]],
            y[test_by_repeat[b["repeat"]]], batch[test_by_repeat[b["repeat"]]],
            patterns, b["repeat"], algorithm)
        for b in bundles
    )
    wall = time.time() - t0

    results = pd.concat([o[0] for o in outputs], ignore_index=True)
    confusion = pd.concat([o[1] for o in outputs], ignore_index=True)
    prefix = "medium" if args.variant == "base" else "medium_dropout"
    results.to_csv(results_path(f"{prefix}_failure_sweep", args.smoke), index=False)
    confusion.to_csv(results_path(f"{prefix}_confusion", args.smoke), index=False)

    _, summary = summarise_robustness(results[results.test_batch == "all"])
    summary.insert(0, "algorithm", algorithm)
    summary.to_csv(results_path(f"{prefix}_robustness_summary", args.smoke), index=False)
    print(summary.round(3).to_string(index=False))
    n_preds = (results.test_batch == "all").sum()
    print(f"\nWall-clock: {wall:.1f}s for {n_preds} predictions "
          f"({len(bundles)} repeats, n_jobs={args.n_jobs})")


if __name__ == "__main__":
    main()
