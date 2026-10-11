"""
Nested temporal CV for the RBF-SVM on the full 16-sensor array.

Outer: the 10 Phase 1 splits. Inner: shared_protocol.make_inner_temporal_splits
(1-2 -> 3-6, 1-5 -> 6, 1-6 -> 7) inside each outer-train sample. The feature
pipeline is refitted on every inner-train set. The grid point with the highest
mean inner macro-F1 is refitted on the full outer-train sample and cached.

Outputs
    cache/medium/base/repeat_XX.joblib
    results/medium_tuning_log.csv     algorithm, outer_repeat, hparam, chosen_value, inner_mean_score
    results/medium_inner_scores.csv   every (outer repeat, grid point, inner split) score
    results/medium_k0_test.csv        clean outer-test metrics + wall-clock per repeat

    python -m medium.run_tuning [--smoke] [--n-jobs N]
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from medium.common import load_data, results_path, save_bundle
from medium.model import ALGORITHM, GRID, SMOKE_GRID, make_model
from shared_features import FeaturePipeline
from shared_metrics import classification_metrics_from_scratch
from shared_protocol import make_inner_temporal_splits


def grid_points(grid: dict[str, list]) -> list[dict]:
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def tune(df, X_raw, y, outer_train, grid) -> tuple[dict, float, list[dict]]:
    """Grid search on the shared inner temporal splits; ties go to the earlier grid point."""
    points = grid_points(grid)
    inner = make_inner_temporal_splits(df, outer_train)
    scores = np.zeros((len(points), len(inner)))
    rows = []
    for s, split in enumerate(inner):
        tr, va = split["train_indices"], split["val_indices"]
        pipe = FeaturePipeline()
        Z_tr = pipe.fit_transform(X_raw[tr])
        Z_va = pipe.transform(X_raw[va])
        for p, params in enumerate(points):
            model = make_model(params, Z_tr).fit(Z_tr, y[tr])
            metrics, _ = classification_metrics_from_scratch(y[va], model.predict(Z_va))
            scores[p, s] = metrics["macro_f1"]
            rows.append({**params, "inner_split": split["inner_split"],
                         "score": scores[p, s]})
    mean_scores = scores.mean(axis=1)
    best = int(np.argmax(mean_scores))
    return points[best], float(mean_scores[best]), rows


def run_outer_repeat(df, X_raw, y, split, grid) -> dict:
    t0 = time.time()
    tr, te = split["train_indices"], split["test_indices"]
    best, best_score, inner_rows = tune(df, X_raw, y, tr, grid)

    pipe = FeaturePipeline()
    Z_tr = pipe.fit_transform(X_raw[tr])
    model = make_model(best, Z_tr).fit(Z_tr, y[tr])
    test_k0, _ = classification_metrics_from_scratch(y[te], model.predict(pipe.transform(X_raw[te])))

    repeat = split["repeat"]
    for row in inner_rows:
        row.update(algorithm=ALGORITHM, outer_repeat=repeat)
    return {
        "bundle": {"repeat": repeat, "params": best, "pipeline": pipe, "model": model,
                   "train_mean": X_raw[tr].mean(axis=0)},
        "tuning_log": [{"algorithm": ALGORITHM, "outer_repeat": repeat, "hparam": h,
                        "chosen_value": v, "inner_mean_score": best_score}
                       for h, v in best.items()],
        "inner_rows": inner_rows,
        "k0": {"algorithm": ALGORITHM, "outer_repeat": repeat, **best,
               "n_support_vectors": int(model.n_support_.sum()), **test_k0,
               "seconds": time.time() - t0},
    }


def middle_value_check(log: pd.DataFrame, grid: dict) -> None:
    print("\nChosen-value frequency per hyperparameter (middle value should win):")
    for hparam, values in grid.items():
        counts = log[log["hparam"] == hparam]["chosen_value"].value_counts()
        freq = {v: int(counts.get(v, 0)) for v in values}
        middle = values[len(values) // 2]
        top = max(freq, key=freq.get)
        status = "OK" if top == middle else f"EDGE VALUE {top} WINS - shift grid"
        print(f"  {hparam:10s} {freq}  -> {status}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="1 repeat, tiny grid")
    parser.add_argument("--n-jobs", type=int, default=10)
    args = parser.parse_args()

    df, X_raw, y, _, splits = load_data()
    grid = SMOKE_GRID if args.smoke else GRID
    if args.smoke:
        splits = splits[:1]

    t0 = time.time()
    outputs = Parallel(n_jobs=args.n_jobs)(
        delayed(run_outer_repeat)(df, X_raw, y, split, grid) for split in splits)
    wall = time.time() - t0

    for out in outputs:
        save_bundle(out["bundle"], "base", args.smoke)
    log = pd.DataFrame([r for out in outputs for r in out["tuning_log"]])
    log.to_csv(results_path("medium_tuning_log", args.smoke), index=False)
    pd.DataFrame([r for out in outputs for r in out["inner_rows"]]).to_csv(
        results_path("medium_inner_scores", args.smoke), index=False)
    k0 = pd.DataFrame([out["k0"] for out in outputs])
    k0.to_csv(results_path("medium_k0_test", args.smoke), index=False)

    print(k0[["outer_repeat", "C", "gamma_mult", "n_support_vectors", "accuracy",
              "macro_f1", "seconds"]].to_string(index=False))
    print(f"\nClean test accuracy {k0.accuracy.mean():.3f} +/- {k0.accuracy.std():.3f}, "
          f"macro-F1 {k0.macro_f1.mean():.3f} +/- {k0.macro_f1.std():.3f}")
    middle_value_check(log, grid)
    print(f"\nWall-clock: {wall:.1f}s ({len(splits)} repeats, n_jobs={args.n_jobs})")


if __name__ == "__main__":
    main()
