"""
Single shallow decision tree under the team shared protocol.

Pipeline (for each of the 10 outer temporal repeats):
  1. Inner temporal CV (3 splits) -> pick hyperparameters by mean Macro-F1
  2. Refit FeaturePipeline + DecisionTree on the full outer-train set
  3. Evaluate on the clean outer-test set
  4. Sensor-failure robustness (raw_mean imputation on RAW x1..x128,
     then FeaturePipeline -> derived features inherit the failure; no retraining)

Put this file in team_shared_code/ (same folder as shared_*.py).
"""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

from shared_protocol import (
    SEED,
    N_CLASSES,
    RAW_FEATURES,
    load_phase1,
    get_outer_split,
    make_inner_temporal_splits,
)
from shared_metrics import classification_metrics_from_scratch
from shared_robustness import (
    load_failure_patterns,
    evaluate_failure_patterns,
    summarise_robustness,
)
from shared_features import FeaturePipeline, feature_names

BASE = Path(__file__).resolve().parent
OUT_DIR = BASE / "tree_output"
OUT_DIR.mkdir(exist_ok=True)

FEATURE_NAMES = feature_names()  # 396 constructed features

# ---------------------------------------------------------------------------
# Hyperparameter grid: "shallow" = max_depth capped at 6
# 5 x 3 x 2 = 30 combos; a single tree is cheap, so a wider grid is fine
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "max_depth": [2, 3, 4, 5, 6],
    "min_samples_leaf": [1, 20, 50],
    "class_weight": [None, "balanced"],
}
FIXED_PARAMS = {
    "criterion": "gini",
    "random_state": SEED,
}


def param_combos() -> list[dict]:
    keys = list(PARAM_GRID.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*PARAM_GRID.values())]


def make_tree(params: dict) -> DecisionTreeClassifier:
    return DecisionTreeClassifier(**FIXED_PARAMS, **params)


class TreeWithFeatures:
    """FeaturePipeline + DecisionTree. Takes RAW x1..x128 in both fit and predict."""

    def fit(self, X_raw: np.ndarray, y: np.ndarray, params: dict) -> "TreeWithFeatures":
        self.pipe_ = FeaturePipeline()
        Z = self.pipe_.fit_transform(X_raw)
        self.tree_ = make_tree(params).fit(Z, y)
        return self

    def predict(self, X_raw: np.ndarray) -> np.ndarray:
        return self.tree_.predict(self.pipe_.transform(X_raw)).astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df, outer_splits = load_phase1(BASE)
    patterns = load_failure_patterns(BASE / "sensor_failure_patterns.json")

    X = df[RAW_FEATURES].to_numpy(dtype=np.float64)
    y_raw = df["target"].to_numpy(dtype=np.int64)

    # Labels must be 0..5 for shared_metrics
    label_offset = int(y_raw.min())
    y = y_raw - label_offset
    if sorted(np.unique(y)) != list(range(N_CLASSES)):
        raise ValueError(f"Unexpected target values: {sorted(np.unique(y_raw))}")
    print(f"Labels shifted by {label_offset} -> 0..{N_CLASSES - 1}")
    print(f"Constructed features: {len(FEATURE_NAMES)}")

    combos = param_combos()
    inner_rows, clean_rows, robustness_tables, importance_tables = [], [], [], []
    tree_texts = []

    for repeat in range(1, 11):
        t0 = time.time()
        split = get_outer_split(outer_splits, repeat)
        tr = np.asarray(split["train_indices"], dtype=np.int64)
        te = np.asarray(split["test_indices"], dtype=np.int64)
        inner_splits = make_inner_temporal_splits(df, tr)

        # ---- 1. Inner temporal CV ----------------------------------------
        # Features built once per inner split (pipeline fitted on inner-train only)
        inner_data = []
        for inner in inner_splits:
            itr, iva = inner["train_indices"], inner["val_indices"]
            pipe = FeaturePipeline()
            inner_data.append({
                "inner_split": inner["inner_split"],
                "Z_tr": pipe.fit_transform(X[itr]),
                "y_tr": y[itr],
                "Z_va": pipe.transform(X[iva]),
                "y_va": y[iva],
            })

        best_score, best_params = -np.inf, None
        for combo_id, params in enumerate(combos, start=1):
            scores = []
            for d in inner_data:
                tree = make_tree(params).fit(d["Z_tr"], d["y_tr"])
                metrics, _ = classification_metrics_from_scratch(
                    d["y_va"], tree.predict(d["Z_va"]), N_CLASSES
                )
                scores.append(metrics["macro_f1"])
                inner_rows.append({
                    "outer_repeat": repeat,
                    "combo_id": combo_id,
                    **{k: str(v) for k, v in params.items()},
                    "inner_split": d["inner_split"],
                    "val_macro_f1": metrics["macro_f1"],
                    "val_accuracy": metrics["accuracy"],
                })

            mean_score = float(np.mean(scores))
            if mean_score > best_score:
                best_score, best_params = mean_score, params

        # ---- 2. Refit pipeline + tree on full outer-train ----------------
        final_model = TreeWithFeatures().fit(X[tr], y[tr], best_params)

        # ---- 3. Clean outer-test evaluation -----------------------------
        clean_metrics, _ = classification_metrics_from_scratch(
            y[te], final_model.predict(X[te]), N_CLASSES
        )
        clean_rows.append({
            "outer_repeat": repeat,
            **{f"best_{k}": str(v) for k, v in best_params.items()},
            "inner_mean_macro_f1": best_score,
            "n_leaves": int(final_model.tree_.get_n_leaves()),
            "actual_depth": int(final_model.tree_.get_depth()),
            **clean_metrics,
        })

        # Interpretability: importance + readable tree rules
        importance_tables.append(pd.DataFrame({
            "outer_repeat": repeat,
            "feature": FEATURE_NAMES,
            "importance": final_model.tree_.feature_importances_,
        }))
        tree_texts.append(
            f"===== Outer repeat {repeat} | params {best_params} =====\n"
            + export_text(final_model.tree_, feature_names=FEATURE_NAMES)
        )

        # ---- 4. Sensor-failure robustness (no retraining) ---------------
        outer_train_mean = X[tr].mean(axis=0)
        rob = evaluate_failure_patterns(
            predict_fn=final_model.predict,
            X_test=X[te],
            y_test=y[te],
            patterns=patterns,
            corruption_mode="raw_mean",
            outer_repeat=repeat,
            outer_train_feature_mean=outer_train_mean,
        )
        robustness_tables.append(rob)

        print(
            f"Repeat {repeat:2d} | best {best_params} | inner F1 {best_score:.4f} | "
            f"test acc {clean_metrics['accuracy']:.4f}, F1 {clean_metrics['macro_f1']:.4f} | "
            f"{time.time() - t0:.0f}s"
        )

    # ---- Save results ----------------------------------------------------
    inner_df = pd.DataFrame(inner_rows)
    clean_df = pd.DataFrame(clean_rows)
    rob_df = pd.concat(robustness_tables, ignore_index=True)
    imp_df = pd.concat(importance_tables, ignore_index=True)
    rob_by_repeat, rob_summary = summarise_robustness(rob_df)

    inner_df.to_csv(OUT_DIR / "tree_inner_cv.csv", index=False)
    clean_df.to_csv(OUT_DIR / "tree_clean_results.csv", index=False)
    rob_df.to_csv(OUT_DIR / "tree_robustness_patterns.csv", index=False)
    rob_by_repeat.to_csv(OUT_DIR / "tree_robustness_by_repeat.csv", index=False)
    rob_summary.to_csv(OUT_DIR / "tree_robustness_summary.csv", index=False)
    imp_df.to_csv(OUT_DIR / "tree_feature_importance.csv", index=False)
    (OUT_DIR / "tree_rules.txt").write_text("\n\n".join(tree_texts), encoding="utf-8")

    print("\n=== Clean test (mean +/- SD over 10 outer repeats) ===")
    for m in ["accuracy", "macro_f1"]:
        print(f"{m:10s}: {clean_df[m].mean():.4f} +/- {clean_df[m].std(ddof=1):.4f}")

    print("\n=== Robustness summary ===")
    print(rob_summary[[
        "n_failed_sensors", "macro_f1_mean",
        "macro_f1_sd_across_outer_repeats", "retained_fraction_mean",
    ]].to_string(index=False))
    print(f"\nAll outputs saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
