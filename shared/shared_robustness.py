"""
Common sensor-failure robustness utilities.

Every model must load the SAME sensor_failure_patterns.json from Phase 6B.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from shared_protocol import (
    N_CLASSES,
    N_SENSORS,
    RAW_FEATURES,
    sensor_feature_positions,
)
from shared_metrics import classification_metrics_from_scratch

EXPECTED_FAILURE_LEVELS = [0, 1, 2, 4, 8]


def load_failure_patterns(
    json_path: str | Path,
) -> dict[int, list[tuple[int, ...]]]:
    json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_patterns = payload.get("failure_patterns")
    if raw_patterns is None:
        raise ValueError("JSON does not contain 'failure_patterns'.")

    patterns = {
        int(k): [tuple(int(s) for s in pattern) for pattern in pattern_list]
        for k, pattern_list in raw_patterns.items()
    }

    if sorted(patterns.keys()) != EXPECTED_FAILURE_LEVELS:
        raise ValueError(
            f"Expected failure levels {EXPECTED_FAILURE_LEVELS}; "
            f"found {sorted(patterns.keys())}"
        )

    expected_counts = {0: 1, 1: 16, 2: 20, 4: 20, 8: 20}

    for k, expected_count in expected_counts.items():
        if len(patterns[k]) != expected_count:
            raise ValueError(
                f"k={k}: expected {expected_count} patterns, "
                f"found {len(patterns[k])}"
            )

        for pattern in patterns[k]:
            if len(pattern) != k:
                raise ValueError(f"k={k}: invalid pattern {pattern}")

            if len(set(pattern)) != len(pattern):
                raise ValueError(f"k={k}: duplicate sensor in {pattern}")

            if not all(1 <= sensor_id <= N_SENSORS for sensor_id in pattern):
                raise ValueError(f"k={k}: invalid sensor ID in {pattern}")

    return patterns


def failed_feature_positions(
    failed_sensors: tuple[int, ...],
    feature_columns: list[str] | None = None,
) -> list[int]:
    if feature_columns is None:
        feature_columns = RAW_FEATURES

    positions = []
    for sensor_id in failed_sensors:
        positions.extend(
            sensor_feature_positions(sensor_id, feature_columns)
        )

    return sorted(positions)


def corrupt_raw_with_outer_train_mean(
    X_test_raw: np.ndarray,
    failed_sensors: tuple[int, ...],
    outer_train_feature_mean: np.ndarray,
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    if feature_columns is None:
        feature_columns = RAW_FEATURES

    X_corrupted = np.asarray(X_test_raw, dtype=np.float64).copy()
    means = np.asarray(outer_train_feature_mean, dtype=np.float64)

    if X_corrupted.shape[1] != len(feature_columns):
        raise ValueError("X_test_raw width does not match feature_columns.")

    if len(means) != len(feature_columns):
        raise ValueError("outer_train_feature_mean width mismatch.")

    positions = failed_feature_positions(failed_sensors, feature_columns)

    if positions:
        X_corrupted[:, positions] = means[positions]

    return X_corrupted


def corrupt_standardised_with_zero(
    X_test_scaled: np.ndarray,
    failed_sensors: tuple[int, ...],
    feature_columns: list[str] | None = None,
) -> np.ndarray:
    if feature_columns is None:
        feature_columns = RAW_FEATURES

    X_corrupted = np.asarray(X_test_scaled, dtype=np.float32).copy()

    if X_corrupted.shape[1] != len(feature_columns):
        raise ValueError("X_test_scaled width does not match feature_columns.")

    positions = failed_feature_positions(failed_sensors, feature_columns)

    if positions:
        X_corrupted[:, positions] = 0.0

    return X_corrupted


def evaluate_failure_patterns(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    X_test: np.ndarray,
    y_test: np.ndarray,
    patterns: dict[int, list[tuple[int, ...]]],
    corruption_mode: str,
    outer_repeat: int,
    outer_train_feature_mean: np.ndarray | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    if feature_columns is None:
        feature_columns = RAW_FEATURES

    rows = []
    clean_macro_f1 = None

    for k in EXPECTED_FAILURE_LEVELS:
        for pattern_id, failed_sensors in enumerate(patterns[k], start=1):

            if corruption_mode == "raw_mean":
                if outer_train_feature_mean is None:
                    raise ValueError(
                        "outer_train_feature_mean is required for raw_mean."
                    )

                X_corrupted = corrupt_raw_with_outer_train_mean(
                    X_test,
                    failed_sensors,
                    outer_train_feature_mean,
                    feature_columns,
                )

            elif corruption_mode == "standardised_zero":
                X_corrupted = corrupt_standardised_with_zero(
                    X_test,
                    failed_sensors,
                    feature_columns,
                )

            else:
                raise ValueError(
                    "corruption_mode must be 'raw_mean' "
                    "or 'standardised_zero'."
                )

            y_pred = np.asarray(predict_fn(X_corrupted), dtype=np.int64)

            metrics, _ = classification_metrics_from_scratch(
                y_test,
                y_pred,
                N_CLASSES,
            )

            if k == 0:
                clean_macro_f1 = metrics["macro_f1"]

            rows.append({
                "outer_repeat": int(outer_repeat),
                "n_failed_sensors": int(k),
                "n_available_sensors": int(N_SENSORS - k),
                "pattern_id": int(pattern_id),
                "failed_sensors": ",".join(map(str, failed_sensors)),
                **metrics,
            })

    if clean_macro_f1 is None:
        raise RuntimeError("Clean k=0 condition was not evaluated.")

    result = pd.DataFrame(rows)

    result["macro_f1_drop_from_clean"] = (
        clean_macro_f1 - result["macro_f1"]
    )

    result["macro_f1_retained_fraction"] = (
        result["macro_f1"] / clean_macro_f1
        if clean_macro_f1 > 0
        else np.nan
    )

    return result


def summarise_robustness(
    pattern_results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_repeat = (
        pattern_results
        .groupby(
            ["outer_repeat", "n_failed_sensors"],
            as_index=False,
        )
        .agg(
            n_patterns=("macro_f1", "size"),
            macro_f1_mean_across_patterns=("macro_f1", "mean"),
            macro_f1_sd_across_patterns=("macro_f1", "std"),
            accuracy_mean_across_patterns=("accuracy", "mean"),
        )
    )

    by_repeat["macro_f1_sd_across_patterns"] = (
        by_repeat["macro_f1_sd_across_patterns"].fillna(0.0)
    )

    final_rows = []

    for k in sorted(by_repeat["n_failed_sensors"].unique()):
        subset = by_repeat[by_repeat["n_failed_sensors"] == k]

        macro_values = subset[
            "macro_f1_mean_across_patterns"
        ].to_numpy(dtype=float)

        accuracy_values = subset[
            "accuracy_mean_across_patterns"
        ].to_numpy(dtype=float)

        final_rows.append({
            "n_failed_sensors": int(k),
            "n_outer_repeats": int(len(subset)),
            "macro_f1_mean": float(macro_values.mean()),
            "macro_f1_sd_across_outer_repeats": (
                float(macro_values.std(ddof=1))
                if len(macro_values) > 1
                else 0.0
            ),
            "accuracy_mean": float(accuracy_values.mean()),
            "accuracy_sd_across_outer_repeats": (
                float(accuracy_values.std(ddof=1))
                if len(accuracy_values) > 1
                else 0.0
            ),
            "mean_within_repeat_macro_f1_sd_across_patterns": float(
                subset["macro_f1_sd_across_patterns"].mean()
            ),
        })

    final_summary = pd.DataFrame(final_rows)

    clean = final_summary.loc[
        final_summary["n_failed_sensors"] == 0,
        "macro_f1_mean",
    ]

    if len(clean) != 1:
        raise ValueError("Expected exactly one clean k=0 row.")

    clean_f1 = float(clean.iloc[0])

    final_summary["macro_f1_drop_mean"] = (
        clean_f1 - final_summary["macro_f1_mean"]
    )

    final_summary["retained_fraction_mean"] = (
        final_summary["macro_f1_mean"] / clean_f1
        if clean_f1 > 0
        else np.nan
    )

    return by_repeat, final_summary
