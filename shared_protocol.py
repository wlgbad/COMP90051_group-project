"""
Common experiment protocol for all three models.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

SEED = 90051
N_CLASSES = 6
N_SENSORS = 16
FEATURES_PER_SENSOR = 8
RAW_FEATURES = [f"x{i}" for i in range(1, 129)]

INNER_TEMPORAL_SPLITS = [
    {"inner_split": 1, "train_batches": [1, 2], "val_batches": [3, 4, 5, 6]},
    {"inner_split": 2, "train_batches": [1, 2, 3, 4, 5], "val_batches": [6]},
    {"inner_split": 3, "train_batches": [1, 2, 3, 4, 5, 6], "val_batches": [7]},
]


def load_phase1(base_dir: str | Path = ".") -> tuple[pd.DataFrame, list[dict]]:
    base_dir = Path(base_dir)
    phase1_dir = base_dir / "phase1_output"
    data_path = phase1_dir / "gas_sensor_clean.csv"
    split_path = phase1_dir / "temporal_outer_splits.json"

    if not data_path.exists():
        raise FileNotFoundError(f"Cannot find {data_path}")
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot find {split_path}")

    df = pd.read_csv(data_path)
    with split_path.open("r", encoding="utf-8") as f:
        outer_splits = json.load(f)

    validate_phase1(df, outer_splits)
    return df, outer_splits


def validate_phase1(df: pd.DataFrame, outer_splits: list[dict]) -> None:
    missing = [c for c in RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing raw features: {missing[:10]}")

    for required in ["batch", "target"]:
        if required not in df.columns:
            raise ValueError(f"Missing required column: {required}")

    repeats = sorted(int(s["repeat"]) for s in outer_splits)
    if repeats != list(range(1, 11)):
        raise ValueError(f"Expected outer repeats 1..10, found {repeats}")

    for split in outer_splits:
        train_idx = np.asarray(split["train_indices"], dtype=np.int64)
        test_idx = np.asarray(split["test_indices"], dtype=np.int64)

        if np.intersect1d(train_idx, test_idx).size:
            raise ValueError(f"Repeat {split['repeat']} has train/test overlap.")

        train_max_batch = int(df.loc[train_idx, "batch"].max())
        test_min_batch = int(df.loc[test_idx, "batch"].min())

        if train_max_batch >= test_min_batch:
            raise ValueError(
                f"Repeat {split['repeat']} is not temporally ordered."
            )


def get_outer_split(outer_splits: list[dict], repeat: int) -> dict:
    matches = [s for s in outer_splits if int(s["repeat"]) == int(repeat)]
    if len(matches) != 1:
        raise ValueError(f"Expected one split for repeat {repeat}, found {len(matches)}")
    return matches[0]


def make_inner_temporal_splits(
    df: pd.DataFrame,
    outer_train_indices: Iterable[int],
) -> list[dict]:
    outer_train_indices = np.asarray(list(outer_train_indices), dtype=np.int64)
    in_outer_train = df.index.isin(outer_train_indices)

    results = []

    for spec in INNER_TEMPORAL_SPLITS:
        train_idx = df.index[
            in_outer_train & df["batch"].isin(spec["train_batches"])
        ].to_numpy(dtype=np.int64)

        val_idx = df.index[
            in_outer_train & df["batch"].isin(spec["val_batches"])
        ].to_numpy(dtype=np.int64)

        if len(train_idx) == 0 or len(val_idx) == 0:
            raise ValueError(f"Inner split {spec['inner_split']} is empty.")

        if int(df.loc[train_idx, "batch"].max()) >= int(df.loc[val_idx, "batch"].min()):
            raise ValueError(f"Inner split {spec['inner_split']} is not temporally ordered.")

        if np.intersect1d(train_idx, val_idx).size:
            raise ValueError(f"Inner split {spec['inner_split']} overlaps.")

        results.append({
            "inner_split": int(spec["inner_split"]),
            "train_indices": train_idx,
            "val_indices": val_idx,
            "train_batches": list(spec["train_batches"]),
            "val_batches": list(spec["val_batches"]),
        })

    return results


def sensor_feature_names(sensor_id: int) -> list[str]:
    if not 1 <= int(sensor_id) <= N_SENSORS:
        raise ValueError(f"sensor_id must be 1..{N_SENSORS}")

    start = (int(sensor_id) - 1) * FEATURES_PER_SENSOR + 1
    return [f"x{i}" for i in range(start, start + FEATURES_PER_SENSOR)]


def sensor_feature_positions(
    sensor_id: int,
    feature_columns: list[str] | None = None,
) -> list[int]:
    if feature_columns is None:
        feature_columns = RAW_FEATURES

    lookup = {name: pos for pos, name in enumerate(feature_columns)}
    positions = []

    for name in sensor_feature_names(sensor_id):
        if name not in lookup:
            raise ValueError(f"{name} not present in feature_columns")
        positions.append(lookup[name])

    return positions


class TrainOnlyStandardScaler:
    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "TrainOnlyStandardScaler":
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        scale = X.std(axis=0, ddof=0)
        self.scale_ = np.where(scale == 0.0, 1.0, scale)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("Scaler must be fitted first.")
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.mean_) / self.scale_).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
