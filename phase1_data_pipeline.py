
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


N_SENSOR_FEATURES = 128
FEATURE_COLUMNS = [f"x{i}" for i in range(1, N_SENSOR_FEATURES + 1)]


def _batch_number(path: Path) -> int:
    """Extract batch number from names such as 'batch10.dat'."""
    match = re.fullmatch(r"batch(\d+)\.dat", path.name)
    if match is None:
        raise ValueError(f"Unexpected batch filename: {path.name}")
    return int(match.group(1))


def parse_dat_line(line: str) -> tuple[int, float, np.ndarray]:
    """
    Parse one line of the gas sensor drift dataset.

    Expected format:
        label;concentration 1:value 2:value ... 128:value
    """
    fields = line.strip().split()
    if not fields:
        raise ValueError("Encountered an empty data line.")

    label_text, concentration_text = fields[0].split(";", maxsplit=1)
    label = int(label_text)
    concentration = float(concentration_text)

    x = np.full(N_SENSOR_FEATURES, np.nan, dtype=np.float64)

    for token in fields[1:]:
        index_text, value_text = token.split(":", maxsplit=1)
        feature_index = int(index_text)

        if not 1 <= feature_index <= N_SENSOR_FEATURES:
            raise ValueError(f"Feature index out of range: {feature_index}")

        if not np.isnan(x[feature_index - 1]):
            raise ValueError(f"Duplicate feature index: {feature_index}")

        x[feature_index - 1] = float(value_text)

    missing = np.flatnonzero(np.isnan(x)) + 1
    if len(missing) > 0:
        raise ValueError(
            "Missing sensor features in a row: "
            + ", ".join(map(str, missing.tolist()))
        )

    return label, concentration, x


def load_dataset(data_dir: str | Path) -> pd.DataFrame:
    """
    Read batch1.dat ... batch10.dat directly from a folder.

    Expected folder:
        gas+sensor+array+drift+dataset+at+different+concentrations/
            batch1.dat
            batch2.dat
            ...
            batch10.dat

    Output columns:
        row_id
        batch
        label_raw
        target
        concentration
        x1 ... x128
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Folder not found: {data_dir}")

    batch_files = sorted(
        data_dir.glob("batch*.dat"),
        key=_batch_number
    )

    if len(batch_files) != 10:
        raise ValueError(
            f"Expected 10 batch files, found {len(batch_files)}: "
            f"{[p.name for p in batch_files]}"
        )

    rows = []
    row_id = 0

    for file_path in batch_files:
        batch = _batch_number(file_path)

        with file_path.open("r", encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                try:
                    label, concentration, features = parse_dat_line(line)
                except Exception as exc:
                    raise ValueError(
                        f"Failed to parse {file_path.name}, line {line_number}"
                    ) from exc

                row = {
                    "row_id": row_id,
                    "batch": batch,
                    "label_raw": label,
                    "target": label - 1,
                    "concentration": concentration,
                }

                row.update(
                    {
                        column: value
                        for column, value in zip(FEATURE_COLUMNS, features)
                    }
                )

                rows.append(row)
                row_id += 1

    df = pd.DataFrame(rows)

    # Integrity checks
    expected_labels = set(range(1, 7))
    observed_labels = set(df["label_raw"].unique())

    if observed_labels != expected_labels:
        raise ValueError(
            f"Expected gas labels {expected_labels}, found {observed_labels}"
        )

    if df[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Missing values found in sensor features.")

    if not np.isfinite(df[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Non-finite sensor values found.")

    return df


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print basic dataset checks before modelling."""
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"Rows: {len(df):,}")
    print(f"Sensor features: {len(FEATURE_COLUMNS)}")
    print(f"Columns in cleaned table: {df.shape[1]}")
    print(f"Missing sensor values: {int(df[FEATURE_COLUMNS].isna().sum().sum())}")
    print()

    print("Samples per batch")
    print(df.groupby("batch").size().to_string())
    print()

    print("Class counts by batch")
    print(pd.crosstab(df["batch"], df["label_raw"]).to_string())
    print()

    print("Concentration range by gas class")
    print(
        df.groupby("label_raw")["concentration"]
        .agg(["min", "max", "nunique"])
        .to_string()
    )
    print()

    early = df[df["batch"].between(1, 7)]
    late = df[df["batch"].between(8, 10)]

    print("Provisional temporal pools")
    print(f"Earlier pool (batches 1-7): {len(early):,} rows")
    print(f"Later pool   (batches 8-10): {len(late):,} rows")
    print("Classes in earlier pool:", sorted(early["label_raw"].unique().tolist()))
    print("Classes in later pool:", sorted(late["label_raw"].unique().tolist()))


def _stratified_sample_indices(
    df: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample approximately the same fraction independently from each class.
    """
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    sampled = []

    for label in sorted(df["target"].unique()):
        class_indices = df.index[df["target"] == label].to_numpy()

        n_take = max(1, int(round(len(class_indices) * fraction)))
        n_take = min(n_take, len(class_indices))

        chosen = rng.choice(
            class_indices,
            size=n_take,
            replace=False
        )

        sampled.extend(chosen.tolist())

    sampled = np.asarray(sampled, dtype=np.int64)
    rng.shuffle(sampled)

    return sampled


def make_temporal_splits(
    df: pd.DataFrame,
    n_repeats: int = 10,
    early_batches: tuple[int, ...] = tuple(range(1, 8)),
    late_batches: tuple[int, ...] = (8, 9, 10),
    train_fraction: float = 0.80,
    test_fraction: float = 0.80,
    seed: int = 90051,
) -> list[dict]:
    """
    Construct repeated temporally ordered outer train/test splits.

    Training observations come from earlier batches.
    Test observations come from later batches.
    """
    early_df = df[df["batch"].isin(early_batches)]
    late_df = df[df["batch"].isin(late_batches)]

    if len(early_df) == 0 or len(late_df) == 0:
        raise ValueError("Early or late temporal pool is empty.")

    if max(early_batches) >= min(late_batches):
        raise ValueError(
            "Training batches must be strictly earlier than test batches."
        )

    expected_targets = set(range(6))

    if set(early_df["target"].unique()) != expected_targets:
        raise ValueError("Earlier pool does not contain all six classes.")

    if set(late_df["target"].unique()) != expected_targets:
        raise ValueError("Later pool does not contain all six classes.")

    splits = []

    for repeat in range(n_repeats):
        rng = np.random.default_rng(seed + repeat)

        train_idx = _stratified_sample_indices(
            early_df,
            fraction=train_fraction,
            rng=rng
        )

        test_idx = _stratified_sample_indices(
            late_df,
            fraction=test_fraction,
            rng=rng
        )

        # Leakage checks
        assert set(train_idx).isdisjoint(set(test_idx))
        assert (
            df.loc[train_idx, "batch"].max()
            < df.loc[test_idx, "batch"].min()
        )

        split = {
            "repeat": repeat + 1,
            "seed": seed + repeat,
            "train_indices": train_idx.tolist(),
            "test_indices": test_idx.tolist(),
            "train_batches": sorted(
                df.loc[train_idx, "batch"]
                .unique()
                .astype(int)
                .tolist()
            ),
            "test_batches": sorted(
                df.loc[test_idx, "batch"]
                .unique()
                .astype(int)
                .tolist()
            ),
            "n_train": int(len(train_idx)),
            "n_test": int(len(test_idx)),
        }

        splits.append(split)

    return splits


def save_outputs(
    df: pd.DataFrame,
    splits: list[dict],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "gas_sensor_clean.csv"
    split_path = output_dir / "temporal_outer_splits.json"

    df.to_csv(csv_path, index=False)

    with split_path.open("w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)

    print()
    print(f"Saved cleaned data to: {csv_path}")
    print(f"Saved temporal splits to: {split_path}")



def main() -> None:
    # Dataset folder placed in the same directory as this Python script.
    base_dir = Path(__file__).resolve().parent

    data_dir = (
        base_dir
        / "gas+sensor+array+drift+dataset+at+different+concentrations"
    )

    output_dir = base_dir / "phase1_output"

    df = load_dataset(data_dir)

    inspect_dataset(df)

    splits = make_temporal_splits(
        df,
        n_repeats=10,
        train_fraction=0.80,
        test_fraction=0.80,
        seed=90051,
    )

    print()
    print("=" * 70)
    print("OUTER SPLIT SUMMARY")
    print("=" * 70)

    for split in splits:
        print(
            f"Repeat {split['repeat']:2d}: "
            f"train={split['n_train']:4d} "
            f"from batches {split['train_batches']} | "
            f"test={split['n_test']:4d} "
            f"from batches {split['test_batches']}"
        )

    save_outputs(
        df,
        splits,
        output_dir
    )


if __name__ == "__main__":
    main()
