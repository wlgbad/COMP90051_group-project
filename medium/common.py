"""
Data loading, caching and the failure-evaluation wrapper for the medium model.

Built on the team's shared_protocol / shared_metrics / shared_robustness modules;
the only extra shared piece used here is shared_features.FeaturePipeline.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from shared_metrics import classification_metrics_from_scratch
from shared_protocol import N_CLASSES, RAW_FEATURES, load_phase1
from shared_robustness import evaluate_failure_patterns, load_failure_patterns

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PATTERNS_PATH = ROOT / "sensor_failure_patterns.json"
TEST_BATCHES = (8, 9, 10)
GAS_NAMES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """df (index == row_id), X_raw (n, 128), y, batch, outer splits with array indices."""
    df, splits = load_phase1(ROOT)
    if not (df["row_id"].to_numpy() == df.index.to_numpy()).all():
        raise ValueError("row_id must equal the DataFrame index.")
    for split in splits:
        split["train_indices"] = np.asarray(split["train_indices"], dtype=np.int64)
        split["test_indices"] = np.asarray(split["test_indices"], dtype=np.int64)
    X_raw = df[RAW_FEATURES].to_numpy(dtype=np.float64)
    return (df, X_raw, df["target"].to_numpy(dtype=np.int64),
            df["batch"].to_numpy(dtype=np.int64), splits)


def load_patterns() -> dict[int, list[tuple[int, ...]]]:
    return load_failure_patterns(PATTERNS_PATH)


def stratified_subsample(indices: np.ndarray, y: np.ndarray, fraction: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Take round(fraction * n_c) rows (at least 1) of every class c in `indices`."""
    chosen = []
    for c in np.unique(y[indices]):
        members = indices[y[indices] == c]
        n_take = min(len(members), max(1, int(round(fraction * len(members)))))
        chosen.append(rng.choice(members, size=n_take, replace=False))
    return np.concatenate(chosen)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """Per-class F1 from a confusion matrix (rows = true), 0 when undefined."""
    tp = np.diag(cm).astype(np.float64)
    pred_pos, true_pos = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros_like(tp), where=true_pos > 0)
    denom = precision + recall
    return np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)


# ---------------------------------------------------------------- caching

def cache_dir(variant: str, smoke: bool) -> Path:
    return ROOT / "cache" / "medium" / (variant + ("_smoke" if smoke else ""))


def results_path(name: str, smoke: bool) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / (f"{name}_smoke.csv" if smoke else f"{name}.csv")


def save_bundle(bundle: dict, variant: str, smoke: bool) -> None:
    out = cache_dir(variant, smoke)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / f"repeat_{bundle['repeat']:02d}.joblib")


def load_bundles(variant: str, smoke: bool) -> list[dict]:
    paths = sorted(cache_dir(variant, smoke).glob("repeat_*.joblib"))
    if not paths:
        raise FileNotFoundError(f"No cached models in {cache_dir(variant, smoke)}")
    return [joblib.load(p) for p in paths]


# ---------------------------------------------------------------- failure sweep

def sweep_with_breakdown(model, pipeline, train_mean: np.ndarray, X_raw_test: np.ndarray,
                         y_test: np.ndarray, batch_test: np.ndarray, patterns: dict,
                         repeat: int, algorithm: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run shared_robustness.evaluate_failure_patterns (raw_mean mode) and, from the
    same predictions, add per-class F1 and per-test-batch rows.

    Returns (results, confusion):
      results  shared columns + algorithm, test_batch ('all', '8', '9', '10'), f1_c0..f1_c5
      confusion long format (test_batch = all): outer_repeat, n_failed_sensors,
               pattern_id, true, pred, count
    """
    predictions: list[np.ndarray] = []

    def predict_fn(X_corrupted_raw: np.ndarray) -> np.ndarray:
        y_pred = model.predict(pipeline.transform(X_corrupted_raw))
        predictions.append(y_pred)
        return y_pred

    shared = evaluate_failure_patterns(predict_fn, X_raw_test, y_test, patterns,
                                       corruption_mode="raw_mean", outer_repeat=repeat,
                                       outer_train_feature_mean=train_mean)
    assert len(predictions) == len(shared)

    id_cols = ["outer_repeat", "n_failed_sensors", "n_available_sensors",
               "pattern_id", "failed_sensors"]
    rows, cm_rows = [], []
    for (_, ids), y_pred in zip(shared[id_cols].iterrows(), predictions):
        ids = ids.to_dict()
        for test_batch in ("all",) + TEST_BATCHES:
            mask = np.ones(len(y_test), bool) if test_batch == "all" else batch_test == test_batch
            metrics, cm = classification_metrics_from_scratch(y_test[mask], y_pred[mask], N_CLASSES)
            f1 = per_class_f1(cm)
            rows.append({"algorithm": algorithm, **ids, **metrics, "test_batch": str(test_batch),
                         **{f"f1_c{c}": f1[c] for c in range(N_CLASSES)}})
            if test_batch == "all":
                cm_rows += [{**ids, "true": t, "pred": p, "count": int(cm[t, p])}
                            for t in range(N_CLASSES) for p in range(N_CLASSES)]

    results = pd.DataFrame(rows)
    # Drop / retained relative to the same repeat's clean score on the same test batch
    # (identical to shared_robustness for test_batch == 'all').
    clean = results[results.n_failed_sensors == 0].set_index("test_batch").macro_f1
    results["macro_f1_drop_from_clean"] = results.test_batch.map(clean) - results.macro_f1
    results["macro_f1_retained_fraction"] = results.macro_f1 / results.test_batch.map(clean)

    check = results[results.test_batch == "all"].reset_index(drop=True)
    assert np.allclose(check.macro_f1, shared.macro_f1)
    assert np.allclose(check.macro_f1_drop_from_clean, shared.macro_f1_drop_from_clean)
    return results, pd.DataFrame(cm_rows)
