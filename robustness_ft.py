"""COMP90051 group project: final FT-Transformer sensor-failure robustness.

No training is performed.  Uses the group's *fixed* sensor_failure_patterns.json,
corrupts RAW 128-D readings with the outer-training mean, then rebuilds and
standardises ALL 396 derived features using the saved train-fitted scaler.

Primary uncertainty: SD of 10 outer-repeat means (not SD across masks).

In the project root, run:
    python robustness_ft.py --device cuda --validate-only
    python robustness_ft.py --device cuda --dry-run-clean --repeats 1
    python robustness_ft.py --device cuda --repeats 1
    python robustness_ft.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
# Shared scripts use absolute imports such as 'from shared_protocol import ...'.
for moddir in (HERE, HERE / "shared"):
    if str(moddir) not in sys.path:
        sys.path.insert(0, str(moddir))

import ft_transformer as ft  # the exact final training architecture
from shared_features import FeaturePipeline, feature_names
from shared_metrics import classification_metrics_from_scratch
from shared_protocol import (N_CLASSES, N_SENSORS, RAW_FEATURES, SEED,
                             TrainOnlyStandardScaler, get_outer_split, load_phase1)
from shared_robustness import (EXPECTED_FAILURE_LEVELS, load_failure_patterns,
                               corrupt_raw_with_outer_train_mean,
                               summarise_robustness)

PROTOCOL_VERSION = "FT_FINAL_SHARED_RAW_MEAN_V1"
EXPECTED_COUNTS = {0: 1, 1: 16, 2: 20, 4: 20, 8: 20}
REQUIRED_SOURCE_HASHES = (
    "phase1_output/gas_sensor_clean.csv",
    "phase1_output/temporal_outer_splits.json",
    "shared/shared_features.py",
    "shared/shared_protocol.py",
    "shared/shared_metrics.py",
)


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-dir", type=Path, default=HERE)
    p.add_argument("--source-dir", type=Path, default=Path("ft_results"))
    p.add_argument("--patterns", type=Path, default=Path("shared/sensor_failure_patterns.json"))
    p.add_argument("--output-dir", type=Path, default=Path("ft_robustness_results"))
    p.add_argument("--repeats", type=int, nargs="+", default=list(range(1, 11)))
    p.add_argument("--device", choices=("auto", "cuda", "cpu", "mps"), default="auto")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size; reduce to 32/16 if CUDA runs out of memory")
    p.add_argument("--validate-only", action="store_true",
                   help="Validate sources, file hashes, checkpoint metadata; no inference")
    p.add_argument("--dry-run-clean", action="store_true",
                   help="Run only the k=0 clean reproduction check; do not save any robustness results")
    return p


def absolute_path(project: Path, path: Path) -> Path:
    return (path if path.is_absolute() else project / path).resolve()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_sha(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
                   encoding="utf-8")
    os.replace(tmp, path)


def get_device(choice: str) -> torch.device:
    dev = ft.device_from_choice(choice)
    return dev


def checkpoint_path(source_dir: Path, repeat: int) -> Path:
    return source_dir / "outer_runs" / f"ft_repeat_{repeat:02d}.pt"


def metrics_path(source_dir: Path, repeat: int) -> Path:
    return source_dir / "outer_runs" / f"repeat_{repeat:02d}.json"


def predictions_path(source_dir: Path, repeat: int) -> Path:
    return source_dir / "outer_runs" / f"repeat_{repeat:02d}_predictions.csv"


def check_source(project: Path, source_dir: Path, meta: dict) -> dict:
    """Verify input provenance before reusing pretrained checkpoints."""
    conf = meta.get('configuration', {})
    if not meta.get('signature') or conf.get('phase') != 'ft_final':
        raise ValueError('Expected final FT run_metadata.json and signature')
    if meta['signature'] != hashlib.sha256(json.dumps(conf, sort_keys=True).encode('utf8')).hexdigest():
        raise ValueError('Final training metadata signature mismatch')
    if sorted(float(x) for x in conf.get('lr_grid', [])) != [0.003, 0.0055, 0.01]:
        raise ValueError('Unexpected final FT learning-rate grid')
    if conf.get('features') != 'shared 396-D FeaturePipeline train-only scaling':
        raise ValueError('Unexpected final feature pipeline')
    if meta.get('feature_names') != feature_names() or len(feature_names()) != 396:
        raise ValueError('Final feature names differ from current shared definitions')
    original = conf.get('input_sha256', {})
    verified = {}
    for rel in REQUIRED_SOURCE_HASHES:
        if rel not in original:
            raise RuntimeError(f'Missing training hash for {rel}')
        path = project / rel
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != original[rel]:
            raise RuntimeError(f'Training input hash differs: {rel}')
        verified[rel] = actual
    return verified


def validate_patterns(patterns: dict[int, list[tuple[int, ...]]]) -> None:
    assert EXPECTED_FAILURE_LEVELS == [0, 1, 2, 4, 8]
    if patterns[0] != [()]:
        raise ValueError("Clean condition must contain exactly one empty sensor set")
    if set(patterns[1]) != {(i,) for i in range(1, N_SENSORS + 1)}:
        raise ValueError("Single-sensor masks must exhaust all 16 sensors exactly once")
    for k, entries in patterns.items():
        if len(entries) != EXPECTED_COUNTS[k] or len(set(entries)) != len(entries):
            raise ValueError(f"Wrong number of distinct patterns for k={k}")


def check_checkpoint(source: Path, repeat: int, signature: str,
                     arch: dict, labels: list[str]) -> dict:
    path = checkpoint_path(source, repeat)
    # Only load checkpoints from the user's own training run.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    run = read_json(metrics_path(source, repeat))
    pred = pd.read_csv(predictions_path(source, repeat))
    if (ckpt.get("outer_repeat") != repeat or run.get("outer_repeat") != repeat
            or ckpt.get("experiment_signature") != signature):
        raise ValueError(f"Final model repeat/signature mismatch: {repeat}")
    if int(ckpt.get("n_features", -1)) != 396 or ckpt.get("architecture") != arch:
        raise ValueError(f"Final checkpoint architecture mismatch: {repeat}")
    if ckpt.get("class_labels") != labels:
        raise ValueError(f"Final checkpoint classes mismatch: {repeat}")
    if (float(ckpt.get("learning_rate")) != float(run["selected_lr"])
            or int(ckpt.get("epochs")) != int(run["fit_epochs"])):
        raise ValueError(f"Final selected learning rate/epoch mismatch: {repeat}")
    for key, width in (("scaler_mean", 396), ("scaler_scale", 396),
                       ("raw_outer_train_mean", 128)):
        value = ckpt.get(key)
        if value is None or tuple(np.asarray(value).shape) != (width,):
            raise ValueError(f"Repeat {repeat}: invalid {key}")
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError(f"Repeat {repeat}: nonfinite {key}")
    if np.any(np.asarray(ckpt["scaler_scale"]) <= 0):
        raise ValueError(f"Repeat {repeat}: nonpositive scaler scale")
    if (len(pred) != int(run["n_test"]) or not (pred["outer_repeat"] == repeat).all()):
        raise ValueError(f"Final repeat {repeat}: prediction row count mismatch")
    if len(set(pred["row_index"])) != len(pred):
        raise ValueError(f"Final repeat {repeat}: duplicated test predictions")
    return ckpt


def build_model_and_pipe(ckpt: dict, device: torch.device):
    net = ft.new_model(SimpleNamespace(**ckpt["architecture"]), 396)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net.to(device).eval()
    pipeline = FeaturePipeline()
    pipeline.scaler_ = TrainOnlyStandardScaler()
    pipeline.scaler_.mean_ = np.asarray(ckpt["scaler_mean"], dtype=np.float64).copy()
    pipeline.scaler_.scale_ = np.asarray(ckpt["scaler_scale"], dtype=np.float64).copy()
    return net, pipeline


def classwise_from_cm(cm: np.ndarray, repeat: int, k: int, pid: int,
                      failed_string: str, raw_labels: list[str]) -> list[dict]:
    rows = []
    for c in range(N_CLASSES):
        tp = float(cm[c, c]); fp = float(cm[:, c].sum() - tp)
        fn = float(cm[c, :].sum() - tp)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        rows.append({"outer_repeat": repeat, "n_failed_sensors": k,
                     "pattern_id": pid, "failed_sensors": failed_string,
                     "class_id": c, "label_raw": raw_labels[c],
                     "precision": prec, "recall": rec, "f1": f1})
    return rows


def evaluate_one(repeat: int, ckpt: dict, df: pd.DataFrame, split: dict,
                 source: Path, patterns: dict[int, list[tuple[int, ...]]],
                 device: torch.device, batch_size: int,
                 dry_clean: bool = False) -> dict:
    tr = np.asarray(split["train_indices"], dtype=np.int64)
    te = np.asarray(split["test_indices"], dtype=np.int64)
    Xraw = df[RAW_FEATURES].to_numpy(dtype=np.float64)
    Xte = Xraw[te]
    # Check checkpoint corruption mean was fitted ONLY on this outer train.
    raw_means = np.asarray(ckpt["raw_outer_train_mean"], dtype=np.float64)
    computed_means = Xraw[tr].mean(axis=0)
    if not np.allclose(raw_means, computed_means, rtol=1e-9, atol=1e-8):
        raise RuntimeError(f"Repeat {repeat}: checkpoint raw mean != outer training mean")
    orig_labels = sorted(pd.unique(df["target"]).tolist())
    if [str(x) for x in orig_labels] != ckpt["class_labels"]:
        raise RuntimeError("Class label encoding differs from the final checkpoint")
    label_ids = {x: i for i, x in enumerate(orig_labels)}
    yte = np.array([label_ids[x] for x in df.loc[te, "target"]], dtype=np.int64)
    stored = read_json(metrics_path(source, repeat))
    old_pred = pd.read_csv(predictions_path(source, repeat))
    if not np.array_equal(old_pred["row_index"].to_numpy(dtype=np.int64), te):
        raise RuntimeError(f"Repeat {repeat}: final training test row order differs from outer split")
    if not np.array_equal(old_pred["y_true_encoded"].to_numpy(dtype=np.int64), yte):
        raise RuntimeError(f"Repeat {repeat}: final training true labels disagree with dataset")
    net, pipeline = build_model_and_pipe(ckpt, device)
    pats = {0: patterns[0]} if dry_clean else patterns
    pattern_rows, cm_records, class_records = [], [], []
    clean_preds = None
    clean_f1 = None
    for k in pats:
        entries = pats[k]
        print(f"  k={k}: {len(entries)} sensor failure patterns", flush=True)
        for pid, failed in enumerate(entries, 1):
            corrupted = corrupt_raw_with_outer_train_mean(
                Xte, failed, raw_means, RAW_FEATURES)
            X396 = pipeline.transform(corrupted)  # critical: reengineer AFTER raw corruption
            yp = ft.predict(net, X396, batch_size, device)
            if not np.isin(yp, np.arange(N_CLASSES)).all():
                raise RuntimeError("Invalid predicted class label")
            metrics, cm = classification_metrics_from_scratch(yte, yp, N_CLASSES)
            failed_str = ",".join(map(str, failed))
            pattern_rows.append({"outer_repeat": repeat, "n_failed_sensors": k,
                "n_available_sensors": N_SENSORS - k,
                "pattern_id": pid, "failed_sensors": failed_str, **metrics})
            cm_records.append({"outer_repeat": repeat, "n_failed_sensors": k,
                "pattern_id": pid, "failed_sensors": list(failed),
                "confusion_matrix": cm.tolist()})
            class_records.extend(classwise_from_cm(cm, repeat, k, pid, failed_str,
                                                   [str(x) for x in orig_labels]))
            if k == 0:
                clean_preds = yp
                clean_f1 = metrics["macro_f1"]
                diff = abs(clean_f1 - float(stored["macro_f1"]))
                if diff > 1e-6:
                    raise RuntimeError(f"Repeat {repeat}: clean Macro-F1 mismatch "
                                       f"{clean_f1:.8f} vs final training {stored['macro_f1']:.8f}")
                if not np.array_equal(yp, old_pred["y_pred_encoded"].to_numpy(dtype=np.int64)):
                    bad = np.flatnonzero(yp != old_pred["y_pred_encoded"].to_numpy(dtype=np.int64))
                    raise RuntimeError(f"Repeat {repeat}: clean predictions disagree with final training "
                                       f"in {len(bad)} rows (first index {int(bad[0])}). "
                                       "Check model/scaler/data/GPU determinism.")
                print(f"  CLEAN CHECK PASSED: Macro-F1={clean_f1:.6f}, "
                      f"exactly {len(yp)} final training predictions reproduced", flush=True)
        if not dry_clean:
            selected = [r["macro_f1"] for r in pattern_rows if r["n_failed_sensors"] == k]
            print(f"  k={k}: mean Macro-F1={np.mean(selected):.5f}", flush=True)
    if clean_preds is None:
        raise RuntimeError("Missing clean condition")
    if dry_clean:
        return {"outer_repeat": repeat, "clean_macro_f1": clean_f1}
    for row in pattern_rows:
        row["macro_f1_drop_from_clean"] = float(clean_f1 - row["macro_f1"])
        row["macro_f1_retained_fraction"] = (
            float(row["macro_f1"] / clean_f1) if clean_f1 > 0 else None)
    return {"outer_repeat": repeat, "pattern_rows": pattern_rows,
            "class_rows": class_records, "confusion_records": cm_records,
            "clean_consistency": {"passed": True,
                "stored_macro_f1": float(stored["macro_f1"]),
                "recomputed_macro_f1": float(clean_f1),
                "predictions_identical": True,
                "n_test": len(te)}}


def sample_sd(values) -> float:
    x = np.asarray(values, dtype=np.float64)
    return float(x.std(ddof=1)) if len(x) > 1 else 0.0


def aggregate(payloads: list[dict], out: Path) -> None:
    if not payloads:
        return
    patterns_df = pd.DataFrame([r for p in payloads for r in p["pattern_rows"]])
    classes_df = pd.DataFrame([r for p in payloads for r in p["class_rows"]])
    confusion = [r for p in payloads for r in p["confusion_records"]]
    clean = pd.DataFrame([{"outer_repeat": p["outer_repeat"], **p["clean_consistency"]}
                          for p in payloads])
    patterns_df = patterns_df.sort_values(["outer_repeat", "n_failed_sensors", "pattern_id"])
    classes_df = classes_df.sort_values(["outer_repeat", "n_failed_sensors", "pattern_id", "class_id"])
    patterns_df.to_csv(out / "ft_sensor_failure_pattern_results.csv", index=False)
    classes_df.to_csv(out / "ft_classwise_pattern_results.csv", index=False)
    clean.to_csv(out / "ft_clean_consistency_checks.csv", index=False)
    atomic_json(out / "ft_sensor_failure_confusion_matrices.json", confusion)

    # Use group's shared summarisation, then augment with paired drops and
    # mean of within-repeat retained fractions (rather than a global ratio).
    by_repeat, final = summarise_robustness(patterns_df)
    by_repeat["n_available_sensors"] = N_SENSORS - by_repeat["n_failed_sensors"]
    by_repeat["macro_f1_drop_mean_from_clean"] = np.nan
    by_repeat["retained_fraction_mean"] = np.nan
    for r in sorted(by_repeat["outer_repeat"].unique()):
        clean_val = float(patterns_df.loc[(patterns_df.outer_repeat == r) &
                              (patterns_df.n_failed_sensors == 0), "macro_f1"].iloc[0])
        relevant = by_repeat["outer_repeat"] == r
        by_repeat.loc[relevant, "macro_f1_drop_mean_from_clean"] = (
            clean_val - by_repeat.loc[relevant, "macro_f1_mean_across_patterns"])
        if clean_val > 0:
            by_repeat.loc[relevant, "retained_fraction_mean"] = (
                by_repeat.loc[relevant, "macro_f1_mean_across_patterns"] / clean_val)
    by_repeat = by_repeat.sort_values(["outer_repeat", "n_failed_sensors"])
    by_repeat.to_csv(out / "ft_sensor_failure_by_repeat.csv", index=False)
    aug = []
    for k, group in by_repeat.groupby("n_failed_sensors"):
        aug.append({"n_failed_sensors": int(k),
                    "macro_f1_drop_mean": float(group["macro_f1_drop_mean_from_clean"].mean()),
                    "macro_f1_drop_sd_across_outer_repeats": sample_sd(group["macro_f1_drop_mean_from_clean"]),
                    "retained_fraction_mean": float(group["retained_fraction_mean"].mean()),
                    "retained_fraction_sd_across_outer_repeats": sample_sd(group["retained_fraction_mean"]),
                    "patterns_per_repeat": int(group["n_patterns"].iloc[0]),
                    "n_available_sensors": int(N_SENSORS - k)})
    aug_df = pd.DataFrame(aug)
    final = final.drop(columns=["macro_f1_drop_mean", "retained_fraction_mean"]).merge(
        aug_df, on="n_failed_sensors", validate="one_to_one").sort_values("n_failed_sensors")
    final.to_csv(out / "ft_final_robustness_summary.csv", index=False)
    final[["n_failed_sensors", "n_available_sensors", "macro_f1_mean",
           "macro_f1_sd_across_outer_repeats", "macro_f1_drop_mean",
           "retained_fraction_mean"]].to_csv(out / "ft_report_ready_robustness_table.csv", index=False)

    single = patterns_df.loc[patterns_df.n_failed_sensors == 1].copy()
    single["sensor_id"] = single["failed_sensors"].astype(int)
    single_by_rep = single[["outer_repeat", "sensor_id", "macro_f1",
                            "macro_f1_drop_from_clean", "macro_f1_retained_fraction"]].copy()
    single_by_rep.to_csv(out / "ft_single_sensor_by_repeat.csv", index=False)
    single_summary = (single_by_rep.groupby("sensor_id", as_index=False)
       .agg(macro_f1_mean=("macro_f1", "mean"),
            macro_f1_sd_across_outer_repeats=("macro_f1", lambda x: sample_sd(x)),
            macro_f1_drop_mean=("macro_f1_drop_from_clean", "mean"),
            macro_f1_drop_sd_across_outer_repeats=("macro_f1_drop_from_clean", lambda x: sample_sd(x)),
            retained_fraction_mean=("macro_f1_retained_fraction", "mean"))
       .sort_values(["macro_f1_drop_mean", "sensor_id"], ascending=[False, True]))
    single_summary["damage_rank"] = np.arange(1, len(single_summary) + 1)
    single_summary.to_csv(out / "ft_final_single_sensor_summary.csv", index=False)

    cls_rep = (classes_df.groupby(["outer_repeat", "n_failed_sensors", "class_id", "label_raw"],
                    as_index=False)
               .agg(precision_mean_across_patterns=("precision", "mean"),
                    recall_mean_across_patterns=("recall", "mean"),
                    f1_mean_across_patterns=("f1", "mean"),
                    f1_sd_across_patterns=("f1", lambda x: sample_sd(x))))
    cls_rep.to_csv(out / "ft_classwise_by_repeat.csv", index=False)
    cls_sum = (cls_rep.groupby(["n_failed_sensors", "class_id", "label_raw"], as_index=False)
               .agg(precision_mean=("precision_mean_across_patterns", "mean"),
                    precision_sd_across_outer_repeats=("precision_mean_across_patterns", lambda x: sample_sd(x)),
                    recall_mean=("recall_mean_across_patterns", "mean"),
                    recall_sd_across_outer_repeats=("recall_mean_across_patterns", lambda x: sample_sd(x)),
                    f1_mean=("f1_mean_across_patterns", "mean"),
                    f1_sd_across_outer_repeats=("f1_mean_across_patterns", lambda x: sample_sd(x)),
                    mean_within_repeat_f1_sd_across_patterns=("f1_sd_across_patterns", "mean")))
    cls_sum.to_csv(out / "ft_final_classwise_robustness_summary.csv", index=False)



def read_completed_cache(output: Path, signature: str, source: Path) -> list[dict]:
    """Reject stale/incomplete cache entries before aggregating any result."""
    values = []
    for path in sorted((output / "repeat_results").glob("repeat_*.json")):
        data = read_json(path)
        r = int(data.get("outer_repeat", -1))
        ckpt = checkpoint_path(source, r)
        if (r not in range(1, 11) or data.get("protocol_version") != PROTOCOL_VERSION
                or data.get("manifest_signature") != signature
                or not ckpt.is_file() or data.get("checkpoint_sha256") != sha256_file(ckpt)
                or not data.get("clean_consistency", {}).get("predictions_identical")
                or len(data.get("pattern_rows", [])) != 77
                or len(data.get("class_rows", [])) != 77 * N_CLASSES
                or len(data.get("confusion_records", [])) != 77):
            raise RuntimeError(f"Stale or incomplete cache {path}; use a new output directory.")
        values.append(data)
    if len({p["outer_repeat"] for p in values}) != len(values):
        raise RuntimeError("Duplicate outer repeats in robustness cache")
    return values

def main() -> None:
    cfg = args_parser().parse_args()
    project = cfg.project_dir.resolve()
    source = absolute_path(project, cfg.source_dir)
    patterns_path = absolute_path(project, cfg.patterns)
    output = absolute_path(project, cfg.output_dir)
    repeats = sorted(set(cfg.repeats))
    if not repeats or any(x not in range(1, 11) for x in repeats):
        raise ValueError("--repeats must be among 1..10")
    if cfg.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    meta = read_json(source / "run_metadata.json")
    source_hashes = check_source(project, source, meta)
    patterns = load_failure_patterns(patterns_path)
    validate_patterns(patterns)
    pattern_file_hash = sha256_file(patterns_path)
    raw_df, outer_splits = load_phase1(project)
    labels = [str(x) for x in sorted(pd.unique(raw_df.target).tolist())]
    expected_labels = meta["configuration"]["class_labels"]
    if labels != expected_labels:
        raise ValueError("Class labels differ from final training metadata")
    conf = meta["configuration"]
    arch = {k: conf[k] for k in ("d_token", "n_heads", "n_layers", "ffn_mult", "dropout")}
    signatures = {}
    for r in repeats:
        ckpt = check_checkpoint(source, r, meta["signature"], arch, labels)
        split = get_outer_split(outer_splits, r)
        pr = pd.read_csv(predictions_path(source, r))
        te = np.asarray(split["test_indices"], dtype=np.int64)
        if not np.array_equal(pr["row_index"].to_numpy(dtype=np.int64), te):
            raise RuntimeError(f"final training prediction indices mismatch outer repeat {r}")
        signatures[r] = sha256_file(checkpoint_path(source, r))
        del ckpt
    print(f"Final FT source verified: {len(repeats)} checkpoint(s), 396 features", flush=True)
    print("Shared fixed masks:", {k: len(v) for k, v in patterns.items()}, flush=True)
    print(f"Pattern SHA256: {pattern_file_hash}", flush=True)
    if cfg.validate_only:
        print("VALIDATION PASSED — no inference or training.")
        return
    device = get_device(cfg.device)
    print(f"Device: {device}. No training will occur.", flush=True)
    if cfg.dry_run_clean:
        for r in repeats:
            ckpt = check_checkpoint(source, r, meta["signature"], arch, labels)
            result = evaluate_one(r, ckpt, raw_df, get_outer_split(outer_splits, r),
                                  source, patterns, device, cfg.batch_size, dry_clean=True)
            print(f"Dry run repeat {r}: {result['clean_macro_f1']:.6f}", flush=True)
        print("ALL CLEAN CHECKS PASSED. No robustness results saved.")
        return
    manifest = {"protocol_version": PROTOCOL_VERSION,
                "source_phase": "ft_final", "model_signature": meta["signature"],
                "source_input_sha256": source_hashes, "pattern_file_sha256": pattern_file_hash,
                "corruption_mode": "raw_mean_before_396_feature_construction",
                "checkpoint_sha256_by_repeat": {str(r): signatures[r] for r in repeats},
                "note": "SD across outer-repeat means; masks are averaged within each repeat"}
    # Allow incremental runs with different --repeats; signature excludes repeat choice
    # but a cache carries its own exact checkpoint digest.
    out_meta = {k: v for k, v in manifest.items() if k != "checkpoint_sha256_by_repeat"}
    out_sig = stable_sha(out_meta)
    out_meta["signature"] = out_sig
    if (output / "run_metadata.json").exists():
        previous = read_json(output / "run_metadata.json")
        if previous != out_meta:
            raise RuntimeError("Output directory metadata differs. Choose a NEW --output-dir.")
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError("Output directory is non-empty without matching metadata")
    output.mkdir(parents=True, exist_ok=True)
    (output / "repeat_results").mkdir(exist_ok=True)
    if not (output / "run_metadata.json").exists():
        atomic_json(output / "run_metadata.json", out_meta)
    for r in repeats:
        cache = output / "repeat_results" / f"repeat_{r:02d}.json"
        if cache.exists():
            prev = read_json(cache)
            if (prev.get("protocol_version") == PROTOCOL_VERSION and
                prev.get("manifest_signature") == out_sig and
                prev.get("checkpoint_sha256") == signatures[r] and
                prev.get("clean_consistency", {}).get("predictions_identical") is True and
                len(prev.get("pattern_rows", [])) == 77 and
                len(prev.get("class_rows", [])) == 77 * N_CLASSES and
                len(prev.get("confusion_records", [])) == 77):
                print(f"Repeat {r}: cached and verified, skipping", flush=True)
                continue
            raise RuntimeError(f"Existing repeat cache failed validation: {cache}. Do not overwrite.")
        print(f"\n======== FT ROBUSTNESS | OUTER REPEAT {r}/10 ========", flush=True)
        ckpt = check_checkpoint(source, r, meta["signature"], arch, labels)
        result = evaluate_one(r, ckpt, raw_df, get_outer_split(outer_splits, r),
                              source, patterns, device, cfg.batch_size)
        result.update({"protocol_version": PROTOCOL_VERSION,
                       "manifest_signature": out_sig,
                       "checkpoint_sha256": signatures[r]})
        atomic_json(cache, result)
        completed = read_completed_cache(output, out_sig, source)
        aggregate(completed, output)
        print(f"Saved repeat {r}, total {len(completed)} repeats.", flush=True)
    completed = read_completed_cache(output, out_sig, source)
    if completed:
        aggregate(completed, output)
    print(f"\nCompleted repeats: {[p['outer_repeat'] for p in completed]}", flush=True)
    print(f"Results: {output}", flush=True)
    if len(completed) == 10:
        result = pd.read_csv(output / "ft_report_ready_robustness_table.csv")
        print("\nFINAL ROBUSTNESS SUMMARY\n" + result.to_string(index=False,
                                                       float_format=lambda x: f"{x:.4f}"), flush=True)
    else:
        print("Run without --repeats to complete the remaining repeats.", flush=True)


if __name__ == "__main__":
    main()
