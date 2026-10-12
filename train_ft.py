"""COMP90051 group project: final FT-Transformer nested temporal evaluation.

Run the same protocol directly, with no dependence on exploratory phases:
  10 outer repeated temporal train/test splits, 3 inner temporal splits,
  LR grid {0.003, 0.0055, 0.01}, train-only 396-feature pipeline.

Hyperparameters are selected ONLY on inner validation Macro-F1. The outer
held-out labels are used once per repeat for reporting, never for tuning.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'shared'))

import numpy as np
import pandas as pd
import torch
from shared_features import FeaturePipeline, feature_names
from shared_metrics import classification_metrics_from_scratch
from shared_protocol import N_CLASSES, RAW_FEATURES, SEED, get_outer_split, load_phase1, make_inner_temporal_splits
from ft_transformer import device_from_choice, inner_fit, final_fit, predict

LRS = (0.003, 0.0055, 0.01)
REQUIRED_FILES = (
    'phase1_output/gas_sensor_clean.csv',
    'phase1_output/temporal_outer_splits.json',
    'shared/shared_features.py',
    'shared/shared_protocol.py',
    'shared/shared_metrics.py',
)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-dir', type=Path, default=ROOT)
    p.add_argument('--output-dir', type=Path, default=Path('ft_results'))
    p.add_argument('--repeats', nargs='+', type=int, default=list(range(1, 11)))
    p.add_argument('--device', choices=('auto', 'cpu', 'cuda', 'mps'), default='auto')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--d-token', type=int, default=32)
    p.add_argument('--n-layers', type=int, default=2)
    p.add_argument('--n-heads', type=int, default=4)
    p.add_argument('--ffn-mult', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--validate-only', action='store_true')
    return p.parse_args()


def absolute(project: Path, path: Path) -> Path:
    return (path if path.is_absolute() else project / path).resolve()


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode('utf8')).hexdigest()


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf8')
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf8'))


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.csv.tmp')
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def lr_filename(lr: float) -> str:
    return format(lr, '.12g').replace('.', 'p')


def summarise(output: Path) -> None:
    inner = [read_json(p) for p in sorted((output / 'inner_cv').glob('*.json'))]
    if inner:
        save_csv(pd.DataFrame(inner), output / 'inner_cv_scores.csv')
    runs, all_preds = [], []
    for p in sorted((output / 'outer_runs').glob('repeat_??.json')):
        row = read_json(p)
        runs.append({k: v for k, v in row.items() if k != 'confusion_matrix'})
        pp = output / 'outer_runs' / f"repeat_{row['outer_repeat']:02d}_predictions.csv"
        if not pp.exists():
            raise RuntimeError(f'Missing predictions for completed repeat: {pp}')
        all_preds.append(pd.read_csv(pp))
    if not runs:
        return
    df = pd.DataFrame(runs).sort_values('outer_repeat')
    save_csv(df, output / 'outer_test_metrics.csv')
    save_csv(df[['outer_repeat', 'selected_lr', 'fit_epochs', 'inner_cv_macro_f1_mean']],
             output / 'selected_lr_by_repeat.csv')
    save_csv(pd.concat(all_preds, ignore_index=True), output / 'outer_test_predictions.csv')
    summary: dict[str, Any] = {'n_outer_repeats_completed': len(df), 'lr_grid': list(LRS)}
    for metric in ('accuracy', 'macro_f1', 'macro_precision', 'macro_recall'):
        summary[metric + '_mean'] = float(df[metric].mean())
        summary[metric + '_sample_sd'] = float(df[metric].std(ddof=1)) if len(df) > 1 else 0.0
    counts = {str(lr): int(np.isclose(df.selected_lr.to_numpy(), lr, rtol=0, atol=1e-12).sum())
              for lr in LRS}
    summary['lr_selection_counts'] = counts
    summary['middle_lr_strictly_most_frequent'] = counts[str(LRS[1])] > max(counts[str(LRS[0])], counts[str(LRS[2])])
    save_json(output / 'summary.json', summary)
    print('Completed:', len(df), '| Macro-F1:', summary['macro_f1_mean'],
          '| LR counts:', counts, flush=True)


def main() -> None:
    cfg = arguments()
    project = cfg.project_dir.resolve()
    output = absolute(project, cfg.output_dir)
    repeats = cfg.repeats
    if not repeats or len(set(repeats)) != len(repeats) or any(r not in range(1, 11) for r in repeats):
        raise ValueError('--repeats must contain distinct values between 1 and 10')
    if cfg.epochs < 1 or cfg.patience < 1 or cfg.batch_size < 1 or cfg.n_layers < 1 or cfg.n_heads < 1:
        raise ValueError('Invalid nonpositive training parameter')
    if cfg.d_token % cfg.n_heads != 0:
        raise ValueError('--d-token must be divisible by --n-heads')
    if len(feature_names()) != 396:
        raise RuntimeError('Shared feature definition is not 396-dimensional')
    data, outer_splits = load_phase1(project)
    X_raw = data[RAW_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(X_raw).all():
        raise ValueError('Non-finite raw input')
    labels = sorted(pd.unique(data.target).tolist())
    if len(labels) != N_CLASSES:
        raise ValueError('Expected 6 target classes')
    label_to_index = {v: i for i, v in enumerate(labels)}
    y = np.array([label_to_index[t] for t in data.target], dtype=np.int64)
    hashes = {p: file_sha(project / p) for p in REQUIRED_FILES}
    conf = {k: getattr(cfg, k) for k in ('epochs', 'patience', 'batch_size', 'd_token',
             'n_layers', 'n_heads', 'ffn_mult', 'dropout', 'weight_decay')}
    conf.update({'phase': 'ft_final', 'lr_grid': list(LRS), 'class_labels': [str(v) for v in labels],
                 'input_sha256': hashes, 'features': 'shared 396-D FeaturePipeline train-only scaling',
                 'protocol': '10 outer temporal repetitions; 3 inner temporal validations'})
    signature = stable_sha(conf)
    if (output / 'run_metadata.json').exists():
        existing = read_json(output / 'run_metadata.json')
        if existing.get('signature') != signature:
            raise RuntimeError('Experiment configuration changed. Choose a NEW --output-dir.')
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError('Output directory is nonempty without a matching run_metadata.json')
    # Test all requested indices/folds before running model training.
    for repeat in repeats:
        outer = get_outer_split(outer_splits, repeat)
        folds = make_inner_temporal_splits(data, outer['train_indices'])
        if len(folds) != 3:
            raise RuntimeError('Expected exactly three inner temporal splits')
    print(f'Input validated: {len(data)} rows; {X_raw.shape[1]} raw -> 396 engineered features')
    if cfg.validate_only:
        print('VALIDATION PASSED; no training was performed.')
        return
    device = device_from_choice(cfg.device)
    print('Device:', device, flush=True)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'inner_cv').mkdir(exist_ok=True)
    (output / 'outer_runs').mkdir(exist_ok=True)
    if not (output / 'run_metadata.json').exists():
        save_json(output / 'run_metadata.json', {'signature': signature,
                   'configuration': conf, 'feature_names': feature_names()})

    for repeat in repeats:
        split = get_outer_split(outer_splits, repeat)
        train_idx = np.asarray(split['train_indices'], dtype=np.int64)
        test_idx = np.asarray(split['test_indices'], dtype=np.int64)
        run_path = output / 'outer_runs' / f'repeat_{repeat:02d}.json'
        model_path = output / 'outer_runs' / f'ft_repeat_{repeat:02d}.pt'
        pred_path = output / 'outer_runs' / f'repeat_{repeat:02d}_predictions.csv'
        if run_path.exists() and model_path.exists() and pred_path.exists():
            print(f'Outer repeat {repeat}: already completed, skipping', flush=True)
            continue
        if any(p.exists() for p in (run_path, model_path, pred_path)):
            raise RuntimeError(f'Incomplete final output on repeat {repeat}; move incomplete files aside')
        folds = make_inner_temporal_splits(data, train_idx)
        print(f'\nOUTER REPEAT {repeat}/10 | train={len(train_idx)} test={len(test_idx)}', flush=True)
        cv_rows = []
        for fold in folds:
            fid = int(fold['inner_split'])
            itr = np.asarray(fold['train_indices'], dtype=np.int64)
            iva = np.asarray(fold['val_indices'], dtype=np.int64)
            pipeline = FeaturePipeline()
            Xtr = pipeline.fit_transform(X_raw[itr])
            Xva = pipeline.transform(X_raw[iva])
            # Same initialization and minibatch seed for competing LRs within fold.
            seed = SEED + repeat * 10000 + fid
            for lr in LRS:
                fp = output / 'inner_cv' / f'outer_{repeat:02d}_lr_{lr_filename(lr)}_split_{fid}.json'
                if fp.exists():
                    result = read_json(fp)
                    if (result.get('outer_repeat') != repeat or
                            float(result.get('learning_rate')) != lr or
                            result.get('inner_split') != fid or result.get('seed') != seed or
                            result.get('n_train') != len(itr) or result.get('n_val') != len(iva)):
                        raise RuntimeError(f'Incompatible CV cache: {fp}')
                    print(f'  cached LR={lr} split={fid}', flush=True)
                else:
                    print(f'  training LR={lr} split={fid}', flush=True)
                    fit = inner_fit(Xtr, y[itr], Xva, y[iva], cfg, lr, seed, device)
                    result = {'outer_repeat': repeat, 'learning_rate': lr,
                              'inner_split': fid, 'n_train': len(itr), 'n_val': len(iva),
                              'seed': seed, **fit}
                    save_json(fp, result)
                cv_rows.append(result)
        cv = pd.DataFrame(cv_rows)
        stat = cv.groupby('learning_rate', as_index=False).agg(
            mean_macro_f1=('best_macro_f1', 'mean'), sd_macro_f1=('best_macro_f1', 'std'))
        stat = stat.sort_values(['mean_macro_f1', 'sd_macro_f1', 'learning_rate'],
                                ascending=[False, True, True])
        chosen = stat.iloc[0]
        lr = float(chosen['learning_rate'])
        fit_epochs = max(1, int(np.median(cv.loc[cv.learning_rate == lr, 'best_epoch']) + 0.5))
        # Refit on outer TRAIN only; held-out labels never drive selection.
        pipe = FeaturePipeline()
        Xtr = pipe.fit_transform(X_raw[train_idx])
        Xte = pipe.transform(X_raw[test_idx])
        model = final_fit(Xtr, y[train_idx], cfg, lr, fit_epochs,
                          SEED + repeat * 10000 + 9999, device)
        pred = predict(model, Xte, cfg.batch_size, device)
        metrics, cm = classification_metrics_from_scratch(y[test_idx], pred, N_CLASSES)
        torch.save({
            'model_state_dict': {k: t.detach().cpu() for k, t in model.state_dict().items()},
            'n_features': len(feature_names()),
            'architecture': {k: getattr(cfg, k) for k in ('d_token', 'n_layers', 'n_heads', 'ffn_mult', 'dropout')},
            'scaler_mean': torch.tensor(pipe.scaler_.mean_, dtype=torch.float64),
            'scaler_scale': torch.tensor(pipe.scaler_.scale_, dtype=torch.float64),
            'raw_outer_train_mean': torch.tensor(X_raw[train_idx].mean(axis=0), dtype=torch.float64),
            'class_labels': [str(v) for v in labels], 'outer_repeat': repeat,
            'learning_rate': lr, 'epochs': fit_epochs,
            'experiment_signature': signature,
        }, model_path)
        save_csv(pd.DataFrame({
            'outer_repeat': repeat, 'row_index': test_idx,
            'batch': data.loc[test_idx, 'batch'].to_numpy(),
            'y_true_encoded': y[test_idx], 'y_pred_encoded': pred,
            'y_true_original': [labels[i] for i in y[test_idx]],
            'y_pred_original': [labels[i] for i in pred],
        }), pred_path)
        save_json(run_path, {'outer_repeat': repeat, 'n_train': len(train_idx),
                 'n_test': len(test_idx), 'selected_lr': lr, 'fit_epochs': fit_epochs,
                 'inner_cv_macro_f1_mean': float(chosen['mean_macro_f1']),
                 'confusion_matrix': cm.tolist(), **metrics})
        print(f'  SELECTED LR={lr} epochs={fit_epochs}, test Macro-F1={metrics["macro_f1"]:.5f}', flush=True)
        summarise(output)
    summarise(output)
    print('Results:', output)


if __name__ == '__main__':
    main()
