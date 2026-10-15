"""Paired FT-Transformer sensor-dropout augmentation experiment.

- original outer-train rows + one corrupted copy of every row
- corrupted copy: k sampled uniformly from {1,...,8} per row
- k distinct sensors selected randomly per row (not test patterns)
- each failed raw sensor channel replaced by outer-train mean
- fit FeaturePipeline scaler on CLEAN outer training only
- keep the baseline final model's random seed for initialisation
- test on the same clean and shared 77 fixed sensor-failure patterns
- report k-wise mean over masks WITHIN repeat, then mean/SD across repeats.

Unlike ordinary neural Dropout, this augments the *raw sensor readings*.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
for folder in (ROOT, ROOT / 'shared'):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import numpy as np
import pandas as pd
import torch

from shared_protocol import RAW_FEATURES, SEED, N_CLASSES, N_SENSORS, TrainOnlyStandardScaler, get_outer_split, load_phase1
from shared_features import FeaturePipeline, feature_names
from shared_metrics import classification_metrics_from_scratch
from shared_robustness import (failed_feature_positions, load_failure_patterns,
                               corrupt_raw_with_outer_train_mean)
from ft_transformer import final_fit, predict, new_model, device_from_choice
import robustness_ft as base_rob

VERSION = 'FT_SENSOR_DROPOUT_AUGMENTATION_V1'


def resolve(root: Path, p: Path) -> Path:
    return (p if p.is_absolute() else root / p).resolve()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj: object) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project-dir', type=Path, default=ROOT)
    p.add_argument('--source-dir', type=Path, default=Path('ft_results'))
    p.add_argument('--patterns', type=Path, default=Path('shared/sensor_failure_patterns.json'))
    p.add_argument('--output-dir', type=Path, default=Path('ft_sensor_dropout_results'))
    p.add_argument('--baseline-robustness-dir', type=Path, default=Path('ft_robustness_results'))
    p.add_argument('--mode', choices=('validate', 'train', 'evaluate', 'all'), default='all')
    p.add_argument('--device', choices=('auto', 'cuda', 'cpu', 'mps'), default='auto')
    p.add_argument('--repeats', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--batch-size', type=int, default=None,
                   help='Training batch size; by default use baseline FT training batch size')
    p.add_argument('--inference-batch-size', type=int, default=64)
    return p.parse_args()


def make_corrupted_copy(raw_train: np.ndarray, raw_mean: np.ndarray,
                        repeat: int) -> tuple[np.ndarray, np.ndarray]:
    """Exact sensor-mask sampling scheme used by the teammate's SVM script."""
    raw_train = np.asarray(raw_train, dtype=np.float64)
    raw_mean = np.asarray(raw_mean, dtype=np.float64)
    if raw_train.ndim != 2 or raw_train.shape[1] != 128 or raw_mean.shape != (128,):
        raise ValueError('Expected clean raw training (n,128) and mean (128,)')
    rng = np.random.default_rng([SEED, 1, int(repeat)])
    augmented = raw_train.copy()
    k_values = np.empty(len(augmented), dtype=np.int64)
    for i in range(len(augmented)):
        k = int(rng.integers(1, 9))
        failed = tuple(int(s) for s in rng.choice(np.arange(1, N_SENSORS+1), size=k, replace=False))
        cols = failed_feature_positions(failed)
        augmented[i, cols] = raw_mean[cols]
        k_values[i] = k
    return augmented, k_values


def source_checkpoint(source: Path, repeat: int, signature: str,
                      baseline_row: pd.Series, meta: dict) -> dict:
    cp = source / 'outer_runs' / f'ft_repeat_{repeat:02d}.pt'
    ckpt = torch.load(cp, map_location='cpu', weights_only=True)
    expected_arch = {k: meta['configuration'][k] for k in ('d_token', 'n_heads','n_layers','ffn_mult','dropout')}
    if (ckpt.get('experiment_signature') != signature or
        ckpt.get('outer_repeat') != repeat or ckpt.get('n_features') != 396 or
        ckpt.get('architecture') != expected_arch or
        not np.isclose(float(ckpt['learning_rate']), float(baseline_row['selected_lr'])) or
        int(ckpt['epochs']) != int(baseline_row['fit_epochs'])):
        raise RuntimeError(f'Incompatible baseline FT checkpoint: repeat {repeat}')
    for name, dim in (('scaler_mean',396),('scaler_scale',396),('raw_outer_train_mean',128)):
        if tuple(np.asarray(ckpt[name]).shape) != (dim,):
            raise ValueError(f'Invalid {name} for repeat {repeat}')
    return ckpt


def get_scaler_from_base(base_ckpt: dict) -> FeaturePipeline:
    pipe = FeaturePipeline()
    pipe.scaler_ = TrainOnlyStandardScaler()
    pipe.scaler_.mean_ = np.asarray(base_ckpt['scaler_mean'], dtype=np.float64).copy()
    pipe.scaler_.scale_ = np.asarray(base_ckpt['scaler_scale'], dtype=np.float64).copy()
    return pipe


def validate_source(project: Path, source: Path, patterns_path: Path,
                    repeats: list[int]):
    meta = load_json(source / 'run_metadata.json')
    # Already tests original data + shared code SHA256 signatures, feature order etc.
    base_rob.check_source(project, source, meta)
    data, splits = load_phase1(project)
    raw = data[RAW_FEATURES].to_numpy(dtype=np.float64)
    if raw.shape[1] != 128 or not np.isfinite(raw).all():
        raise RuntimeError('Invalid raw sensor input')
    classes = sorted(pd.unique(data['target']).tolist())
    if [str(x) for x in classes] != meta['configuration']['class_labels']:
        raise RuntimeError('Class encoding differs from baseline')
    y_map = {v:i for i,v in enumerate(classes)}
    y = np.array([y_map[c] for c in data['target']], dtype=np.int64)
    patterns = load_failure_patterns(patterns_path)
    base_rob.validate_patterns(patterns)
    expected_levels = {0:1,1:16,2:20,4:20,8:20}
    if {k:len(v) for k,v in patterns.items()} != expected_levels:
        raise RuntimeError('Shared failure mask counts changed')
    base_df = pd.read_csv(source / 'outer_test_metrics.csv')
    if len(base_df) != 10 or sorted(base_df.outer_repeat.astype(int).tolist()) != list(range(1,11)):
        raise RuntimeError('Expected full 10-repeat base FT results')
    for r in repeats:
        ckpt = source_checkpoint(source,r,meta['signature'],
                                 base_df.loc[base_df.outer_repeat==r].iloc[0],meta)
        split = get_outer_split(splits,r)
        tr = np.asarray(split['train_indices'],dtype=np.int64)
        if len(tr) != int(base_df.loc[base_df.outer_repeat==r, 'n_train'].iloc[0]):
            raise RuntimeError(f'Base outer training count mismatch repeat {r}')
        mean = raw[tr].mean(axis=0)
        if not np.allclose(mean,np.asarray(ckpt['raw_outer_train_mean']),rtol=1e-9,atol=1e-8):
            raise RuntimeError(f'Training raw imputation mean mismatch repeat {r}')
    return meta, data, raw, y, splits, patterns, base_df


def train_one(r: int, raw: np.ndarray, y: np.ndarray, tr: np.ndarray,
              cfg: SimpleNamespace, base_ckpt: dict, device: torch.device,
              batch_size: int, output: Path, signature: str,
              baseline_sha: str) -> Path:
    cp_out = output / 'models' / f'ft_dropout_repeat_{r:02d}.pt'
    if cp_out.exists():
        saved = torch.load(cp_out, map_location='cpu', weights_only=True)
        if saved.get('experiment_signature') != signature or saved.get('baseline_checkpoint_sha256') != baseline_sha:
            raise RuntimeError(f'Stale sensor-dropout checkpoint {cp_out}; use new output folder.')
        print(f'Repeat {r}: augmented checkpoint already exists, skipping training', flush=True)
        return cp_out
    clean = raw[tr]
    means = clean.mean(axis=0)
    pipe = FeaturePipeline().fit(clean)
    if not np.allclose(pipe.scaler_.mean_, np.asarray(base_ckpt['scaler_mean']), rtol=1e-9, atol=1e-8):
        raise RuntimeError(f'Repeat {r}: different train-only feature means from baseline')
    if not np.allclose(pipe.scaler_.scale_, np.asarray(base_ckpt['scaler_scale']), rtol=1e-9, atol=1e-8):
        raise RuntimeError(f'Repeat {r}: different train-only feature scales from baseline')
    corrupted, k = make_corrupted_copy(clean, means, r)
    Z = pipe.transform(np.vstack([clean, corrupted]))
    y_aug = np.concatenate([y[tr],y[tr]])
    train_cfg = SimpleNamespace(**vars(cfg))
    train_cfg.batch_size = batch_size
    lr = float(base_ckpt['learning_rate'])
    epochs = int(base_ckpt['epochs'])
    # Same initialization seed as in final baseline training.
    seed = SEED + r * 10000 + 9999
    print(f'\nTRAIN repeat {r}/10 | {len(clean)} clean + {len(corrupted)} corrupted; '
          f'LR={lr}, epochs={epochs}, device={device}',flush=True)
    net = final_fit(Z,y_aug,train_cfg,lr,epochs,seed,device)
    cp_out.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model_state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},
                'outer_repeat':r,'experiment_signature':signature,
                'baseline_checkpoint_sha256':baseline_sha,'baseline_experiment_signature':base_ckpt['experiment_signature'],
                'architecture':base_ckpt['architecture'],'n_features':396,
                'scaler_mean':torch.tensor(pipe.scaler_.mean_,dtype=torch.float64),
                'scaler_scale':torch.tensor(pipe.scaler_.scale_,dtype=torch.float64),
                'raw_outer_train_mean':torch.tensor(means,dtype=torch.float64),
                'learning_rate':lr,'epochs':epochs,'class_labels':base_ckpt['class_labels'],
                'augmentation':{'clean_rows':len(clean),'augmented_rows':len(corrupted),
                               'min_failed_sensors':1,'max_failed_sensors':8,
                               'mask_rng_seed':[SEED,1,r], 'mask_count_histogram':{str(i):int((k==i).sum()) for i in range(1,9)},
                               'scaler_fit_on':'clean_outer_train_only'}},cp_out)
    print(f'Saved: {cp_out}',flush=True)
    return cp_out


def evaluate_one(r: int, ckpt: dict, data: pd.DataFrame, raw: np.ndarray, y: np.ndarray,
                 te: np.ndarray, patterns: dict, device: torch.device,
                 batch_size: int) -> dict:
    pipe = get_scaler_from_base(ckpt)
    net = new_model(SimpleNamespace(**ckpt['architecture']),396).to(device)
    net.load_state_dict(ckpt['model_state_dict'],strict=True)
    net.eval()
    Xte = raw[te]
    yte = y[te]
    train_mean = np.asarray(ckpt['raw_outer_train_mean'],dtype=np.float64)
    class_labels = ckpt['class_labels']
    rows, cm_records, class_rows = [], [], []
    clean_f1 = None
    print(f'EVALUATE repeat {r}/10 | 77 fixed patterns',flush=True)
    for k in sorted(patterns):
        for pid, failed in enumerate(patterns[k], start=1):
            corrupted = corrupt_raw_with_outer_train_mean(Xte, failed, train_mean, RAW_FEATURES)
            Zte = pipe.transform(corrupted)
            pred = predict(net,Zte,batch_size,device)
            metrics, cm = classification_metrics_from_scratch(yte,pred,N_CLASSES)
            if k == 0:
                clean_f1 = float(metrics['macro_f1'])
            fstr = ','.join(map(str,failed))
            rows.append({'outer_repeat':r,'n_failed_sensors':k,
                         'n_available_sensors':N_SENSORS-k,
                         'pattern_id':pid,'failed_sensors':fstr,**metrics})
            cm_records.append({'outer_repeat':r,'n_failed_sensors':k,'pattern_id':pid,
                               'failed_sensors':list(failed),'confusion_matrix':cm.tolist()})
            class_rows.extend(base_rob.classwise_from_cm(cm,r,k,pid,fstr,class_labels))
        mean_f1 = float(np.mean([x['macro_f1'] for x in rows if x['n_failed_sensors']==k]))
        print(f'  k={k}: mean Macro-F1={mean_f1:.5f}',flush=True)
    if clean_f1 is None:
        raise RuntimeError('Missing clean condition')
    for row in rows:
        row['macro_f1_drop_from_clean'] = float(clean_f1-row['macro_f1'])
        row['macro_f1_retained_fraction'] = float(row['macro_f1']/clean_f1) if clean_f1>0 else None
    return {'outer_repeat':r,'pattern_rows':rows,'confusion_records':cm_records,
            'class_rows':class_rows,
            'clean_consistency':{'passed':True,'dropout_clean_macro_f1':clean_f1,
                                  'n_test':len(te),
                                  'note':'Augmented model is not expected to reproduce baseline clean predictions'}}


def aggregate(output: Path, signatures: dict[int,str], baseline_dir: Path) -> None:
    cache_dir = output / 'repeat_results'
    if not cache_dir.exists():
        return
    payloads = []
    for path in sorted(cache_dir.glob('repeat_??.json')):
        obj=load_json(path)
        r=int(obj['outer_repeat'])
        if (obj.get('experiment_signature')!=signatures.get(r) or
            obj.get('n_patterns')!=77 or len(obj.get('pattern_rows',[]))!=77):
            raise RuntimeError(f'Invalid dropout eval cache: {path}')
        payloads.append(obj)
    if not payloads:
        return
    folder = output / 'robustness'
    folder.mkdir(parents=True,exist_ok=True)
    # Reuse same aggregation as original FT robustness script: mask-average within
    # repeat and then mean + sample SD across outer repeats.
    base_rob.aggregate(payloads,folder)
    out_summary = pd.read_csv(folder / 'ft_final_robustness_summary.csv')
    out_summary.insert(0,'algorithm','ft_transformer_sensor_dropout')
    out_summary.to_csv(output / 'ft_dropout_robustness_summary.csv',index=False)
    old_path = baseline_dir / 'ft_sensor_failure_by_repeat.csv'
    new_path = folder / 'ft_sensor_failure_by_repeat.csv'
    if old_path.exists():
        b = pd.read_csv(old_path)
        n = pd.read_csv(new_path)
        joined = n.merge(b[['outer_repeat','n_failed_sensors','macro_f1_mean_across_patterns']],
                         on=['outer_repeat','n_failed_sensors'],suffixes=('_dropout','_baseline'),validate='one_to_one')
        joined['macro_f1_improvement']=(joined['macro_f1_mean_across_patterns_dropout']-
                                         joined['macro_f1_mean_across_patterns_baseline'])
        joined.to_csv(output/'ft_dropout_vs_baseline_by_repeat.csv',index=False)
        rows=[]
        for k, sub in joined.groupby('n_failed_sensors'):
            vals=sub['macro_f1_improvement'].to_numpy(dtype=np.float64)
            rows.append({'n_failed_sensors':int(k),'n_outer_repeats':len(vals),
                         'mean_macro_f1_improvement':float(vals.mean()),
                         'sd_macro_f1_improvement':float(vals.std(ddof=1)) if len(vals)>1 else 0.,
                         'baseline_mean_macro_f1':float(sub['macro_f1_mean_across_patterns_baseline'].mean()),
                         'dropout_mean_macro_f1':float(sub['macro_f1_mean_across_patterns_dropout'].mean())})
        compare = pd.DataFrame(rows).sort_values('n_failed_sensors')
        compare.to_csv(output/'ft_dropout_vs_baseline_summary.csv',index=False)
        print('\nPAIRED MACRO-F1 COMPARISON (positive means augmented model improves)\n'+
              compare.to_string(index=False,float_format=lambda x:f'{x:.4f}'),flush=True)
    else:
        print(f'Baseline robustness per-repeat file not found: {old_path}; comparison CSV skipped.',flush=True)


def main() -> None:
    args=arguments()
    project=args.project_dir.resolve()
    source=resolve(project,args.source_dir)
    output=resolve(project,args.output_dir)
    baseline_dir=resolve(project,args.baseline_robustness_dir)
    patterns_file=resolve(project,args.patterns)
    repeats=sorted(set(args.repeats))
    if len(repeats)!=len(args.repeats) or not repeats or any(r not in range(1,11) for r in repeats):
        raise ValueError('Repeats must be unique and in 1..10')
    if args.inference_batch_size<1 or (args.batch_size is not None and args.batch_size<1):
        raise ValueError('Batch sizes must be positive')
    meta,data,raw,y,splits,patterns,base_df = validate_source(project,source,patterns_file,repeats)
    base_cfg=SimpleNamespace(**meta['configuration'])
    train_batch_size=args.batch_size or int(base_cfg.batch_size)
    manifest={'protocol_version':VERSION,'base_signature':meta['signature'],
              'source_result_manifest_sha256':sha(source/'run_metadata.json'),
              'patterns_sha256':sha(patterns_file),
              'training_augmentation':'clean plus one copy per training row, k ~ discrete uniform 1..8',
              'mask_rng':'numpy.default_rng([SEED,1,outer_repeat])',
              'sensor_values':'outer-training raw mean before 396-D feature construction',
              'pipeline':'fit on clean outer train only',
              'hyperparameters':'reuse same selected LR, epochs and architecture as original FT',
              'training_batch_size':train_batch_size,'inference_batch_size':args.inference_batch_size}
    sig=digest(manifest)
    manifest['signature']=sig
    ckpt_base_sha={r:sha(source/'outer_runs'/f'ft_repeat_{r:02d}.pt') for r in repeats}
    if output.exists() and any(output.iterdir()):
        previous=output/'run_metadata.json'
        if not previous.exists() or load_json(previous)!=manifest:
            raise RuntimeError(f'Existing output directory has incompatible metadata: {output}')
    print(f'Validated source: {len(data)} rows, 10 baseline repeats, 77 shared failure patterns.',flush=True)
    if args.mode == 'validate':
        print('VALIDATION PASSED. No training or prediction performed.',flush=True)
        return
    device=device_from_choice(args.device)
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'run_metadata.json').exists():
        save_json(output/'run_metadata.json',manifest)
    source_signatures={}
    for r in repeats:
        split=get_outer_split(splits,r)
        tr=np.asarray(split['train_indices'],dtype=np.int64)
        te=np.asarray(split['test_indices'],dtype=np.int64)
        base_row=base_df.loc[base_df.outer_repeat==r].iloc[0]
        base_ckpt=source_checkpoint(source,r,meta['signature'],base_row,meta)
        dest=output/'models'/f'ft_dropout_repeat_{r:02d}.pt'
        if args.mode in ('train','all'):
            train_one(r,raw,y,tr,base_cfg,base_ckpt,device,train_batch_size,output,sig,ckpt_base_sha[r])
        if args.mode in ('evaluate','all'):
            if not dest.exists():
                raise FileNotFoundError(f'No augmented model for repeat {r}: {dest}. Run --mode train first.')
            dropped_ckpt=torch.load(dest,map_location='cpu',weights_only=True)
            if (dropped_ckpt.get('experiment_signature')!=sig or
                dropped_ckpt.get('baseline_checkpoint_sha256')!=ckpt_base_sha[r] or
                dropped_ckpt.get('outer_repeat')!=r):
                raise RuntimeError(f'Wrong dropout checkpoint for repeat {r}')
            cache=output/'repeat_results'/f'repeat_{r:02d}.json'
            cphash=sha(dest)
            source_signatures[r]=cphash
            if cache.exists():
                c=load_json(cache)
                if (c.get('experiment_signature')!=cphash or c.get('n_patterns')!=77 or
                    len(c.get('pattern_rows',[]))!=77):
                    raise RuntimeError(f'Stale dropout robustness cache: {cache}')
                print(f'Repeat {r}: robustness already cached, skipping',flush=True)
            else:
                print(f'\nDROPOUT ROBUSTNESS | repeat {r}/10',flush=True)
                result=evaluate_one(r,dropped_ckpt,data,raw,y,te,patterns,device,args.inference_batch_size)
                result.update({'experiment_signature':cphash,'n_patterns':77})
                save_json(cache,result)
            # Recheck hash for every previous repeat that may appear in resumed results
            for old in (output/'repeat_results').glob('repeat_??.json'):
                rid=int(old.stem.split('_')[1])
                model_path=output/'models'/f'ft_dropout_repeat_{rid:02d}.pt'
                if model_path.exists():
                    source_signatures[rid]=sha(model_path)
            aggregate(output,source_signatures,baseline_dir)
    if args.mode=='train':
        print('Training done. Next: --mode evaluate --device cuda',flush=True)
    else:
        print(f'Outputs: {output}',flush=True)

if __name__ == '__main__':
    main()
