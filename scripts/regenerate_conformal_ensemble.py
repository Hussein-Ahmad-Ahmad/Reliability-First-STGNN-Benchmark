"""Re-evaluate fixed ensemble members using their identified checkpoints."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'framework'))

from pipelines.run_sensor_dropout import load_config
from scripts.run_mc_dropout_inference import load_model
from scripts.generate_conformal_intervals import compute_conformal_metrics
from scripts.run_conformal_sigma_control import compute_plain_control


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(part)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--member-config', action='append', default=[], help='MODEL=CONFIG for a missing historical config')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--array-dir', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.path.insert(0, str(args.archive_root))
    supplied = dict(value.split('=', 1) for value in args.member_config)
    manifest = json.loads(args.manifest.read_text(encoding='utf-8'))
    predictions, records, common_target = [], [], None
    device = torch.device(args.device)
    for member in manifest['member_order']:
        name = member['model']
        checkpoint_path = args.archive_root / member['checkpoint_path']
        if digest(checkpoint_path) != member['checkpoint_sha256']:
            raise ValueError(f'Checkpoint identity mismatch: {name}')
        reconstructed = not member.get('config_path')
        config_path = Path(supplied[name]) if reconstructed else args.archive_root / member['config_path']
        if not reconstructed and digest(config_path) != member['config_sha256']:
            raise ValueError(f'Configuration identity mismatch: {name}')
        fallback = args.archive_root / 'baselines' / ('STGCN' if name == 'STGCNChebGraphConv' else name)
        cfg = load_config(config_path, f'ensemble_export_{name}', fallback_paths=[fallback])
        dataset = cfg.DATASET.TYPE(mode='test', **cfg.DATASET.PARAM)
        scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)
        model = load_model(cfg, checkpoint_path, device)
        model.eval()
        batches, targets = [], []
        print(f'{name}: complete chronological inference', flush=True)
        with torch.no_grad():
            for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0):
                history = scaler.transform(batch['inputs'].to(device).clone())[..., cfg.MODEL.FORWARD_FEATURES]
                future = batch['target'].to(device)
                decoder = scaler.transform(future.clone())[..., cfg.MODEL.FORWARD_FEATURES].clone()
                decoder[..., 0] = 0.0
                result = model(history_data=history, future_data=decoder, batch_seen=None, epoch=None, train=False)
                prediction = result['prediction'] if isinstance(result, dict) else result
                batches.append(scaler.inverse_transform(prediction).cpu().numpy().squeeze(-1))
                targets.append(future[..., cfg.MODEL.TARGET_FEATURES].cpu().numpy().squeeze(-1))
        prediction = np.concatenate(batches).astype(np.float32)
        target = np.concatenate(targets).astype(np.float32)
        if not np.isfinite(prediction).all() or not np.isfinite(target).all():
            raise ValueError(f'Nonfinite ensemble arrays: {name}')
        if common_target is not None and not np.array_equal(common_target, target):
            raise ValueError(f'Ensemble target alignment mismatch: {name}')
        common_target = target
        valid = ~np.isclose(target, 0, atol=5e-5, rtol=0)
        mae = float(np.abs(prediction - target)[valid].mean(dtype=np.float64))
        metrics_path = checkpoint_path.parent / 'test_metrics.json'
        archived = json.loads(metrics_path.read_text())['overall']['MAE']
        if abs(mae - archived) > 1e-5:
            raise ValueError(f'Clean MAE mismatch for {name}: {mae} versus {archived}')
        args.array_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.array_dir / f'{name}_predictions.npy', prediction)
        records.append({**member,
            'historical_prediction_sha256': member['prediction_sha256'],
            'prediction_path': f'{name}_predictions.npy',
            'prediction_sha256': digest(args.array_dir / f'{name}_predictions.npy'),
            'reexecution_config': config_path.relative_to(args.archive_root).as_posix(),
            'reexecution_config_sha256': digest(config_path),
            'reexecution_config_reconstructed': reconstructed,
            'member_mae': mae, 'archived_online_mae': archived,
            'clean_mae_difference': mae - archived, 'verification_absolute_tolerance': 1e-5})
        predictions.append(prediction)
        del model, batches, targets, dataset, scaler
        torch.cuda.empty_cache()
        gc.collect()
        print(f'{name}: MAE={mae:.9f}, difference={mae - archived:+.3g}', flush=True)
    ensemble = np.stack(predictions)
    mean, std = ensemble.mean(axis=0), ensemble.std(axis=0, ddof=0)
    np.save(args.array_dir / 'ensemble_predictions.npy', ensemble)
    np.save(args.array_dir / 'targets.npy', common_target)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_conformal_metrics(mean, std, common_target, manifest['dataset'], member_count=len(records))
    for variant, result in metrics.items():
        (args.output_dir / f'{manifest["dataset"]}_conformal_{variant}_metrics.json').write_text(
            json.dumps(result, indent=2) + '\n', encoding='utf-8')
    plain = compute_plain_control(mean, common_target, manifest['dataset'], .1)
    normalized = metrics['fixed']
    comparison = {'dataset': manifest['dataset'], 'normalized_split_conformal': normalized,
                  'plain_sigma1_control': plain,
                  'comparison': {'dataset': manifest['dataset'], 'alpha': .1,
                    'normalized_variant': {k: normalized['evaluation_set'][k] for k in ('PICP', 'MPIW')},
                    'plain_sigma1_variant': {k: plain['evaluation_set'][k] for k in ('PICP', 'MPIW')},
                    'picp_difference_normalized_minus_plain': normalized['evaluation_set']['PICP'] - plain['evaluation_set']['PICP'],
                    'mpiw_difference_normalized_minus_plain': normalized['evaluation_set']['MPIW'] - plain['evaluation_set']['MPIW']}}
    (args.output_dir / f'{manifest["dataset"]}_conformal_sigma_control.json').write_text(
        json.dumps(comparison, indent=2) + '\n', encoding='utf-8')
    manifest['selection_rule']['inclusion_rule'] = 'Fixed historical six-member checkpoint set, re-evaluated in a complete chronological pass; no selection using coverage'
    manifest.update({'member_order': records, 'array_export': 'Concatenated chronological batches',
        'targets_sha256': digest(args.array_dir / 'targets.npy'),
        'ensemble_predictions_sha256': digest(args.array_dir / 'ensemble_predictions.npy'),
        'training_performed': False, 'torch_version': torch.__version__,
        'device': str(device), 'batch_size': args.batch_size,
        'cuda_matmul_allow_tf32': False, 'cudnn_allow_tf32': False,
        'source_manifest_sha256': digest(args.manifest),
        'reexecution_note': 'Same identified checkpoint members; any reconstructed configuration is labeled, not represented as a recovered original record'})
    (args.output_dir / f'{manifest["dataset"]}_ensemble_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(comparison['comparison'], indent=2))


if __name__ == '__main__':
    main()
