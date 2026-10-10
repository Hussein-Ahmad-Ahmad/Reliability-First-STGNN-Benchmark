"""Generate a clean, deterministic (no MC Dropout) inference dump for one
model/dataset/seed, matching the format run_block_bootstrap_pairwise.py and
related scripts expect under results/task1_point_forecasting/fresh_inference_dumps/.

Uses an existing checkpoint for inference only, without retraining.
Single deterministic forward pass per
batch, model.eval() throughout (no dropout sampling) - this is a point
prediction, not MC Dropout.

Usage:
    python scripts/generate_fresh_inference_dump.py --model MegaCRN --dataset PEMS04 --seed 44
    python scripts/generate_fresh_inference_dump.py --model MegaCRN --dataset PEMS04 --seed 45
    python scripts/generate_fresh_inference_dump.py --model MTGNN --dataset PEMS04 --seed 44
    python scripts/generate_fresh_inference_dump.py --model MTGNN --dataset PEMS04 --seed 45
    python scripts/generate_fresh_inference_dump.py --model MegaCRN --dataset PEMS-BAY --seed 43
    ... (seeds 43/44/45 for MegaCRN and MTGNN on PEMS-BAY)

Output:
    results/task1_point_forecasting/fresh_inference_dumps/<model>/<dataset>/seed<seed>/predictions.npy
    results/task1_point_forecasting/fresh_inference_dumps/<model>/<dataset>/seed<seed>/targets.npy

Also verifies the clean-pass MAE against the archived checkpoints/<model>/<dataset>_seed<seed>/test_metrics.json,
matching the same tolerance-checked pattern used elsewhere in this repo (pipelines/run_sensor_dropout.py).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
for path in (ROOT, FRAMEWORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

OUTPUT_ROOT = ROOT / "results" / "task1_point_forecasting" / "fresh_inference_dumps"


def _load_mc_dropout_module():
    spec = importlib.util.spec_from_file_location(
        "run_mc_dropout_inference", ROOT / "scripts" / "run_mc_dropout_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_mc_dropout_module()


def run_clean_inference(model_name: str, dataset: str, seed: int, batch_size: int, device: torch.device,
                        archive_root: Path | None = None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if archive_root is None:
        cfg = mc.load_config(model_name, dataset, seed)
        config_path = ROOT / 'configs' / model_name / f'{dataset}_seed{seed}.py'
        checkpoint_path = ROOT / "checkpoints" / model_name / f"{dataset}_seed{seed}" / "best_model.pt"
        metrics_path = checkpoint_path.parent / 'test_metrics.json'
        source_root = ROOT
    else:
        sys.path.insert(0, str(archive_root))
        from pipelines.run_sensor_dropout import find_run, load_config
        run, config_path, checkpoint_path = find_run(archive_root, dataset, model_name, seed)
        cfg = load_config(config_path, f'export_{model_name}_{dataset}_{seed}',
                          fallback_paths=[archive_root / 'baselines' / model_name])
        metrics_path = run / 'test_metrics.json'
        source_root = archive_root
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_obj = cfg.DATASET.TYPE(mode="test", **cfg.DATASET.PARAM)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
    scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)
    model = mc.load_model(cfg, checkpoint_path, device)
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            history = batch["inputs"].to(device=device, dtype=torch.float32)
            future = batch["target"].to(device=device, dtype=torch.float32)
            history = scaler.transform(history.clone())
            future_scaled = scaler.transform(future.clone())
            history = history[..., cfg.MODEL.FORWARD_FEATURES]
            decoder_future = future_scaled[..., cfg.MODEL.FORWARD_FEATURES].clone()
            decoder_future[..., 0] = 0.0

            output = model(
                history_data=history,
                future_data=decoder_future,
                batch_seen=None,
                epoch=None,
                train=False,
            )
            prediction = output["prediction"] if isinstance(output, dict) else output
            prediction = scaler.inverse_transform(prediction).detach().cpu().numpy()
            # `future` was never forward-transformed (only `future_scaled` was,
            # for the decoder input) - it is already in raw units, so it must
            # NOT be passed through inverse_transform again here.
            target = future[..., cfg.MODEL.TARGET_FEATURES].cpu().numpy()
            predictions.append(prediction.astype(np.float32))
            targets.append(target.astype(np.float32))

    pred = np.concatenate(predictions, axis=0)
    targ = np.concatenate(targets, axis=0)
    # cfg output shape is [origins, horizon, num_nodes, 1] typically; squeeze to
    # [origins, horizon, num_nodes] to match reshape_prediction's expectations.
    if pred.ndim == 4 and pred.shape[-1] == 1:
        pred = pred[..., 0]
    if targ.ndim == 4 and targ.shape[-1] == 1:
        targ = targ[..., 0]
    metadata = {
        'model': model_name, 'dataset': dataset, 'seed': seed,
        'checkpoint_selection': ('minimum validation MAE' if archive_root else
                                 'provided checkpoint; selection provenance is not established by its filename'),
        'config': config_path.relative_to(source_root).as_posix(),
        'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'checkpoint': checkpoint_path.relative_to(source_root).as_posix(),
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'test_metrics': metrics_path.relative_to(source_root).as_posix(),
        'test_metrics_sha256': hashlib.sha256(metrics_path.read_bytes()).hexdigest(),
        'test_origins': int(len(dataset_obj)), 'batch_size': batch_size,
        'array_order': 'chronological forecast origin, horizon, sensor',
        'training_performed': False,
        'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU',
    }
    with metrics_path.open(encoding='utf-8') as handle:
        metadata['archived_mae'] = json.load(handle)['overall']['MAE']
    return pred, targ, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean deterministic inference dump")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=("METR-LA", "PEMS-BAY", "PEMS04"))
    parser.add_argument("--seed", type=int, required=True, choices=(43, 44, 45))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument('--archive-root', type=Path, help='Original BasicTS run tree with validation-selected checkpoints')
    args = parser.parse_args()

    device = torch.device(args.device)
    pred, targ, metadata = run_clean_inference(args.model, args.dataset, args.seed, args.batch_size, device,
                                              args.archive_root)

    null_val = 0.0
    valid = np.isfinite(targ) & ~np.isclose(targ, null_val, atol=5e-5, rtol=0.0)
    if not valid.any() or not np.isfinite(pred).all():
        raise ValueError('Inference must contain finite predictions and valid targets.')
    recomputed_mae = float(np.abs(pred - targ)[valid].mean(dtype=np.float64))

    archived_mae = metadata['archived_mae']
    if abs(recomputed_mae - archived_mae) > 1e-5:
        raise RuntimeError(f'Clean-pass mismatch: {recomputed_mae:.9f} versus {archived_mae:.9f}; no export written.')
    metadata.update({'exported_mae': recomputed_mae, 'mae_difference': recomputed_mae - archived_mae,
                     'verification_absolute_tolerance': 1e-5, 'verified': True,
                     'device': str(device), 'torch_version': torch.__version__,
                     'null_atol': 5e-5, 'null_rtol': 0.0, 'valid_target_count': int(valid.sum())})

    out_dir = args.output_root / args.model / args.dataset / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", pred)
    np.save(out_dir / "targets.npy", targ)
    metadata['predictions_sha256'] = hashlib.sha256((out_dir / 'predictions.npy').read_bytes()).hexdigest()
    metadata['targets_sha256'] = hashlib.sha256((out_dir / 'targets.npy').read_bytes()).hexdigest()
    (out_dir / 'export_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')

    print(f"{args.model}/{args.dataset}/seed{args.seed}: shape={pred.shape}")
    print(f"  recomputed clean-pass MAE (pooled, atol=5e-5, rtol=0) = {recomputed_mae:.6f}")
    if archived_mae is not None:
        print(f"  archived test_metrics.json MAE                                = {archived_mae:.6f}")
        print(f"  difference = {recomputed_mae - archived_mae:+.6f}")
    print(f"Wrote {out_dir / 'predictions.npy'}")
    print(f"Wrote {out_dir / 'targets.npy'}")


if __name__ == "__main__":
    main()
