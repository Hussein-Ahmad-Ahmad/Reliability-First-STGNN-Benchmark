"""Generate a clean, deterministic (no MC Dropout) inference dump for one
model/dataset/seed, matching the format run_block_bootstrap_pairwise.py and
related scripts expect under results/task1_point_forecasting/fresh_inference_dumps/.

This closes specific gaps identified while auditing the C3/C5/C6 bootstrap
and conformal analyses: MegaCRN and MTGNN are missing seed-44/45 dumps for
PEMS04, and missing all three seeds for PEMS-BAY. Both checkpoints already
exist (inference-only; no retraining). Single deterministic forward pass per
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


def run_clean_inference(model_name: str, dataset: str, seed: int, batch_size: int, device: torch.device):
    cfg = mc.load_config(model_name, dataset, seed)
    checkpoint_path = ROOT / "checkpoints" / model_name / f"{dataset}_seed{seed}" / "best_model.pt"
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
            decoder_future[..., 0] = torch.empty_like(decoder_future[..., 0])

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
    return pred, targ


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a clean deterministic inference dump")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=("METR-LA", "PEMS-BAY", "PEMS04"))
    parser.add_argument("--seed", type=int, required=True, choices=(43, 44, 45))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()

    device = torch.device(args.device)
    pred, targ = run_clean_inference(args.model, args.dataset, args.seed, args.batch_size, device)

    null_val = 0.0
    valid = np.isfinite(targ) & ~np.isclose(targ, null_val)
    recomputed_mae = float(np.abs(pred - targ)[valid].mean())

    metrics_path = ROOT / "checkpoints" / args.model / f"{args.dataset}_seed{args.seed}" / "test_metrics.json"
    archived_mae = None
    if metrics_path.is_file():
        archived_mae = json.load(open(metrics_path))["overall"]["MAE"]

    out_dir = args.output_root / args.model / args.dataset / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "predictions.npy", pred)
    np.save(out_dir / "targets.npy", targ)

    print(f"{args.model}/{args.dataset}/seed{args.seed}: shape={pred.shape}")
    print(f"  recomputed clean-pass MAE (pooled, numpy default null tolerance) = {recomputed_mae:.6f}")
    if archived_mae is not None:
        print(f"  archived test_metrics.json MAE                                = {archived_mae:.6f}")
        print(f"  difference = {recomputed_mae - archived_mae:+.6f}")
    print(f"Wrote {out_dir / 'predictions.npy'}")
    print(f"Wrote {out_dir / 'targets.npy'}")


if __name__ == "__main__":
    main()
