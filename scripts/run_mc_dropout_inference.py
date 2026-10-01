"""Run native MC Dropout inference from a retained traffic checkpoint.

This command is deliberately separate from the manuscript-generation workflow.
It uses the archived configuration, test partition, training-fitted scaler, and
checkpoint for one model/dataset/seed case. It writes a new artifact directory
and never overwrites the existing summary JSON files.
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
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
for path in (ROOT, FRAMEWORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

Z_90 = 1.6448536269514722


def load_config(model: str, dataset: str, seed: int):
    path = ROOT / "configs" / model / f"{dataset}_seed{seed}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Configuration not found: {path}")

    module_name = f"configs.{model}._mc_{dataset.replace('-', '_')}_seed{seed}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load configuration: {path}")
    module = importlib.util.module_from_spec(spec)
    module.__package__ = f"configs.{model}"
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CFG


def enable_native_dropout(model: nn.Module) -> int:
    """Keep the model in evaluation mode while enabling registered dropout modules.

    nn.Dropout instances are the common case, but nn.MultiheadAttention (used by
    D2STGNN's inherent-block temporal attention) applies its own attention-weight
    dropout internally, gated by the module's own .training flag rather than a
    child nn.Dropout submodule. Setting only nn.Dropout instances to train mode
    leaves that attention dropout permanently disabled (silently under-covering
    the stochastic forward pass) because model.eval() already set it to False and
    nothing here would flip it back. Both module types are enabled here so a
    model's whole registered dropout surface participates in MC Dropout.
    """
    model.eval()
    plain_dropout = [module for module in model.modules() if isinstance(module, nn.Dropout)]
    attention_dropout = [
        module for module in model.modules()
        if isinstance(module, nn.MultiheadAttention) and module.dropout > 0
    ]
    for module in plain_dropout + attention_dropout:
        module.train()

    # MTGNN (and potentially other legacy architectures) applies dropout via a bare
    # F.dropout(x, self.dropout, training=self.training) call written directly in the
    # top-level forward(), where self.dropout is a plain float, not an nn.Dropout
    # submodule. That pattern is invisible to the scan above and stays permanently
    # disabled once model.eval() runs, unless the top-level module's own .training
    # flag is restored. This is only safe to do when nothing else in the model reads
    # .training for eval-sensitive behavior (e.g. BatchNorm running statistics) -
    # verified for MTGNN, which has no BatchNorm anywhere in its architecture.
    functional_dropout_active = False
    has_batchnorm = any(
        isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
        for module in model.modules()
    )
    top_level_dropout = getattr(model, "dropout", None)
    if isinstance(top_level_dropout, float) and top_level_dropout > 0 and not has_batchnorm:
        model.training = True
        functional_dropout_active = True

    return len(plain_dropout) + len(attention_dropout) + (1 if functional_dropout_active else 0)


def load_model(cfg, checkpoint_path: Path, device: torch.device) -> nn.Module:
    model = cfg.MODEL.ARCH(**cfg.MODEL.PARAM).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    return model


def run_case(model_name: str, dataset: str, seed: int, n_passes: int, batch_size: int, device: torch.device) -> Path:
    cfg = load_config(model_name, dataset, seed)
    checkpoint_path = ROOT / "checkpoints" / model_name / f"{dataset}_seed{seed}" / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    dataset_obj = cfg.DATASET.TYPE(mode="test", **cfg.DATASET.PARAM)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
    scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)
    model = load_model(cfg, checkpoint_path, device)
    dropout_modules = enable_native_dropout(model)

    total_batches = len(loader)
    print(
        f"  {model_name}/{dataset}/seed{seed}: {total_batches} batches x {n_passes} "
        f"stochastic passes on {device} (no per-pass progress below this line until each batch finishes)",
        flush=True,
    )

    predictions: list[np.ndarray] = []
    predictive_std: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    pass_digests = [hashlib.sha256() for _ in range(n_passes)]

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            history = batch["inputs"].to(device=device, dtype=torch.float32)
            future = batch["target"].to(device=device, dtype=torch.float32)
            history = scaler.transform(history.clone())
            future = scaler.transform(future.clone())
            history = history[..., cfg.MODEL.FORWARD_FEATURES]
            decoder_future = future[..., cfg.MODEL.FORWARD_FEATURES].clone()
            decoder_future[..., 0] = torch.empty_like(decoder_future[..., 0])

            batch_passes = []
            for pass_index in range(n_passes):
                output = model(
                    history_data=history,
                    future_data=decoder_future,
                    batch_seen=None,
                    epoch=None,
                    train=False,
                )
                prediction = output["prediction"] if isinstance(output, dict) else output
                prediction = scaler.inverse_transform(prediction).detach().cpu().numpy()
                pass_digests[pass_index].update(prediction.tobytes())
                batch_passes.append(prediction)

            samples = np.stack(batch_passes, axis=0)
            predictions.append(samples.mean(axis=0).astype(np.float32))
            predictive_std.append(samples.std(axis=0, ddof=0).astype(np.float32))
            targets.append(scaler.inverse_transform(future[..., cfg.MODEL.TARGET_FEATURES]).cpu().numpy().astype(np.float32))

            if (batch_index + 1) % max(1, total_batches // 10) == 0 or batch_index + 1 == total_batches:
                print(f"    batch {batch_index + 1}/{total_batches}", flush=True)

    mean_prediction = np.concatenate(predictions, axis=0)
    std_prediction = np.concatenate(predictive_std, axis=0)
    target = np.concatenate(targets, axis=0)
    lower = mean_prediction - Z_90 * std_prediction
    upper = mean_prediction + Z_90 * std_prediction

    coverage = float(np.mean((target >= lower) & (target <= upper)))
    width = float(np.mean(upper - lower))
    variance = float(np.mean(np.square(std_prediction)))
    status = "functional" if variance > 1e-12 else "degenerate"

    output_dir = ROOT / "results" / "task2_uncertainty" / "mc_dropout_generated" / model_name / dataset / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / f"{model_name}_{dataset}_seed{seed}_{n_passes}pass_moments.npz",
        mean_prediction=mean_prediction,
        predictive_std=std_prediction,
        target=target,
    )
    summary = {
        "model": model_name,
        "dataset": dataset,
        "seed": seed,
        "n_passes": n_passes,
        "device": str(device),
        "dropout_module_count": dropout_modules,
        "test_origins": int(mean_prediction.shape[0]),
        "interval_z_value": Z_90,
        "variance_mean": variance,
        "picp_uncalibrated": coverage,
        "mpiw_uncalibrated": width,
        "status": status,
        "pass_prediction_sha256": [digest.hexdigest() for digest in pass_digests],
        "artifact": f"{model_name}_{dataset}_seed{seed}_{n_passes}pass_moments.npz",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Native MC Dropout checkpoint inference")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True, choices=("METR-LA", "PEMS-BAY", "PEMS04"))
    parser.add_argument("--seed", type=int, default=43, choices=(43, 44, 45))
    parser.add_argument("--passes", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.passes < 2:
        raise ValueError("--passes must be at least 2.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    output_dir = run_case(
        args.model,
        args.dataset,
        args.seed,
        args.passes,
        args.batch_size,
        torch.device(args.device),
    )
    print(f"Wrote generated MC Dropout artifacts to {output_dir}")


if __name__ == "__main__":
    main()
