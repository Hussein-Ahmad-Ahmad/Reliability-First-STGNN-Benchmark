"""Diagnose which dropout layers MC Dropout actually activates per model (C8).

run_mc_dropout_inference.py's enable_native_dropout() keeps the model in
eval() and then flips specific submodules back to train() so BatchNorm-style
statistics stay frozen while dropout still samples. Before the fix applied
alongside this script, it only looked for nn.Dropout instances. D2STGNN's
inherent-block temporal self-attention uses torch.nn.MultiheadAttention,
which applies its own attention-weight dropout internally gated by the
module's *own* .training flag rather than a child nn.Dropout — so it was
silently left disabled (still eval-mode) under the old coverage rule, even
though the "dropout_module_count" printed as if dropout were fully active.

This script is read-only diagnostics: for every model, it lists the
dropout-bearing modules found by (a) the old nn.Dropout-only rule and
(b) the corrected rule that also enables nn.MultiheadAttention, and it runs
a small number of stochastic passes over the first few test batches under
both rules so you can see whether the fix actually changes predictive
variance for that architecture (it should be a no-op for models that only
use plain nn.Dropout, and should increase variance for D2STGNN).

Usage:
    python scripts/diagnose_mc_dropout_coverage.py --dataset METR-LA --seed 43 --n-passes 10 --n-batches 3
"""

from __future__ import annotations

import argparse
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

MODELS = ("D2STGNN", "MegaCRN", "MTGNN", "STGCNChebGraphConv", "STID", "STNorm", "STAEformer")


def _load_mc_dropout_module():
    spec = importlib.util.spec_from_file_location(
        "run_mc_dropout_inference", Path(__file__).resolve().parent / "run_mc_dropout_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_mc_dropout_module()


def inspect_modules(model: nn.Module) -> dict:
    plain = [(name, module.p) for name, module in model.named_modules() if isinstance(module, nn.Dropout)]
    attention = [
        (name, module.dropout)
        for name, module in model.named_modules()
        if isinstance(module, nn.MultiheadAttention)
    ]
    return {
        "plain_dropout_modules": [{"name": n, "p": float(p)} for n, p in plain],
        "multihead_attention_modules": [{"name": n, "dropout_p": float(p)} for n, p in attention],
        "multihead_attention_dropout_active_after_fix": any(p > 0 for _, p in attention),
    }


def enable_old_rule(model: nn.Module) -> None:
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def few_batch_passes(model, loader, cfg, scaler, device, n_passes: int, n_batches: int) -> dict:
    stds = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= n_batches:
                break
            history = batch["inputs"].to(device=device, dtype=torch.float32)
            future = batch["target"].to(device=device, dtype=torch.float32)
            history = scaler.transform(history.clone())
            future_scaled = scaler.transform(future.clone())
            history = history[..., cfg.MODEL.FORWARD_FEATURES]
            decoder_future = future_scaled[..., cfg.MODEL.FORWARD_FEATURES].clone()
            decoder_future[..., 0] = torch.empty_like(decoder_future[..., 0])

            passes = []
            for _ in range(n_passes):
                output = model(history_data=history, future_data=decoder_future, batch_seen=None, epoch=None, train=False)
                prediction = output["prediction"] if isinstance(output, dict) else output
                passes.append(scaler.inverse_transform(prediction).detach().cpu().numpy())
            samples = np.stack(passes, axis=0)
            stds.append(samples.std(axis=0, ddof=0).mean())
    return {"mean_predictive_std": float(np.mean(stds)) if stds else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose MC Dropout module coverage per model")
    parser.add_argument("--dataset", default="METR-LA")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--n-passes", type=int, default=10)
    parser.add_argument("--n-batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "task2_uncertainty" / "mc_dropout_generated" / "dropout_coverage_diagnostic.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    report = {}

    for model_name in MODELS:
        print(f"=== {model_name} ===")
        cfg = mc.load_config(model_name, args.dataset, args.seed)
        checkpoint_path = ROOT / "checkpoints" / model_name / f"{args.dataset}_seed{args.seed}" / "best_model.pt"
        if not checkpoint_path.is_file():
            print(f"  skip: checkpoint not found at {checkpoint_path}")
            continue

        model = mc.load_model(cfg, checkpoint_path, device)
        module_report = inspect_modules(model)

        dataset_obj = cfg.DATASET.TYPE(mode="test", **cfg.DATASET.PARAM)
        loader = DataLoader(dataset_obj, batch_size=args.batch_size, shuffle=False, num_workers=0)
        scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)

        enable_old_rule(model)
        old_result = few_batch_passes(model, loader, cfg, scaler, device, args.n_passes, args.n_batches)

        mc.enable_native_dropout(model)
        new_result = few_batch_passes(model, loader, cfg, scaler, device, args.n_passes, args.n_batches)

        module_report["old_rule_mean_predictive_std"] = old_result["mean_predictive_std"]
        module_report["fixed_rule_mean_predictive_std"] = new_result["mean_predictive_std"]
        module_report["fix_changes_variance"] = bool(
            abs(new_result["mean_predictive_std"] - old_result["mean_predictive_std"]) > 1e-8
        )
        report[model_name] = module_report
        print(json.dumps(module_report, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
