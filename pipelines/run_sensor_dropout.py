"""Run fixed-mask sensor-dropout inference from archived BasicTS runs.

This pipeline performs checkpoint inference. It does not interpolate or
simulate degradation curves. One nested sensor mask per dataset is shared
across all evaluated models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT / "results" / "robustness" / "sensor_dropout_fixed_masks_seed42.json"
)
DEFAULT_EVALUATIONS = (
    ("METR-LA", "D2STGNN"),
    ("METR-LA", "MegaCRN"),
    ("METR-LA", "STID"),
    ("PEMS-BAY", "D2STGNN"),
    ("PEMS-BAY", "STID"),
    ("PEMS04", "D2STGNN"),
    ("PEMS04", "STID"),
)
SENSOR_COUNTS = {"METR-LA": 207, "PEMS-BAY": 325, "PEMS04": 307}
DROPOUT_RATES = (0.0, 0.10, 0.30)


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_run(root: Path, dataset: str, model: str, seed: int):
    experiment = root / "checkpoints" / model / f"{dataset}_100_12_12_seed{seed}"
    if not experiment.is_dir():
        raise FileNotFoundError(f"Experiment not found: {experiment}")
    runs = sorted(
        path
        for path in experiment.iterdir()
        if path.is_dir() and (path / f"{model}_best_val_MAE.pt").is_file()
    )
    if len(runs) != 1:
        raise RuntimeError(f"Expected one archived run under {experiment}")
    run = runs[0]
    configs = sorted(run.glob(f"*seed{seed}.py"))
    if len(configs) != 1:
        raise RuntimeError(f"Expected one seed-{seed} config under {run}")
    return run, configs[0], run / f"{model}_best_val_MAE.pt"


def load_config(
    path: Path,
    token: str,
    fallback_paths: list[Path] | None = None,
) -> Any:
    """Load a copied config while preserving its relative .arch import."""
    package_name = f"_dropout_{token.replace('-', '_')}"
    package = types.ModuleType(package_name)
    package.__path__ = [
        str(item)
        for item in [path.parent, *(fallback_paths or [])]
    ]
    package.__package__ = package_name
    sys.modules[package_name] = package
    module_name = f"{package_name}.config"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load config: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CFG


def make_masks(num_sensors: int, seed: int) -> dict[float, list[int]]:
    permutation = np.random.RandomState(seed).permutation(num_sensors)
    return {
        rate: sorted(int(index) for index in permutation[: int(num_sensors * rate)])
        for rate in DROPOUT_RATES
    }


def evaluate_rate(runner: Any, dropped: list[int]) -> dict[str, Any]:
    original_preprocessing = runner.preprocessing

    def masked_preprocessing(data):
        data = original_preprocessing(data)
        if dropped:
            data["inputs"][:, :, dropped, :] = 0.0
        return data

    runner.preprocessing = masked_preprocessing
    runner.model.eval()
    error_sum = 0.0
    valid_count = 0
    origins = 0
    try:
        with torch.no_grad():
            for data in runner.test_data_loader:
                result = runner.forward(data, epoch=None, iter_num=None, train=False)
                prediction = result["prediction"]
                target = result["target"]
                valid = torch.isfinite(target) & target.ne(0)
                error_sum += torch.abs(prediction - target)[valid].double().sum().item()
                valid_count += int(valid.sum().item())
                origins += int(target.shape[0])
    finally:
        runner.preprocessing = original_preprocessing
    if valid_count == 0:
        raise RuntimeError("No valid target values were found")
    return {
        "mae": error_sum / valid_count,
        "valid_target_count": valid_count,
        "test_origins": origins,
    }


def reference_mae(run: Path) -> float | None:
    path = run / "test_metrics.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["overall"]["MAE"])


def evaluate_model(
    basicts_root: Path,
    dataset: str,
    model: str,
    checkpoint_seed: int,
    masks: dict[float, list[int]],
) -> dict[str, Any]:
    from easytorch.config import init_cfg

    run, config_path, checkpoint_path = find_run(
        basicts_root, dataset, model, checkpoint_seed
    )
    cfg = init_cfg(
        load_config(config_path, f"{dataset}_{model}_{checkpoint_seed}"),
        save=False,
    )
    runner = cfg.RUNNER(cfg)
    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)
    runner.init_test(cfg)
    runner.load_model(ckpt_path=str(checkpoint_path), strict=True)

    rates = {}
    for rate in DROPOUT_RATES:
        print(f"{dataset} / {model} / {int(rate * 100)}%")
        rates[f"{int(rate * 100)}%"] = evaluate_rate(runner, masks[rate])
    baseline = rates["0%"]["mae"]
    for result in rates.values():
        result["relative_mae_change_percent"] = (
            (result["mae"] - baseline) / baseline * 100.0
        )

    archived = reference_mae(run)
    verification = None
    if archived is not None:
        difference = baseline - archived
        verification = {
            "archived_test_mae": archived,
            "rerun_minus_archived_mae": difference,
            "absolute_tolerance": 1e-5,
            "within_tolerance": abs(difference) <= 1e-5,
        }
        if not verification["within_tolerance"]:
            raise RuntimeError(
                f"Clean-pass mismatch for {dataset}/{model}: "
                f"rerun={baseline:.8f}, archived={archived:.8f}"
            )

    return {
        "checkpoint_seed": checkpoint_seed,
        "config_path": config_path.relative_to(basicts_root).as_posix(),
        "config_sha256": file_hash(config_path),
        "checkpoint_path": checkpoint_path.relative_to(basicts_root).as_posix(),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "clean_pass_verification": verification,
        "rates": rates,
    }


def parse_evaluations(values: list[str] | None):
    if not values:
        return list(DEFAULT_EVALUATIONS)
    evaluations = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected DATASET:MODEL, received {value}")
        dataset, model = value.split(":", 1)
        if dataset not in SENSOR_COUNTS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        evaluations.append((dataset, model))
    return evaluations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic fixed-mask sensor-dropout inference"
    )
    parser.add_argument("--basicts-root", required=True, type=Path)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--evaluation", action="append", help="DATASET:MODEL")
    parser.add_argument("--checkpoint-seed", type=int, default=43)
    parser.add_argument("--mask-seed", type=int, default=42)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    basicts_root = args.basicts_root.resolve()
    if not (basicts_root / "basicts").is_dir():
        raise FileNotFoundError(f"Not a BasicTS root: {basicts_root}")
    if args.extra_site_packages:
        sys.path.append(str(args.extra_site_packages.resolve()))
    sys.path.insert(0, str(basicts_root))

    from easytorch.device import set_device_type
    from easytorch.utils import set_visible_devices

    set_device_type(args.device)
    if args.device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        set_visible_devices(args.gpu)

    evaluations = parse_evaluations(args.evaluation)
    masks = {
        dataset: make_masks(SENSOR_COUNTS[dataset], args.mask_seed)
        for dataset, _ in evaluations
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "generation_script": "pipelines/run_sensor_dropout.py",
            "checkpoint_seed": args.checkpoint_seed,
            "checkpoint_seed_role": (
                "the first fixed seed in the benchmark seed set, used uniformly "
                "as the deterministic checkpoint reference"
            ),
            "checkpoint_seed_comparison": (
                "alternative checkpoint seeds were not evaluated in this stress test"
            ),
            "mask_seed": args.mask_seed,
            "mask_generator": "numpy.random.RandomState(seed).permutation",
            "mask_count_rule": "floor(num_sensors * dropout_rate)",
            "mask_nesting": "10% indices are a subset of 30% indices",
            "application_point": "after BasicTS normalization",
            "masked_values": "all input features set to zero",
            "masked_history": "all 12 input steps",
            "mask_reuse": "one fixed mask per dataset/rate, shared across models",
            "target_rule": "finite targets not equal to zero",
            "dropout_rates": list(DROPOUT_RATES),
        },
        "datasets": {},
    }

    previous_cwd = Path.cwd()
    os.chdir(basicts_root)
    try:
        for dataset, model in evaluations:
            dataset_entry = output["datasets"].setdefault(
                dataset,
                {
                    "num_sensors": SENSOR_COUNTS[dataset],
                    "masks": {
                        f"{int(rate * 100)}%": {
                            "dropped_count": len(indices),
                            "dropped_sensor_indices_zero_based": indices,
                        }
                        for rate, indices in masks[dataset].items()
                    },
                    "models": {},
                },
            )
            dataset_entry["models"][model] = evaluate_model(
                basicts_root,
                dataset,
                model,
                args.checkpoint_seed,
                masks[dataset],
            )
    finally:
        os.chdir(previous_cwd)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
