"""Regenerate cross-dataset conformal summaries from BasicTS checkpoints."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipelines.run_sensor_dropout import file_hash, find_run, load_config
from scripts.generate_conformal_intervals import compute_conformal_metrics


DEFAULT_MEMBERS = (
    ("D2STGNN", 43),
    ("D2STGNN", 44),
    ("D2STGNN", 45),
    ("MTGNN", 43),
    ("MTGNN", 44),
    ("MTGNN", 45),
    ("STID", 43),
    ("STID", 44),
    ("STID", 45),
)


def archived_mae(run: Path) -> float | None:
    path = run / "test_metrics.json"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return float(json.load(handle)["overall"]["MAE"])


def checkpoint_training_metadata(path: Path) -> tuple[int | None, float | None]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    epoch = payload.get("epoch") if isinstance(payload, dict) else None
    metrics = payload.get("best_metrics", {}) if isinstance(payload, dict) else {}
    best_val_mae = metrics.get("val/MAE") if isinstance(metrics, dict) else None
    return (
        None if epoch is None else int(epoch),
        None if best_val_mae is None else float(best_val_mae),
    )


def infer_member(
    basicts_root: Path,
    dataset: str,
    model: str,
    seed: int,
):
    from easytorch.config import init_cfg

    run, config_path, checkpoint_path = find_run(
        basicts_root, dataset, model, seed
    )
    cfg = init_cfg(
        load_config(
            config_path,
            f"conformal_{dataset}_{model}_{seed}",
            fallback_paths=[basicts_root / "baselines" / model],
        )
    )
    cfg.MD5 = run.name
    runner = cfg.RUNNER(cfg)
    if runner.need_setup_graph:
        runner.setup_graph(cfg=cfg, train=False)
    runner.init_test(cfg)
    runner.load_model(ckpt_path=str(checkpoint_path), strict=True)
    runner.model.eval()

    predictions = []
    targets = []
    with torch.no_grad():
        for data in runner.test_data_loader:
            result = runner.forward(data, epoch=None, iter_num=None, train=False)
            predictions.append(
                result["prediction"].detach().cpu().numpy().squeeze(-1)
            )
            targets.append(result["target"].detach().cpu().numpy().squeeze(-1))
    prediction = np.concatenate(predictions, axis=0).astype(np.float32, copy=False)
    target = np.concatenate(targets, axis=0).astype(np.float32, copy=False)
    valid = np.isfinite(target) & target.astype(bool)
    clean_mae = float(np.abs(prediction - target)[valid].mean())
    archived = archived_mae(run)
    if archived is not None and abs(clean_mae - archived) > 1e-5:
        raise RuntimeError(
            f"Clean-pass mismatch for {dataset}/{model}/seed{seed}: "
            f"{clean_mae:.8f} versus {archived:.8f}"
        )

    checkpoint_epoch, checkpoint_best_val_mae = checkpoint_training_metadata(
        checkpoint_path
    )
    metadata = {
        "model": model,
        "seed": seed,
        "config_path": config_path.relative_to(basicts_root).as_posix(),
        "config_sha256": file_hash(config_path),
        "checkpoint_path": checkpoint_path.relative_to(basicts_root).as_posix(),
        "checkpoint_sha256": file_hash(checkpoint_path),
        "checkpoint_selector": "minimum validation MAE during training",
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_best_validation_mae": checkpoint_best_val_mae,
        "checkpoint_bytes_in_public_repository": False,
        "rerun_clean_mae": clean_mae,
        "archived_test_mae": archived,
        "rerun_minus_archived_mae": (
            None if archived is None else clean_mae - archived
        ),
    }
    del runner
    torch.cuda.empty_cache()
    gc.collect()
    return prediction, target, metadata


def parse_members(values: list[str] | None):
    if not values:
        return list(DEFAULT_MEMBERS)
    members = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"Expected MODEL:SEED, received {value}")
        model, seed = value.split(":", 1)
        members.append((model, int(seed)))
    return members


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cross-dataset normalized conformal summaries"
    )
    parser.add_argument("--basicts-root", required=True, type=Path)
    parser.add_argument("--extra-site-packages", type=Path)
    parser.add_argument("--dataset", required=True, choices=("PEMS-BAY", "PEMS04"))
    parser.add_argument("--member", action="append", help="MODEL:SEED")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument("--gpu", default="0")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "task2_uncertainty" / "conformal",
    )
    args = parser.parse_args()

    basicts_root = args.basicts_root.resolve()
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

    members = parse_members(args.member)
    running_mean = None
    running_m2 = None
    reference_target = None
    member_metadata = []
    previous_cwd = Path.cwd()
    os.chdir(basicts_root)
    try:
        for count, (model, seed) in enumerate(members, start=1):
            print(f"{args.dataset} / {model} / seed {seed}")
            prediction, target, metadata = infer_member(
                basicts_root, args.dataset, model, seed
            )
            if reference_target is None:
                reference_target = target
                running_mean = np.zeros_like(prediction, dtype=np.float32)
                running_m2 = np.zeros_like(prediction, dtype=np.float32)
            elif not np.allclose(target, reference_target, rtol=1e-6, atol=1e-6):
                raise RuntimeError(f"Target mismatch for {model}/seed{seed}")
            delta = prediction - running_mean
            running_mean += delta / count
            running_m2 += delta * (prediction - running_mean)
            member_metadata.append(metadata)
            del prediction, target, delta
            gc.collect()
    finally:
        os.chdir(previous_cwd)

    if reference_target is None or running_mean is None or running_m2 is None:
        raise RuntimeError("No members were evaluated")
    ensemble_std = np.sqrt(running_m2 / len(members)).astype(
        np.float32, copy=False
    )
    metrics = compute_conformal_metrics(
        running_mean,
        ensemble_std,
        reference_target,
        dataset=args.dataset,
        alpha=args.alpha,
        member_count=len(members),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = f"{args.dataset}_ensemble_manifest.json"
    manifest = {
        "schema_version": 2,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "member_count": len(members),
        "selection_rule": (
            "The member list was fixed from the requested MODEL:SEED entries "
            "before conformal calibration; calibration and evaluation coverage "
            "were not used to select members or checkpoints."
        ),
        "coverage_used_for_member_selection": False,
        "member_order": member_metadata,
        "aggregation": {
            "mean": "online arithmetic mean in float32",
            "standard_deviation": "population standard deviation (ddof=0)",
            "target_match_tolerance": {"rtol": 1e-6, "atol": 1e-6},
            "clean_mae_verification_tolerance": 1e-5,
        },
        "generation_script": "scripts/regenerate_cross_dataset_conformal.py",
        "artifact_availability": {
            "public_repository": "generation code, member metadata, and compact metrics",
            "large_prediction_arrays": "not retained; regenerated in memory from checkpoints",
            "checkpoint_bytes": "retained internal archive; not distributed in the GitHub release",
            "public_download_url": None,
            "raw_rerun_requirement": "access to the listed BasicTS checkpoints and public datasets",
        },
        "statistics_arrays_archived": False,
        "statistics_note": (
            "Ensemble mean and standard deviation are regenerated in memory "
            "from the listed checkpoints; only compact metrics are archived."
        ),
    }
    with (args.output_dir / manifest_name).open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")

    for variant, result in metrics.items():
        result["ensemble_manifest_path"] = (
            f"results/task2_uncertainty/conformal/{manifest_name}"
        )
        path = args.output_dir / f"{args.dataset}_conformal_{variant}_metrics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(path)


if __name__ == "__main__":
    main()
