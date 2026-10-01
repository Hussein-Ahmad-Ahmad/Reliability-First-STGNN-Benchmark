"""Evaluate sensor zero-ablation across checkpoint seeds and repeated nested masks."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARCHIVED_SEED42_PATH = ROOT / "results" / "robustness" / "sensor_dropout_fixed_masks_seed42.json"
DEFAULT_EVALUATIONS = (
    ("METR-LA", "D2STGNN"),
    ("METR-LA", "MegaCRN"),
    ("METR-LA", "STID"),
    ("PEMS-BAY", "D2STGNN"),
    ("PEMS-BAY", "STID"),
    ("PEMS04", "D2STGNN"),
    ("PEMS04", "STID"),
)
DROPOUT_RATES = (0.0, 0.10, 0.30)
DEFAULT_MASK_SEEDS = (7, 11, 19, 23, 31)


def _load_degree_matched_module():
    spec = importlib.util.spec_from_file_location(
        "run_xai_degree_matched_control", Path(__file__).resolve().parent / "run_xai_degree_matched_control.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


xc = _load_degree_matched_module()


def num_sensors_for(dataset: str) -> int:
    with (ROOT / "datasets" / dataset / "desc.json").open(encoding="utf-8") as handle:
        return int(json.load(handle)["num_nodes"])


def make_masks(num_sensors: int, seed: int) -> dict[float, list[int]]:
    permutation = np.random.RandomState(seed).permutation(num_sensors)
    return {
        rate: sorted(int(index) for index in permutation[: int(num_sensors * rate)])
        for rate in DROPOUT_RATES
    }


def evaluate_one(dataset: str, model: str, mask_seed: int, checkpoint_seed: int, batch_size: int, device: str) -> dict:
    import torch

    num_sensors = num_sensors_for(dataset)
    masks = make_masks(num_sensors, mask_seed)
    rates = {}
    for rate in DROPOUT_RATES:
        result = xc.run_masked_pass(model, dataset, checkpoint_seed, masks[rate], batch_size, torch.device(device))
        rates[f"{int(rate * 100)}%"] = {"mae": result["mae"], "valid_target_count": result["valid_target_count"]}
    baseline = rates["0%"]["mae"]
    for r in rates.values():
        r["relative_mae_change_percent"] = (r["mae"] - baseline) / baseline * 100.0
    return {"mask_seed": mask_seed, "checkpoint_seed": checkpoint_seed, "rates": rates}


def load_archived_seed42(dataset: str, model: str) -> dict | None:
    if not ARCHIVED_SEED42_PATH.is_file():
        return None
    with ARCHIVED_SEED42_PATH.open(encoding="utf-8") as handle:
        archived = json.load(handle)
    try:
        return archived["datasets"][dataset]["models"][model]
    except KeyError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeated-mask-seed robustness sensitivity check")
    parser.add_argument("--mask-seeds", type=int, nargs="+", default=list(DEFAULT_MASK_SEEDS))
    parser.add_argument("--checkpoint-seed", type=int, default=43, choices=(43, 44, 45),
                         help="Which trained checkpoint to run the mask sweep against ("
                              "default 43 matches the archived "
                              "seed-42 mask run; 44/45 extend coverage to the other two checkpoints.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None, help="Defaults to cuda if available, else cpu")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "robustness")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    # The archived seed-42 mask realization was only ever run against the
    # seed-43 checkpoint. Reusing it as a "realization" for a seed-44/45
    # checkpoint sweep would silently mix corruption-draw variability with
    # checkpoint variability - only include it when checkpoint_seed==43.
    reuse_archived = args.checkpoint_seed == 43

    detailed: dict[str, dict[str, dict]] = {}
    summary_rows = []

    for dataset, model in DEFAULT_EVALUATIONS:
        detailed.setdefault(dataset, {})[model] = {"checkpoint_seed": args.checkpoint_seed, "mask_runs": []}
        all_realizations = []
        if reuse_archived:
            archived = load_archived_seed42(dataset, model)
            if archived is not None:
                all_realizations.append({"mask_seed": 42, "checkpoint_seed": 43, "rates": archived["rates"], "source": "archived"})

        for mask_seed in args.mask_seeds:
            print(f"=== {dataset}/{model}: checkpoint_seed={args.checkpoint_seed} mask_seed={mask_seed} ===")
            result = evaluate_one(dataset, model, mask_seed, args.checkpoint_seed, args.batch_size, args.device)
            result["source"] = "newly_computed"
            detailed[dataset][model]["mask_runs"].append(result)
            all_realizations.append(result)
            for rate_key, rate_result in result["rates"].items():
                print(f"  {rate_key}: relative_mae_change_percent={rate_result['relative_mae_change_percent']:.3f}")

        for rate in DROPOUT_RATES:
            rate_key = f"{int(rate * 100)}%"
            values = [r["rates"][rate_key]["relative_mae_change_percent"] for r in all_realizations if rate_key in r["rates"]]
            summary_rows.append({
                "dataset": dataset,
                "model": model,
                "checkpoint_seed": args.checkpoint_seed,
                "dropout_rate": rate_key,
                "n_mask_realizations": len(values),
                "mean_relative_mae_change_percent": float(np.mean(values)),
                "std_relative_mae_change_percent": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.checkpoint_seed == 43 else f"_ckpt{args.checkpoint_seed}"
    detail_path = args.out_dir / f"sensor_dropout_additional_mask_seeds{suffix}.json"
    detail_path.write_text(json.dumps({
        "checkpoint_seed": args.checkpoint_seed,
        "checkpoint_seed_role": (
            "same fixed checkpoint used by the archived seed-42 mask run" if reuse_archived
            else "different checkpoint seed; the archived seed-42 mask run used checkpoint seed 43"
        ),
        "new_mask_seeds": args.mask_seeds,
        "archived_mask_seed_reused": 42 if reuse_archived else None,
        "dropout_rates": list(DROPOUT_RATES),
        "results": detailed,
    }, indent=2) + "\n", encoding="utf-8")

    summary_path = args.out_dir / f"sensor_dropout_mask_seed_sensitivity_summary{suffix}.json"
    summary_path.write_text(json.dumps({
        "checkpoint_seed": args.checkpoint_seed,
        "n_mask_realizations_total": len(args.mask_seeds) + (1 if reuse_archived else 0),
        "mask_seeds_used": ([42] if reuse_archived else []) + args.mask_seeds,
        "note": (
            "mean/std computed across all mask realizations for this checkpoint seed. This "
            "reports whether the single-mask relative-MAE-change numbers in the "
            "main robustness table are representative or an artifact of one draw."
        ),
        "rows": summary_rows,
    }, indent=2) + "\n", encoding="utf-8")

    print(detail_path)
    print(summary_path)


if __name__ == "__main__":
    main()
