"""Compare normalized split conformal against a plain (sigma = 1) control.

Reviewer concern C3 asks for an unnormalized conformal control: the same
chronological calibration/evaluation split as the normalized pipeline in
generate_conformal_intervals.py, but with the calibration score defined as
the raw absolute residual (no division by predictive std) and a constant
half-width applied uniformly to every sensor/horizon. Both PICP and MPIW are
reported for both variants so the manuscript can state whether normalization
changes coverage, width, or both.

This does not retrain or re-run inference. It re-reads the same ensemble
prediction and target arrays already used by generate_conformal_intervals.py.

Usage (repeat per dataset with its existing ensemble prediction arrays):

    python scripts/run_conformal_sigma_control.py ^
        --ensemble-predictions results/task2_uncertainty/<dataset>_ensemble_predictions.npy ^
        --targets results/task2_uncertainty/<dataset>_targets.npy ^
        --dataset METR-LA ^
        --output-dir results/task2_uncertainty/conformal_sigma_control

Output:
    <output-dir>/<dataset>_conformal_sigma_control.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_DIR = Path(__file__).resolve().parent
EPSILON = 1e-6


def _load_generate_conformal_intervals():
    spec = importlib.util.spec_from_file_location(
        "generate_conformal_intervals", SCRIPTS_DIR / "generate_conformal_intervals.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


gci = _load_generate_conformal_intervals()


def plain_interval_metrics(
    targets: np.ndarray, mean: np.ndarray, half_width: float
) -> dict:
    lower = mean - half_width
    upper = mean + half_width
    valid = np.isfinite(targets) & targets.astype(bool)
    covered = (targets >= lower) & (targets <= upper)
    return {
        "PICP": float(covered[valid].mean()),
        "MPIW": float(2.0 * half_width),
        "valid_target_count": int(valid.sum()),
    }


def compute_plain_control(
    mean: np.ndarray, targets: np.ndarray, dataset: str, alpha: float
) -> dict:
    split = mean.shape[0] // 2
    cal_mean, eval_mean = mean[:split], mean[split:]
    cal_targets, eval_targets = targets[:split], targets[split:]
    cal_valid = np.isfinite(cal_targets) & cal_targets.astype(bool)
    scores = np.full(cal_targets.shape, np.nan, dtype=np.float32)
    scores[cal_valid] = np.abs(cal_targets[cal_valid] - cal_mean[cal_valid])

    half_width, n_scores, level = gci.higher_quantile(scores, alpha)
    eval_metrics = plain_interval_metrics(eval_targets, eval_mean, half_width)
    per_horizon = {}
    for horizon in range(targets.shape[1]):
        per_horizon[f"horizon_{horizon + 1}"] = plain_interval_metrics(
            eval_targets[:, horizon, :], eval_mean[:, horizon, :], half_width
        )
    eval_metrics["per_horizon"] = per_horizon

    return {
        "dataset": dataset,
        "method": "plain_split_conformal_sigma1_control",
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "note": (
            "Calibration score is the raw absolute residual (no division by "
            "predictive std). The resulting interval half-width is constant "
            "across sensors and horizons, unlike the normalized pipeline."
        ),
        "chronological_split": {
            "calibration_origins": split,
            "evaluation_origins": mean.shape[0] - split,
        },
        "calibration": {
            "half_width": half_width,
            "valid_score_count": n_scores,
            "quantile_level": level,
            "quantile_method": "higher",
        },
        "evaluation_set": eval_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plain (sigma=1) vs normalized split-conformal control"
    )
    parser.add_argument("--ensemble-predictions", required=True, type=Path)
    parser.add_argument("--targets", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    predictions = np.load(args.ensemble_predictions, mmap_mode="r")
    targets = np.load(args.targets, mmap_mode="r")
    if predictions.ndim != 4:
        raise ValueError("Expected ensemble predictions [members, origins, horizons, sensors]")

    mean = np.asarray(predictions.mean(axis=0), dtype=np.float32)
    std = np.asarray(predictions.std(axis=0, ddof=0), dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    normalized = gci.compute_conformal_metrics(
        mean, std, targets, dataset=args.dataset, alpha=args.alpha,
        member_count=predictions.shape[0],
    )["fixed"]
    plain = compute_plain_control(mean, targets, args.dataset, args.alpha)

    comparison = {
        "dataset": args.dataset,
        "alpha": args.alpha,
        "normalized_variant": {
            "PICP": normalized["evaluation_set"]["PICP"],
            "MPIW": normalized["evaluation_set"]["MPIW"],
        },
        "plain_sigma1_variant": {
            "PICP": plain["evaluation_set"]["PICP"],
            "MPIW": plain["evaluation_set"]["MPIW"],
        },
        "picp_difference_normalized_minus_plain": (
            normalized["evaluation_set"]["PICP"] - plain["evaluation_set"]["PICP"]
        ),
        "mpiw_difference_normalized_minus_plain": (
            normalized["evaluation_set"]["MPIW"] - plain["evaluation_set"]["MPIW"]
        ),
    }

    output = {
        "dataset": args.dataset,
        "normalized_split_conformal": normalized,
        "plain_sigma1_control": plain,
        "comparison": comparison,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.dataset}_conformal_sigma_control.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
