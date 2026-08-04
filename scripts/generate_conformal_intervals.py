"""Generate fixed and per-horizon normalized conformal diagnostics.

The calibration split is the first half of the held-out origins and the
evaluation split is the second half. Scores pool valid origin-sensor-horizon
elements; the output therefore reports marginal diagnostics and does not claim
an exchangeability guarantee for the dependent pooled elements.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


EPSILON = 1e-6


def higher_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int, float]:
    valid = scores[np.isfinite(scores)]
    if valid.size == 0:
        raise ValueError("No valid calibration scores")
    level = min((valid.size + 1) * (1.0 - alpha) / valid.size, 1.0)
    return float(np.quantile(valid, level, method="higher")), int(valid.size), level


def interval_metrics(
    targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: np.ndarray | float,
) -> dict[str, Any]:
    width = thresholds * (std + EPSILON)
    lower = mean - width
    upper = mean + width
    valid = np.isfinite(targets) & targets.astype(bool)
    covered = (targets >= lower) & (targets <= upper)
    return {
        "PICP": float(covered[valid].mean()),
        "MPIW": float((upper - lower)[valid].mean()),
        "valid_target_count": int(valid.sum()),
    }


def per_horizon_metrics(
    targets: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    thresholds: np.ndarray | float,
) -> dict[str, Any]:
    result = {}
    for horizon in range(targets.shape[1]):
        threshold = (
            float(thresholds[horizon])
            if isinstance(thresholds, np.ndarray)
            else thresholds
        )
        result[f"horizon_{horizon + 1}"] = interval_metrics(
            targets[:, horizon, :],
            mean[:, horizon, :],
            std[:, horizon, :],
            threshold,
        )
    return result


def compute_conformal_metrics(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    dataset: str,
    alpha: float = 0.1,
    member_count: int | None = None,
) -> dict[str, dict[str, Any]]:
    if mean.shape != std.shape or mean.shape != targets.shape:
        raise ValueError(
            f"Shape mismatch: mean={mean.shape}, std={std.shape}, "
            f"targets={targets.shape}"
        )
    if mean.ndim != 3:
        raise ValueError("Expected arrays with shape [origins, horizons, sensors]")

    split = mean.shape[0] // 2
    cal_mean, eval_mean = mean[:split], mean[split:]
    cal_std, eval_std = std[:split], std[split:]
    cal_targets, eval_targets = targets[:split], targets[split:]
    cal_valid = np.isfinite(cal_targets) & cal_targets.astype(bool)
    scores = np.full(cal_targets.shape, np.nan, dtype=np.float32)
    scores[cal_valid] = (
        np.abs(cal_targets[cal_valid] - cal_mean[cal_valid])
        / (cal_std[cal_valid] + EPSILON)
    )

    shared = {
        "dataset": dataset,
        "method": "normalized_split_conformal_marginal_diagnostic",
        "alpha": alpha,
        "target_coverage": 1.0 - alpha,
        "member_count": member_count,
        "ensemble_std_ddof": 0,
        "epsilon": EPSILON,
        "chronological_split": {
            "calibration_origins": split,
            "evaluation_origins": mean.shape[0] - split,
        },
        "score_pooling": "valid origin-sensor-horizon elements",
        "null_target_rule": "finite targets not equal to zero",
        "dependence_caveat": (
            "Coverage is an empirical marginal diagnostic. Pooled elements "
            "share origins, sensors, and horizons, so no independent-element "
            "or finite-sample exchangeability guarantee is claimed."
        ),
    }

    fixed_q, fixed_n, fixed_level = higher_quantile(scores, alpha)
    fixed_eval = interval_metrics(
        eval_targets, eval_mean, eval_std, fixed_q
    )
    fixed_eval["per_horizon"] = per_horizon_metrics(
        eval_targets, eval_mean, eval_std, fixed_q
    )
    fixed = {
        **shared,
        "variant": "fixed",
        "calibration": {
            "threshold": fixed_q,
            "valid_score_count": fixed_n,
            "quantile_level": fixed_level,
            "quantile_method": "higher",
            "score_min": float(np.nanmin(scores)),
            "score_max": float(np.nanmax(scores)),
            "score_mean": float(np.nanmean(scores)),
            "score_median": float(np.nanmedian(scores)),
        },
        "evaluation_set": fixed_eval,
    }

    horizon_thresholds = np.empty(mean.shape[1], dtype=np.float64)
    horizon_counts = []
    horizon_levels = []
    for horizon in range(mean.shape[1]):
        q, count, level = higher_quantile(scores[:, horizon, :], alpha)
        horizon_thresholds[horizon] = q
        horizon_counts.append(count)
        horizon_levels.append(level)
    per_horizon_eval = interval_metrics(
        eval_targets,
        eval_mean,
        eval_std,
        horizon_thresholds.reshape(1, -1, 1),
    )
    per_horizon_eval["per_horizon"] = per_horizon_metrics(
        eval_targets, eval_mean, eval_std, horizon_thresholds
    )
    per_horizon = {
        **shared,
        "variant": "per_horizon",
        "calibration": {
            "thresholds": horizon_thresholds.tolist(),
            "valid_score_counts": horizon_counts,
            "quantile_levels": horizon_levels,
            "quantile_method": "higher",
        },
        "evaluation_set": per_horizon_eval,
    }
    return {"fixed": fixed, "per_horizon": per_horizon}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate normalized conformal diagnostics"
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
    metrics = compute_conformal_metrics(
        mean,
        std,
        targets,
        dataset=args.dataset,
        alpha=args.alpha,
        member_count=predictions.shape[0],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant, result in metrics.items():
        path = args.output_dir / f"{args.dataset}_conformal_{variant}_metrics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        print(path)


if __name__ == "__main__":
    main()
