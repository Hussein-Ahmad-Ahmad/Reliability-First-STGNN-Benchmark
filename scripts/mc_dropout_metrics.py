"""Masked diagnostics for retained MC-Dropout predictive moments."""

from __future__ import annotations

import numpy as np

NULL_ATOL = 5e-5


def summarize_moments(mean, std, target, z_value, pass_hashes):
    mean, std, target = (np.asarray(a) for a in (mean, std, target))
    if not (mean.shape == std.shape == target.shape):
        raise ValueError("Predictive moments and targets must have identical shapes")
    if not np.isfinite(z_value) or z_value <= 0:
        raise ValueError("Interval multiplier must be finite and positive")
    valid = np.isfinite(target) & ~np.isclose(target, 0, atol=NULL_ATOL, rtol=0)
    count = int(valid.sum())
    if count == 0:
        raise ValueError("No valid targets for interval evaluation")
    mu = mean[valid].astype(np.float64)
    sigma = std[valid].astype(np.float64)
    y = target[valid].astype(np.float64)
    if not (np.isfinite(mu).all() and np.isfinite(sigma).all()) or (sigma < 0).any():
        raise ValueError("Valid targets have invalid predictive moments")
    hashes = list(pass_hashes)
    if len(hashes) < 2:
        raise ValueError("At least two pass hashes are required")
    distinct = len(set(hashes))
    variance = float(np.square(sigma).mean())
    return {
        "variance_mean": variance,
        "predictive_std_mean": float(sigma.mean()),
        "picp_uncalibrated": float((np.abs(y - mu) <= z_value * sigma).mean()),
        "mpiw_uncalibrated": float(2 * z_value * sigma.mean()),
        "status": "functional" if distinct > 1 and variance > 0 else "inactive",
        "distinct_pass_predictions": distinct,
        "total_target_count": int(target.size),
        "valid_target_count": count,
        "excluded_target_count": int(target.size - count),
        "target_mask": {"finite": True, "null_value": 0.0, "atol": NULL_ATOL, "rtol": 0.0},
        "aggregation": "flat pooled over valid target elements; float64 moment diagnostics",
    }
