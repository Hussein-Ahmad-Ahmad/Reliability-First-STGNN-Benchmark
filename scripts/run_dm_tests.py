"""
Diebold-Mariano Test — Full 21-Pair Matrix with Holm-Bonferroni
=================================================================
Runs pairwise DM tests for all 7 models (21 unique pairs) using
Newey-West HAC standard errors and applies Holm-Bonferroni correction
for multiple comparisons.

NOTE: Pre-computed results are already in:
  results/task1_point_forecasting/dm_full_21pairs_holm_corrected.json

This script regenerates them from prediction dumps if needed.

Output:
  results/task1_point_forecasting/dm_full_21pairs_holm_corrected.json

Run:
    python scripts/run_dm_tests.py
"""

import json
import numpy as np
from itertools import combinations
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
PRED_DIR = PROJECT_ROOT / "results" / "task1_point_forecasting" / "prediction_dumps"
OUTPUT_PATH = PROJECT_ROOT / "results" / "task1_point_forecasting" / "dm_full_21pairs_holm_corrected.json"

MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]


def dm_test_newey_west(errors1: np.ndarray, errors2: np.ndarray, h: int = 12) -> dict:
    """
    Diebold-Mariano test with Newey-West HAC standard errors.

    Args:
        errors1: absolute errors from model 1, shape (T,)
        errors2: absolute errors from model 2, shape (T,)
        h: forecast horizon (for HAC bandwidth selection)

    Returns:
        dict with dm_stat, p_value, significant (alpha=0.05)
    """
    d = errors1 - errors2  # loss differential
    T = len(d)
    d_mean = np.mean(d)

    # Newey-West HAC variance estimator
    lags = int(np.floor(4 * (T / 100) ** (2 / 9)))  # automatic bandwidth
    gamma_0 = np.mean((d - d_mean) ** 2)
    hac_var = gamma_0

    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma_lag = np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean))
        hac_var += 2 * weight * gamma_lag

    hac_se = np.sqrt(max(hac_var, 1e-10) / T)
    dm_stat = d_mean / hac_se

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

    return {
        "dm_stat": round(float(dm_stat), 4),
        "p_value": round(float(p_value), 6),
        "mean_loss_diff": round(float(d_mean), 6),
        "hac_se": round(float(hac_se), 6),
        "n_obs": int(T),
    }


def holm_bonferroni(p_values: list) -> list:
    """Apply Holm-Bonferroni correction. Returns adjusted p-values."""
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adjusted[orig_idx] = min(adj, 1.0)

    # Ensure monotonicity
    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    return adjusted


def main():
    # Check if prediction dumps exist
    if not PRED_DIR.exists():
        print("  Prediction dumps not found. Using pre-computed results.")
        pre_computed = PROJECT_ROOT / "results" / "task1_point_forecasting" / "dm_full_21pairs_holm_corrected.json"
        if pre_computed.exists():
            print(f"  Pre-computed DM results available at: {pre_computed}")
        else:
            print("  No DM results found. Run training first, then generate prediction dumps.")
        return

    results = {}
    pairs = list(combinations(MODELS, 2))
    raw_pvalues = []
    pair_order = []

    print(f"Computing DM tests for {len(pairs)} pairs...")

    for m1, m2 in pairs:
        pair_key = f"{m1}_vs_{m2}"

        # Load predictions (seed43 as reference)
        try:
            pred1 = np.load(PRED_DIR / m1 / "METR-LA_seed43_predictions.npy")
            pred2 = np.load(PRED_DIR / m2 / "METR-LA_seed43_predictions.npy")
            targets = np.load(PRED_DIR / m1 / "METR-LA_seed43_targets.npy")

            errors1 = np.abs(pred1 - targets).mean(axis=(-1, -2)).flatten()
            errors2 = np.abs(pred2 - targets).mean(axis=(-1, -2)).flatten()

            dm_result = dm_test_newey_west(errors1, errors2)
            results[pair_key] = dm_result
            raw_pvalues.append(dm_result["p_value"])
            pair_order.append(pair_key)

            print(f"  ✓ {pair_key}: DM={dm_result['dm_stat']}, p={dm_result['p_value']:.4f}")
        except FileNotFoundError as e:
            print(f"  ✗ {pair_key}: {e}")
            results[pair_key] = {"error": str(e)}

    # Apply Holm-Bonferroni correction
    if raw_pvalues:
        adjusted = holm_bonferroni(raw_pvalues)
        for i, pair_key in enumerate(pair_order):
            if "p_value" in results[pair_key]:
                results[pair_key]["p_value_holm_corrected"] = round(adjusted[i], 6)
                results[pair_key]["significant_after_correction"] = adjusted[i] < 0.05

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")
    sig = sum(1 for k in pair_order if results.get(k, {}).get("significant_after_correction", False))
    print(f"Significant pairs (after Holm-Bonferroni): {sig}/{len(pair_order)}")


if __name__ == "__main__":
    main()
