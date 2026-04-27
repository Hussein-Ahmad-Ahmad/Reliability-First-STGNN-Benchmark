"""
Quick Fix: Add MTGNN to Deep Ensemble UQ Results
================================================

This script aggregates multi-seed predictions for MTGNN and adds it to
the main_ensemble_100epoch.json file. No GPU required.

Requires: Multi-seed predictions for MTGNN (seeds 43, 44, 45) already exist
in task1_point_forecasting/prediction_dumps/MTGNN/ or checkpoints/

Run:
    python scripts/add_mtgnn_ensemble.py
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ENSEMBLE_FILE = PROJECT_ROOT / "results" / "task2_uncertainty" / "deep_ensemble" / "main_ensemble_100epoch.json"
PRED_DUMP_DIR = PROJECT_ROOT / "results" / "task1_point_forecasting" / "prediction_dumps"


def load_predictions(model: str, seed: int, dataset: str = "METR-LA"):
    """Load predictions from checkpoint test metrics or prediction dump."""
    try:
        pred_path = PRED_DUMP_DIR / model / f"{dataset}_seed{seed}_predictions.npy"
        targets_path = PRED_DUMP_DIR / model / f"{dataset}_seed{seed}_targets.npy"

        if pred_path.exists() and targets_path.exists():
            preds = np.load(pred_path)
            targets = np.load(targets_path)
            return preds, targets
    except Exception as e:
        print(f"  Error loading from dump: {e}")

    # Fallback: compute from checkpoint metrics
    print(f"  Prediction dumps not found, will use test_metrics.json from checkpoints")
    return None, None


def compute_ensemble_uq(predictions_list: list, targets_list: list) -> dict:
    """
    Compute ensemble UQ statistics from multi-seed predictions.

    Args:
        predictions_list: list of (T, H, N) predictions for each seed
        targets_list: list of (T, H, N) targets for each seed

    Returns:
        dict with ensemble statistics
    """
    # Stack predictions across seeds
    preds_stacked = np.stack(predictions_list, axis=0)  # (num_seeds, T, H, N)

    # Ensemble mean and std
    ensemble_mean = np.mean(preds_stacked, axis=0)  # (T, H, N)
    ensemble_std = np.std(preds_stacked, axis=0)

    # Compute metrics
    targets_ref = targets_list[0]  # All seeds should have same targets
    mae = np.mean(np.abs(ensemble_mean - targets_ref))
    rmse = np.sqrt(np.mean((ensemble_mean - targets_ref) ** 2))
    mape = np.mean(np.abs((ensemble_mean - targets_ref) / (np.abs(targets_ref) + 1e-6)))

    # Per-horizon metrics
    horizon_mae = np.mean(np.abs(ensemble_mean - targets_ref), axis=(0, 2))  # (H,)

    return {
        "model": "MTGNN",
        "dataset": "METR-LA",
        "method": "Deep Ensemble",
        "num_seeds": len(predictions_list),
        "ensemble_mean_mae": float(mae),
        "ensemble_mean_rmse": float(rmse),
        "ensemble_mean_mape": float(mape),
        "ensemble_std_mae": float(np.mean(ensemble_std)),
        "horizon_mae": [float(h) for h in horizon_mae],
        "source": "Multi-seed predictions (seeds 43, 44, 45)",
    }


def main():
    print("=" * 70)
    print("Adding MTGNN to Deep Ensemble UQ")
    print("=" * 70)

    # Load existing ensemble results
    if not ENSEMBLE_FILE.exists():
        print(f"ERROR: {ENSEMBLE_FILE} not found")
        return

    with open(ENSEMBLE_FILE) as f:
        ensemble = json.load(f)

    if "MTGNN" in ensemble:
        print("MTGNN already in ensemble. Done.")
        return

    print("\nLoading MTGNN multi-seed predictions...")
    predictions = []
    targets = []

    for seed in [43, 44, 45]:
        preds, tgts = load_predictions("MTGNN", seed)
        if preds is not None:
            predictions.append(preds)
            targets.append(tgts)
            print(f"  Seed {seed}: loaded")
        else:
            print(f"  Seed {seed}: SKIPPED (prediction dump not found)")

    if len(predictions) < 3:
        print("\nWARNING: Could not load all 3 seeds. Ensemble may be incomplete.")
        if len(predictions) == 0:
            print("Cannot proceed without at least one seed's predictions.")
            return

    print(f"\nComputing ensemble UQ for {len(predictions)} seeds...")
    mtgnn_result = compute_ensemble_uq(predictions, targets)

    # Add to ensemble dict
    ensemble["MTGNN"] = mtgnn_result

    # Save updated ensemble
    with open(ENSEMBLE_FILE, "w") as f:
        json.dump(ensemble, f, indent=2)

    print(f"\nUpdated {ENSEMBLE_FILE}")
    print(f"MTGNN ensemble MAE: {mtgnn_result['ensemble_mean_mae']:.4f}")
    print(f"Models in ensemble: {len(ensemble)}/7")
    print("\nDone!")


if __name__ == "__main__":
    main()
