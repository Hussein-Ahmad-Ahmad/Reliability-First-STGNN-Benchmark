"""
Task 2 Pipeline: Uncertainty Quantification
============================================
Evaluates three UQ methods on trained STGNN checkpoints:
  1. MC Dropout  — 50-pass stochastic inference
  2. Deep Ensemble — multi-seed prediction aggregation
  3. Conformal Prediction — fixed (global) and per-horizon variants

Usage:
    # Run all UQ methods for all models
    python pipelines/task2_run.py --method all

    # Run MC Dropout only
    python pipelines/task2_run.py --method mc_dropout --model D2STGNN --dataset METR-LA

    # Run Deep Ensemble
    python pipelines/task2_run.py --method ensemble

    # Run Conformal Prediction
    python pipelines/task2_run.py --method conformal --dataset PEMS-BAY
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.uncertainty.mc_dropout import MCDropoutEvaluator
from src.uncertainty.ensemble import EnsembleUQEvaluator
from src.uncertainty.calibration import ConformalPredictor

MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
DATASETS = ["METR-LA", "PEMS-BAY", "PEMS04"]


def run_mc_dropout(model: str, dataset: str):
    """Run 50-pass MC Dropout evaluation."""
    ckpt_dir = PROJECT_ROOT / "checkpoints" / model / f"{dataset}_seed43"
    ckpt_path = ckpt_dir / "best_model.pt"
    output_path = PROJECT_ROOT / "results" / "task2_uncertainty" / "mc_dropout" / f"{model}_{dataset}_mc_dropout_50pass.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = MCDropoutEvaluator(
        checkpoint_path=str(ckpt_path),
        n_passes=50,
        output_path=str(output_path),
    )
    evaluator.evaluate()
    print(f"  ✓ MC Dropout: {model}/{dataset}")


def run_ensemble(dataset: str):
    """Aggregate multi-seed predictions into ensemble UQ."""
    pred_dir = PROJECT_ROOT / "results" / "task2_uncertainty" / "deep_ensemble"
    evaluator = EnsembleUQEvaluator(
        checkpoint_dir=str(PROJECT_ROOT / "checkpoints"),
        dataset=dataset,
        seeds=[43, 44, 45],
        output_dir=str(pred_dir),
    )
    evaluator.evaluate()
    print(f"  ✓ Ensemble: {dataset}")


def run_conformal(dataset: str):
    """Run fixed and per-horizon conformal prediction."""
    for variant in ["fixed", "per_horizon"]:
        output_path = PROJECT_ROOT / "results" / "task2_uncertainty" / "conformal" / f"{dataset}_conformal_{variant}_metrics.json"
        predictor = ConformalPredictor(
            dataset=dataset,
            variant=variant,
            output_path=str(output_path),
        )
        predictor.calibrate_and_evaluate()
        print(f"  ✓ Conformal ({variant}): {dataset}")


def main():
    parser = argparse.ArgumentParser(description="Task 2: Uncertainty Quantification Pipeline")
    parser.add_argument("--method", choices=["mc_dropout", "ensemble", "conformal", "all"], default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.method in ("mc_dropout", "all"):
        print("=== MC Dropout (50-pass) ===")
        for model in models:
            for dataset in datasets:
                try:
                    run_mc_dropout(model, dataset)
                except Exception as e:
                    print(f"  ✗ {model}/{dataset}: {e}")

    if args.method in ("ensemble", "all"):
        print("=== Deep Ensemble ===")
        for dataset in datasets:
            try:
                run_ensemble(dataset)
            except Exception as e:
                print(f"  ✗ {dataset}: {e}")

    if args.method in ("conformal", "all"):
        print("=== Conformal Prediction ===")
        for dataset in datasets:
            try:
                run_conformal(dataset)
            except Exception as e:
                print(f"  ✗ {dataset}: {e}")


if __name__ == "__main__":
    main()
