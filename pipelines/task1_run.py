"""
Task 1 Pipeline: Multi-Seed Point Forecasting
==============================================
Trains 7 STGNN models on 3 datasets (METR-LA, PEMS-BAY, PEMS04) with seeds 43/44/45.
Computes multi-seed aggregation and Diebold-Mariano statistical tests.

Usage:
    # Train all configurations (63 runs)
    python pipelines/task1_run.py --mode train --model all --dataset all

    # Train a specific model/dataset/seed
    python pipelines/task1_run.py --mode train --model D2STGNN --dataset METR-LA --seed 43

    # Aggregate results (after training)
    python pipelines/task1_run.py --mode aggregate

    # Run DM tests (after aggregation)
    python pipelines/task1_run.py --mode dm_test
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
BASICTS_ROOT = PROJECT_ROOT / "framework" / "basicts"

MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
DATASETS = ["METR-LA", "PEMS-BAY", "PEMS04"]
SEEDS = [43, 44, 45]


def train_config(model: str, dataset: str, seed: int):
    """Train a single model/dataset/seed configuration."""
    config_path = PROJECT_ROOT / "configs" / model / f"{dataset}_seed{seed}.py"
    if not config_path.exists():
        print(f"  [SKIP] Config not found: {config_path}")
        return False

    cmd = [
        sys.executable, "-m", "basicts.launcher",
        "--cfg", str(config_path),
        "--gpus", "0"
    ]
    print(f"  Training: {model}/{dataset}/seed{seed}")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "framework"))
    return result.returncode == 0


def aggregate_results():
    """Aggregate multi-seed results from checkpoints into summary JSON."""
    script = PROJECT_ROOT / "scripts" / "aggregate_multiseed.py"
    subprocess.run([sys.executable, str(script)])


def run_dm_tests():
    """Run full 21-pair Diebold-Mariano test matrix with Holm-Bonferroni correction."""
    script = PROJECT_ROOT / "scripts" / "run_dm_tests.py"
    subprocess.run([sys.executable, str(script)])


def main():
    parser = argparse.ArgumentParser(description="Task 1: Point Forecasting Pipeline")
    parser.add_argument("--mode", choices=["train", "aggregate", "dm_test"], required=True)
    parser.add_argument("--model", default="all")
    parser.add_argument("--dataset", default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "train":
        models = MODELS if args.model == "all" else [args.model]
        datasets = DATASETS if args.dataset == "all" else [args.dataset]
        seeds = SEEDS if args.seed is None else [args.seed]

        total = len(models) * len(datasets) * len(seeds)
        print(f"Training {total} configurations...")
        success = 0
        for model in models:
            for dataset in datasets:
                for seed in seeds:
                    if train_config(model, dataset, seed):
                        success += 1
        print(f"\nCompleted: {success}/{total}")

    elif args.mode == "aggregate":
        print("Aggregating multi-seed results...")
        aggregate_results()

    elif args.mode == "dm_test":
        print("Running Diebold-Mariano tests (21 pairs, Holm-Bonferroni)...")
        run_dm_tests()


if __name__ == "__main__":
    main()
