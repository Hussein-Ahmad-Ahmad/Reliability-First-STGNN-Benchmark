"""
Aggregate Multi-Seed Training Results
======================================
Reads test_metrics.json from each checkpoint directory and computes
mean ± std across seeds 43/44/45 for each model/dataset combination.

Output:
  results/task1_point_forecasting/multiseed_aggregation.json

Run:
    python scripts/aggregate_multiseed.py
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CKPT_ROOT = PROJECT_ROOT / "checkpoints"
OUTPUT_PATH = PROJECT_ROOT / "results" / "task1_point_forecasting" / "multiseed_aggregation.json"

MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
DATASETS = ["METR-LA", "PEMS-BAY", "PEMS04"]
SEEDS = [43, 44, 45]
METRICS = ["MAE", "RMSE", "MAPE"]


def main():
    results = {}

    for model in MODELS:
        results[model] = {}
        for dataset in DATASETS:
            seed_metrics = {m: [] for m in METRICS}
            found_seeds = []

            for seed in SEEDS:
                metrics_path = CKPT_ROOT / model / f"{dataset}_seed{seed}" / "test_metrics.json"
                if not metrics_path.exists():
                    continue

                with open(metrics_path) as f:
                    data = json.load(f)

                for metric in METRICS:
                    val = data.get(metric) or data.get(metric.lower())
                    if val is not None:
                        seed_metrics[metric].append(float(val))

                found_seeds.append(seed)

            if not found_seeds:
                print(f"  MISSING: {model}/{dataset} — no metrics found")
                continue

            agg = {"seeds_found": found_seeds}
            for metric in METRICS:
                vals = seed_metrics[metric]
                if vals:
                    agg[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
                    agg[f"{metric}_std"] = round(float(np.std(vals)), 4)
                    agg[f"{metric}_values"] = [round(v, 4) for v in vals]

            results[model][dataset] = agg
            mae = agg.get("MAE_mean", "N/A")
            print(f"  ✓ {model}/{dataset}: MAE={mae} (seeds={found_seeds})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
