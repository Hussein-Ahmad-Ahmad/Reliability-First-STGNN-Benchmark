"""Regenerate three-seed point-forecast statistics from checkpoint metrics.

The reported standard deviations use the sample convention (ddof=1). The
three-seed 95% t intervals use the same sample standard deviation and two
degrees of freedom.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results" / "task1_point_forecasting"
AGGREGATE_PATH = RESULTS / "multiseed_aggregation_clean.json"
SUMMARY_PATH = RESULTS / "statistical_reporting_summary.csv"

MODELS = (
    "D2STGNN",
    "MegaCRN",
    "MTGNN",
    "STGCNChebGraphConv",
    "STID",
    "STNorm",
    "STAEformer",
)
DATASETS = ("METR-LA", "PEMS-BAY", "PEMS04")
SEEDS = (43, 44, 45)
T_CRITICAL_DF2 = 4.302653


def mean_and_sample_sd(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        raise ValueError("At least two seed values are required for a sample SD.")
    return float(np.mean(values)), float(np.std(values, ddof=1))


def read_seed_metrics(model: str, dataset: str, seed: int) -> dict:
    public = RESULTS / "seed_metrics.json"
    if public.exists():
        return json.loads(public.read_text(encoding="utf-8"))[dataset][model][str(seed)]
    path = CHECKPOINTS / model / f"{dataset}_seed{seed}" / "test_metrics.json"
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    aggregate: dict[str, dict[str, dict[str, float | int]]] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for dataset in DATASETS:
        aggregate[dataset] = {}
        dataset_mae_means: dict[str, float] = {}
        pending_rows: list[tuple[str, dict[str, float | int]]] = []

        for model in MODELS:
            runs = [read_seed_metrics(model, dataset, seed) for seed in SEEDS]
            record: dict[str, float | int] = {
                "n_seeds": len(SEEDS),
                "standard_deviation_ddof": 1,
            }
            for metric in ("MAE", "RMSE", "MAPE"):
                mean, sd = mean_and_sample_sd([run["overall"][metric] for run in runs])
                record[f"{metric}_mean"] = round(mean, 6)
                record[f"{metric}_std"] = round(sd, 6)

            for label, key in (("H3", "horizon_3"), ("H6", "horizon_6"), ("H12", "horizon_12")):
                mean, _ = mean_and_sample_sd([run[key]["MAE"] for run in runs])
                record[f"{label}_MAE_mean"] = round(mean, 6)

            aggregate[dataset][model] = record
            dataset_mae_means[model] = float(record["MAE_mean"])
            pending_rows.append((model, record))

        best_mae = min(dataset_mae_means.values())
        for model, record in pending_rows:
            mae_sd = float(record["MAE_std"])
            summary_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "n_seeds": len(SEEDS),
                    "seeds": ",".join(map(str, SEEDS)),
                    "mae_mean": record["MAE_mean"],
                    "mae_sample_sd": record["MAE_std"],
                    "mae_95ci_half_width": round(T_CRITICAL_DF2 * mae_sd / np.sqrt(len(SEEDS)), 6),
                    "rmse_mean": record["RMSE_mean"],
                    "mape_mean": record["MAPE_mean"],
                    "effect_mae_pct_vs_dataset_best": round(
                        100.0 * (float(record["MAE_mean"]) - best_mae) / best_mae, 6
                    ),
                    "standard_deviation_ddof": 1,
                    "ci_scope": "across three seed-level results",
                }
            )

    AGGREGATE_PATH.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {AGGREGATE_PATH}")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
