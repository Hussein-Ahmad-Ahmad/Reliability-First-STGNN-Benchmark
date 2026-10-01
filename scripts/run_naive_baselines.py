"""Persistence and seasonal-naive anchor baselines (C11).

Reviewer concern C11 wants simple non-learned anchors reported alongside the
seven model classes, on the same chronological test windows, so a reader can
see how much of each model's accuracy comes from the forecasting task being
easy (strong autocorrelation / seasonality) rather than from the model.

Two anchors, per dataset:
  - persistence: repeat the last observed value across all output-horizon steps.
  - seasonal-naive: use the value exactly one seasonal period earlier for each
    horizon step (one day earlier for the 5-minute traffic datasets, one year
    (52 weeks) earlier for the weekly Chickenpox series).

Traffic datasets (METR-LA / PEMS-BAY / PEMS04) are read directly from the
raw test_data.npy (already the chronological test partition, in original
units) with the last day of the validation partition prepended so the
seasonal-naive lookback is defined for the first test origins too. This
windowing (non-overlapping origin step of 1 over the whole test array) is
the standard BasicTS convention; it may include a handful more or fewer
origins than a given model's own test loader if that loader trims boundary
origins, so treat small origin-count differences as expected, not an error.

Chickenpox reuses the exact target-disjoint test-partition window indices
from run_chickenpox_all_baselines.py so the comparison is on the identical
99 test origins used by the seven model classes there.

Usage:
    python scripts/run_naive_baselines.py
    python scripts/run_naive_baselines.py --datasets METR-LA PEMS04

Output:
    results/task1_point_forecasting/naive_baselines.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TRAFFIC_DATASETS = ("METR-LA", "PEMS-BAY", "PEMS04")


def _load_chickenpox_module():
    spec = importlib.util.spec_from_file_location(
        "run_chickenpox_all_baselines", ROOT / "scripts" / "run_chickenpox_all_baselines.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def day_block_length(frequency_minutes: int) -> int:
    return int(round((24 * 60) / frequency_minutes))


def mae_rmse(pred: np.ndarray, target: np.ndarray, null_val: float) -> dict:
    if np.isnan(null_val):
        valid = ~np.isnan(target)
    else:
        valid = np.isfinite(target) & ~np.isclose(target, null_val)
    error = (pred - target)[valid]
    return {
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "valid_target_count": int(valid.sum()),
        "test_origins": int(target.shape[0]),
    }


def traffic_baselines(dataset: str) -> dict:
    with (ROOT / "datasets" / dataset / "desc.json").open(encoding="utf-8") as handle:
        desc = json.load(handle)
    settings = desc["regular_settings"]
    input_len = int(settings["INPUT_LEN"])
    output_len = int(settings["OUTPUT_LEN"])
    null_val = float(settings.get("NULL_VAL", 0.0))
    frequency_minutes = int(desc["frequency (minutes)"])
    day_len = day_block_length(frequency_minutes)

    val_data = np.load(ROOT / "datasets" / dataset / "val_data.npy").astype(np.float64)
    test_data = np.load(ROOT / "datasets" / dataset / "test_data.npy").astype(np.float64)
    if val_data.shape[0] < day_len:
        raise ValueError(f"{dataset} validation partition shorter than one seasonal period")

    lookback_buffer = val_data[-day_len:]
    full_series = np.concatenate([lookback_buffer, test_data], axis=0)
    offset = day_len

    n_origins = test_data.shape[0] - input_len - output_len + 1
    if n_origins <= 0:
        raise ValueError(f"{dataset} test partition too short for input_len={input_len}, output_len={output_len}")

    persistence_pred = np.empty((n_origins, output_len, test_data.shape[1]), dtype=np.float64)
    seasonal_pred = np.empty_like(persistence_pred)
    target = np.empty_like(persistence_pred)

    for t in range(n_origins):
        idx = offset + t
        last_observed = full_series[idx + input_len - 1]
        persistence_pred[t] = np.tile(last_observed, (output_len, 1))
        target_start = idx + input_len
        target[t] = full_series[target_start: target_start + output_len]
        seasonal_pred[t] = full_series[target_start - day_len: target_start - day_len + output_len]

    return {
        "dataset": dataset,
        "seasonal_period": "one day",
        "seasonal_period_steps": day_len,
        "input_len": input_len,
        "output_len": output_len,
        "n_test_origins": n_origins,
        "windowing_note": (
            "Non-overlapping-origin sliding window over the full raw test_data.npy "
            "array with a one-day lookback buffer borrowed from val_data.npy for "
            "seasonal-naive; origin count may differ slightly from a given model's "
            "own test loader boundary handling."
        ),
        "persistence": mae_rmse(persistence_pred, target, null_val),
        "seasonal_naive": mae_rmse(seasonal_pred, target, null_val),
    }


def chickenpox_baselines() -> dict:
    cp = _load_chickenpox_module()
    dataset = cp.fetch_dataset()
    source_data = np.array(dataset["FX"], dtype=np.float64)
    steps_per_year = 52

    x, y = cp.build_windows(source_data.astype(np.float32), cp.INPUT_LEN, cp.OUTPUT_LEN)
    _, split_indices, _ = cp.split_data(x, y)
    test_starts = split_indices["test"]

    n_origins = test_starts.shape[0]
    output_len = cp.OUTPUT_LEN
    input_len = cp.INPUT_LEN
    n_counties = source_data.shape[1]

    persistence_pred = np.empty((n_origins, output_len, n_counties), dtype=np.float64)
    seasonal_pred = np.empty_like(persistence_pred)
    target = np.empty_like(persistence_pred)

    for i, start in enumerate(test_starts):
        last_observed = source_data[start + input_len - 1]
        persistence_pred[i] = np.tile(last_observed, (output_len, 1))
        target_start = start + input_len
        target[i] = source_data[target_start: target_start + output_len]
        seasonal_start = target_start - steps_per_year
        if seasonal_start < 0:
            raise ValueError("Seasonal lookback underflows the source series for this test window")
        seasonal_pred[i] = source_data[seasonal_start: seasonal_start + output_len]

    return {
        "dataset": "Hungarian Chickenpox Cases",
        "seasonal_period": "52 weeks (one year)",
        "seasonal_period_steps": steps_per_year,
        "input_len": input_len,
        "output_len": output_len,
        "n_test_origins": n_origins,
        "windowing_note": (
            "Uses the identical target-disjoint test-window start indices as "
            "run_chickenpox_all_baselines.py, in the dataset-provided county-wise "
            "standardized FX signal units (not raw weekly case counts)."
        ),
        "persistence": mae_rmse(persistence_pred, target, float("nan")),
        "seasonal_naive": mae_rmse(seasonal_pred, target, float("nan")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistence and seasonal-naive anchor baselines")
    parser.add_argument("--datasets", nargs="+", default=list(TRAFFIC_DATASETS) + ["Chickenpox"])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "task1_point_forecasting" / "naive_baselines.json")
    args = parser.parse_args()

    results = {}
    for dataset in args.datasets:
        print(f"=== {dataset} ===")
        if dataset == "Chickenpox":
            results[dataset] = chickenpox_baselines()
        elif dataset in TRAFFIC_DATASETS:
            results[dataset] = traffic_baselines(dataset)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        print(json.dumps({k: v for k, v in results[dataset].items() if k in ("persistence", "seasonal_naive")}, indent=2))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
