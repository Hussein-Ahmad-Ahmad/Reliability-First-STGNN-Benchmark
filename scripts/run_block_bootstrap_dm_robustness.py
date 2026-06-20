"""
Dependence-aware forecast-origin bootstrap for pairwise point-forecast results.

This check resamples contiguous blocks of forecast origins, preserving short-range
temporal dependence that an iid resample would break. It is intended as a compact
robustness companion to the pairwise DM/Holm statistical reporting.

Default run:
    python scripts/run_block_bootstrap_dm_robustness.py

Outputs:
    results/task1_point_forecasting/block_bootstrap_dm_robustness_pems04_seeds43-44-45.json
    results/task1_point_forecasting/block_bootstrap_dm_robustness_pems04_seeds43-44-45.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DUMP_ROOT = PROJECT_ROOT / "results" / "task1_point_forecasting" / "fresh_inference_dumps"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "task1_point_forecasting"

MODEL_ORDER = [
    "D2STGNN",
    "STAEformer",
    "MegaCRN",
    "MTGNN",
    "STID",
    "STNorm",
    "STGCNChebGraphConv",
]


def load_dataset_desc(dataset: str) -> dict:
    path = PROJECT_ROOT / "datasets" / dataset / "desc.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_prediction_array(path: Path, num_nodes: int, horizon: int) -> np.ndarray:
    try:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.dtype == object and loaded.shape == ():
            loaded = loaded.item()
        arr = np.asarray(loaded)
        if arr.dtype == object:
            raise ValueError("object array cannot be coerced safely")
        return arr.astype(np.float32, copy=False)
    except Exception:
        return read_raw_float32(path, num_nodes=num_nodes, horizon=horizon)


def read_raw_float32(path: Path, num_nodes: int, horizon: int) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float32)
    values_per_origin = num_nodes * horizon
    if raw.size % values_per_origin != 0:
        raise ValueError(
            f"{path} has {raw.size} float32 values, not divisible by "
            f"{horizon} * {num_nodes} = {values_per_origin}"
        )
    num_origins = raw.size // values_per_origin
    return raw.reshape(num_origins, horizon, num_nodes)


def reshape_prediction(arr: np.ndarray, num_nodes: int, horizon: int, path: Path) -> np.ndarray:
    if arr.ndim == 3 and arr.shape[1] == horizon and arr.shape[2] == num_nodes:
        return arr
    if arr.ndim == 4 and arr.shape[1] == horizon and arr.shape[2] == num_nodes:
        if arr.shape[3] == 1:
            return arr[..., 0]
        return arr[..., 0]

    values_per_origin = horizon * num_nodes
    if arr.size % values_per_origin != 0:
        raise ValueError(
            f"{path} with shape {arr.shape} has {arr.size} values, not divisible by "
            f"{horizon} * {num_nodes} = {values_per_origin}"
        )
    return arr.reshape(-1, horizon, num_nodes)


def day_block_length(frequency_minutes: int) -> int:
    return int(round((24 * 60) / frequency_minutes))


def moving_block_bootstrap_means(
    diff: np.ndarray,
    block_len: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if diff.ndim != 1:
        raise ValueError("diff must be a one-dimensional loss-differential series")
    if block_len <= 0:
        raise ValueError("block_len must be positive")
    if block_len > diff.size:
        raise ValueError(f"block_len={block_len} exceeds series length={diff.size}")

    max_start = diff.size - block_len
    blocks_needed = math.ceil(diff.size / block_len)
    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        starts = rng.integers(0, max_start + 1, size=blocks_needed)
        sampled = np.concatenate([diff[s : s + block_len] for s in starts])[: diff.size]
        boot_means[b] = float(sampled.mean())

    return boot_means


def format_model_name(model: str) -> str:
    return "STGCN-Cheb" if model == "STGCNChebGraphConv" else model


def find_original_test_dir(root: Path, model: str, dataset: str, seed: int) -> Path | None:
    model_root = root / model
    if not model_root.exists():
        return None

    candidates = []
    for candidate in model_root.iterdir():
        if not candidate.is_dir():
            continue
        name = candidate.name
        if name.startswith(dataset + "_") and f"_seed{seed}" in name:
            candidates.append(candidate)

    for candidate in sorted(candidates):
        for run_dir in sorted(x for x in candidate.iterdir() if x.is_dir()):
            test_dir = run_dir / "test_results"
            if (test_dir / "predictions.npy").exists() and (test_dir / "targets.npy").exists():
                return test_dir
    return None


def find_dump_dir(args: argparse.Namespace, model: str, dataset: str, seed: int) -> Path | None:
    fresh_dir = args.dump_root / model / dataset / f"seed{seed}"
    if (fresh_dir / "predictions.npy").exists() and (fresh_dir / "targets.npy").exists():
        return fresh_dir
    if args.original_checkpoint_root is not None:
        return find_original_test_dir(args.original_checkpoint_root, model, dataset, seed)
    return None


def masked_per_origin_mae(pred: np.ndarray, target: np.ndarray, null_val: float) -> np.ndarray:
    abs_error = np.abs(pred - target).astype(np.float64)
    if np.isnan(null_val):
        valid = ~np.isnan(target)
    else:
        valid = ~np.isclose(target, null_val)

    counts = valid.sum(axis=(1, 2))
    sums = (abs_error * valid).sum(axis=(1, 2))
    errors = np.full(target.shape[0], np.nan, dtype=np.float64)
    valid_origins = counts > 0
    errors[valid_origins] = sums[valid_origins] / counts[valid_origins]
    return errors


def load_errors_for_seed(
    dump_dir: Path,
    num_nodes: int,
    horizon: int,
    null_val: float,
) -> tuple[np.ndarray, np.ndarray]:
    pred_path = dump_dir / "predictions.npy"
    target_path = dump_dir / "targets.npy"
    pred = reshape_prediction(read_prediction_array(pred_path, num_nodes, horizon), num_nodes, horizon, pred_path)
    target = reshape_prediction(read_prediction_array(target_path, num_nodes, horizon), num_nodes, horizon, target_path)

    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch in {dump_dir}: {pred.shape} vs {target.shape}")

    errors = masked_per_origin_mae(pred, target, null_val).astype(np.float64)
    return errors, target


def bootstrap_pair_across_seeds(
    seed_diffs: list[np.ndarray],
    block_len: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    boot_means = np.empty(n_bootstrap, dtype=np.float64)
    lengths = [diff.size for diff in seed_diffs]
    total_len = int(sum(lengths))

    for b in range(n_bootstrap):
        pieces = []
        for diff in seed_diffs:
            max_start = diff.size - block_len
            blocks_needed = math.ceil(diff.size / block_len)
            starts = rng.integers(0, max_start + 1, size=blocks_needed)
            sampled = np.concatenate([diff[s : s + block_len] for s in starts])[: diff.size]
            pieces.append(sampled)
        boot_means[b] = float(np.concatenate(pieces).mean())

    if not all(length >= block_len for length in lengths):
        raise ValueError("All seed series must be at least one block long")
    if total_len <= 0:
        raise ValueError("No bootstrap observations")
    return boot_means


def main() -> None:
    parser = argparse.ArgumentParser(description="Run day-block bootstrap robustness for forecast-origin losses.")
    parser.add_argument("--dataset", default="PEMS04")
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--block-len", type=int, default=None, help="Forecast origins per block. Defaults to one day.")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument(
        "--original-checkpoint-root",
        type=Path,
        default=None,
        help=(
            "Optional external BasicTS checkpoint root used as a fallback for test_results. "
            "Use only after verifying those arrays match the official manuscript metrics."
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rng-seed", type=int, default=20260616)
    args = parser.parse_args()

    desc = load_dataset_desc(args.dataset)
    settings = desc["regular_settings"]
    horizon = int(settings["OUTPUT_LEN"])
    null_val = float(settings.get("NULL_VAL", 0.0))
    num_nodes = int(desc["num_nodes"])
    frequency_minutes = int(desc["frequency (minutes)"])
    block_len = args.block_len or day_block_length(frequency_minutes)

    per_model_seed_errors: dict[str, dict[int, np.ndarray]] = {model: {} for model in MODEL_ORDER}
    target_reference: dict[int, np.ndarray] = {}
    source_dirs: dict[str, dict[int, str]] = {model: {} for model in MODEL_ORDER}

    for model in MODEL_ORDER:
        for seed in args.seeds:
            dump_dir = find_dump_dir(args, model, args.dataset, seed)
            if dump_dir is None:
                continue
            errors, target = load_errors_for_seed(
                dump_dir,
                num_nodes=num_nodes,
                horizon=horizon,
                null_val=null_val,
            )

            if seed not in target_reference:
                target_reference[seed] = target
            elif target.shape != target_reference[seed].shape or not np.allclose(target, target_reference[seed], atol=1e-5):
                raise ValueError(f"Targets differ for {model}/{args.dataset}/seed{seed}")

            if errors.size < block_len:
                raise ValueError(f"{model}/{args.dataset}/seed{seed} has fewer forecast origins than block_len")

            per_model_seed_errors[model][seed] = errors
            source_dirs[model][seed] = str(dump_dir)

    available_models = [model for model in MODEL_ORDER if per_model_seed_errors[model]]
    if len(available_models) < 2:
        raise RuntimeError(f"Need at least two models with usable dumps for {args.dataset}")

    rng = np.random.default_rng(args.rng_seed)
    lower_q = 100 * args.alpha / 2
    upper_q = 100 * (1 - args.alpha / 2)

    rows = []
    for model_a, model_b in itertools.combinations(available_models, 2):
        seeds_used = [seed for seed in args.seeds if seed in per_model_seed_errors[model_a] and seed in per_model_seed_errors[model_b]]
        if not seeds_used:
            continue

        err_a_by_seed = [per_model_seed_errors[model_a][seed] for seed in seeds_used]
        err_b_by_seed = [per_model_seed_errors[model_b][seed] for seed in seeds_used]
        filtered_a_by_seed = []
        filtered_b_by_seed = []
        seed_diffs = []
        valid_origin_counts = []
        for err_a_seed, err_b_seed in zip(err_a_by_seed, err_b_by_seed):
            finite = np.isfinite(err_a_seed) & np.isfinite(err_b_seed)
            filtered_a = err_a_seed[finite]
            filtered_b = err_b_seed[finite]
            if filtered_a.size < block_len:
                raise ValueError(f"{model_a} vs {model_b} has fewer valid origins than block_len")
            filtered_a_by_seed.append(filtered_a)
            filtered_b_by_seed.append(filtered_b)
            seed_diffs.append(filtered_a - filtered_b)
            valid_origin_counts.append(int(finite.sum()))

        err_a = np.concatenate(filtered_a_by_seed)
        err_b = np.concatenate(filtered_b_by_seed)
        diff = np.concatenate(seed_diffs)
        original_mean_diff = float(diff.mean())
        boot_means = bootstrap_pair_across_seeds(seed_diffs, block_len, args.n_bootstrap, rng)
        ci_low, ci_high = np.percentile(boot_means, [lower_q, upper_q])
        direction = "model_a_better" if original_mean_diff < 0 else "model_b_better"
        sign_agree = float(np.mean(boot_means < 0)) if original_mean_diff < 0 else float(np.mean(boot_means > 0))

        rows.append(
            {
                "dataset": args.dataset,
                "requested_seeds": ",".join(str(seed) for seed in args.seeds),
                "model_a": model_a,
                "model_b": model_b,
                "model_a_display": format_model_name(model_a),
                "model_b_display": format_model_name(model_b),
                "mae_a": float(err_a.mean()),
                "mae_b": float(err_b.mean()),
                "mean_loss_diff_a_minus_b": original_mean_diff,
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
                "direction": direction,
                "sign_agreement_fraction": sign_agree,
                "seeds_used": ",".join(str(seed) for seed in seeds_used),
                "n_seeds_used": len(seeds_used),
                "n_forecast_origins": int(diff.size),
                "valid_origins_by_seed": ",".join(str(n) for n in valid_origin_counts),
                "block_len": int(block_len),
                "n_bootstrap": int(args.n_bootstrap),
            }
        )

    seed_stem = "seeds" + "-".join(str(seed) for seed in args.seeds)
    out_stem = f"block_bootstrap_dm_robustness_{args.dataset.lower()}_{seed_stem}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"{out_stem}.json"
    csv_path = args.out_dir / f"{out_stem}.csv"

    output = {
        "method": "moving block bootstrap over forecast origins",
        "purpose": (
            "Dependence-aware robustness check for pairwise loss-difference conclusions; "
            "negative mean_loss_diff_a_minus_b means model_a has lower MAE."
        ),
        "dataset": args.dataset,
        "requested_seeds": args.seeds,
        "frequency_minutes": frequency_minutes,
        "horizon": horizon,
        "num_nodes": num_nodes,
        "null_val": null_val,
        "block_len": int(block_len),
        "block_interpretation": "one day" if block_len == day_block_length(frequency_minutes) else "custom",
        "n_bootstrap": int(args.n_bootstrap),
        "alpha": args.alpha,
        "models": available_models,
        "source_dirs": source_dirs,
        "pairs": rows,
        "summary": {
            "n_pairs": len(rows),
            "n_ci_excluding_zero": int(sum(row["ci_excludes_zero"] for row in rows)),
            "min_sign_agreement_fraction": float(min(row["sign_agreement_fraction"] for row in rows)),
            "pairs_by_seed_count": {
                str(n): int(sum(row["n_seeds_used"] == n for row in rows))
                for n in sorted({row["n_seeds_used"] for row in rows})
            },
        },
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    print(
        f"{output['summary']['n_ci_excluding_zero']}/{output['summary']['n_pairs']} "
        f"pairwise CIs exclude zero; min sign agreement = "
        f"{output['summary']['min_sign_agreement_fraction']:.3f}"
    )


if __name__ == "__main__":
    main()
