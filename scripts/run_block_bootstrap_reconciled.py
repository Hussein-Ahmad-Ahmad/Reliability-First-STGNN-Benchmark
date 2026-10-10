"""Forecast-origin moving-block bootstrap with flat-pooled sum/count aggregation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load_pairwise_module():
    spec = importlib.util.spec_from_file_location(
        "run_block_bootstrap_pairwise", ROOT / "scripts" / "run_block_bootstrap_pairwise.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bb = _load_pairwise_module()

# Matches framework/basicts/metrics/mae.py's null-value tolerance exactly
# (atol=5e-5, rtol=0), not numpy.isclose's default tolerance.
NULL_ATOL = 5e-5


def per_origin_sum_and_count(pred: np.ndarray, target: np.ndarray, null_val: float) -> tuple[np.ndarray, np.ndarray]:
    abs_error = np.abs(pred - target).astype(np.float64)
    valid = ~np.isclose(target, null_val, atol=NULL_ATOL, rtol=0.0)
    counts = valid.sum(axis=(1, 2)).astype(np.float64)
    sums = (abs_error * valid).sum(axis=(1, 2))
    return sums, counts


def load_sum_count_for_seed(dump_dir: Path, num_nodes: int, horizon: int, null_val: float):
    pred_path = dump_dir / "predictions.npy"
    target_path = dump_dir / "targets.npy"
    metadata_path = dump_dir / 'export_metadata.json'
    if not metadata_path.is_file():
        raise ValueError(f'Validated export metadata is required: {dump_dir}')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if not metadata.get('verified'):
        raise ValueError(f'Unverified export: {dump_dir}')
    for path in (pred_path, target_path):
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != metadata[f'{path.stem}_sha256']:
            raise ValueError(f'Export checksum mismatch: {path}')
    pred = bb.reshape_prediction(bb.read_prediction_array(pred_path, num_nodes, horizon), num_nodes, horizon, pred_path)
    target = bb.reshape_prediction(bb.read_prediction_array(target_path, num_nodes, horizon), num_nodes, horizon, target_path)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/target shape mismatch in {dump_dir}: {pred.shape} vs {target.shape}")
    if pred.shape[0] != metadata['test_origins'] or not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError(f'Incomplete or nonfinite chronological export: {dump_dir}')
    sums, counts = per_origin_sum_and_count(pred, target, null_val)
    return sums, counts, target


def flat_pooled_mae(sums: np.ndarray, counts: np.ndarray) -> float:
    return float(sums.sum() / counts.sum())


def aggregate_sum_count_across_seeds(
    sums_by_seed: list[np.ndarray], counts_by_seed: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse per-seed per-origin (sum, count) arrays into one seed-aggregated
    per-origin sequence, origin-index-aligned. Valid because the test-set windowing
    (and therefore the forecast-origin timestamp at each index) does not depend on
    the training seed - verified elsewhere that origin counts and target arrays are
    identical across seeds 43/44/45 for a given (model, dataset)."""
    total_sum = np.sum(np.stack(sums_by_seed, axis=0), axis=0)
    total_count = np.sum(np.stack(counts_by_seed, axis=0), axis=0)
    return total_sum, total_count


def bootstrap_flat_pooled_diff(
    seed_agg_sum_a: np.ndarray,
    seed_agg_count_a: np.ndarray,
    seed_agg_sum_b: np.ndarray,
    seed_agg_count_b: np.ndarray,
    block_len: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Block-bootstrap a single seed-aggregated origin sequence.

    One shared set of block-start indices is drawn per replicate and applied to
    both models (they share the same origin/target timestamps), matching
    the seed-aggregated forecast-origin estimand: origin-level
    losses are seed-averaged (here, seed-summed, which is equivalent up to a
    constant factor for the ratio in mae_a/mae_b) before differencing, so the
    resampling unit should be one seed-aggregated temporal sequence, not
    independently-resampled per-seed blocks recombined afterward.
    """
    arrays = [np.asarray(value, dtype=np.float64) for value in
              (seed_agg_sum_a, seed_agg_count_a, seed_agg_sum_b, seed_agg_count_b)]
    n = arrays[0].size
    if any(value.ndim != 1 or value.size != n or not np.isfinite(value).all() for value in arrays):
        raise ValueError('Paired sums/counts must be finite, equally sized origin sequences.')
    if not 0 < block_len <= n or n_bootstrap <= 0:
        raise ValueError('Block length must be within the origin count and replicate count must be positive.')
    if any(np.any(value < 0) or value.sum() <= 0 for value in (arrays[1], arrays[3])):
        raise ValueError('Valid-element counts must be nonnegative and have positive totals.')
    max_start = n - block_len
    blocks_needed = math.ceil(n / block_len)
    boot_diffs = np.empty(n_bootstrap, dtype=np.float64)
    prefixes = [np.concatenate(([0.0], np.cumsum(value))) for value in arrays]
    lengths = np.full(blocks_needed, block_len, dtype=np.int64)
    lengths[-1] = n - block_len * (blocks_needed - 1)
    # Prefix sums preserve the shared blocks and final truncation without materializing every origin index.
    for offset in range(0, n_bootstrap, 512):
        stop = min(offset + 512, n_bootstrap)
        starts = rng.integers(0, max_start + 1, size=(stop - offset, blocks_needed))
        ends = starts + lengths
        totals = [(prefix[ends] - prefix[starts]).sum(axis=1) for prefix in prefixes]
        if np.any(totals[1] <= 0) or np.any(totals[3] <= 0):
            raise ValueError('A bootstrap replicate contains no valid target elements.')
        boot_diffs[offset:stop] = totals[0] / totals[1] - totals[2] / totals[3]
    return boot_diffs


def main() -> None:
    parser = argparse.ArgumentParser(description="Flat-pooled forecast-origin bootstrap")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--block-len", type=int, default=None, help="Defaults to one day")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--dump-root", type=Path, default=bb.DEFAULT_DUMP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "task1_point_forecasting")
    parser.add_argument("--rng-seed", type=int, default=20260616)
    args = parser.parse_args()

    desc = bb.load_dataset_desc(args.dataset)
    settings = desc["regular_settings"]
    horizon = int(settings["OUTPUT_LEN"])
    null_val = float(settings.get("NULL_VAL", 0.0))
    num_nodes = int(desc["num_nodes"])
    frequency_minutes = int(desc["frequency (minutes)"])
    block_len = args.block_len or bb.day_block_length(frequency_minutes)

    per_model_seed_sc: dict[str, dict[int, tuple[np.ndarray, np.ndarray]]] = {m: {} for m in bb.MODEL_ORDER}
    for model in bb.MODEL_ORDER:
        for seed in args.seeds:
            dump_dir = args.dump_root / model / args.dataset / f"seed{seed}"
            if not (dump_dir / "predictions.npy").is_file():
                continue
            sums, counts, _ = load_sum_count_for_seed(dump_dir, num_nodes, horizon, null_val)
            per_model_seed_sc[model][seed] = (sums, counts)

    available_models = [m for m in bb.MODEL_ORDER if per_model_seed_sc[m]]
    if len(available_models) < 2:
        raise RuntimeError(f"Need at least two models with usable dumps for {args.dataset}")

    # Headline per-model flat-pooled MAE using each model's own full available
    # seed set (matches the archived metric convention when all 3 seeds are present).
    # NOT necessarily what a given pair's point estimate uses - see below.
    model_mae_full = {}
    model_seed_counts = {}
    for model in available_models:
        seeds_used = sorted(per_model_seed_sc[model])
        total_sum = sum(per_model_seed_sc[model][s][0].sum() for s in seeds_used)
        total_count = sum(per_model_seed_sc[model][s][1].sum() for s in seeds_used)
        model_mae_full[model] = total_sum / total_count
        model_seed_counts[model] = len(seeds_used)

    rng = np.random.default_rng(args.rng_seed)
    lower_q = 100 * args.alpha / 2
    upper_q = 100 * (1 - args.alpha / 2)

    rows = []
    for model_a, model_b in itertools.combinations(available_models, 2):
        seeds_used = sorted(set(per_model_seed_sc[model_a]) & set(per_model_seed_sc[model_b]))
        sums_a = [per_model_seed_sc[model_a][s][0] for s in seeds_used]
        counts_a = [per_model_seed_sc[model_a][s][1] for s in seeds_used]
        sums_b = [per_model_seed_sc[model_b][s][0] for s in seeds_used]
        counts_b = [per_model_seed_sc[model_b][s][1] for s in seeds_used]

        # Seed-aggregate first (origins are index-aligned across seeds - same
        # forecast-origin timestamps regardless of training seed), then draw one
        # shared block-start sample per bootstrap replicate from that single
        # sequence, rather than independently resampling blocks per seed.
        agg_sum_a, agg_count_a = aggregate_sum_count_across_seeds(sums_a, counts_a)
        agg_sum_b, agg_count_b = aggregate_sum_count_across_seeds(sums_b, counts_b)

        # Point estimate MUST be computed from the same seeds_used intersection
        # the bootstrap resamples, not each model's own full seed set - otherwise
        # a model missing some seeds (e.g. MegaCRN/MTGNN on PEMS04 only have
        # seed 43 in this repo's fresh_inference_dumps) silently produces a point
        # estimate that mixes a multi-seed average for one model against a
        # single-seed value for the other, while the CI is built from the
        # single-seed-only intersection - an apples-to-oranges mismatch that can
        # place the point estimate outside its own reported CI for no genuine
        # statistical reason.
        seed_mismatch = seeds_used != sorted(per_model_seed_sc[model_a]) or seeds_used != sorted(per_model_seed_sc[model_b])
        mae_a_matched = agg_sum_a.sum() / agg_count_a.sum()
        mae_b_matched = agg_sum_b.sum() / agg_count_b.sum()
        point_diff = mae_a_matched - mae_b_matched
        boot_diffs = bootstrap_flat_pooled_diff(agg_sum_a, agg_count_a, agg_sum_b, agg_count_b, block_len, args.n_bootstrap, rng)
        ci_low, ci_high = np.percentile(boot_diffs, [lower_q, upper_q])

        rows.append({
            "dataset": args.dataset,
            "model_a": model_a,
            "model_b": model_b,
            "model_a_display": bb.format_model_name(model_a),
            "model_b_display": bb.format_model_name(model_b),
            "mae_a_flat_pooled_seeds_used": mae_a_matched,
            "mae_b_flat_pooled_seeds_used": mae_b_matched,
            "mae_a_flat_pooled_full_seed_set": model_mae_full[model_a],
            "mae_b_flat_pooled_full_seed_set": model_mae_full[model_b],
            "mean_loss_diff_a_minus_b": point_diff,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "point_estimate_inside_ci": bool(ci_low <= point_diff <= ci_high),
            "block_len": block_len,
            "n_bootstrap": args.n_bootstrap,
            "seeds_used": seeds_used,
            "seed_mismatch_vs_full_set": seed_mismatch,
        })

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_stem = f"block_bootstrap_reconciled_{args.dataset.lower()}_seeds" + "-".join(str(s) for s in args.seeds)
    json_path = args.out_dir / f"{out_stem}.json"
    csv_path = args.out_dir / f"{out_stem}.csv"

    output = {
        "method": "moving block bootstrap over forecast origins, flat-pooled (sum/count) aggregation matching official masked_mae",
        "aggregation_note": "Flat-pooled MAE is the sum of absolute errors over valid elements divided by their count. Input exports are chronological complete passes verified against archived online MAE within 1e-5, with array checksums and origin counts checked before loading. Pairwise estimates use the same common seed intersection as their resampling intervals.",
        "block_sampling_design": "One shared block-start sample is drawn per replicate from a seed-aggregated per-origin sum/count sequence and applied to both models. Seeds are not independently resampled; each pair uses its disclosed common seed intersection.",
        "dataset": args.dataset,
        "block_len": block_len,
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "model_flat_pooled_mae_full_seed_set": model_mae_full,
        "model_seed_counts": model_seed_counts,
        "point_estimate_seed_note": (
            "Each pair's mean_loss_diff_a_minus_b and its CI are computed from the "
            "SAME seed intersection (seeds_used) for both models in that pair, not "
            "each model's own full seed set. When seed_mismatch_vs_full_set is true, "
            "this pair's point estimate differs from what you'd get by subtracting "
            "the two model_flat_pooled_mae_full_seed_set values directly."
        ),
        "pairs": rows,
    }
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV:  {csv_path}")
    for model, mae in model_mae_full.items():
        print(f"  {model}: flat-pooled MAE ({model_seed_counts[model]} seeds) = {mae:.6f}")
    for row in rows:
        flag = " [SEED MISMATCH]" if row["seed_mismatch_vs_full_set"] else ""
        print(f"  {row['model_a_display']} vs {row['model_b_display']}: diff={row['mean_loss_diff_a_minus_b']:+.4f} CI=[{row['ci_low']:+.4f},{row['ci_high']:+.4f}]{flag}")


if __name__ == "__main__":
    main()
