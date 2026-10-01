"""Block-length sensitivity sweep using the RECONCILED (Table-12-consistent) bootstrap.

scripts/run_block_bootstrap_sensitivity.py shares the same aggregation bug
found and fixed in scripts/run_block_bootstrap_reconciled.py: it calls the
original run_block_bootstrap_pairwise.py's masked_per_origin_mae (mean of
per-origin means), not the flat-pooled aggregation that Table 12 and every
archived test_metrics.json actually use. That produced pairwise differences
disagreeing with Table 12 by up to ~0.017 MAE, which changes which CIs
exclude zero. This script re-runs the same 72/144/288-origin sensitivity
sweep using run_block_bootstrap_reconciled.py's per-origin (sum, count)
resampling instead, and is the corrected replacement for that sweep's
results. The original run_block_bootstrap_sensitivity.py is left in place
for provenance; do not use its output for Table 15 sensitivity reporting.

Usage:
    python scripts/run_block_bootstrap_sensitivity_reconciled.py
    python scripts/run_block_bootstrap_sensitivity_reconciled.py --dataset METR-LA
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLOCK_LENGTHS = (72, 144, 288)


def _load_reconciled_module():
    spec = importlib.util.spec_from_file_location(
        "run_block_bootstrap_reconciled", ROOT / "scripts" / "run_block_bootstrap_reconciled.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


rc = _load_reconciled_module()


def run_one(dataset: str, block_len: int, seeds: list[int], n_bootstrap: int, alpha: float, rng_seed: int, dump_root: Path) -> dict:
    desc = rc.bb.load_dataset_desc(dataset)
    settings = desc["regular_settings"]
    horizon = int(settings["OUTPUT_LEN"])
    null_val = float(settings.get("NULL_VAL", 0.0))
    num_nodes = int(desc["num_nodes"])

    per_model_seed_sc: dict[str, dict[int, tuple]] = {m: {} for m in rc.bb.MODEL_ORDER}
    for model in rc.bb.MODEL_ORDER:
        for seed in seeds:
            dump_dir = dump_root / model / dataset / f"seed{seed}"
            if not (dump_dir / "predictions.npy").is_file():
                continue
            sums, counts, _ = rc.load_sum_count_for_seed(dump_dir, num_nodes, horizon, null_val)
            per_model_seed_sc[model][seed] = (sums, counts)

    available_models = [m for m in rc.bb.MODEL_ORDER if per_model_seed_sc[m]]
    if len(available_models) < 2:
        raise RuntimeError(f"Need at least two models with usable dumps for {dataset}")

    rng = np.random.default_rng(rng_seed)
    lower_q = 100 * alpha / 2
    upper_q = 100 * (1 - alpha / 2)

    import itertools

    rows = []
    for model_a, model_b in itertools.combinations(available_models, 2):
        seeds_used = sorted(set(per_model_seed_sc[model_a]) & set(per_model_seed_sc[model_b]))
        sums_a = [per_model_seed_sc[model_a][s][0] for s in seeds_used]
        counts_a = [per_model_seed_sc[model_a][s][1] for s in seeds_used]
        sums_b = [per_model_seed_sc[model_b][s][0] for s in seeds_used]
        counts_b = [per_model_seed_sc[model_b][s][1] for s in seeds_used]
        agg_sum_a, agg_count_a = rc.aggregate_sum_count_across_seeds(sums_a, counts_a)
        agg_sum_b, agg_count_b = rc.aggregate_sum_count_across_seeds(sums_b, counts_b)

        # Point estimate from the SAME seed intersection the bootstrap uses (not
        # each model's own full seed set) - see run_block_bootstrap_reconciled.py's
        # identical fix for why this matters when seed coverage is incomplete.
        point_diff = (agg_sum_a.sum() / agg_count_a.sum()) - (agg_sum_b.sum() / agg_count_b.sum())
        boot_diffs = rc.bootstrap_flat_pooled_diff(agg_sum_a, agg_count_a, agg_sum_b, agg_count_b, block_len, n_bootstrap, rng)
        ci_low, ci_high = np.percentile(boot_diffs, [lower_q, upper_q])
        rows.append({
            "model_a": model_a,
            "model_b": model_b,
            "mean_loss_diff_a_minus_b": point_diff,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_excludes_zero": bool(ci_low > 0 or ci_high < 0),
            "seeds_used": seeds_used,
        })

    return {
        "block_len": block_len,
        "n_pairs": len(rows),
        "n_ci_excluding_zero": int(sum(r["ci_excludes_zero"] for r in rows)),
        "pairs": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconciled block-length sensitivity sweep")
    parser.add_argument("--dataset", action="append", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--rng-seed", type=int, default=20260616)
    parser.add_argument("--dump-root", type=Path, default=rc.bb.DEFAULT_DUMP_ROOT)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "task1_point_forecasting")
    args = parser.parse_args()

    datasets = args.dataset or ["METR-LA", "PEMS-BAY", "PEMS04"]

    for dataset in datasets:
        by_block_len = {}
        for block_len in BLOCK_LENGTHS:
            print(f"=== {dataset}: block_len={block_len} ===")
            by_block_len[str(block_len)] = run_one(dataset, block_len, args.seeds, args.n_bootstrap, args.alpha, args.rng_seed, args.dump_root)

        output = {
            "dataset": dataset,
            "seeds": args.seeds,
            "n_bootstrap": args.n_bootstrap,
            "alpha": args.alpha,
            "block_lengths_tested": list(BLOCK_LENGTHS),
            "aggregation": "flat-pooled sum/count, matching Table 12 and official masked_mae (see run_block_bootstrap_reconciled.py)",
            "supersedes": "results/task1_point_forecasting/block_bootstrap_sensitivity_<dataset>_seeds43-44-45.json (buggy mean-of-per-origin-means aggregation)",
            "results_by_block_len": by_block_len,
        }

        seed_stem = "seeds" + "-".join(str(s) for s in args.seeds)
        out_path = args.out_dir / f"block_bootstrap_sensitivity_reconciled_{dataset.lower()}_{seed_stem}.json"
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        print(out_path)
        for block_len in BLOCK_LENGTHS:
            r = by_block_len[str(block_len)]
            print(f"  block_len={block_len}: {r['n_ci_excluding_zero']}/{r['n_pairs']} CIs exclude zero")


if __name__ == "__main__":
    main()
