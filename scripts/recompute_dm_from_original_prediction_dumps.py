from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ORIGINAL_PRED_ROOT = Path(r"D:/Hussein-Files/original/results/evidence_hub/future_obtained_results/prediction_dumps")
OUT_JSON = Path("results/task1_point_forecasting/dm_recomputed_from_original_prediction_dumps.json")
OUT_MD = Path("results/task1_point_forecasting/dm_recomputed_from_original_prediction_dumps.md")


def dm_test_newey_west(errors1: np.ndarray, errors2: np.ndarray) -> dict:
    d = errors1 - errors2
    t = len(d)
    d_mean = float(np.mean(d))
    lags = int(np.floor(4 * (t / 100) ** (2 / 9)))
    gamma_0 = float(np.mean((d - d_mean) ** 2))
    hac_var = gamma_0
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma_lag = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        hac_var += 2 * weight * gamma_lag
    hac_se = float(np.sqrt(max(hac_var, 1e-10) / t))
    dm_stat = d_mean / hac_se
    p_value = float(2 * (1 - stats.norm.cdf(abs(dm_stat))))
    return {
        "dm_stat": dm_stat,
        "p_value": p_value,
        "mean_loss_diff": d_mean,
        "hac_se": hac_se,
        "n_obs": int(t),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted[orig_idx] = min(1.0, p * (n - rank))
    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    return adjusted


def list_coverage() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if not ORIGINAL_PRED_ROOT.exists():
        return out
    for model_dir in sorted(ORIGINAL_PRED_ROOT.iterdir()):
        if model_dir.is_dir():
            out[model_dir.name] = [d.name for d in sorted(model_dir.iterdir()) if d.is_dir()]
    return out


def main() -> None:
    coverage = list_coverage()
    all_datasets = sorted({d for ds in coverage.values() for d in ds})
    out: dict[str, object] = {
        "source_root": str(ORIGINAL_PRED_ROOT),
        "datasets": {},
        "notes": "Only pairs with common-seed prediction+target dumps are included.",
    }

    for dataset in all_datasets:
        models = sorted([m for m, ds in coverage.items() if dataset in ds])
        pairs = list(itertools.combinations(models, 2))
        rows = []
        pvals = []

        for m1, m2 in pairs:
            e1_all, e2_all, used_seeds = [], [], []
            for seed in (43, 44, 45):
                p1 = ORIGINAL_PRED_ROOT / m1 / dataset / f"seed{seed}" / "predictions.npy"
                t1 = ORIGINAL_PRED_ROOT / m1 / dataset / f"seed{seed}" / "targets.npy"
                p2 = ORIGINAL_PRED_ROOT / m2 / dataset / f"seed{seed}" / "predictions.npy"
                t2 = ORIGINAL_PRED_ROOT / m2 / dataset / f"seed{seed}" / "targets.npy"
                if not (p1.exists() and t1.exists() and p2.exists() and t2.exists()):
                    continue
                pred1, tgt1 = np.load(p1), np.load(t1)
                pred2, tgt2 = np.load(p2), np.load(t2)
                if pred1.shape != pred2.shape or tgt1.shape != tgt2.shape or tgt1.shape != pred1.shape:
                    continue
                if not np.allclose(tgt1, tgt2, atol=1e-8):
                    continue
                e1_all.append(np.abs(pred1 - tgt1).mean(axis=(-1, -2)).reshape(-1))
                e2_all.append(np.abs(pred2 - tgt2).mean(axis=(-1, -2)).reshape(-1))
                used_seeds.append(seed)

            if not e1_all:
                continue

            e1 = np.concatenate(e1_all)
            e2 = np.concatenate(e2_all)
            dm = dm_test_newey_west(e1, e2)
            winner = m1 if dm["dm_stat"] < 0 else m2
            row = {
                "pair": f"{m1}_vs_{m2}",
                "model1": m1,
                "model2": m2,
                "winner_by_dm_sign": winner,
                "seeds_used": used_seeds,
                **dm,
            }
            rows.append(row)
            pvals.append(dm["p_value"])

        if rows:
            adj = holm_bonferroni(pvals)
            for i, r in enumerate(rows):
                r["p_value_holm_corrected"] = float(adj[i])
                r["significant_after_holm"] = bool(adj[i] < 0.05)

        out["datasets"][dataset] = {
            "models_covered": models,
            "pairs_tested": len(rows),
            "rows": rows,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = ["# DM Recomputed From Original Prediction Dumps", "", f"Source: {ORIGINAL_PRED_ROOT}", ""]
    for dataset, payload in out["datasets"].items():
        md.append(f"## {dataset}")
        md.append("")
        mlist = ", ".join(payload["models_covered"]) if payload["models_covered"] else "(none)"
        md.append(f"- Models covered: {mlist}")
        md.append(f"- Pairs tested: {payload['pairs_tested']}")
        md.append("")
        md.append("| Pair | Winner by DM sign | DM stat | p(raw) | p(Holm) | Holm<0.05 | Seeds |")
        md.append("|---|---|---:|---:|---:|---|---|")
        for r in payload["rows"]:
            md.append(
                f"| {r['pair']} | {r['winner_by_dm_sign']} | {r['dm_stat']:.4f} | {r['p_value']:.6g} | "
                f"{r['p_value_holm_corrected']:.6g} | {'Yes' if r['significant_after_holm'] else 'No'} | "
                f"{','.join(map(str, r['seeds_used']))} |"
            )
        md.append("")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
