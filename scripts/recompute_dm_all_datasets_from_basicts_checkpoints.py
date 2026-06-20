import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(r"D:/Hussein-Files/original/experiments/basicts/checkpoints")
FRESH_ROOT = Path("results/task1_point_forecasting/fresh_inference_dumps")
MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
DATASETS = ["METR-LA", "PEMS-BAY", "PEMS04"]
SEEDS = [43, 44, 45]
OUT_JSON = Path("results/task1_point_forecasting/dm_recomputed_all_datasets_from_basicts_checkpoints.json")
OUT_MD = Path("results/task1_point_forecasting/dm_recomputed_all_datasets_from_basicts_checkpoints.md")


def dm_test_newey_west(errors1: np.ndarray, errors2: np.ndarray):
    d = errors1 - errors2
    t = len(d)
    d_mean = float(np.mean(d))
    lags = int(np.floor(4 * (t / 100) ** (2 / 9)))
    gamma_0 = float(np.mean((d - d_mean) ** 2))
    hac_var = gamma_0
    for lag in range(1, lags + 1):
        w = 1 - lag / (lags + 1)
        g = float(np.mean((d[lag:] - d_mean) * (d[:-lag] - d_mean)))
        hac_var += 2 * w * g
    se = float(np.sqrt(max(hac_var, 1e-10) / t))
    dm = d_mean / se
    p = float(2 * (1 - stats.norm.cdf(abs(dm))))
    return {"dm_stat": dm, "p_value": p, "mean_loss_diff": d_mean, "hac_se": se, "n_obs": int(t)}


def holm_bonferroni(p_values):
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    for rank, (idx, p) in enumerate(indexed):
        adjusted[idx] = min(1.0, p * (n - rank))
    for i in range(len(adjusted) - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    return adjusted


def dataset_nodes(dataset: str) -> int:
    return {"METR-LA": 207, "PEMS-BAY": 325, "PEMS04": 307}[dataset]


def load_pred_or_target(path: Path, nodes: int, horizon: int = 12):
    try:
        return np.load(path, allow_pickle=True)
    except Exception:
        raw = np.fromfile(path, dtype=np.float32)
        step = horizon * nodes
        if raw.size % step != 0:
            raise
        return raw.reshape(-1, horizon, nodes)


def coerce_array(x):
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        x = x.item()
    return np.asarray(x)


def to_errors(pred, tgt):
    pred = coerce_array(pred)
    tgt = coerce_array(tgt)
    if pred.shape != tgt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {tgt.shape}")
    ae = np.abs(pred - tgt)
    if ae.ndim >= 3:
        return ae.mean(axis=tuple(range(1, ae.ndim))).reshape(-1)
    return ae.reshape(-1)


def find_seed_test_dir(model: str, dataset: str, seed: int):
    fresh_dir = FRESH_ROOT / model / dataset / f"seed{seed}"
    if (fresh_dir / "predictions.npy").exists() and (fresh_dir / "targets.npy").exists():
        return fresh_dir

    mroot = ROOT / model
    if not mroot.exists():
        return None

    candidates = []
    for d in mroot.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(dataset + "_"):
            continue
        if f"_seed{seed}" not in name:
            continue
        candidates.append(d)

    for d in sorted(candidates):
        runs = sorted([x for x in d.iterdir() if x.is_dir()])
        for run in runs:
            tr = run / "test_results"
            if (tr / "predictions.npy").exists() and (tr / "targets.npy").exists():
                return tr
    return None


def main():
    result = {"source_root": str(ROOT), "fresh_root": str(FRESH_ROOT), "datasets": {}}

    for ds in DATASETS:
        nodes = dataset_nodes(ds)
        cache = {m: {} for m in MODELS}
        for m in MODELS:
            for s in SEEDS:
                tr = find_seed_test_dir(m, ds, s)
                if tr is None:
                    continue
                pred_p = tr / "predictions.npy"
                tgt_p = tr / "targets.npy"
                try:
                    pred = load_pred_or_target(pred_p, nodes=nodes)
                    tgt = load_pred_or_target(tgt_p, nodes=nodes)
                    cache[m][s] = {"pred": pred, "tgt": tgt, "test_results_dir": str(tr)}
                except Exception:
                    continue

        covered_models = [m for m in MODELS if len(cache[m]) > 0]
        rows = []
        for m1, m2 in itertools.combinations(MODELS, 2):
            e1_all, e2_all, used = [], [], []
            for s in SEEDS:
                if s not in cache[m1] or s not in cache[m2]:
                    continue
                t1 = coerce_array(cache[m1][s]["tgt"])
                t2 = coerce_array(cache[m2][s]["tgt"])
                if t1.shape != t2.shape:
                    continue
                if not np.allclose(t1, t2, atol=1e-8):
                    continue
                try:
                    e1_all.append(to_errors(cache[m1][s]["pred"], t1))
                    e2_all.append(to_errors(cache[m2][s]["pred"], t2))
                    used.append(s)
                except Exception:
                    continue

            if not e1_all:
                rows.append({"pair": f"{m1}_vs_{m2}", "model1": m1, "model2": m2, "missing": True, "seeds_used": used})
                continue

            e1 = np.concatenate(e1_all)
            e2 = np.concatenate(e2_all)
            dm = dm_test_newey_west(e1, e2)
            winner = m1 if dm["dm_stat"] < 0 else m2
            rows.append({
                "pair": f"{m1}_vs_{m2}",
                "model1": m1,
                "model2": m2,
                "winner_by_dm_sign": winner,
                "seeds_used": used,
                **dm,
            })

        tested = [r for r in rows if not r.get("missing")]
        if tested:
            adj = holm_bonferroni([r["p_value"] for r in tested])
            for r, a in zip(tested, adj):
                r["p_value_holm_corrected"] = float(a)
                r["significant_after_holm"] = bool(a < 0.05)

        result["datasets"][ds] = {
            "models_with_any_seed": covered_models,
            "pairs_total": len(rows),
            "pairs_tested": len(tested),
            "rows": rows,
        }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# DM Recomputed All Datasets (from basicts checkpoints + fresh dumps)",
        "",
        f"Source root: {ROOT}",
        f"Fresh root: {FRESH_ROOT}",
        "",
    ]
    for ds in DATASETS:
        payload = result["datasets"][ds]
        lines.append(f"## {ds}")
        lines.append("")
        lines.append(f"- Models with at least one usable seed dump: {', '.join(payload['models_with_any_seed']) if payload['models_with_any_seed'] else '(none)'}")
        lines.append(f"- Pairs tested: {payload['pairs_tested']}/{payload['pairs_total']}")
        lines.append("")
        lines.append("| Pair | Winner by DM sign | DM stat | p(raw) | p(Holm) | Holm<0.05 | Seeds |")
        lines.append("|---|---|---:|---:|---:|---|---|")
        for r in payload["rows"]:
            if r.get("missing"):
                lines.append(f"| {r['pair']} | NA | NA | NA | NA | NA | {','.join(map(str,r['seeds_used'])) if r['seeds_used'] else '-'} |")
            else:
                lines.append(
                    f"| {r['pair']} | {r['winner_by_dm_sign']} | {r['dm_stat']:.4f} | {r['p_value']:.6g} | "
                    f"{r['p_value_holm_corrected']:.6g} | {'Yes' if r['significant_after_holm'] else 'No'} | "
                    f"{','.join(map(str,r['seeds_used']))} |"
                )
        lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    for ds in DATASETS:
        p = result['datasets'][ds]
        print(ds, f"pairs_tested={p['pairs_tested']}/{p['pairs_total']}")


if __name__ == '__main__':
    main()
