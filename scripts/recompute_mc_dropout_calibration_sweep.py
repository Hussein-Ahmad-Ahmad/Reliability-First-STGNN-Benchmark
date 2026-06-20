import json
from pathlib import Path

import numpy as np

SRC = Path(r"D:/Hussein-Files/original/experiments/publishable_experiment/results/mc_dropout/calibration_sweep.json")
OUT_JSON = Path("results/task2_uncertainty/mc_dropout_calibration_recomputed.json")
OUT_MD = Path("results/task2_uncertainty/mc_dropout_calibration_recomputed.md")

TARGET_COVERAGE_PERCENT = 90.0


def main() -> None:
    data = json.loads(SRC.read_text(encoding="utf-8"))
    rates = data.get("dropout_rates", [])
    passes = data.get("forward_passes", [])
    models = data.get("models", {})

    out = {
        "source": str(SRC),
        "target_coverage_percent": TARGET_COVERAGE_PERCENT,
        "rows": [],
    }

    for model, payload in models.items():
        picp = np.array(payload.get("picp_grid", []), dtype=float)
        mpiw = np.array(payload.get("mpiw_grid", []), dtype=float)
        if picp.size == 0 or mpiw.size == 0:
            continue

        idx_max = np.unravel_index(np.argmax(picp), picp.shape)
        max_picp_percent = float(picp[idx_max])
        max_picp_fraction = max_picp_percent / 100.0
        mpiw_at_max_picp = float(mpiw[idx_max])
        tightest_mpiw = float(np.min(mpiw))
        idx_tightest = np.unravel_index(np.argmin(mpiw), mpiw.shape)

        feasible = np.where(picp >= TARGET_COVERAGE_PERCENT)
        best_feasible = None
        if feasible[0].size > 0:
            cands = list(zip(feasible[0], feasible[1]))
            best_idx = min(cands, key=lambda ij: mpiw[ij])
            i, j = best_idx
            best_feasible = {
                "picp_percent": float(picp[i, j]),
                "picp_fraction": float(picp[i, j] / 100.0),
                "mpiw": float(mpiw[i, j]),
                "dropout_rate": float(rates[i]),
                "forward_passes": int(passes[j]),
            }

        row = {
            "model": model,
            "max_picp_percent": max_picp_percent,
            "max_picp_fraction": max_picp_fraction,
            "mpiw_at_max_picp": mpiw_at_max_picp,
            "max_picp_dropout_rate": float(rates[idx_max[0]]),
            "max_picp_forward_passes": int(passes[idx_max[1]]),
            "tightest_mpiw": tightest_mpiw,
            "tightest_mpiw_dropout_rate": float(rates[idx_tightest[0]]),
            "tightest_mpiw_forward_passes": int(passes[idx_tightest[1]]),
            "meets_90pct_target_anywhere": best_feasible is not None,
            "best_feasible_90pct": best_feasible,
        }
        out["rows"].append(row)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    lines = [
        "# MC-Dropout Calibration Sweep Recomputed",
        "",
        f"Source: {SRC}",
        f"Target coverage (percent scale): {TARGET_COVERAGE_PERCENT}",
        "",
        "| Model | Max PICP % | Max PICP frac | MPIW at max PICP | Meets 90% target? | Best feasible MPIW @>=90% |", 
        "|---|---:|---:|---:|---|---:|",
    ]
    for r in out["rows"]:
        best_mpiw = "NA"
        if r["best_feasible_90pct"] is not None:
            best_mpiw = f"{r['best_feasible_90pct']['mpiw']:.6f}"
        lines.append(
            f"| {r['model']} | {r['max_picp_percent']:.6f} | {r['max_picp_fraction']:.6f} | "
            f"{r['mpiw_at_max_picp']:.6f} | {'Yes' if r['meets_90pct_target_anywhere'] else 'No'} | {best_mpiw} |"
        )
    lines.append("")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
