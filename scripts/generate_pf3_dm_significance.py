"""Generate the METR-LA Diebold-Mariano significance figure.

Input:
    results/task1_point_forecasting/dm_metrla_21pairs_mae_aligned.json

Output:
    figures/main/pf3_dm_significance_metrla.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "results/task1_point_forecasting/dm_metrla_21pairs_mae_aligned.json"
OUT_PATH = ROOT / "figures/main/pf3_dm_significance_metrla.png"


def stars(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    models = payload["models_ordered_by_mae"]
    display = payload["model_display_names"]
    labels = [display[m] for m in models]
    wins = [payload["wins"][m] for m in models]

    p_lookup = {}
    better_lookup = {}
    for pair in payload["pairs"]:
        m1, m2 = pair["model1"], pair["model2"]
        p = max(float(pair["p_value_holm_adjusted"]), 1e-15)
        p_lookup[(m1, m2)] = p
        p_lookup[(m2, m1)] = p
        better_lookup[(m1, m2)] = pair["better_model"]
        better_lookup[(m2, m1)] = pair["better_model"]

    n = len(models)
    heatmap = np.full((n, n), np.nan)
    for i, row_model in enumerate(models):
        for j, col_model in enumerate(models):
            if i == j:
                continue
            p = p_lookup[(row_model, col_model)]
            value = -np.log10(p)
            heatmap[i, j] = value if better_lookup[(row_model, col_model)] == row_model else -value

    colors = ["#1f77b4", "#e377c2", "#ff7f0e", "#2ca02c", "#17becf", "#bcbd22", "#7ecef4"]
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig, (ax_bar, ax_heat) = plt.subplots(
        1,
        2,
        figsize=(12, 5),
        gridspec_kw={"width_ratios": [1, 1.8]},
    )
    fig.suptitle(
        "Diebold-Mariano Test: Pairwise Significance on METR-LA",
        fontweight="bold",
        fontsize=12,
        y=1.01,
    )

    y_pos = np.arange(n)
    bars = ax_bar.barh(y_pos, wins, color=colors, edgecolor="white", height=0.6)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels)
    ax_bar.invert_yaxis()
    ax_bar.set_xlim(0, n)
    ax_bar.set_xlabel("Significant win count")
    ax_bar.set_xticks(range(0, n + 1))
    ax_bar.grid(axis="x", linestyle="--", alpha=0.35)
    ax_bar.set_title("MAE-Aligned DM Wins", fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    for bar, count in zip(bars, wins):
        x = count + 0.1 if count > 0 else 0.1
        ax_bar.text(x, bar.get_y() + bar.get_height() / 2, str(count), va="center", ha="left")

    vmax = 15.0
    im = ax_heat.imshow(
        np.ma.masked_invalid(heatmap),
        cmap=plt.cm.RdYlGn,
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        aspect="auto",
    )
    for i in range(n):
        ax_heat.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="lightgray", zorder=2))
        ax_heat.text(i, i, "-", ha="center", va="center", color="gray", zorder=3)

    for i, row_model in enumerate(models):
        for j, col_model in enumerate(models):
            if i == j:
                continue
            p = p_lookup[(row_model, col_model)]
            label = stars(p)
            if label:
                color = "white" if abs(heatmap[i, j]) > 5 else "black"
                ax_heat.text(j, i, label, ha="center", va="center", fontsize=7, color=color)

    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(labels, rotation=45, ha="right")
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(labels)
    ax_heat.set_title("Signed -log10(Holm p-value)\nGreen = row model better", fontsize=10)
    cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("signed -log10(Holm p)")

    plt.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
