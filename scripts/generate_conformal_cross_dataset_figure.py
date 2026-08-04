"""Generate the corrected cross-dataset conformal summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "task2_uncertainty" / "conformal"
OUTPUT = ROOT / "figures" / "appendix" / "uq2_conformal_cross_dataset.png"
DATASETS = ("METR-LA", "PEMS-BAY", "PEMS04")
VARIANTS = (("fixed", "Fixed"), ("per_horizon", "Per horizon"))
COLORS = ("#167D8D", "#D56A4C")


def load_metrics():
    coverage = np.zeros((len(VARIANTS), len(DATASETS)))
    width = np.zeros_like(coverage)
    for variant_index, (variant, _) in enumerate(VARIANTS):
        for dataset_index, dataset in enumerate(DATASETS):
            path = RESULTS / f"{dataset}_conformal_{variant}_metrics.json"
            with path.open("r", encoding="utf-8") as handle:
                metrics = json.load(handle)["evaluation_set"]
            coverage[variant_index, dataset_index] = metrics["PICP"] * 100
            width[variant_index, dataset_index] = metrics["MPIW"]
    return coverage, width


def annotate(ax, bars, decimals):
    for bar in bars:
        value = bar.get_height()
        ax.annotate(
            f"{value:.{decimals}f}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def main():
    coverage, width = load_metrics()
    x = np.arange(len(DATASETS))
    bar_width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8), dpi=180)

    for index, (_, label) in enumerate(VARIANTS):
        offset = (index - 0.5) * bar_width
        bars = axes[0].bar(
            x + offset,
            coverage[index],
            bar_width,
            label=label,
            color=COLORS[index],
        )
        annotate(axes[0], bars, 2)
    axes[0].axhline(90, color="#30343B", linestyle="--", linewidth=1.2)
    axes[0].set_ylim(88.8, 91.4)
    axes[0].set_ylabel("Evaluation PICP (%)")
    axes[0].set_title("Empirical marginal coverage")
    axes[0].text(
        2.47,
        90.04,
        "90% target",
        fontsize=8.5,
        ha="right",
        va="bottom",
        color="#30343B",
    )

    for index, (_, label) in enumerate(VARIANTS):
        offset = (index - 0.5) * bar_width
        bars = axes[1].bar(
            x + offset,
            width[index],
            bar_width,
            label=label,
            color=COLORS[index],
        )
        annotate(axes[1], bars, 1)
    axes[1].set_ylabel("Evaluation MPIW")
    axes[1].set_title("Mean interval width (dataset units)")

    for ax in axes:
        ax.set_xticks(x, DATASETS)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#D9DDE2", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.suptitle(
        "Corrected normalized ensemble-conformal diagnostics",
        y=1.08,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(OUTPUT)


if __name__ == "__main__":
    main()
