"""Generate the corrected fixed-mask sensor-dropout summary figure."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "results" / "robustness" / "sensor_dropout_fixed_masks_seed42.json"
OUTPUT = ROOT / "figures" / "main" / "xai7_sensor_dropout_robustness.png"
DATASETS = ("METR-LA", "PEMS-BAY", "PEMS04")
COLORS = {"D2STGNN": "#167D8D", "MegaCRN": "#D56A4C", "STID": "#715A9B"}


def main():
    with SOURCE.open("r", encoding="utf-8") as handle:
        results = json.load(handle)["datasets"]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), dpi=180, sharey=True)
    rates = np.array([0, 10, 30])
    for ax, dataset in zip(axes, DATASETS):
        models = results[dataset]["models"]
        for model, model_result in models.items():
            changes = [
                model_result["rates"][f"{rate}%"]["relative_mae_change_percent"]
                for rate in rates
            ]
            ax.plot(
                rates,
                changes,
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                color=COLORS[model],
                label=model,
            )
        ax.axhline(0, color="#30343B", linewidth=0.9)
        ax.set_title(dataset, fontweight="bold")
        ax.set_xlabel("Dropped sensors (%)")
        ax.set_xticks(rates)
        ax.grid(color="#D9DDE2", linewidth=0.7, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Relative MAE change (%)")

    handles = {}
    for ax in axes:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            handles[label] = handle
    fig.legend(
        handles.values(),
        handles.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Deterministic fixed-mask sensor-dropout stress test",
        y=1.10,
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
