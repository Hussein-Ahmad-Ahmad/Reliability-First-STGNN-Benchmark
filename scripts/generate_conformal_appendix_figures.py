"""Generate appendix conformal figures from compact metrics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "task2_uncertainty" / "conformal"
OUTPUT = ROOT / "figures" / "appendix"
DATASETS = ("METR-LA", "PEMS-BAY", "PEMS04")
COLORS = {
    "METR-LA": "#1769AA",
    "PEMS-BAY": "#00856A",
    "PEMS04": "#C74440",
}
VARIANTS = {
    "fixed": {"label": "Fixed", "marker": "o", "linestyle": "-"},
    "per_horizon": {
        "label": "Per horizon",
        "marker": "s",
        "linestyle": "--",
    },
}


def load_metrics(dataset: str, variant: str) -> dict:
    path = RESULTS / f"{dataset}_conformal_{variant}_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def horizon_values(metrics: dict, key: str) -> np.ndarray:
    values = metrics["evaluation_set"]["per_horizon"]
    return np.asarray(
        [values[f"horizon_{index}"][key] for index in range(1, 13)],
        dtype=float,
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
        }
    )


def generate_tradeoff() -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.7))
    offsets = {"fixed": (5, 5), "per_horizon": (5, -12)}
    for dataset in DATASETS:
        for variant, style in VARIANTS.items():
            metrics = load_metrics(dataset, variant)["evaluation_set"]
            axis.scatter(
                metrics["MPIW"],
                100.0 * metrics["PICP"],
                color=COLORS[dataset],
                marker=style["marker"],
                edgecolor="white",
                linewidth=0.8,
                s=78,
                zorder=3,
            )
            axis.annotate(
                dataset,
                (metrics["MPIW"], 100.0 * metrics["PICP"]),
                xytext=offsets[variant],
                textcoords="offset points",
                color=COLORS[dataset],
                fontsize=8,
            )

    axis.axhline(90.0, color="#333333", linestyle=":", linewidth=1.2)
    axis.set_xscale("log")
    axis.set_xlabel("Mean prediction-interval width (dataset units, log scale)")
    axis.set_ylabel("Empirical marginal coverage (%)")
    axis.set_title("Coverage-width diagnostics for normalized ensemble conformal prediction")
    axis.set_ylim(89.75, 90.8)

    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=COLORS[dataset],
            label=dataset,
        )
        for dataset in DATASETS
    ]
    variant_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            linestyle="",
            color="#555555",
            label=style["label"],
        )
        for style in VARIANTS.values()
    ]
    axis.legend(
        handles=dataset_handles + variant_handles,
        ncol=3,
        loc="lower left",
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT / "uq5_coverage_width_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def generate_horizon_figure(key: str, output_name: str) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 3.5), sharex=True)
    horizons = np.arange(1, 13)
    for axis, dataset in zip(axes, DATASETS):
        for variant, style in VARIANTS.items():
            values = horizon_values(load_metrics(dataset, variant), key)
            if key == "PICP":
                values = 100.0 * values
            axis.plot(
                horizons,
                values,
                color=COLORS[dataset],
                marker=style["marker"],
                markersize=3.6,
                linewidth=1.45,
                linestyle=style["linestyle"],
                label=style["label"],
            )
        if key == "PICP":
            axis.axhline(90.0, color="#333333", linestyle=":", linewidth=1.0)
        axis.set_title(dataset)
        axis.set_xticks((1, 3, 6, 9, 12))
        axis.set_xlabel("Forecast horizon")

    axes[0].set_ylabel(
        "Empirical marginal coverage (%)"
        if key == "PICP"
        else "Mean prediction-interval width"
    )
    axes[-1].legend(loc="best", fontsize=8)
    figure.suptitle(
        "Per-horizon empirical marginal coverage"
        if key == "PICP"
        else "Per-horizon mean interval width",
        y=1.02,
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(OUTPUT / output_name, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    generate_tradeoff()
    generate_horizon_figure("MPIW", "uq6_pi_horizon_bands.png")
    generate_horizon_figure("PICP", "uq7_pi_coverage_calibration.png")
    print("Generated uq5, uq6, and uq7 from conformal metrics")


if __name__ == "__main__":
    main()
