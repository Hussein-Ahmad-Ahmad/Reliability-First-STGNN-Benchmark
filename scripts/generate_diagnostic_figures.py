"""Render masked stochastic diagnostics and dataset-specific calibration panels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#009E73", "#6B7280"
MODEL_ORDER = ["D2STGNN", "STGCNChebGraphConv", "STID", "STAEformer", "MTGNN", "MegaCRN", "STNorm"]


def read(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def configure():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 9,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white", "legend.frameon": False})


def save(fig, folder, name):
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(folder / name)


def stochastic(folder):
    data = [read(f"results/task2_uncertainty/mc_dropout_generated/{m}/METR-LA/seed43/summary.json") for m in MODEL_ORDER]
    labels = ["STGCN-Cheb" if m == "STGCNChebGraphConv" else m for m in MODEL_ORDER]
    colors = [ORANGE if d["status"] == "inactive" else BLUE for d in data]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.8))
    for axis, key, ylabel in zip(axes, ["variance_mean", "picp_uncalibrated"], ["Mean predictive variance", "Empirical coverage"]):
        axis.bar(range(7), [d[key] for d in data], color=colors)
        axis.set_xticks(range(7), labels, rotation=30, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Valid-target variance")
    axes[1].set_title("(b) Uncalibrated 90% intervals")
    axes[1].axhline(.9, color="black", linestyle=":", linewidth=1)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    save(fig, folder, "mc_dropout_diagnostics.png")


def calibration(folder):
    datasets = ["METR-LA", "PEMS-BAY", "PEMS04"]
    fig, axes = plt.subplots(2, 3, figsize=(7.35, 3.9))
    for col, ds in enumerate(datasets):
        members = None
        for row, key in enumerate(["PICP", "MPIW"]):
            values = []
            for variant in ["fixed", "per_horizon"]:
                d = read(f"results/task2_uncertainty/conformal/{ds}_conformal_{variant}_metrics.json")
                members = d["member_count"]
                values.append(d["evaluation_set"][key])
            axes[row, col].bar([0, 1], values, color=[BLUE, ORANGE])
            axes[row, col].set_xticks([0, 1], ["Pooled", "Per horizon"])
            axes[row, col].grid(axis="y", alpha=.25)
            axes[row, col].set_axisbelow(True)
            if row == 0:
                axes[row, col].set_ylim(0, 1)
                axes[row, col].axhline(.9, color="black", linestyle=":", linewidth=1)
                axes[row, col].set_title(f"{ds} ({members} members)")
            else:
                axes[row, col].set_ylabel("Width (dataset units)")
        axes[0, col].set_ylabel("Coverage")
    fig.tight_layout()
    save(fig, folder, "cross_dataset_conformal.png")


def ablation(folder):
    models = ["D2STGNN", "MegaCRN", "STID", "STAEformer"]
    budgets = [5, 10, 20, 50]
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.5))
    for axis, model in zip(axes, models):
        d = read(f"results/task3_explainability/gnnexplainer/{model}/METR-LA_fidelity_metrics.json")
        axis.bar(np.arange(4) - .18, [d[f"k={k}"]["mean_delta_important"] for k in budgets], .36, color=BLUE, label="Selected")
        axis.bar(np.arange(4) + .18, [d[f"k={k}"]["mean_delta_random"] for k in budgets], .36, color=GRAY, label="Random")
        axis.set_xticks(range(4), budgets)
        axis.set_xlabel("Deleted sensors")
        axis.set_title(model)
        axis.set_ylabel("Absolute forecast change")
        axis.grid(axis="y", alpha=.25)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.035))
    fig.tight_layout(rect=(0, .13, 1, 1))
    save(fig, folder, "selected_random_ablation.png")


def degrees(folder):
    directory = ROOT / "results/task3_explainability/degree_matched_control"
    data = [json.loads(p.read_text()) for p in sorted(directory.glob("*.json"))]
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 3.1))
    for axis, ds in zip(axes, ["METR-LA", "PEMS-BAY", "PEMS04"]):
        rows = [d for d in data if d["dataset"] == ds]
        x = np.arange(len(rows))
        axis.bar(x - .25, [d["important_delta"] for d in rows], .25, color=BLUE, label="Selected")
        for offset, key, color, label in [(0, "uniform_random", ORANGE, "Uniform"), (.25, "degree_matched_random", GREEN, "Degree matched")]:
            axis.bar(x + offset, [d[key]["mean_delta"] for d in rows], .25,
                yerr=[d[key]["std_delta"] for d in rows], capsize=2, color=color, label=label)
        labels = ["STGCN-Cheb" if d["model"] == "STGCNChebGraphConv" else d["model"] + ("*" if not d["degree_matching"]["exact"] else "") for d in rows]
        axis.set_xticks(x, labels, rotation=35, ha="right")
        axis.set_title(ds)
        axis.set_ylabel("Test MAE change")
        axis.grid(axis="y", alpha=.25)
        axis.set_axisbelow(True)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .17, 1, 1))
    save(fig, folder, "degree_matched_fidelity.png")


def ranking_panels(folder):
    labels = ["STGCN-Cheb" if m == "STGCNChebGraphConv" else m for m in MODEL_ORDER]
    values = [read(f"results/task3_explainability/gnnexplainer/{m}/METR-LA_fidelity_metrics.json")["k=10"]["mean_fidelity_ratio"] for m in MODEL_ORDER]
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.95), gridspec_kw={"width_ratios": (1, 1.45)})
    cross_models = ["D2STGNN", "MTGNN", "STID"]
    datasets = ["METR-LA", "PEMS-BAY", "PEMS04"]
    matrix = []
    for ds in datasets:
        row = []
        for model in cross_models:
            relative = f"results/task3_explainability/gnnexplainer/{model}/{ds}_fidelity_metrics.json" if ds == "METR-LA" else f"results/task3_explainability/cross_dataset/{model}/{ds}_fidelity_metrics.json"
            row.append(read(relative)["k=10"]["mean_fidelity_ratio"])
        matrix.append(row)
    matrix = np.asarray(matrix)
    heat = axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=max(2.1, matrix.max()))
    axes[0].set_xticks(range(3), cross_models)
    axes[0].set_yticks(range(3), datasets)
    axes[0].set_title("(a) Archived dataset-specific ratios")
    for row in range(3):
        for col in range(3):
            axes[0].text(col, row, f"{matrix[row,col]:.3f}", ha="center", va="center",
                color="white" if matrix[row,col] > 1.25 else "black", fontsize=8)
    fig.colorbar(heat, ax=axes[0], fraction=.046, pad=.03)
    axis = axes[1]
    axis.bar(range(7), values, color=BLUE)
    axis.axhline(1, color="black", linestyle=":", linewidth=1)
    axis.set_ylabel("Selected/random forecast-change ratio")
    axis.set_xticks(range(7), labels, rotation=30, ha="right")
    axis.set_title("(b) METR-LA, deletion budget 10")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save(fig, folder, "sensor_ranking_fidelity.png")

    payload = read("results/task3_explainability/statistical_tests/ig_vs_gnn_agreement_7models.json")
    records = {d["model"]: d for d in payload["models"]}
    overlaps = [records[m]["overlap_count"] for m in MODEL_ORDER]
    fig, axis = plt.subplots(figsize=(7.35, 2.65))
    axis.bar(range(7), overlaps, color=GREEN)
    axis.set_ylabel("Shared sensors in top 10")
    axis.set_xticks(range(7), labels, rotation=30, ha="right")
    axis.set_ylim(0, 3)
    axis.set_title("Archived sensor-input ranking and Integrated Gradients sets")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save(fig, folder, "cross_method_ranking_agreement.png")

    provenance = read("results/task3_explainability/ranking_provenance.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.85))
    cohort = provenance["cohort_saliency_values"]
    axes[0].bar(range(len(cohort)), [v["stability_jaccard"] for v in cohort.values()], color=GREEN)
    axes[0].set_xticks(range(len(cohort)), list(cohort), rotation=30, ha="right")
    axes[0].set_title("(a) Gradient-saliency cohort overlap")
    noise_models = [m for m in MODEL_ORDER if m not in ("MegaCRN", "STID")]
    values = []
    for model in noise_models:
        data = read(f"results/task3_explainability/gnnexplainer/{model}/METR-LA_stability_metrics.json")
        values.append(data["noise_0.1"]["mean_jaccard_across_samples"])
    axes[1].bar(range(len(values)), values, color=GRAY)
    axes[1].set_xticks(range(len(values)), ["STGCN-Cheb" if m == "STGCNChebGraphConv" else m for m in noise_models], rotation=30, ha="right")
    axes[1].set_title("(b) Archived noise-response overlap")
    for axis in axes:
        axis.set_ylabel("Jaccard overlap")
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    save(fig, folder, "ranking_perturbation_stability.png")


def within_pipeline_tradeoff(folder):
    fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.7))
    for axis, ds in zip(axes, ["METR-LA", "PEMS-BAY", "PEMS04"]):
        for variant, color, label in [("fixed", BLUE, "Pooled"), ("per_horizon", ORANGE, "Per horizon")]:
            data = read(f"results/task2_uncertainty/conformal/{ds}_conformal_{variant}_metrics.json")["evaluation_set"]
            axis.scatter(data["MPIW"], data["PICP"], color=color, s=36, label=label)
        axis.set_title(ds)
        axis.set_xlabel("Width (dataset units)")
        axis.set_ylabel("Coverage")
        axis.set_ylim(.88, .92)
        axis.axhline(.9, color="black", linestyle=":", linewidth=1)
        axis.grid(alpha=.2)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(.5, -.025))
    fig.tight_layout(rect=(0, .13, 1, 1))
    save(fig, folder, "coverage_width_tradeoff.png")


def horizon_profiles(folder):
    datasets = ['METR-LA', 'PEMS-BAY', 'PEMS04']
    horizons = np.arange(1, 13)
    variants = [('fixed', ORANGE, 'Pooled'), ('per_horizon', BLUE, 'Per horizon')]
    for key, filenames in [('PICP', ['horizon_coverage.png', 'horizon_marginal_coverage.png']),
                           ('MPIW', ['horizon_interval_width.png'])]:
        fig, axes = plt.subplots(1, 3, figsize=(7.35, 2.75))
        for axis, dataset in zip(axes, datasets):
            for variant, color, label in variants:
                data = read(f'results/task2_uncertainty/conformal/{dataset}_conformal_{variant}_metrics.json')
                values = [data['evaluation_set']['per_horizon'][f'horizon_{h}'][key] for h in horizons]
                axis.plot(horizons, values, color=color, label=label, marker='o', markersize=2.5, linewidth=1.2)
            axis.set_title(dataset)
            axis.set_xlabel('Forecast horizon')
            axis.set_xticks([1, 3, 6, 9, 12])
            axis.set_ylabel('Coverage' if key == 'PICP' else 'Width (dataset units)')
            axis.grid(alpha=.2)
            if key == 'PICP':
                axis.axhline(.9, color='black', linestyle=':', linewidth=1, label='90% nominal')
                axis.set_ylim(.80, .98)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3)
        fig.tight_layout(rect=(0, .12, 1, 1))
        for filename in filenames:
            folder.mkdir(parents=True, exist_ok=True)
            fig.savefig(folder / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)

    data = read('results/task2_uncertainty/conformal/METR-LA_conformal_per_horizon_metrics.json')
    fig, axes = plt.subplots(1, 2, figsize=(7.35, 2.65))
    for axis, key, color, ylabel in zip(axes, ['PICP', 'MPIW'], [BLUE, GREEN], ['Coverage', 'Width (target units)']):
        values = [data['evaluation_set']['per_horizon'][f'horizon_{h}'][key] for h in horizons]
        axis.bar(horizons, values, color=color, width=.72)
        axis.set_xlabel('Forecast horizon')
        axis.set_xticks([1, 3, 6, 9, 12])
        axis.set_ylabel(ylabel)
        axis.grid(axis='y', alpha=.2)
        if key == 'PICP':
            axis.axhline(.9, color='black', linestyle=':', linewidth=1)
            axis.set_ylim(.85, .95)
    fig.tight_layout()
    save(fig, folder, 'metr_la_horizon_conformal.png')


def forecast_window(folder):
    data = read('results/release/conformal_forecast_example.json')
    horizons = np.arange(1, len(data['target']) + 1)
    fig, axis = plt.subplots(figsize=(7.35, 2.75))
    mean, std = np.asarray(data['mean']), np.asarray(data['std'])
    for variant, color, label in [('fixed', ORANGE, 'Pooled interval'), ('per_horizon', BLUE, 'Per-horizon interval')]:
        threshold = np.asarray(data['thresholds'][variant])
        half_width = threshold * (std + data['epsilon'])
        axis.fill_between(horizons, mean - half_width, mean + half_width, color=color, alpha=.20, label=label)
    axis.plot(horizons, data['target'], color='black', marker='o', markersize=3, linewidth=1.2, label='Observed')
    axis.plot(horizons, mean, color=BLUE, marker='s', markersize=3, linewidth=1.2, label='Ensemble mean')
    axis.set_xlabel('Forecast horizon')
    axis.set_ylabel('Traffic speed')
    axis.set_xticks([1, 3, 6, 9, 12])
    axis.grid(alpha=.2)
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4)
    fig.tight_layout(rect=(0, .12, 1, 1))
    save(fig, folder, 'forecast_interval_example.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures/results")
    parser.add_argument("--include-degrees", action="store_true")
    args = parser.parse_args()
    configure()
    stochastic(args.output_dir)
    calibration(args.output_dir)
    ablation(args.output_dir)
    ranking_panels(args.output_dir)
    within_pipeline_tradeoff(args.output_dir)
    horizon_profiles(args.output_dir)
    forecast_window(args.output_dir)
    if args.include_degrees:
        degrees(args.output_dir)


if __name__ == "__main__":
    main()
