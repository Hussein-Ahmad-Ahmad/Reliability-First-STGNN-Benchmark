"""
Generate a compact cross-dataset XAI transfer summary from stored artifacts.

This is intentionally lighter than the METR-LA IG-vs-GNNExplainer case study:
PEMS-BAY and PEMS04 currently have stored GNNExplainer transfer checks for a
small model subset, not full Integrated Gradients outputs.

Default run:
    python scripts/generate_xai_transfer_summary.py

Outputs:
    results/task3_explainability/case_studies/xai_cross_dataset_transfer_summary.json
    results/task3_explainability/case_studies/xai_cross_dataset_transfer_summary.md
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
XAI_ROOT = PROJECT_ROOT / "results" / "task3_explainability"
SOURCE_CSV = XAI_ROOT / "xai_cross_dataset_summary.csv"
OUT_ROOT = XAI_ROOT / "case_studies"


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def main() -> None:
    rows = []
    with SOURCE_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["dataset"] == "METR-LA":
                continue
            rows.append(
                {
                    "model": row["model"],
                    "dataset": row["dataset"],
                    "n_samples": int(row["n_samples"]),
                    "top3_sensors": [int(x.strip()) for x in row["top3_sensors"].split(",")],
                    "fidelity_ratio_k10": as_float(row["fidelity_ratio_k10"]),
                    "fidelity_ratio_k20": as_float(row["fidelity_ratio_k20"]),
                    "delta_important_k20": as_float(row["delta_important_k20"]),
                    "delta_random_k20": as_float(row["delta_random_k20"]),
                    "jaccard_noise_0.1": as_float(row["jaccard_noise_0.1"]),
                    "jaccard_noise_0.2": as_float(row["jaccard_noise_0.2"]),
                }
            )

    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})
    output = {
        "scope": "stored cross-dataset GNNExplainer transfer check",
        "source": str(SOURCE_CSV.relative_to(PROJECT_ROOT)),
        "datasets": datasets,
        "models": models,
        "n_rows": len(rows),
        "interpretation_guardrail": (
            "These PEMS-BAY/PEMS04 results are lightweight transfer diagnostics from stored "
            "GNNExplainer artifacts. They are not full cross-dataset causal or IG-vs-GNN validation."
        ),
        "rows": rows,
    }

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = OUT_ROOT / "xai_cross_dataset_transfer_summary.json"
    md_path = OUT_ROOT / "xai_cross_dataset_transfer_summary.md"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    lines = [
        "# XAI Cross-Dataset Transfer Summary",
        "",
        "Stored GNNExplainer transfer diagnostics are available for PEMS-BAY and PEMS04.",
        "These are used only as lightweight support for method-dependent diagnostic behavior; no causal or universal XAI generalization is claimed.",
        "",
        "| Dataset | Model | Top-3 Sensors | Fidelity k10 | Fidelity k20 | Stability Jaccard 0.1 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {model} | {top3} | {k10:.3g} | {k20:.3g} | {jac:.3g} |".format(
                dataset=row["dataset"],
                model=row["model"],
                top3=", ".join(str(x) for x in row["top3_sensors"]),
                k10=row["fidelity_ratio_k10"],
                k20=row["fidelity_ratio_k20"],
                jac=row["jaccard_noise_0.1"],
            )
        )
    lines.extend(
        [
            "",
            "Takeaway: the non-METR-LA datasets already have small stored GNNExplainer checks for D2STGNN, MTGNN, and STID.",
            "The detailed IG-vs-GNNExplainer case study remains METR-LA only because stored IG outputs are METR-LA-focused.",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Saved JSON: {json_path}")
    print(f"Saved Markdown: {md_path}")


if __name__ == "__main__":
    main()
