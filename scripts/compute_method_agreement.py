"""
Cross-Method Agreement: IG vs GNNExplainer vs Attention
=========================================================
Computes pairwise top-K sensor overlap (Jaccard) across the three XAI methods
for D2STGNN on METR-LA (primary triangulation model).

Pre-computed triangulation results are already in:
  results/task3_explainability/attention/batch7_attention_summary.json

This script regenerates them if needed.

Output:
  results/task3_explainability/statistical_tests/cross_method_agreement.json

Run:
    python scripts/compute_method_agreement.py
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
XAI_DIR = PROJECT_ROOT / "results" / "task3_explainability"
OUTPUT_PATH = XAI_DIR / "statistical_tests" / "cross_method_agreement.json"


def jaccard(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


def main():
    # Load pre-computed triangulation from batch7_attention_summary
    summary_path = XAI_DIR / "attention" / "batch7_attention_summary.json"

    if summary_path.exists():
        with open(summary_path) as f:
            batch7 = json.load(f)

        if "d2_triangulation" in batch7:
            print("  Using pre-computed triangulation from batch7_attention_summary.json")
            tri = batch7["d2_triangulation"]

            output = {
                "model": "D2STGNN",
                "dataset": "METR-LA",
                "k": 10,
                "triangulation": tri,
                "source": "batch7_attention_summary.json",
                "note": (
                    "Three-way top-10 sensor overlap between IG, GNNExplainer, and Attention. "
                    "Low pairwise overlap is expected: different methods capture different aspects "
                    "of model behavior (gradient attribution vs. structural explanation vs. temporal weights)."
                ),
            }

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(OUTPUT_PATH, "w") as f:
                json.dump(output, f, indent=2)
            print(f"  ✓ Agreement saved to {OUTPUT_PATH}")

            # Print summary
            for key, val in tri.items():
                print(f"    {key}: {val}")
            return

    # Fallback: compute from raw results
    print("  batch7_attention_summary.json not found, computing from raw results...")

    k = 10
    results = {}

    # Load IG top sensors
    ig_path = XAI_DIR / "integrated_gradients" / "D2STGNN_METR-LA_ig_results.json"
    gnn_path = XAI_DIR / "gnnexplainer" / "D2STGNN" / "METR-LA_summary.json"
    att_path = XAI_DIR / "attention" / "D2STGNN_attention_results.json"

    methods = {}
    for name, path in [("ig", ig_path), ("gnnexplainer", gnn_path), ("attention", att_path)]:
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            top_sensors = data.get(f"top_{k}_sensors") or data.get("top_sensors", [])[:k]
            methods[name] = set(top_sensors)
            print(f"  ✓ {name}: {len(methods[name])} top sensors")

    if len(methods) >= 2:
        for m1, m2 in combinations(methods.keys(), 2):
            j = jaccard(methods[m1], methods[m2])
            results[f"{m1}_vs_{m2}_jaccard"] = round(j, 4)
            print(f"  {m1} vs {m2}: Jaccard={j:.4f}")

        output = {
            "model": "D2STGNN",
            "dataset": "METR-LA",
            "k": k,
            "pairwise_jaccard": results,
        }

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  ✓ Agreement saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    from itertools import combinations
    main()
