"""
Consolidate Jaccard Stability Results (Gap 2 Fix)
==================================================
MegaCRN and STID Jaccard values for METR-LA exist in xai_results_final_source.json
(from experiments/basicts/xai_results_final.json).

This script extracts those values and writes formal stability_metrics.json files
to match the format used by the other 5 models.

Output files:
  results/task3_explainability/jaccard_stability/MegaCRN_METR-LA_stability_metrics.json
  results/task3_explainability/jaccard_stability/STID_METR-LA_stability_metrics.json

Run:
    python scripts/consolidate_jaccard_stability.py
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
JACCARD_DIR = PROJECT_ROOT / "results" / "task3_explainability" / "jaccard_stability"
SOURCE_FILE = JACCARD_DIR / "xai_results_final_source.json"


def main():
    with open(SOURCE_FILE) as f:
        source = json.load(f)

    # Models to extract: MegaCRN and STID
    for model in ["MegaCRN", "STID"]:
        if model not in source:
            print(f"  ✗ {model} not in source file")
            continue

        entry = source[model]
        jaccard = entry.get("stability_jaccard")
        validity = entry.get("validity_mae_increase")

        # Write formal stability_metrics.json matching format of other models
        output = {
            "model": model,
            "dataset": "METR-LA",
            "method": "GNNExplainer",
            "jaccard_stability": jaccard,
            "top_k": 10,
            "note": (
                "Jaccard stability extracted from xai_results_final.json "
                "(computed by test_xai_metrics.py in BasicTS pipeline). "
                "Formal per-run breakdown not available for this model."
            ),
            "source": "experiments/basicts/xai_results_final.json",
        }

        # Also include validity as context
        if validity is not None:
            output["validity_mae_increase"] = validity

        out_path = JACCARD_DIR / f"{model}_METR-LA_stability_metrics.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  OK {model}: stability_jaccard = {jaccard:.4f} -> {out_path.name}")

    # Summary
    print("\nJaccard METR-LA coverage:")
    models = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
    for model in models:
        path = JACCARD_DIR / f"{model}_METR-LA_stability_metrics.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            j = data.get("jaccard_stability") or data.get("mean_jaccard", "N/A")
            print(f"  OK {model}: {j}")
        else:
            print(f"  XX {model}: MISSING")


if __name__ == "__main__":
    main()
