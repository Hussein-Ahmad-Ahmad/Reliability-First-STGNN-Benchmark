"""
Task 3 Pipeline: Explainability (XAI)
======================================
Evaluates three XAI methods on trained STGNN models:
  1. GNNExplainer — spatial saliency with deletion fidelity (k=5,10,20,50)
  2. Integrated Gradients — attribution maps (50 steps, zero baseline)
  3. Attention Extraction — temporal attention weight analysis

And computes:
  4. Jaccard Stability — top-K explanation consistency under noise (K=10, 100 runs)
  5. Cross-method Agreement — IG vs GNNExplainer vs Attention triangulation

Usage:
    # Run all XAI methods
    python pipelines/task3_run.py --method all

    # Run GNNExplainer for a specific model
    python pipelines/task3_run.py --method gnnexplainer --model D2STGNN --dataset METR-LA

    # Run Integrated Gradients
    python pipelines/task3_run.py --method ig

    # Compute Jaccard stability
    python pipelines/task3_run.py --method jaccard

    # Compute cross-method agreement
    python pipelines/task3_run.py --method agreement
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.explainability.spatial_saliency import GNNExplainerWrapper
from src.explainability.feature_importance import IntegratedGradients
from src.explainability.temporal_attention import TemporalAttentionAnalyzer

MODELS = ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCNChebGraphConv", "STID", "STAEformer"]
DATASETS = ["METR-LA", "PEMS-BAY", "PEMS04"]

# Models with attention mechanisms
ATTENTION_MODELS = ["D2STGNN", "MegaCRN", "STAEformer", "STID"]


def run_gnnexplainer(model: str, dataset: str):
    """Run GNNExplainer and compute fidelity/stability metrics."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / model / f"{dataset}_seed43" / "best_model.pt"
    out_dir = PROJECT_ROOT / "results" / "task3_explainability" / "gnnexplainer" / model
    out_dir.mkdir(parents=True, exist_ok=True)

    explainer = GNNExplainerWrapper(
        model_name=model,
        checkpoint_path=str(ckpt_path),
        dataset=dataset,
        output_dir=str(out_dir),
    )
    explainer.explain_and_evaluate(fidelity_k=[5, 10, 20, 50])
    print(f"  ✓ GNNExplainer: {model}/{dataset}")


def run_ig(model: str, dataset: str = "METR-LA"):
    """Run Integrated Gradients (50 steps, zero baseline)."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / model / f"{dataset}_seed43" / "best_model.pt"
    out_dir = PROJECT_ROOT / "results" / "task3_explainability" / "integrated_gradients"
    out_dir.mkdir(parents=True, exist_ok=True)

    ig = IntegratedGradients(
        model_name=model,
        checkpoint_path=str(ckpt_path),
        dataset=dataset,
        n_steps=50,
        output_dir=str(out_dir),
    )
    ig.compute_attributions()
    print(f"  ✓ Integrated Gradients: {model}/{dataset}")


def run_attention(model: str, dataset: str = "METR-LA"):
    """Extract and analyze temporal attention weights."""
    ckpt_path = PROJECT_ROOT / "checkpoints" / model / f"{dataset}_seed43" / "best_model.pt"
    out_dir = PROJECT_ROOT / "results" / "task3_explainability" / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = TemporalAttentionAnalyzer(
        model_name=model,
        checkpoint_path=str(ckpt_path),
        dataset=dataset,
        output_dir=str(out_dir),
    )
    analyzer.extract_and_analyze()
    print(f"  ✓ Attention: {model}/{dataset}")


def run_jaccard(model: str, dataset: str = "METR-LA", k: int = 10, n_runs: int = 100):
    """Compute Jaccard stability of top-K explanations under noise."""
    explainer_output = (
        PROJECT_ROOT / "results" / "task3_explainability" / "gnnexplainer" / model / f"{dataset}_explanations.pkl"
    )
    out_dir = PROJECT_ROOT / "results" / "task3_explainability" / "jaccard_stability"
    out_dir.mkdir(parents=True, exist_ok=True)

    from src.explainability.spatial_saliency import compute_jaccard_stability
    jaccard = compute_jaccard_stability(
        explanations_path=str(explainer_output),
        k=k,
        n_runs=n_runs,
        output_path=str(out_dir / f"{model}_{dataset}_stability_metrics.json"),
    )
    print(f"  ✓ Jaccard (K={k}, runs={n_runs}): {model}/{dataset} = {jaccard:.4f}")


def run_agreement():
    """Compute cross-method agreement (IG vs GNN vs Attention)."""
    script = PROJECT_ROOT / "scripts" / "compute_method_agreement.py"
    import subprocess
    subprocess.run([sys.executable, str(script)])
    print("  ✓ Cross-method agreement computed")


def main():
    parser = argparse.ArgumentParser(description="Task 3: XAI Pipeline")
    parser.add_argument("--method", choices=["gnnexplainer", "ig", "attention", "jaccard", "agreement", "all"], default="all")
    parser.add_argument("--model", default="all")
    parser.add_argument("--dataset", default="all")
    args = parser.parse_args()

    models = MODELS if args.model == "all" else [args.model]
    datasets = DATASETS if args.dataset == "all" else [args.dataset]

    if args.method in ("gnnexplainer", "all"):
        print("=== GNNExplainer ===")
        for model in models:
            for dataset in datasets:
                try:
                    run_gnnexplainer(model, dataset)
                except Exception as e:
                    print(f"  ✗ {model}/{dataset}: {e}")

    if args.method in ("ig", "all"):
        print("=== Integrated Gradients (METR-LA only) ===")
        for model in models:
            try:
                run_ig(model, dataset="METR-LA")
            except Exception as e:
                print(f"  ✗ {model}: {e}")

    if args.method in ("attention", "all"):
        print("=== Attention Extraction ===")
        for model in [m for m in models if m in ATTENTION_MODELS]:
            try:
                run_attention(model, dataset="METR-LA")
            except Exception as e:
                print(f"  ✗ {model}: {e}")

    if args.method in ("jaccard", "all"):
        print("=== Jaccard Stability ===")
        for model in models:
            try:
                run_jaccard(model, dataset="METR-LA")
            except Exception as e:
                print(f"  ✗ {model}: {e}")

    if args.method in ("agreement", "all"):
        print("=== Cross-Method Agreement ===")
        try:
            run_agreement()
        except Exception as e:
            print(f"  ✗ Agreement: {e}")


if __name__ == "__main__":
    main()
