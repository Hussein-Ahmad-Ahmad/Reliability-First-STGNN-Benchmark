"""
Generate a concrete XAI method-dependence case study.

The default case compares Integrated Gradients and GNNExplainer for MTGNN on
METR-LA, maps internal node indices to real sensor IDs, and adds graph context
from the dataset adjacency matrix.

Default run:
    python scripts/generate_xai_case_study.py

Output:
    results/task3_explainability/case_studies/MTGNN_METR-LA_ig_vs_gnn_case_study.json
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
XAI_ROOT = PROJECT_ROOT / "results" / "task3_explainability"
OUT_ROOT = XAI_ROOT / "case_studies"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_graph(dataset: str) -> tuple[list[str], np.ndarray]:
    path = PROJECT_ROOT / "datasets" / dataset / "adj_mx.pkl"
    with path.open("rb") as f:
        obj = pickle.load(f, encoding="latin1")

    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        sensor_ids = [str(x) for x in obj[0]]
        adjacency = np.asarray(obj[2], dtype=np.float64)
    else:
        adjacency = np.asarray(obj, dtype=np.float64)
        sensor_ids = [str(i) for i in range(adjacency.shape[0])]

    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(f"Expected square adjacency for {dataset}, got {adjacency.shape}")
    if len(sensor_ids) != adjacency.shape[0]:
        raise ValueError(f"Sensor-id count does not match adjacency shape for {dataset}")

    return sensor_ids, adjacency


def top_neighbors(node: int, adjacency: np.ndarray, sensor_ids: list[str], k: int) -> list[dict]:
    weights = np.asarray(adjacency[node], dtype=np.float64).copy()
    weights[node] = 0.0
    candidate_idx = np.flatnonzero(weights > 0)
    if candidate_idx.size == 0:
        return []

    order = candidate_idx[np.argsort(weights[candidate_idx])[::-1]][:k]
    return [
        {
            "node_index": int(idx),
            "sensor_id": sensor_ids[int(idx)],
            "edge_weight": float(weights[int(idx)]),
        }
        for idx in order
    ]


def sensor_context(
    node_indices: list[int],
    sensor_ids: list[str],
    adjacency: np.ndarray,
    highlighted_other_method: set[int],
    neighbor_k: int,
) -> list[dict]:
    undirected = (adjacency > 0) | (adjacency.T > 0)
    weighted_degree = adjacency.sum(axis=1) + adjacency.sum(axis=0)

    context = []
    for rank, node in enumerate(node_indices, start=1):
        node = int(node)
        neighbors = top_neighbors(node, adjacency, sensor_ids, neighbor_k)
        adjacent_other = sorted(
            int(idx)
            for idx in highlighted_other_method
            if idx != node and idx < undirected.shape[0] and undirected[node, idx]
        )
        context.append(
            {
                "rank": rank,
                "node_index": node,
                "sensor_id": sensor_ids[node],
                "binary_degree": int(undirected[node].sum()),
                "weighted_degree": float(weighted_degree[node]),
                "top_weighted_neighbors": neighbors,
                "adjacent_to_other_method_top10": [
                    {"node_index": idx, "sensor_id": sensor_ids[idx]} for idx in adjacent_other
                ],
            }
        )
    return context


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one concrete IG-vs-GNNExplainer XAI case study.")
    parser.add_argument("--model", default="MTGNN")
    parser.add_argument("--dataset", default="METR-LA")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--neighbor-k", type=int, default=5)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    ig_path = XAI_ROOT / "integrated_gradients" / f"{args.model}_{args.dataset}_ig_results.json"
    gnn_path = XAI_ROOT / "gnnexplainer" / args.model / f"{args.dataset}_summary.json"

    ig = load_json(ig_path)
    gnn = load_json(gnn_path)
    sensor_ids, adjacency = load_graph(args.dataset)

    ig_top = [int(x) for x in ig["top_10_sensors"][: args.k]]
    gnn_top = [int(x) for x in gnn["top_10_sensors"][: args.k]]
    ig_set = set(ig_top)
    gnn_set = set(gnn_top)
    overlap = sorted(ig_set & gnn_set)

    output = {
        "model": args.model,
        "dataset": args.dataset,
        "case": "Integrated Gradients vs GNNExplainer top-k sensor comparison",
        "k": args.k,
        "source_files": {
            "integrated_gradients": str(ig_path.relative_to(PROJECT_ROOT)),
            "gnnexplainer": str(gnn_path.relative_to(PROJECT_ROOT)),
            "adjacency": str((PROJECT_ROOT / "datasets" / args.dataset / "adj_mx.pkl").relative_to(PROJECT_ROOT)),
        },
        "interpretation_guardrail": (
            "This case study treats explanations as diagnostic views of model behavior. "
            "Disagreement between methods should not be read as causal evidence about sensors."
        ),
        "summary": {
            "ig_top_indices": ig_top,
            "ig_top_sensor_ids": [sensor_ids[i] for i in ig_top],
            "gnn_top_indices": gnn_top,
            "gnn_top_sensor_ids": [sensor_ids[i] for i in gnn_top],
            "overlap_count": len(overlap),
            "overlap_indices": overlap,
            "overlap_sensor_ids": [sensor_ids[i] for i in overlap],
            "jaccard": jaccard(ig_set, gnn_set),
            "ig_mean_importance": ig.get("sensor_importance_mean"),
            "ig_importance_std": ig.get("sensor_importance_std"),
            "gnn_mean_node_importance_stats": gnn.get("mean_node_importance_stats"),
        },
        "integrated_gradients_sensor_context": sensor_context(
            ig_top, sensor_ids, adjacency, highlighted_other_method=gnn_set, neighbor_k=args.neighbor_k
        ),
        "gnnexplainer_sensor_context": sensor_context(
            gnn_top, sensor_ids, adjacency, highlighted_other_method=ig_set, neighbor_k=args.neighbor_k
        ),
        "paper_ready_takeaway": (
            f"For {args.model} on {args.dataset}, IG and GNNExplainer select "
            f"{len(overlap)}/{args.k} common top sensors "
            f"(Jaccard={jaccard(ig_set, gnn_set):.3f}). The low overlap, together with graph-neighborhood "
            "context, supports using XAI as a diagnostic and method-dependent probe rather than a causal claim."
        ),
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    out_path = args.out_root / f"{args.model}_{args.dataset}_ig_vs_gnn_case_study.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved case study: {out_path}")
    print(output["paper_ready_takeaway"])


if __name__ == "__main__":
    main()
