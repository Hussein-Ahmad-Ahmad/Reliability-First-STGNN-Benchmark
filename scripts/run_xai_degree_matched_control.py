"""Degree-matched random control for GNNExplainer sensor-deletion fidelity.

Reviewer concern C4: the existing fidelity artifacts
(results/task3_explainability/gnnexplainer/<model>/<dataset>_fidelity_metrics.json)
compare deleting the top-K "important" sensors against deleting K
*uniformly* random sensors. A uniform random control can differ from the
important set simply because high-degree (well-connected) sensors are
easier to predict from neighbors, independent of whether GNNExplainer
picked them for a meaningful reason. A degree-matched control samples
random sensors whose graph degree distribution matches the important set,
isolating the "is this explanation better than a degree-equivalent guess"
question.

The original GNNExplainerWrapper class referenced by pipelines/task3_run.py
is no longer present in this repository (src/explainability/spatial_saliency.py
does not define it), so this script does not attempt to reproduce its exact
deletion-fidelity numbers. Instead it defines one explicit, self-contained
deletion protocol and applies it identically to all three sensor sets
(important / uniform-random / degree-matched-random), so the three numbers
are directly comparable even though they are not bit-identical to the
archived fidelity_metrics.json values.

Protocol: after BasicTS scaler normalization, set the selected sensors'
full 12-step input history to zero (same masking convention as the sensor
dropout robustness pipeline). Run the retained seed-43 checkpoint over the
full test set and report the increase in overall test MAE relative to an
unmasked clean pass.

Usage:
    python scripts/run_xai_degree_matched_control.py ^
        --model D2STGNN --dataset METR-LA --seed 43 --k 10 ^
        --n-random-draws 30

Output:
    results/task3_explainability/degree_matched_control/<model>_<dataset>_seed<seed>_degree_matched_control.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = ROOT / "framework"
for path in (ROOT, FRAMEWORK):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_mc_dropout_module():
    spec = importlib.util.spec_from_file_location(
        "run_mc_dropout_inference", Path(__file__).resolve().parent / "run_mc_dropout_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mc = _load_mc_dropout_module()


def load_adjacency(dataset: str) -> np.ndarray:
    path = ROOT / "datasets" / dataset / "adj_mx.pkl"
    with path.open("rb") as handle:
        try:
            obj = pickle.load(handle, encoding="latin1")
        except UnicodeDecodeError:
            handle.seek(0)
            obj = pickle.load(handle)
    if isinstance(obj, (list, tuple)) and len(obj) == 3:
        matrix = obj[2]
    else:
        matrix = obj
    return np.asarray(matrix, dtype=np.float64)


def compute_degree(adj: np.ndarray) -> np.ndarray:
    binary = (adj > 0).astype(np.int64)
    np.fill_diagonal(binary, 0)
    return binary.sum(axis=1)


def load_important_sensors(model: str, dataset: str, k: int) -> list[int]:
    summary_path = (
        ROOT / "results" / "task3_explainability" / "gnnexplainer" / model / f"{dataset}_summary.json"
    )
    if not summary_path.is_file():
        raise FileNotFoundError(
            f"No existing GNNExplainer summary at {summary_path}. "
            "Pass --important-sensors explicitly."
        )
    with summary_path.open(encoding="utf-8") as handle:
        summary = json.load(handle)
    key = f"top_{k}_sensors"
    if key not in summary:
        raise KeyError(f"{summary_path} has no '{key}' field; pass --important-sensors explicitly.")
    return list(summary[key])[:k]


def sample_degree_matched(
    degree: np.ndarray,
    important: list[int],
    rng: np.random.Generator,
) -> list[int]:
    """Sample one random sensor per important sensor with the closest-matching
    degree, breaking ties randomly so repeated draws are not identical."""
    n = degree.shape[0]
    excluded = set(important)
    chosen: list[int] = []
    for target_index in important:
        target_degree = int(degree[target_index])
        distances = np.abs(degree.astype(np.int64) - target_degree)
        best_distance = None
        pool = []
        for candidate in range(n):
            if candidate in excluded:
                continue
            d = distances[candidate]
            if best_distance is None or d < best_distance:
                best_distance = d
                pool = [candidate]
            elif d == best_distance:
                pool.append(candidate)
        choice = int(rng.choice(pool))
        chosen.append(choice)
        excluded.add(choice)
    return chosen


def sample_uniform_random(
    degree: np.ndarray,
    important: list[int],
    k: int,
    rng: np.random.Generator,
) -> list[int]:
    n = degree.shape[0]
    pool = [i for i in range(n) if i not in set(important)]
    return [int(x) for x in rng.choice(pool, size=k, replace=False)]


def run_masked_pass(
    model_name: str,
    dataset: str,
    seed: int,
    dropped: list[int],
    batch_size: int,
    device: torch.device,
) -> dict:
    cfg = mc.load_config(model_name, dataset, seed)
    checkpoint_path = ROOT / "checkpoints" / model_name / f"{dataset}_seed{seed}" / "best_model.pt"
    dataset_obj = cfg.DATASET.TYPE(mode="test", **cfg.DATASET.PARAM)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=False, num_workers=0)
    scaler = cfg.SCALER.TYPE(**cfg.SCALER.PARAM)
    model = mc.load_model(cfg, checkpoint_path, device)
    model.eval()

    error_sum = 0.0
    valid_count = 0
    with torch.no_grad():
        for batch in loader:
            history = batch["inputs"].to(device=device, dtype=torch.float32)
            future = batch["target"].to(device=device, dtype=torch.float32)
            history = scaler.transform(history.clone())
            future_scaled = scaler.transform(future.clone())
            if dropped:
                history[:, :, dropped, :] = 0.0
            history = history[..., cfg.MODEL.FORWARD_FEATURES]
            decoder_future = future_scaled[..., cfg.MODEL.FORWARD_FEATURES].clone()
            decoder_future[..., 0] = torch.empty_like(decoder_future[..., 0])

            output = model(
                history_data=history,
                future_data=decoder_future,
                batch_seen=None,
                epoch=None,
                train=False,
            )
            prediction = output["prediction"] if isinstance(output, dict) else output
            prediction = scaler.inverse_transform(prediction)
            target = future[..., cfg.MODEL.TARGET_FEATURES]

            valid = torch.isfinite(target) & target.ne(0)
            error_sum += torch.abs(prediction - target)[valid].double().sum().item()
            valid_count += int(valid.sum().item())

    if valid_count == 0:
        raise RuntimeError("No valid target values found")
    return {"mae": error_sum / valid_count, "valid_target_count": valid_count, "dropped_sensors": dropped}


def main() -> None:
    parser = argparse.ArgumentParser(description="Degree-matched random control for XAI deletion fidelity")
    parser.add_argument("--model", default="D2STGNN")
    parser.add_argument("--dataset", default="METR-LA")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--important-sensors", type=int, nargs="+", default=None)
    parser.add_argument("--n-random-draws", type=int, default=30)
    parser.add_argument("--mask-seed", type=int, default=20260920)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "task3_explainability" / "degree_matched_control")
    args = parser.parse_args()

    device = torch.device(args.device)
    adj = load_adjacency(args.dataset)
    degree = compute_degree(adj)

    important = args.important_sensors or load_important_sensors(args.model, args.dataset, args.k)
    important = important[: args.k]
    important_degrees = [int(degree[i]) for i in important]

    print(f"Baseline (unmasked) pass: {args.model}/{args.dataset}/seed{args.seed}")
    baseline = run_masked_pass(args.model, args.dataset, args.seed, [], args.batch_size, device)

    print(f"Important-set pass (k={args.k}): sensors={important}")
    important_result = run_masked_pass(args.model, args.dataset, args.seed, important, args.batch_size, device)

    rng = np.random.default_rng(args.mask_seed)
    uniform_deltas = []
    uniform_draws = []
    for draw in range(args.n_random_draws):
        sensors = sample_uniform_random(degree, important, args.k, rng)
        result = run_masked_pass(args.model, args.dataset, args.seed, sensors, args.batch_size, device)
        delta = result["mae"] - baseline["mae"]
        uniform_deltas.append(delta)
        uniform_draws.append({"sensors": sensors, "mae": result["mae"], "delta": delta})
        print(f"  uniform-random draw {draw + 1}/{args.n_random_draws}: delta={delta:.4f}")

    degree_deltas = []
    degree_draws = []
    for draw in range(args.n_random_draws):
        sensors = sample_degree_matched(degree, important, rng)
        result = run_masked_pass(args.model, args.dataset, args.seed, sensors, args.batch_size, device)
        delta = result["mae"] - baseline["mae"]
        degree_deltas.append(delta)
        degree_draws.append({
            "sensors": sensors,
            "degrees": [int(degree[i]) for i in sensors],
            "mae": result["mae"],
            "delta": delta,
        })
        print(f"  degree-matched draw {draw + 1}/{args.n_random_draws}: delta={delta:.4f}")

    important_delta = important_result["mae"] - baseline["mae"]
    output = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "k": args.k,
        "protocol": (
            "Post-scaler zero-masking of the selected sensors' full input history; "
            "delta = masked test MAE minus unmasked clean-pass test MAE. "
            "This is a self-contained protocol defined for this comparison; it does "
            "not reproduce the archived GNNExplainerWrapper fidelity numbers, which "
            "used a different (now-missing) implementation."
        ),
        "baseline_mae": baseline["mae"],
        "important_sensors": important,
        "important_sensor_degrees": important_degrees,
        "important_delta": important_delta,
        "uniform_random": {
            "n_draws": args.n_random_draws,
            "mean_delta": float(np.mean(uniform_deltas)),
            "std_delta": float(np.std(uniform_deltas, ddof=1)) if len(uniform_deltas) > 1 else 0.0,
            "draws": uniform_draws,
        },
        "degree_matched_random": {
            "n_draws": args.n_random_draws,
            "mean_delta": float(np.mean(degree_deltas)),
            "std_delta": float(np.std(degree_deltas, ddof=1)) if len(degree_deltas) > 1 else 0.0,
            "draws": degree_draws,
        },
        "mask_seed": args.mask_seed,
        "interpretation": (
            "important_delta should exceed both control means for the explanation "
            "to be credited with finding sensors that matter beyond what their "
            "connectivity alone would predict."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"{args.model}_{args.dataset}_seed{args.seed}_degree_matched_control.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(out_path)
    print(f"important_delta={important_delta:.4f}  uniform_mean={np.mean(uniform_deltas):.4f}  degree_matched_mean={np.mean(degree_deltas):.4f}")


if __name__ == "__main__":
    main()
