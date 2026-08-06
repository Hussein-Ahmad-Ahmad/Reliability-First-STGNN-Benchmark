"""Run the target-disjoint Chickenpox Hungary protocol illustration.

The experiment reuses the seven traffic-benchmark model classes with compact,
dataset-specific dimensions. Model selection, interval calibration, and final
evaluation use distinct chronological target periods.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "framework"))

from models.D2STGNN.arch import D2STGNN
from models.MTGNN.arch import MTGNN
from models.MegaCRN.arch import MegaCRN
from models.STAEformer.arch import STAEformer
from models.STGCNChebGraphConv.arch.stgcn_arch import STGCNChebGraphConv
from models.STID.arch import STID
from models.STNorm.arch import STNorm


DATA_URL = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
OUT_DIR = Path("results/nontraffic_graph_sanity")
DATA_PATH = OUT_DIR / "chickenpox.json"
RUN_ARTIFACT_DIR = OUT_DIR / "chickenpox_run_artifacts"
PROTOCOL_ARRAYS_PATH = OUT_DIR / "chickenpox_protocol_arrays.npz"
INPUT_LEN = 12
OUTPUT_LEN = 12
ALPHA = 0.10
SEEDS = (43, 44, 45)
MODELS = (
    "D2STGNN",
    "MegaCRN",
    "MTGNN",
    "STNorm",
    "STGCN-Cheb",
    "STID",
    "STAEformer",
)
PARTITION_BOUNDS = {
    "train": (0, 286),
    "val": (297, 327),
    "calibration": (338, 388),
    "test": (399, 498),
}
EXCLUDED_BOUNDS = ((286, 297), (327, 338), (388, 399))
SUMMARY_FIELDS = (
    "mae",
    "rmse",
    "conformal_coverage_90",
    "conformal_interval_width",
    "best_val_mae",
    "best_epoch",
    "train_seconds",
    "num_parameters",
)


@dataclass
class Metrics:
    mae: float
    rmse: float
    conformal_coverage_90: float
    conformal_interval_width: float
    best_val_mae: float
    best_epoch: int
    train_seconds: float
    num_parameters: int
    validation_mae_history: list[float]
    coverage_by_horizon: list[float]
    interval_width_by_horizon: list[float]
    conformal_rank: int
    run_artifact_path: str
    run_artifact_sha256: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fetch_dataset() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            DATA_PATH.write_bytes(response.read())
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def build_windows(data: np.ndarray, input_len: int, output_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(data.shape[0] - input_len - output_len + 1):
        xs.append(data[i : i + input_len].T)
        ys.append(data[i + input_len : i + input_len + output_len].T)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def split_data(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray]],
    dict[str, np.ndarray],
    np.ndarray,
]:
    if x.shape[0] != 498 or y.shape[0] != 498:
        raise ValueError(
            "The documented Chickenpox protocol expects 498 windows; "
            f"found x={x.shape[0]} and y={y.shape[0]}"
        )

    indices = {
        name: np.arange(start, stop, dtype=np.int64)
        for name, (start, stop) in PARTITION_BOUNDS.items()
    }
    excluded = np.concatenate(
        [
            np.arange(start, stop, dtype=np.int64)
            for start, stop in EXCLUDED_BOUNDS
        ]
    )
    assigned = np.concatenate(list(indices.values()))
    if len(np.unique(np.concatenate([assigned, excluded]))) != x.shape[0]:
        raise ValueError("Partition and exclusion indices do not cover each window once")

    split_names = list(indices)
    for left_name, right_name in zip(split_names, split_names[1:]):
        left_last_target = int(indices[left_name][-1] + INPUT_LEN + OUTPUT_LEN - 1)
        right_first_target = int(indices[right_name][0] + INPUT_LEN)
        if left_last_target >= right_first_target:
            raise ValueError(
                f"Targets overlap between {left_name} and {right_name}: "
                f"{left_last_target} >= {right_first_target}"
            )

    splits = {name: (x[idx], y[idx]) for name, idx in indices.items()}
    return splits, indices, excluded


def standardize(splits: dict[str, tuple[np.ndarray, np.ndarray]]):
    train_y = splits["train"][1]
    mean = float(train_y.mean())
    std = float(train_y.std() + 1e-6)
    out = {}
    for name, (x, y) in splits.items():
        out[name] = ((x - mean) / std, (y - mean) / std)
    return out, mean, std


def add_time_features(x: np.ndarray, start_indices: np.ndarray, steps_per_year: int = 52) -> np.ndarray:
    # x: [B, N, T] -> [B, T, N, 3]
    b, n, t = x.shape
    values = np.transpose(x, (0, 2, 1))[..., None]
    offsets = np.arange(t, dtype=np.int64)[None, :]
    week = ((start_indices[:, None] + offsets) % steps_per_year).astype(np.float32) / steps_per_year
    week = np.repeat(week[:, :, None], n, axis=2)[..., None]
    dummy = np.zeros_like(week)
    return np.concatenate([values, week, dummy], axis=-1).astype(np.float32)


def prepare_tensors(splits_scaled, split_indices):
    tensors = {}
    for name in ["train", "val", "calibration", "test"]:
        x_scaled, y_scaled = splits_scaled[name]
        starts = split_indices[name]
        hist = add_time_features(x_scaled, starts)
        future_features = add_time_features(y_scaled, starts + INPUT_LEN)
        target = np.transpose(y_scaled, (0, 2, 1))[..., None].astype(np.float32)
        tensors[name] = (hist, future_features, target)
    return tensors


def normalized_adj(edges: list[list[int]], n: int) -> torch.Tensor:
    adj = np.eye(n, dtype=np.float32)
    for src, dst in edges:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return torch.tensor(adj / np.maximum(degree, 1.0), dtype=torch.float32)


def norm_lap(adj: torch.Tensor) -> torch.Tensor:
    a = (adj.detach().cpu().numpy() > 0).astype(np.float32)
    a = ((a + np.eye(a.shape[0], dtype=np.float32)) > 0).astype(np.float32)
    d = a.sum(axis=1)
    inv_sqrt = np.power(np.maximum(d, 1.0), -0.5)
    return torch.tensor(np.eye(a.shape[0], dtype=np.float32) - inv_sqrt[:, None] * a * inv_sqrt[None, :])


def double_transition(adj: torch.Tensor) -> list[torch.Tensor]:
    a = (adj.detach().cpu().numpy() > 0).astype(np.float32)
    out = a / np.maximum(a.sum(axis=1, keepdims=True), 1.0)
    inn = a.T / np.maximum(a.T.sum(axis=1, keepdims=True), 1.0)
    return [torch.tensor(out, dtype=torch.float32), torch.tensor(inn, dtype=torch.float32)]


def make_loader(parts, model_name: str, batch_size: int, shuffle: bool) -> DataLoader:
    hist, future, target = parts
    if model_name in {"STGCN-Cheb", "STNorm", "MTGNN"}:
        hist = hist[..., [0]]
        future = future[..., [0]]
    elif model_name == "MegaCRN":
        hist = hist[..., [0, 1]]
        future = future[..., [0, 1]]
    ds = TensorDataset(
        torch.tensor(hist, dtype=torch.float32),
        torch.tensor(future, dtype=torch.float32),
        torch.tensor(target, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def unwrap_prediction(output):
    return output["prediction"] if isinstance(output, dict) else output


def normalize_output_shape(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim != 4:
        raise ValueError(f"Unexpected prediction shape: {tuple(pred.shape)}")
    # STNorm returns [B, output_len, N, 1] with the config below; keep generic.
    return pred


def make_model(name: str, n: int, input_len: int, output_len: int, adj: torch.Tensor) -> nn.Module:
    if name == "STID":
        return STID(
            num_nodes=n,
            input_len=input_len,
            input_dim=3,
            embed_dim=32,
            output_len=output_len,
            num_layer=2,
            if_node=True,
            node_dim=8,
            if_T_i_D=True,
            if_D_i_W=False,
            temp_dim_tid=8,
            temp_dim_diw=0,
            time_of_day_size=52,
            day_of_week_size=1,
        )
    if name == "STNorm":
        return STNorm(n, True, True, 1, output_len, 16, 2, 4, 2)
    if name == "MTGNN":
        return MTGNN(True, True, 2, n, None, None, 0.1, min(20, n), 16, 1, 16, 16, 32, 64, input_len, 1, output_len, 2)
    if name == "STAEformer":
        return STAEformer(n, input_len, output_len, 52, 3, 1, 8, 8, 0, 0, 16, 64, 4, 1, 0.1, True)
    if name == "STGCN-Cheb":
        return STGCNChebGraphConv(3, 3, [[1], [16, 8, 16], [16, 8, 16], [32, 32], [output_len]], input_len, n, "glu", "cheb_graph_conv", norm_lap(adj), True, 0.2)
    if name == "MegaCRN":
        return MegaCRN(n, 1, 1, output_len, 32, 1, 2, 1, 10, 16, 2000, False)
    if name == "D2STGNN":
        return D2STGNN(
            num_feat=1,
            num_hidden=16,
            dropout=0.1,
            seq_length=output_len,
            k_t=3,
            k_s=2,
            gap=3,
            num_nodes=n,
            adjs=double_transition(adj),
            num_layers=5,
            num_modalities=2,
            node_hidden=8,
            time_emb_dim=8,
            time_in_day_size=52,
            day_in_week_size=1,
        )
    raise ValueError(name)


def eval_model(model, loader, mean, std):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for hist, future, target in loader:
            pred = normalize_output_shape(unwrap_prediction(model(hist, future_data=future, batch_seen=1, epoch=1, train=False)))
            preds.append(pred.numpy() * std + mean)
            targets.append(target.numpy() * std + mean)
    y_pred = np.concatenate(preds, axis=0)
    y_true = np.concatenate(targets, axis=0)
    err = y_pred - y_true
    return y_pred, y_true, float(np.abs(err).mean()), float(np.sqrt(np.mean(err**2)))


def coordinate_conformal_quantile(
    calibration_pred: np.ndarray,
    calibration_true: np.ndarray,
    alpha: float = ALPHA,
) -> tuple[np.ndarray, np.ndarray, int]:
    scores = np.abs(calibration_pred - calibration_true)
    calibration_origins = scores.shape[0]
    rank = min(
        math.ceil((calibration_origins + 1) * (1.0 - alpha)),
        calibration_origins,
    )
    quantiles = np.sort(scores, axis=0)[rank - 1]
    return scores, quantiles, rank


def run_artifact_path(name: str, seed: int) -> Path:
    slug = name.lower().replace("-", "_")
    return RUN_ARTIFACT_DIR / f"{slug}_seed{seed}.npz"


def train_one(name, tensors, adj, mean, std, seed, epochs=120, patience=25) -> Metrics:
    set_seed(seed)
    model = make_model(name, 20, 12, 12, adj)
    num_params = int(sum(p.numel() for p in model.parameters()))
    lr = {"D2STGNN": 0.002, "MegaCRN": 0.005, "STGCN-Cheb": 0.001, "STAEformer": 0.001}.get(name, 0.003)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    train_loader = make_loader(tensors["train"], name, 32, True)
    val_loader = make_loader(tensors["val"], name, 128, False)
    calibration_loader = make_loader(
        tensors["calibration"], name, 128, False
    )
    test_loader = make_loader(tensors["test"], name, 128, False)
    best_state, best_val, best_epoch, stale = None, math.inf, -1, 0
    validation_mae_history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for hist, future, target in train_loader:
            optim.zero_grad(set_to_none=True)
            pred = normalize_output_shape(unwrap_prediction(model(hist, future_data=future, batch_seen=epoch, epoch=epoch, train=True)))
            loss = loss_fn(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optim.step()
        _, _, val_mae, _ = eval_model(model, val_loader, mean, std)
        validation_mae_history.append(val_mae)
        if val_mae < best_val:
            best_val, best_epoch, stale = val_mae, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    train_seconds = time.time() - start
    calibration_pred, calibration_true, _, _ = eval_model(
        model, calibration_loader, mean, std
    )
    test_pred, test_true, mae, rmse = eval_model(model, test_loader, mean, std)
    calibration_residuals, coordinate_quantiles, rank = (
        coordinate_conformal_quantile(calibration_pred, calibration_true)
    )
    covered = np.abs(test_pred - test_true) <= coordinate_quantiles
    coverage = float(covered.mean())
    interval_width = float((2.0 * coordinate_quantiles).mean())
    coverage_by_horizon = covered.mean(axis=(0, 2, 3)).tolist()
    width_by_horizon = (2.0 * coordinate_quantiles).mean(
        axis=(1, 2)
    ).tolist()

    RUN_ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = run_artifact_path(name, seed)
    np.savez_compressed(
        artifact_path,
        calibration_residuals=calibration_residuals.astype(np.float32),
        coordinate_quantiles=coordinate_quantiles.astype(np.float32),
        test_predictions=test_pred.astype(np.float32),
        coverage_by_horizon=np.asarray(
            coverage_by_horizon, dtype=np.float64
        ),
        interval_width_by_horizon=np.asarray(
            width_by_horizon, dtype=np.float64
        ),
    )
    return Metrics(
        mae=mae,
        rmse=rmse,
        conformal_coverage_90=coverage,
        conformal_interval_width=interval_width,
        best_val_mae=best_val,
        best_epoch=best_epoch,
        train_seconds=train_seconds,
        num_parameters=num_params,
        validation_mae_history=validation_mae_history,
        coverage_by_horizon=coverage_by_horizon,
        interval_width_by_horizon=width_by_horizon,
        conformal_rank=rank,
        run_artifact_path=artifact_path.as_posix(),
        run_artifact_sha256=file_sha256(artifact_path),
    )


def summarize(results):
    summary = {}
    for name, rows in results.items():
        summary[name] = {}
        for field in SUMMARY_FIELDS:
            values = np.array([getattr(row, field) for row in rows], dtype=float)
            summary[name][f"{field}_mean"] = float(values.mean())
            summary[name][f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_manifest(
    source_data: np.ndarray,
    mean: float,
    std: float,
    split_indices: dict[str, np.ndarray],
    excluded_indices: np.ndarray,
    total_windows: int,
    protocol_arrays_sha256: str,
) -> dict:
    column_means = source_data.mean(axis=0)
    column_stds = source_data.std(axis=0)
    partition_details = {
        name: {
            "first_window_index": int(indices[0]),
            "last_window_index": int(indices[-1]),
            "window_count": int(len(indices)),
            "first_target_week_index": int(indices[0] + INPUT_LEN),
            "last_target_week_index": int(
                indices[-1] + INPUT_LEN + OUTPUT_LEN - 1
            ),
        }
        for name, indices in split_indices.items()
    }
    return {
        "schema_version": 2,
        "role": (
            "secondary graph-native non-traffic experiment with target-disjoint "
            "model-selection, calibration, and evaluation periods"
        ),
        "ranking_scope": "not pooled with the main traffic-domain rankings",
        "dataset": {
            "name": "Hungarian Chickenpox Cases",
            "source_url": DATA_URL,
            "snapshot_path": DATA_PATH.as_posix(),
            "snapshot_sha256": file_sha256(DATA_PATH),
            "time_steps": 521,
            "nodes": 20,
            "source_edges": 102,
            "value_scale": (
                "county-wise standardized FX signal units in the dataset-provided JSON; "
                "not raw weekly case counts"
            ),
            "source_preprocessing_limitation": (
                "The dataset-provided FX columns were standardized over all 521 "
                "source weeks before this experiment. Only the additional "
                "optimization transform below is estimated from training data."
            ),
            "source_fx_standardization": {
                "axis": "each county column over all 521 source weeks",
                "maximum_absolute_column_mean": float(
                    np.abs(column_means).max()
                ),
                "minimum_column_population_std": float(column_stds.min()),
                "maximum_column_population_std": float(column_stds.max()),
            },
        },
        "forecast_task": {
            "input_weeks": INPUT_LEN,
            "output_weeks": OUTPUT_LEN,
            "total_forecast_origin_windows": total_windows,
            "assigned_windows": int(
                sum(len(indices) for indices in split_indices.values())
            ),
            "split_windows": {
                name: int(len(indices))
                for name, indices in split_indices.items()
            },
            "partition_details": partition_details,
            "excluded_boundary_window_indices": excluded_indices.tolist(),
            "excluded_boundary_ranges": [
                {
                    "first_window_index": start,
                    "last_window_index": stop - 1,
                    "window_count": stop - start,
                }
                for start, stop in EXCLUDED_BOUNDS
            ],
            "boundary_exclusion_reason": (
                "Eleven forecast origins (output horizon minus one) are omitted at "
                "each boundary so adjacent partitions share no target weeks."
            ),
            "target_periods_are_disjoint": True,
            "test_partition_matches_previous_protocol": True,
        },
        "preprocessing": {
            "optimization_transform": (
                "additional single scalar mean and standard deviation estimated "
                "from training targets only"
            ),
            "training_target_mean": mean,
            "training_target_std": std,
            "same_transform_for_all_models": True,
            "evaluation_inverse_transform": (
                "reverses only the additional training-target scalar transform"
            ),
            "reported_error_units": (
                "dataset-provided county-wise standardized FX signal units"
            ),
            "time_features": "week index modulo 52 plus one zero dummy channel",
        },
        "graph": {
            "construction": "source neighbor edges symmetrized with self-loops",
            "base_normalization": "row normalization",
            "model_specific_supports": {
                "D2STGNN": "forward and reverse random-walk transitions",
                "STGCN-Cheb": "symmetric normalized Laplacian",
                "other_models": "their released adaptive, memory, normalization, identity, or embedding paths",
            },
        },
        "optimization": {
            "seeds": [43, 44, 45],
            "execution_device": "CPU",
            "maximum_epochs": 120,
            "early_stopping_patience": 25,
            "checkpoint_selector": "minimum validation MAE",
            "checkpoint_storage": (
                "best state retained in memory; no checkpoint files written by this script"
            ),
            "loss": "L1",
            "optimizer": "Adam",
            "weight_decay": 1e-5,
            "gradient_clip_max_norm": 5.0,
            "train_batch_size": 32,
            "validation_test_batch_size": 128,
            "learning_rates": {
                "D2STGNN": 0.002,
                "MegaCRN": 0.005,
                "MTGNN": 0.003,
                "STNorm": 0.003,
                "STGCN-Cheb": 0.001,
                "STID": 0.003,
                "STAEformer": 0.001,
            },
        },
        "model_dimensions": {
            "D2STGNN": "hidden=16, node_hidden=8, time_embedding=8, layers=5, k_t=3, k_s=2",
            "MegaCRN": "rnn_units=32, layers=1, cheb_k=2, memory=10x16",
            "MTGNN": "gcn_depth=2, node_dim=16, conv/residual=16, skip=32, end=64, layers=2",
            "STNorm": "channels=16, kernel_size=2, blocks=4, layers=2",
            "STGCN-Cheb": "Kt=3, Ks=3, two 16/8/16 blocks, dropout=0.2",
            "STID": "embedding=32, layers=2, node/time embeddings=8",
            "STAEformer": "input/time embeddings=8, adaptive embedding=16, feed-forward=64, heads=4, layer=1",
        },
        "interval_diagnostic": {
            "nominal_coverage": 0.90,
            "calibration_split": "dedicated chronological calibration partition",
            "calibration_origins": int(len(split_indices["calibration"])),
            "evaluation_split": "chronological test split",
            "evaluation_origins": int(len(split_indices["test"])),
            "coordinates_per_origin": 20 * OUTPUT_LEN,
            "calibration_coordinate_values": int(
                len(split_indices["calibration"]) * 20 * OUTPUT_LEN
            ),
            "evaluation_coordinate_values": int(
                len(split_indices["test"]) * 20 * OUTPUT_LEN
            ),
            "score": "absolute residual at each horizon-county coordinate",
            "replication_axis": "forecast origin",
            "pooling": (
                "No pooling across horizons or counties; each coordinate uses "
                "its 50 origin-level calibration residuals."
            ),
            "finite_sample_rank": 46,
            "rank_formula": "ceil((50 + 1) * (1 - 0.10))",
            "interval": (
                "point prediction plus or minus the model-and-seed coordinate-wise "
                "calibration residual quantile"
            ),
            "temporal_dependence_caveat": (
                "reported coverage is an empirical diagnostic, not an exchangeability guarantee"
            ),
        },
        "artifacts": {
            "protocol_arrays_path": PROTOCOL_ARRAYS_PATH.as_posix(),
            "protocol_arrays_sha256": protocol_arrays_sha256,
            "protocol_arrays_contents": (
                "partition indices, excluded boundary indices, calibration targets, "
                "and unchanged test targets in dataset-provided FX units"
            ),
        },
    }


def write_outputs(
    summary,
    results,
    source_data,
    window_targets,
    mean,
    std,
    split_indices,
    excluded_indices,
):
    np.savez_compressed(
        PROTOCOL_ARRAYS_PATH,
        train_window_indices=split_indices["train"],
        validation_window_indices=split_indices["val"],
        calibration_window_indices=split_indices["calibration"],
        test_window_indices=split_indices["test"],
        excluded_boundary_window_indices=excluded_indices,
        calibration_targets=window_targets[split_indices["calibration"]],
        test_targets=window_targets[split_indices["test"]],
    )
    protocol = protocol_manifest(
        source_data,
        mean,
        std,
        split_indices,
        excluded_indices,
        int(window_targets.shape[0]),
        file_sha256(PROTOCOL_ARRAYS_PATH),
    )
    protocol["optimization"]["best_epochs"] = {
        name: {
            str(seed): row.best_epoch
            for seed, row in zip(SEEDS, rows)
        }
        for name, rows in results.items()
    }
    protocol["artifacts"]["run_artifacts"] = [
        {
            "model": name,
            "seed": seed,
            "path": row.run_artifact_path,
            "sha256": row.run_artifact_sha256,
            "contents": (
                "calibration residuals, coordinate quantiles, test predictions, "
                "and horizon-wise coverage and width"
            ),
        }
        for name, rows in results.items()
        for seed, row in zip(SEEDS, rows)
    ]
    protocol["generation_script"] = "scripts/run_chickenpox_all_baselines.py"
    protocol["result_path"] = (
        "results/nontraffic_graph_sanity/chickenpox_all_baselines_summary.json"
    )
    payload = {
        "dataset": "Hungarian Chickenpox Cases",
        "task": (
            "12-week input to 12-week output forecasting in the dataset-provided "
            "county-wise standardized FX signal units"
        ),
        "split": (
            "target-disjoint chronological train/validation/calibration/test "
            "partitions with 11 excluded origins at each boundary"
        ),
        "models": list(results.keys()),
        "seeds": list(SEEDS),
        "protocol_manifest_path": "results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json",
        "protocol": protocol,
        "summary": summary,
        "per_seed": {name: [asdict(row) for row in rows] for name, rows in results.items()},
        "interpretation": (
            "Secondary graph-native non-traffic experiment using the same seven "
            "model classes; not pooled with the main traffic ranking."
        ),
    }
    json_path = OUT_DIR / "chickenpox_all_baselines_summary.json"
    protocol_path = OUT_DIR / "chickenpox_protocol_manifest.json"
    md_path = OUT_DIR / "chickenpox_all_baselines_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    def fmt(name, field):
        return f"{summary[name][field + '_mean']:.4f} +/- {summary[name][field + '_std']:.4f}"

    lines = [
        "# Chickenpox Hungary Graph-Native Protocol Illustration",
        "",
        "Secondary graph-native non-traffic experiment using the same seven model classes from the traffic benchmark with compact Chickenpox-specific dimensions.",
        "",
        "The target-disjoint split uses 286 training, 30 validation, 50 calibration, and 99 unchanged test origins. Eleven origins are excluded at each boundary so 12-week forecast targets do not overlap across partitions.",
        "",
        "| Model | MAE | RMSE | 90% coverage | Width | Params |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in sorted(results, key=lambda m: summary[m]["mae_mean"]):
        lines.append(
            f"| {name} | {fmt(name, 'mae')} | {fmt(name, 'rmse')} | {fmt(name, 'conformal_coverage_90')} | "
            f"{fmt(name, 'conformal_interval_width')} | {summary[name]['num_parameters_mean']:.0f} |"
        )
    lines += [
        "",
        "The protocol, validation histories, per-seed best epochs, calibration residuals, coordinate quantiles, and test predictions are retained with the release. This secondary experiment is not pooled with the traffic-domain model rankings.",
        "",
        "MAE, RMSE, and interval width remain in the dataset-provided county-wise standardized FX signal units. They are not numbers of weekly cases.",
        "",
        "Coverage is an empirical chronological diagnostic under temporal dependence, not a distribution-free guarantee under arbitrary temporal shift.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataset = fetch_dataset()
    source_data = np.array(dataset["FX"], dtype=np.float64)
    data = source_data.astype(np.float32)
    x, y = build_windows(data, INPUT_LEN, OUTPUT_LEN)
    raw, split_indices, excluded_indices = split_data(x, y)
    scaled, mean, std = standardize(raw)
    tensors = prepare_tensors(scaled, split_indices)
    adj = normalized_adj(dataset["edges"], data.shape[1])
    results = {}
    for name in MODELS:
        print(f"Running {name}...")
        rows = []
        for seed in SEEDS:
            rows.append(train_one(name, tensors, adj, mean, std, seed))
            print(f"  seed {seed}: MAE={rows[-1].mae:.4f}, coverage={rows[-1].conformal_coverage_90:.4f}")
        results[name] = rows
    write_outputs(
        summarize(results),
        results,
        source_data,
        y,
        mean,
        std,
        split_indices,
        excluded_indices,
    )
    print((OUT_DIR / "chickenpox_all_baselines_summary.md").resolve())


if __name__ == "__main__":
    main()
