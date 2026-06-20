"""Train the seven paper baselines on Chickenpox Hungary as a small sanity check.

This is not a full non-traffic benchmark. It reuses the same model classes from
the traffic paper with small Chickenpox-specific dimensions.
"""

from __future__ import annotations

import json
import math
import random
import ssl
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
ORIGINAL_BASICTS = Path("D:/Hussein-Files/original/experiments/basicts")
if ORIGINAL_BASICTS.exists():
    sys.path.insert(0, str(ORIGINAL_BASICTS))

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fetch_dataset() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not DATA_PATH.exists():
        context = ssl._create_unverified_context()
        with urllib.request.urlopen(DATA_URL, context=context, timeout=60) as response:
            DATA_PATH.write_bytes(response.read())
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def build_windows(data: np.ndarray, input_len: int, output_len: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(data.shape[0] - input_len - output_len + 1):
        xs.append(data[i : i + input_len].T)
        ys.append(data[i + input_len : i + input_len + output_len].T)
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32)


def split_data(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = x.shape[0]
    n_train = round(n * 0.7)
    n_val = round(n * 0.1)
    return {
        "train": (x[:n_train], y[:n_train]),
        "val": (x[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        "test": (x[n_train + n_val :], y[n_train + n_val :]),
    }


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


def prepare_tensors(splits_raw, splits_scaled, input_len: int, output_len: int):
    tensors = {}
    offset = 0
    for name in ["train", "val", "test"]:
        x_scaled, y_scaled = splits_scaled[name]
        count = x_scaled.shape[0]
        starts = np.arange(offset, offset + count)
        offset += count
        hist = add_time_features(x_scaled, starts)
        future_features = add_time_features(y_scaled, starts + input_len)
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


def train_one(name, tensors, adj, mean, std, seed, epochs=120, patience=25) -> Metrics:
    set_seed(seed)
    model = make_model(name, 20, 12, 12, adj)
    num_params = int(sum(p.numel() for p in model.parameters()))
    lr = {"D2STGNN": 0.002, "MegaCRN": 0.005, "STGCN-Cheb": 0.001, "STAEformer": 0.001}.get(name, 0.003)
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.L1Loss()
    train_loader = make_loader(tensors["train"], name, 32, True)
    val_loader = make_loader(tensors["val"], name, 128, False)
    test_loader = make_loader(tensors["test"], name, 128, False)
    best_state, best_val, best_epoch, stale = None, math.inf, -1, 0
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
    val_pred, val_true, _, _ = eval_model(model, val_loader, mean, std)
    test_pred, test_true, mae, rmse = eval_model(model, test_loader, mean, std)
    q = float(np.quantile(np.abs(val_pred - val_true).reshape(-1), 0.9, method="higher"))
    coverage = float((np.abs(test_pred - test_true) <= q).mean())
    return Metrics(mae, rmse, coverage, 2.0 * q, best_val, best_epoch, train_seconds, num_params)


def summarize(results):
    summary = {}
    for name, rows in results.items():
        summary[name] = {}
        for field in Metrics.__dataclass_fields__:
            values = np.array([getattr(row, field) for row in rows], dtype=float)
            summary[name][f"{field}_mean"] = float(values.mean())
            summary[name][f"{field}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return summary


def write_outputs(summary, results):
    payload = {
        "dataset": "Hungarian Chickenpox Cases",
        "task": "12-week input to 12-week output county-level case-count forecasting",
        "split": "chronological 70/10/20 train/validation/test",
        "models": list(results.keys()),
        "seeds": [43, 44, 45],
        "summary": summary,
        "per_seed": {name: [asdict(row) for row in rows] for name, rows in results.items()},
        "interpretation": "Appendix-only non-traffic graph sanity check using the same seven model classes; not part of the main traffic ranking.",
    }
    json_path = OUT_DIR / "chickenpox_all_baselines_summary.json"
    md_path = OUT_DIR / "chickenpox_all_baselines_summary.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def fmt(name, field):
        return f"{summary[name][field + '_mean']:.4f} +/- {summary[name][field + '_std']:.4f}"

    lines = [
        "# Chickenpox Hungary Same-Baseline Sanity Check",
        "",
        "Appendix-only graph-native non-traffic check using the same seven model classes from the traffic benchmark, with small Chickenpox-specific dimensions.",
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
        "Interpret this as a feasibility/sanity check only. Hyperparameters were scaled down for the 20-node weekly dataset and are not intended as a full non-traffic benchmark.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    dataset = fetch_dataset()
    data = np.array(dataset["FX"], dtype=np.float32)
    x, y = build_windows(data, 12, 12)
    raw = split_data(x, y)
    scaled, mean, std = standardize(raw)
    tensors = prepare_tensors(raw, scaled, 12, 12)
    adj = normalized_adj(dataset["edges"], data.shape[1])
    results = {}
    for name in ["D2STGNN", "MegaCRN", "MTGNN", "STNorm", "STGCN-Cheb", "STID", "STAEformer"]:
        print(f"Running {name}...")
        rows = []
        for seed in [43, 44, 45]:
            rows.append(train_one(name, tensors, adj, mean, std, seed))
            print(f"  seed {seed}: MAE={rows[-1].mae:.4f}, coverage={rows[-1].conformal_coverage_90:.4f}")
        results[name] = rows
    write_outputs(summarize(results), results)
    print((OUT_DIR / "chickenpox_all_baselines_summary.md").resolve())


if __name__ == "__main__":
    main()
