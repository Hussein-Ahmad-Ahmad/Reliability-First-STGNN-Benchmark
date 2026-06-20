"""Smoke-test paper baseline models on Chickenpox-shaped tensors.

This does not train models. It checks whether the seven benchmarked model
classes can be instantiated with small Chickenpox-compatible dimensions and can
produce a forward pass for 12-to-12 weekly forecasting.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

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


OUT_DIR = Path("results/nontraffic_graph_sanity")
DATA_PATH = OUT_DIR / "chickenpox.json"
OUT_PATH = OUT_DIR / "chickenpox_baseline_compatibility.json"


def load_data() -> dict:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} missing. Run scripts/run_chickenpox_graph_sanity.py first."
        )
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


def normalized_adj(edges: list[list[int]], n: int) -> torch.Tensor:
    adj = np.eye(n, dtype=np.float32)
    for src, dst in edges:
        adj[src, dst] = 1.0
        adj[dst, src] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return torch.tensor(adj / np.maximum(degree, 1.0), dtype=torch.float32)


def norm_lap(adj: torch.Tensor) -> torch.Tensor:
    a = adj.detach().cpu().numpy().astype(np.float32)
    a = ((a > 0).astype(np.float32) + np.eye(a.shape[0], dtype=np.float32))
    a = (a > 0).astype(np.float32)
    d = a.sum(axis=1)
    inv_sqrt = np.power(np.maximum(d, 1.0), -0.5)
    l = np.eye(a.shape[0], dtype=np.float32) - inv_sqrt[:, None] * a * inv_sqrt[None, :]
    return torch.tensor(l, dtype=torch.float32)


def double_transition(adj: torch.Tensor) -> list[torch.Tensor]:
    a = adj.detach().cpu().numpy().astype(np.float32)
    a = (a > 0).astype(np.float32)
    out = a / np.maximum(a.sum(axis=1, keepdims=True), 1.0)
    inn = a.T / np.maximum(a.T.sum(axis=1, keepdims=True), 1.0)
    return [torch.tensor(out, dtype=torch.float32), torch.tensor(inn, dtype=torch.float32)]


def make_history(batch: int, input_len: int, num_nodes: int, channels: int) -> torch.Tensor:
    x = torch.randn(batch, input_len, num_nodes, channels)
    if channels >= 2:
        week = torch.arange(input_len, dtype=torch.float32).view(1, input_len, 1)
        x[..., 1] = (week % 52) / 52.0
    if channels >= 3:
        x[..., 2] = 0.0
    return x


def make_future(batch: int, output_len: int, num_nodes: int, channels: int) -> torch.Tensor:
    y = torch.randn(batch, output_len, num_nodes, channels)
    if channels >= 2:
        week = torch.arange(output_len, dtype=torch.float32).view(1, output_len, 1)
        y[..., 1] = (week % 52) / 52.0
    if channels >= 3:
        y[..., 2] = 0.0
    return y


def unwrap_prediction(output):
    if isinstance(output, dict):
        return output.get("prediction")
    return output


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    n = len(data["FX"][0])
    input_len = 12
    output_len = 12
    adj = normalized_adj(data["edges"], n)
    results = {}

    configs = {
        "STID": (
            lambda: STID(
                num_nodes=n,
                input_len=input_len,
                input_dim=3,
                embed_dim=32,
                output_len=output_len,
                num_layer=2,
                if_node=True,
                node_dim=8,
                if_T_i_D=True,
                if_D_i_W=True,
                temp_dim_tid=8,
                temp_dim_diw=4,
                time_of_day_size=52,
                day_of_week_size=1,
            ),
            make_history(2, input_len, n, 3),
            make_future(2, output_len, n, 3),
        ),
        "STNorm": (
            lambda: STNorm(
                num_nodes=n,
                tnorm_bool=True,
                snorm_bool=True,
                in_dim=1,
                out_dim=output_len,
                channels=16,
                kernel_size=2,
                blocks=4,
                layers=2,
            ),
            make_history(2, input_len, n, 1),
            make_future(2, output_len, n, 1),
        ),
        "MTGNN": (
            lambda: MTGNN(
                gcn_true=True,
                buildA_true=True,
                gcn_depth=2,
                num_nodes=n,
                predefined_A=None,
                dropout=0.1,
                subgraph_size=min(20, n),
                node_dim=16,
                dilation_exponential=1,
                conv_channels=16,
                residual_channels=16,
                skip_channels=32,
                end_channels=64,
                seq_length=input_len,
                in_dim=1,
                out_dim=output_len,
                layers=2,
                propalpha=0.05,
                tanhalpha=3,
                layer_norm_affline=True,
            ),
            make_history(2, input_len, n, 1),
            make_future(2, output_len, n, 1),
        ),
        "STAEformer": (
            lambda: STAEformer(
                num_nodes=n,
                in_steps=input_len,
                out_steps=output_len,
                steps_per_day=52,
                input_dim=3,
                output_dim=1,
                input_embedding_dim=8,
                tod_embedding_dim=8,
                dow_embedding_dim=0,
                spatial_embedding_dim=0,
                adaptive_embedding_dim=16,
                feed_forward_dim=64,
                num_heads=4,
                num_layers=1,
                dropout=0.1,
                use_mixed_proj=True,
            ),
            make_history(2, input_len, n, 3),
            make_future(2, output_len, n, 3),
        ),
        "STGCN-Cheb": (
            lambda: STGCNChebGraphConv(
                Ks=3,
                Kt=3,
                blocks=[[1], [16, 8, 16], [16, 8, 16], [32, 32], [output_len]],
                T=input_len,
                n_vertex=n,
                act_func="glu",
                graph_conv_type="cheb_graph_conv",
                gso=norm_lap(adj),
                bias=True,
                droprate=0.2,
            ),
            make_history(2, input_len, n, 1),
            make_future(2, output_len, n, 1),
        ),
        "MegaCRN": (
            lambda: MegaCRN(
                num_nodes=n,
                input_dim=1,
                output_dim=1,
                horizon=output_len,
                rnn_units=32,
                num_layers=1,
                cheb_k=2,
                ycov_dim=1,
                mem_num=10,
                mem_dim=16,
                cl_decay_steps=2000,
                use_curriculum_learning=False,
            ),
            make_history(2, input_len, n, 2),
            make_future(2, output_len, n, 2),
        ),
        "D2STGNN": (
            lambda: D2STGNN(
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
            ),
            make_history(2, input_len, n, 3),
            make_future(2, output_len, n, 3),
        ),
    }

    for name, (factory, history, future) in configs.items():
        try:
            torch.manual_seed(1)
            model = factory()
            model.eval()
            with torch.no_grad():
                out = unwrap_prediction(
                    model(history, future_data=future, batch_seen=1, epoch=1, train=False)
                )
            results[name] = {
                "status": "forward_pass_ok",
                "output_shape": list(out.shape) if out is not None else None,
                "num_parameters": int(sum(p.numel() for p in model.parameters())),
            }
        except Exception as exc:
            results[name] = {"status": "failed", "error": repr(exc)}

    OUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
