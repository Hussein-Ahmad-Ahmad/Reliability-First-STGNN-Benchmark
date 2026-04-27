from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ORIG_ROOT = Path(r"D:/Hussein-Files/original/experiments/basicts")
CHECKPOINTS_ROOT = ORIG_ROOT / "checkpoints"
EXPORT_ROOT = Path("results/task1_point_forecasting/fresh_inference_dumps")


def find_run_dir(model: str, dataset: str, seed: int) -> Path:
    mroot = CHECKPOINTS_ROOT / model
    if not mroot.exists():
        raise FileNotFoundError(f"model checkpoint root missing: {mroot}")

    # Prefer 100-epoch configs first, then any seed-matching config directory.
    candidates = []
    for d in mroot.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if not name.startswith(dataset + "_"):
            continue
        if f"seed{seed}" not in name:
            continue
        pri = 0 if "_100_" in name else 1
        candidates.append((pri, name, d))

    if not candidates:
        raise FileNotFoundError(f"no seed dir for {model}/{dataset}/seed{seed}")

    for _, _, seed_dir in sorted(candidates):
        run_dirs = sorted([x for x in seed_dir.iterdir() if x.is_dir()])
        for run_dir in run_dirs:
            cfg_file = run_dir / f"{dataset}_seed{seed}.py"
            ckpts = sorted(run_dir.glob(f"{model}_best_val_*.pt"))
            if cfg_file.exists() and ckpts:
                return run_dir

    raise FileNotFoundError(f"no runnable hash dir for {model}/{dataset}/seed{seed}")


def ensure_arch_shim(run_dir: Path, model: str) -> None:
    mapping = {
        "D2STGNN": ("D2STGNN", "D2STGNN"),
        "MegaCRN": ("MegaCRN", "MegaCRN"),
        "MTGNN": ("MTGNN", "MTGNN"),
        "STNorm": ("STNorm", "STNorm"),
        "STGCNChebGraphConv": ("STGCN", "STGCN"),
        "STID": ("STID", "STID"),
        "STAEformer": ("STAEformer", "STAEformer"),
    }
    if model not in mapping:
        return
    pkg, cls = mapping[model]
    arch_file = run_dir / "arch.py"
    # STGCN checkpoints can carry stale shims with a wrong symbol import; rewrite for this model.
    if arch_file.exists() and model != "STGCNChebGraphConv":
        return
    arch_file.write_text(f"from baselines.{pkg}.arch import {cls}\n\n__all__ = ['{cls}']\n", encoding="utf-8")


def ensure_support_shims(run_dir: Path, model: str) -> None:
    # Some checkpoint hash dirs expect local helper modules that are absent; provide passthrough shims.
    if model == "MegaCRN":
        loss_file = run_dir / "loss.py"
        if not loss_file.exists():
            loss_file.write_text(
                "from baselines.MegaCRN.loss import megacrn_loss\n\n__all__ = ['megacrn_loss']\n",
                encoding="utf-8",
            )

    if model == "MTGNN":
        runner_file = run_dir / "runner.py"
        if not runner_file.exists():
            runner_file.write_text(
                "from baselines.MTGNN.runner import MTGNNRunner\n\n__all__ = ['MTGNNRunner']\n",
                encoding="utf-8",
            )


def run_eval(run_dir: Path, model: str, dataset: str, seed: int) -> None:
    cfg_path = run_dir / f"{dataset}_seed{seed}.py"
    ensure_arch_shim(run_dir, model)
    ensure_support_shims(run_dir, model)
    ckpt = sorted(run_dir.glob(f"{model}_best_val_*.pt"))[0]

    cmd = [
        sys.executable,
        str(ORIG_ROOT / "experiments" / "evaluate.py"),
        "--config",
        str(cfg_path.relative_to(ORIG_ROOT)).replace('\\', '/'),
        "--checkpoint",
        str(ckpt),
        "--device_type",
        "cpu",
        "--gpus",
        "0",
    ]
    subprocess.run(cmd, cwd=str(ORIG_ROOT), check=True)


def copy_dump(run_dir: Path, model: str, dataset: str, seed: int) -> tuple[Path, Path]:
    test_dir = run_dir / "test_results"
    pred = test_dir / "predictions.npy"
    tgt = test_dir / "targets.npy"
    if not pred.exists() or not tgt.exists():
        raise FileNotFoundError(f"missing fresh dump after eval: {test_dir}")

    out_dir = EXPORT_ROOT / model / dataset / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pred = out_dir / "predictions.npy"
    out_tgt = out_dir / "targets.npy"
    shutil.copy2(pred, out_pred)
    shutil.copy2(tgt, out_tgt)
    return out_pred, out_tgt


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict fresh-inference dump generator")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[43, 44, 45])
    args = parser.parse_args()

    print(f"Running strict fresh inference for {args.model}/{args.dataset} seeds={args.seeds}")
    results = []

    for seed in args.seeds:
        run_dir = find_run_dir(args.model, args.dataset, seed)
        print(f"seed{seed}: run_dir={run_dir}")
        run_eval(run_dir, args.model, args.dataset, seed)
        pred, tgt = copy_dump(run_dir, args.model, args.dataset, seed)
        print(f"seed{seed}: exported {pred} and {tgt}")
        results.append((seed, str(pred), str(tgt)))

    print("Completed fresh inference dump generation")
    for seed, pred, tgt in results:
        print(f"seed{seed}: {pred} | {tgt}")


if __name__ == "__main__":
    main()
