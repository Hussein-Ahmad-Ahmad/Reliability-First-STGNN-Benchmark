"""Build ensemble prediction/target arrays for a dataset from the in-repo
fresh_inference_dumps, for datasets that don't have a pre-materialized
ensemble archive (PEMS-BAY, PEMS04).

Stacks every available seed-43 model dump for the dataset into a single
[members, origins, horizons, sensors] array plus a matching targets array,
verifying that every member's targets agree (same test partition) before
stacking. Output is written next to where run_conformal_sigma_control.py
expects its inputs.

Usage:
    python scripts/build_ensemble_from_dumps.py --dataset PEMS04
    python scripts/build_ensemble_from_dumps.py --dataset PEMS-BAY
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP_ROOT = ROOT / "results" / "task1_point_forecasting" / "fresh_inference_dumps"
MODEL_ORDER = (
    "D2STGNN",
    "MegaCRN",
    "MTGNN",
    "STGCNChebGraphConv",
    "STID",
    "STNorm",
    "STAEformer",
)


def _load_pairwise_module():
    spec = importlib.util.spec_from_file_location(
        "run_block_bootstrap_pairwise", ROOT / "scripts" / "run_block_bootstrap_pairwise.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bb = _load_pairwise_module()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ensemble arrays from fresh_inference_dumps")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dump-root", type=Path, default=DEFAULT_DUMP_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "task2_uncertainty" / "conformal_sigma_control" / "ensemble_inputs",
    )
    args = parser.parse_args()

    desc = bb.load_dataset_desc(args.dataset)
    settings = desc["regular_settings"]
    horizon = int(settings["OUTPUT_LEN"])
    num_nodes = int(desc["num_nodes"])

    members = []
    reference_target = None
    used_models = []
    for model in MODEL_ORDER:
        dump_dir = args.dump_root / model / args.dataset / f"seed{args.seed}"
        pred_path = dump_dir / "predictions.npy"
        target_path = dump_dir / "targets.npy"
        if not (pred_path.is_file() and target_path.is_file()):
            print(f"  skip {model}: no dump at {dump_dir}")
            continue

        pred = bb.reshape_prediction(
            bb.read_prediction_array(pred_path, num_nodes, horizon), num_nodes, horizon, pred_path
        )
        target = bb.reshape_prediction(
            bb.read_prediction_array(target_path, num_nodes, horizon), num_nodes, horizon, target_path
        )

        if reference_target is None:
            reference_target = target
        elif target.shape != reference_target.shape or not np.allclose(target, reference_target, atol=1e-4):
            raise ValueError(f"{model}'s targets do not match the reference target for {args.dataset}")

        members.append(pred.astype(np.float32))
        used_models.append(model)
        print(f"  included {model}: shape={pred.shape}")

    if len(members) < 2:
        raise RuntimeError(f"Need at least two model dumps to build an ensemble for {args.dataset}")

    ensemble = np.stack(members, axis=0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_out = args.output_dir / f"{args.dataset}_ensemble_predictions.npy"
    target_out = args.output_dir / f"{args.dataset}_targets.npy"
    np.save(pred_out, ensemble)
    np.save(target_out, reference_target.astype(np.float32))

    manifest = {
        "dataset": args.dataset,
        "seed": args.seed,
        "source": "results/task1_point_forecasting/fresh_inference_dumps (in-repo, not the external archive)",
        "member_count": len(used_models),
        "members": used_models,
        "excluded_models": [m for m in MODEL_ORDER if m not in used_models],
        "ensemble_shape": list(ensemble.shape),
        "note": (
            "Built from whichever seed-43 model dumps are present in this repo's "
            "fresh_inference_dumps directory; not necessarily the same member set "
            "as the archived METR-LA ensemble manifest."
        ),
    }
    manifest_out = args.output_dir / f"{args.dataset}_ensemble_manifest.json"
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {pred_out} (shape {ensemble.shape})")
    print(f"Wrote {target_out}")
    print(f"Wrote {manifest_out}")


if __name__ == "__main__":
    main()
