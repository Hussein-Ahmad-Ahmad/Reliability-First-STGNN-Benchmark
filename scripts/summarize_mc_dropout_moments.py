"""Summarize saved predictive moments without loading checkpoints or running inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.mc_dropout_metrics import summarize_moments


def summarize_case(folder: Path, output: Path):
    previous = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    archive_path = folder / previous["artifact"]
    with np.load(archive_path) as arrays:
        diagnostics = summarize_moments(arrays["mean_prediction"], arrays["predictive_std"],
            arrays["target"], previous["interval_z_value"], previous["pass_prediction_sha256"])
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    updated = {**previous, **diagnostics,
        "moments_sha256": digest.hexdigest(),
        "metric_source": "retained predictive moments; no new inference",
        "sampling_seed": previous.get("sampling_seed"),
        "sampling_seed_note": "Historical sampling seed was not recorded" if previous.get("sampling_seed") is None else "Explicit sampling seed",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")
    return {"model": updated["model"], **diagnostics, "moments_sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/task2_uncertainty/mc_dropout_generated")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for path in sorted(args.root.glob("*/METR-LA/seed43/summary.json")):
        relative = path.relative_to(args.root)
        row = summarize_case(path.parent, args.output_root / relative)
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
