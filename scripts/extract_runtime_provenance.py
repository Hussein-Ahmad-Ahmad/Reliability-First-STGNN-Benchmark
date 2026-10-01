"""Extract the METR-LA runtime table from the canonical seed logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "results" / "compute" / "METR-LA_runtime_provenance.json"
TIMESTAMP_PATTERN = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
)
TRAIN_TIME_PATTERN = re.compile(r"train/time:\s*([0-9.]+)")

CANONICAL_RUNS = (
    ("D2STGNN", 43, "df0cf6f0146be24966504e70a2f32880", "training_log_20260212124013.log"),
    ("D2STGNN", 44, "e98ec97bc3505401c5aa34252fad9a9d", "training_log_20260212180448.log"),
    ("D2STGNN", 45, "314de509ae4c09a882a2df45cce6de4c", "training_log_20260212232457.log"),
    ("MegaCRN", 43, "5da0891349a74d561cc671d791d650f5", "training_log_20260213075120.log"),
    ("MegaCRN", 44, "0249ac62895c58a2a348a0e6167406f8", "training_log_20260213093230.log"),
    ("MegaCRN", 45, "7f4070757032836bd729775ac17f4f96", "training_log_20260213111339.log"),
    ("MTGNN", 43, "b49973cb6528058f8eafa1751de49e57", "training_log_20260213044839.log"),
    ("MTGNN", 44, "5f07e537758f747405ba3122fca637f4", "training_log_20260213053548.log"),
    ("MTGNN", 45, "365fa4eaa01f1a0d46779c73085ac308", "training_log_20260213062310.log"),
    ("STAEformer", 43, "6e41ab1fdb68ffe63475b8df1dca8ae8", "training_log_20260213140135.log"),
    ("STAEformer", 44, "dcd1c91d4ec3a2ae6af040965460b9c7", "training_log_20260213160257.log"),
    ("STAEformer", 45, "89c665df504bdd324c7ece2617a87d09", "training_log_20260213180343.log"),
    ("STGCNChebGraphConv", 43, "e6a1d567e4c62e3516fc087a00db8bec", "training_log_20260212091319.log"),
    ("STGCNChebGraphConv", 44, "58eccaeb0bde9543fa28fa70ce14b537", "training_log_20260212102144.log"),
    ("STGCNChebGraphConv", 45, "7ae43e309b4bedd51d49c9173e41a47d", "training_log_20260212113135.log"),
    ("STID", 43, "38e0392ef9dec6dbdddc6a767d3023d4", "training_log_20260213071032.log"),
    ("STID", 44, "bbe9dfdbdd5c9ea9fa8edb60d6ff0094", "training_log_20260213072437.log"),
    ("STID", 45, "9a1e85eb62155c45f724c2b330178a7e", "training_log_20260213073812.log"),
    ("STNorm", 43, "70d43e707a12d602460787b0917023e7", "training_log_20260213125425.log"),
    ("STNorm", 44, "8c6a2abeefe9ca0847647d4da6970827", "training_log_20260213131653.log"),
    ("STNorm", 45, "c4780ebd0790bd61499f070376f786a8", "training_log_20260213133907.log"),
)


def source_path(
    root: Path,
    model: str,
    seed: int,
    experiment_id: str,
    log_name: str,
) -> Path:
    return (
        root
        / "checkpoints"
        / model
        / f"METR-LA_100_12_12_seed{seed}"
        / experiment_id
        / log_name
    )


def extract_run(path: Path) -> dict:
    train_times = []
    timestamps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        train_match = TRAIN_TIME_PATTERN.search(line)
        if train_match:
            train_times.append(float(train_match.group(1)))
        timestamp_match = TIMESTAMP_PATTERN.match(line)
        if timestamp_match:
            timestamps.append(
                datetime.strptime(
                    timestamp_match.group(1), "%Y-%m-%d %H:%M:%S,%f"
                )
            )
    if len(train_times) < 10 or len(timestamps) < 2:
        raise ValueError(f"Incomplete runtime log: {path}")

    final_ten = train_times[-10:]
    return {
        "observed_train_epoch_count": len(train_times),
        "final_ten_train_seconds": final_ten,
        "median_final_ten_train_seconds": statistics.median(final_ten),
        "first_timestamp": timestamps[0].isoformat(timespec="milliseconds"),
        "last_timestamp": timestamps[-1].isoformat(timespec="milliseconds"),
        "recorded_log_span_minutes": (
            timestamps[-1] - timestamps[0]
        ).total_seconds()
        / 60.0,
    }


def build_payload(basicts_root: Path) -> dict:
    per_seed = []
    grouped = defaultdict(list)
    for model, seed, experiment_id, log_name in CANONICAL_RUNS:
        path = source_path(
            basicts_root, model, seed, experiment_id, log_name
        )
        values = extract_run(path)
        row = {
            "model": model,
            "seed": seed,
            "experiment_id": experiment_id,
            "source_log": path.relative_to(basicts_root).as_posix(),
            **values,
        }
        per_seed.append(row)
        grouped[model].append(row)

    aggregate = []
    for model in (
        "D2STGNN",
        "MegaCRN",
        "MTGNN",
        "STAEformer",
        "STGCNChebGraphConv",
        "STID",
        "STNorm",
    ):
        rows = grouped[model]
        aggregate.append(
            {
                "model": model,
                "seed_count": len(rows),
                "train_seconds_per_epoch": statistics.median(
                    row["median_final_ten_train_seconds"] for row in rows
                ),
                "recorded_log_span_minutes": statistics.median(
                    row["recorded_log_span_minutes"] for row in rows
                ),
            }
        )

    return {
        "schema_version": 1,
        "dataset": "METR-LA",
        "seeds": [43, 44, 45],
        "hardware_context": "single NVIDIA GeForce RTX 4090 reference setup",
        "aggregation": {
            "train_seconds_per_epoch": (
                "median across three per-seed medians; each per-seed median "
                "uses the final ten logged training epochs"
            ),
            "recorded_log_span_minutes": (
                "median across three per-seed spans between the first and last "
                "timestamped records in each selected canonical training log"
            ),
            "rounding": "display values rounded to two decimal places",
        },
        "source_boundary": {
            "raw_logs_in_public_repository": False,
            "public_values": (
                "the final-ten observations, boundary timestamps, per-seed "
                "statistics, and aggregate statistics are retained below"
            ),
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basicts-root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_payload(args.basicts_root.resolve())
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(output)


if __name__ == "__main__":
    main()
