"""Inventory the benchmark result plots and their SHA-256 checksums."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    entries = []
    for path in sorted((ROOT / "figures/results").glob("*.png")):
        entries.append({"name": path.stem,
                        "path": path.relative_to(ROOT).as_posix(),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    output = ROOT / "results/release/plot_index.json"
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump({"plots": entries}, handle, indent=2)
        handle.write("\n")
    print(f"Indexed {len(entries)} benchmark plots")


if __name__ == "__main__":
    main()
