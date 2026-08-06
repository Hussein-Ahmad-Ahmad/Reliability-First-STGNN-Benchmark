"""Verify the public release manifest, focused tests, and Git identity."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REQUIRED_PATHS = (
    "ARTIFACT_MANIFEST.sha256",
    "README.md",
    "REPRODUCIBILITY_CHECK.md",
    "scripts/generate_artifact_manifest.py",
    "scripts/generate_conformal_intervals.py",
    "scripts/regenerate_cross_dataset_conformal.py",
    "pipelines/run_sensor_dropout.py",
    "results/task1_point_forecasting",
    "results/task2_uncertainty/conformal",
    "results/robustness",
    "results/task3_explainability",
    "results/compute/METR-LA_runtime_provenance.json",
    "results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json",
    "results/nontraffic_graph_sanity/chickenpox_protocol_arrays.npz",
    "results/nontraffic_graph_sanity/chickenpox_run_artifacts",
)


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def git(*args: str) -> str:
    return run(["git", *args]).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-tag")
    parser.add_argument("--expect-commit")
    parser.add_argument("--expect-manifest-entries", type=int)
    parser.add_argument("--expect-tests", type=int)
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Write the final machine-readable summary to this path.",
    )
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="Fail if the Git worktree differs from the checked-out commit.",
    )
    args = parser.parse_args()

    missing = [
        relative
        for relative in REQUIRED_PATHS
        if not (ROOT / relative).exists()
    ]
    if missing:
        raise SystemExit("Missing required release paths:\n" + "\n".join(missing))

    head = git("rev-parse", "HEAD")
    tags = sorted(filter(None, git("tag", "--points-at", "HEAD").splitlines()))
    if args.expect_commit and head != args.expect_commit:
        raise SystemExit(
            f"Commit mismatch: expected {args.expect_commit}, found {head}"
        )
    if args.expect_tag and args.expect_tag not in tags:
        raise SystemExit(
            f"Tag {args.expect_tag!r} does not point at HEAD; found {tags}"
        )
    if args.require_clean:
        status = git("status", "--porcelain")
        if status:
            raise SystemExit("Worktree is not clean:\n" + status)

    manifest_check = run(
        [
            sys.executable,
            "scripts/generate_artifact_manifest.py",
            "--check",
        ]
    )
    print(manifest_check.stdout.strip())
    manifest_entries = sum(
        1
        for line in (ROOT / "ARTIFACT_MANIFEST.sha256").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    )
    if (
        args.expect_manifest_entries is not None
        and manifest_entries != args.expect_manifest_entries
    ):
        raise SystemExit(
            "Manifest-entry mismatch: "
            f"expected {args.expect_manifest_entries}, found {manifest_entries}"
        )

    sys.path.insert(0, str(ROOT))
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    test_count = suite.countTestCases()
    if args.expect_tests is not None and test_count != args.expect_tests:
        raise SystemExit(
            f"Test-count mismatch: expected {args.expect_tests}, found {test_count}"
        )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit("Focused regression tests failed")

    summary = {
        "status": "passed",
        "commit": head,
        "tags_at_head": tags,
        "manifest_entries": manifest_entries,
        "tests_run": result.testsRun,
    }
    rendered = json.dumps(summary, indent=2)
    print(rendered)
    if args.json_output:
        output = args.json_output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
        print(f"Wrote verification summary to {output}")


if __name__ == "__main__":
    main()
