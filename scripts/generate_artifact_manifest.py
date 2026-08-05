"""Generate or verify the SHA-256 manifest for the public release tree."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "ARTIFACT_MANIFEST.sha256"
RELEASE_ROOTS = (
    "configs/",
    "datasets/",
    "figures/",
    "framework/",
    "models/",
    "pipelines/",
    "results/",
    "scripts/",
    "src/",
    "tests/",
)
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "LICENSE",
    "README.md",
    "REPRODUCIBILITY_CHECK.md",
    "environment.yml",
    "requirements.txt",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_release_path(relative: str) -> bool:
    return relative in ROOT_FILES or relative.startswith(RELEASE_ROOTS)


def discover_release_files(output: Path) -> list[Path]:
    try:
        completed = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        relative_paths = completed.stdout.splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError):
        relative_paths = [
            path.relative_to(ROOT).as_posix()
            for prefix in RELEASE_ROOTS
            for path in (ROOT / prefix).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        relative_paths.extend(
            name for name in ROOT_FILES if (ROOT / name).is_file()
        )

    output_relative = output.resolve().relative_to(ROOT).as_posix()
    paths = {
        ROOT / relative
        for relative in relative_paths
        if is_release_path(relative)
        and relative != output_relative
        and (ROOT / relative).is_file()
        and "__pycache__" not in Path(relative).parts
    }
    return sorted(paths, key=lambda path: path.relative_to(ROOT).as_posix())


def generate(output: Path) -> None:
    lines = [
        f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}"
        for path in discover_release_files(output)
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(lines)} entries to {output}")


def verify(manifest: Path) -> None:
    failures = []
    checked = 0
    for line_number, line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"line {line_number}: malformed")
            continue
        path = ROOT / relative
        checked += 1
        if not path.is_file():
            failures.append(f"{relative}: missing")
        elif sha256(path) != expected:
            failures.append(f"{relative}: checksum mismatch")
    if failures:
        raise SystemExit(
            "Manifest verification failed:\n" + "\n".join(failures)
        )
    print(f"Verified {checked} files from {manifest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify an existing manifest instead of regenerating it",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if args.check:
        verify(output)
    else:
        generate(output)


if __name__ == "__main__":
    main()
