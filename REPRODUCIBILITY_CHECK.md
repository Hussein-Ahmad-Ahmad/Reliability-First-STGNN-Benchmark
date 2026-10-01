# Release Verification

This guide verifies the identity and integrity of the public code-and-artifact
snapshot. It checks the Git revision, required artifact families, SHA-256
manifest, and sixteen focused regression tests. The manifest describes tracked
release files, not unpublished local working files.

## Clean-Clone Check

```bash
git clone --branch v1.4.1 --depth 1 https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark.git
cd Reliability-First-STGNN-Benchmark
python -m pip install numpy==1.24.4
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python scripts/verify_release.py --expect-tag v1.4.1 --expect-tests 16 --require-clean
```

To check a specific full commit identifier as well, append
`--expect-commit <full-commit-hash>` to the verification command. A successful
run prints a JSON summary containing the checked commit, tag, manifest-entry
count, and test count.

## Current Main Branch

Verify the current tracked release tree with:

```bash
python scripts/verify_release.py --expect-tests 16
```

The same command runs automatically through
[`.github/workflows/verify.yml`](.github/workflows/verify.yml) on pushes, pull
requests, a weekly schedule, and manual dispatch.

## Scope

The public snapshot supports source inspection, compact-artifact verification,
and retraining-based reproduction. Large traffic prediction arrays and trained
checkpoints are not distributed in this Git repository, so the verifier does
not claim direct regeneration from those internally archived artifacts.
