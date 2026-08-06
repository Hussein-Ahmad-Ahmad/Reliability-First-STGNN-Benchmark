# Release Verification

This guide verifies the identity and integrity of the public code-and-artifact
snapshot. It checks the Git revision, required artifact families, SHA-256
manifest, and eight focused regression tests.

## Clean-Clone Check

```bash
git clone --branch v1.3.0 --depth 1 https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark.git
cd Reliability-First-STGNN-Benchmark
python scripts/verify_release.py --expect-tag v1.3.0 --expect-manifest-entries 378 --expect-tests 8 --require-clean
```

To check the manuscript's full commit identifier as well, append
`--expect-commit <full-commit-hash>` to the verification command. A successful
run prints a JSON summary containing the checked commit, tag, manifest-entry
count, and test count.

## Scope

The public snapshot supports source inspection, compact-artifact verification,
and retraining-based reproduction. Large traffic prediction arrays and trained
checkpoints are not distributed in this Git repository, so the verifier does
not claim direct regeneration from those internally archived artifacts.
