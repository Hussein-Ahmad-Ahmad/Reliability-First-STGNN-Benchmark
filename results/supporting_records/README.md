# Benchmark Supporting Records

Supporting tables, data and analysis scripts for the benchmark snapshot
[v1.4.4](https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.4.4).

## Files

| Material | Files |
|---|---|
| Reader-facing Tables S1 and S2 | [Supporting Tables.pdf](Supporting%20Tables.pdf) |
| Ten sensor-ablation controls and their source hashes | [Table S1.csv](Table%20S1.csv) |
| Aggregate and horizon-wise calibration-boundary results | [Table S2 all horizons.csv](Table%20S2%20all%20horizons.csv), [calculation record](calibration_boundary_sensitivity.json) |
| Comparison of 27 checkpoint states | [checkpoint_states.json](checkpoint_states.json) |
| Literature codes and synthesis accounting | [RER evidence map.csv](RER%20evidence%20map.csv), [Synthesis sources.csv](Synthesis%20sources.csv), [Screening accounting.csv](Screening%20accounting.csv) |
| Figure inputs and plotting scripts | [figure_generation.json](figure_generation.json), [render_selected_figures.py](render_selected_figures.py), [ig_overlap_figure.json](ig_overlap_figure.json), [render_ig_overlap.py](render_ig_overlap.py) |
| Array and checkpoint calculations | [calibration_boundary_sensitivity.py](calibration_boundary_sensitivity.py), [check_checkpoint_states.py](check_checkpoint_states.py) |

## Measurement Scope

Table S1 reports target-MAE changes on epoch-100 checkpoints, with a clean
reference for each checkpoint. Its ranks are descriptive positions among 30
random controls. Sample-specific XAI tests instead measure forecast changes
on each model's output scale using inputs passed directly from the data loader.
These procedures use different input processing and response definitions.

Table S2 keeps calibration and thresholds fixed and excludes 11 evaluation
origins whose targets are all masked. Both evaluations retain the same
7,118,436 valid elements, so coverage and width are unchanged. This checks
boundary accounting rather than the effect of removing valid observations.

## Prediction Records

The [53 benchmark exports](../release/export_verification.json) and the
[separate six-member METR-LA conformal exports](../task2_uncertainty/conformal/METR-LA_ensemble_manifest.json)
have their own online-MAE agreement checks within `1e-5`. PEMS-BAY and PEMS04
conformal predictions were collected directly from inference batches.

Earlier offline exports had a final-batch indexing mismatch affecting
94/6,831 METR-LA origins (1.38%), 64/10,400 PEMS-BAY origins (0.62%), and
94/3,375 PEMS04 origins (2.79%). The reported bootstrap and METR-LA conformal
analyses use complete, origin-aligned records instead. Point-forecast tables
use the online measurements.

## Verification

From the repository root, using Python 3.9 or later:

```text
python results/supporting_records/verify_records.py
```

This standard-library check verifies file hashes, control arithmetic, boundary
record consistency, checkpoint summaries and literature counts.

Plotting uses NumPy, Matplotlib and the frozen `Evaluation-Artifacts-v1.4.4.zip`
release asset. Array recalculation uses Python 3.12 and NumPy 2.4.4. Checkpoint
comparison uses PyTorch. Each script's `--help` lists the required paths.

Full traffic prediction arrays and weights remain internal. The array and
checkpoint scripts require those files, while the compact-record verifier and
plotting scripts use public inputs. `SHA256SUMS.txt` covers this folder, and
`ARTIFACT_MANIFEST.sha256` covers the tracked repository.
