# Benchmark Results

Compact metrics and analysis outputs are grouped by task. Plots are in [figures/results/](figures/results/); the [plot inventory](results/release/plot_index.json) records paths and checksums.

| Task | Artifacts | Protocol |
|---|---|---|
| Point forecasting | [Seed metrics](results/task1_point_forecasting/seed_metrics.json), [aggregates](results/task1_point_forecasting/multiseed_aggregation_clean.json), [statistics](results/task1_point_forecasting/statistical_reporting_summary.csv) | Seven configurations, three traffic datasets, three seeds; sample SD (`ddof=1`) |
| Naive anchors | [Persistence and seasonal-naive](results/task1_point_forecasting/naive_baselines.json) | Traffic loader boundaries can differ slightly |
| Ranking sensitivity | [Bootstrap outputs](results/task1_point_forecasting/) | Use `block_bootstrap_reconciled_*` and `block_bootstrap_sensitivity_reconciled_*`; flat-pooled sum/count aggregation, 72/144/288-origin blocks, A-minus-B differences |
| Calibration | [Normalized conformal](results/task2_uncertainty/conformal/), [plain control](results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json) | Dataset-specific ensembles; matched plain control on METR-LA only |
| Stochastic inference | [MC-Dropout](results/task2_uncertainty/mc_dropout_generated/) | Seven seed-43 configurations, 50 passes; coverage, width and variance share one valid-target mask; MegaCRN and STNorm return identical pass predictions |
| Robustness | [Sensor zero-ablation](results/robustness/) | Fixed masks and a checkpoint-by-mask subset: 14 model-dataset-severity cells |
| Explanation | [Ranking provenance](results/task3_explainability/ranking_provenance.json), [degree-matched control](results/task3_explainability/degree_matched_control/) | Ten cases, 30 draws per control; supplied-topology degree matching and sensor-input ablation, not edge-mask validation |
| Non-traffic illustration | [Chickenpox](results/nontraffic_graph_sanity/), [coverage drift](results/release/chickenpox_drift.json) | 21 public run arrays; separate selection and calibration sets; supplied representation was standardized over the source series |
| Compute | [Cost profile](results/flops_profile_results.json), [timing provenance](results/compute/) | Parameters/GFLOPs; archived epoch durations are not hardware-matched runtime rankings |
| Export validation | [53-case record](results/release/export_verification.json) | Complete chronological inference; validation-selected checkpoints; absolute MAE agreement within `1e-5` |
| Literature coding | [Paired assignments and agreement](results/literature_coding/) | Ten studies, nine dimensions; primary and Taha Benarbia's second assignments; 66/90 agreements, kappa 0.610 |

## Interpretation

- Three-seed intervals describe run variation. Bootstrap intervals are pointwise, not multiplicity-adjusted, and do not include seed uncertainty.
- Conformal coverage is an empirical diagnostic under dependent pooling, not an independent-element guarantee. Reliability coverage differs by dataset and configuration.
- Sensor zero-ablation is a controlled corruption; explanation diagnostics do not establish causal importance.
- Retained METR-LA rankings match learned sensor-input masks, not incident-edge sums. Historical forecast-change ratios and current test-MAE changes are different quantities. Archived Jaccard records mix cohort saliency and noise-response settings and are not a common same-horizon stability comparison.
- Earlier bootstrap and MC-Dropout outputs remain labeled as historical. Use the current paths above rather than combining versions.

## Availability

Code, configurations, compact results, coding assignments, and plots are public. Large traffic prediction arrays and trained checkpoints are not distributed. Retraining supports protocol-level reproduction, not recovery of identical historical checkpoint bytes. Current bootstrap inputs use verified chronological exports. The fixed six-member METR-LA ensemble is re-evaluated from identified checkpoint hashes; its historical training seeds and one original copied configuration remain unavailable, and the re-execution configuration is labeled explicitly.

The [SHA-256 manifest](ARTIFACT_MANIFEST.sha256) checks missing or changed listed public files. It does not certify completeness of unpublished data or independently establish scientific validity.
