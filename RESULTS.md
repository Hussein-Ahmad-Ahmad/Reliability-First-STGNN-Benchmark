# Benchmark Results

Compact metrics and analysis outputs are grouped by task. Plots are in [figures/results/](figures/results/); the [plot inventory](results/release/plot_index.json) records paths and checksums.

| Task | Artifacts | Protocol |
|---|---|---|
| Point forecasting | [Seed metrics](results/task1_point_forecasting/seed_metrics.json), [aggregates](results/task1_point_forecasting/multiseed_aggregation_clean.json), [statistics](results/task1_point_forecasting/statistical_reporting_summary.csv) | Seven configurations, three traffic datasets, three seeds; sample SD (`ddof=1`) |
| Naive anchors | [Persistence and seasonal-naive](results/task1_point_forecasting/naive_baselines.json) | Traffic loader boundaries can differ slightly |
| Ranking sensitivity | [Bootstrap outputs](results/task1_point_forecasting/) | Use `block_bootstrap_reconciled_*` and `block_bootstrap_sensitivity_reconciled_*`; flat-pooled sum/count aggregation, 72/144/288-origin blocks, A-minus-B differences |
| Calibration | [Normalized conformal](results/task2_uncertainty/conformal/), [plain control](results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json) | Dataset-specific ensembles; matched plain control on METR-LA only |
| Stochastic inference | [MC-Dropout](results/task2_uncertainty/mc_dropout_generated/) | Seven seed-43 configurations, 50 passes; MegaCRN and STNorm lack an active stochastic path in the evaluated configuration |
| Robustness | [Sensor zero-ablation](results/robustness/) | Fixed masks and a checkpoint-by-mask subset: 14 model-dataset-severity cells |
| Explanation | [XAI outputs](results/task3_explainability/), [degree-matched control](results/task3_explainability/degree_matched_control/) | Seven controlled cases, 30 random draws each; checkpoint-level diagnostics |
| Non-traffic illustration | [Chickenpox](results/nontraffic_graph_sanity/), [coverage drift](results/release/chickenpox_drift.json) | 21 public run arrays; separate selection and calibration sets; supplied representation was standardized over the source series |
| Compute | [Cost profile](results/flops_profile_results.json), [timing provenance](results/compute/) | Parameters/GFLOPs; archived epoch durations are not hardware-matched runtime rankings |

## Interpretation

- Three-seed intervals describe run variation. Bootstrap intervals are pointwise, not multiplicity-adjusted, and do not include seed uncertainty.
- Conformal coverage is an empirical diagnostic under dependent pooling, not an independent-element guarantee. Reliability coverage differs by dataset and configuration.
- Sensor zero-ablation is a controlled corruption; explanation diagnostics do not establish causal importance.
- Earlier bootstrap and MC-Dropout outputs remain labeled as historical. Use the current paths above rather than combining versions.

## Availability

Code, configurations, compact results, and plots are public. Large traffic prediction arrays and trained checkpoints are not distributed. Retraining supports protocol-level reproduction, not recovery of identical historical checkpoint bytes. Historical loading provenance remains incomplete.

The [SHA-256 manifest](ARTIFACT_MANIFEST.sha256) checks missing or changed listed public files. It does not certify completeness of unpublished data or independently establish scientific validity.
