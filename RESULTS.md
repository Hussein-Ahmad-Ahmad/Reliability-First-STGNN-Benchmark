# Results Index

The current result set follows the supplied 35-page PDF. Its SHA-256 fingerprint and exact embedded figure pixels are recorded in [figure_index.json](results/release/figure_index.json). Figures were exported without redrawing. The PDF, author portraits, editorial files, and large traffic arrays are not distributed here.

## Numerical Results

| Tables / figures | Current artifacts | Scope |
|---|---|---|
| Tables 6, 18 | [Cost profile](results/flops_profile_results.json), [configurations](configs/) | Parameters/GFLOPs and graph/scaling provenance; epoch durations are not hardware-matched runtime rankings |
| Table 7 | [Naive baselines](results/task1_point_forecasting/naive_baselines.json) | Persistence and seasonal-naive anchors; traffic loader boundaries can differ slightly |
| Tables 8, 9; Figures 4, A.1 | [Seed aggregation](results/task1_point_forecasting/multiseed_aggregation_clean.json), [statistics](results/task1_point_forecasting/statistical_reporting_summary.csv) | 21 model-dataset summaries, sample SD (`ddof=1`); three-seed intervals describe run variation |
| Table 10 | `results/task1_point_forecasting/block_bootstrap_reconciled_*.json` | Flat-pooled sum/count resampling; artifacts use **A minus B**, while the table uses **B minus A**; negate and reverse CI endpoints |
| Table 11 | `results/task1_point_forecasting/block_bootstrap_sensitivity_reconciled_*.json` | 72/144/288 origins; 17/21, 9/10, 19/21 intervals exclude zero on METR-LA, PEMS-BAY, PEMS04 respectively; pointwise, not multiplicity-adjusted |
| Table 12 | [Plain conformal control](results/task2_uncertainty/conformal_sigma_control/METR-LA_conformal_sigma_control.json) | METR-LA six-member ensemble only; no matched plain control is asserted for the other datasets |
| Figures 5, 6, A.2-A.6 | [Normalized conformal metrics](results/task2_uncertainty/conformal/) | Dataset-specific pipelines, empirical marginal/horizon coverage; no independent-element guarantee |
| Table 13; Figure 7 | [Fixed masks](results/robustness/sensor_dropout_fixed_masks_seed42.json), `results/robustness/sensor_dropout_additional_mask_seeds*.json` | Checkpoints 43/44/45 crossed with masks 7/11/19/23/31; D2STGNN and STID on all datasets, MegaCRN on METR-LA: 14 model-dataset-severity cells |
| Tables 14-16; Figures 8-10, A.8-A.13 | [XAI summaries](results/task3_explainability/), [degree-matched controls](results/task3_explainability/degree_matched_control/) | Seven degree-matched cases, 30 draws each; checkpoint-level diagnostics, not causal importance |
| Tables A.1, A.2 | [Chickenpox summary](results/nontraffic_graph_sanity/chickenpox_all_baselines_summary.json), [protocol](results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json), [drift](results/release/chickenpox_drift.json) | 21 public run arrays; 286/30/50/99 split; supplied FX representation was standardized over the source series |
| Table A.7; Figure A.7 | [MC-Dropout summaries](results/task2_uncertainty/mc_dropout_generated/) | Seven seed-43 configurations, 50 passes; MegaCRN and STNorm lack an active stochastic path in the evaluated configuration |
| Tables A.5, A.8 | [Evidence map](results/release/literature_evidence_map.csv), [coding summary](results/release/evidence_coding_summary.json) | Publication-reported 66/90 agreement and kappa 0.610; five reconciled changes. Raw independent coding sheets are not included |

Tables 1-4, 17 and A.3-A.6 also contain descriptive classifications, evidence rules, coverage maps, and source notes rather than additional model experiments. The literature map is purposive, not a prevalence estimate. Historical source-level search dates and individual screening transitions were not retained. The historical-target comparison reports 94/6831, 64/10400, and 94/3375 differing origins; exact historical loading provenance remains unavailable.

## Figure Gallery

All 10 main and 13 appendix figures are in [figures/results/](figures/results/), with numbers, captions, panel paths, source pages, and checksums in the [figure index](results/release/figure_index.json). Figure A.13 has two panels. Older assets under `figures/main/` and `figures/appendix/` remain historical plotting assets; the indexed exports are the current numbered set.

## Version Boundaries

`v1.3.2` remains an earlier snapshot. Its 380-entry manifest and eight-test statement do not describe the expanded `v1.4.0` release. Earlier bootstrap and MC-Dropout files remain for compatibility, with their historical status marked in their directories. Use the current paths above for comparisons.

SHA-256 checks detect missing or changed **listed public artifacts**. They do not establish completeness of unpublished arrays, independently validate scientific conclusions, or recreate historical checkpoint bytes through retraining.
