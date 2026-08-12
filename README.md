# Reliability-First STGNN Benchmark

<p align="center">
  <img src="figures/readme/reliability_first_stgnn_banner.svg" width="100%" alt="Reliability-First STGNN Benchmark banner"/>
</p>

<p align="center">
  <strong>Reproducible traffic forecasting across accuracy, calibration, robustness, explanation, and compute.</strong>
</p>

<p align="center">
  <a href="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/actions/workflows/verify.yml"><img alt="Project checks" src="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/actions/workflows/verify.yml/badge.svg?branch=main"/></a>
  <a href="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.3.2"><img alt="Project state v1.3.2" src="https://img.shields.io/badge/project_state-v1.3.2-0f766e?style=flat-square"/></a>
  <a href="#quick-start"><img alt="Python 3.9" src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="#models"><img alt="Seven models" src="https://img.shields.io/badge/models-7-0891b2?style=flat-square"/></a>
  <a href="#benchmark-scope"><img alt="Three traffic datasets" src="https://img.shields.io/badge/traffic_datasets-3-f59e0b?style=flat-square"/></a>
  <a href="#license"><img alt="MIT License" src="https://img.shields.io/github/license/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark?style=flat-square&color=334155"/></a>
</p>

<p align="center">
  <a href="#overview">Overview</a> |
  <a href="#benchmark-scope">Scope</a> |
  <a href="#models">Models</a> |
  <a href="#result-gallery">Gallery</a> |
  <a href="#project-status">Status</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#citation">Citation</a>
</p>

<p align="center">
  Research artifact for <strong>Reliability-First Spatio-Temporal Graph Forecasting: A Survey and Traffic-Domain Benchmark for Calibration, Robustness, and Explanation Diagnostics</strong>
</p>

## Project Snapshot

The versioned tag `v1.3.2` identifies the public project state used for the
reported study. It includes evaluation code, compact results, run metadata,
focused tests, and a SHA-256 artifact manifest. This public package supports
code inspection, compact-artifact checks, and retraining-based reproduction.
Large traffic prediction arrays and trained checkpoint files are generated
locally when running the full training pipeline.

The clean-clone procedure is described in
[`REPRODUCIBILITY_CHECK.md`](REPRODUCIBILITY_CHECK.md) and automated by
[`scripts/verify_release.py`](scripts/verify_release.py). GitHub Actions runs
the same integrity and regression checks on pushes, pull requests, manual
dispatches, and a weekly schedule.

## Overview

This repository provides the code structure, configurations, figures, and compact result artifacts for a traffic-domain reliability benchmark of spatio-temporal graph neural network forecasting models. The benchmark compares models beyond point accuracy by adding dependence-aware ranking sensitivity, uncertainty calibration, robustness checks, explanation diagnostics, and computational cost.

<table>
  <tr>
    <td align="center" width="25%"><strong>7 models</strong><br/>Graph recurrent, convolutional, normalization, identity, and transformer families</td>
    <td align="center" width="25%"><strong>3 traffic datasets</strong><br/>METR-LA, PEMS-BAY, and PEMS04 under a shared benchmark protocol</td>
    <td align="center" width="25%"><strong>63 main traffic runs</strong><br/>Point-forecast training: 3 datasets x 7 models x 3 seeds</td>
    <td align="center" width="25%"><strong>Reliability diagnostics</strong><br/>Accuracy, ranking sensitivity, uncertainty, robustness, XAI, and compute</td>
  </tr>
</table>

## Project Status

| Signal | Current evidence |
|---|---|
| Project state | [`v1.3.2`](https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.3.2), the versioned code and artifact state |
| Project checks | [Automated checks](https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/actions/workflows/verify.yml) on every push and pull request, weekly, and on demand |
| Artifact integrity | SHA-256 checks plus completeness coverage for the tracked project tree |
| Regression coverage | Eight focused tests for conformal calibration, fixed sensor masks, archived summaries, and the Chickenpox protocol |
| Reproduction scope | Compact artifacts are public; full traffic prediction arrays and checkpoint files are produced by local retraining |

## Benchmark Scope

The main benchmark is intentionally focused on traffic forecasting. The secondary Chickenpox protocol illustration is kept separate from the traffic-domain ranking.

| Dataset | Role | Coverage |
|---|---|---|
| METR-LA | Primary diagnostic traffic dataset | Broadest diagnostic coverage |
| PEMS-BAY | Traffic-domain transfer check | Point forecasting and selected diagnostics |
| PEMS04 | Traffic-domain transfer check | Point forecasting, bootstrap sensitivity, and selected diagnostics |

## Models

| Model | Family |
|---|---|
| D2STGNN | Decoupled dynamic spatial-temporal GNN |
| MegaCRN | Memory-augmented recurrent graph model |
| MTGNN | Multivariate temporal graph neural network |
| STNorm | Spatial-temporal normalization baseline |
| STGCN-Cheb | Chebyshev spectral graph convolution baseline |
| STID | Lightweight spatial-temporal identity baseline |
| STAEformer | Adaptive embedding transformer |

## Result Gallery

<table>
  <tr>
    <td align="center" width="50%">
      <img src="figures/main/pf1_cross_dataset_mae.png" width="100%" alt="Cross-dataset MAE comparison"/><br/>
      <sub><strong>Point forecasting.</strong> Mean test MAE over seeds 43, 44, and 45.</sub>
    </td>
    <td align="center" width="50%">
      <img src="figures/appendix/uq2_conformal_cross_dataset.png" width="100%" alt="Cross-dataset conformal diagnostics"/><br/>
      <sub><strong>Calibration diagnostics.</strong> Fixed and per-horizon conformal results.</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <img src="figures/main/uq1_conformal_per_horizon_metrla.png" width="100%" alt="Conformal per-horizon diagnostics"/><br/>
      <sub><strong>Uncertainty quantification.</strong> Per-horizon conformal calibration diagnostics.</sub>
    </td>
    <td align="center" width="50%">
      <img src="figures/main/xai3_jaccard_stability_heatmap.png" width="100%" alt="XAI Jaccard stability heatmap"/><br/>
      <sub><strong>Explanation diagnostics.</strong> Cross-method stability and agreement patterns.</sub>
    </td>
  </tr>
</table>

## Results

### Point Forecasting

Mean test MAE over seeds 43, 44, and 45:

| Model | METR-LA | PEMS-BAY | PEMS04 |
|---|---:|---:|---:|
| D2STGNN | **2.878** | **1.513** | 18.393 |
| STAEformer | 2.942 | 1.573 | **18.222** |
| MegaCRN | 3.011 | 1.551 | 18.819 |
| MTGNN | 3.021 | 1.591 | 19.059 |
| STID | 3.119 | 1.563 | 18.419 |
| STNorm | 3.132 | 1.603 | 19.042 |
| STGCN-Cheb | 3.137 | 1.702 | 19.963 |

### Pairwise Ranking Sensitivity

Pairwise ranking sensitivity is reported through forecast-origin bootstrap
artifacts. The earlier flattened comparison record is retained only as a
historical note in
[`DM_WITHDRAWAL.md`](results/task1_point_forecasting/DM_WITHDRAWAL.md).

### Dependence-Aware Bootstrap

Forecast-origin block-bootstrap sensitivity checks are provided for METR-LA and PEMS04:

```text
results/task1_point_forecasting/block_bootstrap_pairwise_metr-la_seeds43-44-45.csv
results/task1_point_forecasting/block_bootstrap_pairwise_pems04_seeds43-44-45.csv
```

These are the study's dependence-aware pairwise checks. They
aggregate loss at the forecast-origin level and resample one-day blocks.

### Uncertainty Quantification

The conformal analysis uses a six-architecture METR-LA ensemble and
nine-member D2STGNN/MTGNN/STID ensembles on PEMS-BAY and PEMS04. Exact member
order, checkpoint-selection epochs, seeds where recorded, selection rules, and
generation code are stored under
[`results/task2_uncertainty/conformal/`](results/task2_uncertainty/conformal/).
The compact metrics and metadata are public. Full prediction arrays and
checkpoint files are generated by full local training runs; the compact
manifests state which public summaries each analysis uses.

Fixed-variant PICP/MPIW are 0.9056/23.31 on METR-LA, 0.9063/11.69 on
PEMS-BAY, and 0.9010/91.75 on PEMS04. Cross-dataset widths are descriptive
because target units, scales, and ensemble compositions differ.

```bash
python scripts/generate_conformal_intervals.py --help
python scripts/regenerate_cross_dataset_conformal.py --help
python scripts/generate_conformal_appendix_figures.py
```

### Sensor-Dropout Stress Test

The sensor-dropout artifact comes from checkpoint inference with nested fixed
seed-42 masks, seed-43 checkpoints, and clean-pass checks. Seed 43 is the
first fixed benchmark seed and serves uniformly as the deterministic reference;
alternative checkpoint seeds were not compared in this stress test.
The JSON records every zero-based sensor index plus configuration and checkpoint
hashes:

```text
results/robustness/sensor_dropout_fixed_masks_seed42.json
```

```bash
python pipelines/run_sensor_dropout.py --help
python scripts/generate_sensor_dropout_figure.py
```

### Graph-Native Non-Traffic Protocol Illustration

The secondary Chickenpox experiment uses the same seven model classes with
compact dimensions on a 20-node weekly graph. It uses 286 training, 30
validation, 50 calibration, and 99 unchanged test origins. Eleven forecast
origins are omitted at each boundary so the 12-week target periods do not
overlap. Checkpoint selection uses validation only; coordinate-wise 90%
intervals use the dedicated calibration period with finite-sample rank 46.
The split, graph preprocessing, optimization settings, per-seed best epochs,
and compact check arrays are recorded in
[`chickenpox_protocol_manifest.json`](results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json).
The dataset-provided `FX` matrix was standardized independently by county over
all 521 source weeks. Only the experiment's additional scalar transform is
estimated from training targets. Evaluation reverses that additional transform,
so MAE, RMSE, and interval width remain in standardized `FX` signal units rather
than weekly case counts.

### Explanation Diagnostics

XAI artifacts include GNNExplainer deletion fidelity, perturbation stability, Integrated Gradients, attention diagnostics, and cross-method agreement. The METR-LA MTGNN case study and reduced cross-dataset transfer summary are stored in:

```text
results/task3_explainability/case_studies/
```

## Compute Environment

The main traffic benchmark was run as a multi-seed GPU experiment. The
canonical timing logs support the extracted wall-clock measurements; timing
values should therefore be read as setup-dependent. Post-hoc analysis scripts
can be run on CPU when prediction and result artifacts are already available.

| Item | Public package usage |
|---|---|
| Timing context | Wall-clock values are setup-dependent |
| Reproduction environment | Python 3.9; PyTorch >= 2.0; pinned core packages are listed in `requirements.txt` and `environment.yml` |
| Main framework | BasicTS + EasyTorch 1.3.3 |
| Python stack | NumPy 1.24.4, TensorBoard 2.18.0, PyG >= 2.3.0, SciPy >= 1.10, Captum >= 0.6 |
| Main traffic point-forecast training | 63 runs: 3 datasets x 7 models x 3 seeds, 100 epochs per run |
| Auxiliary executions | UQ, robustness, XAI, profiling, bootstrap, and Chickenpox experiments are reported separately |
| Seeds | 43, 44, 45 |
| Forecasting setting | 12 input steps to 12 output steps |
| Bootstrap / figures | Post-hoc artifact scripts; GPU not required with stored results |

Wall-clock time depends on dataset storage, dataloader settings, GPU
availability, and whether checkpoints or prediction dumps are already present.
The METR-LA table aggregation and per-seed observations are archived in
[`METR-LA_runtime_provenance.json`](results/compute/METR-LA_runtime_provenance.json).

## Repository Layout

```text
configs/                         Model/dataset/seed experiment configs
datasets/                        Dataset metadata and download notes
figures/
  main/                          Main study figures
  appendix/                      Supplementary diagnostic figures
  readme/                        README visual assets
models/                          Local model architecture implementations
pipelines/                       End-to-end task entry points
results/
  task1_point_forecasting/       Point forecasting and bootstrap artifacts
  robustness/                    Sensor-dropout metrics and mask metadata
  task2_uncertainty/             UQ and conformal artifacts
  task3_explainability/          XAI and stability artifacts
  nontraffic_graph_sanity/       Chickenpox protocol illustration artifacts
  compute/                       Runtime aggregation provenance
scripts/                         Reproduction, figure, and diagnostic scripts
src/                             Shared reliability utilities
```

## Quick Start

The repository is organized so readers can inspect the published artifacts without retraining the traffic models. For local script execution, create the environment with either:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate stgnn-benchmark
```

Full traffic-model retraining requires the original datasets, checkpoint storage, and GPU resources.

Check the current checkout, required artifact families, complete SHA-256
manifest, and focused tests together:

```bash
python scripts/verify_release.py --expect-tests 8
```

For the exact versioned project state:

```bash
git switch --detach v1.3.2
python scripts/verify_release.py --expect-tag v1.3.2 --expect-manifest-entries 380 --expect-tests 8 --require-clean
```

## Automated Checks

The [project-check workflow](.github/workflows/verify.yml) runs on
every push and pull request, each Monday, and by manual dispatch. It installs a
minimal CPU test environment, checks that the manifest covers the complete
tracked project tree, checks every recorded SHA-256 digest, and runs the
focused test suite. Each run publishes a machine-readable summary as a workflow
artifact.

## Key Artifacts

| Area | Location |
|---|---|
| Point forecasting and bootstrap results | [`results/task1_point_forecasting/`](results/task1_point_forecasting/) |
| Uncertainty and conformal diagnostics | [`results/task2_uncertainty/`](results/task2_uncertainty/) |
| Sensor-dropout inference and mask metadata | [`pipelines/run_sensor_dropout.py`](pipelines/run_sensor_dropout.py), [`results/robustness/`](results/robustness/) |
| XAI summaries and case-study artifacts | [`results/task3_explainability/`](results/task3_explainability/) |
| Runtime aggregation provenance | [`results/compute/METR-LA_runtime_provenance.json`](results/compute/METR-LA_runtime_provenance.json) |
| Non-traffic protocol illustration | [`results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json`](results/nontraffic_graph_sanity/chickenpox_protocol_manifest.json) |
| Main study figures | [`figures/main/`](figures/main/) |
| Reproduction and utility scripts | [`scripts/`](scripts/) |
| Checksum manifest | [`ARTIFACT_MANIFEST.sha256`](ARTIFACT_MANIFEST.sha256) |
| Clean-clone guide | [`REPRODUCIBILITY_CHECK.md`](REPRODUCIBILITY_CHECK.md) |

## Citation

GitHub exposes the repository's citation through [`CITATION.cff`](CITATION.cff).

```bibtex
@software{ahmad2026reliability,
  title  = {Reliability-First Spatio-Temporal Graph Forecasting: A Survey and
            Traffic-Domain Benchmark for Calibration, Robustness, and
            Explanation Diagnostics},
  author = {Ahmad, Hussein Ahmad and Mortazavi, Seyyed Kasra and Benarbia, Taha
            and Al Machot, Fadi and Kyamakya, Kyandoghere},
  year   = {2026},
  version = {1.3.2},
  url    = {https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark}
}
```

## License

This project is released under the MIT License. See `LICENSE`.
