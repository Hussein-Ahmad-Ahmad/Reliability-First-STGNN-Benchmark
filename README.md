# Reliability-First STGNN Benchmark

<p align="center">
  <img src="figures/readme/reliability_first_stgnn_banner.svg" width="100%" alt="Reliability-First STGNN Benchmark banner"/>
</p>

<p align="center">
  <strong>Traffic forecasting across accuracy, calibration, robustness, explanation, and compute.</strong>
</p>

<p align="center">
  <a href="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/actions/workflows/verify.yml"><img alt="Project checks" src="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/actions/workflows/verify.yml/badge.svg?branch=main"/></a>
  <a href="https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.3.2"><img alt="Project state v1.3.2" src="https://img.shields.io/badge/project_state-v1.3.2-0f766e?style=flat-square"/></a>
  <a href="#quick-start"><img alt="Python 3.9" src="https://img.shields.io/badge/Python-3.9-3776AB?style=flat-square&logo=python&logoColor=white"/></a>
  <a href="#license"><img alt="MIT License" src="https://img.shields.io/github/license/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark?style=flat-square&color=334155"/></a>
</p>

## Overview

This project contains code, configuration files, figures, and compact result
artifacts for a traffic-domain study of spatio-temporal graph neural network
forecasting. The evaluation covers point accuracy, ranking sensitivity,
uncertainty calibration, sensor-dropout robustness, explanation diagnostics,
and compute summaries.

| Scope | Details |
|---|---|
| Models | D2STGNN, MegaCRN, MTGNN, STNorm, STGCN-Cheb, STID, STAEformer |
| Traffic datasets | METR-LA, PEMS-BAY, PEMS04 |
| Main traffic runs | 3 datasets x 7 models x 3 seeds |
| Versioned state | [`v1.3.2`](https://github.com/Hussein-Ahmad-Ahmad/Reliability-First-STGNN-Benchmark/tree/v1.3.2) |
| Automated checks | SHA-256 manifest coverage plus eight focused tests |

## Snapshot

The `v1.3.2` project state includes compact metrics, run metadata, plotting
assets, utility scripts, and a 380-entry SHA-256 manifest. Full traffic-model
training produces large checkpoint and prediction files locally; the public
tree keeps compact artifacts for inspection and reproducible post-processing.

```bash
python scripts/verify_release.py --expect-tests 8
```

For the versioned project state:

```bash
git switch --detach v1.3.2
python scripts/verify_release.py --expect-tag v1.3.2 --expect-manifest-entries 380 --expect-tests 8 --require-clean
```

## Results At A Glance

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

<table>
  <tr>
    <td align="center" width="50%">
      <img src="figures/main/pf1_cross_dataset_mae.png" width="100%" alt="Cross-dataset MAE comparison"/><br/>
      <sub><strong>Point forecasting.</strong> Mean test MAE across fixed seeds.</sub>
    </td>
    <td align="center" width="50%">
      <img src="figures/appendix/uq2_conformal_cross_dataset.png" width="100%" alt="Cross-dataset conformal diagnostics"/><br/>
      <sub><strong>Calibration.</strong> Fixed and per-horizon conformal summaries.</sub>
    </td>
  </tr>
</table>

## Key Locations

| Area | Path |
|---|---|
| Configurations | [`configs/`](configs/) |
| Main figures | [`figures/main/`](figures/main/) |
| Point forecasting and bootstrap summaries | [`results/task1_point_forecasting/`](results/task1_point_forecasting/) |
| Uncertainty and conformal summaries | [`results/task2_uncertainty/`](results/task2_uncertainty/) |
| Sensor-dropout summaries | [`results/robustness/`](results/robustness/) |
| Explanation summaries | [`results/task3_explainability/`](results/task3_explainability/) |
| Secondary graph-native protocol illustration | [`results/nontraffic_graph_sanity/`](results/nontraffic_graph_sanity/) |
| Runtime summaries | [`results/compute/`](results/compute/) |
| Utility scripts | [`scripts/`](scripts/) |
| Checksum manifest | [`ARTIFACT_MANIFEST.sha256`](ARTIFACT_MANIFEST.sha256) |

## Quick Start

Create a local environment:

```bash
pip install -r requirements.txt
```

or:

```bash
conda env create -f environment.yml
conda activate stgnn-benchmark
```

Full traffic-model training requires the original datasets, checkpoint storage,
and GPU resources. Compact artifact checks and plotting utilities can be run
from the public tree.

## Citation

GitHub exposes the repository citation through [`CITATION.cff`](CITATION.cff).

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

This project is released under the MIT License. See [`LICENSE`](LICENSE).
