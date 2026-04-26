# Reliability-First Spatio-Temporal Graph Forecasting: A Survey and Traffic-Domain Benchmark for Calibration, Robustness, and Explanation Diagnostics

**Authors:** Hussein Ahmad Ahmad¹\*, Seyyed Kasra Mortazavi¹, Taha Benarbia², Fadi Al Machot³, and Kyandoghere Kyamakya¹

> ¹ University of Klagenfurt · ² [Affiliation 2] · ³ [Affiliation 3]  
> \* Corresponding author

A reproducible benchmark evaluating **reliability** (calibration, robustness, and explainability) of Spatio-Temporal Graph Neural Networks for traffic forecasting.

## Overview

This benchmark evaluates 7 state-of-the-art STGNN models across three complementary dimensions:

| Task | Dimension | Methods |
|------|-----------|---------|
| **Task 1** | Accuracy & Statistical Significance | Multi-seed (seeds 43/44/45), Diebold-Mariano test (21 pairs, Holm-Bonferroni) |
| **Task 2** | Uncertainty Quantification | MC Dropout (50-pass), Deep Ensemble (3 seeds), Conformal Prediction (fixed + per-horizon) |
| **Task 3** | Explainability | GNNExplainer + Jaccard Stability, Integrated Gradients, Temporal Attention, Cross-method Agreement |

### Models

| Model | Architecture | Key Feature |
|-------|-------------|-------------|
| D2STGNN | Diffusion + Spectral GNN | Decoupled spatial-temporal |
| MegaCRN | Memory-augmented GCN + RNN | Adaptive memory |
| MTGNN | Multi-scale temporal GNN | Dilated convolutions |
| STNorm | Spatial-Temporal Norm | Instance + group normalization |
| STGCNChebGraphConv | Chebyshev spectral GCN | Spectral convolution |
| STID | Spatial-Temporal Identity | Lightweight MLP-based |
| STAEformer | Adaptive Embedding Transformer | Attention with learned node embeddings |

### Datasets

| Dataset | Sensors | Time Steps | Freq | Location |
|---------|---------|-----------|------|----------|
| METR-LA | 207 | ~34,272 | 5 min | Los Angeles, CA |
| PEMS-BAY | 325 | ~52,116 | 5 min | San Francisco Bay Area, CA |
| PEMS04 | 307 | ~16,992 | 5 min | Guangdong Province, China |

---

## Project Structure

```
STGNN-Reliability-Benchmark/
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ framework/basicts/          # BasicTS training framework (vendored)
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ models/                     # 7 model architecture implementations
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ D2STGNN/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ MegaCRN/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ MTGNN/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ STNorm/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ STGCNChebGraphConv/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ STID/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ STAEformer/arch/
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ configs/                    # 63 training configs (7 ÃƒÆ’Ã¢â‚¬â€ 3 datasets ÃƒÆ’Ã¢â‚¬â€ 3 seeds)
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ {MODEL}/{DATASET}_seed{N}.py
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ datasets/                   # METR-LA, PEMS-BAY, PEMS04 metadata
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ {DATASET}/{adj_mx.pkl, desc.json, meta.json, *.npy}
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ checkpoints/                # Trained model weights (63 checkpoints)
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ {MODEL}/{DATASET}_seed{N}/best_model.pt
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ src/                        # Unified source code
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ uncertainty/            # MC Dropout, Ensemble, Conformal
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ explainability/         # GNNExplainer, IG, Attention
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ pipelines/                  # End-to-end pipeline entry points
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task1_run.py            # Training + DM tests
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task2_run.py            # UQ evaluation
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task3_run.py            # XAI evaluation
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ results/                    # Pre-computed results (all JSON)
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task1_point_forecasting/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task2_uncertainty/
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬Å¡   ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ task3_explainability/
ÃƒÂ¢Ã¢â‚¬ÂÃ…â€œÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ scripts/                    # Utility scripts
ÃƒÂ¢Ã¢â‚¬ÂÃ¢â‚¬ÂÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ÃƒÂ¢Ã¢â‚¬ÂÃ¢â€šÂ¬ docs/                       # Methodology and results documentation
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download datasets

The `datasets/` directory contains metadata files. Download the actual `.npy` data files:

```
datasets/METR-LA/{train,val,test}_data.npy    (207 sensors)
datasets/PEMS-BAY/{train,val,test}_data.npy   (325 sensors)
datasets/PEMS04/{train,val,test}_data.npy     (307 sensors)
```

### 3. Reproduce results

**Option A: Use pre-computed results** (fastest)

All results are in `results/`. See `docs/results_summary.md` for interpretation.

**Option B: Re-run training**

```bash
# Train all 63 configurations (requires GPUs, ~72 GPU-hours)
python pipelines/task1_run.py --mode train

# Or train a single config
python pipelines/task1_run.py --mode train --model D2STGNN --dataset METR-LA --seed 43
```

**Option C: Re-run evaluation only** (requires checkpoints)

```bash
# UQ evaluation
python pipelines/task2_run.py --method all

# XAI evaluation
python pipelines/task3_run.py --method all
```

---

## Pre-Computed Results

All experimental results are in `results/` as JSON files.

### Task 1: Point Forecasting

- `results/task1_point_forecasting/multiseed_aggregation_clean.json` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Mean Ãƒâ€šÃ‚Â± std across seeds
- `results/task1_point_forecasting/dm_full_21pairs_holm_corrected.json` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Full DM test matrix

### Task 2: Uncertainty Quantification

- `results/task2_uncertainty/mc_dropout/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 5 models, METR-LA, 50-pass MC Dropout
- `results/task2_uncertainty/deep_ensemble/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â All models, multi-seed ensemble UQ
- `results/task2_uncertainty/conformal/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â 3 datasets ÃƒÆ’Ã¢â‚¬â€ 2 variants (fixed + per-horizon)

### Task 3: Explainability

- `results/task3_explainability/gnnexplainer/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Fidelity (k=5,10,20,50) + stability per model
- `results/task3_explainability/integrated_gradients/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Attribution maps for 7 models (METR-LA)
- `results/task3_explainability/attention/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Temporal attention analysis for 7 models
- `results/task3_explainability/jaccard_stability/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â Top-K stability for all 7 models
- `results/task3_explainability/cross_dataset/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â GNNExplainer on PEMS-BAY + PEMS04
- `results/task3_explainability/statistical_tests/` ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â IG vs GNN cross-method agreement

---

## Known Gaps

| Gap | Description | Status |
|-----|-------------|--------|
| IG cross-dataset | IG only computed for METR-LA | Intentionally scoped; GNNExplainer covers all 3 datasets |

---

## Citation

If you use this benchmark, please cite:

```bibtex
@article{stgnn_reliability_2025,
  title   = {Reliability Benchmark for Spatio-Temporal Graph Neural Networks},
  author  = {...},
  journal = {...},
  year    = {2025}
}
```

---

## License

See `LICENSE` for details.
