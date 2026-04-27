# Datasets

This directory contains **metadata only**. The raw traffic speed `.npy` files must be
downloaded separately from the original sources due to their size.

The exact split files and metadata used in the manuscript are provided in each dataset
folder through `meta.json` and the corresponding configuration files.

---

## Expected directory structure

```
datasets/
|-- METR-LA/
|   |-- train_data.npy      # shape: (24009, 207, 2)
|   |-- val_data.npy        # shape: (3426, 207, 2)
|   |-- test_data.npy       # shape: (6850, 207, 2)
|   |-- adj_mx.pkl          # adjacency matrix (included)
|   |-- desc.json           # dataset description (included)
|   +-- meta.json           # split ratios and normalization config (included)
|-- PEMS-BAY/
|   |-- train_data.npy      # shape: (36469, 325, 2)
|   |-- val_data.npy        # shape: (5209, 325, 2)
|   |-- test_data.npy       # shape: (10419, 325, 2)
|   |-- adj_mx.pkl
|   |-- desc.json
|   +-- meta.json
+-- PEMS04/
    |-- train_data.npy      # shape: (10181, 307, 3)
    |-- val_data.npy        # shape: (3394, 307, 3)
    |-- test_data.npy       # shape: (3394, 307, 3)
    |-- adj_mx.pkl
    |-- desc.json
    +-- meta.json
```

---

## Download instructions

### METR-LA and PEMS-BAY

Originally released with DCRNN (Li et al., ICLR 2018).
Download from the official DCRNN repository:

```
https://github.com/liyaguang/DCRNN
```

The files `metr-la.h5` and `pems-bay.h5` can be converted to `.npy` using the
preprocessing scripts provided in the BasicTS framework (`framework/basicts/`).

### PEMS04

Originally released with STSGCN (Song et al., AAAI 2020).
Download from:

```
https://github.com/Davidham3/STSGCN
```

---

## Train/val/test splits

| Dataset  | Train | Val | Test |
|----------|------:|----:|-----:|
| METR-LA  |   70% | 10% |  20% |
| PEMS-BAY |   70% | 10% |  20% |
| PEMS04   |   60% | 20% |  20% |

---

## Normalization

All datasets are normalized using Z-Score scaling (mean and standard deviation computed
on the training split only), as configured in each dataset's `meta.json`.
