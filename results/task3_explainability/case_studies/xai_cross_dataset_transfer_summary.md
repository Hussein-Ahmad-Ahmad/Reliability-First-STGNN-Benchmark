# XAI Cross-Dataset Transfer Summary

Stored GNNExplainer transfer diagnostics are available for PEMS-BAY and PEMS04.
These are used only as lightweight support for method-dependent diagnostic behavior; no causal or universal XAI generalization is claimed.

| Dataset | Model | Top-3 Sensors | Fidelity k10 | Fidelity k20 | Stability Jaccard 0.1 |
|---|---|---:|---:|---:|---:|
| PEMS-BAY | D2STGNN | 41, 94, 247 | 1.04 | 0.971 | 0.0637 |
| PEMS04 | D2STGNN | 39, 203, 142 | 1.26 | 1.29 | 0.069 |
| PEMS-BAY | MTGNN | 268, 41, 9 | 0.947 | 0.969 | 0.159 |
| PEMS04 | MTGNN | 60, 274, 131 | 1.44 | 1.36 | 0.165 |
| PEMS-BAY | STID | 82, 121, 26 | 0.969 | 0.989 | 0.085 |
| PEMS04 | STID | 31, 266, 128 | 0.793 | 0.851 | 0.103 |

Takeaway: the non-METR-LA datasets already have small stored GNNExplainer checks for D2STGNN, MTGNN, and STID.
The detailed IG-vs-GNNExplainer case study remains METR-LA only because stored IG outputs are METR-LA-focused.
