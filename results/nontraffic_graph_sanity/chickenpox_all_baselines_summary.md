# Chickenpox Hungary Graph-Native Protocol Illustration

Secondary graph-native non-traffic experiment using the same seven model classes from the traffic benchmark with compact Chickenpox-specific dimensions.

The target-disjoint split uses 286 training, 30 validation, 50 calibration, and 99 unchanged test origins. Eleven origins are excluded at each boundary so 12-week forecast targets do not overlap across partitions.

| Model | MAE | RMSE | 90% coverage | Width | Params |
|---|---:|---:|---:|---:|---:|
| STNorm | 0.6394 +/- 0.0005 | 1.0054 +/- 0.0006 | 0.8797 +/- 0.0003 | 3.2174 +/- 0.0012 | 35068 |
| STGCN-Cheb | 0.6403 +/- 0.0007 | 1.0071 +/- 0.0007 | 0.8797 +/- 0.0012 | 3.2381 +/- 0.0094 | 13276 |
| MTGNN | 0.6408 +/- 0.0004 | 1.0059 +/- 0.0011 | 0.8792 +/- 0.0005 | 3.2194 +/- 0.0039 | 22732 |
| D2STGNN | 0.6412 +/- 0.0007 | 1.0091 +/- 0.0017 | 0.8783 +/- 0.0007 | 3.2205 +/- 0.0029 | 230032 |
| MegaCRN | 0.6415 +/- 0.0014 | 1.0091 +/- 0.0030 | 0.8786 +/- 0.0014 | 3.2226 +/- 0.0235 | 42833 |
| STID | 0.6440 +/- 0.0007 | 1.0088 +/- 0.0005 | 0.8792 +/- 0.0008 | 3.2425 +/- 0.0091 | 11756 |
| STAEformer | 0.6694 +/- 0.0014 | 1.0330 +/- 0.0007 | 0.8840 +/- 0.0023 | 3.3750 +/- 0.0226 | 25996 |

The protocol, validation histories, per-seed best epochs, calibration residuals, coordinate quantiles, and test predictions are retained with the release. This secondary experiment is not pooled with the traffic-domain model rankings.

MAE, RMSE, and interval width remain in the dataset-provided county-wise standardized FX signal units. They are not numbers of weekly cases.

Coverage is an empirical chronological diagnostic under temporal dependence, not a distribution-free guarantee under arbitrary temporal shift.
