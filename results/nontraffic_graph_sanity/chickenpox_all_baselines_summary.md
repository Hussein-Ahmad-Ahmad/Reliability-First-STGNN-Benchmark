# Chickenpox Hungary Graph-Native Protocol Illustration

Secondary graph-native non-traffic experiment using the same seven model classes from the traffic benchmark with compact Chickenpox-specific dimensions.

| Model | MAE | RMSE | 90% coverage | Width | Params |
|---|---:|---:|---:|---:|---:|
| STNorm | 0.6389 +/- 0.0004 | 1.0051 +/- 0.0008 | 0.8850 +/- 0.0008 | 2.8992 +/- 0.0193 | 35068 |
| MegaCRN | 0.6394 +/- 0.0007 | 1.0057 +/- 0.0016 | 0.8854 +/- 0.0006 | 2.9050 +/- 0.0179 | 42833 |
| STGCN-Cheb | 0.6396 +/- 0.0010 | 1.0063 +/- 0.0013 | 0.8848 +/- 0.0009 | 2.9134 +/- 0.0063 | 13276 |
| MTGNN | 0.6399 +/- 0.0003 | 1.0053 +/- 0.0006 | 0.8843 +/- 0.0014 | 2.8948 +/- 0.0143 | 22732 |
| D2STGNN | 0.6403 +/- 0.0006 | 1.0089 +/- 0.0010 | 0.8851 +/- 0.0006 | 2.9146 +/- 0.0070 | 230032 |
| STID | 0.6416 +/- 0.0006 | 1.0073 +/- 0.0005 | 0.8844 +/- 0.0007 | 2.9114 +/- 0.0060 | 11756 |
| STAEformer | 0.6649 +/- 0.0019 | 1.0289 +/- 0.0016 | 0.8872 +/- 0.0009 | 3.0320 +/- 0.0078 | 25996 |

The full protocol, model dimensions, split counts, interval construction, and per-seed best epochs are recorded in chickenpox_protocol_manifest.json. This secondary experiment is not pooled with the traffic-domain model rankings.
