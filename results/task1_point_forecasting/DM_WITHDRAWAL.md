# Flattened DM Analysis Withdrawn

The legacy METR-LA 21-pair Diebold-Mariano matrix is not part of the corrected
benchmark evidence.

The June 20, 2026 v1.0.0 artifact flattened sensor-horizon losses from shared
forecast origins and attached direction labels to legacy statistics that could
not be regenerated as one sign-consistent, provenance-complete output. Later
exploratory recomputations also did not provide a stable replacement: they used
different analysis units or masking rules, and one Holm adjustment path was not
valid for the original pair ordering.

For those reasons, the flattened DM JSON, its figure generator, and its
manuscript claims were withdrawn rather than relabeled. The dependence-aware
pairwise sensitivity evidence retained by the corrected study is the
forecast-origin moving-block bootstrap:

- block_bootstrap_pairwise_metr-la_seeds43-44-45.csv
- block_bootstrap_pairwise_pems04_seeds43-44-45.csv

These files aggregate loss at the forecast-origin level before resampling
one-day blocks. They should not be described as Diebold-Mariano tests.
