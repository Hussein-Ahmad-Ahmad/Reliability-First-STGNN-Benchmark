# Statistical Analysis Scope

The legacy flattened Diebold-Mariano output is not part of the supported
analysis workflow. Sensor-horizon losses within a forecast origin are
dependent; the legacy direction labels and multiplicity adjustment could not
be supported by a consistent, reproducible calculation.

Current pairwise sensitivity uses forecast-origin moving-block bootstrap with
flat-pooled sum/count aggregation. Use `block_bootstrap_reconciled_*.json` and
`block_bootstrap_sensitivity_reconciled_*.json`. These outputs are not
Diebold-Mariano tests. Intervals are pointwise exploratory diagnostics and are
not multiplicity-adjusted.
