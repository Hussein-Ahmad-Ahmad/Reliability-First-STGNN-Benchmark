# Point Forecasting

Current seed summaries use sample standard deviations (`ddof=1`). The `block_bootstrap_reconciled_*` and `block_bootstrap_sensitivity_reconciled_*` files use flat-pooled sum/count aggregation and are the current bootstrap results. Files named `block_bootstrap_pairwise_*` belong to the earlier snapshot and use a different aggregation convention; they are not inputs to the current results index.

Bootstrap differences are A minus B. The numbered table reports B minus A, requiring negation and reversal of interval endpoints. Each pair uses its disclosed common seed intersection; bootstrap intervals do not include seed uncertainty.
