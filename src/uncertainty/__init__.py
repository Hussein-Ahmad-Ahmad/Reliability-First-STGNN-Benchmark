"""
Uncertainty quantification module.

Provides methods for quantifying prediction uncertainty:
- MC Dropout: Enable dropout at inference time
- Deep Ensemble: Aggregate predictions from multiple models
- Calibration: Optimize prediction interval coverage
- Metrics: Comprehensive UQ evaluation metrics
"""

from .mc_dropout import MCDropoutWrapper, apply_mc_dropout_to_checkpoint, MCDropoutEvaluator
from .ensemble import DeepEnsemble, create_ensemble_from_seeds, train_ensemble_members, EnsembleUQEvaluator
from .calibration import (
    CalibrationSweep,
    calibrate_multiple_horizons,
    calibrate_per_node,
    compute_adaptive_intervals,
    ConformalPredictor
)
from .metrics import (
    compute_uq_metrics,
    compute_uq_metrics_per_horizon,
    print_uq_metrics,
    picp,
    mpiw,
    interval_score,
    negative_log_likelihood,
    continuous_ranked_probability_score,
    sharpness,
    expected_calibration_error
)

__all__ = [
    # MC Dropout
    'MCDropoutWrapper',
    'apply_mc_dropout_to_checkpoint',
    'MCDropoutEvaluator',
    
    # Deep Ensemble
    'DeepEnsemble',
    'create_ensemble_from_seeds',
    'train_ensemble_members',
    'EnsembleUQEvaluator',
    
    # Calibration
    'CalibrationSweep',
    'calibrate_multiple_horizons',
    'calibrate_per_node',
    'compute_adaptive_intervals',
    'ConformalPredictor',
    
    # Metrics
    'compute_uq_metrics',
    'compute_uq_metrics_per_horizon',
    'print_uq_metrics',
    'picp',
    'mpiw',
    'interval_score',
    'negative_log_likelihood',
    'continuous_ranked_probability_score',
    'sharpness',
    'expected_calibration_error',
]

