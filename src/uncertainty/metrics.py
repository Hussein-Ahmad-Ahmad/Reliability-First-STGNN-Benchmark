"""
Uncertainty quantification metrics for evaluating probabilistic predictions.
"""

import numpy as np
from typing import Dict, Optional
from scipy import stats


def picp(
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Prediction Interval Coverage Probability (PICP).
    
    Measures the proportion of true values that fall within prediction intervals.
    Higher is better (target: 85-95% depending on desired confidence level).
    
    Args:
        predictions: Mean predictions (B, T, N)
        lower_bound: Lower prediction interval (B, T, N)
        upper_bound: Upper prediction interval (B, T, N)
        targets: Ground truth values (B, T, N)
        
    Returns:
        PICP score (0-1)
    """
    within_interval = (targets >= lower_bound) & (targets <= upper_bound)
    return within_interval.mean()


def mpiw(
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    normalize: bool = True,
    targets: Optional[np.ndarray] = None
) -> float:
    """
    Mean Prediction Interval Width (MPIW).
    
    Measures the average width of prediction intervals.
    Lower is better (narrower intervals = more precise predictions).
    
    Args:
        lower_bound: Lower prediction interval (B, T, N)
        upper_bound: Upper prediction interval (B, T, N)
        normalize: Whether to normalize by target range
        targets: Ground truth values (required if normalize=True)
        
    Returns:
        MPIW score
    """
    width = upper_bound - lower_bound
    mean_width = width.mean()
    
    if normalize:
        if targets is None:
            raise ValueError("targets required for normalization")
        target_range = targets.max() - targets.min()
        mean_width = mean_width / (target_range + 1e-8)
    
    return mean_width


def interval_score(
    predictions: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
    targets: np.ndarray,
    alpha: float = 0.1
) -> float:
    """
    Interval Score (IS) - Proper scoring rule for interval forecasts.
    
    Combines interval width and penalties for target values outside interval.
    Lower is better.
    
    Reference:
        Gneiting & Raftery (2007) "Strictly Proper Scoring Rules"
    
    Args:
        predictions: Mean predictions (B, T, N)
        lower_bound: Lower prediction interval (B, T, N)
        upper_bound: Upper prediction interval (B, T, N)
        targets: Ground truth values (B, T, N)
        alpha: Miscoverage level (e.g., 0.1 for 90% intervals)
        
    Returns:
        Interval score (lower is better)
    """
    width = upper_bound - lower_bound
    
    # Penalty for values below lower bound
    below = np.maximum(0, lower_bound - targets)
    below_penalty = (2 / alpha) * below
    
    # Penalty for values above upper bound
    above = np.maximum(0, targets - upper_bound)
    above_penalty = (2 / alpha) * above
    
    # Total score
    score = width + below_penalty + above_penalty
    
    return score.mean()


def negative_log_likelihood(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    eps: float = 1e-8
) -> float:
    """
    Negative Log-Likelihood (NLL) assuming Gaussian distribution.
    
    Proper scoring rule that evaluates both accuracy and calibration.
    Lower is better.
    
    Args:
        mean: Mean predictions (B, T, N)
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        eps: Small constant for numerical stability
        
    Returns:
        NLL score (lower is better)
    """
    # Avoid division by zero
    std = np.maximum(std, eps)
    
    # Log-likelihood for Gaussian
    # log p(y|μ,σ) = -0.5 * log(2πσ²) - (y-μ)²/(2σ²)
    log_likelihood = -0.5 * np.log(2 * np.pi * std**2) - (targets - mean)**2 / (2 * std**2)
    
    # Return negative log-likelihood (lower is better)
    return -log_likelihood.mean()


def continuous_ranked_probability_score(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray
) -> float:
    """
    Continuous Ranked Probability Score (CRPS) for Gaussian predictions.
    
    Proper scoring rule that generalizes MAE to probabilistic forecasts.
    Lower is better.
    
    Args:
        mean: Mean predictions (B, T, N)
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        
    Returns:
        CRPS score (lower is better)
    """
    # Standardize
    z = (targets - mean) / (std + 1e-8)
    
    # CRPS for Gaussian: σ * [z * (2Φ(z) - 1) + 2φ(z) - 1/√π]
    # where Φ is CDF and φ is PDF of standard normal
    pdf = stats.norm.pdf(z)
    cdf = stats.norm.cdf(z)
    
    crps = std * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
    
    return crps.mean()


def sharpness(std: np.ndarray) -> float:
    """
    Sharpness - Measure of prediction uncertainty magnitude.
    
    Lower is better (more confident predictions).
    Should be reported alongside coverage to ensure intervals are not trivially wide.
    
    Args:
        std: Standard deviation (B, T, N)
        
    Returns:
        Mean standard deviation
    """
    return std.mean()


def expected_calibration_error(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Expected Calibration Error (ECE) for regression.
    
    Measures whether predicted confidence matches actual accuracy.
    Lower is better (0 = perfectly calibrated).
    
    Bins predictions by confidence (1/std) and checks if observed error
    matches predicted uncertainty in each bin.
    
    Args:
        mean: Mean predictions (B, T, N)
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        n_bins: Number of confidence bins
        
    Returns:
        ECE score (lower is better)
    """
    # Flatten arrays
    mean_flat = mean.flatten()
    std_flat = std.flatten()
    targets_flat = targets.flatten()
    
    # Compute confidence (inverse of std)
    confidence = 1 / (std_flat + 1e-8)
    
    # Bin edges
    bin_edges = np.linspace(confidence.min(), confidence.max(), n_bins + 1)
    
    # Compute ECE
    ece = 0.0
    n_total = len(mean_flat)
    
    for i in range(n_bins):
        # Get samples in this bin
        in_bin = (confidence >= bin_edges[i]) & (confidence < bin_edges[i + 1])
        n_bin = in_bin.sum()
        
        if n_bin == 0:
            continue
        
        # Observed error in bin
        observed_error = np.abs(mean_flat[in_bin] - targets_flat[in_bin]).mean()
        
        # Expected error (mean std in bin)
        expected_error = std_flat[in_bin].mean()
        
        # Weighted difference
        ece += (n_bin / n_total) * np.abs(observed_error - expected_error)
    
    return ece


def compute_uq_metrics(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    z_factor: float = 1.96,
    alpha: float = 0.05
) -> Dict[str, float]:
    """
    Compute comprehensive uncertainty quantification metrics.
    
    Args:
        mean: Mean predictions (B, T, N)
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        lower_bound: Lower prediction interval (if not provided, computed from mean/std)
        upper_bound: Upper prediction interval (if not provided, computed from mean/std)
        z_factor: Z-factor for interval construction (default: 1.96 for 95%)
        alpha: Miscoverage level for interval score (default: 0.05 for 95%)
        
    Returns:
        Dictionary of UQ metrics
    """
    # Construct intervals if not provided
    if lower_bound is None or upper_bound is None:
        lower_bound = mean - z_factor * std
        upper_bound = mean + z_factor * std
    
    metrics = {
        # Coverage metrics
        'PICP': picp(mean, lower_bound, upper_bound, targets),
        'MPIW': mpiw(lower_bound, upper_bound, normalize=True, targets=targets),
        'MPIW_raw': mpiw(lower_bound, upper_bound, normalize=False),
        
        # Proper scoring rules
        'NLL': negative_log_likelihood(mean, std, targets),
        'CRPS': continuous_ranked_probability_score(mean, std, targets),
        'IS': interval_score(mean, lower_bound, upper_bound, targets, alpha=alpha),
        
        # Calibration
        'ECE': expected_calibration_error(mean, std, targets),
        'Sharpness': sharpness(std),
        
        # Basic accuracy (for reference)
        'MAE': np.abs(mean - targets).mean(),
        'RMSE': np.sqrt(((mean - targets) ** 2).mean())
    }
    
    return metrics


def compute_uq_metrics_per_horizon(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    z_factor: float = 1.96
) -> Dict[int, Dict[str, float]]:
    """
    Compute UQ metrics separately for each forecast horizon.
    
    Args:
        mean: Mean predictions (B, T, N) where T is number of horizons
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        z_factor: Z-factor for interval construction
        
    Returns:
        Dictionary mapping horizon index to metrics
    """
    num_horizons = mean.shape[1]
    horizon_metrics = {}
    
    for h in range(num_horizons):
        horizon_metrics[h] = compute_uq_metrics(
            mean[:, h, :],
            std[:, h, :],
            targets[:, h, :],
            z_factor=z_factor
        )
    
    return horizon_metrics


def print_uq_metrics(metrics: Dict[str, float], title: str = "UQ Metrics"):
    """
    Pretty-print UQ metrics.
    
    Args:
        metrics: Dictionary of metrics
        title: Title to display
    """
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")
    
    # Group metrics
    coverage_metrics = ['PICP', 'MPIW', 'MPIW_raw']
    scoring_metrics = ['NLL', 'CRPS', 'IS']
    calibration_metrics = ['ECE', 'Sharpness']
    accuracy_metrics = ['MAE', 'RMSE']
    
    def print_group(group_name, metric_names):
        print(f"\n{group_name}:")
        for name in metric_names:
            if name in metrics:
                print(f"  {name:12s}: {metrics[name]:.4f}")
    
    print_group("Coverage", coverage_metrics)
    print_group("Proper Scoring Rules", scoring_metrics)
    print_group("Calibration", calibration_metrics)
    print_group("Accuracy (Reference)", accuracy_metrics)
    
    print(f"{'='*60}\n")
