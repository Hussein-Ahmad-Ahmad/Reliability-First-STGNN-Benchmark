"""
Metrics module for evaluation
"""
import torch
import numpy as np
from typing import Dict, Tuple
from sklearn.metrics import r2_score


def masked_mae(pred: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """Mean Absolute Error with masking"""
    mask = (target != mask_value).float()
    mask /= mask.mean()
    loss = torch.abs(pred - target)
    loss = loss * mask
    loss[loss != loss] = 0  # NaN to 0
    return loss.mean()


def masked_rmse(pred: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """Root Mean Squared Error with masking"""
    mask = (target != mask_value).float()
    mask /= mask.mean()
    loss = (pred - target) ** 2
    loss = loss * mask
    loss[loss != loss] = 0
    return torch.sqrt(loss.mean())


def masked_mape(pred: torch.Tensor, target: torch.Tensor, mask_value: float = 0.0, threshold: float = 0.1) -> torch.Tensor:
    """Mean Absolute Percentage Error with masking"""
    mask = (target != mask_value).float() * (torch.abs(target) > threshold).float()
    mask /= mask.mean()
    loss = torch.abs((pred - target) / target)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean() * 100


def smape(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Symmetric Mean Absolute Percentage Error"""
    numerator = torch.abs(pred - target)
    denominator = (torch.abs(pred) + torch.abs(target)) / 2
    return (numerator / denominator).mean() * 100


def r2(pred: torch.Tensor, target: torch.Tensor) -> float:
    """R-squared score"""
    pred_np = pred.detach().cpu().numpy().flatten()
    target_np = target.detach().cpu().numpy().flatten()
    return r2_score(target_np, pred_np)


def compute_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute all accuracy metrics"""
    return {
        'MAE': masked_mae(pred, target).item(),
        'RMSE': masked_rmse(pred, target).item(),
        'MAPE': masked_mape(pred, target).item(),
        'SMAPE': smape(pred, target).item(),
        'R2': r2(pred, target)
    }


# ============================================================================
# UNCERTAINTY METRICS
# ============================================================================

def picp(pred_lower: torch.Tensor, pred_upper: torch.Tensor, target: torch.Tensor) -> float:
    """Prediction Interval Coverage Probability"""
    in_interval = ((target >= pred_lower) & (target <= pred_upper)).float()
    return in_interval.mean().item()


def mpiw(pred_lower: torch.Tensor, pred_upper: torch.Tensor) -> float:
    """Mean Prediction Interval Width"""
    return (pred_upper - pred_lower).mean().item()


def interval_score(
    pred_lower: torch.Tensor,
    pred_upper: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.1
) -> float:
    """Interval Score (lower is better)"""
    width = pred_upper - pred_lower
    lower_penalty = (2 / alpha) * (pred_lower - target) * (target < pred_lower).float()
    upper_penalty = (2 / alpha) * (target - pred_upper) * (target > pred_upper).float()
    score = width + lower_penalty + upper_penalty
    return score.mean().item()


def negative_log_likelihood(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor
) -> float:
    """Negative Log-Likelihood"""
    variance = pred_std ** 2
    nll = 0.5 * torch.log(2 * np.pi * variance) + 0.5 * ((target - pred_mean) ** 2) / variance
    return nll.mean().item()


def expected_calibration_error(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
    num_bins: int = 10
) -> float:
    """Expected Calibration Error"""
    # Compute normalized residuals
    residuals = torch.abs(target - pred_mean) / pred_std
    
    # Bin by predicted confidence (inverse of std)
    confidence = 1 / (1 + pred_std)
    
    ece = 0.0
    for i in range(num_bins):
        lower = i / num_bins
        upper = (i + 1) / num_bins
        
        mask = (confidence >= lower) & (confidence < upper)
        if mask.sum() == 0:
            continue
        
        # Expected confidence in bin
        bin_confidence = confidence[mask].mean()
        
        # Actual accuracy in bin (residuals < 1 means within 1 std)
        bin_accuracy = (residuals[mask] < 1).float().mean()
        
        ece += torch.abs(bin_confidence - bin_accuracy) * mask.sum().float() / mask.numel()
    
    return ece.item()


def sharpness(pred_std: torch.Tensor) -> float:
    """Sharpness (lower is sharper/more confident)"""
    return pred_std.mean().item()


def compute_uncertainty_metrics(
    pred_mean: torch.Tensor,
    pred_std: torch.Tensor,
    target: torch.Tensor,
    z_factor: float = 1.96
) -> Dict[str, float]:
    """Compute all uncertainty quantification metrics"""
    pred_lower = pred_mean - z_factor * pred_std
    pred_upper = pred_mean + z_factor * pred_std
    
    return {
        'PICP': picp(pred_lower, pred_upper, target),
        'MPIW': mpiw(pred_lower, pred_upper),
        'IS': interval_score(pred_lower, pred_upper, target, alpha=0.15),
        'NLL': negative_log_likelihood(pred_mean, pred_std, target),
        'ECE': expected_calibration_error(pred_mean, pred_std, target),
        'Sharpness': sharpness(pred_std)
    }


# ============================================================================
# EXPLAINABILITY METRICS
# ============================================================================

def faithfulness_correlation(
    original_pred: torch.Tensor,
    masked_pred: torch.Tensor,
    importance_scores: torch.Tensor
) -> float:
    """
    Faithfulness: Correlation between importance scores and prediction change
    Higher is better (explanations correlate with model behavior)
    """
    pred_change = torch.abs(original_pred - masked_pred).flatten()
    importance = importance_scores.flatten()
    
    # Pearson correlation
    correlation = torch.corrcoef(torch.stack([pred_change, importance]))[0, 1]
    return correlation.item()


def sparsity(importance_scores: torch.Tensor, threshold: float = 0.1) -> float:
    """
    Sparsity: Percentage of features/nodes with low importance
    Higher sparsity = more concentrated explanations
    """
    important = (importance_scores > threshold).float()
    return 1 - important.mean().item()


def validity_check(
    importance_scores: torch.Tensor,
    min_value: float = 0.0,
    max_value: float = 1.0
) -> float:
    """
    Validity: Check if importance scores are within valid range
    Returns percentage of valid scores
    """
    valid = ((importance_scores >= min_value) & (importance_scores <= max_value)).float()
    return valid.mean().item()


def coverage_consistency(importance_scores_list: list) -> float:
    """
    Coverage: Consistency of explanations across different samples
    Higher = more consistent explanations
    """
    if len(importance_scores_list) < 2:
        return 1.0
    
    # Stack all importance scores
    stacked = torch.stack([s.flatten() for s in importance_scores_list])
    
    # Compute coefficient of variation (std / mean)
    mean = stacked.mean(dim=0)
    std = stacked.std(dim=0)
    
    # Avoid division by zero
    cv = std / (mean + 1e-8)
    
    # Lower CV = higher consistency
    consistency = 1 / (1 + cv.mean())
    return consistency.item()


def compute_explainability_metrics(
    original_pred: torch.Tensor,
    masked_pred: torch.Tensor,
    importance_scores: torch.Tensor,
    all_importance_scores: list = None
) -> Dict[str, float]:
    """Compute all explainability metrics"""
    metrics = {
        'Faithfulness': faithfulness_correlation(original_pred, masked_pred, importance_scores),
        'Sparsity': sparsity(importance_scores),
        'Validity': validity_check(importance_scores)
    }
    
    if all_importance_scores is not None:
        metrics['Coverage'] = coverage_consistency(all_importance_scores)
    
    return metrics


if __name__ == "__main__":
    # Test metrics
    pred = torch.randn(32, 12, 207, 1)
    target = torch.randn(32, 12, 207, 1)
    
    # Accuracy metrics
    acc_metrics = compute_all_metrics(pred, target)
    print("Accuracy Metrics:", acc_metrics)
    
    # Uncertainty metrics
    pred_std = torch.abs(torch.randn(32, 12, 207, 1))
    uq_metrics = compute_uncertainty_metrics(pred, pred_std, target)
    print("Uncertainty Metrics:", uq_metrics)
    
    # Explainability metrics
    importance = torch.rand(32, 207)
    masked = pred + torch.randn_like(pred) * 0.1
    xai_metrics = compute_explainability_metrics(pred, masked, importance)
    print("Explainability Metrics:", xai_metrics)
