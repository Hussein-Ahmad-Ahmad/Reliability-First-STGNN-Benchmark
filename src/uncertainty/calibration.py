"""
Calibration utilities for uncertainty quantification.
Sweeps scaling factors to achieve target coverage levels.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


class CalibrationSweep:
    """
    Calibrates prediction intervals to achieve target coverage.
    
    Prediction intervals are constructed as [mean - z*std, mean + z*std].
    This class sweeps over z values to find the optimal scaling factor
    that achieves the desired coverage probability.
    
    Coverage = % of true values that fall within prediction intervals
    
    Args:
        z_range: Range of z-factors to sweep (min, max, step)
        target_coverage: Target coverage probability (e.g., 0.85 for 85%)
    """
    
    def __init__(
        self,
        z_range: Tuple[float, float, float] = (0.5, 15.0, 0.5),
        target_coverage: float = 0.85
    ):
        self.z_min, self.z_max, self.z_step = z_range
        self.target_coverage = target_coverage
        
        # Generate z-factor candidates
        self.z_candidates = np.arange(self.z_min, self.z_max + self.z_step, self.z_step)
        
    def compute_coverage(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        targets: np.ndarray,
        z_factor: float
    ) -> float:
        """
        Compute prediction interval coverage for a given z-factor.
        
        Args:
            mean: Mean predictions (B, T, N)
            std: Standard deviation (B, T, N)
            targets: Ground truth values (B, T, N)
            z_factor: Scaling factor for std
            
        Returns:
            Coverage probability (0-1)
        """
        # Construct prediction intervals
        lower_bound = mean - z_factor * std
        upper_bound = mean + z_factor * std
        
        # Check if true values fall within intervals
        within_interval = (targets >= lower_bound) & (targets <= upper_bound)
        
        # Coverage = proportion of values within intervals
        coverage = within_interval.mean()
        
        return coverage
    
    def compute_interval_width(
        self,
        std: np.ndarray,
        z_factor: float
    ) -> float:
        """
        Compute mean prediction interval width.
        
        Args:
            std: Standard deviation (B, T, N)
            z_factor: Scaling factor for std
            
        Returns:
            Mean interval width
        """
        # Interval width = 2 * z * std
        width = 2 * z_factor * std
        return width.mean()
    
    def calibrate(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        targets: np.ndarray,
        verbose: bool = True
    ) -> Dict:
        """
        Find optimal z-factor that achieves target coverage.
        
        Args:
            mean: Mean predictions (B, T, N)
            std: Standard deviation (B, T, N)
            targets: Ground truth values (B, T, N)
            verbose: Print progress
            
        Returns:
            Dictionary with calibration results:
                - 'optimal_z': Best z-factor
                - 'achieved_coverage': Coverage at optimal_z
                - 'mean_width': Mean interval width at optimal_z
                - 'sweep_results': Full sweep results
        """
        sweep_results = []
        
        iterator = tqdm(self.z_candidates, desc="Calibration sweep") if verbose else self.z_candidates
        
        for z in iterator:
            coverage = self.compute_coverage(mean, std, targets, z)
            width = self.compute_interval_width(std, z)
            
            sweep_results.append({
                'z_factor': z,
                'coverage': coverage,
                'mean_width': width
            })
        
        # Find z-factor closest to target coverage
        sweep_results = sorted(sweep_results, key=lambda x: abs(x['coverage'] - self.target_coverage))
        optimal_result = sweep_results[0]
        
        if verbose:
            print(f"\nCalibration results:")
            print(f"  Target coverage: {self.target_coverage:.2%}")
            print(f"  Optimal z-factor: {optimal_result['z_factor']:.2f}")
            print(f"  Achieved coverage: {optimal_result['coverage']:.2%}")
            print(f"  Mean interval width: {optimal_result['mean_width']:.4f}")
        
        return {
            'optimal_z': optimal_result['z_factor'],
            'achieved_coverage': optimal_result['coverage'],
            'mean_width': optimal_result['mean_width'],
            'sweep_results': sweep_results
        }
    
    def apply_calibration(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        z_factor: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply calibrated z-factor to compute prediction intervals.
        
        Args:
            mean: Mean predictions (B, T, N)
            std: Standard deviation (B, T, N)
            z_factor: Calibrated z-factor
            
        Returns:
            lower_bound: Lower prediction interval (B, T, N)
            upper_bound: Upper prediction interval (B, T, N)
        """
        lower_bound = mean - z_factor * std
        upper_bound = mean + z_factor * std
        
        return lower_bound, upper_bound


def calibrate_multiple_horizons(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    target_coverage: float = 0.85,
    z_range: Tuple[float, float, float] = (0.5, 15.0, 0.5)
) -> Dict[int, Dict]:
    """
    Calibrate prediction intervals separately for each forecast horizon.
    
    Different horizons may require different z-factors to achieve
    the same coverage due to varying uncertainty levels.
    
    Args:
        mean: Mean predictions (B, T, N) where T is number of horizons
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        target_coverage: Target coverage probability
        z_range: Range of z-factors to sweep
        
    Returns:
        Dictionary mapping horizon index to calibration results
        
    Example:
        >>> results = calibrate_multiple_horizons(mean, std, targets)
        >>> for horizon, calib in results.items():
        ...     print(f"Horizon {horizon+1}: z={calib['optimal_z']:.2f}, "
        ...           f"coverage={calib['achieved_coverage']:.2%}")
    """
    num_horizons = mean.shape[1]
    calibration_results = {}
    
    for h in range(num_horizons):
        print(f"\nCalibrating horizon {h+1}/{num_horizons}")
        
        calibrator = CalibrationSweep(
            z_range=z_range,
            target_coverage=target_coverage
        )
        
        # Calibrate for this horizon
        calib_result = calibrator.calibrate(
            mean[:, h, :],
            std[:, h, :],
            targets[:, h, :],
            verbose=False
        )
        
        calibration_results[h] = calib_result
        
        print(f"  Optimal z: {calib_result['optimal_z']:.2f}")
        print(f"  Coverage: {calib_result['achieved_coverage']:.2%}")
    
    return calibration_results


def calibrate_per_node(
    mean: np.ndarray,
    std: np.ndarray,
    targets: np.ndarray,
    target_coverage: float = 0.85,
    z_range: Tuple[float, float, float] = (0.5, 15.0, 0.5)
) -> Dict[int, Dict]:
    """
    Calibrate prediction intervals separately for each spatial node.
    
    Different locations may have different uncertainty characteristics,
    requiring node-specific calibration.
    
    Args:
        mean: Mean predictions (B, T, N) where N is number of nodes
        std: Standard deviation (B, T, N)
        targets: Ground truth values (B, T, N)
        target_coverage: Target coverage probability
        z_range: Range of z-factors to sweep
        
    Returns:
        Dictionary mapping node index to calibration results
        
    Example:
        >>> results = calibrate_per_node(mean, std, targets)
        >>> for node, calib in results.items():
        ...     print(f"Node {node}: z={calib['optimal_z']:.2f}, "
        ...           f"coverage={calib['achieved_coverage']:.2%}")
    """
    num_nodes = mean.shape[2]
    calibration_results = {}
    
    for n in range(num_nodes):
        print(f"\nCalibrating node {n+1}/{num_nodes}")
        
        calibrator = CalibrationSweep(
            z_range=z_range,
            target_coverage=target_coverage
        )
        
        # Calibrate for this node
        calib_result = calibrator.calibrate(
            mean[:, :, n],
            std[:, :, n],
            targets[:, :, n],
            verbose=False
        )
        
        calibration_results[n] = calib_result
        
        print(f"  Optimal z: {calib_result['optimal_z']:.2f}")
        print(f"  Coverage: {calib_result['achieved_coverage']:.2%}")
    
    return calibration_results


def compute_adaptive_intervals(
    mean: np.ndarray,
    std: np.ndarray,
    base_z: float = 1.96,
    temperature: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute adaptive prediction intervals that vary by uncertainty magnitude.
    
    Instead of fixed z-factor, scales intervals based on local uncertainty.
    High uncertainty regions get wider intervals.
    
    Args:
        mean: Mean predictions (B, T, N)
        std: Standard deviation (B, T, N)
        base_z: Base z-factor (e.g., 1.96 for 95% confidence)
        temperature: Scaling temperature (>1 = wider intervals)
        
    Returns:
        lower_bound: Lower prediction interval (B, T, N)
        upper_bound: Upper prediction interval (B, T, N)
    """
    # Normalize std to [0, 1] range
    std_norm = (std - std.min()) / (std.max() - std.min() + 1e-8)
    
    # Adaptive z-factor: larger for higher uncertainty
    adaptive_z = base_z * (1 + temperature * std_norm)
    
    lower_bound = mean - adaptive_z * std
    upper_bound = mean + adaptive_z * std
    
    return lower_bound, upper_bound


class ConformalPredictor:
    """Run the archived normalized split-conformal implementation."""

    def __init__(
        self,
        dataset: str,
        variant: str,
        output_path: str,
        ensemble_predictions_path: str,
        targets_path: str,
        alpha: float = 0.1,
    ):
        from pathlib import Path

        if variant not in {"fixed", "per_horizon"}:
            raise ValueError(f"Unsupported conformal variant: {variant}")
        self.dataset = dataset
        self.variant = variant
        self.output_path = Path(output_path)
        self.ensemble_predictions_path = Path(ensemble_predictions_path)
        self.targets_path = Path(targets_path)
        self.alpha = alpha

    def calibrate_and_evaluate(self):
        import json
        from scripts.generate_conformal_intervals import compute_conformal_metrics

        predictions = np.load(self.ensemble_predictions_path, mmap_mode="r")
        targets = np.load(self.targets_path, mmap_mode="r")
        if predictions.ndim != 4:
            raise ValueError(
                "Expected ensemble predictions with shape "
                "[members, origins, horizons, sensors]"
            )
        mean = np.asarray(predictions.mean(axis=0), dtype=np.float32)
        std = np.asarray(predictions.std(axis=0, ddof=0), dtype=np.float32)
        result = compute_conformal_metrics(
            mean,
            std,
            targets,
            dataset=self.dataset,
            alpha=self.alpha,
            member_count=predictions.shape[0],
        )[self.variant]

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        return result
