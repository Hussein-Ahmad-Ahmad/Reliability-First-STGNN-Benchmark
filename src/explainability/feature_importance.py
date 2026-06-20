"""
Feature importance analysis using Integrated Gradients and permutation importance.
Identifies which input features contribute most to predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from tqdm import tqdm


class IntegratedGradients:
    """
    Integrated Gradients for feature attribution.
    
    Computes attributions by integrating gradients along a path from
    a baseline (e.g., zero) to the actual input.
    
    Reference:
        Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
    
    Args:
        model: Trained PyTorch model
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def compute_attributions(
        self,
        inputs: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        n_steps: int = 50,
        target_node: Optional[int] = None,
        target_time: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute integrated gradients attributions.
        
        Args:
            inputs: Input tensor (1, T, N, F) or (T, N, F)
            baseline: Baseline input (if None, use zeros)
            n_steps: Number of integration steps
            target_node: Node to explain (if None, average over all)
            target_time: Time step to explain (if None, average over all)
            
        Returns:
            Attribution values same shape as inputs
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.to(self.device)
        
        # Create baseline if not provided
        if baseline is None:
            baseline = torch.zeros_like(inputs)
        else:
            baseline = baseline.to(self.device)
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps, device=self.device)
        
        # Interpolate between baseline and input
        interpolated_inputs = []
        for alpha in alphas:
            interpolated = baseline + alpha * (inputs - baseline)
            interpolated_inputs.append(interpolated)
        
        interpolated_inputs = torch.cat(interpolated_inputs, dim=0)  # (n_steps, T, N, F)
        interpolated_inputs.requires_grad = True
        
        # Forward pass on all interpolated inputs
        outputs = self.model(interpolated_inputs)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            outputs = outputs['prediction']
        
        # Select target output
        if target_node is not None and target_time is not None:
            target_outputs = outputs[:, target_time, target_node]
        elif target_node is not None:
            target_outputs = outputs[:, :, target_node].mean(dim=1)
        elif target_time is not None:
            target_outputs = outputs[:, target_time, :].mean(dim=1)
        else:
            target_outputs = outputs.mean(dim=(1, 2))
        
        # Compute gradients
        gradients = []
        for i in range(n_steps):
            self.model.zero_grad()
            target_outputs[i].backward(retain_graph=True)
            gradients.append(interpolated_inputs.grad[i].detach())
            interpolated_inputs.grad.zero_()
        
        # Average gradients
        avg_gradients = torch.stack(gradients).mean(dim=0)  # (T, N, F)
        
        # Integrated gradients: (input - baseline) * avg_gradients
        attributions = (inputs[0] - baseline[0]) * avg_gradients
        
        return attributions.cpu().numpy()
    
    def analyze_feature_importance(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_samples: Optional[int] = None,
        n_steps: int = 50
    ) -> Dict[str, np.ndarray]:
        """
        Compute feature importance across multiple samples.
        
        Args:
            data_loader: Data loader for input data
            n_samples: Number of samples to analyze (if None, use all)
            n_steps: Number of integration steps
            
        Returns:
            Dictionary containing:
                - 'mean_importance': Average attribution per feature (F,)
                - 'std_importance': Std of attributions per feature (F,)
                - 'feature_rankings': Feature indices sorted by importance
        """
        all_attributions = []
        
        sample_count = 0
        for batch_data in tqdm(data_loader, desc="Computing feature importance"):
            if isinstance(batch_data, dict):
                inputs = batch_data['inputs']
            else:
                inputs, _ = batch_data
            
            for i in range(inputs.size(0)):
                if n_samples is not None and sample_count >= n_samples:
                    break
                
                sample_input = inputs[i:i+1]
                
                # Compute attributions
                attributions = self.compute_attributions(
                    sample_input,
                    n_steps=n_steps
                )
                
                # Average over time and nodes to get feature importance
                feature_importance = np.abs(attributions).mean(axis=(0, 1))  # (F,)
                all_attributions.append(feature_importance)
                
                sample_count += 1
            
            if n_samples is not None and sample_count >= n_samples:
                break
        
        all_attributions = np.array(all_attributions)
        
        mean_importance = all_attributions.mean(axis=0)
        std_importance = all_attributions.std(axis=0)
        feature_rankings = np.argsort(mean_importance)[::-1]
        
        return {
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'feature_rankings': feature_rankings
        }


def compute_feature_importance(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str = 'integrated_gradients',
    n_steps: int = 50,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Quick feature importance computation.
    
    Args:
        model: Trained model
        inputs: Input tensor (1, T, N, F) or (T, N, F)
        method: 'integrated_gradients' or 'gradient'
        n_steps: Integration steps (for IG)
        device: 'cuda' or 'cpu'
        
    Returns:
        Feature importance scores (F,)
        
    Example:
        >>> from basicts.baselines import D2STGNN
        >>> model = D2STGNN(...)
        >>> inputs = torch.randn(1, 12, 207, 3)  # 3 features
        >>> importance = compute_feature_importance(model, inputs)
        >>> print(f"Most important feature: {importance.argmax()}")
    """
    if method == 'integrated_gradients':
        ig = IntegratedGradients(model, device)
        attributions = ig.compute_attributions(inputs, n_steps=n_steps)
        # Average over time and nodes
        importance = np.abs(attributions).mean(axis=(0, 1))
    
    elif method == 'gradient':
        # Simple gradient-based importance
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.to(device)
        inputs.requires_grad = True
        
        model.eval()
        outputs = model(inputs)
        
        if isinstance(outputs, dict):
            outputs = outputs['prediction']
        
        # Backward pass
        model.zero_grad()
        outputs.mean().backward()
        
        # Use gradient magnitude as importance
        importance = inputs.grad.abs().mean(dim=(0, 1, 2)).cpu().numpy()
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance


def permutation_importance(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    n_repeats: int = 10,
    metric_fn: Optional[Callable] = None,
    device: str = 'cuda'
) -> Dict[str, np.ndarray]:
    """
    Compute permutation importance for each feature.
    
    Measures importance by randomly shuffling each feature and
    observing the decrease in model performance.
    
    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        n_repeats: Number of permutation repeats
        metric_fn: Metric function (if None, use MSE)
        device: 'cuda' or 'cpu'
        
    Returns:
        Dictionary with:
            - 'importance': Mean importance per feature (F,)
            - 'importance_std': Std of importance per feature (F,)
            - 'baseline_score': Baseline performance (no permutation)
    """
    if metric_fn is None:
        # Default: MSE
        def metric_fn(pred, target):
            return ((pred - target) ** 2).mean()
    
    model.eval()
    model.to(device)
    
    # Compute baseline performance
    baseline_scores = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if isinstance(batch_data, dict):
                inputs = batch_data['inputs'].to(device)
                targets = batch_data['targets'].to(device)
            else:
                inputs, targets = batch_data
                inputs = inputs.to(device)
                targets = targets.to(device)
            
            outputs = model(inputs)
            if isinstance(outputs, dict):
                outputs = outputs['prediction']
            
            score = metric_fn(outputs, targets)
            baseline_scores.append(score.item())
    
    baseline_score = np.mean(baseline_scores)
    
    # Get number of features
    sample_batch = next(iter(data_loader))
    if isinstance(sample_batch, dict):
        n_features = sample_batch['inputs'].shape[-1]
    else:
        n_features = sample_batch[0].shape[-1]
    
    # Compute importance for each feature
    feature_importances = []
    
    for feature_idx in tqdm(range(n_features), desc="Computing permutation importance"):
        feature_scores = []
        
        for repeat in range(n_repeats):
            repeat_scores = []
            
            with torch.no_grad():
                for batch_data in data_loader:
                    if isinstance(batch_data, dict):
                        inputs = batch_data['inputs'].to(device).clone()
                        targets = batch_data['targets'].to(device)
                    else:
                        inputs, targets = batch_data
                        inputs = inputs.to(device).clone()
                        targets = targets.to(device)
                    
                    # Permute feature
                    perm_indices = torch.randperm(inputs.size(0))
                    inputs[..., feature_idx] = inputs[perm_indices, :, :, feature_idx]
                    
                    # Compute score with permuted feature
                    outputs = model(inputs)
                    if isinstance(outputs, dict):
                        outputs = outputs['prediction']
                    
                    score = metric_fn(outputs, targets)
                    repeat_scores.append(score.item())
            
            feature_scores.append(np.mean(repeat_scores))
        
        # Importance = decrease in performance
        importance = np.array(feature_scores) - baseline_score
        feature_importances.append(importance)
    
    feature_importances = np.array(feature_importances)  # (F, n_repeats)
    
    return {
        'importance': feature_importances.mean(axis=1),
        'importance_std': feature_importances.std(axis=1),
        'baseline_score': baseline_score
    }


def analyze_feature_interactions(
    attributions: np.ndarray,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Analyze interactions between features based on attributions.
    
    Args:
        attributions: Attribution values (n_samples, T, N, F)
        feature_names: Names of features (if None, use indices)
        
    Returns:
        Dictionary with:
            - 'correlation_matrix': Correlation between feature attributions
            - 'dominant_features': Most consistently important features
    """
    n_features = attributions.shape[-1]
    
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(n_features)]
    
    # Flatten to (n_samples * T * N, F)
    flat_attributions = attributions.reshape(-1, n_features)
    
    # Compute correlation between features
    correlation_matrix = np.corrcoef(flat_attributions.T)
    
    # Find dominant features (high attribution variance)
    attribution_variance = flat_attributions.var(axis=0)
    dominant_indices = np.argsort(attribution_variance)[::-1]
    
    return {
        'correlation_matrix': correlation_matrix,
        'dominant_features': [feature_names[i] for i in dominant_indices],
        'attribution_variance': attribution_variance
    }


def compute_temporal_feature_importance(
    attributions: np.ndarray,
    time_window: Optional[int] = None
) -> np.ndarray:
    """
    Compute feature importance separately for different time windows.
    
    Args:
        attributions: Attribution values (n_samples, T, N, F)
        time_window: Window size (if None, compute for each time step)
        
    Returns:
        Feature importance per time window (n_windows, F)
    """
    n_samples, T, N, F = attributions.shape
    
    if time_window is None:
        # Per time step
        importance = np.abs(attributions).mean(axis=(0, 2))  # (T, F)
    else:
        # Per window
        n_windows = T // time_window
        importance = []
        
        for w in range(n_windows):
            start_t = w * time_window
            end_t = start_t + time_window
            window_attributions = attributions[:, start_t:end_t, :, :]
            window_importance = np.abs(window_attributions).mean(axis=(0, 1, 2))  # (F,)
            importance.append(window_importance)
        
        importance = np.array(importance)  # (n_windows, F)
    
    return importance
