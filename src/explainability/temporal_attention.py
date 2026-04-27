"""
Temporal attention analysis for transformer-based models.
Extracts and visualizes attention weights to understand temporal dependencies.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class TemporalAttentionAnalyzer:
    """
    Extracts and analyzes attention weights from transformer models.
    
    Helps understand which historical time steps are most important
    for making predictions at each forecast horizon.
    
    Args:
        model: Trained transformer model with attention layers
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
        
        # Store attention weights
        self.attention_weights = {}
        self.attention_layer_names = []
        
        # Register hooks on attention layers
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks on all attention layers."""
        
        def attention_hook(name):
            def hook(module, input, output):
                # Store attention weights
                # Different transformers have different output formats
                if isinstance(output, tuple):
                    # Standard: (output, attention_weights)
                    if len(output) > 1 and output[1] is not None:
                        self.attention_weights[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    # Custom attribute
                    self.attention_weights[name] = module.attention_weights.detach()
            return hook
        
        # Register on common attention layer types
        attention_layer_types = (
            'MultiheadAttention', 'SelfAttention', 'CrossAttention',
            'TransformerEncoderLayer', 'TransformerDecoderLayer'
        )
        
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            if any(attn_type in layer_type for attn_type in attention_layer_types):
                module.register_forward_hook(attention_hook(name))
                self.attention_layer_names.append(name)
    
    def extract_attention(
        self,
        inputs: torch.Tensor,
        return_all_layers: bool = False
    ) -> Dict[str, np.ndarray]:
        """
        Extract attention weights for a single input.
        
        Args:
            inputs: Input tensor (1, T, N, F) or (T, N, F)
            return_all_layers: Return attention from all layers or just last
            
        Returns:
            Dictionary mapping layer names to attention weights
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)
        
        inputs = inputs.to(self.device)
        
        # Clear previous attention weights
        self.attention_weights = {}
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(inputs)
        
        # Extract attention weights
        attention_dict = {}
        
        if return_all_layers:
            for name in self.attention_layer_names:
                if name in self.attention_weights:
                    attention_dict[name] = self.attention_weights[name].cpu().numpy()
        else:
            # Return last layer only
            if self.attention_layer_names:
                last_layer = self.attention_layer_names[-1]
                if last_layer in self.attention_weights:
                    attention_dict['last_layer'] = self.attention_weights[last_layer].cpu().numpy()
        
        return attention_dict
    
    def analyze_temporal_patterns(
        self,
        data_loader: torch.utils.data.DataLoader,
        n_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Analyze attention patterns across multiple samples.
        
        Args:
            data_loader: Data loader for input data
            n_samples: Number of samples to analyze (if None, use all)
            
        Returns:
            Dictionary containing:
                - 'mean_attention': Average attention weights (T_out, T_in)
                - 'std_attention': Std of attention weights (T_out, T_in)
                - 'all_attention': All attention maps (n_samples, T_out, T_in)
        """
        all_attention_maps = []
        
        sample_count = 0
        for batch_data in tqdm(data_loader, desc="Extracting attention weights"):
            if isinstance(batch_data, dict):
                inputs = batch_data['inputs']
            else:
                inputs, _ = batch_data
            
            for i in range(inputs.size(0)):
                if n_samples is not None and sample_count >= n_samples:
                    break
                
                sample_input = inputs[i:i+1]
                attention_dict = self.extract_attention(sample_input, return_all_layers=False)
                
                if 'last_layer' in attention_dict:
                    attention = attention_dict['last_layer']
                    
                    # Average over batch and heads if needed
                    if attention.ndim == 4:  # (batch, heads, T_out, T_in)
                        attention = attention[0].mean(axis=0)  # (T_out, T_in)
                    elif attention.ndim == 3:  # (heads, T_out, T_in)
                        attention = attention.mean(axis=0)  # (T_out, T_in)
                    
                    all_attention_maps.append(attention)
                    sample_count += 1
            
            if n_samples is not None and sample_count >= n_samples:
                break
        
        if not all_attention_maps:
            return {
                'mean_attention': None,
                'std_attention': None,
                'all_attention': None
            }
        
        all_attention_maps = np.array(all_attention_maps)
        
        return {
            'mean_attention': all_attention_maps.mean(axis=0),
            'std_attention': all_attention_maps.std(axis=0),
            'all_attention': all_attention_maps
        }
    
    def compute_attention_entropy(
        self,
        attention_weights: np.ndarray
    ) -> np.ndarray:
        """
        Compute entropy of attention distribution.
        
        High entropy = attention spread across many time steps
        Low entropy = attention focused on few time steps
        
        Args:
            attention_weights: Attention matrix (T_out, T_in)
            
        Returns:
            Entropy for each output time step (T_out,)
        """
        # Add small epsilon to avoid log(0)
        attention_weights = attention_weights + 1e-10
        
        # Normalize to probabilities
        attention_probs = attention_weights / attention_weights.sum(axis=1, keepdims=True)
        
        # Compute entropy
        entropy = -(attention_probs * np.log(attention_probs)).sum(axis=1)
        
        return entropy


def extract_attention_weights(
    model: nn.Module,
    inputs: torch.Tensor,
    device: str = 'cuda'
) -> Optional[np.ndarray]:
    """
    Quick extraction of attention weights from last layer.
    
    Args:
        model: Trained transformer model
        inputs: Input tensor
        device: 'cuda' or 'cpu'
        
    Returns:
        Attention weights (T_out, T_in) or None if not found
        
    Example:
        >>> from basicts.baselines import Informer
        >>> model = Informer(...)
        >>> inputs = torch.randn(1, 12, 207, 1)
        >>> attention = extract_attention_weights(model, inputs)
        >>> if attention is not None:
        ...     print(f"Attention shape: {attention.shape}")
        ...     print(f"Most attended step: {attention[0].argmax()}")
    """
    analyzer = TemporalAttentionAnalyzer(model, device)
    attention_dict = analyzer.extract_attention(inputs, return_all_layers=False)
    
    if 'last_layer' in attention_dict:
        attention = attention_dict['last_layer']
        
        # Average over batch and heads if needed
        if attention.ndim == 4:  # (batch, heads, T_out, T_in)
            attention = attention[0].mean(axis=0)
        elif attention.ndim == 3:  # (heads, T_out, T_in)
            attention = attention.mean(axis=0)
        
        return attention
    
    return None


def compute_temporal_importance(
    attention_weights: np.ndarray,
    top_k: int = 5
) -> Dict:
    """
    Identify most important historical time steps.
    
    Args:
        attention_weights: Attention matrix (n_samples, T_out, T_in) or (T_out, T_in)
        top_k: Number of top time steps to return
        
    Returns:
        Dictionary with:
            - 'top_steps': Most attended time steps for each forecast horizon
            - 'attention_scores': Attention scores for top steps
            - 'mean_attention': Average attention per input time step
    """
    if attention_weights.ndim == 3:
        # Average across samples
        mean_attention = attention_weights.mean(axis=0)  # (T_out, T_in)
    else:
        mean_attention = attention_weights
    
    T_out, T_in = mean_attention.shape
    
    # For each output step, find top-k input steps
    top_steps = []
    attention_scores = []
    
    for t_out in range(T_out):
        top_indices = np.argsort(mean_attention[t_out])[::-1][:top_k]
        top_steps.append(top_indices)
        attention_scores.append(mean_attention[t_out, top_indices])
    
    # Overall importance of each input step (averaged across all output steps)
    overall_importance = mean_attention.mean(axis=0)
    
    return {
        'top_steps': np.array(top_steps),
        'attention_scores': np.array(attention_scores),
        'mean_attention': overall_importance
    }


def analyze_attention_patterns(
    attention_weights: np.ndarray
) -> Dict:
    """
    Analyze patterns in attention weights.
    
    Args:
        attention_weights: Attention matrix (T_out, T_in)
        
    Returns:
        Dictionary with pattern metrics:
            - 'locality': How much attention focuses on recent steps
            - 'periodicity': Whether attention shows periodic patterns
            - 'entropy': Distribution uniformity of attention
    """
    T_out, T_in = attention_weights.shape
    
    # Locality: average distance of attended steps
    time_steps = np.arange(T_in)
    weighted_distances = []
    
    for t_out in range(T_out):
        # Distance from each input step to last input step
        distances = T_in - 1 - time_steps
        # Weighted by attention
        weighted_dist = (attention_weights[t_out] * distances).sum()
        weighted_distances.append(weighted_dist)
    
    locality = 1 - (np.mean(weighted_distances) / T_in)  # Higher = more local
    
    # Entropy: uniformity of attention distribution
    attention_probs = attention_weights / (attention_weights.sum(axis=1, keepdims=True) + 1e-10)
    max_entropy = np.log(T_in)
    entropy = -(attention_probs * np.log(attention_probs + 1e-10)).sum(axis=1)
    normalized_entropy = entropy.mean() / max_entropy
    
    # Periodicity: check if attention has periodic patterns (via FFT)
    mean_attention = attention_weights.mean(axis=0)
    fft = np.fft.fft(mean_attention)
    power_spectrum = np.abs(fft[:T_in//2])
    
    # Find dominant frequency (excluding DC component)
    if len(power_spectrum) > 1:
        dominant_freq_idx = np.argmax(power_spectrum[1:]) + 1
        periodicity_strength = power_spectrum[dominant_freq_idx] / power_spectrum.sum()
    else:
        periodicity_strength = 0.0
    
    return {
        'locality': locality,
        'entropy': normalized_entropy,
        'periodicity': periodicity_strength
    }
