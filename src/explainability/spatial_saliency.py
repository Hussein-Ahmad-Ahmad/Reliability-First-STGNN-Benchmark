"""
Spatial saliency analysis for Graph Neural Networks.
Uses GradCAM to identify which spatial nodes most influence predictions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm


class SpatialSaliencyAnalyzer:
    """
    Analyzes spatial importance using Gradient-weighted Class Activation Mapping (GradCAM).
    
    GradCAM generates visual explanations by computing gradients of the output
    with respect to intermediate feature maps. For GNNs, this identifies which
    spatial nodes contribute most to predictions.
    
    Reference:
        Selvaraju et al. (2017) "Grad-CAM: Visual Explanations from Deep Networks 
        via Gradient-based Localization"
    
    Args:
        model: Trained PyTorch model
        target_layer: Layer to compute gradients for (e.g., last GCN layer)
        device: 'cuda' or 'cpu'
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: str = 'cuda'
    ):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # Store activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks on target layer
        if target_layer is not None:
            self.register_hooks(target_layer)
        else:
            # Auto-detect last convolutional/graph layer
            self.target_layer = self._find_target_layer()
            if self.target_layer is not None:
                self.register_hooks(self.target_layer)
    
    def _find_target_layer(self) -> Optional[nn.Module]:
        """Auto-detect the last graph convolutional layer."""
        target_layer = None
        
        # Common GNN layer types
        gnn_layer_types = (
            'GCNConv', 'ChebConv', 'GraphConv', 'GATConv',
            'SAGEConv', 'GINConv', 'TransformerConv'
        )
        
        for name, module in self.model.named_modules():
            layer_type = module.__class__.__name__
            if any(gnn_type in layer_type for gnn_type in gnn_layer_types):
                target_layer = module
        
        return target_layer
    
    def register_hooks(self, layer: nn.Module):
        """Register forward and backward hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        layer.register_forward_hook(forward_hook)
        layer.register_full_backward_hook(backward_hook)
    
    def generate_saliency_map(
        self,
        inputs: torch.Tensor,
        target_node: Optional[int] = None,
        target_time: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate spatial saliency map for a single input.
        
        Args:
            inputs: Input tensor (1, T, N, F) or (T, N, F)
            target_node: Node index to explain (if None, average over all)
            target_time: Time step to explain (if None, average over all)
            
        Returns:
            Saliency map (N,) indicating importance of each node
        """
        if inputs.dim() == 3:
            inputs = inputs.unsqueeze(0)  # Add batch dimension
        
        inputs = inputs.to(self.device)
        inputs.requires_grad = True
        
        # Forward pass
        outputs = self.model(inputs)
        
        # Handle different output formats
        if isinstance(outputs, dict):
            outputs = outputs['prediction']
        
        # Select target output
        if target_node is not None and target_time is not None:
            target_output = outputs[0, target_time, target_node]
        elif target_node is not None:
            target_output = outputs[0, :, target_node].mean()
        elif target_time is not None:
            target_output = outputs[0, target_time, :].mean()
        else:
            target_output = outputs.mean()
        
        # Backward pass
        self.model.zero_grad()
        target_output.backward()
        
        # Compute saliency
        if self.gradients is not None and self.activations is not None:
            # GradCAM: weight activations by gradients
            weights = self.gradients.mean(dim=(0, 1))  # Average over batch and time
            
            # Weighted combination of activations
            if self.activations.dim() == 4:  # (B, T, N, C)
                weighted_activations = (self.activations[0] * weights).sum(dim=-1)  # (T, N)
                saliency = weighted_activations.mean(dim=0).cpu().numpy()  # (N,)
            elif self.activations.dim() == 3:  # (B, N, C)
                weighted_activations = (self.activations[0] * weights).sum(dim=-1)  # (N,)
                saliency = weighted_activations.cpu().numpy()
            else:
                # Fallback: use input gradients
                saliency = inputs.grad.abs().mean(dim=(0, 1, 3)).cpu().numpy()  # (N,)
        else:
            # Fallback: use input gradients
            saliency = inputs.grad.abs().mean(dim=(0, 1, 3)).cpu().numpy()  # (N,)
        
        # Normalize to [0, 1]
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
        
        return saliency
    
    def analyze_dataset(
        self,
        data_loader: torch.utils.data.DataLoader,
        target_nodes: Optional[List[int]] = None,
        n_samples: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Generate saliency maps for multiple samples.
        
        Args:
            data_loader: Data loader for input data
            target_nodes: List of nodes to analyze (if None, all nodes)
            n_samples: Number of samples to analyze (if None, use all)
            
        Returns:
            Dictionary containing:
                - 'mean_saliency': Average saliency across samples (N,)
                - 'std_saliency': Std of saliency across samples (N,)
                - 'all_saliency': All individual saliency maps (n_samples, N)
        """
        all_saliency_maps = []
        
        sample_count = 0
        for batch_data in tqdm(data_loader, desc="Computing spatial saliency"):
            if isinstance(batch_data, dict):
                inputs = batch_data['inputs']
            else:
                inputs, _ = batch_data
            
            # Process each sample in batch
            for i in range(inputs.size(0)):
                if n_samples is not None and sample_count >= n_samples:
                    break
                
                sample_input = inputs[i:i+1]
                
                if target_nodes is not None:
                    # Average saliency across target nodes
                    node_saliencies = []
                    for node in target_nodes:
                        saliency = self.generate_saliency_map(
                            sample_input,
                            target_node=node
                        )
                        node_saliencies.append(saliency)
                    
                    avg_saliency = np.mean(node_saliencies, axis=0)
                    all_saliency_maps.append(avg_saliency)
                else:
                    # Saliency for all nodes
                    saliency = self.generate_saliency_map(sample_input)
                    all_saliency_maps.append(saliency)
                
                sample_count += 1
            
            if n_samples is not None and sample_count >= n_samples:
                break
        
        all_saliency_maps = np.array(all_saliency_maps)
        
        return {
            'mean_saliency': all_saliency_maps.mean(axis=0),
            'std_saliency': all_saliency_maps.std(axis=0),
            'all_saliency': all_saliency_maps
        }


def gradcam_for_gnn(
    model: nn.Module,
    inputs: torch.Tensor,
    target_layer: Optional[nn.Module] = None,
    target_node: Optional[int] = None,
    device: str = 'cuda'
) -> np.ndarray:
    """
    Quick GradCAM computation for a single input.
    
    Args:
        model: Trained model
        inputs: Input tensor
        target_layer: Layer to compute gradients for
        target_node: Node to explain (if None, average over all)
        device: 'cuda' or 'cpu'
        
    Returns:
        Saliency map (N,)
        
    Example:
        >>> from basicts.baselines import D2STGNN
        >>> model = D2STGNN(...)
        >>> inputs = torch.randn(1, 12, 207, 1)
        >>> saliency = gradcam_for_gnn(model, inputs, target_node=0)
        >>> print(f"Most important nodes: {saliency.argsort()[-5:]}")
    """
    analyzer = SpatialSaliencyAnalyzer(model, target_layer, device)
    return analyzer.generate_saliency_map(inputs, target_node=target_node)


def compute_node_importance_ranking(
    saliency_maps: np.ndarray,
    top_k: int = 10
) -> Dict:
    """
    Rank nodes by importance based on saliency maps.
    
    Args:
        saliency_maps: Saliency values (n_samples, n_nodes) or (n_nodes,)
        top_k: Number of top nodes to return
        
    Returns:
        Dictionary with:
            - 'top_nodes': Indices of top-K most important nodes
            - 'importance_scores': Importance scores for top-K nodes
            - 'all_scores': Importance scores for all nodes
    """
    if saliency_maps.ndim == 2:
        # Average across samples
        importance_scores = saliency_maps.mean(axis=0)
    else:
        importance_scores = saliency_maps
    
    # Rank nodes
    ranked_indices = np.argsort(importance_scores)[::-1]
    
    return {
        'top_nodes': ranked_indices[:top_k],
        'importance_scores': importance_scores[ranked_indices[:top_k]],
        'all_scores': importance_scores
    }


def compute_spatial_consistency(
    saliency_maps: np.ndarray,
    adjacency_matrix: Optional[np.ndarray] = None
) -> Dict:
    """
    Measure consistency of spatial saliency patterns.
    
    Args:
        saliency_maps: Saliency values (n_samples, n_nodes)
        adjacency_matrix: Graph adjacency matrix (if available)
        
    Returns:
        Dictionary with consistency metrics:
            - 'temporal_consistency': Consistency across samples
            - 'spatial_consistency': Consistency among neighbors (if adj provided)
    """
    # Temporal consistency: how stable are saliency patterns across samples?
    temporal_consistency = 1 - saliency_maps.std(axis=0).mean()
    
    metrics = {
        'temporal_consistency': temporal_consistency
    }
    
    # Spatial consistency: do neighboring nodes have similar saliency?
    if adjacency_matrix is not None:
        mean_saliency = saliency_maps.mean(axis=0)
        
        # For each node, compute similarity with neighbors
        neighbor_similarities = []
        for i in range(len(mean_saliency)):
            neighbors = np.where(adjacency_matrix[i] > 0)[0]
            if len(neighbors) > 0:
                node_saliency = mean_saliency[i]
                neighbor_saliency = mean_saliency[neighbors].mean()
                similarity = 1 - abs(node_saliency - neighbor_saliency)
                neighbor_similarities.append(similarity)
        
        if neighbor_similarities:
            metrics['spatial_consistency'] = np.mean(neighbor_similarities)
    
    return metrics
