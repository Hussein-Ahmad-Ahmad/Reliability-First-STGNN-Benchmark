"""
Graph Construction Utilities
Uses BasicTS's graph utilities where available, extends for additional methods.
"""

import numpy as np
import torch
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
from typing import Optional, Tuple
import sys
from pathlib import Path

# Add BasicTS to path
BASICTS_ROOT = Path(__file__).parent.parent.parent / "BasicTS-master"
sys.path.insert(0, str(BASICTS_ROOT))

from basicts.utils.adjacent_matrix_norm import normalize_adj_mx


def construct_distance_graph(
    distances: np.ndarray,
    sigma: float = 10.0,
    threshold: float = 0.1,
    normalize: bool = True
) -> np.ndarray:
    """
    Construct adjacency matrix from distance matrix using Gaussian kernel.
    
    This is the standard method used in many spatiotemporal GNN papers.
    Reference: DCRNN, Graph WaveNet, etc.
    
    Args:
        distances: Distance matrix [N, N]
        sigma: Gaussian kernel bandwidth
        threshold: Threshold for sparsification (edges < threshold are removed)
        normalize: Whether to apply graph normalization (for spectral GNNs)
    
    Returns:
        Adjacency matrix [N, N]
    """
    # Apply Gaussian kernel: w_ij = exp(-d_ij^2 / sigma^2)
    adj = np.exp(-distances ** 2 / (sigma ** 2))
    
    # Sparsify: remove weak connections
    adj[adj < threshold] = 0
    
    # Self-loops
    np.fill_diagonal(adj, 1.0)
    
    # Normalize if requested (uses BasicTS's normalization)
    if normalize:
        adj = normalize_adj_mx(adj, adj_type='doubletransition')
    
    return adj


def construct_correlation_graph(
    data: np.ndarray,
    method: str = 'pearson',
    threshold: float = 0.5,
    normalize: bool = True
) -> np.ndarray:
    """
    Construct adjacency matrix based on temporal correlation.
    
    Useful for datasets where spatial relationships are not pre-defined
    (e.g., electricity consumption, weather stations).
    
    Args:
        data: Time series data [T, N, C] or [T, N]
        method: Correlation method ('pearson', 'spearman')
        threshold: Correlation threshold (keep edges > threshold)
        normalize: Whether to apply graph normalization
    
    Returns:
        Adjacency matrix [N, N]
    """
    if data.ndim == 3:
        data = data[..., 0]  # Use first feature if multivariate
    
    N = data.shape[1]
    adj = np.zeros((N, N))
    
    # Compute pairwise correlations
    if method == 'pearson':
        for i in range(N):
            for j in range(i, N):
                if i == j:
                    adj[i, j] = 1.0
                else:
                    corr, _ = pearsonr(data[:, i], data[:, j])
                    corr = abs(corr)  # Use absolute correlation
                    if corr > threshold:
                        adj[i, j] = corr
                        adj[j, i] = corr
    elif method == 'spearman':
        from scipy.stats import spearmanr
        corr_matrix, _ = spearmanr(data, axis=0)
        adj = np.abs(corr_matrix)
        adj[adj < threshold] = 0
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Normalize if requested
    if normalize:
        adj = normalize_adj_mx(adj, adj_type='doubletransition')
    
    return adj


def construct_learnable_graph(
    num_nodes: int,
    init_type: str = 'identity',
    sparsity: Optional[float] = None
) -> np.ndarray:
    """
    Construct initial adjacency matrix for learnable graph models.
    
    Models like Graph WaveNet, MTGNN learn the graph during training.
    This provides a reasonable initialization.
    
    Args:
        num_nodes: Number of nodes
        init_type: Initialization type ('identity', 'random', 'uniform')
        sparsity: If provided, randomly sparsify to this density
    
    Returns:
        Initial adjacency matrix [N, N]
    """
    if init_type == 'identity':
        adj = np.eye(num_nodes, dtype=np.float32)
    elif init_type == 'random':
        adj = np.random.rand(num_nodes, num_nodes).astype(np.float32)
        adj = (adj + adj.T) / 2  # Make symmetric
    elif init_type == 'uniform':
        adj = np.ones((num_nodes, num_nodes), dtype=np.float32) / num_nodes
    else:
        raise ValueError(f"Unknown init_type: {init_type}")
    
    # Apply sparsity if requested
    if sparsity is not None:
        mask = np.random.rand(num_nodes, num_nodes) < sparsity
        mask = mask | mask.T  # Keep symmetric
        adj = adj * mask
    
    return adj


def load_or_construct_graph(
    dataset_name: str,
    dataset_path: str,
    construction_method: str = 'distance',
    **kwargs
) -> torch.Tensor:
    """
    Load pre-computed adjacency matrix or construct it on-the-fly.
    
    This is the main entry point for graph construction.
    Follows BasicTS dataset structure conventions.
    
    Args:
        dataset_name: Name of dataset (e.g., 'METR-LA', 'PEMS-BAY')
        dataset_path: Path to dataset directory
        construction_method: 'distance', 'correlation', 'learnable', or 'precomputed'
        **kwargs: Additional arguments for construction methods
    
    Returns:
        Adjacency matrix as torch.Tensor [N, N]
    """
    from pathlib import Path
    import pandas as pd
    
    dataset_dir = Path(dataset_path) / dataset_name
    
    # Try to load pre-computed adjacency
    if construction_method == 'precomputed':
        adj_path = dataset_dir / 'adj_mx.npy'
        if adj_path.exists():
            adj = np.load(adj_path)
            return torch.FloatTensor(adj)
        else:
            raise FileNotFoundError(f"Precomputed adjacency not found: {adj_path}")
    
    # Distance-based (requires distance.csv)
    if construction_method == 'distance':
        distance_path = dataset_dir / 'distance.csv'
        if distance_path.exists():
            distances = pd.read_csv(distance_path, header=None).values
            adj = construct_distance_graph(distances, **kwargs)
            return torch.FloatTensor(adj)
        else:
            # Fall back to precomputed if distance.csv not available
            adj_path = dataset_dir / 'adj_mx.npy'
            if adj_path.exists():
                adj = np.load(adj_path)
                return torch.FloatTensor(adj)
            raise FileNotFoundError(f"Distance file not found: {distance_path}")
    
    # Correlation-based (requires loading data)
    if construction_method == 'correlation':
        import json
        data_path = dataset_dir / 'data.dat'
        desc_path = dataset_dir / 'desc.json'
        
        with open(desc_path, 'r') as f:
            desc = json.load(f)
        
        data = np.memmap(data_path, dtype='float32', mode='r', shape=tuple(desc['shape']))
        adj = construct_correlation_graph(data, **kwargs)
        return torch.FloatTensor(adj)
    
    # Learnable (no pre-computed graph needed)
    if construction_method == 'learnable':
        # Infer num_nodes from data
        import json
        desc_path = dataset_dir / 'desc.json'
        with open(desc_path, 'r') as f:
            desc = json.load(f)
        num_nodes = desc['shape'][1]  # [T, N, C]
        
        adj = construct_learnable_graph(num_nodes, **kwargs)
        return torch.FloatTensor(adj)
    
    raise ValueError(f"Unknown construction method: {construction_method}")


if __name__ == "__main__":
    # Test graph construction
    print("Testing distance-based graph...")
    distances = np.random.rand(10, 10)
    adj = construct_distance_graph(distances, sigma=10.0, threshold=0.1)
    print(f"Distance-based adjacency shape: {adj.shape}")
    print(f"Sparsity: {(adj > 0).sum() / adj.size}")
    
    print("\nTesting correlation-based graph...")
    data = np.random.randn(1000, 10)
    adj = construct_correlation_graph(data, threshold=0.5)
    print(f"Correlation-based adjacency shape: {adj.shape}")
    print(f"Sparsity: {(adj > 0).sum() / adj.size}")
    
    print("\nTesting learnable graph initialization...")
    adj = construct_learnable_graph(10, init_type='identity')
    print(f"Learnable adjacency shape: {adj.shape}")
    print(f"Identity initialization: diagonal sum = {np.trace(adj)}")
