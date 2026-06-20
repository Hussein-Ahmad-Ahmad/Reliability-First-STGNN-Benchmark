"""
Data loader module for multi-dataset support
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
import yaml


class SpatioTemporalDataset(Dataset):
    """
    Universal dataset class for spatiotemporal forecasting
    Supports both traffic and electricity datasets
    """
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        split: str = "train",
        lookback: int = 12,
        horizon: int = 12,
        normalize: bool = True,
        scaler_type: str = "standard"
    ):
        """
        Args:
            data_path: Path to dataset directory
            dataset_name: Name of dataset (METR-LA, PEMS-BAY, ETTm1, etc.)
            split: 'train', 'val', 'calibration', or 'test'
            lookback: Number of historical timesteps
            horizon: Number of prediction timesteps
            normalize: Whether to apply normalization
            scaler_type: 'standard' or 'minmax'
        """
        self.data_path = Path(data_path)
        self.dataset_name = dataset_name
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.normalize = normalize
        self.scaler_type = scaler_type
        
        # Load data
        self.data, self.adj_matrix = self._load_data()
        
        # Get split indices
        self.indices = self._get_split_indices()
        
        # Fit scaler on training data
        if normalize:
            if split == "train":
                self._fit_scaler()
            self.data = self._transform(self.data)
    
    def _load_data(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Load dataset from disk"""
        dataset_dir = self.data_path / self.dataset_name
        
        # Load main data
        if (dataset_dir / "data.npy").exists():
            data = np.load(dataset_dir / "data.npy")
        elif (dataset_dir / "data.h5").exists():
            import h5py
            with h5py.File(dataset_dir / "data.h5", 'r') as f:
                data = f['data'][:]
        else:
            raise FileNotFoundError(f"No data file found in {dataset_dir}")
        
        # Load adjacency matrix (if exists)
        adj_matrix = None
        if (dataset_dir / "adj_mx.npy").exists():
            adj_matrix = np.load(dataset_dir / "adj_mx.npy")
        elif (dataset_dir / "distance.csv").exists():
            # Construct adjacency from distance matrix
            distances = pd.read_csv(dataset_dir / "distance.csv", header=None).values
            adj_matrix = self._distance_to_adjacency(distances)
        
        return data, adj_matrix
    
    def _distance_to_adjacency(self, distances: np.ndarray, sigma: float = 10.0, threshold: float = 0.1) -> np.ndarray:
        """Convert distance matrix to adjacency using Gaussian kernel"""
        adj = np.exp(-distances ** 2 / (sigma ** 2))
        adj[adj < threshold] = 0
        return adj
    
    def _get_split_indices(self) -> Tuple[int, int]:
        """Get start and end indices for current split"""
        total_len = len(self.data) - self.lookback - self.horizon + 1
        
        # Standard split ratios: 70% train, 10% val, 10% calib, 10% test
        train_size = int(total_len * 0.7)
        val_size = int(total_len * 0.1)
        calib_size = int(total_len * 0.1)
        
        if self.split == "train":
            start, end = 0, train_size
        elif self.split == "val":
            start, end = train_size, train_size + val_size
        elif self.split == "calibration":
            start, end = train_size + val_size, train_size + val_size + calib_size
        elif self.split == "test":
            start, end = train_size + val_size + calib_size, total_len
        else:
            raise ValueError(f"Unknown split: {self.split}")
        
        return start, end
    
    def _fit_scaler(self):
        """Fit scaler on training data"""
        if self.scaler_type == "standard":
            self.mean = np.mean(self.data[:self.indices[1]], axis=(0, 1), keepdims=True)
            self.std = np.std(self.data[:self.indices[1]], axis=(0, 1), keepdims=True)
            self.std[self.std == 0] = 1.0  # Avoid division by zero
        elif self.scaler_type == "minmax":
            self.min = np.min(self.data[:self.indices[1]], axis=(0, 1), keepdims=True)
            self.max = np.max(self.data[:self.indices[1]], axis=(0, 1), keepdims=True)
            self.max[self.max == self.min] = self.min[self.max == self.min] + 1.0
    
    def _transform(self, data: np.ndarray) -> np.ndarray:
        """Apply normalization"""
        if self.scaler_type == "standard":
            return (data - self.mean) / self.std
        elif self.scaler_type == "minmax":
            return (data - self.min) / (self.max - self.min)
        return data
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization"""
        if not self.normalize:
            return data
        
        if self.scaler_type == "standard":
            return data * self.std + self.mean
        elif self.scaler_type == "minmax":
            return data * (self.max - self.min) + self.min
        return data
    
    def __len__(self) -> int:
        return self.indices[1] - self.indices[0]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with keys:
                - 'x': Input sequence [lookback, num_nodes, num_features]
                - 'y': Target sequence [horizon, num_nodes, num_features]
                - 'adj': Adjacency matrix (if available)
        """
        actual_idx = self.indices[0] + idx
        
        x = self.data[actual_idx : actual_idx + self.lookback]
        y = self.data[actual_idx + self.lookback : actual_idx + self.lookback + self.horizon]
        
        batch = {
            'x': torch.FloatTensor(x),
            'y': torch.FloatTensor(y),
        }
        
        if self.adj_matrix is not None:
            batch['adj'] = torch.FloatTensor(self.adj_matrix)
        
        return batch


def create_dataloaders(
    config_path: str,
    dataset_name: str,
    batch_size: int = 64,
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for all splits
    
    Args:
        config_path: Path to datasets.yaml
        dataset_name: Name of dataset
        batch_size: Batch size
        num_workers: Number of dataloader workers
    
    Returns:
        Dictionary with 'train', 'val', 'calibration', 'test' dataloaders
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    dataset_config = config['datasets'][dataset_name]
    
    dataloaders = {}
    for split in ['train', 'val', 'calibration', 'test']:
        dataset = SpatioTemporalDataset(
            data_path=dataset_config['path'],
            dataset_name=dataset_name,
            split=split,
            lookback=dataset_config.get('lookback', 12),
            horizon=dataset_config.get('horizon', 12),
            normalize=True
        )
        
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True
        )
    
    return dataloaders


if __name__ == "__main__":
    # Test data loading
    dataset = SpatioTemporalDataset(
        data_path="d:/original/BasicTS-master/datasets",
        dataset_name="METR-LA",
        split="train",
        lookback=12,
        horizon=12
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Sample batch: {dataset[0].keys()}")
    print(f"X shape: {dataset[0]['x'].shape}")
    print(f"Y shape: {dataset[0]['y'].shape}")
    if 'adj' in dataset[0]:
        print(f"Adjacency shape: {dataset[0]['adj'].shape}")
