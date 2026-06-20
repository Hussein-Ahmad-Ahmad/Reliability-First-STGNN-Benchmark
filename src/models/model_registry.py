"""
Model Registry - Wraps BasicTS model implementations
This module provides a unified interface to all models in BasicTS-master/baselines/
"""

import sys
from pathlib import Path
from typing import Dict, Any
import importlib

# Add BasicTS to path
BASICTS_ROOT = Path(__file__).parent.parent.parent / "BasicTS-master"
sys.path.insert(0, str(BASICTS_ROOT))


# ============================================================================
# MODEL REGISTRY
# ============================================================================

MODEL_REGISTRY = {
    # 1. Graph-Based Models (STGNNs)
    'STGCN': {
        'path': 'baselines.STGCN',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'stgnn',
        'description': 'Temporal CNN + Graph Conv on fixed adjacency'
    },
    'DCRNN': {
        'path': 'baselines.DCRNN',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'stgnn',
        'description': 'Diffusion Graph Conv + GRU'
    },
    'MTGNN': {
        'path': 'baselines.MTGNN',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'stgnn',
        'description': 'Adaptive graph learning + temporal convolutions'
    },
    'D2STGNN': {
        'path': 'baselines.D2STGNN',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'stgnn',
        'description': 'Decoupled dynamic spatial-temporal GNN'
    },
    'MegaCRN': {
        'path': 'baselines.MegaCRN',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'stgnn',
        'description': 'Meta-graph convolutional recurrent network',
        'notes': 'Original has dropout=0.0, needs modification for MC Dropout UQ'
    },
    
    # 2. Sequence-First Models (Transformers)
    'Autoformer': {
        'path': 'baselines.Autoformer',
        'arch': 'arch',
        'runner': None,
        'category': 'transformer',
        'description': 'Decomposition-based with Auto-Correlation mechanism'
    },
    'PatchTST': {
        'path': 'baselines.PatchTST',
        'arch': 'arch',
        'runner': None,
        'category': 'transformer',
        'description': 'Patch-based, channel-independent Transformer'
    },
    'iTransformer': {
        'path': 'baselines.iTransformer',
        'arch': 'arch',
        'runner': None,
        'category': 'transformer',
        'description': 'Inverted Transformer treating channels as sequence'
    },
    
    # 3. Simplicity & Efficiency Models
    'DLinear': {
        'path': 'baselines.DLinear',
        'arch': 'arch',
        'runner': None,
        'category': 'simple_efficient',
        'description': 'Linear trend/seasonal decomposition (LTSF-Linear)'
    },
    'STID': {
        'path': 'baselines.STID',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'simple_efficient',
        'description': 'MLP with spatial & temporal identity embeddings'
    },
    'ModernTCN': {
        'path': 'baselines.ModernTCN',
        'arch': 'arch',
        'runner': None,
        'category': 'simple_efficient',
        'description': 'Pure convolutional architecture'
    },
    'STNorm': {
        'path': 'baselines.STNorm',
        'arch': 'arch',
        'runner': 'runner',
        'category': 'simple_efficient',
        'description': 'Temporal CNN with spatial/temporal normalization'
    }
}


def load_model_architecture(model_name: str):
    """
    Load model architecture class from BasicTS baselines.
    
    Args:
        model_name: Name of model (e.g., 'STGCN', 'D2STGNN')
    
    Returns:
        Model class
    
    Example:
        >>> model_cls = load_model_architecture('D2STGNN')
        >>> model = model_cls(**config)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
    
    model_info = MODEL_REGISTRY[model_name]
    arch_module_path = f"{model_info['path']}.{model_info['arch']}"
    
    try:
        arch_module = importlib.import_module(arch_module_path)
        # Model class name is usually the model name itself
        model_cls = getattr(arch_module, model_name)
        return model_cls
    except (ImportError, AttributeError) as e:
        raise ImportError(f"Failed to load model '{model_name}' from '{arch_module_path}': {e}")


def load_model_runner(model_name: str):
    """
    Load custom runner for model (if exists).
    
    Args:
        model_name: Name of model
    
    Returns:
        Runner class or None (use default BaseTimeSeriesForecastingRunner)
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    
    model_info = MODEL_REGISTRY[model_name]
    if model_info['runner'] is None:
        return None  # Use default runner
    
    runner_module_path = f"{model_info['path']}.{model_info['runner']}"
    
    try:
        runner_module = importlib.import_module(runner_module_path)
        # Runner class name is usually ModelName + Runner
        runner_cls = getattr(runner_module, f"{model_name}Runner")
        return runner_cls
    except (ImportError, AttributeError):
        # If custom runner not found, return None (use default)
        return None


def get_model_info(model_name: str) -> Dict[str, Any]:
    """Get metadata about a model."""
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model '{model_name}' not found in registry")
    return MODEL_REGISTRY[model_name]


def list_available_models(category: str = None) -> list:
    """
    List all available models, optionally filtered by category.
    
    Args:
        category: Filter by category ('stgnn', 'transformer', 'simple_efficient')
    
    Returns:
        List of model names
    """
    if category is None:
        return list(MODEL_REGISTRY.keys())
    
    return [name for name, info in MODEL_REGISTRY.items() if info['category'] == category]


def create_model_from_config(model_name: str, config: Dict) -> Any:
    """
    Create model instance from configuration.
    
    This is a convenience function that loads the model architecture
    and instantiates it with the provided config.
    
    Args:
        model_name: Name of model
        config: Model configuration dictionary
    
    Returns:
        Model instance
    """
    model_cls = load_model_architecture(model_name)
    
    # Extract model-specific parameters from config
    if 'MODEL' in config:
        model_params = config['MODEL'].get('PARAM', {})
    else:
        model_params = config
    
    return model_cls(**model_params)


# ============================================================================
# MODEL CONFIGURATION HELPERS
# ============================================================================

def get_default_model_config(model_name: str, dataset_name: str = None) -> Dict:
    """
    Get default configuration for a model.
    
    This reads from config/models.yaml and provides sensible defaults.
    
    Args:
        model_name: Name of model
        dataset_name: Optional dataset name for dataset-specific tuning
    
    Returns:
        Configuration dictionary
    """
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path, 'r') as f:
        models_config = yaml.safe_load(f)
    
    if model_name not in models_config['models']:
        raise ValueError(f"No config found for model '{model_name}'")
    
    return models_config['models'][model_name]


if __name__ == "__main__":
    # Test model loading
    print("Available models by category:")
    print("\n1. STGNNs (Graph-Based):", list_available_models('stgnn'))
    print("\n2. Transformers (Sequence-First):", list_available_models('transformer'))
    print("\n3. Simple & Efficient:", list_available_models('simple_efficient'))
    
    print("\n" + "="*60)
    print("Testing model loading...")
    test_models = ['D2STGNN', 'iTransformer', 'STID', 'STNorm']
    for model_name in test_models:
        try:
            model_cls = load_model_architecture(model_name)
            info = get_model_info(model_name)
            print(f"✓ {model_name:15s} - {info['description']}")
        except Exception as e:
            print(f"✗ {model_name:15s} - FAILED: {e}")
