"""
Explainability module for spatiotemporal forecasting models.

Provides methods for understanding model predictions:
- Spatial Saliency: GradCAM for identifying important spatial locations
- Temporal Attention: Extracting and analyzing attention weights
- Feature Importance: Integrated Gradients, SHAP, permutation importance
"""

from .spatial_saliency import SpatialSaliencyAnalyzer, gradcam_for_gnn
from .temporal_attention import TemporalAttentionAnalyzer, extract_attention_weights
from .feature_importance import (
    IntegratedGradients,
    compute_feature_importance,
    permutation_importance
)

__all__ = [
    # Spatial Saliency
    'SpatialSaliencyAnalyzer',
    'gradcam_for_gnn',
    
    # Temporal Attention
    'TemporalAttentionAnalyzer',
    'extract_attention_weights',
    
    # Feature Importance
    'IntegratedGradients',
    'compute_feature_importance',
    'permutation_importance',
]
