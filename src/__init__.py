"""
GNN Uncertainty Quantification and Explainability Benchmark
Main package initialization
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import key modules for easier access
from . import data
from . import models
from . import uncertainty
from . import explainability
from . import utils

__all__ = [
    "data",
    "models",
    "uncertainty",
    "explainability",
    "utils",
]
