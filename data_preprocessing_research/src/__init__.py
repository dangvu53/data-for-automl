"""
Data Preprocessing Research Framework

A comprehensive framework for studying the impact of data preprocessing
strategies on AutoML performance for text classification tasks.
"""

__version__ = "0.1.0"
__author__ = "Research Team"

from .utils.logging_config import setup_logging
from .utils.reproducibility import set_random_seeds

# Initialize logging
setup_logging()

# Set default random seeds for reproducibility
set_random_seeds(42)
