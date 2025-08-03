"""
Reproducibility utilities for ensuring consistent experimental results.
"""

import random
import os
import numpy as np
import logging

logger = logging.getLogger(__name__)

def set_random_seeds(seed: int = 42):
    """
    Set random seeds for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # Environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try to set PyTorch seeds if available
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info(f"Set PyTorch random seeds to {seed}")
    except ImportError:
        logger.debug("PyTorch not available, skipping PyTorch seed setting")
    
    # Try to set TensorFlow seeds if available
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        logger.info(f"Set TensorFlow random seed to {seed}")
    except ImportError:
        logger.debug("TensorFlow not available, skipping TensorFlow seed setting")
    
    logger.info(f"Set random seeds to {seed} for reproducibility")

def get_environment_info():
    """
    Get information about the current environment for reproducibility tracking.
    
    Returns:
        dict: Environment information
    """
    import platform
    import sys
    
    env_info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "architecture": platform.architecture(),
        "machine": platform.machine(),
    }
    
    # Add package versions
    try:
        from autogluon.tabular import TabularPredictor
        env_info["autogluon_version"] = "1.4.0+"  # Fallback version
    except ImportError:
        pass
    
    try:
        import sklearn
        env_info["sklearn_version"] = sklearn.__version__
    except ImportError:
        pass
    
    try:
        import pandas as pd
        env_info["pandas_version"] = pd.__version__
    except ImportError:
        pass
    
    try:
        import numpy as np
        env_info["numpy_version"] = np.__version__
    except ImportError:
        pass
    
    return env_info
