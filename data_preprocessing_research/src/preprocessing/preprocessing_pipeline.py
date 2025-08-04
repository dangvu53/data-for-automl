#!/usr/bin/env python3
"""
Preprocessing Pipeline Integration Module

This module provides a unified interface for integrating multiple preprocessing
techniques into a single pipeline:
1. Outlier Detection
2. Duplicate Removal
3. Imbalance Handling

Each technique can be enabled or disabled as needed, and the order of processing
can be specified.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from tqdm import tqdm
import importlib
import sys
import inspect

# Setup logging
logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    """
    A preprocessing pipeline that integrates multiple preprocessing techniques.
    
    Attributes:
        steps (List[Dict]): List of preprocessing steps.
        verbose (bool): Whether to display verbose output.
    """
    
    def __init__(
        self,
        steps: List[Dict] = None,
        verbose: bool = True
    ):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            steps (List[Dict]): List of preprocessing steps. Each step is a dictionary with:
                - 'type': The type of preprocessor ('outlier', 'duplicate', 'imbalance')
                - 'params': Parameters for the preprocessor
                - 'enabled': Whether the step is enabled
            verbose (bool): Whether to display verbose output.
        """
        self.steps = steps or []
        self.verbose = verbose
        self.preprocessors = []
        
        # Validate steps
        self._validate_steps()
        
    def _validate_steps(self):
        """
        Validate the preprocessing steps.
        """
        valid_types = ['outlier', 'duplicate', 'imbalance']
        
        for i, step in enumerate(self.steps):
            # Ensure all required keys are present
            if 'type' not in step:
                raise ValueError(f"Step {i} is missing 'type'")
                
            # Validate type
            if step['type'] not in valid_types:
                raise ValueError(f"Invalid type in step {i}: {step['type']}. "
                                  f"Must be one of {valid_types}")
                
            # Ensure params is a dictionary
            if 'params' in step and not isinstance(step['params'], dict):
                raise ValueError(f"'params' in step {i} must be a dictionary")
                
            # Set default values
            if 'enabled' not in step:
                step['enabled'] = True
                
            if 'params' not in step:
                step['params'] = {}
                
    def add_step(
        self,
        step_type: str,
        params: Dict = None,
        enabled: bool = True
    ):
        """
        Add a preprocessing step to the pipeline.
        
        Args:
            step_type (str): The type of preprocessor ('outlier', 'duplicate', 'imbalance')
            params (Dict): Parameters for the preprocessor
            enabled (bool): Whether the step is enabled
        """
        step = {
            'type': step_type,
            'params': params or {},
            'enabled': enabled
        }
        
        self.steps.append(step)
        self._validate_steps()
        
    def _import_preprocessor(self, step_type: str) -> Any:
        """
        Import the preprocessor class for a given step type.
        
        Args:
            step_type (str): The type of preprocessor.
            
        Returns:
            Any: The preprocessor class.
        """
        # Define mapping from step type to module and class
        mapping = {
            'outlier': ('outlier_detection', 'OutlierDetectionPreprocessor'),
            'duplicate': ('duplicate_removal_enhanced', 'DuplicateRemovalPreprocessor'),
            'imbalance': ('imbalance_handling', 'ImbalanceHandlingPreprocessor')
        }
        
        module_name, class_name = mapping.get(step_type, (None, None))
        
        if module_name is None:
            raise ValueError(f"Unknown step type: {step_type}")
            
        try:
            # Import the module from src.preprocessing
            module = importlib.import_module(f"src.preprocessing.{module_name}")
            
            # Get the preprocessor class
            preprocessor_class = getattr(module, class_name)
            
            return preprocessor_class
            
        except ImportError as e:
            logger.error(f"Failed to import {module_name}: {e}")
            raise ImportError(f"Failed to import {module_name}. Make sure it's installed.") from e
        except AttributeError as e:
            logger.error(f"Failed to find {class_name} in {module_name}: {e}")
            raise AttributeError(f"Failed to find {class_name} in {module_name}.") from e
            
    def _initialize_preprocessors(self):
        """
        Initialize the preprocessors for all enabled steps.
        """
        self.preprocessors = []
        
        for step in self.steps:
            if not step['enabled']:
                continue
                
            # Import the preprocessor class
            preprocessor_class = self._import_preprocessor(step['type'])
            
            # Get valid parameters for this preprocessor
            valid_params = {}
            for param_name, param in inspect.signature(preprocessor_class.__init__).parameters.items():
                if param_name != 'self' and param_name in step['params']:
                    valid_params[param_name] = step['params'][param_name]
                    
            # Add verbose parameter if it's accepted by the preprocessor
            if 'verbose' in inspect.signature(preprocessor_class.__init__).parameters and 'verbose' not in valid_params:
                valid_params['verbose'] = self.verbose
                
            # Initialize the preprocessor
            preprocessor = preprocessor_class(**valid_params)
            
            self.preprocessors.append((step['type'], preprocessor))
            
    def transform(self, df: pd.DataFrame, is_validation_or_test: bool = False) -> pd.DataFrame:
        """
        Transform the input dataframe using all enabled preprocessing steps.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            is_validation_or_test (bool): Whether this is validation or test data. 
                                         If True, only apply transformations that don't change sample count.
            
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        if len(self.steps) == 0:
            logger.warning("No preprocessing steps defined")
            return df
            
        # Initialize preprocessors
        self._initialize_preprocessors()
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Apply each preprocessor in order
        for step_type, preprocessor in self.preprocessors:
            # Skip outlier detection, imbalance handling and duplicate removal for validation/test sets
            # as they would change the number of samples
            if is_validation_or_test and step_type in ['outlier', 'imbalance', 'duplicate']:
                if self.verbose:
                    logger.info(f"Skipping {step_type} preprocessing for validation/test data")
                continue
                
            if self.verbose:
                logger.info(f"Applying {step_type} preprocessing...")
                
            try:
                result_df = preprocessor.transform(result_df)
                
                if self.verbose:
                    logger.info(f"After {step_type} preprocessing: {len(result_df)} samples")
                    
            except Exception as e:
                logger.error(f"Error applying {step_type} preprocessing: {e}")
                raise
                
        return result_df
        
    def fit(self, df: pd.DataFrame):
        """
        Fit the preprocessing pipeline on the training data.
        This learns parameters from the training data that will be applied to all datasets.
        
        Args:
            df (pd.DataFrame): The training dataframe.
        """
        if len(self.steps) == 0:
            logger.warning("No preprocessing steps defined")
            return
            
        # Initialize preprocessors
        self._initialize_preprocessors()
        
        # Fit each preprocessor on the training data
        for step_type, preprocessor in self.preprocessors:
            if self.verbose:
                logger.info(f"Fitting {step_type} preprocessor on training data...")
                
            try:
                # Check if the preprocessor has a fit method
                if hasattr(preprocessor, 'fit'):
                    preprocessor.fit(df)
                elif hasattr(preprocessor, 'fit_transform'):
                    # If only fit_transform is available, call it but don't use the result
                    _ = preprocessor.fit_transform(df)
                    
                if self.verbose:
                    logger.info(f"Fitted {step_type} preprocessor on {len(df)} samples")
                    
            except Exception as e:
                logger.error(f"Error fitting {step_type} preprocessor: {e}")
                raise
    
    def fit_transform(self, df: pd.DataFrame, is_validation_or_test: bool = False) -> pd.DataFrame:
        """
        Fit the pipeline on the data and then transform it.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            is_validation_or_test (bool): Whether this is validation or test data.
            
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        if not is_validation_or_test:
            self.fit(df)
        return self.transform(df, is_validation_or_test=is_validation_or_test)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics from all preprocessors.
        
        Returns:
            Dict[str, Any]: Statistics from all preprocessors.
        """
        stats = {}
        
        for step_type, preprocessor in self.preprocessors:
            if hasattr(preprocessor, 'get_stats'):
                stats[step_type] = preprocessor.get_stats()
                
        return stats

# Example usage and demonstration
def demo():
    """
    Demonstrate the PreprocessingPipeline with a small example.
    """
    # Create a small example dataset
    data = {
        'text': [
            "This is a sample text from class 0",
            "Another example from class 0",
            "Yet another class 0 example",
            "This is a sample text from class 0",  # Duplicate
            "Class 0 text with some outlier features " * 10,  # Outlier
            "Another class 0 example with different words",
            "Example text from class 0 with specific content",
            "Class 0 text talking about specific topics",
            "Class 1 example with very different content",
            "Another text from class 1",
            "Class 2 example text for demonstration",
        ],
        'label': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2]  # Imbalanced classes: 8/2/1
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df[['text', 'label']].groupby('label').count())
    print(f"Original shape: {df.shape}")
    
    # Create a preprocessing pipeline
    pipeline = PreprocessingPipeline(
        steps=[
            {
                'type': 'outlier',
                'params': {
                    'strategy': 'isolation-forest',
                    'text_column': 'text',
                    'contamination': 0.1
                },
                'enabled': True
            },
            {
                'type': 'duplicate',
                'params': {
                    'strategy': 'exact',
                    'text_column': 'text',
                    'label_column': 'label'
                },
                'enabled': True
            },
            {
                'type': 'imbalance',
                'params': {
                    'strategy': 'random-oversampling',
                    'text_column': 'text',
                    'label_column': 'label'
                },
                'enabled': True
            }
        ],
        verbose=True
    )
    
    # Apply the pipeline
    result_df = pipeline.fit_transform(df)
    
    print("\nAfter preprocessing:")
    print(result_df[['text', 'label']].groupby('label').count())
    print(f"New shape: {result_df.shape}")
    
    # Get statistics
    stats = pipeline.get_stats()
    print("\nPreprocessing Statistics:")
    for step_type, step_stats in stats.items():
        print(f"\n{step_type.capitalize()} statistics:")
        for key, value in step_stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    demo()
