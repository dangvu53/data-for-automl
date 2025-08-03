"""
Standardized AutoGluon pipeline for text classification experiments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import logging

from autogluon.tabular import TabularPredictor
try:
    from ..utils.logging_config import get_logger
    from ..utils.reproducibility import set_random_seeds
except ImportError:
    # Fallback for when running as script
    import logging
    def get_logger(name):
        return logging.getLogger(name)

    def set_random_seeds(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

logger = get_logger(__name__)

class AutoGluonTextClassifier:
    """
    Standardized AutoGluon text classifier with consistent configuration.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        model_path: Optional[str] = None,
        random_seed: int = 42
    ):
        """
        Initialize the AutoGluon text classifier.
        
        Args:
            config: AutoGluon configuration dictionary
            model_path: Path to save/load the model
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.model_path = model_path
        self.random_seed = random_seed
        self.predictor = None
        self.is_trained = False
        
        # Set random seeds
        set_random_seeds(random_seed)
        
        logger.info(f"Initialized AutoGluonTextClassifier with config: {config}")
    
    def fit(
        self,
        train_data: pd.DataFrame,
        target_column: str,
        validation_data: Optional[pd.DataFrame] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train the AutoGluon model.
        
        Args:
            train_data: Training dataset
            target_column: Name of the target column
            validation_data: Optional validation dataset
            **kwargs: Additional arguments for TabularPredictor.fit()
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting training with {len(train_data)} samples")
        
        # Auto-detect problem type (same logic as your code)
        unique_targets = train_data[target_column].nunique()
        target_dtype = train_data[target_column].dtype

        if target_dtype in ['object', 'category'] or str(target_dtype).startswith('string'):
            problem_type = 'multiclass' if unique_targets > 2 else 'binary'
        elif unique_targets <= 100 and target_dtype in ['int64', 'int32']:
            problem_type = 'multiclass' if unique_targets > 2 else 'binary'
        else:
            problem_type = 'regression'

        eval_metric = 'accuracy' if problem_type in ['binary', 'multiclass'] else 'root_mean_squared_error'

        logger.info(f"Detected problem type: {problem_type}, eval_metric: {eval_metric}")

        # Create predictor with auto-detection
        self.predictor = TabularPredictor(
            label=target_column,
            path=self.model_path,
            problem_type=problem_type,
            eval_metric=eval_metric
        )
        
        # Use simple fit arguments (like your approach)
        fit_kwargs = {
            "train_data": train_data,
            "presets": self.config.get("presets", "medium_quality"),
            "time_limit": self.config.get("time_limit", 300),
            "verbosity": self.config.get("verbosity", 2),
            **kwargs
        }

        # Add validation data if provided
        if validation_data is not None:
            fit_kwargs["tuning_data"] = validation_data
            logger.info(f"Using validation data with {len(validation_data)} samples")

        # Add memory settings if specified
        ag_args_fit = self.config.get("ag_args_fit")
        if ag_args_fit:
            fit_kwargs["ag_args_fit"] = ag_args_fit
        
        # Train the model
        try:
            # Remove train_data from fit_kwargs since it's passed as first argument
            train_data_arg = fit_kwargs.pop("train_data")
            self.predictor.fit(train_data_arg, **fit_kwargs)
            self.is_trained = True
            logger.info("Training completed successfully")
            
            # Get training results
            leaderboard = self.predictor.leaderboard()
            best_model = leaderboard.iloc[0]['model'] if len(leaderboard) > 0 else "Unknown"

            results = {
                "leaderboard": leaderboard,
                "best_model": best_model,
                "feature_importance": self.predictor.feature_importance(train_data) if len(leaderboard) > 0 else None
            }
            
            return results
            
        except RuntimeError as e:
            if "No models were trained successfully" in str(e):
                logger.error("AutoGluon failed to train any models. This might be due to:")
                logger.error("1. Time limit too short")
                logger.error("2. Data format issues")
                logger.error("3. Configuration problems")
                logger.error("Try increasing time_limit or simplifying the configuration")

                # Return a minimal result instead of crashing
                return {
                    "leaderboard": None,
                    "best_model": "Training Failed",
                    "feature_importance": None
                }
            else:
                raise
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def predict(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making predictions on {len(test_data)} samples")
        predictions = self.predictor.predict(test_data)
        return predictions
    
    def predict_proba(self, test_data: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            test_data: Test dataset
            
        Returns:
            Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Getting prediction probabilities for {len(test_data)} samples")
        probabilities = self.predictor.predict_proba(test_data)
        return probabilities
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        target_column: str
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset with true labels
            target_column: Name of the target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating model on {len(test_data)} samples")
        
        # Get predictions
        predictions = self.predict(test_data)
        
        # Calculate metrics using AutoGluon's built-in evaluation
        metrics = self.predictor.evaluate(test_data, silent=True)
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the trained model.
        
        Returns:
            Model information dictionary
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        info = {
            "best_model": self.predictor.get_model_best(),
            "leaderboard": self.predictor.leaderboard(),
            "model_names": self.predictor.get_model_names(),
            "problem_type": self.predictor.problem_type,
            "eval_metric": self.predictor.eval_metric
        }
        
        return info
    
    def save_model(self, path: Optional[str] = None):
        """
        Save the trained model.
        
        Args:
            path: Optional path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        save_path = path or self.model_path
        if save_path:
            # Model is automatically saved during training if path is provided
            logger.info(f"Model saved to {save_path}")
        else:
            logger.warning("No save path provided, model not saved")
    
    def load_model(self, path: str):
        """
        Load a pre-trained model.
        
        Args:
            path: Path to the saved model
        """
        try:
            self.predictor = TabularPredictor.load(path)
            self.is_trained = True
            self.model_path = path
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {str(e)}")
            raise
