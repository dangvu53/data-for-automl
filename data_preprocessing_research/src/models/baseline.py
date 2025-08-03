"""
Baseline model implementations for comparison.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, classification_report
import logging

try:
    from ..utils.logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class BaselineModel:
    """
    Simple baseline models for text classification comparison.
    """
    
    def __init__(self, model_type: str = "logistic_regression"):
        """
        Initialize baseline model.
        
        Args:
            model_type: Type of baseline model to use
        """
        self.model_type = model_type
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.model = self._create_model(model_type)
        self.is_fitted = False
        
        logger.info(f"BaselineModel initialized with {model_type}")
    
    def _create_model(self, model_type: str):
        """Create the specified model."""
        if model_type == "logistic_regression":
            return LogisticRegression(random_state=42, max_iter=1000)
        elif model_type == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == "naive_bayes":
            return MultinomialNB()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(
        self,
        train_data: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> Dict[str, Any]:
        """
        Train the baseline model.
        
        Args:
            train_data: Training dataset
            text_column: Name of the text column
            target_column: Name of the target column
            
        Returns:
            Training information
        """
        logger.info(f"Training {self.model_type} baseline model")
        
        # Prepare text data
        X_text = train_data[text_column].astype(str)
        y = train_data[target_column]
        
        # Vectorize text
        X = self.vectorizer.fit_transform(X_text)
        
        # Train model
        self.model.fit(X, y)
        self.is_fitted = True
        
        # Calculate training accuracy
        train_pred = self.model.predict(X)
        train_accuracy = accuracy_score(y, train_pred)
        
        logger.info(f"Training completed. Training accuracy: {train_accuracy:.4f}")
        
        return {
            'model_type': self.model_type,
            'training_accuracy': train_accuracy,
            'training_samples': len(train_data),
            'feature_count': X.shape[1]
        }
    
    def predict(self, test_data: pd.DataFrame, text_column: str) -> np.ndarray:
        """
        Make predictions on test data.
        
        Args:
            test_data: Test dataset
            text_column: Name of the text column
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_text = test_data[text_column].astype(str)
        X = self.vectorizer.transform(X_text)
        
        return self.model.predict(X)
    
    def predict_proba(self, test_data: pd.DataFrame, text_column: str) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            test_data: Test dataset
            text_column: Name of the text column
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_text = test_data[text_column].astype(str)
        X = self.vectorizer.transform(X_text)
        
        return self.model.predict_proba(X)
    
    def evaluate(
        self,
        test_data: pd.DataFrame,
        text_column: str,
        target_column: str
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            test_data: Test dataset
            text_column: Name of the text column
            target_column: Name of the target column
            
        Returns:
            Evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Make predictions
        predictions = self.predict(test_data, text_column)
        y_true = test_data[target_column]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, predictions)
        f1_macro = f1_score(y_true, predictions, average='macro')
        f1_weighted = f1_score(y_true, predictions, average='weighted')
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted
        }
        
        logger.info(f"Evaluation completed. Accuracy: {accuracy:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get feature importance for interpretability.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            List of feature importance information
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        feature_names = self.vectorizer.get_feature_names_out()
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            if len(self.model.coef_.shape) == 1:
                importances = np.abs(self.model.coef_)
            else:
                importances = np.abs(self.model.coef_).mean(axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return []
        
        # Get top features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        
        feature_importance = []
        for idx in top_indices:
            feature_importance.append({
                'feature': feature_names[idx],
                'importance': float(importances[idx])
            })
        
        return feature_importance
