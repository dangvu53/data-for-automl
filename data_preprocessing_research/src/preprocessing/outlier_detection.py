#!/usr/bin/env python3
"""
Outlier Detection and Removal Module

This module provides functionality to detect and remove outliers in datasets
using various outlier detection strategies:
1. Z-scores: Remove samples that are n standard deviations away from the mean
2. Inter Quartile Range (IQR): Remove samples outside IQR bounds
3. Local Outlier Factor (LOF): Density-based outlier detection
4. Isolation Forest: Ensemble-based outlier detection

The module can be used as a preprocessing step in a machine learning pipeline.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import scipy.stats as stats

# Setup logging
logger = logging.getLogger(__name__)

class OutlierDetectionPreprocessor:
    """
    Preprocessor to detect and remove outliers from datasets.
    
    Attributes:
        strategy (str): The outlier detection strategy to use.
        text_column (str): The column containing the text data.
        feature_columns (list): The columns to use for outlier detection.
        threshold (float): The threshold for outlier detection.
        contamination (float): The expected proportion of outliers in the dataset.
        verbose (bool): Whether to display verbose output.
    """
    
    def __init__(
        self, 
        strategy: str = 'z-score',
        text_column: str = 'text',
        feature_columns: Optional[List[str]] = None,
        threshold: float = 3.0,
        contamination: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the OutlierDetectionPreprocessor.
        
        Args:
            strategy (str): The outlier detection strategy to use.
                Options: 'z-score', 'iqr', 'lof', 'isolation-forest'
            text_column (str): The column containing the text data.
            feature_columns (list): The columns to use for outlier detection.
                If None, will use text-based features like length, word count, etc.
            threshold (float): The threshold for outlier detection.
                - For z-score: number of standard deviations (default: 3.0)
                - For IQR: multiplier for IQR range (default: 1.5)
            contamination (float): The expected proportion of outliers in the dataset.
                Only used for LOF and Isolation Forest.
            verbose (bool): Whether to display verbose output.
        """
        self.strategy = strategy
        self.text_column = text_column
        self.feature_columns = feature_columns
        self.threshold = threshold
        self.contamination = contamination
        self.verbose = verbose
        
        # Validate strategy
        valid_strategies = ['z-score', 'iqr', 'lof', 'isolation-forest']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")
            
        # Statistics for reporting
        self.stats = {}
        
    def _extract_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from text data for outlier detection.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with extracted features.
        """
        features_df = pd.DataFrame()
        
        # Make sure text column exists
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataframe")
            
        # Extract text-based features
        # 1. Text length
        features_df['text_length'] = df[self.text_column].apply(lambda x: len(str(x)))
        
        # 2. Word count
        features_df['word_count'] = df[self.text_column].apply(lambda x: len(str(x).split()))
        
        # 3. Average word length
        def avg_word_length(text):
            words = str(text).split()
            if not words:
                return 0
            return sum(len(word) for word in words) / len(words)
            
        features_df['avg_word_length'] = df[self.text_column].apply(avg_word_length)
        
        # 4. Sentence count (rough approximation)
        features_df['sentence_count'] = df[self.text_column].apply(
            lambda x: len([s for s in str(x).split('.') if s.strip()])
        )
        
        # 5. Special character ratio
        def special_char_ratio(text):
            if not text:
                return 0
            special_chars = sum(1 for c in str(text) if not c.isalnum() and not c.isspace())
            return special_chars / len(str(text))
            
        features_df['special_char_ratio'] = df[self.text_column].apply(special_char_ratio)
        
        return features_df
        
    def _detect_outliers_z_score(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using Z-scores.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            features_df (pd.DataFrame): Dataframe with features for outlier detection.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        logger.info(f"Detecting outliers using Z-scores (threshold: {self.threshold})...")
        
        # Initialize outlier flags
        outlier_mask = pd.Series(False, index=df.index)
        outlier_reasons = {}
        
        # Check each feature
        for col in features_df.columns:
            # Compute Z-scores
            z_scores = stats.zscore(features_df[col], nan_policy='omit')
            
            # Identify outliers
            col_outliers = np.abs(z_scores) > self.threshold
            
            # Track reasons for outliers
            for idx in df.index[col_outliers]:
                if idx not in outlier_reasons:
                    outlier_reasons[idx] = []
                outlier_reasons[idx].append(f"{col} (z-score: {z_scores[df.index.get_loc(idx)]:.2f})")
            
            # Update outlier mask
            outlier_mask = outlier_mask | col_outliers
        
        # Get non-outlier data
        non_outlier_df = df[~outlier_mask].copy()
        
        # Compute statistics
        total_outliers = outlier_mask.sum()
        outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'outliers_detected': int(total_outliers),
            'outlier_percentage': outlier_pct,
            'outlier_threshold': self.threshold,
            'outlier_reasons': outlier_reasons
        }
        
        self.stats = stats
        
        return non_outlier_df
        
    def _detect_outliers_iqr(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR).
        
        Args:
            df (pd.DataFrame): The input dataframe.
            features_df (pd.DataFrame): Dataframe with features for outlier detection.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        logger.info(f"Detecting outliers using IQR (threshold: {self.threshold})...")
        
        # Initialize outlier flags
        outlier_mask = pd.Series(False, index=df.index)
        outlier_reasons = {}
        
        # Check each feature
        for col in features_df.columns:
            # Compute IQR bounds
            Q1 = features_df[col].quantile(0.25)
            Q3 = features_df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR
            
            # Identify outliers
            col_outliers = (features_df[col] < lower_bound) | (features_df[col] > upper_bound)
            
            # Track reasons for outliers
            for idx in df.index[col_outliers]:
                if idx not in outlier_reasons:
                    outlier_reasons[idx] = []
                outlier_reasons[idx].append(
                    f"{col} (value: {features_df[col][idx]:.2f}, bounds: [{lower_bound:.2f}, {upper_bound:.2f}])"
                )
            
            # Update outlier mask
            outlier_mask = outlier_mask | col_outliers
        
        # Get non-outlier data
        non_outlier_df = df[~outlier_mask].copy()
        
        # Compute statistics
        total_outliers = outlier_mask.sum()
        outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'outliers_detected': int(total_outliers),
            'outlier_percentage': outlier_pct,
            'outlier_threshold': self.threshold,
            'outlier_reasons': outlier_reasons
        }
        
        self.stats = stats
        
        return non_outlier_df
        
    def _detect_outliers_lof(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using Local Outlier Factor (LOF).
        
        Args:
            df (pd.DataFrame): The input dataframe.
            features_df (pd.DataFrame): Dataframe with features for outlier detection.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        logger.info(f"Detecting outliers using Local Outlier Factor (contamination: {self.contamination})...")
        
        # Initialize LOF
        lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        try:
            # Fit and predict
            y_pred = lof.fit_predict(features_df)
            
            # Outliers are labeled as -1
            outlier_mask = y_pred == -1
            
            # Get outlier scores (negative of the decision function)
            outlier_scores = -lof.negative_outlier_factor_
            
            # Track outlier reasons
            outlier_reasons = {}
            for i, (idx, is_outlier) in enumerate(zip(df.index, outlier_mask)):
                if is_outlier:
                    outlier_reasons[idx] = f"LOF outlier score: {outlier_scores[i]:.2f}"
            
            # Get non-outlier data
            non_outlier_df = df[~outlier_mask].copy()
            
            # Compute statistics
            total_outliers = outlier_mask.sum()
            outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
            
            stats = {
                'total_samples': len(df),
                'outliers_detected': int(total_outliers),
                'outlier_percentage': outlier_pct,
                'contamination': self.contamination,
                'outlier_reasons': outlier_reasons
            }
            
            self.stats = stats
            
            return non_outlier_df
            
        except Exception as e:
            logger.error(f"Error in LOF outlier detection: {e}")
            logger.warning("Falling back to Z-score outlier detection.")
            return self._detect_outliers_z_score(df, features_df)
        
    def _detect_outliers_isolation_forest(self, df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect outliers using Isolation Forest.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            features_df (pd.DataFrame): Dataframe with features for outlier detection.
            
        Returns:
            pd.DataFrame: Dataframe with outliers removed.
        """
        logger.info(f"Detecting outliers using Isolation Forest (contamination: {self.contamination})...")
        
        # Use the fitted model if available
        if hasattr(self, 'model'):
            # Use the pre-trained model
            outlier_pred = self.model.predict(features_df)
        else:
            # Fallback: train a new model (should only happen for training data)
            model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            outlier_pred = model.fit_predict(features_df)
        
        # Convert predictions to boolean mask (1: inlier, -1: outlier)
        outlier_mask = outlier_pred == -1
        
        # Get non-outlier data
        non_outlier_df = df[~outlier_mask].copy()
        
        # Compute statistics
        total_outliers = sum(outlier_mask)
        outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'outliers_detected': int(total_outliers),
            'outlier_percentage': outlier_pct,
            'contamination': self.contamination
        }
        
        self.stats = stats
        
        return non_outlier_df
        logger.info(f"Detecting outliers using Isolation Forest (contamination: {self.contamination})...")
        
        # Initialize Isolation Forest
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        try:
            # Fit and predict
            y_pred = iso_forest.fit_predict(features_df)
            
            # Outliers are labeled as -1
            outlier_mask = y_pred == -1
            
            # Get outlier scores (negative of the decision function)
            outlier_scores = -iso_forest.score_samples(features_df)
            
            # Track outlier reasons
            outlier_reasons = {}
            for i, (idx, is_outlier) in enumerate(zip(df.index, outlier_mask)):
                if is_outlier:
                    outlier_reasons[idx] = f"Isolation Forest outlier score: {outlier_scores[i]:.2f}"
            
            # Get non-outlier data
            non_outlier_df = df[~outlier_mask].copy()
            
            # Compute statistics
            total_outliers = outlier_mask.sum()
            outlier_pct = (total_outliers / len(df)) * 100 if len(df) > 0 else 0
            
            stats = {
                'total_samples': len(df),
                'outliers_detected': int(total_outliers),
                'outlier_percentage': outlier_pct,
                'contamination': self.contamination,
                'outlier_reasons': outlier_reasons
            }
            
            self.stats = stats
            
            return non_outlier_df
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest outlier detection: {e}")
            logger.warning("Falling back to Z-score outlier detection.")
            return self._detect_outliers_z_score(df, features_df)
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the outlier detector on the training data.
        This computes statistics or model parameters that will be used for transformation.
        
        Args:
            df (pd.DataFrame): The training dataframe.
        """
        # Extract features for outlier detection
        features_df = self._extract_text_features(df)
        
        # Store feature statistics for transformation
        self.feature_stats = {}
        
        # For z-score and IQR, store mean and std or quartiles
        if self.strategy == 'z-score':
            for col in features_df.columns:
                self.feature_stats[col] = {
                    'mean': features_df[col].mean(),
                    'std': features_df[col].std()
                }
        elif self.strategy == 'iqr':
            for col in features_df.columns:
                q1 = features_df[col].quantile(0.25)
                q3 = features_df[col].quantile(0.75)
                iqr = q3 - q1
                self.feature_stats[col] = {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr
                }
        # For LOF and Isolation Forest, train the model
        elif self.strategy == 'lof':
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,  # Set to True to use predict method later
                n_jobs=-1
            )
            self.model.fit(features_df)
        elif self.strategy == 'isolation-forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(features_df)
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply outlier detection to the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The dataframe with outliers removed.
        """
        # Extract features for outlier detection
        features_df = self._extract_text_features(df)
        
        # Apply the appropriate outlier detection strategy
        if self.strategy == 'z-score':
            return self._detect_outliers_z_score(df, features_df)
        elif self.strategy == 'iqr':
            return self._detect_outliers_iqr(df, features_df)
        elif self.strategy == 'lof':
            return self._detect_outliers_lof(df, features_df)
        elif self.strategy == 'isolation-forest':
            return self._detect_outliers_isolation_forest(df, features_df)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
            
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The transformed dataframe.
        """
        self.fit(df)
        return self.transform(df)
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the input dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The dataframe with outliers removed.
        """
        # For this preprocessor, fit does nothing
        return self.transform(df)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get the outlier detection statistics.
        
        Returns:
            Dict[str, Any]: The outlier detection statistics.
        """
        return self.stats

# Example usage and demonstration
def demo():
    """
    Demonstrate the OutlierDetectionPreprocessor with a small example.
    """
    # Create a small example dataset with outliers
    data = {
        'text': [
            "This is a normal length text with typical characteristics.",
            "This is another normal example that shouldn't be flagged.",
            "Short text.",
            "This is a slightly longer text but still within normal range for demonstration purposes.",
            "This is an extremely long text that goes on and on with many words and characters and sentences and should definitely be detected as an outlier based on its length because it's way longer than all the other texts in this small example dataset that we created for demonstration purposes of the outlier detection module that we implemented in this file to show how the different strategies work on real data examples including z-score, IQR, LOF, and Isolation Forest which are all common methods for outlier detection in machine learning pipelines especially for text data preprocessing where outliers can significantly impact model performance if not handled properly.",
            "This text has a lot of special characters: !@#$%^&*()_+{}|:<>?~`-=[]\\;',./",
            "Normal text again, nothing to see here.",
            "One more normal example to balance the dataset."
        ],
        'label': [0, 0, 1, 1, 2, 2, 0, 1]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df[['text']])
    print(f"Original shape: {df.shape}")
    
    # Example 1: Z-score outlier detection
    preprocessor = OutlierDetectionPreprocessor(strategy='z-score', threshold=2.0)
    clean_df = preprocessor.fit_transform(df)
    
    print("\nAfter Z-score outlier detection (threshold=2.0):")
    print(clean_df[['text']])
    print(f"New shape: {clean_df.shape}")
    
    # Example 2: IQR outlier detection
    preprocessor = OutlierDetectionPreprocessor(strategy='iqr', threshold=1.5)
    clean_df = preprocessor.fit_transform(df)
    
    print("\nAfter IQR outlier detection (threshold=1.5):")
    print(clean_df[['text']])
    print(f"New shape: {clean_df.shape}")
    
    # Example 3: LOF outlier detection
    preprocessor = OutlierDetectionPreprocessor(strategy='lof', contamination=0.2)
    clean_df = preprocessor.fit_transform(df)
    
    print("\nAfter LOF outlier detection (contamination=0.2):")
    print(clean_df[['text']])
    print(f"New shape: {clean_df.shape}")
    
    # Example 4: Isolation Forest outlier detection
    preprocessor = OutlierDetectionPreprocessor(strategy='isolation-forest', contamination=0.2)
    clean_df = preprocessor.fit_transform(df)
    
    print("\nAfter Isolation Forest outlier detection (contamination=0.2):")
    print(clean_df[['text']])
    print(f"New shape: {clean_df.shape}")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    demo()
