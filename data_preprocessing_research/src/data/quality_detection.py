"""
Data quality detection utilities.

This module provides functionality to automatically detect various data quality issues
in text classification datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging

try:
    from ..utils.logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class DataQualityDetector:
    """
    Automated detection of data quality issues in text datasets.
    """
    
    def __init__(self):
        """Initialize the data quality detector."""
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        logger.info("DataQualityDetector initialized")
    
    def detect_all_issues(
        self,
        data: pd.DataFrame,
        text_columns: List[str],
        target_column: str
    ) -> Dict[str, Any]:
        """
        Detect all data quality issues in the dataset.
        
        Args:
            data: Dataset to analyze
            text_columns: List of text column names
            target_column: Target column name
            
        Returns:
            Dictionary of detected issues and their severity
        """
        logger.info(f"Detecting data quality issues in dataset with {len(data)} samples")
        
        issues = {
            'redundancy': self.detect_redundancy(data, text_columns),
            'imbalance': self.detect_imbalance(data, target_column),
            'noise': self.detect_noise(data, text_columns, target_column),
            'outliers': self.detect_outliers(data, text_columns)
        }
        
        # Calculate overall quality score
        issues['overall_quality_score'] = self._calculate_quality_score(issues)
        
        logger.info("Data quality analysis completed")
        return issues
    
    def detect_redundancy(self, data: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Detect redundancy and duplicate content."""
        redundancy_info = {}
        
        for col in text_columns:
            if col not in data.columns:
                continue
                
            # Exact duplicates
            exact_duplicates = data[col].duplicated().sum()
            exact_duplicate_ratio = exact_duplicates / len(data)
            
            # Near duplicates (using cosine similarity)
            try:
                # Sample for efficiency if dataset is large
                sample_size = min(1000, len(data))
                sample_data = data[col].sample(sample_size, random_state=42)
                
                # Vectorize text
                tfidf_matrix = self.vectorizer.fit_transform(sample_data.astype(str))
                
                # Calculate pairwise similarities
                similarities = cosine_similarity(tfidf_matrix)
                
                # Count near duplicates (similarity > 0.8, excluding self-similarity)
                near_duplicate_pairs = 0
                for i in range(len(similarities)):
                    for j in range(i+1, len(similarities)):
                        if similarities[i][j] > 0.8:
                            near_duplicate_pairs += 1
                
                near_duplicate_ratio = (near_duplicate_pairs * 2) / len(sample_data)
                
            except Exception as e:
                logger.warning(f"Could not calculate near duplicates for {col}: {e}")
                near_duplicate_ratio = 0
            
            redundancy_info[col] = {
                'exact_duplicates': exact_duplicates,
                'exact_duplicate_ratio': exact_duplicate_ratio,
                'near_duplicate_ratio': near_duplicate_ratio,
                'severity': self._assess_redundancy_severity(exact_duplicate_ratio, near_duplicate_ratio)
            }
        
        return redundancy_info
    
    def detect_imbalance(self, data: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Detect class imbalance."""
        if target_column not in data.columns:
            return {'error': f'Target column {target_column} not found'}
        
        # Calculate class distribution
        class_counts = data[target_column].value_counts()
        class_ratios = class_counts / len(data)
        
        # Calculate imbalance metrics
        majority_ratio = class_ratios.max()
        minority_ratio = class_ratios.min()
        imbalance_ratio = minority_ratio / majority_ratio
        
        # Gini coefficient for multi-class imbalance
        sorted_ratios = sorted(class_ratios.values, reverse=True)
        n = len(sorted_ratios)
        gini = sum((2 * i - n - 1) * ratio for i, ratio in enumerate(sorted_ratios, 1)) / (n * sum(sorted_ratios))
        
        return {
            'class_counts': class_counts.to_dict(),
            'class_ratios': class_ratios.to_dict(),
            'majority_ratio': majority_ratio,
            'minority_ratio': minority_ratio,
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini,
            'severity': self._assess_imbalance_severity(imbalance_ratio, gini)
        }
    
    def detect_noise(
        self, 
        data: pd.DataFrame, 
        text_columns: List[str], 
        target_column: str
    ) -> Dict[str, Any]:
        """Detect label noise and inconsistencies."""
        noise_info = {}
        
        # Check for missing values
        missing_values = data.isnull().sum().to_dict()
        
        # Check for inconsistent text lengths
        for col in text_columns:
            if col not in data.columns:
                continue
                
            text_lengths = data[col].astype(str).str.len()
            length_std = text_lengths.std()
            length_mean = text_lengths.mean()
            length_cv = length_std / length_mean if length_mean > 0 else 0
            
            noise_info[f'{col}_length_variability'] = {
                'mean_length': length_mean,
                'std_length': length_std,
                'coefficient_of_variation': length_cv,
                'severity': 'high' if length_cv > 1.0 else 'medium' if length_cv > 0.5 else 'low'
            }
        
        # Check for potential label noise (simplified heuristic)
        if target_column in data.columns:
            # Look for very short or very long texts that might be mislabeled
            for col in text_columns:
                if col not in data.columns:
                    continue
                    
                text_lengths = data[col].astype(str).str.len()
                q1, q3 = text_lengths.quantile([0.25, 0.75])
                iqr = q3 - q1
                
                # Potential outliers
                outlier_threshold_low = q1 - 1.5 * iqr
                outlier_threshold_high = q3 + 1.5 * iqr
                
                outliers = ((text_lengths < outlier_threshold_low) | 
                           (text_lengths > outlier_threshold_high)).sum()
                outlier_ratio = outliers / len(data)
                
                noise_info[f'{col}_potential_outliers'] = {
                    'outlier_count': outliers,
                    'outlier_ratio': outlier_ratio,
                    'severity': 'high' if outlier_ratio > 0.1 else 'medium' if outlier_ratio > 0.05 else 'low'
                }
        
        noise_info['missing_values'] = missing_values
        
        return noise_info
    
    def detect_outliers(self, data: pd.DataFrame, text_columns: List[str]) -> Dict[str, Any]:
        """Detect text outliers based on length and content."""
        outlier_info = {}
        
        for col in text_columns:
            if col not in data.columns:
                continue
                
            # Text length outliers
            text_lengths = data[col].astype(str).str.len()
            
            # Statistical outliers using IQR method
            q1, q3 = text_lengths.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            length_outliers = ((text_lengths < lower_bound) | (text_lengths > upper_bound)).sum()
            length_outlier_ratio = length_outliers / len(data)
            
            # Word count outliers
            word_counts = data[col].astype(str).str.split().str.len()
            word_q1, word_q3 = word_counts.quantile([0.25, 0.75])
            word_iqr = word_q3 - word_q1
            word_lower = word_q1 - 1.5 * word_iqr
            word_upper = word_q3 + 1.5 * word_iqr
            
            word_outliers = ((word_counts < word_lower) | (word_counts > word_upper)).sum()
            word_outlier_ratio = word_outliers / len(data)
            
            outlier_info[col] = {
                'length_outliers': length_outliers,
                'length_outlier_ratio': length_outlier_ratio,
                'word_outliers': word_outliers,
                'word_outlier_ratio': word_outlier_ratio,
                'length_stats': {
                    'mean': text_lengths.mean(),
                    'std': text_lengths.std(),
                    'min': text_lengths.min(),
                    'max': text_lengths.max(),
                    'q1': q1,
                    'q3': q3
                },
                'severity': self._assess_outlier_severity(length_outlier_ratio, word_outlier_ratio)
            }
        
        return outlier_info
    
    def _assess_redundancy_severity(self, exact_ratio: float, near_ratio: float) -> str:
        """Assess redundancy severity."""
        total_redundancy = exact_ratio + near_ratio
        if total_redundancy > 0.3:
            return 'high'
        elif total_redundancy > 0.1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_imbalance_severity(self, imbalance_ratio: float, gini: float) -> str:
        """Assess imbalance severity."""
        if imbalance_ratio < 0.1 or gini > 0.7:
            return 'high'
        elif imbalance_ratio < 0.3 or gini > 0.4:
            return 'medium'
        else:
            return 'low'
    
    def _assess_outlier_severity(self, length_ratio: float, word_ratio: float) -> str:
        """Assess outlier severity."""
        max_ratio = max(length_ratio, word_ratio)
        if max_ratio > 0.15:
            return 'high'
        elif max_ratio > 0.05:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_quality_score(self, issues: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-1, higher is better)."""
        penalties = 0
        
        # Redundancy penalty
        if 'redundancy' in issues:
            for col_info in issues['redundancy'].values():
                if isinstance(col_info, dict) and 'severity' in col_info:
                    if col_info['severity'] == 'high':
                        penalties += 0.3
                    elif col_info['severity'] == 'medium':
                        penalties += 0.15
        
        # Imbalance penalty
        if 'imbalance' in issues and 'severity' in issues['imbalance']:
            if issues['imbalance']['severity'] == 'high':
                penalties += 0.25
            elif issues['imbalance']['severity'] == 'medium':
                penalties += 0.1
        
        # Noise penalty
        if 'noise' in issues:
            for key, value in issues['noise'].items():
                if isinstance(value, dict) and 'severity' in value:
                    if value['severity'] == 'high':
                        penalties += 0.2
                    elif value['severity'] == 'medium':
                        penalties += 0.1
        
        # Outlier penalty
        if 'outliers' in issues:
            for col_info in issues['outliers'].values():
                if isinstance(col_info, dict) and 'severity' in col_info:
                    if col_info['severity'] == 'high':
                        penalties += 0.15
                    elif col_info['severity'] == 'medium':
                        penalties += 0.075
        
        # Calculate final score
        quality_score = max(0, 1 - penalties)
        return quality_score
