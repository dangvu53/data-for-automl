#!/usr/bin/env python3
"""
Modern implementation of Learn2Clean duplicate detection and outlier detection
using updated packages to avoid conflicts.
"""

import pandas as pd
import numpy as np
import time
import warnings
from typing import Dict, Optional, Tuple, List, Union
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist, squareform
import jellyfish
from textdistance import damerau_levenshtein, levenshtein, jaro_winkler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ModernDuplicateDetector:
    """
    Modern implementation of duplicate detection using updated packages
    
    Parameters
    ----------
    strategy : str, default='exact'
        Detection strategy: 'exact', 'fuzzy', 'semantic', 'hybrid'
    threshold : float, default=0.8
        Similarity threshold for approximate matching
    text_columns : list, optional
        Specific columns to consider for text-based deduplication
    exclude_columns : list, optional
        Columns to exclude from duplicate detection
    verbose : bool, default=False
        Whether to print detailed information
    """
    
    def __init__(self, strategy='exact', threshold=0.8, text_columns=None, 
                 exclude_columns=None, verbose=False):
        self.strategy = strategy
        self.threshold = threshold
        self.text_columns = text_columns or []
        self.exclude_columns = exclude_columns or []
        self.verbose = verbose
        
    def detect_and_remove_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main method to detect and remove duplicates"""
        if data.empty:
            logger.warning("Empty dataframe provided")
            return data
            
        original_size = len(data)
        logger.info(f"Starting duplicate detection with strategy: {self.strategy}")
        logger.info(f"Original dataset size: {original_size}")
        
        # Add unique identifier for tracking
        data_with_id = data.copy()
        data_with_id['_temp_id'] = range(len(data_with_id))
        
        if self.strategy == 'exact':
            result = self._exact_duplicate_removal(data_with_id)
        elif self.strategy == 'fuzzy':
            result = self._fuzzy_duplicate_removal(data_with_id)
        elif self.strategy == 'semantic':
            result = self._semantic_duplicate_removal(data_with_id)
        elif self.strategy == 'hybrid':
            result = self._hybrid_duplicate_removal(data_with_id)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Remove temporary ID column
        result = result.drop(columns=['_temp_id'], errors='ignore')
        
        final_size = len(result)
        removed_count = original_size - final_size
        
        logger.info(f"Duplicate detection complete: {original_size} -> {final_size} (-{removed_count} duplicates)")
        
        return result
    
    def _exact_duplicate_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove exact duplicates"""
        # Exclude specified columns from duplicate check
        check_columns = [col for col in data.columns 
                        if col not in self.exclude_columns and col != '_temp_id']
        
        if not check_columns:
            logger.warning("No columns available for duplicate detection")
            return data
        
        # Keep first occurrence of duplicates
        duplicated_mask = data.duplicated(subset=check_columns, keep='first')
        
        if self.verbose:
            duplicated_ids = data[duplicated_mask]['_temp_id'].tolist()
            logger.info(f"Found {len(duplicated_ids)} exact duplicates")
            if len(duplicated_ids) > 0:
                logger.info(f"Duplicate IDs: {duplicated_ids[:10]}{'...' if len(duplicated_ids) > 10 else ''}")
        
        return data[~duplicated_mask].reset_index(drop=True)
    
    def _fuzzy_duplicate_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove fuzzy duplicates using string similarity"""
        if not self.text_columns:
            # Auto-detect text columns
            self.text_columns = [col for col in data.columns 
                               if data[col].dtype == 'object' and col != '_temp_id']
        
        if not self.text_columns:
            logger.warning("No text columns found for fuzzy matching, falling back to exact matching")
            return self._exact_duplicate_removal(data)
        
        # Create combined text representation
        combined_text = data[self.text_columns].apply(
            lambda row: ' '.join(row.astype(str)), axis=1
        )
        
        # Find fuzzy duplicates using string similarity
        duplicates_to_remove = set()
        
        for i in range(len(combined_text)):
            if i in duplicates_to_remove:
                continue
                
            text_i = combined_text.iloc[i]
            
            for j in range(i + 1, len(combined_text)):
                if j in duplicates_to_remove:
                    continue
                    
                text_j = combined_text.iloc[j]
                
                # Calculate similarity using multiple metrics
                similarities = [
                    1 - (damerau_levenshtein.normalized_distance(text_i, text_j)),
                    1 - (levenshtein.normalized_distance(text_i, text_j)),
                    jaro_winkler.normalized_similarity(text_i, text_j)
                ]
                
                # Use maximum similarity
                max_similarity = max(similarities)
                
                if max_similarity >= self.threshold:
                    duplicates_to_remove.add(j)  # Remove the later occurrence
                    
                    if self.verbose:
                        logger.info(f"Fuzzy duplicate found: {i} <-> {j} (similarity: {max_similarity:.3f})")
        
        # Remove duplicates
        result = data.drop(data.index[list(duplicates_to_remove)]).reset_index(drop=True)
        
        if self.verbose:
            logger.info(f"Removed {len(duplicates_to_remove)} fuzzy duplicates")
        
        return result
    
    def _semantic_duplicate_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove semantic duplicates using TF-IDF and cosine similarity"""
        if not self.text_columns:
            # Auto-detect text columns
            self.text_columns = [col for col in data.columns 
                               if data[col].dtype == 'object' and col != '_temp_id']
        
        if not self.text_columns:
            logger.warning("No text columns found for semantic matching, falling back to exact matching")
            return self._exact_duplicate_removal(data)
        
        # Create combined text representation
        combined_text = data[self.text_columns].apply(
            lambda row: ' '.join(row.astype(str)), axis=1
        )
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(combined_text)
        except ValueError as e:
            logger.warning(f"TF-IDF vectorization failed: {e}, falling back to exact matching")
            return self._exact_duplicate_removal(data)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Find semantic duplicates
        duplicates_to_remove = set()
        
        for i in range(len(similarity_matrix)):
            if i in duplicates_to_remove:
                continue
                
            for j in range(i + 1, len(similarity_matrix)):
                if j in duplicates_to_remove:
                    continue
                    
                if similarity_matrix[i, j] >= self.threshold:
                    duplicates_to_remove.add(j)  # Remove the later occurrence
                    
                    if self.verbose:
                        logger.info(f"Semantic duplicate found: {i} <-> {j} (similarity: {similarity_matrix[i, j]:.3f})")
        
        # Remove duplicates
        result = data.drop(data.index[list(duplicates_to_remove)]).reset_index(drop=True)
        
        if self.verbose:
            logger.info(f"Removed {len(duplicates_to_remove)} semantic duplicates")
        
        return result
    
    def _hybrid_duplicate_removal(self, data: pd.DataFrame) -> pd.DataFrame:
        """Combine multiple duplicate detection strategies"""
        logger.info("Applying hybrid duplicate detection (exact + fuzzy + semantic)")
        
        # Step 1: Remove exact duplicates
        result = self._exact_duplicate_removal(data)
        size_after_exact = len(result)
        
        # Step 2: Remove fuzzy duplicates
        result = self._fuzzy_duplicate_removal(result)
        size_after_fuzzy = len(result)
        
        # Step 3: Remove semantic duplicates
        result = self._semantic_duplicate_removal(result)
        size_after_semantic = len(result)
        
        if self.verbose:
            logger.info(f"Hybrid results - Exact: -{len(data) - size_after_exact}, "
                       f"Fuzzy: -{size_after_exact - size_after_fuzzy}, "
                       f"Semantic: -{size_after_fuzzy - size_after_semantic}")
        
        return result


class ModernOutlierDetector:
    """
    Modern implementation of outlier detection using updated packages
    
    Parameters
    ----------
    strategy : str, default='lof'
        Detection strategy: 'lof', 'isolation_forest', 'zscore', 'iqr', 'hybrid'
    contamination : float, default=0.1
        Expected proportion of outliers in the data
    text_columns : list, optional
        Text columns to include in feature extraction
    numeric_columns : list, optional
        Numeric columns to include directly
    exclude_columns : list, optional
        Columns to exclude from outlier detection
    verbose : bool, default=False
        Whether to print detailed information
    """
    
    def __init__(self, strategy='lof', contamination=0.1, text_columns=None,
                 numeric_columns=None, exclude_columns=None, verbose=False):
        self.strategy = strategy
        self.contamination = contamination
        self.text_columns = text_columns or []
        self.numeric_columns = numeric_columns or []
        self.exclude_columns = exclude_columns or []
        self.verbose = verbose
        
    def detect_and_remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """Main method to detect and remove outliers"""
        if data.empty:
            logger.warning("Empty dataframe provided")
            return data
            
        original_size = len(data)
        logger.info(f"Starting outlier detection with strategy: {self.strategy}")
        logger.info(f"Original dataset size: {original_size}")
        
        # Extract features for outlier detection
        features = self._extract_features(data)
        
        if features is None or features.shape[1] == 0:
            logger.warning("No features available for outlier detection")
            return data
        
        # Detect outliers based on strategy
        if self.strategy == 'lof':
            outlier_mask = self._lof_detection(features)
        elif self.strategy == 'isolation_forest':
            outlier_mask = self._isolation_forest_detection(features)
        elif self.strategy == 'zscore':
            outlier_mask = self._zscore_detection(features)
        elif self.strategy == 'iqr':
            outlier_mask = self._iqr_detection(features)
        elif self.strategy == 'hybrid':
            outlier_mask = self._hybrid_detection(features)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Remove outliers
        result = data[~outlier_mask].reset_index(drop=True)
        
        final_size = len(result)
        removed_count = original_size - final_size
        
        logger.info(f"Outlier detection complete: {original_size} -> {final_size} (-{removed_count} outliers)")
        
        if self.verbose and removed_count > 0:
            outlier_indices = data.index[outlier_mask].tolist()
            logger.info(f"Outlier indices: {outlier_indices[:10]}{'...' if len(outlier_indices) > 10 else ''}")
        
        return result
    
    def _extract_features(self, data: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract numerical features from the dataframe"""
        features_list = []
        
        # Auto-detect columns if not specified
        if not self.text_columns and not self.numeric_columns:
            self.text_columns = [col for col in data.columns 
                               if data[col].dtype == 'object' and col not in self.exclude_columns]
            self.numeric_columns = [col for col in data.columns 
                                  if pd.api.types.is_numeric_dtype(data[col]) and col not in self.exclude_columns]
        
        # Add numeric features directly
        if self.numeric_columns:
            numeric_features = data[self.numeric_columns].fillna(0).values
            features_list.append(numeric_features)
            logger.info(f"Added {len(self.numeric_columns)} numeric features")
        
        # Add TF-IDF features from text columns
        if self.text_columns:
            try:
                # Combine text columns
                combined_text = data[self.text_columns].apply(
                    lambda row: ' '.join(row.astype(str)), axis=1
                )
                
                # Create TF-IDF features
                vectorizer = TfidfVectorizer(
                    max_features=100,  # Limit features for outlier detection
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.8
                )
                
                tfidf_features = vectorizer.fit_transform(combined_text).toarray()
                features_list.append(tfidf_features)
                logger.info(f"Added {tfidf_features.shape[1]} TF-IDF features from text columns")
                
            except ValueError as e:
                logger.warning(f"Failed to extract TF-IDF features: {e}")
        
        # Add basic text statistics
        if self.text_columns:
            text_stats = []
            for col in self.text_columns:
                if col in data.columns:
                    # Text length
                    text_stats.append(data[col].astype(str).str.len().values)
                    # Word count
                    text_stats.append(data[col].astype(str).str.split().str.len().fillna(0).values)
                    # Character diversity
                    char_diversity = data[col].astype(str).apply(
                        lambda x: len(set(x.lower())) / max(len(x), 1)
                    ).values
                    text_stats.append(char_diversity)
            
            if text_stats:
                text_features = np.column_stack(text_stats)
                features_list.append(text_features)
                logger.info(f"Added {text_features.shape[1]} text statistics features")
        
        if not features_list:
            logger.warning("No features could be extracted")
            return None
        
        # Combine all features
        features = np.hstack(features_list)
        
        # Handle infinite and NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        logger.info(f"Final feature matrix shape: {features_scaled.shape}")
        
        return features_scaled
    
    def _lof_detection(self, features: np.ndarray) -> np.ndarray:
        """Detect outliers using Local Outlier Factor"""
        n_neighbors = min(20, len(features) - 1)
        if n_neighbors < 1:
            return np.zeros(len(features), dtype=bool)
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=self.contamination,
            novelty=False
        )
        
        outlier_labels = lof.fit_predict(features)
        return outlier_labels == -1
    
    def _isolation_forest_detection(self, features: np.ndarray) -> np.ndarray:
        """Detect outliers using Isolation Forest"""
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        
        outlier_labels = iso_forest.fit_predict(features)
        return outlier_labels == -1
    
    def _zscore_detection(self, features: np.ndarray) -> np.ndarray:
        """Detect outliers using Z-score method"""
        from scipy import stats
        
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(features, axis=0, nan_policy='omit'))
        
        # Consider a point an outlier if it has high Z-score in any dimension
        threshold = stats.norm.ppf(1 - self.contamination / 2)  # Two-tailed test
        
        # A point is an outlier if it exceeds threshold in any feature
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        return outlier_mask
    
    def _iqr_detection(self, features: np.ndarray) -> np.ndarray:
        """Detect outliers using Interquartile Range method"""
        outlier_mask = np.zeros(len(features), dtype=bool)
        
        for i in range(features.shape[1]):
            feature_col = features[:, i]
            Q1 = np.percentile(feature_col, 25)
            Q3 = np.percentile(feature_col, 75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Mark outliers in this feature
            feature_outliers = (feature_col < lower_bound) | (feature_col > upper_bound)
            outlier_mask |= feature_outliers
        
        return outlier_mask
    
    def _hybrid_detection(self, features: np.ndarray) -> np.ndarray:
        """Combine multiple outlier detection methods"""
        logger.info("Applying hybrid outlier detection (LOF + Isolation Forest + Z-score)")
        
        # Get predictions from multiple methods
        lof_outliers = self._lof_detection(features)
        iso_outliers = self._isolation_forest_detection(features)
        zscore_outliers = self._zscore_detection(features)
        
        # Combine using majority voting
        votes = lof_outliers.astype(int) + iso_outliers.astype(int) + zscore_outliers.astype(int)
        
        # Consider a point an outlier if at least 2 methods agree
        hybrid_outliers = votes >= 2
        
        if self.verbose:
            logger.info(f"Hybrid results - LOF: {lof_outliers.sum()}, "
                       f"IsolationForest: {iso_outliers.sum()}, "
                       f"Z-score: {zscore_outliers.sum()}, "
                       f"Combined: {hybrid_outliers.sum()}")
        
        return hybrid_outliers


class Learn2CleanProcessor:
    """
    Unified processor that combines duplicate detection and outlier detection
    """
    
    def __init__(self, duplicate_config=None, outlier_config=None):
        """
        Initialize the processor with configuration for both detectors
        
        Parameters
        ----------
        duplicate_config : dict, optional
            Configuration for duplicate detector
        outlier_config : dict, optional
            Configuration for outlier detector
        """
        self.duplicate_config = duplicate_config or {}
        self.outlier_config = outlier_config or {}
        
        self.duplicate_detector = ModernDuplicateDetector(**self.duplicate_config)
        self.outlier_detector = ModernOutlierDetector(**self.outlier_config)
    
    def process(self, data: pd.DataFrame, apply_deduplication=True, apply_outlier_detection=True) -> pd.DataFrame:
        """
        Apply both duplicate detection and outlier detection
        
        Parameters
        ----------
        data : pd.DataFrame
            Input data
        apply_deduplication : bool, default=True
            Whether to apply duplicate detection
        apply_outlier_detection : bool, default=True
            Whether to apply outlier detection
            
        Returns
        -------
        pd.DataFrame
            Cleaned data
        """
        result = data.copy()
        original_size = len(result)
        
        logger.info(f"Starting Learn2Clean processing on {original_size} samples")
        
        # Step 1: Duplicate detection
        if apply_deduplication:
            result = self.duplicate_detector.detect_and_remove_duplicates(result)
            size_after_dedup = len(result)
            logger.info(f"After deduplication: {size_after_dedup} samples")
        
        # Step 2: Outlier detection
        if apply_outlier_detection:
            result = self.outlier_detector.detect_and_remove_outliers(result)
            size_after_outliers = len(result)
            logger.info(f"After outlier detection: {size_after_outliers} samples")
        
        final_size = len(result)
        total_removed = original_size - final_size
        
        logger.info(f"Learn2Clean processing complete: {original_size} -> {final_size} (-{total_removed} samples, {final_size/original_size*100:.1f}% retained)")
        
        return result


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    
    # Generate sample text data with duplicates and outliers
    sample_texts = [
        "This is a sample text for testing",
        "This is a sample text for testing",  # Exact duplicate
        "This is a sample text for testing purposes",  # Near duplicate
        "Machine learning is a subset of artificial intelligence",
        "ML is a subset of AI",  # Semantic duplicate
        "The quick brown fox jumps over the lazy dog",
        "A quick brown fox leaps over the lazy dog",  # Fuzzy duplicate
        "XYZXYZXYZXYZXYZ random noise text",  # Outlier
        "Normal text about data science and analytics",
        "Data science involves statistics and programming",
        "!!!@@@### garbage text ###@@@!!!",  # Outlier
    ]
    
    sample_data = pd.DataFrame({
        'text': sample_texts,
        'category': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D', 'E', 'E', 'F'],
        'score': np.random.normal(50, 10, len(sample_texts))
    })
    
    # Add some outlier scores
    sample_data.loc[7, 'score'] = 150  # Outlier
    sample_data.loc[10, 'score'] = -50  # Outlier
    
    print("Original data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    # Test duplicate detection
    print("Testing Duplicate Detection:")
    for strategy in ['exact', 'fuzzy', 'semantic', 'hybrid']:
        print(f"\n--- {strategy.upper()} Strategy ---")
        detector = ModernDuplicateDetector(strategy=strategy, threshold=0.8, verbose=True)
        result = detector.detect_and_remove_duplicates(sample_data)
        print(f"Result size: {len(result)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test outlier detection
    print("Testing Outlier Detection:")
    for strategy in ['lof', 'isolation_forest', 'zscore', 'iqr', 'hybrid']:
        print(f"\n--- {strategy.upper()} Strategy ---")
        detector = ModernOutlierDetector(strategy=strategy, contamination=0.2, verbose=True)
        result = detector.detect_and_remove_outliers(sample_data)
        print(f"Result size: {len(result)}")
    
    print("\n" + "="*50 + "\n")
    
    # Test combined processing
    print("Testing Combined Processing:")
    processor = Learn2CleanProcessor(
        duplicate_config={'strategy': 'hybrid', 'threshold': 0.8, 'verbose': True},
        outlier_config={'strategy': 'hybrid', 'contamination': 0.2, 'verbose': True}
    )
    
    final_result = processor.process(sample_data)
    print(f"\nFinal result:")
    print(final_result)
    print(f"Final size: {len(final_result)}")
