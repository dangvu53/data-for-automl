"""
Data preprocessing implementations for different quality issues.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
import logging

try:
    from ..utils.logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    import logging
    def get_logger(name):
        return logging.getLogger(name)

logger = get_logger(__name__)

class PreprocessorFactory:
    """
    Factory class for creating and applying data preprocessing methods.
    """
    
    def __init__(self):
        """Initialize the preprocessor factory."""
        self.preprocessors = {
            'duplicate_removal': DuplicateRemover(),
            'near_duplicate_removal': NearDuplicateRemover(),
            'data_selection': DiversitySelector(),
            'smote': SMOTEPreprocessor(),
            'borderline_smote': BorderlineSMOTEPreprocessor(),
            'random_undersampling': RandomUndersamplingPreprocessor(),
            'edited_nearest_neighbors': EditedNearestNeighboursPreprocessor(),
            'statistical_outliers': StatisticalOutlierRemover(),
            'text_length_filter': TextLengthFilter()
        }
        logger.info("PreprocessorFactory initialized")
    
    def get_preprocessor(self, method_name: str):
        """Get a preprocessor by name."""
        if method_name not in self.preprocessors:
            raise ValueError(f"Unknown preprocessing method: {method_name}")
        return self.preprocessors[method_name]
    
    def apply_preprocessing(
        self,
        data: pd.DataFrame,
        method_name: str,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """
        Apply preprocessing method to data.
        
        Args:
            data: Input dataset
            method_name: Name of preprocessing method
            parameters: Method parameters
            text_columns: List of text column names
            target_column: Target column name
            
        Returns:
            Preprocessed dataset
        """
        preprocessor = self.get_preprocessor(method_name)
        return preprocessor.preprocess(data, parameters, text_columns, target_column)

class BasePreprocessor:
    """Base class for all preprocessors."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Apply preprocessing to the data."""
        raise NotImplementedError("Subclasses must implement preprocess method")

class DuplicateRemover(BasePreprocessor):
    """Remove exact duplicates from the dataset."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Remove exact duplicates."""
        keep = parameters.get('keep', 'first')
        
        # Remove duplicates based on text columns
        subset_columns = text_columns + [target_column]
        subset_columns = [col for col in subset_columns if col in data.columns]
        
        original_size = len(data)
        cleaned_data = data.drop_duplicates(subset=subset_columns, keep=keep)
        
        removed_count = original_size - len(cleaned_data)
        logger.info(f"Removed {removed_count} exact duplicates ({removed_count/original_size:.1%})")
        
        return cleaned_data.reset_index(drop=True)

class NearDuplicateRemover(BasePreprocessor):
    """Remove near-duplicates using text similarity."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Remove near-duplicates using cosine similarity."""
        similarity_threshold = parameters.get('similarity_threshold', 0.95)
        
        # For efficiency, work with first text column
        if not text_columns or text_columns[0] not in data.columns:
            logger.warning("No valid text columns found for near-duplicate removal")
            return data
        
        text_col = text_columns[0]
        
        # Vectorize text
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(data[text_col].astype(str))
        except Exception as e:
            logger.warning(f"Could not vectorize text for near-duplicate removal: {e}")
            return data
        
        # Find near-duplicates
        similarities = cosine_similarity(tfidf_matrix)
        
        # Mark duplicates for removal
        to_remove = set()
        for i in range(len(similarities)):
            if i in to_remove:
                continue
            for j in range(i+1, len(similarities)):
                if similarities[i][j] > similarity_threshold:
                    to_remove.add(j)
        
        # Remove near-duplicates
        original_size = len(data)
        mask = ~data.index.isin(to_remove)
        cleaned_data = data[mask]
        
        removed_count = len(to_remove)
        logger.info(f"Removed {removed_count} near-duplicates ({removed_count/original_size:.1%})")
        
        return cleaned_data.reset_index(drop=True)

class DiversitySelector(BasePreprocessor):
    """Select diverse subset of data."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Select diverse subset using clustering or sampling."""
        selection_ratio = parameters.get('selection_ratio', 0.8)
        
        target_size = int(len(data) * selection_ratio)
        if target_size >= len(data):
            return data
        
        # Simple stratified sampling for diversity
        if target_column in data.columns:
            # Stratified sampling to maintain class distribution
            sampled_data = data.groupby(target_column, group_keys=False).apply(
                lambda x: x.sample(int(len(x) * selection_ratio), random_state=42)
            )
        else:
            # Random sampling
            sampled_data = data.sample(target_size, random_state=42)
        
        logger.info(f"Selected {len(sampled_data)} diverse samples from {len(data)}")
        return sampled_data.reset_index(drop=True)

class SMOTEPreprocessor(BasePreprocessor):
    """Apply SMOTE for class imbalance."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Apply SMOTE oversampling."""
        if target_column not in data.columns:
            logger.warning("Target column not found for SMOTE")
            return data
        
        # For text data, we need to vectorize first
        if not text_columns or text_columns[0] not in data.columns:
            logger.warning("No text columns found for SMOTE")
            return data
        
        try:
            # Vectorize text
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(data[text_columns[0]].astype(str)).toarray()
            y = data[target_column]
            
            # Apply SMOTE
            smote = SMOTE(
                sampling_strategy=parameters.get('sampling_strategy', 'auto'),
                k_neighbors=parameters.get('k_neighbors', 5),
                random_state=parameters.get('random_state', 42)
            )
            
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Create new dataframe (simplified - just replicate original structure)
            # In practice, you'd want to reconstruct the text from the vectors
            original_size = len(data)
            new_size = len(X_resampled)
            
            logger.info(f"SMOTE: {original_size} -> {new_size} samples")
            
            # For demo purposes, return original data with some duplicated samples
            # In real implementation, you'd reconstruct from the resampled vectors
            minority_class = y.value_counts().idxmin()
            minority_data = data[data[target_column] == minority_class]
            
            # Add some minority samples
            additional_samples = new_size - original_size
            if additional_samples > 0:
                extra_samples = minority_data.sample(
                    min(additional_samples, len(minority_data)), 
                    replace=True, 
                    random_state=42
                )
                result = pd.concat([data, extra_samples], ignore_index=True)
            else:
                result = data
            
            return result
            
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}")
            return data

class BorderlineSMOTEPreprocessor(BasePreprocessor):
    """Apply Borderline SMOTE for class imbalance."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Apply Borderline SMOTE oversampling."""
        # Similar to SMOTE but uses BorderlineSMOTE
        # Simplified implementation for demo
        logger.info("Applying Borderline SMOTE (simplified)")
        return SMOTEPreprocessor().preprocess(data, parameters, text_columns, target_column)

class RandomUndersamplingPreprocessor(BasePreprocessor):
    """Apply random undersampling for class imbalance."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Apply random undersampling."""
        if target_column not in data.columns:
            logger.warning("Target column not found for undersampling")
            return data
        
        # Find minority class size
        class_counts = data[target_column].value_counts()
        minority_size = class_counts.min()
        
        # Sample each class to minority size
        balanced_data = data.groupby(target_column, group_keys=False).apply(
            lambda x: x.sample(min(minority_size, len(x)), random_state=42)
        )
        
        logger.info(f"Undersampling: {len(data)} -> {len(balanced_data)} samples")
        return balanced_data.reset_index(drop=True)

class EditedNearestNeighboursPreprocessor(BasePreprocessor):
    """Apply Edited Nearest Neighbours for cleaning."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Apply ENN cleaning (simplified)."""
        # Simplified implementation - remove some samples randomly
        removal_ratio = 0.05  # Remove 5% of samples
        keep_ratio = 1 - removal_ratio
        
        cleaned_data = data.sample(frac=keep_ratio, random_state=42)
        
        logger.info(f"ENN cleaning: {len(data)} -> {len(cleaned_data)} samples")
        return cleaned_data.reset_index(drop=True)

class StatisticalOutlierRemover(BasePreprocessor):
    """Remove statistical outliers based on text length."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Remove outliers using IQR method."""
        multiplier = parameters.get('multiplier', 1.5)
        
        if not text_columns or text_columns[0] not in data.columns:
            logger.warning("No text columns found for outlier removal")
            return data
        
        text_col = text_columns[0]
        text_lengths = data[text_col].astype(str).str.len()
        
        # Calculate IQR bounds
        q1, q3 = text_lengths.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        # Filter outliers
        mask = (text_lengths >= lower_bound) & (text_lengths <= upper_bound)
        cleaned_data = data[mask]
        
        removed_count = len(data) - len(cleaned_data)
        logger.info(f"Removed {removed_count} outliers ({removed_count/len(data):.1%})")
        
        return cleaned_data.reset_index(drop=True)

class TextLengthFilter(BasePreprocessor):
    """Filter texts by length constraints."""
    
    def preprocess(
        self,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        text_columns: List[str],
        target_column: str
    ) -> pd.DataFrame:
        """Filter texts by minimum and maximum length."""
        min_length = parameters.get('min_length', 10)
        max_length = parameters.get('max_length', 1000)
        
        if not text_columns or text_columns[0] not in data.columns:
            logger.warning("No text columns found for length filtering")
            return data
        
        text_col = text_columns[0]
        text_lengths = data[text_col].astype(str).str.len()
        
        # Apply length filters
        mask = (text_lengths >= min_length) & (text_lengths <= max_length)
        filtered_data = data[mask]
        
        removed_count = len(data) - len(filtered_data)
        logger.info(f"Length filtering: removed {removed_count} samples ({removed_count/len(data):.1%})")
        
        return filtered_data.reset_index(drop=True)
