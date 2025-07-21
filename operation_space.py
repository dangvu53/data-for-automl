#!/usr/bin/env python3
"""
Operation Space Library for Meta-Learning Framework

This module contains various data preprocessing, selection, and augmentation operations
that can be applied to different data types. Each operation is modular and follows
a standard interface.
"""

import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any, Optional
from meta_learning_framework import Operation, OperationResult, DataSelector, DataAugmenter, DataPreprocessor

logger = logging.getLogger(__name__)

# =============================================================================
# TEXT PREPROCESSING OPERATIONS
# =============================================================================

class TextCleaner(DataPreprocessor):
    """Clean text data by removing noise, normalizing, etc."""
    
    def __init__(self, name: str = "TextCleaner", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
        
    def apply(self, data: pd.DataFrame, embeddings: np.ndarray = None) -> OperationResult:
        start_time = time.time()
        
        try:
            result_data = data.copy()
            
            # Find text column
            text_col = self._find_text_column(data)
            if text_col is None:
                return OperationResult(
                    data=data, metadata={}, operation_name=self.name,
                    execution_time=time.time() - start_time, success=False,
                    error_message="No text column found"
                )
            
            # Apply text cleaning based on parameters
            if self.parameters.get('lowercase', True):
                result_data[text_col] = result_data[text_col].str.lower()
            
            if self.parameters.get('remove_punctuation', False):
                import string
                result_data[text_col] = result_data[text_col].str.translate(
                    str.maketrans('', '', string.punctuation)
                )
            
            if self.parameters.get('remove_extra_whitespace', True):
                result_data[text_col] = result_data[text_col].str.replace(r'\s+', ' ', regex=True)
                result_data[text_col] = result_data[text_col].str.strip()
            
            if self.parameters.get('remove_numbers', False):
                result_data[text_col] = result_data[text_col].str.replace(r'\d+', '', regex=True)
            
            # Remove empty texts
            result_data = result_data[result_data[text_col].str.len() > 0]
            
            metadata = {
                'original_size': len(data),
                'final_size': len(result_data),
                'text_column': text_col,
                'parameters_used': self.parameters
            }
            
            return OperationResult(
                data=result_data, metadata=metadata, operation_name=self.name,
                execution_time=time.time() - start_time, success=True
            )
            
        except Exception as e:
            return OperationResult(
                data=data, metadata={}, operation_name=self.name,
                execution_time=time.time() - start_time, success=False,
                error_message=str(e)
            )
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'lowercase': [True, False],
            'remove_punctuation': [True, False],
            'remove_extra_whitespace': [True, False],
            'remove_numbers': [True, False]
        }
    
    def _find_text_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the main text column in the dataframe"""
        text_candidates = ['text', 'premise', 'hypothesis', 'question', 'content']
        for col in text_candidates:
            if col in data.columns:
                return col
        
        # Look for columns with 'text' in the name
        for col in data.columns:
            if 'text' in col.lower():
                return col
        
        return None

class TextLengthFilter(DataSelector):
    """Filter texts based on length criteria"""
    
    def __init__(self, name: str = "TextLengthFilter", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
    
    def apply(self, data: pd.DataFrame, embeddings: np.ndarray = None) -> OperationResult:
        start_time = time.time()
        
        try:
            result_data = data.copy()
            
            text_col = self._find_text_column(data)
            if text_col is None:
                return OperationResult(
                    data=data, metadata={}, operation_name=self.name,
                    execution_time=time.time() - start_time, success=False,
                    error_message="No text column found"
                )
            
            # Calculate text lengths
            text_lengths = result_data[text_col].str.len()
            
            # Apply length filters
            min_length = self.parameters.get('min_length', 10)
            max_length = self.parameters.get('max_length', 1000)
            
            # Filter based on length
            length_mask = (text_lengths >= min_length) & (text_lengths <= max_length)
            result_data = result_data[length_mask]
            
            metadata = {
                'original_size': len(data),
                'final_size': len(result_data),
                'min_length': min_length,
                'max_length': max_length,
                'avg_length_before': text_lengths.mean(),
                'avg_length_after': result_data[text_col].str.len().mean() if len(result_data) > 0 else 0
            }
            
            return OperationResult(
                data=result_data, metadata=metadata, operation_name=self.name,
                execution_time=time.time() - start_time, success=True
            )
            
        except Exception as e:
            return OperationResult(
                data=data, metadata={}, operation_name=self.name,
                execution_time=time.time() - start_time, success=False,
                error_message=str(e)
            )
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'min_length': [5, 10, 20, 50],
            'max_length': [500, 1000, 2000, 5000]
        }
    
    def _find_text_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the main text column in the dataframe"""
        text_candidates = ['text', 'premise', 'hypothesis', 'question', 'content']
        for col in text_candidates:
            if col in data.columns:
                return col
        
        for col in data.columns:
            if 'text' in col.lower():
                return col
        
        return None

class DifficultyBasedSelector(DataSelector):
    """Select data based on difficulty/hardness scores"""
    
    def __init__(self, name: str = "DifficultyBasedSelector", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
    
    def apply(self, data: pd.DataFrame, embeddings: np.ndarray = None) -> OperationResult:
        start_time = time.time()
        
        try:
            result_data = data.copy()
            
            # Calculate difficulty scores based on text features
            difficulty_scores = self._calculate_difficulty_scores(result_data)
            
            # Select based on difficulty strategy
            strategy = self.parameters.get('strategy', 'balanced')
            selection_ratio = self.parameters.get('selection_ratio', 0.8)
            
            n_select = int(len(result_data) * selection_ratio)
            
            if strategy == 'easy':
                # Select easiest samples
                selected_indices = difficulty_scores.nsmallest(n_select).index
            elif strategy == 'hard':
                # Select hardest samples
                selected_indices = difficulty_scores.nlargest(n_select).index
            elif strategy == 'balanced':
                # Select mix of easy and hard
                n_easy = n_select // 2
                n_hard = n_select - n_easy
                easy_indices = difficulty_scores.nsmallest(n_easy).index
                hard_indices = difficulty_scores.nlargest(n_hard).index
                selected_indices = easy_indices.union(hard_indices)
            else:
                # Random selection
                selected_indices = result_data.sample(n=n_select).index
            
            result_data = result_data.loc[selected_indices]
            
            metadata = {
                'original_size': len(data),
                'final_size': len(result_data),
                'strategy': strategy,
                'selection_ratio': selection_ratio,
                'avg_difficulty_before': difficulty_scores.mean(),
                'avg_difficulty_after': difficulty_scores.loc[selected_indices].mean()
            }
            
            return OperationResult(
                data=result_data, metadata=metadata, operation_name=self.name,
                execution_time=time.time() - start_time, success=True
            )
            
        except Exception as e:
            return OperationResult(
                data=data, metadata={}, operation_name=self.name,
                execution_time=time.time() - start_time, success=False,
                error_message=str(e)
            )
    
    def _calculate_difficulty_scores(self, data: pd.DataFrame) -> pd.Series:
        """Calculate difficulty scores based on text features"""
        text_col = self._find_text_column(data)
        if text_col is None:
            return pd.Series(np.random.random(len(data)), index=data.index)
        
        # Simple difficulty metrics
        text_lengths = data[text_col].str.len()
        word_counts = data[text_col].str.split().str.len()
        
        # Normalize and combine metrics
        length_scores = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min() + 1e-8)
        word_scores = (word_counts - word_counts.min()) / (word_counts.max() - word_counts.min() + 1e-8)
        
        # Combine scores (longer texts are considered harder)
        difficulty_scores = 0.6 * length_scores + 0.4 * word_scores
        
        return difficulty_scores
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'strategy': ['easy', 'hard', 'balanced', 'random'],
            'selection_ratio': [0.5, 0.7, 0.8, 0.9]
        }
    
    def _find_text_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the main text column in the dataframe"""
        text_candidates = ['text', 'premise', 'hypothesis', 'question', 'content']
        for col in text_candidates:
            if col in data.columns:
                return col
        
        for col in data.columns:
            if 'text' in col.lower():
                return col
        
        return None

class TextAugmenter(DataAugmenter):
    """Augment text data using various techniques"""
    
    def __init__(self, name: str = "TextAugmenter", parameters: Dict[str, Any] = None):
        super().__init__(name, parameters)
    
    def apply(self, data: pd.DataFrame, embeddings: np.ndarray = None) -> OperationResult:
        start_time = time.time()
        
        try:
            result_data = data.copy()
            
            text_col = self._find_text_column(data)
            if text_col is None:
                return OperationResult(
                    data=data, metadata={}, operation_name=self.name,
                    execution_time=time.time() - start_time, success=False,
                    error_message="No text column found"
                )
            
            augmentation_ratio = self.parameters.get('augmentation_ratio', 0.1)
            technique = self.parameters.get('technique', 'synonym_replacement')
            
            n_augment = int(len(result_data) * augmentation_ratio)
            
            if n_augment > 0:
                # Sample data to augment
                sample_data = result_data.sample(n=min(n_augment, len(result_data)))
                
                augmented_rows = []
                for _, row in sample_data.iterrows():
                    augmented_row = row.copy()
                    
                    if technique == 'synonym_replacement':
                        augmented_row[text_col] = self._synonym_replacement(row[text_col])
                    elif technique == 'random_insertion':
                        augmented_row[text_col] = self._random_insertion(row[text_col])
                    elif technique == 'random_deletion':
                        augmented_row[text_col] = self._random_deletion(row[text_col])
                    elif technique == 'paraphrase':
                        augmented_row[text_col] = self._simple_paraphrase(row[text_col])
                    
                    augmented_rows.append(augmented_row)
                
                # Add augmented data
                if augmented_rows:
                    augmented_df = pd.DataFrame(augmented_rows)
                    result_data = pd.concat([result_data, augmented_df], ignore_index=True)
            
            metadata = {
                'original_size': len(data),
                'final_size': len(result_data),
                'augmented_samples': len(result_data) - len(data),
                'technique': technique,
                'augmentation_ratio': augmentation_ratio
            }
            
            return OperationResult(
                data=result_data, metadata=metadata, operation_name=self.name,
                execution_time=time.time() - start_time, success=True
            )
            
        except Exception as e:
            return OperationResult(
                data=data, metadata={}, operation_name=self.name,
                execution_time=time.time() - start_time, success=False,
                error_message=str(e)
            )
    
    def _synonym_replacement(self, text: str) -> str:
        """Simple synonym replacement (placeholder implementation)"""
        # This is a simplified version - in practice, you'd use libraries like NLTK or spaCy
        words = text.split()
        if len(words) > 1:
            # Replace random word with a simple variation
            idx = np.random.randint(0, len(words))
            if words[idx].lower() in ['good', 'great', 'excellent']:
                words[idx] = np.random.choice(['good', 'great', 'excellent', 'amazing'])
            elif words[idx].lower() in ['bad', 'terrible', 'awful']:
                words[idx] = np.random.choice(['bad', 'terrible', 'awful', 'horrible'])
        return ' '.join(words)
    
    def _random_insertion(self, text: str) -> str:
        """Insert random words"""
        words = text.split()
        if len(words) > 0:
            insert_words = ['very', 'really', 'quite', 'somewhat', 'rather']
            insert_word = np.random.choice(insert_words)
            insert_pos = np.random.randint(0, len(words))
            words.insert(insert_pos, insert_word)
        return ' '.join(words)
    
    def _random_deletion(self, text: str) -> str:
        """Delete random words"""
        words = text.split()
        if len(words) > 2:  # Keep at least 2 words
            delete_pos = np.random.randint(0, len(words))
            words.pop(delete_pos)
        return ' '.join(words)
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple paraphrasing by word reordering"""
        words = text.split()
        if len(words) > 3:
            # Swap two adjacent words
            swap_pos = np.random.randint(0, len(words) - 1)
            words[swap_pos], words[swap_pos + 1] = words[swap_pos + 1], words[swap_pos]
        return ' '.join(words)
    
    def get_parameter_space(self) -> Dict[str, List[Any]]:
        return {
            'technique': ['synonym_replacement', 'random_insertion', 'random_deletion', 'paraphrase'],
            'augmentation_ratio': [0.05, 0.1, 0.2, 0.3]
        }
    
    def _find_text_column(self, data: pd.DataFrame) -> Optional[str]:
        """Find the main text column in the dataframe"""
        text_candidates = ['text', 'premise', 'hypothesis', 'question', 'content']
        for col in text_candidates:
            if col in data.columns:
                return col
        
        for col in data.columns:
            if 'text' in col.lower():
                return col
        
        return None
