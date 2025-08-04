#!/usr/bin/env python3
"""
Enhanced Duplicate Detection and Removal Module

This module provides functionality to detect and remove duplicates in text datasets
using various duplicate detection strategies:
1. Exact Match: Remove exact duplicate texts
2. Approximate Duplicate: Using fuzzywuzzy, textdistance, and dedupe libraries
   - Levenshtein distance
   - Jaccard similarity
   - Cosine similarity
   - Fingerprint-based matching

The module can be used as a preprocessing step in a machine learning pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from tqdm import tqdm
from collections import defaultdict
import re
import hashlib
import itertools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Try to import optional libraries, but make them optional
try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    FUZZYWUZZY_AVAILABLE = False

try:
    import textdistance
    TEXTDISTANCE_AVAILABLE = True
except ImportError:
    TEXTDISTANCE_AVAILABLE = False

try:
    import dedupe
    DEDUPE_AVAILABLE = True
except ImportError:
    DEDUPE_AVAILABLE = False
    
# Setup logging
logger = logging.getLogger(__name__)

class DuplicateRemovalPreprocessor:
    """
    Preprocessor to remove duplicates from text datasets.
    
    Attributes:
        strategy (str): The duplicate detection strategy to use.
        threshold (float): The similarity threshold for duplicate detection.
        text_column (str): The column containing the text data.
        label_column (str): The column containing the label data.
        method (str): The specific method to use for approximate matching.
        n_gram_range (tuple): The range of n-grams to use for TF-IDF vectorization.
        min_token_length (int): Minimum token length to consider.
        verbose (bool): Whether to display verbose output.
    """
    
    def __init__(
        self, 
        strategy: str = 'exact',
        threshold: float = 0.9,
        text_column: str = 'text',
        label_column: Optional[str] = 'label',
        method: str = 'cosine',  # Changed default to cosine
        similarity_metric: Optional[str] = None,  # Added to support configs using this parameter name
        n_gram_range: Tuple[int, int] = (1, 3),
        min_token_length: int = 2,
        verbose: bool = True
    ):
        """
        Initialize the DuplicateRemovalPreprocessor.
        
        Args:
            strategy (str): The duplicate detection strategy to use.
                Options: 'exact', 'approximate', 'cross-label'
            threshold (float): The similarity threshold for duplicate detection.
                Values close to 1.0 will only detect very similar texts.
            text_column (str): The column containing the text data.
            label_column (str): The column containing the label data (optional for some strategies).
            method (str): The specific method to use for approximate matching.
                Options: 'levenshtein', 'jaccard', 'cosine', 'fingerprint', 'token_sort'
            similarity_metric (str): Alternative parameter name for method (for backward compatibility).
            n_gram_range (tuple): The range of n-grams to use for TF-IDF vectorization.
            min_token_length (int): Minimum token length to consider.
            verbose (bool): Whether to display verbose output.
        """
        self.strategy = strategy
        self.threshold = threshold
        self.text_column = text_column
        self.label_column = label_column
        
        # Use similarity_metric if provided, otherwise use method
        if similarity_metric is not None:
            self.method = similarity_metric
        else:
            self.method = method
            
        self.n_gram_range = n_gram_range
        self.min_token_length = min_token_length
        self.verbose = verbose
        
        # Validate strategy
        valid_strategies = ['exact', 'approximate', 'cross-label']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")
            
        # Validate method for approximate matching
        valid_methods = ['levenshtein', 'jaccard', 'cosine', 'fingerprint', 'token_sort']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method: {self.method}. Must be one of {valid_methods}")
            
        # Log the similarity method being used
        if self.method == 'cosine' and self.strategy == 'approximate':
            logger.info(f"Using {self.method} method (vectorized approach) for duplicate detection with threshold {self.threshold}")
        else:
            logger.info(f"Using {self.method} method for duplicate detection with threshold {self.threshold}")
            
        # Check for library availability
        if self.method == 'levenshtein' and not FUZZYWUZZY_AVAILABLE:
            logger.warning("fuzzywuzzy library not available. Will use built-in Levenshtein implementation.")
            
        if self.method in ['jaccard', 'fingerprint'] and not TEXTDISTANCE_AVAILABLE:
            logger.warning("textdistance library not available. Will use built-in implementation.")
            
        # Statistics for reporting
        self.stats = {}
        
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text by lowercasing, removing extra whitespace, and punctuation.
        
        Args:
            text (str): The text to preprocess.
            
        Returns:
            str: The preprocessed text.
        """
        if not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Optional: Remove punctuation if needed
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
        
    def _compute_text_hash(self, text: str) -> str:
        """
        Compute a hash of the preprocessed text for exact matching.
        
        Args:
            text (str): The text to hash.
            
        Returns:
            str: The hash of the text.
        """
        preprocessed = self._preprocess_text(text)
        return hashlib.md5(preprocessed.encode()).hexdigest()
        
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text (str): The text to tokenize.
            
        Returns:
            List[str]: The tokenized words.
        """
        preprocessed = self._preprocess_text(text)
        return [w for w in preprocessed.split() if len(w) >= self.min_token_length]
        
    def _get_exact_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Identify exact duplicates in the dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The deduplicated dataframe and stats.
        """
        logger.info("Finding exact duplicates...")
        
        # Create a hash column for exact matching
        df['text_hash'] = df[self.text_column].apply(self._compute_text_hash)
        
        # Identify duplicates
        duplicate_mask = df.duplicated(subset=['text_hash'], keep='first')
        
        # Get deduplicated dataframe
        deduplicated_df = df[~duplicate_mask].drop(columns=['text_hash'])
        
        # Compute statistics
        total_duplicates = duplicate_mask.sum()
        duplicate_pct = (total_duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'unique_samples': len(deduplicated_df),
            'duplicates_removed': total_duplicates,
            'duplicate_percentage': duplicate_pct
        }
        
        return deduplicated_df, stats
        
    def _levenshtein_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Levenshtein similarity between two strings.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if FUZZYWUZZY_AVAILABLE:
            # Use fuzzywuzzy for Levenshtein similarity
            return fuzz.ratio(s1, s2) / 100.0
        else:
            # Fallback implementation if fuzzywuzzy is not available
            # This is a simplified version, not as efficient as fuzzywuzzy
            if not s1 and not s2:
                return 1.0
            if not s1 or not s2:
                return 0.0
                
            len_s1, len_s2 = len(s1), len(s2)
            
            # Initialize matrix
            dp = [[0] * (len_s2 + 1) for _ in range(len_s1 + 1)]
            
            # Fill matrix
            for i in range(len_s1 + 1):
                dp[i][0] = i
            for j in range(len_s2 + 1):
                dp[0][j] = j
                
            for i in range(1, len_s1 + 1):
                for j in range(1, len_s2 + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1]
                    else:
                        dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                        
            # Compute similarity
            max_len = max(len_s1, len_s2)
            distance = dp[len_s1][len_s2]
            return 1.0 - (distance / max_len)
            
    def _jaccard_similarity(self, s1: str, s2: str) -> float:
        """
        Compute Jaccard similarity between two strings.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if TEXTDISTANCE_AVAILABLE:
            # Use textdistance for Jaccard similarity
            return textdistance.jaccard.normalized_similarity(s1, s2)
        else:
            # Fallback implementation if textdistance is not available
            tokens1 = set(self._tokenize_text(s1))
            tokens2 = set(self._tokenize_text(s2))
            
            if not tokens1 and not tokens2:
                return 1.0
            if not tokens1 or not tokens2:
                return 0.0
                
            # Compute Jaccard similarity
            intersection = len(tokens1.intersection(tokens2))
            union = len(tokens1.union(tokens2))
            return intersection / union
            
    def _cosine_similarity_text(self, s1: str, s2: str) -> float:
        """
        Compute cosine similarity between two strings using TF-IDF.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        # Preprocess the strings
        s1 = self._preprocess_text(s1)
        s2 = self._preprocess_text(s2)
        
        # Vectorize
        vectorizer = TfidfVectorizer(ngram_range=self.n_gram_range)
        try:
            tfidf_matrix = vectorizer.fit_transform([s1, s2])
            
            # Compute cosine similarity
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return similarity
        except:
            # Fallback if vectorization fails
            return 0.0
            
    def _get_approximate_duplicates_vectorized(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Identify approximate duplicates using vectorized cosine similarity.
        This is much faster than pairwise comparison.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The deduplicated dataframe and stats.
        """
        logger.info(f"Using vectorized approach for cosine similarity (threshold: {self.threshold})...")
        
        # Sample if the dataset is very large to reduce computation time
        MAX_DOCS = 50000  # Maximum number of documents to process at once
        if len(df) > MAX_DOCS:
            logger.info(f"Dataset is large ({len(df)} documents). Processing in batches of {MAX_DOCS}...")
            # Process in batches
            batch_size = MAX_DOCS
            all_indices_to_keep = []
            
            for start_idx in range(0, len(df), batch_size):
                end_idx = min(start_idx + batch_size, len(df))
                logger.info(f"Processing batch {start_idx//batch_size + 1}: documents {start_idx} to {end_idx}...")
                
                batch_df = df.iloc[start_idx:end_idx]
                batch_indices_to_keep = self._process_similarity_batch(batch_df, start_idx)
                all_indices_to_keep.extend(batch_indices_to_keep)
                
            # Create deduplicated dataframe
            deduplicated_df = df.iloc[all_indices_to_keep].reset_index(drop=True)
        else:
            # Process the entire dataset at once
            indices_to_keep = self._process_similarity_batch(df, 0)
            deduplicated_df = df.iloc[indices_to_keep].reset_index(drop=True)
        
        # Compute statistics
        total_duplicates = len(df) - len(deduplicated_df)
        duplicate_pct = (total_duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'unique_samples': len(deduplicated_df),
            'duplicates_removed': total_duplicates,
            'duplicate_percentage': duplicate_pct,
            'similarity_threshold': self.threshold,
            'similarity_method': 'cosine (vectorized)',
            'vectorized_approach': True
        }
        
        return deduplicated_df, stats
        
    def _process_similarity_batch(self, df: pd.DataFrame, offset: int = 0) -> List[int]:
        """
        Process a batch of documents to find duplicates using vectorized cosine similarity.
        
        Args:
            df (pd.DataFrame): The batch dataframe.
            offset (int): The offset index for the batch within the full dataset.
            
        Returns:
            List[int]: Indices of documents to keep (not duplicates).
        """
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            ngram_range=self.n_gram_range,
            min_df=1,  # Include all terms
            max_df=1.0,  # Include all terms
            stop_words='english'
        )
        
        try:
            # If dataset is very large, use sparse matrices to conserve memory
            if len(df) > 10000:
                logger.info(f"Large batch detected ({len(df)} documents). Using memory-efficient approach...")
                # Fit and transform the data
                tfidf_matrix = vectorizer.fit_transform(df[self.text_column].astype(str))
                
                # Process row by row for large matrices to avoid memory issues
                indices_to_remove = set()
                for i in tqdm(range(tfidf_matrix.shape[0]), desc="Processing documents"):
                    if i in indices_to_remove:
                        continue
                        
                    # Get similarities for this document with all others
                    row_vector = tfidf_matrix[i:i+1, :]
                    similarities = cosine_similarity(row_vector, tfidf_matrix[i+1:, :])
                    
                    # Find similar documents
                    similar_indices = np.where(similarities[0] >= self.threshold)[0]
                    
                    # Adjust indices (since we're comparing with i+1 onwards)
                    similar_indices = similar_indices + (i + 1)
                    
                    # Mark these as duplicates
                    indices_to_remove.update(similar_indices)
            else:
                # Standard approach for smaller datasets
                # Fit and transform the data
                tfidf_matrix = vectorizer.fit_transform(df[self.text_column].astype(str))
                
                # Calculate pairwise cosine similarity
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Set diagonal to 0 to avoid self-similarity
                np.fill_diagonal(similarity_matrix, 0)
                
                # Find pairs of documents with similarity above threshold
                # We only need to look at the upper triangular part of the matrix
                similar_pairs = np.where(np.triu(similarity_matrix) >= self.threshold)
                
                # Documents that should be removed (duplicates)
                indices_to_remove = set()
                
                # For each pair of similar documents, keep the first and remove the second
                for i, j in zip(*similar_pairs):
                    if j not in indices_to_remove:  # If the second document is not already marked for removal
                        indices_to_remove.add(j)  # Remove the second document
            
            # Calculate indices to keep
            indices_to_keep = [i + offset for i in range(len(df)) if i not in indices_to_remove]
            
            logger.info(f"Found {len(indices_to_remove)} duplicates in batch of {len(df)} documents.")
            return indices_to_keep
            
        except Exception as e:
            logger.error(f"Error in vectorized similarity calculation: {e}")
            # Fallback: keep all documents
            return list(range(offset, offset + len(df)))
            
    def _fingerprint_similarity(self, s1: str, s2: str) -> float:
        """
        Compute fingerprint similarity between two strings.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if TEXTDISTANCE_AVAILABLE:
            # Use textdistance for fingerprint similarity
            return textdistance.fingerprint.normalized_similarity(s1, s2)
        else:
            # Fallback implementation - simplified fingerprint
            def get_fingerprint(text):
                # Preprocess
                text = self._preprocess_text(text)
                # Split into tokens
                tokens = text.split()
                # Sort and deduplicate
                tokens = sorted(set(tokens))
                # Join back
                return ' '.join(tokens)
                
            fp1 = get_fingerprint(s1)
            fp2 = get_fingerprint(s2)
            
            # Use Jaccard similarity on fingerprints
            return self._jaccard_similarity(fp1, fp2)
            
    def _token_sort_similarity(self, s1: str, s2: str) -> float:
        """
        Compute token sort ratio similarity between two strings.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if FUZZYWUZZY_AVAILABLE:
            # Use fuzzywuzzy for token sort ratio
            return fuzz.token_sort_ratio(s1, s2) / 100.0
        else:
            # Fallback implementation
            def sorted_text(text):
                text = self._preprocess_text(text)
                tokens = text.split()
                return ' '.join(sorted(tokens))
                
            sorted_s1 = sorted_text(s1)
            sorted_s2 = sorted_text(s2)
            
            return self._levenshtein_similarity(sorted_s1, sorted_s2)
            
    def _compute_similarity(self, s1: str, s2: str) -> float:
        """
        Compute similarity between two strings using the specified method.
        
        Args:
            s1 (str): First string.
            s2 (str): Second string.
            
        Returns:
            float: Similarity score between 0 and 1.
        """
        if self.method == 'levenshtein':
            return self._levenshtein_similarity(s1, s2)
        elif self.method == 'jaccard':
            return self._jaccard_similarity(s1, s2)
        elif self.method == 'cosine':
            return self._cosine_similarity_text(s1, s2)
        elif self.method == 'fingerprint':
            return self._fingerprint_similarity(s1, s2)
        elif self.method == 'token_sort':
            return self._token_sort_similarity(s1, s2)
        else:
            # Default to Levenshtein
            return self._levenshtein_similarity(s1, s2)
            
    def _get_approximate_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Identify approximate duplicates using the specified method.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The deduplicated dataframe and stats.
        """
        logger.info(f"Finding approximate duplicates using {self.method} method (threshold: {self.threshold})...")
        
        # Use a vectorized approach for cosine similarity
        if self.method == 'cosine':
            return self._get_approximate_duplicates_vectorized(df)
        
        # For other methods, use the original pairwise comparison approach
        # Preprocess texts
        texts = df[self.text_column].tolist()
        preprocessed_texts = [self._preprocess_text(text) for text in texts]
        
        # Find duplicates - compare each text with all others
        indices_to_keep = []
        indices_to_remove = set()
        similarity_groups = defaultdict(list)
        
        # Progress bar for verbose mode
        iterator = enumerate(preprocessed_texts)
        if self.verbose:
            iterator = tqdm(iterator, total=len(preprocessed_texts), desc=f"Finding {self.method} duplicates")
            
        # Check each document against all others
        for i, text1 in iterator:
            if i in indices_to_remove:
                continue
                
            # Keep this text
            indices_to_keep.append(i)
            group_id = len(similarity_groups)
            similarity_groups[group_id].append(i)
            
            # Compare with remaining texts
            for j in range(i+1, len(preprocessed_texts)):
                if j in indices_to_remove:
                    continue
                    
                text2 = preprocessed_texts[j]
                similarity = self._compute_similarity(text1, text2)
                
                # If similar enough, mark as duplicate
                if similarity >= self.threshold:
                    indices_to_remove.add(j)
                    similarity_groups[group_id].append(j)
        
        # Create deduplicated dataframe
        deduplicated_df = df.iloc[indices_to_keep].reset_index(drop=True)
        
        # Compute statistics
        total_duplicates = len(df) - len(deduplicated_df)
        duplicate_pct = (total_duplicates / len(df)) * 100 if len(df) > 0 else 0
        
        # Prepare similarity group statistics
        groups_with_duplicates = [group for group in similarity_groups.values() if len(group) > 1]
        
        stats = {
            'total_samples': len(df),
            'unique_samples': len(deduplicated_df),
            'duplicates_removed': total_duplicates,
            'duplicate_percentage': duplicate_pct,
            'similarity_threshold': self.threshold,
            'similarity_method': self.method,
            'duplicate_groups': len(groups_with_duplicates),
            'average_group_size': sum(len(g) for g in groups_with_duplicates) / len(groups_with_duplicates) if groups_with_duplicates else 0
        }
        
        return deduplicated_df, stats
        
    def _get_cross_label_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Identify cross-label duplicates (same text, different labels).
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: The deduplicated dataframe and stats.
        """
        logger.info("Finding cross-label duplicates...")
        
        if self.label_column not in df.columns:
            logger.warning(f"Label column {self.label_column} not found. Falling back to exact deduplication.")
            return self._get_exact_duplicates(df)
            
        # Create a hash column for exact matching
        df['text_hash'] = df[self.text_column].apply(self._compute_text_hash)
        
        # Group by text hash and count unique labels
        hash_counts = df.groupby('text_hash')[self.label_column].nunique()
        
        # Find hashes with multiple labels
        conflict_hashes = hash_counts[hash_counts > 1].index.tolist()
        
        # Mark cross-label duplicates
        cross_label_mask = df['text_hash'].isin(conflict_hashes)
        cross_label_df = df[cross_label_mask].copy()
        
        # For each conflicting hash, keep the most frequent label
        resolution_indices = []
        stats_by_class = defaultdict(int)
        
        for hash_val in conflict_hashes:
            hash_group = df[df['text_hash'] == hash_val]
            label_counts = hash_group[self.label_column].value_counts()
            most_common_label = label_counts.index[0]
            
            # Keep one instance with the most common label
            idx_to_keep = hash_group[hash_group[self.label_column] == most_common_label].index[0]
            resolution_indices.append(idx_to_keep)
            
            # Track stats
            stats_by_class[most_common_label] += 1
            
        # Remove all cross-label duplicates
        df = df[~cross_label_mask]
        
        # Add back the resolved instances
        resolved_df = df.loc[resolution_indices]
        df = pd.concat([df, resolved_df]).reset_index(drop=True)
        
        # Now do normal exact deduplication on the remaining data
        duplicate_mask = df.duplicated(subset=['text_hash'], keep='first')
        deduplicated_df = df[~duplicate_mask].drop(columns=['text_hash'])
        
        # Compute statistics
        total_conflicts = len(conflict_hashes)
        samples_with_conflicts = cross_label_mask.sum()
        conflict_pct = (samples_with_conflicts / len(df)) * 100 if len(df) > 0 else 0
        
        stats = {
            'total_samples': len(df),
            'unique_samples': len(deduplicated_df),
            'cross_label_conflicts': total_conflicts,
            'samples_with_conflicts': samples_with_conflicts,
            'conflict_percentage': conflict_pct,
            'resolution_by_class': dict(stats_by_class)
        }
        
        return deduplicated_df, stats
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the duplicate detector on the training data.
        For approximate duplicate detection, this may involve training embedding models or storing text hashes.
        
        Args:
            df (pd.DataFrame): The training dataframe.
        """
        # Create a hash column for exact matching if needed
        if self.strategy == 'exact':
            # Store text hashes from training data
            self.training_hashes = set(df[self.text_column].apply(self._compute_text_hash))
        
        # For TF-IDF based similarity, initialize the vectorizer
        elif self.strategy == 'approximate' and self.method == 'cosine':
            logger.info("Initializing TF-IDF vectorizer for cosine similarity...")
            self.vectorizer = TfidfVectorizer(
                ngram_range=self.n_gram_range,
                min_df=1,
                max_df=1.0,
                stop_words='english'
            )
            
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply duplicate removal to the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The deduplicated dataframe.
        """
        # Apply the appropriate duplicate removal strategy
        if self.strategy == 'exact':
            deduplicated_df, stats = self._get_exact_duplicates(df)
        elif self.strategy == 'approximate':
            deduplicated_df, stats = self._get_approximate_duplicates(df)
        elif self.strategy == 'cross-label':
            deduplicated_df, stats = self._get_cross_label_duplicates(df)
        else:
            raise ValueError(f"Invalid strategy: {self.strategy}")
            
        self.stats = stats
        return deduplicated_df
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the duplicate detector and transform the dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The deduplicated dataframe.
        """
        self.fit(df)
        return self.transform(df)
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the input dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The deduplicated dataframe.
        """
        # For this preprocessor, fit does nothing
        return self.transform(df)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get the deduplication statistics.
        
        Returns:
            Dict[str, Any]: The deduplication statistics.
        """
        return self.stats

# Example usage and demonstration
def demo():
    """
    Demonstrate the DuplicateRemovalPreprocessor with a small example.
    """
    # Create a small example dataset with duplicates
    data = {
        'text': [
            "This is a sample text",
            "This is a sample text",  # Exact duplicate
            "This is a sample text!",  # Near duplicate
            "This text is completely different",
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",  # Exact duplicate
            "The quick fox jumps over the lazy brown dog",  # Word order changed
            "Completely unique text example here",
            "This text is actually quite different from the others"
        ],
        'label': [0, 0, 0, 1, 1, 1, 1, 2, 2]
    }
    
    df = pd.DataFrame(data)
    print("Original dataset:")
    print(df)
    print(f"Original shape: {df.shape}")
    
    # Example 1: Exact duplicate removal
    preprocessor = DuplicateRemovalPreprocessor(strategy='exact')
    deduplicated_df = preprocessor.fit_transform(df)
    
    print("\nAfter exact duplicate removal:")
    print(deduplicated_df)
    print(f"New shape: {deduplicated_df.shape}")
    
    # Example 2: Approximate duplicate removal with Levenshtein
    preprocessor = DuplicateRemovalPreprocessor(
        strategy='approximate', 
        method='levenshtein',
        threshold=0.9
    )
    deduplicated_df = preprocessor.fit_transform(df)
    
    print("\nAfter approximate duplicate removal (Levenshtein, threshold=0.9):")
    print(deduplicated_df)
    print(f"New shape: {deduplicated_df.shape}")
    
    # Example 3: Approximate duplicate removal with Jaccard
    preprocessor = DuplicateRemovalPreprocessor(
        strategy='approximate', 
        method='jaccard',
        threshold=0.8
    )
    deduplicated_df = preprocessor.fit_transform(df)
    
    print("\nAfter approximate duplicate removal (Jaccard, threshold=0.8):")
    print(deduplicated_df)
    print(f"New shape: {deduplicated_df.shape}")
    
    # Example 4: Cross-label duplicate handling
    # Create dataset with cross-label duplicates
    data_cross = {
        'text': [
            "This is a sample text",
            "This is a sample text",  # Same text, different label
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumps over the lazy dog",  # Same text, different label
            "Completely unique text example here"
        ],
        'label': [0, 1, 1, 2, 2]  # Different labels for same text
    }
    
    df_cross = pd.DataFrame(data_cross)
    print("\nDataset with cross-label duplicates:")
    print(df_cross)
    
    preprocessor = DuplicateRemovalPreprocessor(strategy='cross-label')
    deduplicated_df = preprocessor.fit_transform(df_cross)
    
    print("\nAfter cross-label duplicate handling:")
    print(deduplicated_df)
    print(f"New shape: {deduplicated_df.shape}")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    demo()
