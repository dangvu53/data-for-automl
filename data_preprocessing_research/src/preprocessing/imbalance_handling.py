#!/usr/bin/env python3
"""
Imbalance Handling Module

This module provides functionality to handle class imbalance in datasets
using various techniques:
1. Random Oversampling: Randomly duplicate minority class samples
2. SMOTE: Synthetic Minority Over-sampling Technique
3. ADASYN: Adaptive Synthetic Sampling
4. Easy Data Augmentation (EDA): Text-level augmentation techniques

The module can be used as a preprocessing step in a machine learning pipeline.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional, Any
import logging
from tqdm import tqdm
from collections import defaultdict, Counter
import random
import re
import string
from sklearn.neighbors import NearestNeighbors

# Try to import optional libraries, but make them optional
try:
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    
# Setup logging
logger = logging.getLogger(__name__)

class ImbalanceHandlingPreprocessor:
    """
    Preprocessor to handle class imbalance in datasets.
    
    Attributes:
        strategy (str): The imbalance handling strategy to use.
        text_column (str): The column containing the text data.
        label_column (str): The column containing the label data.
        sampling_strategy (Union[str, dict, float]): The sampling strategy for resampling.
        random_state (int): Random state for reproducibility.
        k_neighbors (int): Number of nearest neighbors for SMOTE and ADASYN.
        eda_alpha (float): Parameter for EDA controlling the probability of each operation.
        eda_num_aug (int): Number of augmented samples to generate per original sample.
        verbose (bool): Whether to display verbose output.
    """
    
    def __init__(
        self, 
        strategy: str = 'random-oversampling',
        text_column: str = 'text',
        label_column: str = 'label',
        sampling_strategy: Union[str, dict, float] = 'auto',
        random_state: int = 42,
        k_neighbors: int = 5,
        eda_alpha: float = 0.1,
        eda_num_aug: int = 4,
        verbose: bool = True
    ):
        """
        Initialize the ImbalanceHandlingPreprocessor.
        
        Args:
            strategy (str): The imbalance handling strategy to use.
                Options: 'random-oversampling', 'smote', 'adasyn', 'eda'
            text_column (str): The column containing the text data.
            label_column (str): The column containing the label data.
            sampling_strategy (Union[str, dict, float]): The sampling strategy for resampling.
                - 'auto': Resample all classes to match the majority class.
                - 'minority': Resample only the minority class.
                - 'not minority': Resample all classes except the minority class.
                - 'not majority': Resample all classes except the majority class.
                - float: Ratio of the number of samples in the minority class to the number of samples in the majority class.
                - dict: Keys are the class labels, values are the desired number of samples for each class.
            random_state (int): Random state for reproducibility.
            k_neighbors (int): Number of nearest neighbors for SMOTE and ADASYN.
            eda_alpha (float): Parameter for EDA controlling the probability of each operation.
            eda_num_aug (int): Number of augmented samples to generate per original sample.
            verbose (bool): Whether to display verbose output.
        """
        self.strategy = strategy
        self.text_column = text_column
        self.label_column = label_column
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.eda_alpha = eda_alpha
        self.eda_num_aug = eda_num_aug
        self.verbose = verbose
        
        # Validate strategy
        valid_strategies = ['random-oversampling', 'smote', 'adasyn', 'eda']
        if self.strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {self.strategy}. Must be one of {valid_strategies}")
            
        # Check for imblearn availability
        if self.strategy in ['smote', 'adasyn'] and not IMBLEARN_AVAILABLE:
            raise ImportError(f"The '{self.strategy}' strategy requires the imbalanced-learn package. "
                              "Install it using: pip install imbalanced-learn")
            
        # Initialize resampler
        self.resampler = None
        
        # Statistics for reporting
        self.stats = {}
        
    def _extract_text_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract numerical features from text data.
        For SMOTE and ADASYN, which require numerical features.
        
        Args:
            texts (List[str]): The list of texts.
            
        Returns:
            np.ndarray: Numerical features extracted from text.
        """
        # Use a simple bag-of-words representation
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer(
            max_features=1000,  # Limit to 1000 features to avoid memory issues
            min_df=2,           # Ignore terms that appear in less than 2 documents
            max_df=0.95,        # Ignore terms that appear in more than 95% of documents
            binary=True         # Binary weights
        )
        
        try:
            features = vectorizer.fit_transform(texts).toarray()
            return features
        except:
            # Fallback to simple character counts if vectorization fails
            logger.warning("Vectorization failed. Falling back to simple character counts.")
            features = np.zeros((len(texts), 26))  # 26 letters of the alphabet
            
            for i, text in enumerate(texts):
                # Count occurrences of each letter
                counter = Counter(c.lower() for c in text if c.isalpha())
                for j, char in enumerate(string.ascii_lowercase):
                    features[i, j] = counter.get(char, 0)
                    
            return features
            
    def _random_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply random oversampling to balance the dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Balanced dataframe.
        """
        logger.info("Applying random oversampling...")
        
        # Ensure label column exists
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe")
            
        # Count class distribution
        class_counts = df[self.label_column].value_counts()
        majority_class = class_counts.index[0]
        majority_count = class_counts[majority_class]
        
        # Compute the target counts based on sampling strategy
        target_counts = {}
        
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy == 'auto':
                # Balance all classes to match the majority class
                for cls in class_counts.index:
                    target_counts[cls] = majority_count
            elif self.sampling_strategy == 'minority':
                # Oversample only the minority class
                minority_class = class_counts.index[-1]
                target_counts = {cls: count for cls, count in class_counts.items()}
                target_counts[minority_class] = majority_count
            elif self.sampling_strategy == 'not minority':
                # Oversample all classes except the minority class
                minority_class = class_counts.index[-1]
                target_counts = {cls: majority_count for cls in class_counts.index if cls != minority_class}
                target_counts[minority_class] = class_counts[minority_class]
            elif self.sampling_strategy == 'not majority':
                # Oversample all classes except the majority class
                for cls in class_counts.index:
                    if cls == majority_class:
                        target_counts[cls] = class_counts[cls]
                    else:
                        target_counts[cls] = majority_count
        elif isinstance(self.sampling_strategy, float):
            # Use the specified ratio
            minority_class = class_counts.index[-1]
            minority_target = int(majority_count * self.sampling_strategy)
            target_counts = {cls: count for cls, count in class_counts.items()}
            target_counts[minority_class] = max(minority_target, class_counts[minority_class])
        elif isinstance(self.sampling_strategy, dict):
            # Use the specified counts
            target_counts = self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")
            
        # Perform oversampling
        random.seed(self.random_state)
        balanced_dfs = []
        
        for cls in target_counts:
            class_df = df[df[self.label_column] == cls]
            class_count = len(class_df)
            target_count = target_counts.get(cls, class_count)
            
            if target_count <= class_count:
                # No need to oversample
                balanced_dfs.append(class_df)
            else:
                # Number of additional samples needed
                n_additional = target_count - class_count
                
                # Randomly sample with replacement
                additional_df = class_df.sample(
                    n=n_additional, 
                    replace=True, 
                    random_state=self.random_state
                )
                
                balanced_dfs.append(pd.concat([class_df, additional_df]))
                
        # Combine oversampled classes
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Compute statistics
        before_counts = dict(class_counts)
        after_counts = dict(balanced_df[self.label_column].value_counts())
        
        stats = {
            'total_samples_before': len(df),
            'total_samples_after': len(balanced_df),
            'class_distribution_before': before_counts,
            'class_distribution_after': after_counts,
            'strategy': 'random-oversampling',
            'sampling_strategy': self.sampling_strategy
        }
        
        self.stats = stats
        
        return balanced_df
        
    def _smote_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Balanced dataframe.
        """
        logger.info("Applying SMOTE oversampling...")
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn package not available. Falling back to random oversampling.")
            return self._random_oversampling(df)
            
        # Ensure label column exists
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe")
            
        # Extract features for SMOTE
        logger.info("Extracting features for SMOTE...")
        X = self._extract_text_features(df[self.text_column].tolist())
        y = df[self.label_column].values
        
        # Count class distribution before
        before_counts = dict(Counter(y))
        
        # Initialize SMOTE
        try:
            smote = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                k_neighbors=min(self.k_neighbors, min(Counter(y).values()) - 1)
            )
            
            # Apply SMOTE
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Create new dataframe with synthetic samples
            original_indices = np.arange(len(df))
            synthetic_indices = np.arange(len(df), len(y_resampled))
            
            # Keep track of which samples are original vs. synthetic
            is_synthetic = np.zeros(len(y_resampled), dtype=bool)
            is_synthetic[synthetic_indices] = True
            
            # Create resampled dataframe
            resampled_data = {
                self.label_column: y_resampled,
                'is_synthetic': is_synthetic
            }
            
            # Original samples remain unchanged
            for col in df.columns:
                if col != self.label_column:
                    resampled_data[col] = [None] * len(y_resampled)
                    for i, idx in enumerate(original_indices):
                        resampled_data[col][i] = df.iloc[idx][col]
                        
            # For text column, generate synthetic text for synthetic samples
            # Note: This is a placeholder implementation, as SMOTE doesn't directly handle text
            # We'll implement a simple approach using nearest neighbors
            if len(synthetic_indices) > 0:
                logger.info("Generating synthetic text samples...")
                
                # Create a nearest neighbors model on the original data
                nn = NearestNeighbors(n_neighbors=min(5, len(df)), n_jobs=-1)
                nn.fit(X)
                
                # Get synthetic feature vectors
                X_synthetic = X_resampled[synthetic_indices]
                
                # Find nearest neighbors for each synthetic sample
                distances, indices = nn.kneighbors(X_synthetic)
                
                # For each synthetic sample, use text from the nearest neighbor
                original_texts = df[self.text_column].tolist()
                for i, idx in enumerate(synthetic_indices):
                    nearest_text = original_texts[indices[i, 0]]
                    resampled_data[self.text_column][idx] = f"SYNTHETIC: {nearest_text}"
                    
                    # For other columns, copy from nearest neighbor
                    for col in df.columns:
                        if col != self.label_column and col != self.text_column:
                            resampled_data[col][idx] = df.iloc[indices[i, 0]][col]
                
            # Create balanced dataframe
            balanced_df = pd.DataFrame(resampled_data)
            
            # Compute statistics
            after_counts = dict(Counter(y_resampled))
            
            stats = {
                'total_samples_before': len(df),
                'total_samples_after': len(balanced_df),
                'class_distribution_before': before_counts,
                'class_distribution_after': after_counts,
                'strategy': 'smote',
                'sampling_strategy': self.sampling_strategy,
                'synthetic_samples': len(synthetic_indices)
            }
            
            self.stats = stats
            
            return balanced_df
            
        except Exception as e:
            logger.error(f"Error applying SMOTE: {e}")
            logger.warning("Falling back to random oversampling.")
            return self._random_oversampling(df)
            
    def _adasyn_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply ADASYN (Adaptive Synthetic Sampling) to balance the dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Balanced dataframe.
        """
        logger.info("Applying ADASYN oversampling...")
        
        if not IMBLEARN_AVAILABLE:
            logger.warning("imbalanced-learn package not available. Falling back to random oversampling.")
            return self._random_oversampling(df)
            
        # Ensure label column exists
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe")
            
        # Extract features for ADASYN
        logger.info("Extracting features for ADASYN...")
        X = self._extract_text_features(df[self.text_column].tolist())
        y = df[self.label_column].values
        
        # Count class distribution before
        before_counts = dict(Counter(y))
        
        # Initialize ADASYN
        try:
            adasyn = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
                n_neighbors=min(self.k_neighbors, min(Counter(y).values()) - 1),
                n_jobs=-1
            )
            
            # Apply ADASYN
            X_resampled, y_resampled = adasyn.fit_resample(X, y)
            
            # Create new dataframe with synthetic samples
            original_indices = np.arange(len(df))
            synthetic_indices = np.arange(len(df), len(y_resampled))
            
            # Keep track of which samples are original vs. synthetic
            is_synthetic = np.zeros(len(y_resampled), dtype=bool)
            is_synthetic[synthetic_indices] = True
            
            # Create resampled dataframe
            resampled_data = {
                self.label_column: y_resampled,
                'is_synthetic': is_synthetic
            }
            
            # Original samples remain unchanged
            for col in df.columns:
                if col != self.label_column:
                    resampled_data[col] = [None] * len(y_resampled)
                    for i, idx in enumerate(original_indices):
                        resampled_data[col][i] = df.iloc[idx][col]
                        
            # For text column, generate synthetic text for synthetic samples
            # Similar approach as SMOTE
            if len(synthetic_indices) > 0:
                logger.info("Generating synthetic text samples...")
                
                # Create a nearest neighbors model on the original data
                nn = NearestNeighbors(n_neighbors=min(5, len(df)), n_jobs=-1)
                nn.fit(X)
                
                # Get synthetic feature vectors
                X_synthetic = X_resampled[synthetic_indices]
                
                # Find nearest neighbors for each synthetic sample
                distances, indices = nn.kneighbors(X_synthetic)
                
                # For each synthetic sample, use text from the nearest neighbor
                original_texts = df[self.text_column].tolist()
                for i, idx in enumerate(synthetic_indices):
                    nearest_text = original_texts[indices[i, 0]]
                    resampled_data[self.text_column][idx] = f"SYNTHETIC: {nearest_text}"
                    
                    # For other columns, copy from nearest neighbor
                    for col in df.columns:
                        if col != self.label_column and col != self.text_column:
                            resampled_data[col][idx] = df.iloc[indices[i, 0]][col]
                
            # Create balanced dataframe
            balanced_df = pd.DataFrame(resampled_data)
            
            # Compute statistics
            after_counts = dict(Counter(y_resampled))
            
            stats = {
                'total_samples_before': len(df),
                'total_samples_after': len(balanced_df),
                'class_distribution_before': before_counts,
                'class_distribution_after': after_counts,
                'strategy': 'adasyn',
                'sampling_strategy': self.sampling_strategy,
                'synthetic_samples': len(synthetic_indices)
            }
            
            self.stats = stats
            
            return balanced_df
            
        except Exception as e:
            logger.error(f"Error applying ADASYN: {e}")
            logger.warning("Falling back to random oversampling.")
            return self._random_oversampling(df)
            
    def _synonym_replacement(self, words: List[str], n: int) -> List[str]:
        """
        Replace n random words with their synonyms.
        
        Args:
            words (List[str]): List of words in the text.
            n (int): Number of words to replace.
            
        Returns:
            List[str]: The augmented list of words.
        """
        # Simple synonym list for demonstration
        # In a real implementation, use a proper synonym library like WordNet
        synonyms = {
            'good': ['nice', 'excellent', 'great', 'positive', 'wonderful'],
            'bad': ['poor', 'negative', 'terrible', 'awful', 'horrible'],
            'happy': ['glad', 'joyful', 'pleased', 'content', 'delighted'],
            'sad': ['unhappy', 'depressed', 'gloomy', 'miserable', 'sorrowful'],
            'big': ['large', 'huge', 'enormous', 'massive', 'gigantic'],
            'small': ['tiny', 'little', 'miniature', 'compact', 'microscopic'],
            'smart': ['intelligent', 'clever', 'bright', 'brilliant', 'wise'],
            'stupid': ['dumb', 'foolish', 'idiotic', 'moronic', 'brainless'],
            'beautiful': ['pretty', 'lovely', 'gorgeous', 'attractive', 'stunning'],
            'ugly': ['unattractive', 'hideous', 'repulsive', 'unsightly', 'grotesque'],
            'important': ['significant', 'crucial', 'essential', 'vital', 'critical'],
            'unimportant': ['insignificant', 'trivial', 'minor', 'negligible', 'inconsequential'],
            'interesting': ['fascinating', 'intriguing', 'engaging', 'captivating', 'compelling'],
            'boring': ['dull', 'tedious', 'monotonous', 'uninteresting', 'dreary'],
            'easy': ['simple', 'straightforward', 'effortless', 'uncomplicated', 'painless'],
            'difficult': ['hard', 'challenging', 'complicated', 'complex', 'demanding']
        }
        
        new_words = words.copy()
        random_word_indices = list(range(len(words)))
        random.shuffle(random_word_indices)
        
        num_replaced = 0
        for idx in random_word_indices:
            word = words[idx].lower()
            if word in synonyms:
                synonym = random.choice(synonyms[word])
                new_words[idx] = synonym
                num_replaced += 1
                
            if num_replaced >= n:
                break
                
        return new_words
        
    def _random_insertion(self, words: List[str], n: int) -> List[str]:
        """
        Randomly insert n words into the text.
        
        Args:
            words (List[str]): List of words in the text.
            n (int): Number of words to insert.
            
        Returns:
            List[str]: The augmented list of words.
        """
        # Simple word list for demonstration
        # In a real implementation, use a proper word corpus
        common_words = [
            'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
            'about', 'above', 'across', 'after', 'against', 'along', 'among', 'around',
            'very', 'quite', 'rather', 'somewhat', 'extremely', 'slightly',
            'often', 'sometimes', 'rarely', 'occasionally', 'frequently',
            'well', 'poorly', 'effectively', 'efficiently', 'successfully'
        ]
        
        new_words = words.copy()
        for _ in range(n):
            if not words:
                continue
                
            random_word = random.choice(words)
            random_idx = random.randint(0, len(new_words))
            new_words.insert(random_idx, random_word)
            
        return new_words
        
    def _random_swap(self, words: List[str], n: int) -> List[str]:
        """
        Randomly swap the positions of n pairs of words.
        
        Args:
            words (List[str]): List of words in the text.
            n (int): Number of word pairs to swap.
            
        Returns:
            List[str]: The augmented list of words.
        """
        if len(words) < 2:
            return words
            
        new_words = words.copy()
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
            
        return new_words
        
    def _random_deletion(self, words: List[str], p: float) -> List[str]:
        """
        Randomly delete words with probability p.
        
        Args:
            words (List[str]): List of words in the text.
            p (float): Probability of deletion.
            
        Returns:
            List[str]: The augmented list of words.
        """
        if len(words) == 1:
            return words
            
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
                
        if not new_words:
            # If all words were deleted, keep at least one
            return [random.choice(words)]
            
        return new_words
        
    def _eda_augment(self, text: str, num_aug: int = 4, alpha: float = 0.1) -> List[str]:
        """
        Apply Easy Data Augmentation (EDA) to generate augmented text samples.
        
        Args:
            text (str): The original text.
            num_aug (int): Number of augmented samples to generate.
            alpha (float): Parameter controlling the strength of augmentation.
            
        Returns:
            List[str]: List of augmented text samples.
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()
        words = text.split()
        
        # Skip if text is too short
        if len(words) <= 3:
            return [text] * num_aug
            
        augmented_texts = []
        
        for _ in range(num_aug):
            # 1. Synonym replacement
            n_sr = max(1, int(alpha * len(words)))
            aug_sr = self._synonym_replacement(words, n_sr)
            
            # 2. Random insertion
            n_ri = max(1, int(alpha * len(words)))
            aug_ri = self._random_insertion(words, n_ri)
            
            # 3. Random swap
            n_rs = max(1, int(alpha * len(words)))
            aug_rs = self._random_swap(words, n_rs)
            
            # 4. Random deletion
            aug_rd = self._random_deletion(words, alpha)
            
            # Choose one of the augmentation methods randomly
            aug_methods = [aug_sr, aug_ri, aug_rs, aug_rd]
            augmented = random.choice(aug_methods)
            
            # Convert back to text
            augmented_text = ' '.join(augmented)
            augmented_texts.append(augmented_text)
            
        return augmented_texts
        
    def _eda_oversampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply Easy Data Augmentation (EDA) to balance the dataset.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: Balanced dataframe.
        """
        logger.info("Applying Easy Data Augmentation (EDA)...")
        
        # Ensure label column exists
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe")
            
        # Ensure text column exists
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataframe")
            
        # Count class distribution
        class_counts = df[self.label_column].value_counts()
        majority_class = class_counts.index[0]
        majority_count = class_counts[majority_class]
        
        # Compute the target counts based on sampling strategy
        target_counts = {}
        
        if isinstance(self.sampling_strategy, str):
            if self.sampling_strategy == 'auto':
                # Balance all classes to match the majority class
                for cls in class_counts.index:
                    target_counts[cls] = majority_count
            elif self.sampling_strategy == 'minority':
                # Oversample only the minority class
                minority_class = class_counts.index[-1]
                target_counts = {cls: count for cls, count in class_counts.items()}
                target_counts[minority_class] = majority_count
            elif self.sampling_strategy == 'not minority':
                # Oversample all classes except the minority class
                minority_class = class_counts.index[-1]
                target_counts = {cls: majority_count for cls in class_counts.index if cls != minority_class}
                target_counts[minority_class] = class_counts[minority_class]
            elif self.sampling_strategy == 'not majority':
                # Oversample all classes except the majority class
                for cls in class_counts.index:
                    if cls == majority_class:
                        target_counts[cls] = class_counts[cls]
                    else:
                        target_counts[cls] = majority_count
        elif isinstance(self.sampling_strategy, float):
            # Use the specified ratio
            minority_class = class_counts.index[-1]
            minority_target = int(majority_count * self.sampling_strategy)
            target_counts = {cls: count for cls, count in class_counts.items()}
            target_counts[minority_class] = max(minority_target, class_counts[minority_class])
        elif isinstance(self.sampling_strategy, dict):
            # Use the specified counts
            target_counts = self.sampling_strategy
        else:
            raise ValueError(f"Invalid sampling_strategy: {self.sampling_strategy}")
            
        # Perform EDA oversampling
        random.seed(self.random_state)
        balanced_dfs = []
        
        for cls in target_counts:
            class_df = df[df[self.label_column] == cls]
            class_count = len(class_df)
            target_count = target_counts.get(cls, class_count)
            
            if target_count <= class_count:
                # No need to oversample
                balanced_dfs.append(class_df)
            else:
                # Number of additional samples needed
                n_additional = target_count - class_count
                
                # Apply EDA to generate synthetic samples
                synthetic_samples = []
                
                # Calculate how many augmentations per sample
                aug_per_sample = min(self.eda_num_aug, n_additional // class_count + 1)
                
                # Progress bar for verbose mode
                iterator = class_df.iterrows()
                if self.verbose:
                    iterator = tqdm(iterator, total=len(class_df), desc=f"Augmenting class {cls}")
                    
                # Generate augmented samples for each original sample
                for _, row in iterator:
                    text = row[self.text_column]
                    augmented_texts = self._eda_augment(
                        text,
                        num_aug=aug_per_sample,
                        alpha=self.eda_alpha
                    )
                    
                    for aug_text in augmented_texts:
                        # Create new sample with augmented text
                        new_sample = row.copy()
                        new_sample[self.text_column] = aug_text
                        new_sample['is_synthetic'] = True
                        synthetic_samples.append(new_sample)
                        
                        # Check if we have enough samples
                        if len(synthetic_samples) >= n_additional:
                            break
                            
                    if len(synthetic_samples) >= n_additional:
                        break
                        
                # Create dataframe with synthetic samples
                synthetic_df = pd.DataFrame(synthetic_samples)
                
                # Ensure we don't add too many samples
                if len(synthetic_df) > n_additional:
                    synthetic_df = synthetic_df.sample(n=n_additional, random_state=self.random_state)
                    
                # Add is_synthetic column to original class_df
                class_df = class_df.copy()
                class_df['is_synthetic'] = False
                
                # Combine original and synthetic samples
                balanced_dfs.append(pd.concat([class_df, synthetic_df]))
                
        # Combine all classes
        balanced_df = pd.concat(balanced_dfs).reset_index(drop=True)
        
        # Compute statistics
        before_counts = dict(class_counts)
        after_counts = dict(balanced_df[self.label_column].value_counts())
        synthetic_counts = dict(balanced_df[balanced_df['is_synthetic'] == True][self.label_column].value_counts())
        
        stats = {
            'total_samples_before': len(df),
            'total_samples_after': len(balanced_df),
            'class_distribution_before': before_counts,
            'class_distribution_after': after_counts,
            'synthetic_samples_by_class': synthetic_counts,
            'strategy': 'eda',
            'sampling_strategy': self.sampling_strategy,
            'eda_alpha': self.eda_alpha,
            'eda_num_aug': self.eda_num_aug
        }
        
        self.stats = stats
        
        return balanced_df
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the input dataframe by handling class imbalance.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The balanced dataframe.
        """
        if self.text_column not in df.columns:
            raise ValueError(f"Text column '{self.text_column}' not found in dataframe")
            
        if self.label_column not in df.columns:
            raise ValueError(f"Label column '{self.label_column}' not found in dataframe")
            
        # Check for empty dataframe
        if len(df) == 0:
            logger.warning("Empty dataframe provided")
            return df
            
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Apply the appropriate strategy
        if self.strategy == 'random-oversampling':
            balanced_df = self._random_oversampling(df)
        elif self.strategy == 'smote':
            balanced_df = self._smote_oversampling(df)
        elif self.strategy == 'adasyn':
            balanced_df = self._adasyn_oversampling(df)
        elif self.strategy == 'eda':
            balanced_df = self._eda_oversampling(df)
        else:
            # Should never get here due to validation in __init__
            raise ValueError(f"Invalid strategy: {self.strategy}")
            
        # Print statistics if verbose
        if self.verbose:
            logger.info("Imbalance Handling Statistics:")
            for key, value in self.stats.items():
                if isinstance(value, dict):
                    logger.info(f"  {key}:")
                    for subkey, subvalue in value.items():
                        logger.info(f"    {subkey}: {subvalue}")
                elif isinstance(value, float):
                    logger.info(f"  {key}: {value:.2f}")
                else:
                    logger.info(f"  {key}: {value}")
                    
        return balanced_df
        
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit and transform the input dataframe.
        
        Args:
            df (pd.DataFrame): The input dataframe.
            
        Returns:
            pd.DataFrame: The balanced dataframe.
        """
        # For this preprocessor, fit does nothing
        return self.transform(df)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get the imbalance handling statistics.
        
        Returns:
            Dict[str, Any]: The imbalance handling statistics.
        """
        return self.stats

# Example usage and demonstration
def demo():
    """
    Demonstrate the ImbalanceHandlingPreprocessor with a small example.
    """
    # Create a small example dataset with class imbalance
    data = {
        'text': [
            "This is a sample text from class 0",
            "Another example from class 0",
            "Yet another class 0 example",
            "One more text from class 0",
            "Class 0 text with some keywords",
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
    
    # Example 1: Random oversampling
    preprocessor = ImbalanceHandlingPreprocessor(strategy='random-oversampling')
    balanced_df = preprocessor.fit_transform(df)
    
    print("\nAfter random oversampling:")
    print(balanced_df[['text', 'label']].groupby('label').count())
    print(f"New shape: {balanced_df.shape}")
    
    # Example 2: EDA augmentation
    preprocessor = ImbalanceHandlingPreprocessor(
        strategy='eda',
        eda_alpha=0.1,
        eda_num_aug=4
    )
    balanced_df = preprocessor.fit_transform(df)
    
    print("\nAfter EDA augmentation:")
    print(balanced_df[['text', 'label']].groupby('label').count())
    print(f"New shape: {balanced_df.shape}")
    
    # Show some augmented examples
    print("\nSome augmented examples:")
    synthetic_samples = balanced_df[balanced_df['is_synthetic'] == True]
    for i, (_, row) in enumerate(synthetic_samples.iterrows()):
        if i < 5:  # Show first 5 examples
            print(f"Label: {row['label']}, Text: {row['text']}")
    
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    demo()
