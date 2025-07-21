#!/usr/bin/env python3




import os
import pandas as pd
import numpy as np
import time
import logging
import warnings
from datetime import datetime
from datasets import load_dataset
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Disable warnings
warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds into human readable time"""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"

class FixedTextCleaner:
    """Always-applied text cleaning with optimizable parameters"""
    
    def __init__(self, lowercase=True, remove_extra_whitespace=True, remove_punctuation=False):
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
        self.remove_punctuation = remove_punctuation
    
    def apply(self, data):
        result = data.copy()
        text_col = 'text'
        
        if self.lowercase:
            result[text_col] = result[text_col].str.lower()
        
        if self.remove_extra_whitespace:
            result[text_col] = result[text_col].str.replace(r'\s+', ' ', regex=True).str.strip()
        
        if self.remove_punctuation:
            import string
            result[text_col] = result[text_col].str.translate(str.maketrans('', '', string.punctuation))
        
        # Remove empty texts
        result = result[result[text_col].str.len() > 0].reset_index(drop=True)
        
        return result

class FixedTextFilter:
    """Learn quality criteria from training data with improved metrics"""

    def __init__(self, min_length_percentile=10, max_length_percentile=90, quality_threshold_percentile=20):
        self.min_length_percentile = min_length_percentile
        self.max_length_percentile = max_length_percentile
        self.quality_threshold_percentile = quality_threshold_percentile
        self.min_length_threshold = None
        self.max_length_threshold = None
        self.quality_threshold = None

    def _calculate_quality_score(self, data):
        """Calculate text quality scores based on multiple factors"""
        # 1. Length appropriateness (not too short/long)
        text_lengths = data['text'].str.len()
        length_scores = 1 - np.abs(text_lengths - text_lengths.median()) / (text_lengths.max() - text_lengths.min() + 1e-8)

        # 2. Word count appropriateness
        word_counts = data['text'].str.split().str.len()
        word_scores = 1 - np.abs(word_counts - word_counts.median()) / (word_counts.max() - word_counts.min() + 1e-8)

        # 3. Character diversity (avoid repetitive text)
        char_diversity = data['text'].apply(lambda x: len(set(x.lower())) / max(len(x), 1))

        # 4. Word diversity (avoid repetitive words)
        word_diversity = data['text'].apply(lambda x: len(set(x.lower().split())) / max(len(x.split()), 1))

        # 5. Sentence structure (has proper sentences)
        has_sentences = data['text'].apply(lambda x: 1 if any(p in x for p in '.!?') else 0)

        # 6. Language consistency (basic check for mixed languages/garbage)
        lang_consistency = data['text'].apply(lambda x: sum(1 for c in x if c.isalpha()) / max(len(x), 1))

        # Combine quality factors
        quality = (0.15 * length_scores + 0.15 * word_scores + 0.2 * char_diversity +
                  0.2 * word_diversity + 0.15 * has_sentences + 0.15 * lang_consistency)
        return quality

    def fit(self, train_data):
        """Learn thresholds from training data"""
        # Length thresholds
        text_lengths = train_data['text'].str.len()
        self.min_length_threshold = np.percentile(text_lengths, self.min_length_percentile)
        self.max_length_threshold = np.percentile(text_lengths, self.max_length_percentile)

        # Quality threshold
        quality_scores = self._calculate_quality_score(train_data)
        self.quality_threshold = np.percentile(quality_scores, self.quality_threshold_percentile)

        logger.info(f"Learned length thresholds: {self.min_length_threshold:.0f} - {self.max_length_threshold:.0f}")
        logger.info(f"Learned quality threshold: {self.quality_threshold:.3f}")

    def apply(self, data):
        """Apply learned thresholds"""
        if self.min_length_threshold is None:
            raise ValueError("Must call fit() first")

        original_size = len(data)
        
        # Length filtering
        text_lengths = data['text'].str.len()
        length_mask = (text_lengths >= self.min_length_threshold) & (text_lengths <= self.max_length_threshold)

        # Quality filtering
        quality_scores = self._calculate_quality_score(data)
        quality_mask = quality_scores >= self.quality_threshold

        # Combine filters
        combined_mask = length_mask & quality_mask
        result = data[combined_mask].reset_index(drop=True)

        # Add logging
        logger.info(f"Filtering: {original_size} -> {len(result)} (-{original_size - len(result)} samples)")
        
        return result

class FixedDifficultySelector:
    """Learn difficulty criteria from training data with improved metrics"""

    def __init__(self, strategy='balanced', difficulty_threshold_percentile=50):
        self.strategy = strategy
        self.difficulty_threshold_percentile = difficulty_threshold_percentile
        self.difficulty_threshold = None

    def _calculate_difficulty(self, data):
        """Calculate difficulty scores based on multiple text features"""
        text_lengths = data['text'].str.len()
        word_counts = data['text'].str.split().str.len()

        # 1. Basic complexity (length-based)
        length_norm = (text_lengths - text_lengths.min()) / (text_lengths.max() - text_lengths.min() + 1e-8)
        word_norm = (word_counts - word_counts.min()) / (word_counts.max() - word_counts.min() + 1e-8)

        # 2. Vocabulary complexity (unique words ratio)
        vocab_complexity = data['text'].apply(lambda x: len(set(x.split())) / max(len(x.split()), 1))
        vocab_norm = (vocab_complexity - vocab_complexity.min()) / (vocab_complexity.max() - vocab_complexity.min() + 1e-8)

        # 3. Sentence complexity (average sentence length)
        sentence_complexity = data['text'].apply(lambda x: np.mean([len(s.split()) for s in x.split('.') if s.strip()]))
        sentence_complexity = sentence_complexity.fillna(word_counts)  # Fallback to word count
        sentence_norm = (sentence_complexity - sentence_complexity.min()) / (sentence_complexity.max() - sentence_complexity.min() + 1e-8)

        # 4. Punctuation density (complexity indicator)
        punct_density = data['text'].apply(lambda x: sum(1 for c in x if c in '.,!?;:') / max(len(x), 1))
        punct_norm = (punct_density - punct_density.min()) / (punct_density.max() - punct_density.min() + 1e-8)

        # Combine multiple difficulty factors
        difficulty = (0.3 * length_norm + 0.2 * word_norm + 0.2 * vocab_norm +
                     0.15 * sentence_norm + 0.15 * punct_norm)
        return difficulty
    
    def fit(self, train_data):
        """Learn difficulty threshold from training data"""
        difficulty_scores = self._calculate_difficulty(train_data)
        self.difficulty_threshold = np.percentile(difficulty_scores, self.difficulty_threshold_percentile)
        logger.info(f"Learned difficulty threshold: {self.difficulty_threshold:.3f}")
    
    def apply(self, data):
        """Apply learned difficulty criteria"""
        if self.difficulty_threshold is None:
            raise ValueError("Must call fit() first")
        
        original_size = len(data)
        
        difficulty_scores = self._calculate_difficulty(data)
        
        if self.strategy == 'easy':
            mask = difficulty_scores <= self.difficulty_threshold
        elif self.strategy == 'hard':
            mask = difficulty_scores >= self.difficulty_threshold
        elif self.strategy == 'balanced':
            threshold_range = 0.2
            mask = np.abs(difficulty_scores - self.difficulty_threshold) <= threshold_range
        else:  # random
            mask = np.random.random(len(data)) > 0.3
        
        result = data[mask].reset_index(drop=True)
        
        # Add logging
        logger.info(f"Selection ({self.strategy}): {original_size} -> {len(result)} (-{original_size - len(result)} samples)")
        
        return result

class FixedTextAugmenter:
    """Strategic text augmentation focusing on class balancing and diversity"""

    def __init__(self, technique='mixed', augmentation_ratio=0.15, minority_boost=2.0):
        self.technique = technique
        self.augmentation_ratio = augmentation_ratio
        self.minority_boost = minority_boost  # Boost factor for minority classes

    def apply(self, data):
        """Apply strategic augmentation with class balancing"""
        if 'label' not in data.columns:
            logger.warning("No label column found, skipping augmentation")
            return data

        # Analyze class distribution
        class_counts = data['label'].value_counts()
        median_count = class_counts.median()

        augmented_rows = []

        for label in class_counts.index:
            class_data = data[data['label'] == label]
            class_count = len(class_data)

            # Calculate augmentation amount based on class size
            if class_count < median_count:
                # Minority class - augment more
                aug_ratio = self.augmentation_ratio * self.minority_boost
            else:
                # Majority class - augment less
                aug_ratio = self.augmentation_ratio * 0.5

            n_augment = int(class_count * aug_ratio)

            if n_augment > 0:
                # Sample texts to augment
                sample_data = class_data.sample(n=min(n_augment, len(class_data)), replace=True)

                for _, row in sample_data.iterrows():
                    # Choose augmentation technique
                    if self.technique == 'mixed':
                        chosen_technique = np.random.choice(['synonym', 'insertion', 'deletion', 'swap'])
                    else:
                        chosen_technique = self.technique

                    # Apply augmentation
                    augmented_text = self._augment_text(row['text'], chosen_technique)

                    # Create augmented row
                    augmented_row = row.copy()
                    augmented_row['text'] = augmented_text
                    augmented_rows.append(augmented_row)

        # Add augmented data to original
        if augmented_rows:
            augmented_df = pd.DataFrame(augmented_rows)
            result = pd.concat([data, augmented_df], ignore_index=True)
            logger.info(f"Augmentation: {len(data)} -> {len(result)} (+{len(augmented_rows)} samples)")
        else:
            result = data

        return result

    def _augment_text(self, text, technique):
        """Apply specific augmentation technique"""
        words = text.split()
        if len(words) < 2:
            return text  # Too short to augment

        try:
            if technique == 'synonym':
                return self._synonym_replacement(words)
            elif technique == 'insertion':
                return self._random_insertion(words)
            elif technique == 'deletion':
                return self._random_deletion(words)
            elif technique == 'swap':
                return self._word_swap(words)
            else:
                return text
        except:
            return text  # Fallback to original if augmentation fails

    def _synonym_replacement(self, words):
        """Replace random words with simple synonyms"""
        # Simple synonym dictionary for common words
        synonyms = {
            'good': ['great', 'excellent', 'fine', 'nice'],
            'bad': ['terrible', 'awful', 'poor', 'horrible'],
            'big': ['large', 'huge', 'enormous', 'massive'],
            'small': ['tiny', 'little', 'minor', 'compact'],
            'fast': ['quick', 'rapid', 'swift', 'speedy'],
            'slow': ['sluggish', 'gradual', 'delayed', 'leisurely'],
            'happy': ['joyful', 'cheerful', 'pleased', 'delighted'],
            'sad': ['unhappy', 'sorrowful', 'depressed', 'gloomy']
        }

        result_words = words.copy()
        n_replacements = max(1, len(words) // 10)  # Replace ~10% of words

        for _ in range(n_replacements):
            idx = np.random.randint(0, len(result_words))
            word = result_words[idx].lower()
            if word in synonyms:
                result_words[idx] = np.random.choice(synonyms[word])

        return ' '.join(result_words)

    def _random_insertion(self, words):
        """Insert random words at random positions"""
        insert_words = ['very', 'really', 'quite', 'somewhat', 'rather', 'fairly', 'pretty', 'extremely']

        result_words = words.copy()
        n_insertions = max(1, len(words) // 15)  # Insert ~7% more words

        for _ in range(n_insertions):
            insert_pos = np.random.randint(0, len(result_words))
            insert_word = np.random.choice(insert_words)
            result_words.insert(insert_pos, insert_word)

        return ' '.join(result_words)

    def _random_deletion(self, words):
        """Delete random words (but keep at least half)"""
        if len(words) <= 3:
            return ' '.join(words)  # Too short to delete

        result_words = words.copy()
        n_deletions = max(1, len(words) // 20)  # Delete ~5% of words

        for _ in range(min(n_deletions, len(result_words) // 2)):
            if len(result_words) > 2:
                del_pos = np.random.randint(0, len(result_words))
                result_words.pop(del_pos)

        return ' '.join(result_words)

    def _word_swap(self, words):
        """Swap adjacent words"""
        if len(words) < 3:
            return ' '.join(words)

        result_words = words.copy()
        n_swaps = max(1, len(words) // 20)  # Swap ~5% of adjacent pairs

        for _ in range(n_swaps):
            if len(result_words) >= 2:
                swap_pos = np.random.randint(0, len(result_words) - 1)
                result_words[swap_pos], result_words[swap_pos + 1] = result_words[swap_pos + 1], result_words[swap_pos]

        return ' '.join(result_words)

class FixedPipeline:
    """Fixed pipeline with logical stages including augmentation"""

    def __init__(self, cleaner_params, filter_params, selector_params, augmenter_params):
        self.cleaner = FixedTextCleaner(**cleaner_params)
        self.filter = FixedTextFilter(**filter_params) if filter_params else None
        self.selector = FixedDifficultySelector(**selector_params) if selector_params else None
        self.augmenter = FixedTextAugmenter(**augmenter_params) if augmenter_params else None
        self.fitted = False
    
    def fit(self, train_data):
        """Fit pipeline components on training data only"""
        logger.info("Fitting pipeline on training data...")
        
        original_size = len(train_data)
        current_data = train_data
        
        # Stage 1: Always clean
        current_data = self.cleaner.apply(current_data)
        logger.info(f"After cleaning: {original_size} -> {len(current_data)} (-{original_size - len(current_data)} samples)")

        # Stage 2: Optional filtering
        if self.filter:
            self.filter.fit(current_data)
            before_filter = len(current_data)
            current_data = self.filter.apply(current_data)
            logger.info(f"After filtering: {before_filter} -> {len(current_data)} (-{before_filter - len(current_data)} samples)")

        # Stage 3: Optional selection
        if self.selector:
            self.selector.fit(current_data)
            before_selection = len(current_data)
            current_data = self.selector.apply(current_data)
            logger.info(f"After selection: {before_selection} -> {len(current_data)} (-{before_selection - len(current_data)} samples)")

        # Stage 4: Optional augmentation
        if self.augmenter:
            before_augmentation = len(current_data)
            current_data = self.augmenter.apply(current_data)
            logger.info(f"After augmentation: {before_augmentation} -> {len(current_data)} (+{len(current_data) - before_augmentation} samples)")

        # Overall summary
        logger.info(f"Pipeline complete: {original_size} -> {len(current_data)} (net change: {len(current_data) - original_size:+d} samples)")
        
        self.fitted = True
        return current_data
    
    def transform(self, data, apply_selection=False):
        """Apply fitted pipeline to any data (train/val/test)"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        # Apply same learned criteria
        current_data = self.cleaner.apply(data)

        if self.filter:
            current_data = self.filter.apply(current_data)

        # Selection is optional - only apply if explicitly requested
        if self.selector and apply_selection:
            current_data = self.selector.apply(current_data)

        return current_data

class HybridExploratoryMetaLearner:
    """Hybrid exploratory meta-learning with multiple generation strategies"""

    def __init__(self, population_size=8, generations=5):
        self.population_size = population_size
        self.generations = generations
        self.best_pipeline = None
        self.best_fitness = 0.0

        # Define operation pool for exploratory approaches
        self.operation_pool = {
            # Cleaning operations
            'lowercase': {'apply': [True, False]},
            'remove_punctuation': {'apply': [True, False]},
            'normalize_whitespace': {'apply': [True, False]},
            'remove_numbers': {'apply': [True, False]},

            # Filtering operations (less aggressive to preserve data)
            'length_filter': {
                'min_percentile': [2, 5, 10],  # Less aggressive minimum
                'max_percentile': [90, 95, 98]  # Less aggressive maximum
            },
            'quality_filter': {
                'threshold_percentile': [5, 10, 20]  # Less aggressive quality filtering
            },

            # Selection operations (preserve more data)
            'difficulty_select_easy': {
                'threshold_percentile': [30, 40, 50],
                'ratio': [0.7, 0.8, 0.9]  # Keep more data
            },
            'difficulty_select_hard': {
                'threshold_percentile': [50, 60, 70],
                'ratio': [0.7, 0.8, 0.9]  # Keep more data
            },
            'difficulty_select_balanced': {
                'threshold_percentile': [40, 50, 60],
                'range': [0.2, 0.3, 0.4]  # Wider range = more data
            },
            'random_select': {
                'ratio': [0.8, 0.9, 0.95]  # Keep most data
            },

            # Augmentation operations
            'synonym_augment': {
                'ratio': [0.05, 0.1, 0.15, 0.2],
                'minority_boost': [1.5, 2.0, 2.5]
            },
            'insertion_augment': {
                'ratio': [0.05, 0.1, 0.15],
                'minority_boost': [1.5, 2.0, 2.5]
            },
            'deletion_augment': {
                'ratio': [0.03, 0.05, 0.08],
                'minority_boost': [1.5, 2.0]
            },
            'mixed_augment': {
                'ratio': [0.1, 0.15, 0.2, 0.25],
                'minority_boost': [1.5, 2.0, 2.5, 3.0]
            },
            'class_balance_augment': {
                'target_ratio': [1.0, 1.2, 1.5, 2.0]
            }
        }

        # Objectives for objective-driven generation
        self.objectives = {
            'maximize_accuracy': [
                ('quality_filter', {'threshold_percentile': 30}),
                ('difficulty_select_balanced', {'threshold_percentile': 50, 'range': 0.2}),
                ('synonym_augment', {'ratio': 0.1, 'minority_boost': 2.0})
            ],
            'maximize_robustness': [
                ('mixed_augment', {'ratio': 0.15, 'minority_boost': 2.0}),
                ('difficulty_select_hard', {'threshold_percentile': 60, 'ratio': 0.7}),
                ('insertion_augment', {'ratio': 0.1, 'minority_boost': 1.5})
            ],
            'minimize_data_size': [
                ('quality_filter', {'threshold_percentile': 40}),
                ('difficulty_select_easy', {'threshold_percentile': 40, 'ratio': 0.6}),
                ('length_filter', {'min_percentile': 15, 'max_percentile': 85})
            ],
            'maximize_diversity': [
                ('mixed_augment', {'ratio': 0.2, 'minority_boost': 2.5}),
                ('random_select', {'ratio': 0.8}),
                ('class_balance_augment', {'target_ratio': 1.5})
            ]
        }
    
    def generate_hybrid_pipeline(self):
        """Generate pipeline using hybrid approach with multiple strategies"""
        # Choose generation strategy
        approach = np.random.choice(['staged', 'random', 'objective', 'conditional'],
                                  p=[0.4, 0.3, 0.2, 0.1])

        if approach == 'staged':
            return self._generate_staged_pipeline()
        elif approach == 'random':
            return self._generate_random_pipeline()
        elif approach == 'objective':
            return self._generate_objective_pipeline()
        else:
            return self._generate_conditional_pipeline()

    def _generate_staged_pipeline(self):
        """Generate pipeline with logical stages (current approach)"""
        """Generate random pipeline configuration with all stages"""
        # Stage 1: Cleaning (always included, optimize parameters)
        cleaner_params = {
            'lowercase': True,  # Almost always good
            'remove_extra_whitespace': True,  # Almost always good
            'remove_punctuation': np.random.choice([True, False])
        }

        # Stage 2: Filtering (optional)
        filter_params = None
        if np.random.random() < 0.7:  # 70% chance to include
            filter_params = {
                'min_length_percentile': np.random.choice([5, 10, 15]),
                'max_length_percentile': np.random.choice([85, 90, 95]),
                'quality_threshold_percentile': np.random.choice([10, 20, 30])
            }

        # Stage 3: Selection (optional - strategic choice, not always needed)
        selector_params = None
        if np.random.random() < 0.3:  # Only 30% chance to include selection
            selector_params = {
                'strategy': np.random.choice(['easy', 'hard', 'balanced', 'random']),
                'difficulty_threshold_percentile': np.random.choice([30, 40, 50, 60, 70])
            }

        # Stage 4: Augmentation (optional - for data expansion)
        augmenter_params = None
        if np.random.random() < 0.4:  # 40% chance to include augmentation
            augmenter_params = {
                'technique': np.random.choice(['mixed', 'synonym', 'insertion', 'deletion', 'swap']),
                'augmentation_ratio': np.random.choice([0.1, 0.15, 0.2, 0.25]),
                'minority_boost': np.random.choice([1.5, 2.0, 2.5])
            }

        return {
            'approach': 'staged',
            'operations': [
                ('clean', cleaner_params),
                ('filter', filter_params) if filter_params else None,
                ('select', selector_params) if selector_params else None,
                ('augment', augmenter_params) if augmenter_params else None
            ]
        }

    def _generate_random_pipeline(self):
        """Generate completely random operation sequence"""
        # Random sequence length (1-6 operations)
        seq_length = np.random.randint(1, 7)

        # Sample random operations
        operation_names = list(self.operation_pool.keys())
        selected_ops = np.random.choice(operation_names, seq_length, replace=False)

        operations = []
        for op_name in selected_ops:
            # Sample random parameters for this operation
            params = {}
            for param_name, param_values in self.operation_pool[op_name].items():
                params[param_name] = np.random.choice(param_values)

            operations.append((op_name, params))

        return {
            'approach': 'random',
            'operations': operations
        }

    def _generate_objective_pipeline(self):
        """Generate pipeline driven by specific objectives"""
        # Choose 1-2 random objectives
        chosen_objectives = np.random.choice(list(self.objectives.keys()),
                                           size=np.random.randint(1, 3),
                                           replace=False)

        operations = []
        for objective in chosen_objectives:
            objective_ops = self.objectives[objective]
            # Add random subset of operations for this objective
            n_ops = np.random.randint(1, len(objective_ops) + 1)
            selected_ops = np.random.choice(len(objective_ops), n_ops, replace=False)

            for op_idx in selected_ops:
                op_name, params = objective_ops[op_idx]
                operations.append((op_name, params))

        # Remove duplicates while preserving order
        seen_ops = set()
        unique_operations = []
        for op_name, params in operations:
            if op_name not in seen_ops:
                unique_operations.append((op_name, params))
                seen_ops.add(op_name)

        # Shuffle for exploration
        np.random.shuffle(unique_operations)

        return {
            'approach': 'objective',
            'objectives': chosen_objectives.tolist(),
            'operations': unique_operations
        }

    def _generate_conditional_pipeline(self):
        """Generate pipeline with conditional operation chains"""
        operations = []
        data_state = {'size': 1.0, 'quality': 0.5, 'balance': 0.5, 'diversity': 0.5}

        for step in range(np.random.randint(2, 6)):
            available_ops = self._get_available_operations(data_state)

            if not available_ops:
                break

            op_name = np.random.choice(available_ops)
            params = self._sample_conditional_params(op_name, data_state)

            operations.append((op_name, params))

            # Update simulated data state
            data_state = self._simulate_operation_effect(op_name, params, data_state)

        return {
            'approach': 'conditional',
            'operations': operations
        }

    def _get_available_operations(self, data_state):
        """Get operations that make sense given current data state"""
        available = []

        # Always available: basic cleaning
        available.extend(['lowercase', 'normalize_whitespace'])

        # If data is large, selection operations are useful
        if data_state['size'] > 0.7:
            available.extend(['difficulty_select_easy', 'difficulty_select_hard', 'random_select'])

        # If data is imbalanced, augmentation is useful
        if data_state['balance'] < 0.7:
            available.extend(['class_balance_augment', 'mixed_augment'])

        # If data quality is low, filtering is useful
        if data_state['quality'] < 0.6:
            available.extend(['quality_filter', 'length_filter'])

        # If data lacks diversity, augmentation helps
        if data_state['diversity'] < 0.6:
            available.extend(['synonym_augment', 'insertion_augment'])

        # If data is small, augmentation is beneficial
        if data_state['size'] < 0.5:
            available.extend(['mixed_augment', 'synonym_augment'])

        return available

    def _sample_conditional_params(self, op_name, data_state):
        """Sample parameters based on data state"""
        if op_name in self.operation_pool:
            params = {}
            for param_name, param_values in self.operation_pool[op_name].items():
                if param_name == 'ratio' and data_state['size'] < 0.5:
                    # Use higher ratios for small datasets
                    params[param_name] = np.random.choice([v for v in param_values if v >= 0.1])
                elif param_name == 'threshold_percentile' and data_state['quality'] < 0.5:
                    # Use lower thresholds for low quality data
                    params[param_name] = np.random.choice([v for v in param_values if v <= 30])
                else:
                    params[param_name] = np.random.choice(param_values)
            return params
        else:
            return {'apply': True}

    def _simulate_operation_effect(self, op_name, params, data_state):
        """Simulate how operation affects data state"""
        new_state = data_state.copy()

        if 'select' in op_name or 'filter' in op_name:
            # Selection/filtering reduces size
            ratio = params.get('ratio', 0.8)
            new_state['size'] *= ratio
            new_state['quality'] += 0.1  # Assume filtering improves quality

        elif 'augment' in op_name:
            # Augmentation increases size and diversity
            ratio = params.get('ratio', 0.1)
            new_state['size'] *= (1 + ratio)
            new_state['diversity'] += 0.1
            if 'balance' in op_name:
                new_state['balance'] += 0.2

        elif op_name in ['lowercase', 'normalize_whitespace', 'remove_punctuation']:
            # Cleaning improves quality slightly
            new_state['quality'] += 0.05

        # Clamp values between 0 and 1
        for key in new_state:
            new_state[key] = max(0.0, min(1.0, new_state[key]))

        return new_state
    
    def evaluate_pipeline(self, pipeline_config, train_data, target_col):
        """Evaluate flexible pipeline using only training data with data retention penalty"""
        try:
            original_size = len(train_data)

            # Apply operations in sequence
            processed_train = self._apply_pipeline_operations(pipeline_config, train_data)
            processed_size = len(processed_train)

            # Calculate data retention ratio
            retention_ratio = processed_size / original_size

            # Penalize aggressive data reduction
            if retention_ratio < 0.3:  # Less than 30% retained
                return 0.0  # Completely reject
            elif processed_size < 50:  # Too few samples
                return 0.0

            # Split processed training data for evaluation
            train_subset, val_subset = train_test_split(
                processed_train, test_size=0.3, random_state=42,
                stratify=processed_train[target_col]
            )

            # Get accuracy
            accuracy = self._quick_evaluate(train_subset, val_subset, target_col)

            # Balanced fitness: accuracy + data retention bonus
            # Encourage keeping more data while maintaining quality
            retention_bonus = min(retention_ratio, 1.0)  # Cap at 1.0

            # Weighted fitness: 70% accuracy + 30% retention
            fitness = 0.7 * accuracy + 0.3 * retention_bonus

            return fitness

        except Exception as e:
            logger.warning(f"Pipeline evaluation failed: {e}")
            return 0.0

    def _apply_pipeline_operations(self, pipeline_config, data):
        """Apply sequence of operations to data"""
        current_data = data.copy()

        for operation in pipeline_config['operations']:
            if operation is None:
                continue

            op_name, params = operation

            try:
                if op_name == 'clean':
                    # Apply cleaning operations
                    current_data = self._apply_cleaning(current_data, params)
                elif op_name == 'filter':
                    # Apply filtering operations
                    current_data = self._apply_filtering(current_data, params)
                elif op_name == 'select':
                    # Apply selection operations
                    current_data = self._apply_selection(current_data, params)
                elif op_name == 'augment':
                    # Apply augmentation operations
                    current_data = self._apply_augmentation(current_data, params)
                else:
                    # Apply individual operations
                    current_data = self._apply_individual_operation(current_data, op_name, params)

            except Exception as e:
                logger.warning(f"Operation {op_name} failed: {e}")
                continue

        return current_data

    def _apply_cleaning(self, data, params):
        """Apply cleaning operations"""
        cleaner = FixedTextCleaner(**params)
        return cleaner.apply(data)

    def _apply_filtering(self, data, params):
        """Apply filtering operations"""
        filter_op = FixedTextFilter(**params)
        filter_op.fit(data)  # Fit on current data
        return filter_op.apply(data)

    def _apply_selection(self, data, params):
        """Apply selection operations"""
        selector = FixedDifficultySelector(**params)
        selector.fit(data)  # Fit on current data
        return selector.apply(data)

    def _apply_augmentation(self, data, params):
        """Apply augmentation operations"""
        augmenter = FixedTextAugmenter(**params)
        return augmenter.apply(data)

    def _apply_individual_operation(self, data, op_name, params):
        """Apply individual operations from the operation pool"""
        if op_name == 'lowercase':
            if params.get('apply', True):
                data = data.copy()
                data['text'] = data['text'].str.lower()

        elif op_name == 'remove_punctuation':
            if params.get('apply', True):
                import string
                data = data.copy()
                data['text'] = data['text'].str.translate(str.maketrans('', '', string.punctuation))

        elif op_name == 'normalize_whitespace':
            if params.get('apply', True):
                data = data.copy()
                data['text'] = data['text'].str.replace(r'\s+', ' ', regex=True).str.strip()

        elif op_name == 'remove_numbers':
            if params.get('apply', True):
                data = data.copy()
                data['text'] = data['text'].str.replace(r'\d+', '', regex=True)

        elif op_name == 'length_filter':
            text_lengths = data['text'].str.len()
            min_threshold = np.percentile(text_lengths, params['min_percentile'])
            max_threshold = np.percentile(text_lengths, params['max_percentile'])
            mask = (text_lengths >= min_threshold) & (text_lengths <= max_threshold)
            data = data[mask].reset_index(drop=True)

        elif op_name == 'quality_filter':
            # Simple quality filter based on character diversity
            char_diversity = data['text'].apply(lambda x: len(set(x.lower())) / max(len(x), 1))
            threshold = np.percentile(char_diversity, params['threshold_percentile'])
            data = data[char_diversity >= threshold].reset_index(drop=True)

        elif 'difficulty_select' in op_name:
            # Calculate difficulty and select based on strategy
            text_lengths = data['text'].str.len()
            word_counts = data['text'].str.split().str.len()
            difficulty = 0.6 * (text_lengths / text_lengths.max()) + 0.4 * (word_counts / word_counts.max())

            if 'easy' in op_name:
                threshold = np.percentile(difficulty, params['threshold_percentile'])
                mask = difficulty <= threshold
            elif 'hard' in op_name:
                threshold = np.percentile(difficulty, params['threshold_percentile'])
                mask = difficulty >= threshold
            elif 'balanced' in op_name:
                threshold = np.percentile(difficulty, params['threshold_percentile'])
                range_val = params.get('range', 0.2)
                mask = np.abs(difficulty - threshold/difficulty.max()) <= range_val

            if 'ratio' in params:
                # Apply ratio-based selection
                n_select = int(len(data) * params['ratio'])
                selected_indices = data[mask].sample(n=min(n_select, mask.sum())).index
                data = data.loc[selected_indices].reset_index(drop=True)
            else:
                data = data[mask].reset_index(drop=True)

        elif op_name == 'random_select':
            n_select = int(len(data) * params['ratio'])
            data = data.sample(n=n_select).reset_index(drop=True)

        elif 'augment' in op_name:
            # Apply specific augmentation
            augmenter = FixedTextAugmenter(
                technique=op_name.replace('_augment', ''),
                augmentation_ratio=params.get('ratio', 0.1),
                minority_boost=params.get('minority_boost', 2.0)
            )
            data = augmenter.apply(data)

        return data
    
    def _quick_evaluate(self, train_data, val_data, target_col):
        """Quick evaluation using RandomForest"""
        try:
            # Prepare features (just use text length and word count as simple features)
            def extract_features(data):
                features = pd.DataFrame({
                    'text_length': data['text'].str.len(),
                    'word_count': data['text'].str.split().str.len(),
                    'avg_word_length': data['text'].str.len() / data['text'].str.split().str.len()
                })
                return features.fillna(0)
            
            X_train = extract_features(train_data)
            y_train = train_data[target_col]
            X_val = extract_features(val_data)
            y_val = val_data[target_col]
            
            # Encode target if needed
            if y_train.dtype == 'object':
                le = LabelEncoder()
                y_train = le.fit_transform(y_train.astype(str))
                y_val = le.transform(y_val.astype(str))
            
            # Train and evaluate
            model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            return accuracy
            
        except Exception as e:
            logger.warning(f"Quick evaluation failed: {e}")
            return 0.0
    
    def run_meta_learning(self, train_data, target_col):
        """Run meta-learning on training data only"""
        logger.info("Starting meta-learning...")
        start_time = time.time()
        
        # Generate initial population using hybrid approach
        population = [self.generate_hybrid_pipeline() for _ in range(self.population_size)]
        
        for generation in range(self.generations):
            gen_start = time.time()
            logger.info(f"Generation {generation + 1}/{self.generations}")
            
            # Evaluate all pipelines
            fitness_scores = []
            for i, pipeline_config in enumerate(population):
                fitness = self.evaluate_pipeline(pipeline_config, train_data, target_col)
                fitness_scores.append(fitness)

                # Log data retention for transparency
                try:
                    processed_data = self._apply_pipeline_operations(pipeline_config, train_data)
                    retention = len(processed_data) / len(train_data)
                    logger.info(f"Pipeline {i+1}: fitness={fitness:.4f}, retention={retention:.2f} ({len(processed_data)}/{len(train_data)})")
                except:
                    logger.info(f"Pipeline {i+1}: fitness={fitness:.4f}")

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_pipeline = pipeline_config
                    logger.info(f"🏆 New best fitness: {fitness:.4f} (retention: {retention:.2f})")
            
            gen_time = time.time() - gen_start
            logger.info(f"Generation {generation + 1} completed in {gen_time:.2f}s - "
                       f"Best: {max(fitness_scores):.4f}, Avg: {np.mean(fitness_scores):.4f}")
            
            # Simple evolution: keep best half, mutate them for next generation
            if generation < self.generations - 1:
                # Sort by fitness
                sorted_indices = np.argsort(fitness_scores)[::-1]
                best_half = [population[i] for i in sorted_indices[:self.population_size//2]]
                
                # Create next generation
                new_population = best_half.copy()
                while len(new_population) < self.population_size:
                    # Mutate a random good pipeline
                    parent = np.random.choice(best_half)
                    child = self._mutate_pipeline(parent)
                    new_population.append(child)
                
                population = new_population
        
        total_time = time.time() - start_time
        logger.info(f"Meta-learning completed in {format_time(total_time)}")
        logger.info(f"Best fitness found: {self.best_fitness:.4f}")
        
        return self.best_pipeline
    
    def _mutate_pipeline(self, pipeline_config):
        """Mutate a flexible pipeline configuration"""
        mutated = pipeline_config.copy()
        mutated['operations'] = pipeline_config['operations'].copy()

        mutation_type = np.random.choice(['add', 'remove', 'modify', 'reorder'])

        if mutation_type == 'add' and len(mutated['operations']) < 8:
            # Add random operation
            operation_names = list(self.operation_pool.keys())
            new_op_name = np.random.choice(operation_names)

            # Sample parameters
            params = {}
            for param_name, param_values in self.operation_pool[new_op_name].items():
                params[param_name] = np.random.choice(param_values)

            # Insert at random position
            insert_pos = np.random.randint(0, len(mutated['operations']) + 1)
            mutated['operations'].insert(insert_pos, (new_op_name, params))

        elif mutation_type == 'remove' and len(mutated['operations']) > 1:
            # Remove random operation (but keep at least one)
            non_none_ops = [i for i, op in enumerate(mutated['operations']) if op is not None]
            if non_none_ops:
                remove_idx = np.random.choice(non_none_ops)
                mutated['operations'].pop(remove_idx)

        elif mutation_type == 'modify' and mutated['operations']:
            # Modify parameters of random operation
            non_none_ops = [i for i, op in enumerate(mutated['operations']) if op is not None]
            if non_none_ops:
                modify_idx = np.random.choice(non_none_ops)
                op_name, old_params = mutated['operations'][modify_idx]

                if op_name in self.operation_pool:
                    # Sample new parameters
                    new_params = {}
                    for param_name, param_values in self.operation_pool[op_name].items():
                        new_params[param_name] = np.random.choice(param_values)

                    mutated['operations'][modify_idx] = (op_name, new_params)

        elif mutation_type == 'reorder' and len(mutated['operations']) > 1:
            # Reorder operations
            non_none_ops = [op for op in mutated['operations'] if op is not None]
            if len(non_none_ops) > 1:
                np.random.shuffle(non_none_ops)
                # Replace non-None operations with shuffled versions
                non_none_idx = 0
                for i, op in enumerate(mutated['operations']):
                    if op is not None:
                        mutated['operations'][i] = non_none_ops[non_none_idx]
                        non_none_idx += 1

        return mutated

def load_anli_dataset():
    """Load ANLI R1 dataset"""
    start_time = time.time()
    logger.info("Loading ANLI R1 dataset...")

    try:
        dataset = load_dataset("facebook/anli")

        def prepare_anli_data(split_data):
            data = []
            for item in split_data:
                text_features = f"[PREMISE] {item['premise']} [HYPOTHESIS] {item['hypothesis']}"
                data.append({
                    'text': text_features,
                    'premise': item['premise'],
                    'hypothesis': item['hypothesis'],
                    'label': item['label']
                })
            return pd.DataFrame(data)

        train_df = prepare_anli_data(dataset['train_r1'])
        val_df = prepare_anli_data(dataset['dev_r1'])
        test_df = prepare_anli_data(dataset['test_r1'])

        load_time = time.time() - start_time
        logger.info(f"ANLI R1 loaded in {format_time(load_time)}")
        logger.info(f"Sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df, 'label'

    except Exception as e:
        logger.error(f"Error loading ANLI: {e}")
        return None, None, None, None

def load_casehold_dataset(self):
        """Load and prepare CaseHold dataset"""
        start_time = time.time()
        logger.info("Loading CaseHold dataset...")
        
        try:
            dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
            
            def prepare_casehold_data(split_data):
                data = []
                for item in split_data:
                    # Combine citing prompt with all holdings for context
                    text_features = item['citing_prompt']
                    for i in range(5):  # holdings 0-4
                        text_features += f" [HOLDING_{i}] " + item[f'holding_{i}']
                    
                    data.append({
                        'text': text_features,
                        'label': item['label']
                    })
                return pd.DataFrame(data)
            
            train_df = prepare_casehold_data(dataset['train'])
            val_df = prepare_casehold_data(dataset['validation'])
            test_df = prepare_casehold_data(dataset['test'])
            
            load_time = time.time() - start_time
            logger.info(f"CaseHold loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"CaseHold loading time: {format_time(load_time)}")
            return train_df, val_df, test_df, 'label'
            
        except Exception as e:
            logger.error(f"Error loading CaseHold: {e}")
            return None, None, None, None
    

def run_fast_experiment():
    """Run fast experiment on ANLI R1 only"""
    total_start = time.time()
    logger.info("="*80)
    logger.info("FAST FIXED META-LEARNING EXPERIMENT - ANLI R1 ONLY")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # 1. Load data
    train_df, val_df, test_df, target_col = load_anli_dataset()
    if train_df is None:
        logger.error("Failed to load dataset")
        return

    # 2. Run meta-learning (ONLY on training data)
    logger.info("\n" + "="*60)
    logger.info("PHASE 1: HYBRID EXPLORATORY META-LEARNING (Training data only)")
    logger.info("="*60)

    meta_learner = HybridExploratoryMetaLearner(population_size=8, generations=5)

    # Use subset for faster experimentation
    subset_size = min(2000, len(train_df))
    train_subset = train_df.sample(n=subset_size, random_state=42)
    logger.info(f"Using {len(train_subset)} training samples for meta-learning")

    best_pipeline_config = meta_learner.run_meta_learning(train_subset, target_col)

    if not best_pipeline_config:
        logger.error("No best pipeline found")
        return

    # 3. Apply discovered pipeline
    logger.info("\n" + "="*60)
    logger.info("PHASE 2: APPLYING DISCOVERED PIPELINE")
    logger.info("="*60)

    logger.info("Best pipeline configuration:")
    logger.info(f"  Approach: {best_pipeline_config['approach']}")
    if 'objectives' in best_pipeline_config:
        logger.info(f"  Objectives: {best_pipeline_config['objectives']}")
    logger.info("  Operations:")
    for i, operation in enumerate(best_pipeline_config['operations']):
        if operation is not None:
            op_name, params = operation
            logger.info(f"    {i+1}. {op_name}: {params}")
        else:
            logger.info(f"    {i+1}. (skipped)")

    # Apply discovered pipeline to FULL training data
    processing_start = time.time()
    processed_train = meta_learner._apply_pipeline_operations(best_pipeline_config, train_df)

    # Keep val/test completely untouched - no processing at all
    processed_val = val_df.copy()  # Original validation data
    processed_test = test_df.copy()  # Original test data

    processing_time = time.time() - processing_start
    logger.info(f"Data processing completed in {format_time(processing_time)}")

    logger.info("Data sizes after processing:")
    logger.info(f"  Train: {len(train_df)} -> {len(processed_train)} ({len(processed_train)/len(train_df)*100:.1f}%)")
    logger.info(f"  Val: {len(val_df)} -> {len(processed_val)} ({len(processed_val)/len(val_df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} -> {len(processed_test)} ({len(processed_test)/len(test_df)*100:.1f}%)")

    # 4. Train AutoGluon (using your exact settings)
    logger.info("\n" + "="*60)
    logger.info("PHASE 3: AUTOGLUON TRAINING (Your exact settings)")
    logger.info("="*60)

    autogluon_start = time.time()

    # Create model directory
    model_dir = "./anli_r1_model"
    os.makedirs(model_dir, exist_ok=True)

    # Use your exact train_text_model method for ANLI R1
    def train_text_model(train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Your exact TabularPredictor for text data using AutoGluon defaults"""
        logger.info(f"Using TabularPredictor for text data for {dataset_name}")

        predictor = TabularPredictor(
            label=target_col,
            path=model_dir,
            problem_type='multiclass',
            eval_metric='accuracy'
        )

        # Use AutoGluon's default settings - no custom model specifications
        logger.info("Using AutoGluon's default models and settings")

        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',  # Use AutoGluon's default preset
            time_limit=3000,
            verbosity=2,
            # Allow AutoGluon to use more memory
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
            # No included_model_types - let AutoGluon choose default models
        )

        training_time = time.time() - start_time
        logger.info(f"Text model training completed for {dataset_name} in {format_time(training_time)}")
        return predictor, training_time

    # Train using your exact method
    predictor, training_time = train_text_model(
        processed_train, processed_val, target_col, "anli_r1", model_dir, autogluon_start
    )

    # 5. Evaluate
    logger.info("\n" + "="*60)
    logger.info("PHASE 4: EVALUATION")
    logger.info("="*60)

    eval_start = time.time()
    performance = predictor.evaluate(processed_test, silent=True)
    leaderboard = predictor.leaderboard(processed_test, silent=True)
    eval_time = time.time() - eval_start

    logger.info(f"Evaluation completed in {format_time(eval_time)}")
    logger.info(f"Test performance: {performance}")
    logger.info(f"Best model: {leaderboard.iloc[0]['model']}")
    logger.info(f"Test score: {leaderboard.iloc[0]['score_test']}")

    # 6. Summary
    total_time = time.time() - total_start
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    logger.info(f"Meta-learning fitness: {meta_learner.best_fitness:.4f} (70% accuracy + 30% retention)")

    retention_ratio = len(processed_train) / len(train_df)
    logger.info(f"Data retention: {len(train_df)} -> {len(processed_train)} ({retention_ratio*100:.1f}%)")

    if retention_ratio < 0.5:
        logger.warning("⚠️  Low data retention - pipeline may be too aggressive")
    elif retention_ratio > 0.8:
        logger.info("✅ Good data retention - pipeline preserves most training data")
    else:
        logger.info("✅ Balanced data retention - reasonable trade-off")

    logger.info(f"AutoGluon performance: {performance}")
    logger.info(f"Processing time: {format_time(processing_time)}")
    logger.info(f"AutoGluon time: {format_time(training_time)}")
    logger.info(f"Total time: {format_time(total_time)}")
    logger.info("="*80)

    # Save results (convert non-serializable objects)
    def make_json_serializable(obj):
        """Convert objects to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(item) for item in obj]
        elif isinstance(obj, (bool, int, float, str, type(None))):
            return obj
        else:
            return str(obj)

    results = {
        'meta_learning_fitness': float(meta_learner.best_fitness),
        'best_pipeline_config': make_json_serializable(best_pipeline_config),
        'data_sizes': {
            'original': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
            'processed': {'train': len(processed_train), 'val': len(processed_val), 'test': len(processed_test)}
        },
        'autogluon_performance': str(performance),
        'timing': {
            'processing_time': float(processing_time),
            'autogluon_time': float(training_time),
            'total_time': float(total_time)
        }
    }

    import json
    with open('fast_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Results saved to fast_experiment_results.json")

if __name__ == "__main__":
    run_fast_experiment()
