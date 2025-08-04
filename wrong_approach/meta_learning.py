#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import time
import logging
import warnings
import random
from datetime import datetime
from datasets import load_dataset
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from PIL import Image
import io

# Set seeds for deterministic behavior
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

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

class UnifiedDataRepresentation:
    """Convert all data types to common embedding space for unified processing"""

    def __init__(self):
        self.text_vectorizer = None
        self.label_encoders = {}
        self.fitted = False

    def fit(self, data, target_col):
        """Fit the unified representation on training data"""
        from sklearn.feature_extraction.text import TfidfVectorizer

        logger.info("Fitting unified data representation...")

        # Fit text vectorizer on text column
        if 'text' in data.columns:
            self.text_vectorizer = TfidfVectorizer(
                max_features=1000,  # Limit features for efficiency
                stop_words='english',
                ngram_range=(1, 2),
                random_state=RANDOM_SEED
            )
            self.text_vectorizer.fit(data['text'].fillna(''))

        # Fit label encoders for categorical columns
        for col in data.columns:
            if col not in ['text', target_col, 'image', 'timestamp', 'item_id']:
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(data[col].fillna('unknown'))

        self.fitted = True
        logger.info("Unified data representation fitted successfully")

    def transform(self, data):
        """Transform data to unified embedding space"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        features = []
        feature_names = []

        # 1. Text features (TF-IDF)
        if 'text' in data.columns and self.text_vectorizer:
            text_features = self.text_vectorizer.transform(data['text'].fillna(''))
            features.append(text_features.toarray())
            feature_names.extend([f'text_tfidf_{i}' for i in range(text_features.shape[1])])

        # 2. Numerical features
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col not in ['label', 'answer', 'target']]

        if len(numerical_cols) > 0:
            numerical_features = data[numerical_cols].fillna(0).values
            features.append(numerical_features)
            feature_names.extend(numerical_cols.tolist())

        # 3. Categorical features (encoded)
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                try:
                    # Handle unseen categories
                    col_data = data[col].fillna('unknown')
                    encoded_col = []
                    for val in col_data:
                        try:
                            encoded_col.append(encoder.transform([str(val)])[0])
                        except ValueError:
                            # Unseen category, use 0 (first class)
                            encoded_col.append(0)

                    features.append(np.array(encoded_col).reshape(-1, 1))
                    feature_names.append(f'{col}_encoded')
                except Exception as e:
                    logger.warning(f"Error encoding column {col}: {e}")

        # 4. Time series features (if applicable)
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
            time_features = np.column_stack([
                timestamps.dt.hour.fillna(0),
                timestamps.dt.day.fillna(1),
                timestamps.dt.month.fillna(1),
                timestamps.dt.dayofweek.fillna(0)
            ])
            features.append(time_features)
            feature_names.extend(['hour', 'day', 'month', 'dayofweek'])

        # Combine all features
        if features:
            combined_features = np.hstack(features)

            # Create DataFrame with unified features
            unified_df = pd.DataFrame(combined_features, columns=feature_names)

            # Add target column if present
            for target_col in ['label', 'answer', 'target']:
                if target_col in data.columns:
                    unified_df[target_col] = data[target_col].values
                    break

            # Add item_id for time series
            if 'item_id' in data.columns:
                unified_df['item_id'] = data['item_id'].values

            # Add timestamp for time series
            if 'timestamp' in data.columns:
                unified_df['timestamp'] = data['timestamp'].values

            return unified_df
        else:
            logger.warning("No features extracted, returning original data")
            return data

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

class MLBasedPipelinePredictor:
    """ML-based pipeline parameter predictor using embeddings"""

    def __init__(self, n_candidates=10):
        self.n_candidates = n_candidates
        self.parameter_models = {}
        self.operation_models = {}
        self.fitted = False
        self.training_data = []
        self.best_pipeline = None
        self.best_fitness = 0.0

        # Define parameter spaces for ML prediction
        self.parameter_space = {
            'cleaner_params': {
                'lowercase': [True, False],
                'remove_extra_whitespace': [True, False],
                'remove_punctuation': [True, False]
            },
            'filter_params': {
                'min_length_percentile': [2, 5, 10, 15],
                'max_length_percentile': [85, 90, 95, 98],
                'quality_threshold_percentile': [5, 10, 20, 30]
            },
            'selector_params': {
                'strategy': ['easy', 'hard', 'balanced', 'random'],
                'difficulty_threshold_percentile': [30, 40, 50, 60, 70]
            },
            'augmenter_params': {
                'technique': ['mixed', 'synonym', 'insertion', 'deletion', 'swap'],
                'augmentation_ratio': [0.05, 0.1, 0.15, 0.2, 0.25],
                'minority_boost': [1.5, 2.0, 2.5, 3.0]
            }
        }

    def extract_embedding_features(self, data, target_col):
        """Extract statistical features from embeddings for ML prediction"""
        features = {}

        # Get embedding columns (exclude target and metadata)
        embedding_cols = [col for col in data.columns
                         if col not in [target_col, 'item_id', 'timestamp']
                         and data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

        if not embedding_cols:
            logger.warning("No embedding columns found for feature extraction")
            return pd.DataFrame([features])

        embedding_data = data[embedding_cols].values

        # Dataset size features
        features['dataset_size'] = len(data)
        features['n_features'] = len(embedding_cols)

        # Class distribution features
        if target_col in data.columns:
            class_counts = data[target_col].value_counts()
            features['n_classes'] = len(class_counts)
            features['class_imbalance'] = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
            features['majority_class_ratio'] = class_counts.max() / len(data)
            features['minority_class_ratio'] = class_counts.min() / len(data)
        else:
            features['n_classes'] = 1
            features['class_imbalance'] = 1.0
            features['majority_class_ratio'] = 1.0
            features['minority_class_ratio'] = 1.0

        # Embedding statistical features
        features['embedding_mean'] = np.mean(embedding_data)
        features['embedding_std'] = np.std(embedding_data)
        features['embedding_min'] = np.min(embedding_data)
        features['embedding_max'] = np.max(embedding_data)
        features['embedding_median'] = np.median(embedding_data)
        features['embedding_skewness'] = float(pd.DataFrame(embedding_data).skew().mean())
        features['embedding_kurtosis'] = float(pd.DataFrame(embedding_data).kurtosis().mean())

        # Feature correlation and complexity
        corr_matrix = np.corrcoef(embedding_data.T)
        features['avg_feature_correlation'] = np.mean(np.abs(corr_matrix[np.triu_indices_from(corr_matrix, k=1)]))
        features['feature_variance_ratio'] = np.var(np.var(embedding_data, axis=0))

        # Dimensionality features
        features['feature_density'] = len(data) / len(embedding_cols) if len(embedding_cols) > 0 else 0

        # Sample-wise features
        sample_norms = np.linalg.norm(embedding_data, axis=1)
        features['avg_sample_norm'] = np.mean(sample_norms)
        features['sample_norm_std'] = np.std(sample_norms)

        # Distance-based features
        if len(data) > 1:
            sample_distances = []
            for i in range(min(100, len(data))):  # Sample for efficiency
                for j in range(i+1, min(100, len(data))):
                    dist = np.linalg.norm(embedding_data[i] - embedding_data[j])
                    sample_distances.append(dist)

            if sample_distances:
                features['avg_sample_distance'] = np.mean(sample_distances)
                features['sample_distance_std'] = np.std(sample_distances)
            else:
                features['avg_sample_distance'] = 0
                features['sample_distance_std'] = 0
        else:
            features['avg_sample_distance'] = 0
            features['sample_distance_std'] = 0

        return pd.DataFrame([features])

    def predict_pipeline_parameters(self, data, target_col):
        """Predict optimal pipeline parameters using ML models trained on embeddings"""
        if not self.fitted:
            logger.warning("ML predictor not fitted, using random parameters")
            return self._generate_random_pipeline_config()

        # Extract embedding features
        embedding_features = self.extract_embedding_features(data, target_col)

        # Predict parameters for each component
        predicted_config = {
            'approach': 'ml_predicted',
            'operations': []
        }

        # Predict cleaner parameters
        cleaner_params = self._predict_component_params('cleaner', embedding_features)
        predicted_config['operations'].append(('clean', cleaner_params))

        # Predict whether to include filter and its parameters
        if self._should_include_component('filter', embedding_features):
            filter_params = self._predict_component_params('filter', embedding_features)
            predicted_config['operations'].append(('filter', filter_params))
        else:
            predicted_config['operations'].append(None)

        # Predict whether to include selector and its parameters
        if self._should_include_component('selector', embedding_features):
            selector_params = self._predict_component_params('selector', embedding_features)
            predicted_config['operations'].append(('select', selector_params))
        else:
            predicted_config['operations'].append(None)

        # Predict whether to include augmenter and its parameters
        if self._should_include_component('augmenter', embedding_features):
            augmenter_params = self._predict_component_params('augmenter', embedding_features)
            predicted_config['operations'].append(('augment', augmenter_params))
        else:
            predicted_config['operations'].append(None)

        return predicted_config

    def predict_pipeline_parameters_from_features(self, embedding_features, target_col):
        """Predict optimal pipeline parameters from pre-computed embedding features"""
        if not self.fitted:
            logger.warning("ML predictor not fitted, using random parameters")
            return self._generate_random_pipeline_config()

        # Predict parameters for each component
        predicted_config = {
            'approach': 'ml_predicted',
            'operations': []
        }

        # Predict cleaner parameters
        cleaner_params = self._predict_component_params('cleaner', embedding_features)
        predicted_config['operations'].append(('clean', cleaner_params))

        # Predict whether to include filter and its parameters
        if self._should_include_component('filter', embedding_features):
            filter_params = self._predict_component_params('filter', embedding_features)
            predicted_config['operations'].append(('filter', filter_params))
        else:
            predicted_config['operations'].append(None)

        # Predict whether to include selector and its parameters
        if self._should_include_component('selector', embedding_features):
            selector_params = self._predict_component_params('selector', embedding_features)
            predicted_config['operations'].append(('select', selector_params))
        else:
            predicted_config['operations'].append(None)

        # Predict whether to include augmenter and its parameters
        if self._should_include_component('augmenter', embedding_features):
            augmenter_params = self._predict_component_params('augmenter', embedding_features)
            predicted_config['operations'].append(('augment', augmenter_params))
        else:
            predicted_config['operations'].append(None)

        return predicted_config

    def _predict_component_params(self, component_type, embedding_features):
        """Predict parameters for a specific component using trained ML models"""
        params = {}
        param_space = self.parameter_space.get(f'{component_type}_params', {})

        for param_name, param_values in param_space.items():
            model_key = f'{component_type}_{param_name}'

            if model_key in self.parameter_models:
                try:
                    # Predict parameter value
                    predicted_idx = self.parameter_models[model_key].predict(embedding_features)[0]
                    predicted_idx = max(0, min(predicted_idx, len(param_values) - 1))
                    params[param_name] = param_values[predicted_idx]
                except Exception as e:
                    logger.warning(f"Parameter prediction failed for {model_key}: {e}")
                    params[param_name] = np.random.choice(param_values)
            else:
                # Fallback to random if model not available
                params[param_name] = np.random.choice(param_values)

        return params

    def _should_include_component(self, component_type, embedding_features):
        """Predict whether to include a component using trained ML models"""
        model_key = f'include_{component_type}'

        if model_key in self.operation_models:
            try:
                prediction = self.operation_models[model_key].predict_proba(embedding_features)[0]
                return prediction[1] > 0.5  # Include if probability > 0.5
            except Exception as e:
                logger.warning(f"Component inclusion prediction failed for {model_key}: {e}")

        # Fallback probabilities based on component type
        fallback_probs = {
            'filter': 0.7,
            'selector': 0.3,
            'augmenter': 0.4
        }
        return np.random.random() < fallback_probs.get(component_type, 0.5)

    def _generate_random_pipeline_config(self):
        """Generate random pipeline configuration as fallback"""
        cleaner_params = {
            'lowercase': True,
            'remove_extra_whitespace': True,
            'remove_punctuation': np.random.choice([True, False])
        }

        filter_params = None
        if np.random.random() < 0.7:
            filter_params = {
                'min_length_percentile': np.random.choice([5, 10, 15]),
                'max_length_percentile': np.random.choice([85, 90, 95]),
                'quality_threshold_percentile': np.random.choice([10, 20, 30])
            }

        selector_params = None
        if np.random.random() < 0.3:
            selector_params = {
                'strategy': np.random.choice(['easy', 'hard', 'balanced', 'random']),
                'difficulty_threshold_percentile': np.random.choice([30, 40, 50, 60, 70])
            }

        augmenter_params = None
        if np.random.random() < 0.4:
            augmenter_params = {
                'technique': np.random.choice(['mixed', 'synonym', 'insertion', 'deletion', 'swap']),
                'augmentation_ratio': np.random.choice([0.1, 0.15, 0.2, 0.25]),
                'minority_boost': np.random.choice([1.5, 2.0, 2.5])
            }

        return {
            'approach': 'random_fallback',
            'operations': [
                ('clean', cleaner_params),
                ('filter', filter_params) if filter_params else None,
                ('select', selector_params) if selector_params else None,
                ('augment', augmenter_params) if augmenter_params else None
            ]
        }

    def train_ml_models(self, training_data_list):
        """Train ML models on collected training data with proper sample alignment"""
        if not training_data_list:
            logger.warning("No training data available for ML models")
            return

        logger.info(f"Training ML models on {len(training_data_list)} examples...")

        # Prepare training data with proper alignment
        X_list = []
        y_dict = {}  # Store targets for different parameters

        # First pass: collect all data and initialize y_dict with proper length
        for i, (data_features, pipeline_config, fitness) in enumerate(training_data_list):
            X_list.append(data_features)

            # Initialize all possible targets with default values
            for component_type in ['cleaner', 'filter', 'selector', 'augmenter']:
                include_key = f'include_{component_type}'
                if include_key not in y_dict:
                    y_dict[include_key] = [0] * len(training_data_list)  # Default to not included

                # Initialize parameter targets
                param_space = self.parameter_space.get(f'{component_type}_params', {})
                for param_name in param_space.keys():
                    param_key = f'{component_type}_{param_name}'
                    if param_key not in y_dict:
                        y_dict[param_key] = [0] * len(training_data_list)  # Default to first option

        # Second pass: fill in actual values
        for i, (data_features, pipeline_config, fitness) in enumerate(training_data_list):
            # Track which components are included
            included_components = set()

            # Extract parameter targets from pipeline configuration
            for operation in pipeline_config.get('operations', []):
                if operation is None:
                    continue

                op_name, params = operation
                component_type = self._get_component_type(op_name)

                if component_type:
                    included_components.add(component_type)

                    # Record that this component was included
                    include_key = f'include_{component_type}'
                    y_dict[include_key][i] = 1

                    # Record parameter values
                    for param_name, param_value in params.items():
                        param_key = f'{component_type}_{param_name}'

                        # Convert parameter value to index
                        param_space = self.parameter_space.get(f'{component_type}_params', {})
                        param_values = param_space.get(param_name, [param_value])

                        try:
                            param_idx = param_values.index(param_value)
                        except ValueError:
                            param_idx = 0  # Default to first value if not found

                        y_dict[param_key][i] = param_idx

            # Ensure components not included remain 0 (already initialized)

        # Convert to arrays
        X = np.vstack(X_list)

        # Train models for each parameter
        for param_key, y_values in y_dict.items():
            if len(set(y_values)) > 1:  # Only train if there's variation
                try:
                    y = np.array(y_values)

                    # Ensure X and y have same length
                    if len(X) != len(y):
                        logger.warning(f"Skipping {param_key}: X has {len(X)} samples, y has {len(y)} samples")
                        continue

                    if param_key.startswith('include_'):
                        # Binary classification for component inclusion
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
                        model.fit(X, y)
                        self.operation_models[param_key] = model
                    else:
                        # Multi-class classification for parameter values
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=50, random_state=RANDOM_SEED)
                        model.fit(X, y)
                        self.parameter_models[param_key] = model

                except Exception as e:
                    logger.warning(f"Failed to train model for {param_key}: {e}")

        self.fitted = True
        logger.info(f"Trained {len(self.parameter_models)} parameter models and {len(self.operation_models)} operation models")

    def _get_component_type(self, operation_name):
        """Map operation name to component type"""
        mapping = {
            'clean': 'cleaner',
            'filter': 'filter',
            'select': 'selector',
            'augment': 'augmenter'
        }
        return mapping.get(operation_name)

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

            # Check if we have enough data for evaluation
            if original_size < 10:
                return 0.0

            # Apply operations in sequence
            processed_train = self._apply_pipeline_operations(pipeline_config, train_data)
            processed_size = len(processed_train)

            # Calculate data retention ratio
            retention_ratio = processed_size / original_size

            # Penalize aggressive data reduction
            if retention_ratio < 0.3:  # Less than 30% retained
                return 0.0  # Completely reject
            elif processed_size < 10:  # Too few samples
                return 0.0

            # Check if we have enough samples for train/val split
            min_samples_per_class = 2
            class_counts = processed_train[target_col].value_counts()
            if class_counts.min() < min_samples_per_class:
                # Not enough samples per class for stratified split
                # Use simple accuracy estimation
                accuracy = self._simple_accuracy_estimate(processed_train, target_col)
            else:
                # Split processed training data for evaluation
                try:
                    train_subset, val_subset = train_test_split(
                        processed_train, test_size=0.3, random_state=42,
                        stratify=processed_train[target_col]
                    )
                    # Get accuracy
                    accuracy = self._quick_evaluate(train_subset, val_subset, target_col)
                except Exception as e:
                    logger.debug(f"Stratified split failed: {e}, using simple split")
                    train_subset, val_subset = train_test_split(
                        processed_train, test_size=0.3, random_state=42
                    )
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

    def _simple_accuracy_estimate(self, data, target_col):
        """Simple accuracy estimate when we can't do proper train/val split"""
        try:
            # Use majority class baseline as simple estimate
            class_counts = data[target_col].value_counts()
            majority_class_ratio = class_counts.max() / len(data)

            # Add small bonus for class balance
            balance_bonus = 1.0 - (class_counts.max() - class_counts.min()) / len(data)

            return majority_class_ratio * 0.8 + balance_bonus * 0.2
        except:
            return 0.3  # Default fallback

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
        """Apply cleaning operations - skip for embedded data"""
        if 'text' in data.columns:
            cleaner = FixedTextCleaner(**params)
            return cleaner.apply(data)
        else:
            # For embedded data, cleaning doesn't apply - return unchanged
            logger.debug("Skipping text cleaning for embedded data")
            return data

    def _apply_filtering(self, data, params):
        """Apply filtering operations - adapted for embedded data"""
        if 'text' in data.columns:
            filter_op = FixedTextFilter(**params)
            filter_op.fit(data)  # Fit on current data
            return filter_op.apply(data)
        else:
            # For embedded data, use feature-based filtering
            return self._apply_embedding_filtering(data, params)

    def _apply_selection(self, data, params):
        """Apply selection operations - adapted for embedded data"""
        if 'text' in data.columns:
            selector = FixedDifficultySelector(**params)
            selector.fit(data)  # Fit on current data
            return selector.apply(data)
        else:
            # For embedded data, use feature-based selection
            return self._apply_embedding_selection(data, params)

    def _apply_augmentation(self, data, params):
        """Apply augmentation operations - adapted for embedded data"""
        if 'text' in data.columns:
            augmenter = FixedTextAugmenter(**params)
            return augmenter.apply(data)
        else:
            # For embedded data, use feature-based augmentation
            return self._apply_embedding_augmentation(data, params)

    def _apply_embedding_filtering(self, data, params):
        """Apply filtering operations to embedded data"""
        try:
            # Get feature columns
            feature_cols = [col for col in data.columns
                           if col not in ['label', 'answer', 'target', 'item_id', 'timestamp']
                           and data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

            if not feature_cols:
                return data

            original_size = len(data)

            # Use feature magnitude as proxy for text length
            feature_magnitudes = np.linalg.norm(data[feature_cols].values, axis=1)
            min_threshold = np.percentile(feature_magnitudes, params.get('min_length_percentile', 10))
            max_threshold = np.percentile(feature_magnitudes, params.get('max_length_percentile', 90))
            length_mask = (feature_magnitudes >= min_threshold) & (feature_magnitudes <= max_threshold)

            # Use feature variance as proxy for quality
            feature_variance = np.var(data[feature_cols].values, axis=1)
            quality_threshold = np.percentile(feature_variance, params.get('quality_threshold_percentile', 20))
            quality_mask = feature_variance >= quality_threshold

            # Combine filters
            combined_mask = length_mask & quality_mask
            result = data[combined_mask].reset_index(drop=True)

            logger.debug(f"Embedding filtering: {original_size} -> {len(result)} samples")
            return result

        except Exception as e:
            logger.warning(f"Embedding filtering failed: {e}")
            return data

    def _apply_embedding_selection(self, data, params):
        """Apply selection operations to embedded data"""
        try:
            # Get feature columns
            feature_cols = [col for col in data.columns
                           if col not in ['label', 'answer', 'target', 'item_id', 'timestamp']
                           and data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

            if not feature_cols:
                return data

            original_size = len(data)
            strategy = params.get('strategy', 'balanced')

            # Calculate difficulty based on feature complexity
            feature_magnitudes = np.linalg.norm(data[feature_cols].values, axis=1)
            feature_variance = np.var(data[feature_cols].values, axis=1)

            # Normalize for combined difficulty score
            norm_magnitude = (feature_magnitudes - feature_magnitudes.min()) / (feature_magnitudes.max() - feature_magnitudes.min() + 1e-8)
            norm_variance = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min() + 1e-8)

            difficulty = 0.6 * norm_magnitude + 0.4 * norm_variance

            # Apply strategy
            threshold = np.percentile(difficulty, params.get('difficulty_threshold_percentile', 50))

            if strategy == 'easy':
                mask = difficulty <= threshold
            elif strategy == 'hard':
                mask = difficulty >= threshold
            elif strategy == 'balanced':
                range_val = 0.2
                mask = np.abs(difficulty - threshold) <= range_val * (difficulty.max() - difficulty.min())
            else:  # random
                mask = np.random.random(len(data)) > 0.3

            result = data[mask].reset_index(drop=True)
            logger.debug(f"Embedding selection ({strategy}): {original_size} -> {len(result)} samples")
            return result

        except Exception as e:
            logger.warning(f"Embedding selection failed: {e}")
            return data

    def _apply_embedding_augmentation(self, data, params):
        """Apply augmentation operations to embedded data"""
        try:
            # Get feature and target columns
            feature_cols = [col for col in data.columns
                           if col not in ['label', 'answer', 'target', 'item_id', 'timestamp']
                           and data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

            target_cols = [col for col in data.columns if col in ['label', 'answer', 'target']]

            if not feature_cols or not target_cols:
                return data

            target_col = target_cols[0]
            augmentation_ratio = params.get('augmentation_ratio', 0.1)
            minority_boost = params.get('minority_boost', 2.0)

            # Identify class distribution
            class_counts = data[target_col].value_counts()
            median_count = class_counts.median()

            augmented_rows = []
            for label in class_counts.index:
                class_data = data[data[target_col] == label]
                class_count = len(class_data)

                # Calculate augmentation amount (boost minorities)
                if class_count < median_count:
                    aug_ratio = augmentation_ratio * minority_boost
                else:
                    aug_ratio = augmentation_ratio * 0.5

                n_augment = int(class_count * aug_ratio)

                if n_augment > 0:
                    # Sample data to augment
                    sample_data = class_data.sample(n=min(n_augment, len(class_data)),
                                                  replace=True, random_state=RANDOM_SEED)

                    # Add noise to embeddings
                    noise_scale = 0.02
                    technique = params.get('technique', 'mixed')

                    if technique == 'synonym':
                        noise_scale = 0.01
                    elif technique == 'insertion':
                        noise_scale = 0.03
                    elif technique == 'deletion':
                        noise_scale = 0.02
                    elif technique == 'mixed':
                        noise_scale = np.random.uniform(0.01, 0.04)

                    for _, row in sample_data.iterrows():
                        # Create augmented row
                        augmented_row = row.copy()

                        # Add appropriate noise to feature columns
                        for col in feature_cols:
                            if technique == 'deletion':
                                # Deletion: reduce some values
                                factor = np.random.uniform(0.7, 1.0)
                                augmented_row[col] = row[col] * factor
                            else:
                                # Other techniques: add noise
                                noise = np.random.normal(0, noise_scale * (abs(row[col]) + 0.01))
                                augmented_row[col] = row[col] + noise

                        augmented_rows.append(augmented_row)

            # Add augmented data
            if augmented_rows:
                augmented_df = pd.DataFrame(augmented_rows)
                result = pd.concat([data, augmented_df], ignore_index=True)
                logger.debug(f"Embedding augmentation: {len(data)} -> {len(result)} samples (+{len(augmented_rows)})")
                return result
            else:
                return data

        except Exception as e:
            logger.warning(f"Embedding augmentation failed: {e}")
            return data

    def _apply_individual_operation(self, data, op_name, params):
        """Apply individual operations from the operation pool - adapted for embedded data"""

        # More robust feature detection for different embedders
        numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
        
        # Get possible feature columns (more inclusive pattern matching)
        feature_cols = [col for col in numeric_cols if 
                       any(col.startswith(prefix) for prefix in 
                          ['text_', 'emb_', 'feature_', 'tfidf_']) or
                       col in ['hour', 'day', 'month', 'dayofweek']]
        
        # If no embedded features found, use all numeric columns except target-related ones
        if not feature_cols:
            excluded_cols = ['label', 'answer', 'target', 'item_id', 'timestamp']
            feature_cols = [col for col in numeric_cols if col not in excluded_cols]
        
        target_cols = [col for col in data.columns if col in ['label', 'answer', 'target']]
        id_cols = [col for col in data.columns if col in ['item_id', 'timestamp']]
        
        logger.debug(f"Operation {op_name} on {len(feature_cols)} feature columns")

        if op_name in ['lowercase', 'remove_punctuation', 'normalize_whitespace', 'remove_numbers']:
            # These text operations don't apply to embedded data - return unchanged
            logger.debug(f"Text operation {op_name} skipped for embedded data")
            return data

        elif op_name == 'length_filter':
            # Use feature magnitude as proxy for text length
            if feature_cols:
                try:
                    feature_magnitudes = np.linalg.norm(data[feature_cols].values, axis=1)
                    min_threshold = np.percentile(feature_magnitudes, params.get('min_percentile', 10))
                    max_threshold = np.percentile(feature_magnitudes, params.get('max_percentile', 90))
                    mask = (feature_magnitudes >= min_threshold) & (feature_magnitudes <= max_threshold)
                    data = data[mask].reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Length filter failed: {e}")

        elif op_name == 'quality_filter':
            # Use feature variance or distance from mean as proxy for quality
            if feature_cols:
                try:
                    # Calculate feature variance for each sample
                    feature_variance = np.var(data[feature_cols].values, axis=1)
                    # Calculate distance from feature mean
                    feature_mean = data[feature_cols].mean().values
                    distances = np.sqrt(np.sum((data[feature_cols].values - feature_mean)**2, axis=1))
                    
                    # Combine metrics for quality score (normalized)
                    quality_score = (
                        0.5 * (feature_variance / feature_variance.max()) + 
                        0.5 * (1 - distances / distances.max())
                    )
                    threshold = np.percentile(quality_score, params.get('threshold_percentile', 20))
                    data = data[quality_score >= threshold].reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Quality filter failed: {e}")

        elif 'difficulty_select' in op_name:
            # Calculate difficulty based on feature complexity
            if feature_cols:
                try:
                    # Calculate feature norms and variances
                    feature_magnitudes = np.linalg.norm(data[feature_cols].values, axis=1)
                    feature_variance = np.var(data[feature_cols].values, axis=1)
                    
                    # Normalize for combined difficulty score
                    norm_magnitude = (feature_magnitudes - feature_magnitudes.min()) / (feature_magnitudes.max() - feature_magnitudes.min() + 1e-8)
                    norm_variance = (feature_variance - feature_variance.min()) / (feature_variance.max() - feature_variance.min() + 1e-8)
                    
                    difficulty = 0.6 * norm_magnitude + 0.4 * norm_variance

                    if 'easy' in op_name:
                        threshold = np.percentile(difficulty, params.get('threshold_percentile', 30))
                        mask = difficulty <= threshold
                    elif 'hard' in op_name:
                        threshold = np.percentile(difficulty, params.get('threshold_percentile', 70))
                        mask = difficulty >= threshold
                    elif 'balanced' in op_name:
                        threshold = np.percentile(difficulty, params.get('threshold_percentile', 50))
                        range_val = params.get('range', 0.2)
                        mask = np.abs(difficulty - threshold) <= range_val * (difficulty.max() - difficulty.min())
                    else:
                        mask = np.ones(len(data), dtype=bool)  # Default to keeping all

                    if 'ratio' in params and sum(mask) > 0:
                        # Apply ratio-based selection
                        n_select = int(len(data) * params['ratio'])
                        selected_indices = data[mask].sample(n=min(n_select, sum(mask)), 
                                                            random_state=RANDOM_SEED).index
                        data = data.loc[selected_indices].reset_index(drop=True)
                    else:
                        data = data[mask].reset_index(drop=True)
                except Exception as e:
                    logger.warning(f"Difficulty selection failed: {e}")

        elif op_name == 'random_select':
            try:
                n_select = int(len(data) * params.get('ratio', 0.8))
                if n_select > 0:
                    data = data.sample(n=min(n_select, len(data)), 
                                      random_state=RANDOM_SEED).reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Random selection failed: {e}")

        elif 'augment' in op_name:
            # For embedded data, apply perturbation-based augmentation
            if feature_cols and len(target_cols) > 0:
                try:
                    target_col = target_cols[0]
                    augmentation_ratio = params.get('ratio', 0.1)
                    minority_boost = params.get('minority_boost', 2.0)

                    # Identify class distribution
                    class_counts = data[target_col].value_counts()
                    median_count = class_counts.median()

                    augmented_rows = []
                    for label in class_counts.index:
                        class_data = data[data[target_col] == label]
                        class_count = len(class_data)

                        # Calculate augmentation amount (boost minorities)
                        if class_count < median_count:
                            aug_ratio = augmentation_ratio * minority_boost
                        else:
                            aug_ratio = augmentation_ratio * 0.5

                        n_augment = int(class_count * aug_ratio)

                        if n_augment > 0:
                            # Sample data to augment
                            sample_data = class_data.sample(n=min(n_augment, len(class_data)),
                                                          replace=True, random_state=RANDOM_SEED)

                            # Different augmentation strategies for different techniques
                            noise_scale = 0.02  # Base noise scale
                            
                            if 'synonym' in op_name:
                                # Synonym = small noise
                                noise_scale = 0.01
                            elif 'insertion' in op_name:
                                # Insertion = additive noise
                                noise_scale = 0.03
                            elif 'deletion' in op_name:
                                # Deletion = multiplicative noise (reducing values)
                                noise_scale = 0.02
                            elif 'mixed' in op_name:
                                # Mixed = variable noise
                                noise_scale = np.random.uniform(0.01, 0.04)

                            for _, row in sample_data.iterrows():
                                # Create augmented row
                                augmented_row = row.copy()
                                
                                # Add appropriate noise to feature columns
                                for col in feature_cols:
                                    if 'deletion' in op_name:
                                        # Deletion: reduce some values
                                        factor = np.random.uniform(0.7, 1.0)
                                        augmented_row[col] = row[col] * factor
                                    else:
                                        # Other techniques: add noise
                                        noise = np.random.normal(0, noise_scale * (abs(row[col]) + 0.01))
                                        augmented_row[col] = row[col] + noise

                            augmented_rows.append(augmented_row)

                    # Add augmented data
                    if augmented_rows:
                        augmented_df = pd.DataFrame(augmented_rows)
                        data = pd.concat([data, augmented_df], ignore_index=True)
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")

        return data
    
    def _quick_evaluate(self, train_data, val_data, target_col):
        """Quick evaluation using RandomForest on embedded data"""
        try:
            # Check if we have embedded data or text data
            if 'text' in train_data.columns:
                # Original text data - use text features
                def extract_features(data):
                    features = pd.DataFrame({
                        'text_length': data['text'].str.len(),
                        'word_count': data['text'].str.split().str.len(),
                        'avg_word_length': data['text'].str.len() / data['text'].str.split().str.len()
                    })
                    return features.fillna(0)

                X_train = extract_features(train_data)
                X_val = extract_features(val_data)
            else:
                # Embedded data - use embedding features directly
                feature_cols = [col for col in train_data.columns
                               if col not in [target_col, 'item_id', 'timestamp']
                               and train_data[col].dtype in ['float64', 'float32', 'int64', 'int32']]

                if not feature_cols:
                    logger.warning("No suitable feature columns found for evaluation")
                    return 0.0

                X_train = train_data[feature_cols].fillna(0)
                X_val = val_data[feature_cols].fillna(0)

            y_train = train_data[target_col]
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

class MLEnhancedMetaLearner:
    """Meta-learner that combines evolutionary search with ML-based parameter prediction"""

    def __init__(self, population_size=8, generations=3, ml_candidates=5):
        self.population_size = population_size
        self.generations = generations
        self.ml_candidates = ml_candidates
        self.ml_predictor = MLBasedPipelinePredictor(n_candidates=ml_candidates)
        self.evolutionary_learner = HybridExploratoryMetaLearner(population_size=population_size//2, generations=generations)
        self.best_pipeline = None
        self.best_fitness = 0.0
        self.training_history = []
        self.dataset_characteristics = None

    def set_dataset_characteristics(self, embedded_data, target_col):
        """Set dataset characteristics for ML prediction"""
        self.dataset_characteristics = self.ml_predictor.extract_embedding_features(embedded_data, target_col)
        logger.info("Dataset characteristics set for ML prediction")

    def run_meta_learning(self, train_data, target_col):
        """Run hybrid meta-learning combining ML prediction with evolutionary search"""
        logger.info("Starting ML-enhanced meta-learning...")
        start_time = time.time()

        # Phase 1: Collect initial training data using evolutionary approach
        logger.info("Phase 1: Collecting training data with evolutionary search...")

        # Run a short evolutionary search to collect training examples
        initial_population = [self.evolutionary_learner.generate_hybrid_pipeline()
                            for _ in range(self.population_size)]

        training_data = []
        for pipeline_config in initial_population:
            fitness = self.evolutionary_learner.evaluate_pipeline(pipeline_config, train_data, target_col)

            # Use dataset characteristics if available, otherwise extract from current data
            if self.dataset_characteristics is not None:
                embedding_features = self.dataset_characteristics
            else:
                # For raw text data, we need to create temporary embeddings to extract characteristics
                temp_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", max_features=50)
                temp_embedder.fit(train_data, target_col)
                temp_embedded = temp_embedder.transform(train_data.sample(n=min(200, len(train_data))))
                embedding_features = self.ml_predictor.extract_embedding_features(temp_embedded, target_col)
            training_data.append((embedding_features.values[0], pipeline_config, fitness))

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_pipeline = pipeline_config

        self.training_history.extend(training_data)

        # Phase 2: Train ML models on collected data
        logger.info("Phase 2: Training ML models on collected examples...")
        self.ml_predictor.train_ml_models(training_data)

        # Phase 3: Generate ML-predicted candidates
        logger.info("Phase 3: Generating ML-predicted pipeline candidates...")
        ml_candidates = []
        for _ in range(self.ml_candidates):
            if self.dataset_characteristics is not None:
                # Use pre-computed dataset characteristics
                ml_pipeline = self.ml_predictor.predict_pipeline_parameters_from_features(self.dataset_characteristics, target_col)
            else:
                # Fallback to extracting features from current data
                ml_pipeline = self.ml_predictor.predict_pipeline_parameters(train_data, target_col)
            ml_candidates.append(ml_pipeline)

        # Evaluate ML candidates
        for i, pipeline_config in enumerate(ml_candidates):
            fitness = self.evolutionary_learner.evaluate_pipeline(pipeline_config, train_data, target_col)
            logger.info(f"ML candidate {i+1}: fitness={fitness:.4f}")

            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_pipeline = pipeline_config
                logger.info(f"🤖 New best from ML prediction: {fitness:.4f}")

            # Add to training data for future iterations
            if self.dataset_characteristics is not None:
                embedding_features = self.dataset_characteristics
            else:
                # For raw text data, create temporary embeddings
                temp_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", max_features=50)
                temp_embedder.fit(train_data, target_col)
                temp_embedded = temp_embedder.transform(train_data.sample(n=min(200, len(train_data))))
                embedding_features = self.ml_predictor.extract_embedding_features(temp_embedded, target_col)
            self.training_history.append((embedding_features.values[0], pipeline_config, fitness))

        # Phase 4: Hybrid evolution with ML guidance
        logger.info("Phase 4: Hybrid evolution with ML guidance...")

        # Combine best evolutionary and ML candidates
        combined_population = []

        # Add best ML candidates
        ml_fitness_pairs = [(ml_candidates[i], self.evolutionary_learner.evaluate_pipeline(ml_candidates[i], train_data, target_col))
                           for i in range(len(ml_candidates))]
        ml_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        combined_population.extend([pair[0] for pair in ml_fitness_pairs[:self.population_size//2]])

        # Add best evolutionary candidates
        evolutionary_fitness_pairs = [(initial_population[i], self.evolutionary_learner.evaluate_pipeline(initial_population[i], train_data, target_col))
                                    for i in range(len(initial_population))]
        evolutionary_fitness_pairs.sort(key=lambda x: x[1], reverse=True)
        combined_population.extend([pair[0] for pair in evolutionary_fitness_pairs[:self.population_size//2]])

        # Run evolution on combined population
        for generation in range(self.generations):
            gen_start = time.time()
            logger.info(f"Generation {generation + 1}/{self.generations}")

            # Evaluate population
            fitness_scores = []
            for i, pipeline_config in enumerate(combined_population):
                fitness = self.evolutionary_learner.evaluate_pipeline(pipeline_config, train_data, target_col)
                fitness_scores.append(fitness)

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_pipeline = pipeline_config
                    logger.info(f"🏆 New best fitness: {fitness:.4f}")

                # Update training data
                if self.dataset_characteristics is not None:
                    embedding_features = self.dataset_characteristics
                else:
                    # For raw text data, create temporary embeddings
                    temp_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2", max_features=50)
                    temp_embedder.fit(train_data, target_col)
                    temp_embedded = temp_embedder.transform(train_data.sample(n=min(200, len(train_data))))
                    embedding_features = self.ml_predictor.extract_embedding_features(temp_embedded, target_col)
                self.training_history.append((embedding_features.values[0], pipeline_config, fitness))

            gen_time = time.time() - gen_start
            logger.info(f"Generation {generation + 1} completed in {gen_time:.2f}s - "
                       f"Best: {max(fitness_scores):.4f}, Avg: {np.mean(fitness_scores):.4f}")

            # Evolution step
            if generation < self.generations - 1:
                # Keep best half
                sorted_indices = np.argsort(fitness_scores)[::-1]
                best_half = [combined_population[i] for i in sorted_indices[:self.population_size//2]]

                # Generate new population
                new_population = best_half.copy()

                # Add ML-predicted candidates (retrain models with updated data)
                if len(self.training_history) > 10:  # Only retrain if we have enough data
                    self.ml_predictor.train_ml_models(self.training_history[-50:])  # Use recent examples

                    for _ in range(self.population_size//4):
                        if self.dataset_characteristics is not None:
                            ml_pipeline = self.ml_predictor.predict_pipeline_parameters_from_features(self.dataset_characteristics, target_col)
                        else:
                            ml_pipeline = self.ml_predictor.predict_pipeline_parameters(train_data, target_col)
                        new_population.append(ml_pipeline)

                # Add mutated candidates
                while len(new_population) < self.population_size:
                    parent = np.random.choice(best_half)
                    child = self.evolutionary_learner._mutate_pipeline(parent)
                    new_population.append(child)

                combined_population = new_population

        total_time = time.time() - start_time
        logger.info(f"ML-enhanced meta-learning completed in {format_time(total_time)}")
        logger.info(f"Best fitness found: {self.best_fitness:.4f}")
        logger.info(f"Training examples collected: {len(self.training_history)}")

        return self.best_pipeline

    def _apply_pipeline_operations(self, pipeline_config, data):
        """Apply pipeline operations to data - delegate to evolutionary learner"""
        return self.evolutionary_learner._apply_pipeline_operations(pipeline_config, data)

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

def load_casehold_dataset():
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

# ============================================================================
# NEW MODULAR COMPONENTS FOR DATASET-SPECIFIC EMBEDDINGS AND AUTOGLUON TRAINING
# ============================================================================

from abc import ABC, abstractmethod

class DatasetEmbedder(ABC):
    """Abstract base class for dataset-specific embedders"""

    @abstractmethod
    def fit(self, data, target_col):
        """Fit the embedder on training data"""
        pass

    @abstractmethod
    def transform(self, data):
        """Transform data to embedding space"""
        pass

    @abstractmethod
    def get_feature_names(self):
        """Get feature names for the embedding"""
        pass

class TextEmbedder(DatasetEmbedder):
    """Embedder for text datasets using sentence transformers"""

    def __init__(self, model_name="all-MiniLM-L6-v2", max_features=512):
        self.model_name = model_name
        self.max_features = max_features
        self.model = None
        self.fitted = False

    def fit(self, data, target_col):
        """Fit sentence transformer model"""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence transformer: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.fitted = True
            logger.info("Text embedder fitted successfully")
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to TF-IDF")
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.model = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                ngram_range=(1, 2),
                random_state=RANDOM_SEED
            )
            self.model.fit(data['text'].fillna(''))
            self.fitted = True

    def transform(self, data):
        """Transform text to embeddings"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        text_data = data['text'].fillna('').tolist()

        if hasattr(self.model, 'encode'):
            # Sentence transformer
            embeddings = self.model.encode(text_data, show_progress_bar=False)
        else:
            # TF-IDF fallback
            embeddings = self.model.transform(text_data).toarray()

        # Create DataFrame with embeddings
        feature_names = self.get_feature_names()
        embedding_df = pd.DataFrame(embeddings, columns=feature_names)

        # Add target column if present
        for target_col in ['label', 'answer', 'target']:
            if target_col in data.columns:
                embedding_df[target_col] = data[target_col].values
                break

        return embedding_df

    def get_feature_names(self):
        """Get feature names for the embedding"""
        if hasattr(self.model, 'encode'):
            # Sentence transformer - dynamically determine dimensions
            dummy_embedding = self.model.encode("test")
            return [f'text_emb_{i}' for i in range(len(dummy_embedding))]
        else:
            # TF-IDF
            try:
                return [f'text_tfidf_{i}' for i in range(len(self.model.get_feature_names_out()))]
            except:
                # Fallback if get_feature_names_out not available
                return [f'text_tfidf_{i}' for i in range(self.max_features)]

class MultimodalEmbedder(DatasetEmbedder):
    """Embedder for multimodal datasets (text + images + metadata)"""

    def __init__(self):
        self.text_embedder = None
        self.label_encoders = {}
        self.fitted = False

    def fit(self, data, target_col):
        """Fit multimodal embedder"""
        logger.info("Fitting multimodal embedder...")

        # Fit text embedder
        self.text_embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
        self.text_embedder.fit(data, target_col)

        # Fit label encoders for categorical columns
        categorical_cols = ['task', 'grade', 'subject', 'topic', 'category']
        for col in categorical_cols:
            if col in data.columns:
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(data[col].fillna('unknown'))

        self.fitted = True
        logger.info("Multimodal embedder fitted successfully")

    def transform(self, data):
        """Transform multimodal data to unified embedding space"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        features = []
        feature_names = []

        # 1. Text embeddings
        text_embeddings = self.text_embedder.transform(data)
        text_cols = [col for col in text_embeddings.columns if col.startswith('text_')]
        features.append(text_embeddings[text_cols].values)
        feature_names.extend(text_cols)

        # 2. Image features (simple placeholder - could use vision models)
        if 'image' in data.columns:
            # Simple image availability feature
            has_image = data['image'].notna().astype(int).values.reshape(-1, 1)
            features.append(has_image)
            feature_names.append('has_image')

        # 3. Categorical features
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                try:
                    encoded_col = []
                    for val in data[col].fillna('unknown'):
                        try:
                            encoded_col.append(encoder.transform([str(val)])[0])
                        except ValueError:
                            encoded_col.append(0)  # Unknown category

                    features.append(np.array(encoded_col).reshape(-1, 1))
                    feature_names.append(f'{col}_encoded')
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {e}")

        # Combine features
        combined_features = np.hstack(features)
        result_df = pd.DataFrame(combined_features, columns=feature_names)

        # Add target column
        for target_col in ['answer', 'label', 'target']:
            if target_col in data.columns:
                result_df[target_col] = data[target_col].values
                break

        return result_df

    def get_feature_names(self):
        """Get feature names"""
        names = self.text_embedder.get_feature_names()
        names.append('has_image')
        names.extend([f'{col}_encoded' for col in self.label_encoders.keys()])
        return names

class TimeSeriesEmbedder(DatasetEmbedder):
    """Embedder for time series datasets"""

    def __init__(self):
        self.fitted = False
        self.feature_names = []

    def fit(self, data, target_col):
        """Fit time series embedder"""
        logger.info("Fitting time series embedder...")

        # Define feature names
        self.feature_names = [
            'hour', 'day', 'month', 'dayofweek', 'quarter',
            'is_weekend', 'is_month_start', 'is_month_end',
            'target_lag_1', 'target_lag_7', 'target_lag_30',
            'target_rolling_mean_7', 'target_rolling_std_7'
        ]

        self.fitted = True
        logger.info("Time series embedder fitted successfully")

    def transform(self, data):
        """Transform time series data to feature space"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        # Convert timestamp to datetime if needed
        if 'timestamp' in data.columns:
            timestamps = pd.to_datetime(data['timestamp'])
        else:
            # Create dummy timestamps
            timestamps = pd.date_range(start='2020-01-01', periods=len(data), freq='D')

        # Extract time features
        features = {
            'hour': timestamps.hour,
            'day': timestamps.day,
            'month': timestamps.month,
            'dayofweek': timestamps.dayofweek,
            'quarter': timestamps.quarter,
            'is_weekend': (timestamps.dayofweek >= 5).astype(int),
            'is_month_start': timestamps.is_month_start.astype(int),
            'is_month_end': timestamps.is_month_end.astype(int)
        }

        # Add lag features if target column exists
        if 'target' in data.columns:
            if 'item_id' in data.columns:
                # Group-wise lag features
                features['target_lag_1'] = data.groupby('item_id')['target'].shift(1).fillna(0)
                features['target_lag_7'] = data.groupby('item_id')['target'].shift(7).fillna(0)
                features['target_lag_30'] = data.groupby('item_id')['target'].shift(30).fillna(0)
                features['target_rolling_mean_7'] = data.groupby('item_id')['target'].rolling(7).mean().fillna(0).values
                features['target_rolling_std_7'] = data.groupby('item_id')['target'].rolling(7).std().fillna(0).values
            else:
                # Simple lag features
                features['target_lag_1'] = data['target'].shift(1).fillna(0)
                features['target_lag_7'] = data['target'].shift(7).fillna(0)
                features['target_lag_30'] = data['target'].shift(30).fillna(0)
                features['target_rolling_mean_7'] = data['target'].rolling(7).mean().fillna(0)
                features['target_rolling_std_7'] = data['target'].rolling(7).std().fillna(0)
        else:
            # Fill with zeros if no target
            for feat in ['target_lag_1', 'target_lag_7', 'target_lag_30', 'target_rolling_mean_7', 'target_rolling_std_7']:
                features[feat] = 0

        # Create result DataFrame
        result_df = pd.DataFrame(features)

        # Add original columns
        for col in ['item_id', 'timestamp', 'target']:
            if col in data.columns:
                result_df[col] = data[col].values

        return result_df

    def get_feature_names(self):
        """Get feature names"""
        return self.feature_names

class TabularEmbedder(DatasetEmbedder):
    """Embedder for general tabular datasets"""

    def __init__(self):
        self.label_encoders = {}
        self.fitted = False

    def fit(self, data, target_col):
        """Fit tabular embedder"""
        logger.info("Fitting tabular embedder...")

        # Fit label encoders for categorical columns
        for col in data.columns:
            if col != target_col and (data[col].dtype == 'object' or data[col].dtype.name == 'category'):
                self.label_encoders[col] = LabelEncoder()
                self.label_encoders[col].fit(data[col].fillna('unknown'))

        self.fitted = True
        logger.info("Tabular embedder fitted successfully")

   

   

    def transform(self, data):
        """Transform tabular data"""
        if not self.fitted:
            raise ValueError("Must call fit() first")

        result_df = data.copy()

        # Encode categorical columns
        for col, encoder in self.label_encoders.items():
            if col in result_df.columns:
                try:
                    encoded_col = []
                    for val in result_df[col].fillna('unknown'):
                        try:
                            encoded_col.append(encoder.transform([str(val)])[0])
                        except ValueError:
                            encoded_col.append(0)  # Unknown category
                    result_df[col] = encoded_col
                except Exception as e:
                    logger.warning(f"Error encoding {col}: {e}")

        # Fill missing values
        result_df = result_df.fillna(0)

        return result_df

    def get_feature_names(self):
        """Get feature names"""
        return list(self.label_encoders.keys())

def get_embedder_for_dataset(dataset_name):
    """Factory function to get appropriate embedder for dataset"""
    if dataset_name in ['anli', 'casehold']:
        return TextEmbedder(model_name="all-MiniLM-L6-v2")
    elif dataset_name == 'scienceqa':
        return MultimodalEmbedder()
    elif dataset_name == 'temperature_rain':
        return TimeSeriesEmbedder()
    else:
        return TabularEmbedder()

class AutoGluonTrainer:
    """Handles AutoGluon training for different dataset types"""

    def __init__(self, output_dir="./models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def train_model(self, train_df, val_df, target_col, dataset_name, problem_type=None):
        """Train appropriate AutoGluon model based on dataset type"""
        logger.info(f"Training AutoGluon model for {dataset_name}...")

        start_time = time.time()

        # Create output directory for this dataset
        model_dir = os.path.join(self.output_dir, f"{dataset_name}_model")
        os.makedirs(model_dir, exist_ok=True)

        # Determine predictor type and problem type based on dataset
        if dataset_name == 'temperature_rain':
            return self.train_timeseries_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        elif dataset_name == 'scienceqa':
            return self.train_multimodal_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        elif dataset_name in ['casehold', 'anli']:
            return self.train_text_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        else:
            return self.train_tabular_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)

    def train_timeseries_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train TimeSeriesPredictor"""
        from autogluon.timeseries import TimeSeriesPredictor

        logger.info(f"Using TimeSeriesPredictor for {dataset_name}")

        # TimeSeriesPredictor expects specific format
        predictor = TimeSeriesPredictor(
            path=model_dir,
            target=target_col,
            prediction_length=24,  # Predict 24 time steps ahead
            eval_metric='MASE',
            freq='D'  # Daily frequency for temperature_rain dataset
        )

        # Use AutoGluon's default settings
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',
            time_limit=3000,
            verbosity=2
        )

        training_time = time.time() - start_time
        logger.info(f"TimeSeriesPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
        return predictor, training_time

    def train_multimodal_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train MultiModalPredictor with fallback to TabularPredictor"""
        from autogluon.multimodal import MultiModalPredictor
        from autogluon.tabular import TabularPredictor

        try:
            # Create MultiModalPredictor with AutoGluon's default settings
            predictor = MultiModalPredictor(
                label=target_col,
                path=model_dir
            )

            # Combine train and validation data for MultiModalPredictor
            # MultiModalPredictor handles train/val split internally
            combined_train_df = pd.concat([train_df, val_df], ignore_index=True)

            logger.info(f"Training MultiModalPredictor with {len(combined_train_df)} samples")
            logger.info(f"Columns: {combined_train_df.columns.tolist()}")
            logger.info(f"Images available: {combined_train_df['image'].notna().sum()}/{len(combined_train_df)}")

            # Train MultiModalPredictor with AutoGluon's default settings
            predictor.fit(
                train_data=combined_train_df,
                time_limit=3000,
                presets='medium_quality'
            )

            training_time = time.time() - start_time
            logger.info(f"MultiModalPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
            return predictor, training_time

        except Exception as e:
            logger.error(f"MultiModalPredictor failed: {e}")
            import traceback
            traceback.print_exc()

            logger.warning("Falling back to TabularPredictor without images")

            # Prepare data for TabularPredictor (remove image column)
            train_df_tabular = train_df.copy()
            val_df_tabular = val_df.copy()

            # Remove image column for TabularPredictor as it can't handle PIL Images
            if 'image' in train_df_tabular.columns:
                logger.info("Removing image column for TabularPredictor fallback")
                train_df_tabular = train_df_tabular.drop(columns=['image'])
                val_df_tabular = val_df_tabular.drop(columns=['image'])

            # Fallback to TabularPredictor with AutoGluon defaults
            predictor = TabularPredictor(
                label=target_col,
                path=model_dir,
                problem_type='multiclass',
                eval_metric='accuracy'
            )

            # Use AutoGluon's default settings with increased memory allowance
            predictor.fit(
                train_data=train_df_tabular,
                tuning_data=val_df_tabular,
                presets='medium_quality',
                time_limit=3000,
                verbosity=2,
                ag_args_fit={'ag.max_memory_usage_ratio': 5}
            )

            training_time = time.time() - start_time
            logger.info(f"Fallback TabularPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
            return predictor, training_time

    def train_text_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train TabularPredictor for text data using AutoGluon defaults"""
        from autogluon.tabular import TabularPredictor

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
            presets='medium_quality',
            time_limit=3000,
            verbosity=2,
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
        )

        training_time = time.time() - start_time
        logger.info(f"Text model training completed for {dataset_name} in {training_time:.2f} seconds")
        return predictor, training_time

    def train_tabular_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train standard TabularPredictor using AutoGluon defaults"""
        from autogluon.tabular import TabularPredictor

        logger.info(f"Using TabularPredictor for {dataset_name}")

        # Auto-detect problem type
        unique_targets = train_df[target_col].nunique()
        target_dtype = train_df[target_col].dtype

        if target_dtype in ['object', 'category'] or str(target_dtype).startswith('string'):
            problem_type = 'multiclass' if unique_targets > 2 else 'binary'
        elif unique_targets <= 10 and target_dtype in ['int64', 'int32']:
            problem_type = 'multiclass' if unique_targets > 2 else 'binary'
        else:
            problem_type = 'regression'

        eval_metric = 'accuracy' if problem_type in ['binary', 'multiclass'] else 'root_mean_squared_error'

        predictor = TabularPredictor(
            label=target_col,
            path=model_dir,
            problem_type=problem_type,
            eval_metric=eval_metric
        )

        # Use AutoGluon's default settings with increased memory allowance
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',
            time_limit=3000,
            verbosity=2,
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
        )

        training_time = time.time() - start_time
        logger.info(f"Tabular model training completed for {dataset_name} in {training_time:.2f} seconds")
        return predictor, training_time

class MetaLearningFramework:
    """Complete meta-learning framework with dataset-specific embeddings and AutoGluon integration"""

    def __init__(self, dataset_name, output_dir="./meta_learning_results", use_ml_enhanced=True):
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.embedder = get_embedder_for_dataset(dataset_name)

        # Choose meta-learning approach
        if use_ml_enhanced:
            self.meta_learner = MLEnhancedMetaLearner(population_size=8, generations=3, ml_candidates=5)
            logger.info("Using ML-enhanced meta-learner")
        else:
            self.meta_learner = HybridExploratoryMetaLearner(population_size=8, generations=5)
            logger.info("Using traditional evolutionary meta-learner")

        self.autogluon_trainer = AutoGluonTrainer(output_dir)

        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Initialized meta-learning framework for {dataset_name}")
        logger.info(f"Using embedder: {type(self.embedder).__name__}")

    def run_complete_experiment(self, train_df, val_df, test_df, target_col):
        """Run complete meta-learning + AutoGluon experiment with embedding-guided pipeline prediction"""
        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING-GUIDED META-LEARNING EXPERIMENT: {self.dataset_name.upper()}")
        logger.info(f"{'='*80}")

        total_start_time = time.time()

        # Phase 1: Create embeddings for dataset characterization
        logger.info("\n🔄 Phase 1: Creating embeddings for dataset analysis...")
        embedding_start = time.time()

        # Create embeddings to understand dataset characteristics
        self.embedder.fit(train_df, target_col)

        # Use subset for analysis to save time
        analysis_subset = train_df.sample(n=min(1000, len(train_df)), random_state=RANDOM_SEED)
        embedded_analysis = self.embedder.transform(analysis_subset)

        embedding_time = time.time() - embedding_start
        logger.info(f"Initial embedding completed in {embedding_time:.2f}s")
        logger.info(f"Dataset characteristics extracted from {len(embedded_analysis)} samples")

        # Phase 2: ML-guided pipeline discovery using embedding characteristics
        logger.info("\n🧠 Phase 2: ML-guided pipeline discovery...")
        meta_start = time.time()

        # Pass embedding characteristics to meta-learner for ML prediction
        if hasattr(self.meta_learner, 'set_dataset_characteristics'):
            self.meta_learner.set_dataset_characteristics(embedded_analysis, target_col)

        # Use subset of RAW data for meta-learning
        subset_size = min(2000, len(train_df))
        train_subset = train_df.sample(n=subset_size, random_state=RANDOM_SEED)

        # Meta-learning on RAW data, guided by embedding characteristics
        best_pipeline_config = self.meta_learner.run_meta_learning(train_subset, target_col)

        meta_time = time.time() - meta_start
        logger.info(f"Meta-learning completed in {meta_time:.2f}s")
        logger.info(f"Best fitness: {self.meta_learner.best_fitness:.4f}")

        # Phase 3: Apply discovered pipeline to RAW data
        logger.info("\n⚙️ Phase 3: Applying discovered pipeline to raw data...")
        pipeline_start = time.time()

        # Apply pipeline operations to RAW data
        processed_train = self.meta_learner._apply_pipeline_operations(best_pipeline_config, train_df)
        processed_val = val_df.copy()  # Keep validation untouched
        processed_test = test_df.copy()  # Keep test untouched

        pipeline_time = time.time() - pipeline_start
        retention_ratio = len(processed_train) / len(train_df)

        logger.info(f"Pipeline applied in {pipeline_time:.2f}s")
        logger.info(f"Data retention: {retention_ratio:.2f} ({len(processed_train)}/{len(train_df)})")

        # Phase 4: Create final embeddings from processed data
        logger.info("\n🔄 Phase 4: Creating final embeddings from processed data...")
        final_embedding_start = time.time()

        # Re-fit embedder on processed data and create final embeddings
        self.embedder.fit(processed_train, target_col)
        final_embedded_train = self.embedder.transform(processed_train)
        final_embedded_val = self.embedder.transform(processed_val)
        final_embedded_test = self.embedder.transform(processed_test)

        final_embedding_time = time.time() - final_embedding_start
        logger.info(f"Final embedding completed in {final_embedding_time:.2f}s")
        logger.info(f"Final features: {final_embedded_train.shape[1]}")

        # Phase 5: AutoGluon training
        logger.info("\n🚀 Phase 5: AutoGluon training...")

        predictor, training_time = self.autogluon_trainer.train_model(
            final_embedded_train, final_embedded_val, target_col, self.dataset_name
        )

        # Phase 6: Evaluation
        logger.info("\n📊 Phase 6: Evaluation...")
        eval_start = time.time()

        performance = predictor.evaluate(final_embedded_test)

        eval_time = time.time() - eval_start
        total_time = time.time() - total_start_time

        # Compile results
        results = {
            'dataset': self.dataset_name,
            'timestamp': datetime.now().isoformat(),
            'embedder_type': type(self.embedder).__name__,
            'meta_learning_fitness': float(self.meta_learner.best_fitness),
            'best_pipeline_config': best_pipeline_config,
            'data_sizes': {
                'original': {'train': len(train_df), 'val': len(val_df), 'test': len(test_df)},
                'processed': {'train': len(processed_train), 'val': len(processed_val), 'test': len(processed_test)},
                'final_embedded': {'train': len(final_embedded_train), 'val': len(final_embedded_val), 'test': len(final_embedded_test)}
            },
            'retention_ratio': float(retention_ratio),
            'autogluon_performance': str(performance),
            'timing': {
                'initial_embedding_time': float(embedding_time),
                'meta_learning_time': float(meta_time),
                'pipeline_time': float(pipeline_time),
                'final_embedding_time': float(final_embedding_time),
                'training_time': float(training_time),
                'evaluation_time': float(eval_time),
                'total_time': float(total_time)
            },
            'random_seed': RANDOM_SEED
        }

        # Save results
        results_file = os.path.join(self.output_dir, f'{self.dataset_name}_complete_results.json')
        import json
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        logger.info(f"{'='*80}")
        logger.info(f"EXPERIMENT SUMMARY: {self.dataset_name.upper()}")
        logger.info(f"{'='*60}")
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Timing: Initial-Embedding={format_time(embedding_time)}, Meta-learning={format_time(meta_time)}, " +
                   f"Pipeline={format_time(pipeline_time)}, Final-Embedding={format_time(final_embedding_time)}, " +
                   f"Training={format_time(training_time)}, Evaluation={format_time(eval_time)}, Total={format_time(total_time)}")
        logger.info(f"AutoGluon performance: {performance}")
        logger.info(f"Data retention: {len(train_df)} -> {len(processed_train)} ({retention_ratio*100:.1f}%)")
        logger.info(f"Meta-learning fitness: {self.meta_learner.best_fitness:.4f}")
        logger.info(f"Embedder: {type(self.embedder).__name__}")
        logger.info(f"Flow: Raw Data → Embeddings → ML Pipeline Prediction → Pipeline on Raw Data → Final Embeddings → AutoGluon")
        logger.info(f"{'='*80}")

        return results

def run_ml_enhanced_experiment():
    """Run experiment with ML-enhanced meta-learning approach"""
    total_start = time.time()
    logger.info("="*80)
    logger.info("ML-ENHANCED META-LEARNING EXPERIMENT")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)

    # 1. Load data
    train_df, val_df, test_df, target_col = load_anli_dataset()
    if train_df is None:
        logger.error("Failed to load dataset")
        return

    # 2. Initialize ML-enhanced framework
    logger.info("\n" + "="*60)
    logger.info("INITIALIZING ML-ENHANCED META-LEARNING FRAMEWORK")
    logger.info("="*60)

    framework = MetaLearningFramework("anli", use_ml_enhanced=True)

    # Use subset for faster experimentation
    subset_size = min(2000, len(train_df))
    train_subset = train_df.sample(n=subset_size, random_state=RANDOM_SEED)
    val_subset = val_df.sample(n=min(500, len(val_df)), random_state=RANDOM_SEED)
    test_subset = test_df.sample(n=min(500, len(test_df)), random_state=RANDOM_SEED)

    logger.info(f"Using subsets - Train: {len(train_subset)}, Val: {len(val_subset)}, Test: {len(test_subset)}")

    # 3. Run complete experiment
    results = framework.run_complete_experiment(train_subset, val_subset, test_subset, target_col)

    # 4. Compare with traditional approach
    logger.info("\n" + "="*60)
    logger.info("COMPARISON WITH TRADITIONAL APPROACH")
    logger.info("="*60)

    # Run traditional approach for comparison
    traditional_framework = MetaLearningFramework("anli_traditional", use_ml_enhanced=False)
    traditional_results = traditional_framework.run_complete_experiment(train_subset, val_subset, test_subset, target_col)

    # 5. Summary comparison
    total_time = time.time() - total_start
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT COMPARISON SUMMARY")
    logger.info("="*80)

    logger.info("ML-Enhanced Approach:")
    logger.info(f"  Meta-learning fitness: {results['meta_learning_fitness']:.4f}")
    logger.info(f"  Data retention: {results['retention_ratio']:.2f}")
    logger.info(f"  Total time: {format_time(results['timing']['total_time'])}")
    logger.info(f"  AutoGluon performance: {results['autogluon_performance']}")

    logger.info("\nTraditional Evolutionary Approach:")
    logger.info(f"  Meta-learning fitness: {traditional_results['meta_learning_fitness']:.4f}")
    logger.info(f"  Data retention: {traditional_results['retention_ratio']:.2f}")
    logger.info(f"  Total time: {format_time(traditional_results['timing']['total_time'])}")
    logger.info(f"  AutoGluon performance: {traditional_results['autogluon_performance']}")

    # Determine winner
    ml_fitness = results['meta_learning_fitness']
    traditional_fitness = traditional_results['meta_learning_fitness']

    if ml_fitness > traditional_fitness:
        logger.info(f"\n🏆 ML-Enhanced approach wins! (+{ml_fitness - traditional_fitness:.4f} fitness)")
    elif traditional_fitness > ml_fitness:
        logger.info(f"\n🏆 Traditional approach wins! (+{traditional_fitness - ml_fitness:.4f} fitness)")
    else:
        logger.info(f"\n🤝 Tie! Both approaches achieved similar fitness")

    logger.info(f"\nTotal experiment time: {format_time(total_time)}")
    logger.info("="*80)

    # Save comparison results
    comparison_results = {
        'ml_enhanced': results,
        'traditional': traditional_results,
        'comparison': {
            'ml_fitness_advantage': float(ml_fitness - traditional_fitness),
            'ml_time_advantage': float(traditional_results['timing']['total_time'] - results['timing']['total_time']),
            'winner': 'ml_enhanced' if ml_fitness > traditional_fitness else 'traditional' if traditional_fitness > ml_fitness else 'tie'
        },
        'experiment_time': float(total_time),
        'timestamp': datetime.now().isoformat()
    }

    import json
    with open('ml_enhanced_comparison_results.json', 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    logger.info("Comparison results saved to ml_enhanced_comparison_results.json")

if __name__ == "__main__":
    run_ml_enhanced_experiment()
