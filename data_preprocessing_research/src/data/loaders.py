"""
Dataset loading utilities for text classification experiments.

This module provides functionality to load and prepare various text classification
datasets with consistent preprocessing and train/test splits.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import logging
import pickle
from sklearn.model_selection import train_test_split

# Dataset-specific imports will be done dynamically to avoid authentication issues

try:
    from ..utils.logging_config import get_logger
except ImportError:
    # Fallback for when running as script
    import logging
    def get_logger(name):
        return logging.getLogger(name)
try:
    from ..utils.reproducibility import set_random_seeds
except ImportError:
    # Fallback for when running as script
    def set_random_seeds(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

logger = get_logger(__name__)

class DatasetLoader:
    """
    Unified dataset loader for text classification datasets.
    """
    
    def __init__(self, cache_dir: str = "datasets/processed"):
        """
        Initialize the dataset loader.
        
        Args:
            cache_dir: Directory to cache processed datasets
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"DatasetLoader initialized with cache dir: {cache_dir}")
    
    def load_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any],
        force_reload: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load a dataset with train/validation/test splits.

        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration
            force_reload: Whether to force reload from source

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Check cache first
        cache_file = self.cache_dir / f"{dataset_name}_processed.pkl"

        if cache_file.exists() and not force_reload:
            logger.info(f"Loading cached dataset from {cache_file}")
            with open(cache_file, 'rb') as f:
                cached_data = pickle.load(f)
                if len(cached_data) == 3:
                    train_data, val_data, test_data = cached_data
                else:
                    # Old cache format with only train/test
                    train_data, test_data = cached_data
                    val_data = None

            if val_data is not None:
                return train_data, val_data, test_data

        # Load dataset from your HuggingFace repository
        train_data, val_data, test_data = self._load_from_hf_repo(dataset_name, config)

        # Apply sampling if max_samples is specified
        max_samples = config.get("max_samples")
        if max_samples and len(train_data) + len(val_data) + len(test_data) > max_samples:
            train_data, val_data, test_data = self._apply_sampling_three_way(
                train_data, val_data, test_data, max_samples, config
            )

        # Cache the processed dataset
        logger.info(f"Caching processed dataset to {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump((train_data, val_data, test_data), f)

        logger.info(f"Dataset loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")
        return train_data, val_data, test_data

    def _load_from_hf_repo(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load dataset from your HuggingFace repository.

        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")

        logger.info(f"Loading dataset from HF repo: {dataset_name}")

        # Your repository ID
        repo_id = "MothMalone/data-preprocessing-automl-benchmarks"

        # Maximum rows per split
        max_rows = 50000

        try:
            # Load dataset from your repository
            dataset = load_dataset(repo_id, dataset_name)

            # Convert to pandas DataFrames with row limits
            train_data = dataset["train"].to_pandas()
            val_data = dataset["validation"].to_pandas()
            test_data = dataset["test"].to_pandas()

            # Apply row limits
            if len(train_data) > max_rows:
                logger.info(f"Limiting train data from {len(train_data)} to {max_rows} rows")
                train_data = train_data.sample(n=max_rows, random_state=42).reset_index(drop=True)

            if len(val_data) > max_rows:
                logger.info(f"Limiting validation data from {len(val_data)} to {max_rows} rows")
                val_data = val_data.sample(n=max_rows, random_state=42).reset_index(drop=True)

            if len(test_data) > max_rows:
                logger.info(f"Limiting test data from {len(test_data)} to {max_rows} rows")
                test_data = test_data.sample(n=max_rows, random_state=42).reset_index(drop=True)

            logger.info(f"Loaded {dataset_name}: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

            return train_data, val_data, test_data

        except Exception as e:
            logger.error(f"Failed to load {dataset_name} from HF repo: {e}")
            raise ValueError(f"Could not load {dataset_name} from repository {repo_id}: {e}")
    
    def _load_huggingface_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from HuggingFace datasets."""
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("datasets library not installed. Install with: pip install datasets")

        logger.info(f"Loading HuggingFace dataset: {dataset_name}")

        # Get HuggingFace dataset ID from config
        hf_dataset_id = config.get("hf_dataset_id")
        if not hf_dataset_id:
            raise ValueError(f"No hf_dataset_id specified for {dataset_name}")

        # Load dataset
        dataset = load_dataset(hf_dataset_id)
        
        # Convert to pandas - combine all available splits
        dfs = []
        for split in ["train", "test", "validation"]:
            if split in dataset:
                dfs.append(dataset[split].to_pandas())

        if not dfs:
            raise ValueError(f"No data splits found in dataset {hf_dataset_id}")

        # Combine all splits
        train_df = pd.concat(dfs, ignore_index=True)

        # Ensure target column exists
        target_col = config["target_column"]
        if target_col not in train_df.columns:
            # Try common alternatives
            if "labels" in train_df.columns:
                train_df[target_col] = train_df["labels"]
            elif "label" in train_df.columns and target_col != "label":
                train_df[target_col] = train_df["label"]
            else:
                raise ValueError(f"Target column '{target_col}' not found in dataset")

        # Ensure text columns exist
        text_cols = config["text_columns"]
        for text_col in text_cols:
            if text_col not in train_df.columns:
                # Try common alternatives
                if "text" in train_df.columns and text_col != "text":
                    train_df[text_col] = train_df["text"]
                elif "sentence" in train_df.columns:
                    train_df[text_col] = train_df["sentence"]
                elif "content" in train_df.columns:
                    train_df[text_col] = train_df["content"]
                else:
                    raise ValueError(f"Text column '{text_col}' not found in dataset")
        
        # Create train/test split
        target_col = config["target_column"]
        test_size = config.get("test_split", 0.2)
        random_state = config.get("random_seed", 42)
        
        train_data, test_data = train_test_split(
            train_df,
            test_size=test_size,
            random_state=random_state,
            stratify=train_df[target_col]
        )
        
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def _load_sklearn_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from scikit-learn."""
        try:
            from sklearn.datasets import fetch_20newsgroups
        except ImportError:
            raise ImportError("scikit-learn library not installed. Install with: pip install scikit-learn")

        logger.info(f"Loading sklearn dataset: {dataset_name}")

        if dataset_name == "twenty_newsgroups":
            # Load 20 newsgroups dataset
            categories = None  # Use all categories

            # Fetch train and test data
            train_data_raw = fetch_20newsgroups(subset='train', categories=categories,
                                              remove=('headers', 'footers', 'quotes'))
            test_data_raw = fetch_20newsgroups(subset='test', categories=categories,
                                             remove=('headers', 'footers', 'quotes'))

            # Convert to DataFrame
            train_df = pd.DataFrame({
                'data': train_data_raw.data,
                'target': train_data_raw.target
            })

            test_df = pd.DataFrame({
                'data': test_data_raw.data,
                'target': test_data_raw.target
            })

            # Combine for unified processing
            train_df['split'] = 'train'
            test_df['split'] = 'test'
            combined_df = pd.concat([train_df, test_df], ignore_index=True)

        else:
            raise ValueError(f"Unknown sklearn dataset: {dataset_name}")

        # Create train/test split from combined data
        target_col = config["target_column"]
        test_size = config.get("test_split", 0.2)
        random_state = config.get("random_seed", 42)

        train_data, test_data = train_test_split(
            combined_df.drop('split', axis=1),
            test_size=test_size,
            random_state=random_state,
            stratify=combined_df[target_col]
        )

        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def _load_kaggle_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from Kaggle."""
        try:
            import kaggle
        except ImportError:
            raise ImportError("kaggle library not installed. Install with: pip install kaggle")

        logger.info(f"Loading Kaggle dataset: {dataset_name}")
        
        if dataset_name == "quora_question_pairs":
            # Download Quora Question Pairs dataset
            kaggle.api.dataset_download_files(
                'c1udexl/quora-insincere-questions-classification',
                path=self.cache_dir / "raw",
                unzip=True
            )
            
            # Load the dataset
            data_file = self.cache_dir / "raw" / "train.csv"
            df = pd.read_csv(data_file)
            
            # Prepare columns
            df = df.rename(columns={
                'question_text': 'question1',
                'target': 'is_duplicate'
            })
            df['question2'] = df['question1']  # Simplified for this example
                
        elif dataset_name == "news_category":
            kaggle.api.dataset_download_files(
                'rmisra/news-category-dataset',
                path=self.cache_dir / "raw",
                unzip=True
            )
            
            # Load the dataset
            data_file = self.cache_dir / "raw" / "News_Category_Dataset_v3.json"
            df = pd.read_json(data_file, lines=True)
            
            # Prepare columns
            df = df.rename(columns={
                'headline': 'headline',
                'short_description': 'short_description',
                'category': 'category'
            })
        
        else:
            raise ValueError(f"Unknown Kaggle dataset: {dataset_name}")
        
        # Create train/test split
        target_col = config["target_column"]
        test_size = config.get("test_split", 0.2)
        random_state = config.get("random_seed", 42)
        
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col] if target_col in df.columns else None
        )
        
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)
    
    def _load_nltk_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from NLTK."""
        try:
            import nltk
            from nltk.corpus import reuters
        except ImportError:
            raise ImportError("nltk library not installed. Install with: pip install nltk")

        logger.info(f"Loading NLTK dataset: {dataset_name}")
        
        if dataset_name == "reuters_news":
            # Download required NLTK data
            try:
                nltk.data.find('corpora/reuters')
            except LookupError:
                nltk.download('reuters')
            
            # Load Reuters corpus
            categories = reuters.categories()
            
            # Get documents and their categories
            documents = []
            labels = []
            
            for category in categories[:10]:  # Limit to top 10 categories
                fileids = reuters.fileids(category)
                for fileid in fileids[:500]:  # Limit documents per category
                    text = reuters.raw(fileid)
                    documents.append(text)
                    labels.append(category)
            
            df = pd.DataFrame({
                'text': documents,
                'category': labels
            })
            
        else:
            raise ValueError(f"Unknown NLTK dataset: {dataset_name}")
        
        # Create train/test split
        target_col = config["target_column"]
        test_size = config.get("test_split", 0.2)
        random_state = config.get("random_seed", 42)
        
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col]
        )
        
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)

    def _apply_sampling(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        max_samples: int,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply sampling to limit dataset size."""
        total_samples = len(train_data) + len(test_data)

        if total_samples <= max_samples:
            return train_data, test_data

    def _apply_sampling_three_way(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        test_data: pd.DataFrame,
        max_samples: int,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Apply sampling to limit dataset size while maintaining proportions.

        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data
            max_samples: Maximum total samples
            config: Dataset configuration

        Returns:
            Tuple of sampled (train_data, val_data, test_data)
        """
        total_samples = len(train_data) + len(val_data) + len(test_data)

        if total_samples <= max_samples:
            return train_data, val_data, test_data

        # Calculate proportional sizes
        train_ratio = len(train_data) / total_samples
        val_ratio = len(val_data) / total_samples
        test_ratio = len(test_data) / total_samples

        train_limit = int(max_samples * train_ratio)
        val_limit = int(max_samples * val_ratio)
        test_limit = max_samples - train_limit - val_limit

        # Ensure we don't exceed original sizes
        train_limit = min(train_limit, len(train_data))
        val_limit = min(val_limit, len(val_data))
        test_limit = min(test_limit, len(test_data))

        # Sample data
        sampled_train = train_data.sample(n=train_limit, random_state=42).reset_index(drop=True)
        sampled_val = val_data.sample(n=val_limit, random_state=42).reset_index(drop=True)
        sampled_test = test_data.sample(n=test_limit, random_state=42).reset_index(drop=True)

        logger.info(f"Applied sampling: {len(train_data)}->{len(sampled_train)} train, "
                   f"{len(val_data)}->{len(sampled_val)} val, {len(test_data)}->{len(sampled_test)} test")

        return sampled_train, sampled_val, sampled_test

        # Calculate sampling ratio
        sampling_ratio = max_samples / total_samples

        # Sample from both train and test sets
        target_col = config["target_column"]
        random_state = config.get("random_seed", 42)

        if target_col in train_data.columns:
            # Stratified sampling
            train_sampled = train_data.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(int(len(x) * sampling_ratio), random_state=random_state)
            ).reset_index(drop=True)

            test_sampled = test_data.groupby(target_col, group_keys=False).apply(
                lambda x: x.sample(int(len(x) * sampling_ratio), random_state=random_state)
            ).reset_index(drop=True)
        else:
            # Random sampling
            train_sampled = train_data.sample(
                int(len(train_data) * sampling_ratio),
                random_state=random_state
            ).reset_index(drop=True)

            test_sampled = test_data.sample(
                int(len(test_data) * sampling_ratio),
                random_state=random_state
            ).reset_index(drop=True)

        logger.info(f"Applied sampling: {total_samples} -> {len(train_sampled) + len(test_sampled)} samples")
        return train_sampled, test_sampled

    def get_dataset_info(self, dataset_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a dataset without loading it.

        Args:
            dataset_name: Name of the dataset
            config: Dataset configuration

        Returns:
            Dataset information dictionary
        """
        info = {
            "name": dataset_name,
            "source": config.get("source"),
            "task_type": config.get("task_type"),
            "target_column": config.get("target_column"),
            "text_columns": config.get("text_columns"),
            "max_samples": config.get("max_samples"),
            "description": config.get("description"),
            "quality_issues": config.get("quality_issues", [])
        }

        return info
    
    def _load_uci_dataset(
        self,
        dataset_name: str,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load dataset from UCI repository."""
        logger.info(f"Loading UCI dataset: {dataset_name}")
        
        if dataset_name == "spam_detection":
            # Load SMS Spam Collection dataset
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
            
            import urllib.request
            import zipfile
            
            # Download and extract
            zip_path = self.cache_dir / "raw" / "smsspam.zip"
            zip_path.parent.mkdir(parents=True, exist_ok=True)
            
            urllib.request.urlretrieve(url, zip_path)
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cache_dir / "raw")
            
            # Load the data
            data_file = self.cache_dir / "raw" / "SMSSpamCollection"
            df = pd.read_csv(data_file, sep='\t', header=None, names=['label', 'message'])
            
            # Convert labels
            df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        
        else:
            raise ValueError(f"Unknown UCI dataset: {dataset_name}")
        
        # Create train/test split
        target_col = config["target_column"]
        test_size = config.get("test_split", 0.2)
        random_state = config.get("random_seed", 42)
        
        train_data, test_data = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df[target_col]
        )
        
        return train_data.reset_index(drop=True), test_data.reset_index(drop=True)
