import os
import sys
import importlib
import subprocess
import time
import json
import logging
import warnings
from datetime import datetime


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import torch
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import FunctionTransformer
from sentence_transformers import SentenceTransformer
from autogluon.tabular import TabularPredictor


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


warnings.filterwarnings('ignore')
pd.set_option('future.no_silent_downcasting', True)


# Create output directories
BASE_DIR = "/storage/nammt/autogluon"
OUTPUT_DIR = os.path.join(BASE_DIR, "autogluon_comparison_results")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")
CONFUSION_MATRICES_DIR = os.path.join(OUTPUT_DIR, "confusion_matrices")



use_gpu = torch.cuda.is_available()
device = 'cuda' if use_gpu else 'cpu'
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Small but effective embedding model
EMBEDDING_DIM = 384  # Dimension of the all-MiniLM-L6-v2 model

logger.info(f"Initializing SentenceTransformer model '{EMBEDDING_MODEL}'")
logger.info(f"Using device: {device}")

try:
    sentence_transformer = SentenceTransformer(EMBEDDING_MODEL, device=device)
    logger.info("SentenceTransformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load SentenceTransformer model: {e}")
    sentence_transformer = None
    logger.warning("Will use fallback random embeddings since SentenceTransformer failed to load")

def generate_text_embeddings(texts, batch_size=32):
    """Generate embeddings for text using SentenceTransformer
    
    Args:
        texts (list or Series): List or pandas Series of text strings to encode
        batch_size (int): Batch size for encoding
        
    Returns:
        numpy.ndarray: Array of embeddings with shape (len(texts), embedding_dim)
    """
    if sentence_transformer is None:
        logger.error("SentenceTransformer model not available")
        # Return a small random embedding as a fallback
        logger.warning(f"Using random embeddings with dimension {EMBEDDING_DIM}")
        # Use a consistent random seed for deterministic behavior
        np_random = np.random.RandomState(42)
        return np_random.normal(size=(len(texts), EMBEDDING_DIM)).astype(np.float32)
    
    logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
    try:
        if hasattr(texts, 'isna'):
            # For pandas Series
            missing_mask = texts.isna()
            if missing_mask.any():
                logger.warning(f"Found {missing_mask.sum()} missing text values. Replacing with empty strings.")
                texts = texts.fillna('')
        
        # Convert texts to list if it's a pandas Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
            
        # Handle None values in list
        texts = [t if t is not None else '' for t in texts]
        
        embeddings = sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=device
        )
        logger.info(f"Generated embeddings with shape {embeddings.shape}")
        
        if not isinstance(embeddings, np.ndarray) or embeddings.dtype != np.float32:
            embeddings = np.array(embeddings, dtype=np.float32)
            
        return embeddings
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Return random embeddings as a fallback with consistent seed
        logger.warning(f"Falling back to random embeddings with dimension {EMBEDDING_DIM}")
        np_random = np.random.RandomState(42)
        return np_random.normal(size=(len(texts), EMBEDDING_DIM)).astype(np.float32)

def format_time(seconds):
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

def load_dataset_from_csv(base_path, dataset_name=None, label_column='label'):
    """Generic function to load any dataset from CSV files
    
    Args:
        base_path: Path to directory containing train_subset.csv, val_subset.csv, and test_subset.csv
        dataset_name: Optional name for logging purposes
        label_column: Name of the label column (defaults to 'label')
    
    Returns:
        train_df, val_df, test_df, label_column
    """
    start_time = time.time()
    if dataset_name:
        logger.info(f"Loading {dataset_name} dataset from {base_path}...")
    else:
        logger.info(f"Loading dataset from {base_path}...")

    try:
        train_df = pd.read_csv(f"{base_path}/train_subset.csv")
        val_df = pd.read_csv(f"{base_path}/val_subset.csv")
        test_df = pd.read_csv(f"{base_path}/test_subset.csv")
        
        # Ensure text column exists - construct if needed based on available columns
        if 'text' not in train_df.columns:
            # For ANLI
            if 'premise' in train_df.columns and 'hypothesis' in train_df.columns:
                for df in [train_df, val_df, test_df]:
                    df['text'] = df['premise'] + " [SEP] " + df['hypothesis']
            # For CaseHold 
            elif 'citing_prompt' in train_df.columns and 'holding_0' in train_df.columns:
                for df in [train_df, val_df, test_df]:
                    holdings = []
                    for i in range(5): 
                        if f'holding_{i}' in df.columns:
                            holdings.append(df[f'holding_{i}'])
                    df['text'] = df['citing_prompt'] + " [SEP] " + " [SEP] ".join(holdings)
        
        load_time = time.time() - start_time
        logger.info(f"Dataset loaded from CSV in {format_time(load_time)}")
        logger.info(f"Sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        if label_column not in train_df.columns:
            available_columns = train_df.columns.tolist()
            logger.error(f"Label column '{label_column}' not found. Available columns: {available_columns}")
            return None, None, None, None
            
        return train_df, val_df, test_df, label_column

    except Exception as e:
        logger.error(f"Error loading dataset from CSV: {e}")
        return None, None, None, None

def identity_function(x):
    """Identity function that properly handles pandas DataFrames and numpy arrays"""
    if isinstance(x, pd.DataFrame):
        return x.values  # Convert DataFrame to numpy array
    return x

def generate_text_embeddings(texts, batch_size=32):
    """Generate embeddings for text using SentenceTransformer
    
    Args:
        texts: List or Series of texts to embed
        batch_size: Batch size for embedding generation
        
    Returns:
        numpy array of embeddings
    """
    if sentence_transformer is None:
        logger.error("SentenceTransformer model not available")
        return np.random.normal(size=(len(texts), 384))
    
    start_time = time.time()
    logger.info(f"Generating embeddings for {len(texts)} texts")
    
    try:
        if isinstance(texts, pd.Series):
            texts = texts.tolist()
        
        texts = [str(t) if t is not None else "" for t in texts]
        
        embeddings = sentence_transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            device=device
        )
        
        # Check for NaN or Inf values
        if np.isnan(embeddings).any() or np.isinf(embeddings).any():
            logger.warning("NaN or Inf values found in embeddings, replacing with zeros")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        generation_time = time.time() - start_time
        logger.info(f"Generated {embeddings.shape[1]}-dimensional embeddings in {format_time(generation_time)}")
        
        return embeddings
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        # Return random embeddings as fallback
        logger.warning("Returning random embeddings as fallback")
        return np.random.normal(size=(len(texts), 384))

class TPOTPreprocessor:
    """Class to find and apply preprocessing pipeline using TPOT without the final classifier"""
    
    def __init__(self, population_size=20, generations=5, cv=5, n_jobs=-1, random_state=42):
        """Initialize TPOT preprocessor with specified parameters
        
        Args:
            population_size: Population size for TPOT (default: 50)
            generations: Number of generations to evolve (default: 5)
            cv: Number of cross-validation folds (default: 5)
            n_jobs: Number of parallel jobs (default: -1 for all CPUs)
            random_state: Random state for reproducibility (default: 42)
        """
        self.population_size = population_size
        self.generations = generations
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_pipeline = None
        self.preprocessing_pipeline = None
    
    def fit(self, X, y, cache_path=None):
        """Find best preprocessing pipeline using TPOT
        
        Args:
            X: Features (dataframe or array)
            y: Target labels
            cache_path: Path to save/load TPOT pipeline (optional)
        
        Returns:
            self: The fitted preprocessor
        """
        start_time = time.time()
        self.fit_time = 0  
        
        if X.shape[0] == 0 or X.shape[1] == 0:
            logger.error(f"Empty dataset provided: {X.shape}")
            self.best_pipeline = Pipeline([('identity', FunctionTransformer(identity_function, validate=False))])
            self.preprocessing_pipeline = self.best_pipeline
            return self
            
        # Check data types to avoid TPOT errors
        if isinstance(X, pd.DataFrame):
            # Check for non-numeric columns
            non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"Dropping non-numeric columns before TPOT processing: {non_numeric_cols}")
                X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
                
            # If no columns left, create simple numeric features
            if X.shape[1] == 0:
                logger.warning("No numeric columns left. Creating simple numeric features.")
                X = pd.DataFrame({
                    'synthetic_feat1': np.random.normal(size=len(y)),
                    'synthetic_feat2': np.random.normal(size=len(y))
                })
        
        # Check if cached pipeline exists
        if cache_path and os.path.exists(cache_path):
            logger.info(f"Loading cached TPOT pipeline from {cache_path}")
            try:
                self.best_pipeline = joblib.load(cache_path)
                # Set minimal fit time for cached pipeline
                self.fit_time = 0.1
            except Exception as e:
                logger.error(f"Failed to load cached pipeline: {e}")
                # Will proceed to fit a new pipeline
                pass
        
        # If no cached pipeline or loading failed, fit a new one
        if not hasattr(self, 'best_pipeline') or self.best_pipeline is None:
            logger.info(f"Starting TPOT optimization with population_size={self.population_size}, generations={self.generations}...")
            
            # Define minimal config with only preprocessing operations + LogisticRegression and RandomForest
            config_dict = {
                # Preprocessing
                'sklearn.preprocessing.StandardScaler': {},
                'sklearn.preprocessing.MinMaxScaler': {},
                'sklearn.preprocessing.RobustScaler': {},
                'sklearn.preprocessing.Normalizer': {'norm': ['l1', 'l2']},
                'sklearn.decomposition.PCA': {'n_components': [0.5, 0.75, 0.9, 0.95]},
                'sklearn.decomposition.FastICA': {'n_components': [0.5, 0.75, 0.9, 0.95]},
                'sklearn.feature_selection.SelectPercentile': {'percentile': [10, 25, 50, 75], 
                                                              'score_func': {'sklearn.feature_selection.f_classif'}},
                'sklearn.feature_selection.SelectFromModel': {
                    'estimator': {'sklearn.ensemble.ExtraTreesClassifier': {'n_estimators': [100],
                                                                          'criterion': ['gini', 'entropy'],
                                                                          'random_state': [self.random_state]}},
                    'threshold': [0.01, 0.05, 0.1, 0.2]
                },
                
                # Classifiers (limited to LogisticRegression and RandomForest)
                'sklearn.linear_model.LogisticRegression': {
                    'penalty': ['l2'],
                    'C': [0.01, 0.1, 1.0, 10.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [1000]
                },
                'sklearn.ensemble.RandomForestClassifier': {
                    'n_estimators': [100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_features': ['sqrt', 'log2'],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            }
            
            try:
                # Run TPOT
                tpot = TPOTClassifier(
                    generations=self.generations,
                    population_size=self.population_size,
                    verbosity=2,
                    random_state=self.random_state,
                    cv=self.cv,
                    n_jobs=self.n_jobs,
                    config_dict=config_dict
                )
                
                tpot.fit(X, y)
                self.best_pipeline = tpot.fitted_pipeline_
                
                # Save pipeline if cache_path is provided
                if cache_path and self.best_pipeline:
                    joblib.dump(self.best_pipeline, cache_path)
                    logger.info(f"Saved TPOT pipeline to {cache_path}")
                
            except Exception as e:
                logger.error(f"TPOT optimization failed: {e}")
                logger.info("Using fallback preprocessing pipeline")
                # Create a simple pipeline with standard scaler as fallback
                from sklearn.preprocessing import StandardScaler
                self.best_pipeline = Pipeline([('scaler', StandardScaler())])
            
            # Store TPOT's actual fit time for more accurate timing
            self.fit_time = time.time() - start_time
            
        # Check if TPOT produced a valid pipeline
        if not hasattr(self, 'best_pipeline') or self.best_pipeline is None:
            logger.warning("No pipeline was produced. Creating a simple fallback pipeline.")
            self.best_pipeline = Pipeline([('identity', FunctionTransformer(identity_function, validate=False))])
        
        # Extract only preprocessing steps (all steps except the last one which is the classifier)
        try:
            if self.best_pipeline and isinstance(self.best_pipeline, Pipeline):
                if len(self.best_pipeline.steps) > 1:
                    # Extract all steps except the last one (classifier)
                    preprocessing_steps = self.best_pipeline.steps[:-1]

                    if preprocessing_steps:
                        self.preprocessing_pipeline = Pipeline(preprocessing_steps)
                        logger.info(f"Extracted preprocessing pipeline: {self.preprocessing_pipeline}")
                    else:
                        # If no preprocessing steps, create identity transformer
                        logger.info("No preprocessing steps found, using identity transformer")
                        self.preprocessing_pipeline = Pipeline([
                            ('identity', FunctionTransformer(identity_function, validate=False))
                        ])
                else:
                    # Only one step (likely just the classifier)
                    logger.info("Pipeline has only one step, using identity transformer")
                    self.preprocessing_pipeline = Pipeline([
                        ('identity', FunctionTransformer(identity_function, validate=False))
                    ])
            else:
                logger.warning("Failed to extract preprocessing pipeline, using identity transformer")
                self.preprocessing_pipeline = Pipeline([
                    ('identity', FunctionTransformer(identity_function, validate=False))
                ])
        except Exception as e:
            logger.error(f"Error extracting preprocessing pipeline: {e}")
            self.preprocessing_pipeline = Pipeline([
                ('identity', FunctionTransformer(identity_function, validate=False))
            ])
            
        fit_time = time.time() - start_time
        logger.info(f"TPOT preprocessing pipeline found in {format_time(fit_time)}")
        return self
    
    def transform(self, X):
        """Apply preprocessing pipeline to features
        
        Args:
            X: Features to transform
        
        Returns:
            Transformed features
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        try:
            # Check for empty or invalid data
            if X.shape[0] == 0 or X.shape[1] == 0:
                logger.warning(f"Empty dataset provided for transform: {X.shape}")
                # Return empty array with correct dimensions
                return np.zeros((X.shape[0], 1))
                
            # Handle non-numeric data if DataFrame
            if isinstance(X, pd.DataFrame):
                # Get only numeric columns
                non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64', 'int32', 'float32']).columns.tolist()
                if non_numeric_cols:
                    logger.warning(f"Dropping non-numeric columns before transform: {non_numeric_cols}")
                    X = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32'])
                    
                # If no columns left, create simple numeric features
                if X.shape[1] == 0:
                    logger.warning("No numeric columns left for transform. Creating simple features.")
                    X = pd.DataFrame({
                        'synthetic_feat1': np.random.normal(size=X.shape[0]),
                        'synthetic_feat2': np.random.normal(size=X.shape[0])
                    })
            
            # Apply transformation
            X_transformed = self.preprocessing_pipeline.transform(X)
            
            # Check for NaN or inf values
            if np.isnan(X_transformed).any() or np.isinf(X_transformed).any():
                logger.warning("NaN or Inf values found in transformed data. Replacing with zeros.")
                X_transformed = np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)
                
            return X_transformed
            
        except Exception as e:
            logger.error(f"Error during transform: {e}")
            # Return original data as fallback
            if isinstance(X, pd.DataFrame):
                X = X.values
            
            # If that fails, return zeros
            logger.warning("Transformation failed. Returning array of zeros as fallback.")
            return np.zeros((X.shape[0], 1))
    
    def fit_transform(self, X, y, cache_path=None):
        """Fit and transform in one step
        
        Args:
            X: Features
            y: Target labels
            cache_path: Path to save/load TPOT pipeline (optional)
            
        Returns:
            Transformed features
        """
        self.fit(X, y, cache_path)
        return self.transform(X)

def train_tabular_model(train_df, val_df, target_col, dataset_name, variant, model_dir, start_time):
    """Train standard TabularPredictor using AutoGluon defaults"""
    logger.info(f"Using TabularPredictor for {dataset_name} ({variant})")

    # Check for valid features
    feature_cols = [col for col in train_df.columns if col != target_col]
    if not feature_cols:
        logger.error("No feature columns found in the dataset")
        return None, 0
    
    # Ensure we have at least some numeric features for the models
    numeric_features = train_df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    if not numeric_features:
        # If no numeric features, convert all features to float
        logger.warning("No numeric features found. Converting all feature columns to float.")
        for col in feature_cols:
            try:
                train_df[col] = train_df[col].astype(float)
                if val_df is not None:
                    val_df[col] = val_df[col].astype(float)
            except Exception as e:
                logger.warning(f"Could not convert column {col} to float: {e}")
    
    # Log feature information
    logger.info(f"Training with {len(feature_cols)} features")
    numeric_features = train_df[feature_cols].select_dtypes(include=['number']).columns.tolist()
    logger.info(f"Number of numeric features: {len(numeric_features)}")
    
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

    # Add a simple check before training
    try:
        # Use AutoGluon's default settings with increased memory allowance
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',
            time_limit=3000,
            verbosity=2,
            # Allow AutoGluon to use more memory
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
        )
    except Exception as e:
        logger.error(f"AutoGluon training failed: {e}")
        # Create emergency fallback predictor
        logger.warning("Attempting to create fallback predictor")
        
        # Save the processed data for debugging
        train_df.to_csv(os.path.join(model_dir, "debug_train_data.csv"), index=False)
        if val_df is not None:
            val_df.to_csv(os.path.join(model_dir, "debug_val_data.csv"), index=False)
            
        return None, time.time() - start_time

    training_time = time.time() - start_time
    logger.info(f"Tabular model training completed for {dataset_name} ({variant}) in {format_time(training_time)}")
    return predictor, training_time

def get_model_info(predictor, dataset_name, variant):
    """Get model size and parameter information"""
    logger.info(f"Getting model information for {dataset_name} ({variant})...")

    try:
        model_info = {
            'model_size_mb': 'N/A',
            'num_parameters': 'N/A',
            'model_type': 'Unknown'
        }

        # Get model type
        if hasattr(predictor, '__class__'):
            model_info['model_type'] = predictor.__class__.__name__

        # Try to get model size from disk
        if hasattr(predictor, 'path') and predictor.path:
            try:
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(predictor.path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)
                model_info['model_size_mb'] = round(total_size / (1024 * 1024), 2)
            except Exception as e:
                logger.warning(f"Could not calculate model size: {e}")

        # Try to get parameter count for neural network models
        try:
            if hasattr(predictor, 'get_model_names'):
                model_names = predictor.get_model_names()
                total_params = 0
                for model_name in model_names:
                    if 'NN' in model_name or 'Neural' in model_name or 'FASTAI' in model_name:
                        # Try to get model object and count parameters
                        try:
                            model_obj = predictor._trainer.load_model(model_name)
                            if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'parameters'):
                                params = sum(p.numel() for p in model_obj.model.parameters())
                                total_params += params
                        except:
                            pass
                if total_params > 0:
                    model_info['num_parameters'] = total_params
        except Exception as e:
            logger.warning(f"Could not count parameters: {e}")

        logger.info(f"Model info for {dataset_name} ({variant}): {model_info}")
        return model_info

    except Exception as e:
        logger.error(f"Error getting model info for {dataset_name}: {e}")
        return {
            'model_size_mb': 'Error',
            'num_parameters': 'Error',
            'model_type': 'Error'
        }

def save_predictions_and_confusion_matrix(predictor, test_df, target_col, dataset_name, variant, predictions_dir, confusion_matrices_dir):
    """Save predictions as CSV and create confusion matrix for classification tasks"""
    logger.info(f"Saving predictions and creating confusion matrix for {dataset_name} ({variant})...")

    try:
        # Make predictions
        test_features = test_df.drop(columns=[target_col])
        
        # Remove image column if it exists and contains PIL Images (for TabularPredictor)
        if 'image' in test_features.columns:
            # Check if image column contains PIL Images or serialized data
            sample_image = test_features['image'].iloc[0]
            if isinstance(sample_image, str) and (sample_image.startswith('{') and sample_image.endswith('}')):
                logger.info("Removing image column for TabularPredictor prediction")
                test_features = test_features.drop(columns=['image'])

        predictions = predictor.predict(test_features)

        # Get prediction probabilities if available
        try:
            pred_proba = predictor.predict_proba(test_features)
            has_proba = True
        except Exception as e:
            logger.warning(f"Could not get prediction probabilities: {e}")
            pred_proba = None
            has_proba = False

        # Create predictions DataFrame
        pred_df = test_df.copy()
        pred_df['prediction'] = predictions

        if has_proba and pred_proba is not None:
            # Add probability columns
            if isinstance(pred_proba, pd.DataFrame):
                for col in pred_proba.columns:
                    pred_df[f'prob_{col}'] = pred_proba[col]
            else:
                pred_df['prob_positive'] = pred_proba[:, 1] if pred_proba.shape[1] > 1 else pred_proba[:, 0]

        # Save predictions
        pred_file = os.path.join(predictions_dir, f"{dataset_name}_{variant}_predictions.csv")
        pred_df.to_csv(pred_file, index=False)
        logger.info(f"Predictions saved to: {pred_file}")

        # Create confusion matrix for classification tasks
        y_true = test_df[target_col]
        y_pred = predictions
        
        # Check if it's a classification task
        unique_classes = len(set(y_true) | set(y_pred))
        if unique_classes <= 100:  # Reasonable number of classes for a confusion matrix
            try:
                # Create confusion matrix
                cm = confusion_matrix(y_true, y_pred)

                # Get unique labels
                labels = sorted(list(set(y_true) | set(y_pred)))

                # Create confusion matrix plot
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                           xticklabels=labels, yticklabels=labels)
                plt.title(f'Confusion Matrix - {dataset_name} ({variant})')
                plt.xlabel('Predicted')
                plt.ylabel('Actual')

                # Save confusion matrix
                cm_file = os.path.join(confusion_matrices_dir, f"{dataset_name}_{variant}_confusion_matrix.png")
                plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Confusion matrix saved to: {cm_file}")

                # Save classification report
                report = classification_report(y_true, y_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                report_file = os.path.join(confusion_matrices_dir, f"{dataset_name}_{variant}_classification_report.csv")
                report_df.to_csv(report_file)
                logger.info(f"Classification report saved to: {report_file}")

                return predictions, cm
            except Exception as e:
                logger.warning(f"Could not create confusion matrix: {e}")
                return predictions, None
        else:
            logger.info(f"Too many classes ({unique_classes}) for a meaningful confusion matrix")
            return predictions, None

    except Exception as e:
        logger.error(f"Error saving predictions/confusion matrix for {dataset_name}: {e}")
        return None, None

def evaluate_model(predictor, test_df, target_col, dataset_name, variant, predictions_dir, confusion_matrices_dir):
    """Evaluate the trained model with detailed metrics"""
    logger.info(f"Evaluating model for {dataset_name} ({variant})...")
    
    # Check if predictor is None (training failed)
    if predictor is None:
        logger.warning(f"Cannot evaluate model for {dataset_name} ({variant}): predictor is None")
        return {
            'dataset': dataset_name,
            'variant': variant,
            'performance': 'Error',
            'best_model': 'Error - Training Failed',
            'best_val_score': 'Error',
            'test_score': 'Error',
            'accuracy': 'Error',
            'num_models_trained': 0,
            'total_models_attempted': 0,
            'leaderboard_top3': [],
            'model_training_times': {},
            'problem_type': 'Error',
            'eval_metric': 'Error',
            'predictions_saved': False,
            'confusion_matrix_saved': False,
            'model_size_mb': 'N/A',
            'num_parameters': 'N/A',
            'model_type': 'N/A',
            'error': 'AutoGluon training failed'
        }

    try:
        # Get model information (size, parameters)
        model_info = get_model_info(predictor, dataset_name, variant)

        # Save predictions and create confusion matrix first
        predictions, confusion_matrix = save_predictions_and_confusion_matrix(
            predictor, test_df, target_col, dataset_name, variant, 
            predictions_dir, confusion_matrices_dir
        )

        # Handle different predictor types
        if hasattr(predictor, 'predict'):
            # TabularPredictor or MultiModalPredictor evaluation
            test_features = test_df.drop(columns=[target_col])

            # For TabularPredictor fallback, we need to remove image column
            if hasattr(predictor, '__class__') and 'MultiModal' not in predictor.__class__.__name__:
                # This is TabularPredictor fallback, remove image column if it exists
                if 'image' in test_features.columns:
                    logger.info("Removing image column for TabularPredictor evaluation")
                    test_features = test_features.drop(columns=['image'])

            predictions = predictor.predict(test_features)

            # Handle different evaluation APIs
            try:
                performance = predictor.evaluate(test_df, silent=True)
            except TypeError:
                # MultiModalPredictor might not accept silent parameter
                try:
                    performance = predictor.evaluate(test_df)
                except:
                    # Fallback to manual calculation
                    if len(set(test_df[target_col])) <= 10:  # Classification
                        performance = accuracy_score(test_df[target_col], predictions)
                    else:  # Regression
                        from sklearn.metrics import mean_squared_error
                        performance = mean_squared_error(test_df[target_col], predictions, squared=False)

            # Extract accuracy for classification tasks
            accuracy = 'N/A'
            if isinstance(performance, dict):
                accuracy = performance.get('accuracy', performance.get('acc', 'N/A'))
            elif isinstance(performance, (int, float)):
                # If performance is a single number and it's classification, it's likely accuracy
                problem_type = getattr(predictor, 'problem_type', 'unknown')
                if problem_type in ['binary', 'multiclass']:
                    accuracy = performance

            # Get leaderboard if available
            try:
                # Try TabularPredictor leaderboard first
                leaderboard = predictor.leaderboard(test_df, silent=True)
                best_model = leaderboard.iloc[0]['model']
                best_val_score = leaderboard.iloc[0]['score_val']
                test_score = leaderboard.iloc[0]['score_test'] if 'score_test' in leaderboard.columns else None

                # Get training times for individual models
                training_times = {}
                if 'fit_time' in leaderboard.columns:
                    for _, row in leaderboard.iterrows():
                        training_times[row['model']] = row['fit_time']

                leaderboard_top3 = leaderboard.head(3)[['model', 'score_val']].to_dict('records')
                num_models = len(leaderboard)
            except Exception as e:
                logger.info(f"Leaderboard not available (likely MultiModalPredictor): {e}")
                # Fallback for MultiModalPredictor which might not have leaderboard
                predictor_type = 'MultiModalPredictor' if hasattr(predictor, '__class__') and 'MultiModal' in predictor.__class__.__name__ else 'TabularPredictor'
                best_model = predictor_type
                best_val_score = performance if isinstance(performance, (int, float)) else 'N/A'
                test_score = performance if isinstance(performance, (int, float)) else 'N/A'
                training_times = {}
                leaderboard_top3 = [{'model': predictor_type, 'score_val': best_val_score}]
                num_models = 1

            results = {
                'dataset': dataset_name,
                'variant': variant,
                'performance': performance,
                'best_model': best_model,
                'best_val_score': best_val_score,
                'test_score': test_score,
                'accuracy': accuracy,
                'num_models_trained': num_models,
                'total_models_attempted': num_models,
                'leaderboard_top3': leaderboard_top3,
                'model_training_times': training_times,
                'problem_type': getattr(predictor, 'problem_type', 'unknown'),
                'eval_metric': getattr(predictor, 'eval_metric', 'unknown'),
                'predictions_saved': predictions is not None,
                'confusion_matrix_saved': confusion_matrix is not None,
                'model_size_mb': model_info.get('model_size_mb', 'N/A'),
                'num_parameters': model_info.get('num_parameters', 'N/A'),
                'model_type': model_info.get('model_type', 'Unknown')
            }

            # Log detailed results
            logger.info(f"Evaluation completed for {dataset_name} ({variant})")
            logger.info(f"Problem type: {results['problem_type']}")
            logger.info(f"Eval metric: {results['eval_metric']}")
            logger.info(f"Best model: {results['best_model']}")
            logger.info(f"Best validation score: {results['best_val_score']}")
            if results['test_score'] is not None:
                logger.info(f"Test score: {results['test_score']}")
            if results['accuracy'] != 'N/A':
                logger.info(f"Accuracy: {results['accuracy']}")
            logger.info(f"Total models trained: {results['num_models_trained']}")
            logger.info(f"Model size: {results['model_size_mb']} MB")
            logger.info(f"Model parameters: {results['num_parameters']}")

            # Log top models
            logger.info("Top models:")
            for i, model_info in enumerate(results['leaderboard_top3'], 1):
                logger.info(f"  {i}. {model_info['model']}: {model_info['score_val']}")

            return results

    except Exception as e:
        logger.error(f"Error evaluating {dataset_name}: {e}")
        return {
            'dataset': dataset_name,
            'variant': variant,
            'performance': 'Error',
            'best_model': 'Error',
            'best_val_score': 'Error',
            'test_score': 'Error',
            'accuracy': 'Error',
            'num_models_trained': 0,
            'total_models_attempted': 0,
            'leaderboard_top3': [],
            'model_training_times': {},
            'problem_type': 'Error',
            'eval_metric': 'Error',
            'predictions_saved': False,
            'confusion_matrix_saved': False,
            'model_size_mb': 'Error',
            'num_parameters': 'Error',
            'model_type': 'Error'
        }

def process_dataset(dataset_name, dataset_path, label_column='label', use_cache=True):
    """Process a single dataset with TPOT preprocessing + AutoGluon
    
    Args:
        dataset_name: Name of the dataset
        dataset_path: Path to dataset directory
        label_column: Name of the label column
        use_cache: Whether to use cached embeddings and processed data (default: True)
    """
    logger.info(f"="*80)
    logger.info(f"Processing dataset: {dataset_name}")
    logger.info(f"Cache mode: {'Enabled' if use_cache else 'Disabled'}")
    logger.info(f"="*80)
    
    # Create dataset-specific  directory
    dataset_output_dir = os.path.join(OUTPUT_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    try:
        # Load dataset
        train_df, val_df, test_df, label_col = load_dataset_from_csv(
            dataset_path, dataset_name, label_column
        )
        
        if train_df is None:
            logger.error(f"Failed to load dataset {dataset_name}")
            return
            
        logger.info(f"Dataset loaded successfully: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test samples")
        
        # Create cache directory
        cache_dir = os.path.join(BASE_DIR, "preprocessed_cache", dataset_name)

        
        # Define cache paths
        tpot_cache_path = os.path.join(cache_dir, f"{dataset_name}_tpot_pipeline.joblib")
        embeddings_cache_dir = os.path.join(cache_dir, "embeddings")
        processed_data_cache_dir = os.path.join(cache_dir, "processed_data")
        

        
        # Define paths for cached embeddings
        train_embeddings_path = os.path.join(embeddings_cache_dir, "train_embeddings.npy")
        val_embeddings_path = os.path.join(embeddings_cache_dir, "val_embeddings.npy")
        test_embeddings_path = os.path.join(embeddings_cache_dir, "test_embeddings.npy")
        
        # Define paths for processed data
        train_processed_path = os.path.join(processed_data_cache_dir, "train_processed.csv")
        val_processed_path = os.path.join(processed_data_cache_dir, "val_processed.csv")
        test_processed_path = os.path.join(processed_data_cache_dir, "test_processed.csv")
        
        # PHASE 1: Generate embeddings and find preprocessing pipeline with TPOT
        start_time = time.time()
        
        def is_valid_cached_data(file_path):
            """Check if a cached data file is valid
            
            Args:
                file_path (str): Path to the cached data file
                
            Returns:
                bool: True if the file exists and is valid, False otherwise
            """
            if not os.path.exists(file_path):
                return False
                
            try:
                if file_path.endswith('.csv'):
                    # For CSV files, try to load and check if it has rows
                    df = pd.read_csv(file_path, nrows=5)
                    return len(df) > 0 and not df.empty
                elif file_path.endswith('.npy'):
                    # For numpy arrays, check if it has elements
                    arr = np.load(file_path, allow_pickle=True)
                    return arr.size > 0
                elif file_path.endswith('.joblib'):
                    # For joblib files, just check if they can be loaded
                    obj = joblib.load(file_path)
                    return obj is not None
                else:
                    # Unknown file type
                    return False
            except Exception as e:
                logger.warning(f"Invalid cached file {file_path}: {str(e)}")
                return False

        # Since we're working with text data, we'll create embeddings first
        # Check for cached train embeddings (only if use_cache is True)
        cached_train_embeddings = False
        if use_cache:
            cached_train_embeddings = is_valid_cached_data(train_embeddings_path)
        if not use_cache:
            logger.info("Cache is disabled. Will generate new embeddings.")
        elif cached_train_embeddings:
            logger.info(f"Loading cached train embeddings from {train_embeddings_path}")
            try:
                train_embeddings = np.load(train_embeddings_path)
                logger.info(f"Loaded train embeddings with shape {train_embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load cached train embeddings: {e}")
                cached_train_embeddings = False
        
        # Generate embeddings if not cached
        if not cached_train_embeddings:
            logger.info("Generating embeddings for training data...")
            if 'text' in train_df.columns:
                train_embeddings = generate_text_embeddings(train_df['text'])
            elif 'premise' in train_df.columns and 'hypothesis' in train_df.columns:
                # For datasets like ANLI with separate premise/hypothesis
                combined_texts = train_df['premise'] + " [SEP] " + train_df['hypothesis']
                train_embeddings = generate_text_embeddings(combined_texts)
            elif 'citing_prompt' in train_df.columns:
                # For CaseHold dataset
                train_embeddings = generate_text_embeddings(train_df['citing_prompt'])
            else:
                logger.error("No suitable text columns found for embedding generation")
                return None
                
            # Save embeddings for future use
            try:
                np.save(train_embeddings_path, train_embeddings)
                logger.info(f"Saved train embeddings to {train_embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to cache train embeddings: {e}")
            
        # Create DataFrame with embeddings as features
        logger.info(f"Creating feature DataFrame from {train_embeddings.shape[1]} embedding dimensions")
        embedding_cols = [f'embed_{i}' for i in range(train_embeddings.shape[1])]
        X_train = pd.DataFrame(train_embeddings, columns=embedding_cols)
        
        y_train = train_df[label_col]
        
        # Function to validate cached data files
       
        
        # Check if processed data cache exists and is valid (only if use_cache is True)
        all_processed_exists = False
        if use_cache:
            all_processed_exists = (
                is_valid_cached_data(train_processed_path) and
                is_valid_cached_data(val_processed_path) and
                is_valid_cached_data(test_processed_path)
            )
        
        if not use_cache:
            logger.info("Cache is disabled. Will perform TPOT preprocessing.")
        elif all_processed_exists:
            logger.info("Found valid cached processed data. Loading...")
            try:
                train_processed_df = pd.read_csv(train_processed_path)
                val_processed_df = pd.read_csv(val_processed_path)
                test_processed_df = pd.read_csv(test_processed_path)
                logger.info(f"Loaded cached processed data. Train: {train_processed_df.shape}, Val: {val_processed_df.shape}, Test: {test_processed_df.shape}")
                
                # Skip TPOT preprocessing since we already have processed data
                logger.info("Using cached processed data, skipping TPOT preprocessing")
                transform_time = 0
                preprocessing_time = time.time() - start_time
            except Exception as e:
                logger.error(f"Failed to load cached processed data: {e}")
                all_processed_exists = False
        
        if not all_processed_exists:
            logger.info(f"Training TPOT preprocessor on {X_train.shape[1]} features")
        
        # Initialize and fit TPOTPreprocessor
        preprocessor = TPOTPreprocessor(
            population_size=20,  # Larger population for better exploration
            generations=5,       # More generations for better optimization
            cv=5,                # 5-fold cross-validation
            n_jobs=-1,           # Use all available cores
            random_state=42
        )
        
        # Check for sufficient data before proceeding
        if X_train.shape[0] < 10 or X_train.shape[1] < 1:
            logger.error(f"Insufficient data for TPOT: {X_train.shape[0]} samples, {X_train.shape[1]} features")
            return
        
        preprocessor.fit(X_train, y_train, tpot_cache_path)
        
        # PHASE 2: Apply preprocessing to all datasets
        X_train_processed = preprocessor.transform(X_train)
        
        # Process validation and test data
        transform_start_time = time.time()
        
        # Check for cached validation embeddings (only if use_cache is True)
        cached_val_embeddings = False
        if use_cache:
            cached_val_embeddings = os.path.exists(val_embeddings_path)
        if cached_val_embeddings:
            logger.info(f"Loading cached validation embeddings from {val_embeddings_path}")
            try:
                val_embeddings = np.load(val_embeddings_path)
                logger.info(f"Loaded validation embeddings with shape {val_embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load cached validation embeddings: {e}")
                cached_val_embeddings = False
        
        # Generate validation embeddings if not cached
        if not cached_val_embeddings:
            logger.info("Generating embeddings for validation data...")
            if 'text' in val_df.columns:
                val_embeddings = generate_text_embeddings(val_df['text'])
            elif 'premise' in val_df.columns and 'hypothesis' in val_df.columns:
                combined_texts_val = val_df['premise'] + " [SEP] " + val_df['hypothesis']
                val_embeddings = generate_text_embeddings(combined_texts_val)
            elif 'citing_prompt' in val_df.columns:
                val_embeddings = generate_text_embeddings(val_df['citing_prompt'])
            else:
                logger.error("No suitable text columns found in validation data for embedding generation")
                return None
            
            # Save validation embeddings for future use
            try:
                np.save(val_embeddings_path, val_embeddings)
                logger.info(f"Saved validation embeddings to {val_embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to cache validation embeddings: {e}")
            
        # Check for cached test embeddings (only if use_cache is True)
        cached_test_embeddings = False
        if use_cache:
            cached_test_embeddings = os.path.exists(test_embeddings_path)
        if cached_test_embeddings:
            logger.info(f"Loading cached test embeddings from {test_embeddings_path}")
            try:
                test_embeddings = np.load(test_embeddings_path)
                logger.info(f"Loaded test embeddings with shape {test_embeddings.shape}")
            except Exception as e:
                logger.error(f"Failed to load cached test embeddings: {e}")
                cached_test_embeddings = False
                
        # Generate test embeddings if not cached
        if not cached_test_embeddings:
            logger.info("Generating embeddings for test data...")
            if 'text' in test_df.columns:
                test_embeddings = generate_text_embeddings(test_df['text'])
            elif 'premise' in test_df.columns and 'hypothesis' in test_df.columns:
                combined_texts_test = test_df['premise'] + " [SEP] " + test_df['hypothesis']
                test_embeddings = generate_text_embeddings(combined_texts_test)
            elif 'citing_prompt' in test_df.columns:
                test_embeddings = generate_text_embeddings(test_df['citing_prompt'])
            else:
                logger.error("No suitable text columns found in test data for embedding generation")
                return None
                
            # Save test embeddings for future use
            try:
                np.save(test_embeddings_path, test_embeddings)
                logger.info(f"Saved test embeddings to {test_embeddings_path}")
            except Exception as e:
                logger.warning(f"Failed to cache test embeddings: {e}")
        
        # Create DataFrames with embeddings
        X_val = pd.DataFrame(val_embeddings, columns=embedding_cols)
        X_test = pd.DataFrame(test_embeddings, columns=embedding_cols)
        
        # Only apply TPOT preprocessing if we don't have cached processed data
        if not all_processed_exists:
            X_val_processed = preprocessor.transform(X_val)
            X_test_processed = preprocessor.transform(X_test)
            transform_time = time.time() - transform_start_time
            logger.info(f"Transformation time for val/test data: {format_time(transform_time)}")
        
        # If we didn't use cached processed data, create the dataframes and save them
        if not all_processed_exists:
            # Create dataframes with processed features
            feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
            
            train_processed_df = pd.DataFrame(X_train_processed, columns=feature_names)
            train_processed_df[label_col] = y_train.values
            
            val_processed_df = pd.DataFrame(X_val_processed, columns=feature_names)
            val_processed_df[label_col] = val_df[label_col].values
            
            test_processed_df = pd.DataFrame(X_test_processed, columns=feature_names)
            test_processed_df[label_col] = test_df[label_col].values
            
            # Save processed dataframes to run directory
            run_train_path = os.path.join(dataset_output_dir, f"{dataset_name}_train_processed.csv")
            run_val_path = os.path.join(dataset_output_dir, f"{dataset_name}_val_processed.csv")
            run_test_path = os.path.join(dataset_output_dir, f"{dataset_name}_test_processed.csv")
            
            save_start_time = time.time()
            
            # Save to both run directory and cache directory
            train_processed_df.to_csv(run_train_path, index=False)
            val_processed_df.to_csv(run_val_path, index=False)
            test_processed_df.to_csv(run_test_path, index=False)
            
            # Save to cache for future runs
            train_processed_df.to_csv(train_processed_path, index=False)
            val_processed_df.to_csv(val_processed_path, index=False)
            test_processed_df.to_csv(test_processed_path, index=False)
            
            save_time = time.time() - save_start_time
            logger.info(f"Saved processed data to both run directory and cache")
        else:
            # We already loaded the processed data, no need to save again
            save_time = 0
            logger.info("Using cached processed data, no need to save")
        
        preprocessing_time = time.time() - start_time
        logger.info(f"Data saving time: {format_time(save_time)}")
        logger.info(f"Preprocessing completed in {format_time(preprocessing_time)}")
        logger.info(f"Processed data shape: {X_train_processed.shape[1]} features")
        
        # PHASE 3: Train AutoGluon on preprocessed data
        autogluon_start_time = time.time()
        
        # Create model directory
        model_dir = os.path.join(dataset_output_dir, f"{dataset_name}_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if the processed data is valid for AutoGluon
        # Ensure we have numeric features by checking and converting if needed
        feature_cols = [col for col in train_processed_df.columns if col != label_col]
        numeric_features = train_processed_df[feature_cols].select_dtypes(include=['number']).columns.tolist()
        
        # Check for NaN/Inf values in processed data
        has_invalid_values = False
        for df, name in [(train_processed_df, 'train'), (val_processed_df, 'val'), (test_processed_df, 'test')]:
            for col in numeric_features:
                if col in df.columns:
                    invalid_count = np.sum(~np.isfinite(df[col]))
                    if invalid_count > 0:
                        logger.warning(f"Found {invalid_count} invalid values (NaN/Inf) in {name} dataset, column {col}")
                        has_invalid_values = True
        
        # If we have invalid values or no numeric features, use raw embeddings instead
        if not numeric_features or has_invalid_values:
            if not numeric_features:
                logger.warning("No numeric features in processed data. Using raw embeddings directly.")
            else:
                logger.warning("Found invalid values in processed data. Using raw embeddings directly.")
                
            # Use embeddings directly instead of TPOT transformed features
            logger.info("Creating dataframes with raw embeddings as features")
            
            # Create dataframes with raw embeddings
            train_processed_df = pd.DataFrame(train_embeddings, columns=embedding_cols)
            train_processed_df[label_col] = y_train.values
            
            val_processed_df = pd.DataFrame(val_embeddings, columns=embedding_cols)
            val_processed_df[label_col] = val_df[label_col].values
            
            test_processed_df = pd.DataFrame(test_embeddings, columns=embedding_cols)
            test_processed_df[label_col] = test_df[label_col].values
            
            logger.info(f"Using raw embeddings with {len(embedding_cols)} dimensions")
        else:
            logger.info(f"Using TPOT processed features: {len(numeric_features)} numeric features")
            
            # Double-check that we don't have any NaN values in the target column
            for df_name, df in [("train", train_processed_df), ("val", val_processed_df), ("test", test_processed_df)]:
                if df[label_col].isna().any():
                    logger.warning(f"Found NaN values in {df_name} target column. Replacing with most frequent value.")
                    most_frequent = df[label_col].mode()[0]
                    df[label_col] = df[label_col].fillna(most_frequent)
        
        # Train AutoGluon model
        predictor, training_time = train_tabular_model(
            train_processed_df, 
            val_processed_df, 
            label_col, 
            dataset_name,
            "tpot_preprocessed", 
            model_dir, 
            autogluon_start_time
        )
        
        # Check if predictor was created successfully
        if predictor is None:
            logger.error(f"AutoGluon training failed for {dataset_name}")
            
            # Create a basic results dict for error case
            results = {
                'dataset': dataset_name,
                'preprocessing_method': 'tpot',
                'status': 'failed',
                'error': 'AutoGluon training failed - no valid models produced',
                'timing': {
                    'preprocessing_time': preprocessing_time,
                    'total_time': time.time() - start_time
                }
            }
        else:
            # PHASE 4: Evaluate model
            results = evaluate_model(
                predictor,
                test_processed_df,
                label_col,
                dataset_name,
                "tpot_preprocessed",
                PREDICTIONS_DIR,
                CONFUSION_MATRICES_DIR
            )
        
        # Save results
        total_time = time.time() - start_time
        
        # Log detailed timing information
        logger.info(f"Full pipeline timing breakdown:")
        logger.info(f"  TPOT preprocessing time: {format_time(preprocessing_time)}")
        logger.info(f"  AutoGluon training time: {format_time(training_time)}")
        logger.info(f"  Total pipeline time: {format_time(total_time)}")
        
        # Get TPOT pipeline details if available
        tpot_pipeline_details = "Unknown"
        if hasattr(preprocessor, 'best_pipeline') and preprocessor.best_pipeline:
            tpot_pipeline_details = str(preprocessor.best_pipeline)
            logger.info(f"TPOT Pipeline: {tpot_pipeline_details}")
        
        summary = {
            'dataset': dataset_name,
            'preprocessing_method': 'tpot',
            'num_features_after_preprocessing': X_train_processed.shape[1],
            'timing': {
                'tpot_fit_time': preprocessor.fit_time if hasattr(preprocessor, 'fit_time') else preprocessing_time,
                'transform_time': transform_time,
                'data_save_time': save_time,
                'preprocessing_total_time': preprocessing_time,
                'autogluon_training_time': training_time,
                'total_pipeline_time': total_time,
                'formatted_total_time': format_time(total_time)
            },
            'tpot_pipeline': tpot_pipeline_details,
            'accuracy': results.get('accuracy', 'N/A'),
            'performance': results.get('performance', 'N/A'),
            'best_model': results.get('best_model', 'N/A'),
            'leaderboard_top3': results.get('leaderboard_top3', [])
        }
        
        # Save summary as JSON
        import json
        summary_path = os.path.join(dataset_output_dir, f"{dataset_name}_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Results saved to {summary_path}")
        logger.info(f"Total processing time: {format_time(total_time)}")
        
        return summary
        
    except Exception as e:
        logger.error(f"Error processing dataset {dataset_name}: {e}", exc_info=True)
        return None

def main():
    """Main function to process all datasets"""
    # Parse command-line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Run TPOT preprocessing + AutoGluon experiment')
    parser.add_argument('--no-cache', action='store_true', help='Disable using cached embeddings and processed data')
    parser.add_argument('--datasets', nargs='+', default=['anli_r1_noisy', 'casehold_imbalanced'],
                       help='List of datasets to process (default: anli_r1_noisy casehold_imbalanced)')
    args = parser.parse_args()
    
    use_cache = not args.no_cache
    
    overall_start_time = time.time()
    logger.info("Starting TPOT preprocessing + AutoGluon experiment")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Cache mode: {'Enabled' if use_cache else 'Disabled'}")
    logger.info(f"Processing datasets: {', '.join(args.datasets)}")
    
    # Define all available datasets
    all_datasets = {
        'anli_r1_noisy': {
            'name': 'anli_r1_noisy',
            'path': os.path.join(BASE_DIR, 'anli_r1_noisy'),
            'label_column': 'label'
        },
        'casehold_imbalanced': {
            'name': 'casehold_imbalanced',
            'path': os.path.join(BASE_DIR, 'casehold_imbalanced'),
            'label_column': 'label'
        },
        'anli_r1_full': {
            'name': 'anli_r1_full',
            'path': os.path.join(BASE_DIR, 'anli_r1_full'),
            'label_column': 'label'
        },
        'casehold_full': {
            'name': 'casehold_full',
            'path': os.path.join(BASE_DIR, 'casehold_full'),
            'label_column': 'label'
        }
    }
    
    # Filter datasets based on command-line arguments
    datasets = [all_datasets[name] for name in args.datasets if name in all_datasets]
    
    if not datasets:
        logger.error(f"No valid datasets specified. Available datasets: {list(all_datasets.keys())}")
        return
    
    # Process each dataset
    results = []
    dataset_times = {}
    
    for dataset in datasets:
        dataset_start_time = time.time()
        logger.info(f"\n\n{'='*40}\nProcessing {dataset['name']}\n{'='*40}")
        
        result = process_dataset(
            dataset['name'],
            dataset['path'],
            dataset['label_column'],
            use_cache=use_cache
        )
        
        dataset_time = time.time() - dataset_start_time
        dataset_times[dataset['name']] = dataset_time
        logger.info(f"Total time for {dataset['name']}: {format_time(dataset_time)}")
        
        if result:
            results.append(result)
    
    # Calculate and log overall experiment time
    overall_time = time.time() - overall_start_time
    
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    
    for dataset_name, dataset_time in dataset_times.items():
        logger.info(f"{dataset_name}: {format_time(dataset_time)}")
    
    logger.info(f"Total experiment time: {format_time(overall_time)}")
    
    # Save overall results
    if results:
        import json
        overall_results_path = os.path.join(OUTPUT_DIR, f"overall_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Add timing information to results
        for result in results:
            result['experiment_timing'] = {
                'dataset_processing_time': dataset_times.get(result['dataset'], 0),
                'overall_experiment_time': overall_time
            }
        
        with open(overall_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Overall results saved to {overall_results_path}")
    
    logger.info(f"Experiment completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
