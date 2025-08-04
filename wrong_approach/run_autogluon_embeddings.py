import os
import pandas as pd
import numpy as np
import time
import joblib
import argparse
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from autogluon.tabular import TabularPredictor
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autogluon_embeddings_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoGluonEmbeddingsEvaluator:
    """Class to run AutoGluon on embeddings data extracted from joblib files"""
    
    def __init__(self, output_dir=None, time_limit=3000, presets="medium_quality"):
        """Initialize the evaluator"""
        # Set output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or os.path.join("autogluon_embeddings_results", f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create subdirectories for outputs
        self.predictions_dir = os.path.join(self.output_dir, "predictions")
        self.confusion_matrices_dir = os.path.join(self.output_dir, "confusion_matrices")
        self.models_dir = os.path.join(self.output_dir, "models")
        
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.confusion_matrices_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Training parameters
        self.time_limit = time_limit
        self.presets = presets
        self.results = {}
        
        logger.info(f"Initialized AutoGluonEmbeddingsEvaluator")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Time limit per model: {self.time_limit} seconds")
        logger.info(f"Using preset: {self.presets}")
    
    def load_embeddings(self, embeddings_path, labels_path=None):
        """Load embeddings data from joblib file"""
        logger.info(f"Loading embeddings from {embeddings_path}")
        
        try:
            # Try to load the file
            embeddings_data = joblib.load(embeddings_path)
            
            # Debug: Directly inspect the loaded data structure
            logger.info(f"DEBUG: Loaded data type: {type(embeddings_data)}")
            if isinstance(embeddings_data, tuple):
                logger.info(f"DEBUG: Tuple length: {len(embeddings_data)}")
                for i, item in enumerate(embeddings_data):
                    logger.info(f"DEBUG: Item {i} type: {type(item)}, shape: {getattr(item, 'shape', None)}")
            elif isinstance(embeddings_data, dict):
                logger.info(f"DEBUG: Dictionary keys: {embeddings_data.keys()}")
            elif isinstance(embeddings_data, np.ndarray):
                logger.info(f"DEBUG: Array shape: {embeddings_data.shape}, dtype: {embeddings_data.dtype}")
            
            # Check different possible formats
            if isinstance(embeddings_data, tuple):
                # Case 1: tuple with 3 elements (X_train, X_val, X_test)
                if len(embeddings_data) == 3:
                    logger.info("Successfully loaded embeddings as (X_train, X_val, X_test) tuple")
                    X_train, X_val, X_test = embeddings_data
                    
                    # Check if any of the sets is None
                    if X_train is None or X_val is None or X_test is None:
                        logger.warning("One or more embedding sets is None")
                    
                    # Log shapes of the loaded data
                    logger.info(f"Train embeddings shape: {X_train.shape if X_train is not None else 'None'}")
                    logger.info(f"Val embeddings shape: {X_val.shape if X_val is not None else 'None'}")
                    logger.info(f"Test embeddings shape: {X_test.shape if X_test is not None else 'None'}")
                    
                    # DEBUG: Check what the embedding data actually is
                    logger.info(f"DEBUG: Train embeddings type: {type(X_train)}")
                    
                    # Inspect deeper
                    try:
                        # Check for structured array
                        if hasattr(X_train, 'dtype') and hasattr(X_train.dtype, 'names') and X_train.dtype.names:
                            logger.info(f"DEBUG: Structured array with fields: {X_train.dtype.names}")
                            # Check for label fields
                            label_fields = [f for f in X_train.dtype.names if f and ('label' in f.lower() or 'class' in f.lower() or 'target' in f.lower())]
                            if label_fields:
                                logger.info(f"DEBUG: Potential label fields: {label_fields}")
                        # Check for DataFrame
                        elif isinstance(X_train, pd.DataFrame):
                            logger.info(f"DEBUG: DataFrame with columns: {X_train.columns.tolist()}")
                            # Check for label columns
                            label_cols = [c for c in X_train.columns if 'label' in c.lower() or 'class' in c.lower() or 'target' in c.lower()]
                            if label_cols:
                                logger.info(f"DEBUG: Potential label columns: {label_cols}")
                        # Check array size
                        elif isinstance(X_train, np.ndarray):
                            logger.info(f"DEBUG: NumPy array with shape {X_train.shape}")
                            # Sample of values
                            logger.info(f"DEBUG: First row sample: {X_train[0][:5]}")
                            
                            # More detailed analysis of the array
                        logger.info(f"DEBUG: Full shape details - X_train: {X_train.shape}")
                        if X_val is not None:
                            logger.info(f"DEBUG: X_val shape: {X_val.shape}")
                        if X_test is not None:
                            logger.info(f"DEBUG: X_test shape: {X_test.shape}")
                        
                        # Try to find where the labels might be
                        # Option 1: Labels might be in the last column
                        last_column = X_train[:, -1]
                        unique_values = np.unique(last_column)
                        
                        # Check if the last column looks like class labels (few unique values or all integers)
                        is_likely_label = len(unique_values) < 100 or np.all(np.equal(np.mod(last_column, 1), 0))
                        
                        logger.info(f"DEBUG: Last column sample values: {last_column[:10]}")
                        logger.info(f"DEBUG: Unique values in last column: {len(unique_values)} (first 10: {unique_values[:10]})")
                        logger.info(f"DEBUG: Is likely label column: {is_likely_label}")
                        
                        # For CaseHOLD dataset, the expected labels should be 0-4
                        if len(unique_values) <= 10:
                            logger.info(f"DEBUG: All unique values in last column: {sorted(unique_values.tolist())}")
                        
                        if X_train.shape[1] > 384 or is_likely_label:
                            logger.info(f"DEBUG: Array has {X_train.shape[1]} columns (expected embedding dim: 384)")
                            logger.info(f"DEBUG: Last column likely contains labels (will extract later)")
                    except Exception as e:
                        logger.info(f"DEBUG: Error during inspection: {e}")
                    else:
                        logger.info(f"DEBUG: Train embeddings type: {type(X_train)}, no label fields detected")
                    
                # Case 2: tuple with 6 elements (X_train, y_train, X_val, y_val, X_test, y_test)
                elif len(embeddings_data) == 6:
                    logger.info("Successfully loaded embeddings and labels from a 6-tuple format")
                    X_train, y_train, X_val, y_val, X_test, y_test = embeddings_data
                    
                    logger.info(f"Train embeddings shape: {X_train.shape if X_train is not None else 'None'}")
                    logger.info(f"Train labels shape: {y_train.shape if hasattr(y_train, 'shape') else 'length ' + str(len(y_train)) if y_train is not None else 'None'}")
                    logger.info(f"Val embeddings shape: {X_val.shape if X_val is not None else 'None'}")
                    logger.info(f"Test embeddings shape: {X_test.shape if X_test is not None else 'None'}")
                    
                    return X_train, X_val, X_test, y_train, y_val, y_test
                
                else:
                    logger.warning(f"Embeddings file contains a tuple with {len(embeddings_data)} elements, expected 3 or 6")
                
            # Case 3: dictionary with embeddings and labels
            elif isinstance(embeddings_data, dict):
                logger.info("Loaded embeddings from dictionary format")
                X_train = embeddings_data.get('X_train') or embeddings_data.get('embeddings_train')
                X_val = embeddings_data.get('X_val') or embeddings_data.get('embeddings_val')
                X_test = embeddings_data.get('X_test') or embeddings_data.get('embeddings_test')
                y_train = embeddings_data.get('y_train') or embeddings_data.get('labels_train')
                y_val = embeddings_data.get('y_val') or embeddings_data.get('labels_val')
                y_test = embeddings_data.get('y_test') or embeddings_data.get('labels_test')
                
                if X_train is not None:
                    logger.info(f"Train embeddings shape: {X_train.shape}")
                    if y_train is not None:
                        logger.info(f"Train labels shape: {y_train.shape if hasattr(y_train, 'shape') else 'length ' + str(len(y_train))}")
                        return X_train, X_val, X_test, y_train, y_val, y_test
                
            # If we're here and it's a tuple with 3 elements, we need to check for separate labels
            if isinstance(embeddings_data, tuple) and len(embeddings_data) == 3:
                X_train, X_val, X_test = embeddings_data
                
                # Check if embeddings include labels as the last column
                y_train, y_val, y_test = None, None, None
                
                # If X_train has more than 384 columns (expected embedding dim), assume last column is label
                if isinstance(X_train, np.ndarray) and X_train.shape[1] >= 384:
                    # Check if the last column looks like a label (examine values)
                    last_column = X_train[:, -1]
                    unique_values = np.unique(last_column)
                    
                    # Check if the last column looks like it contains class labels
                    # Criteria: Either few unique values (<100) or all values are integers
                    is_likely_label = len(unique_values) < 100 or np.all(np.equal(np.mod(last_column, 1), 0))
                    
                    # Debug: show sample of last column values
                    logger.info(f"DEBUG: Last column sample values: {last_column[:10]}")
                    logger.info(f"DEBUG: Unique values in last column: {len(unique_values)} (first 10: {unique_values[:10]})")
                    logger.info(f"DEBUG: Is likely label column: {is_likely_label}")
                    
                    if is_likely_label or X_train.shape[1] > 384:
                        logger.info(f"Detected potential label column in embeddings (shape: {X_train.shape}), using last column as labels")
                        
                        # Extract labels from last column
                        y_train = X_train[:, -1].copy()
                        y_val = X_val[:, -1].copy() if X_val is not None else None
                        y_test = X_test[:, -1].copy() if X_test is not None else None
                        
                        # Remove label column from features
                        X_train = X_train[:, :-1]
                        X_val = X_val[:, :-1] if X_val is not None else None
                        X_test = X_test[:, :-1] if X_test is not None else None
                        
                        logger.info(f"Extracted labels. New shapes: X_train {X_train.shape}, y_train {y_train.shape}")
                        return X_train, X_val, X_test, y_train, y_val, y_test
                
                # If no labels found in embeddings, check for separate labels file
                if labels_path and os.path.exists(labels_path):
                    try:
                        labels_data = joblib.load(labels_path)
                        if isinstance(labels_data, tuple) and len(labels_data) == 3:
                            y_train, y_val, y_test = labels_data
                            logger.info("Successfully loaded labels from separate file")
                        else:
                            logger.warning("Labels file doesn't contain a tuple of 3 elements")
                    except Exception as e:
                        logger.error(f"Error loading labels: {e}")
                
                return X_train, X_val, X_test, y_train, y_val, y_test
                
            else:
                # It might be a single array or some other format
                logger.warning("Embeddings file doesn't contain a tuple of 3 elements")
                logger.info(f"Embeddings type: {type(embeddings_data)}")
                
                if isinstance(embeddings_data, np.ndarray):
                    logger.info(f"Loaded a single array with shape: {embeddings_data.shape}")
                    
                    # DEBUG: Check if the array has a structured dtype with potential label field
                    if hasattr(embeddings_data, 'dtype') and hasattr(embeddings_data.dtype, 'names'):
                        logger.info(f"DEBUG: Embeddings array is structured with fields: {embeddings_data.dtype.names}")
                        # Look for label-like fields
                        label_candidates = [f for f in embeddings_data.dtype.names if 'label' in f.lower() or 'class' in f.lower() or 'target' in f.lower()]
                        if label_candidates:
                            logger.info(f"DEBUG: Potential label fields found in array: {label_candidates}")
                            
                            # Extract features and labels
                            feature_fields = [f for f in embeddings_data.dtype.names if f not in label_candidates]
                            label_field = label_candidates[0]  # Use the first label field
                            
                            logger.info(f"DEBUG: Using '{label_field}' as label and these as features: {feature_fields}")
                            
                            # Extract features and labels
                            X = embeddings_data[feature_fields]
                            y = embeddings_data[label_field]
                            
                            # Split into train/val/test
                            train_size = int(0.7 * len(embeddings_data))
                            val_size = int(0.15 * len(embeddings_data))
                            
                            X_train = X[:train_size]
                            X_val = X[train_size:train_size+val_size]
                            X_test = X[train_size+val_size:]
                            
                            y_train = y[:train_size]
                            y_val = y[train_size:train_size+val_size]
                            y_test = y[train_size+val_size:]
                            
                            logger.info(f"Split into train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
                            logger.info(f"With labels shapes: train: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}, val: {y_val.shape if hasattr(y_val, 'shape') else len(y_val)}, test: {y_test.shape if hasattr(y_test, 'shape') else len(y_test)}")
                            
                            return X_train, X_val, X_test, y_train, y_val, y_test
                    
                    # Regular numpy array - check for label column first
                    # Check if the last column looks like it contains class labels
                    last_column = embeddings_data[:, -1]
                    unique_values = np.unique(last_column)
                    is_likely_label = len(unique_values) < 100 or np.all(np.equal(np.mod(last_column, 1), 0))
                    
                    logger.info(f"DEBUG: Last column sample values: {last_column[:10]}")
                    logger.info(f"DEBUG: Unique values in last column: {len(unique_values)} (first 10: {unique_values[:10]})")
                    logger.info(f"DEBUG: Is likely label column: {is_likely_label}")
                    
                    if is_likely_label or embeddings_data.shape[1] > 384:
                        logger.info(f"Single array with {embeddings_data.shape[1]} columns detected, using last column as label")
                        
                        # Extract features and labels
                        X = embeddings_data[:, :-1]  # All columns except last
                        y = embeddings_data[:, -1]   # Only last column
                        
                        # Split into train/val/test
                        train_size = int(0.7 * len(embeddings_data))
                        val_size = int(0.15 * len(embeddings_data))
                        
                        X_train = X[:train_size]
                        X_val = X[train_size:train_size+val_size]
                        X_test = X[train_size+val_size:]
                        
                        y_train = y[:train_size]
                        y_val = y[train_size:train_size+val_size]
                        y_test = y[train_size+val_size:]
                        
                        logger.info(f"Split into train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
                        logger.info(f"With labels shapes: train: {y_train.shape}, val: {y_val.shape}, test: {y_test.shape}")
                        return X_train, X_val, X_test, y_train, y_val, y_test
                    else:
                        # No label column detected - split into train/val/test without labels
                        train_size = int(0.7 * len(embeddings_data))
                        val_size = int(0.15 * len(embeddings_data))
                        
                        X_train = embeddings_data[:train_size]
                        X_val = embeddings_data[train_size:train_size+val_size]
                        X_test = embeddings_data[train_size+val_size:]
                        
                        logger.info(f"Split into train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")
                        return X_train, X_val, X_test, None, None, None
                else:
                    logger.error("Unsupported embeddings format")
                    return None, None, None, None, None, None
                
        except Exception as e:
            logger.error(f"Error loading embeddings: {e}")
            return None, None, None, None, None, None
    
    def prepare_dataframes(self, X_train, X_val, X_test, y_train=None, y_val=None, y_test=None, target_col='label'):
        """Convert numpy arrays to pandas DataFrames for AutoGluon"""
        logger.info("Preparing DataFrames for AutoGluon")
        
        # Create column names for the embeddings
        feature_cols = [f'feature_{i}' for i in range(X_train.shape[1])]
        
        # Create DataFrames
        train_df = pd.DataFrame(X_train, columns=feature_cols)
        val_df = pd.DataFrame(X_val, columns=feature_cols)
        test_df = pd.DataFrame(X_test, columns=feature_cols)
        
        # Add target column if labels are provided
        if y_train is not None:
            train_df[target_col] = y_train
        if y_val is not None:
            val_df[target_col] = y_val
        if y_test is not None:
            test_df[target_col] = y_test
        
        logger.info(f"Created DataFrames with shapes: train={train_df.shape}, val={val_df.shape}, test={test_df.shape}")
        return train_df, val_df, test_df
    
    def generate_synthetic_labels(self, X_train, X_val, X_test, num_classes=2, target_col='label'):
        """Generate synthetic labels for cases when labels are not provided"""
        logger.warning("Generating synthetic labels as no labels were provided")
        
        # Use K-means to generate cluster labels
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_classes, random_state=42, n_init=10)
        kmeans.fit(X_train)
        
        y_train = kmeans.predict(X_train)
        y_val = kmeans.predict(X_val)
        y_test = kmeans.predict(X_test)
        
        logger.info(f"Generated synthetic labels with {num_classes} classes")
        return y_train, y_val, y_test
    
    def train_model(self, train_df, val_df, target_col, dataset_name, variant="embeddings"):
        """Train AutoGluon TabularPredictor on embeddings data"""
        logger.info(f"Training AutoGluon model for {dataset_name} ({variant})...")

        start_time = time.time()

        # Create output directory for this dataset variant
        model_dir = os.path.join(self.models_dir, f"{dataset_name}_{variant}_model")
        os.makedirs(model_dir, exist_ok=True)

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
            presets=self.presets,
            time_limit=self.time_limit,
            verbosity=2,
            # Allow AutoGluon to use more memory
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
        )

        training_time = time.time() - start_time
        logger.info(f"Model training completed for {dataset_name} ({variant}) in {training_time:.2f} seconds")
        return predictor, training_time
    
    def evaluate_model(self, predictor, test_df, target_col, dataset_name, variant="embeddings"):
        """Evaluate the trained model with detailed metrics"""
        logger.info(f"Evaluating model for {dataset_name} ({variant})...")

        try:
            # Get model information (size, parameters)
            model_info = self.get_model_info(predictor, dataset_name, variant)

            # Save predictions and create confusion matrix first
            predictions, confusion_matrix = self.save_predictions_and_confusion_matrix(
                predictor, test_df, target_col, dataset_name, variant
            )

            # Evaluate with test set
            test_features = test_df.drop(columns=[target_col])
            predictions = predictor.predict(test_features)

            # Handle different evaluation APIs
            try:
                performance = predictor.evaluate(test_df, silent=True)
            except TypeError:
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
                logger.info(f"Leaderboard not available: {e}")
                best_model = 'Unknown'
                best_val_score = 'N/A'
                test_score = 'N/A'
                training_times = {}
                leaderboard_top3 = []
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
            
    def get_model_info(self, predictor, dataset_name, variant):
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
            
    def save_predictions_and_confusion_matrix(self, predictor, test_df, target_col, dataset_name, variant):
        """Save predictions as CSV and create confusion matrix for classification tasks"""
        logger.info(f"Saving predictions and creating confusion matrix for {dataset_name} ({variant})...")

        try:
            # Make predictions
            test_features = test_df.drop(columns=[target_col])
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
            pred_file = os.path.join(self.predictions_dir, f"{dataset_name}_{variant}_predictions.csv")
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
                    cm_file = os.path.join(self.confusion_matrices_dir, f"{dataset_name}_{variant}_confusion_matrix.png")
                    plt.savefig(cm_file, dpi=300, bbox_inches='tight')
                    plt.close()
                    logger.info(f"Confusion matrix saved to: {cm_file}")

                    # Save classification report
                    report = classification_report(y_true, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    report_file = os.path.join(self.confusion_matrices_dir, f"{dataset_name}_{variant}_classification_report.csv")
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
            
    def save_results_summary(self, results, dataset_name):
        """Save experiment results to CSV and JSON"""
        import json
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results as JSON
        results_file_json = os.path.join(self.output_dir, f"{dataset_name}_results_{timestamp}.json")
        with open(results_file_json, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file_json}")
        
        # Convert results to DataFrame for CSV
        if isinstance(results, dict):
            results_df = pd.DataFrame([results])
            results_file_csv = os.path.join(self.output_dir, f"{dataset_name}_results_{timestamp}.csv")
            results_df.to_csv(results_file_csv, index=False)
            logger.info(f"Results summary saved to: {results_file_csv}")
    
    def run_evaluation(self, embeddings_path, labels_path=None, dataset_name="embeddings_dataset", 
                       target_col="label", num_classes=None):
        """Run full evaluation on embeddings data"""
        logger.info(f"Starting evaluation for {dataset_name} using embeddings from {embeddings_path}")
        
        # Try to locate label files in the same directory as the embeddings file
        csv_labels = None
        if labels_path is None:
            # Get the directory where the embeddings are stored
            embeddings_dir = os.path.dirname(embeddings_path)
            
            # Look for CSV files that might contain labels
            potential_label_files = [
                os.path.join(embeddings_dir, 'after_train.csv'),
                os.path.join(embeddings_dir, 'train_full.csv'),
                os.path.join(embeddings_dir, 'train_subset.csv'),
                os.path.join(embeddings_dir, 'train.csv')
            ]
            
            for file_path in potential_label_files:
                if os.path.exists(file_path):
                    logger.info(f"Found potential labels file: {file_path}")
                    try:
                        label_df = pd.read_csv(file_path)
                        # Try common label column names
                        label_col = None
                        for col in [target_col, 'label', 'answer', 'class', 'target']:
                            if col in label_df.columns:
                                logger.info(f"Found label column '{col}' in {file_path}")
                                label_col = col
                                break
                        
                        if label_col:
                            # Found a label column, extract labels
                            csv_labels = label_df[label_col].values
                            logger.info(f"Extracted {len(csv_labels)} labels from {file_path}")
                            # Set as explicit labels_path to use later
                            labels_path = file_path
                            break
                    except Exception as e:
                        logger.warning(f"Error reading {file_path}: {e}")
        
        # Load embeddings
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_embeddings(embeddings_path, labels_path)
        
        if X_train is None or X_val is None or X_test is None:
            logger.error("Failed to load embeddings properly. Exiting.")
            return None
        
        # If no labels are provided, generate synthetic ones
        if y_train is None or y_val is None or y_test is None:
            # If num_classes is not provided, default to binary classification (2 classes)
            if num_classes is None:
                logger.warning("No labels found in embeddings and num_classes not specified. Defaulting to binary classification (2 classes).")
                num_classes = 2
                
            y_train, y_val, y_test = self.generate_synthetic_labels(
                X_train, X_val, X_test, num_classes=num_classes, target_col=target_col
            )
        
        # Prepare DataFrames
        train_df, val_df, test_df = self.prepare_dataframes(
            X_train, X_val, X_test, y_train, y_val, y_test, target_col=target_col
        )
        
        # Train model
        predictor, training_time = self.train_model(
            train_df, val_df, target_col, dataset_name
        )
        
        # Evaluate model
        results = self.evaluate_model(predictor, test_df, target_col, dataset_name)
        results['total_training_time'] = training_time
        results['train_size'] = len(train_df)
        results['val_size'] = len(val_df)
        results['test_size'] = len(test_df)
        results['feature_count'] = len(train_df.columns) - 1  # Exclude target column
        
        # Save results
        self.save_results_summary(results, dataset_name)
        
        logger.info(f"Completed evaluation for {dataset_name}")
        logger.info(f"Results: {results}")
        
        return results

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run AutoGluon evaluation on embeddings data from joblib files")
    parser.add_argument("--embeddings_path", type=str, default="/storage/nammt/autogluon/embeddings_full.joblib", required=True, help="Path to embeddings joblib file")
    parser.add_argument("--labels_path", type=str, help="Optional path to labels joblib file")
    parser.add_argument("--dataset_name", type=str, default="embeddings_dataset", help="Name for this dataset")
    parser.add_argument("--target_col", type=str, default="label", help="Name for the target column")
    parser.add_argument("--num_classes", type=int, help="Number of classes (defaults to 2 if no labels found)")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    parser.add_argument("--time_limit", type=int, default=3000, help="Time limit in seconds for model training")
    parser.add_argument("--presets", type=str, default="medium_quality", help="AutoGluon preset quality level")
    
    args = parser.parse_args()
    
    print("AutoGluon Embeddings Evaluation Tool")
    print("="*50)
    
    # Check if embeddings path exists
    if not os.path.exists(args.embeddings_path):
        print(f"Error: Embeddings file {args.embeddings_path} not found!")
        return
    
    # Check if labels path exists if provided
    if args.labels_path and not os.path.exists(args.labels_path):
        print(f"Warning: Labels file {args.labels_path} not found! Will generate synthetic labels.")
        args.labels_path = None
    
    # If no labels path provided, but we don't need it since we auto-detect labels from embeddings
    if args.labels_path is None and args.num_classes is None:
        print("Note: No labels path or num_classes specified. Will try to auto-detect labels from embeddings.")
        print("      If no labels are found in embeddings, will default to binary classification (2 classes).")
        args.num_classes = 2  # Default to binary classification if needed
    
    # Create evaluator and run
    evaluator = AutoGluonEmbeddingsEvaluator(
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        presets=args.presets
    )
    
    results = evaluator.run_evaluation(
        embeddings_path=args.embeddings_path,
        labels_path=args.labels_path,
        dataset_name=args.dataset_name,
        target_col=args.target_col,
        num_classes=args.num_classes
    )
    
    if results:
        print("\nEvaluation completed successfully!")
        print(f"Check the output directory for details: {evaluator.output_dir}")
    else:
        print("\nEvaluation failed. Check the log file for details.")

if __name__ == "__main__":
    main()
