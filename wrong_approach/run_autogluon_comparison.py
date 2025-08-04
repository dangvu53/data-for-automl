import os
import pandas as pd
import numpy as np
import time
import warnings
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from PIL import Image
import logging
import glob
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("autogluon_comparison.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoGluonComparisonTrainer:
    """Class to compare AutoGluon performance on datasets before and after preprocessing"""
    
    def __init__(self, output_dir=None, time_limit=600, presets="medium_quality"):
        """Initialize the trainer"""
        # Set output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or os.path.join("autogluon_comparison_results", f"run_{timestamp}")
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
        
        logger.info(f"Initialized AutoGluonComparisonTrainer")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Time limit per model: {self.time_limit} seconds")
        logger.info(f"Using preset: {self.presets}")
    
    def load_dataset(self, dataset_path, dataset_name, variant, target_col=None):
        """Load dataset from CSV files"""
        logger.info(f"Loading {variant} dataset from {dataset_path}")
        
        # Define file paths
        train_path = os.path.join(dataset_path, f"{variant}_train.csv")
        val_path = os.path.join(dataset_path, f"{variant}_val.csv") 
        test_path = os.path.join(dataset_path, f"{variant}_test.csv")
        
        # Check if files exist
        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            logger.error(f"Dataset files not found at {dataset_path} for variant {variant}")
            return None, None, None, None
        
        # Load data
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            # Load validation set if available, otherwise split train set
            if os.path.exists(val_path):
                val_df = pd.read_csv(val_path)
            else:
                logger.info(f"No validation set found, using 20% of training data as validation")
                from sklearn.model_selection import train_test_split
                train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)
            
            # Identify target column if not provided
            if target_col is None:
                # For original datasets (before preprocessing)
                if variant == "train_subset":
                    # Common target columns
                    possible_targets = ['label', 'class', 'target', 'y']
                    for col in possible_targets:
                        if col in train_df.columns:
                            target_col = col
                            break
                    
                    # If still not found, assume it's the last column
                    if target_col is None:
                        target_col = train_df.columns[-1]
                        logger.warning(f"Target column not specified, using last column: {target_col}")
                
                # For after_* datasets (after preprocessing)
                elif variant == "after":
                    # Typically for preprocessed datasets, the target is often the last column
                    target_col = train_df.columns[-1]
                    logger.info(f"Target column for preprocessed data assumed to be: {target_col}")
            
            logger.info(f"Loaded {dataset_name} {variant} dataset:")
            logger.info(f"  Training samples: {len(train_df)}")
            logger.info(f"  Validation samples: {len(val_df)}")
            logger.info(f"  Testing samples: {len(test_df)}")
            logger.info(f"  Features: {len(train_df.columns) - 1}")
            logger.info(f"  Target column: {target_col}")
            
            return train_df, val_df, test_df, target_col
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {str(e)}")
            return None, None, None, None
    
    def train_model(self, train_df, val_df, target_col, dataset_name, variant, problem_type=None):
        """Train appropriate AutoGluon model based on dataset type"""
        logger.info(f"Training AutoGluon model for {dataset_name} ({variant})...")

        start_time = time.time()

        # Create output directory for this dataset variant
        model_dir = os.path.join(self.models_dir, f"{dataset_name}_{variant}_model")
        os.makedirs(model_dir, exist_ok=True)

        # Check if dataset contains image data
        has_image_column = False
        if 'image' in train_df.columns:
            # Check if the image column contains PIL Images or paths
            sample = train_df['image'].iloc[0]
            if isinstance(sample, str) and (sample.startswith('{') and sample.endswith('}')):
                try:
                    # This might be a serialized image
                    has_image_column = True
                    logger.info("Dataset contains image data, using MultiModalPredictor")
                except:
                    has_image_column = False
        
        # For text datasets (ANLI-R1, CaseHOLD)
        has_text = False
        text_columns = ['premise', 'hypothesis', 'text', 'context', 'query', 'question']
        for col in text_columns:
            if col in train_df.columns:
                has_text = True
                logger.info(f"Found text column: {col}, using TabularPredictor with text features")
                break

        # Choose model type based on dataset characteristics
        if has_image_column:
            return self.train_multimodal_model(train_df, val_df, target_col, dataset_name, variant, model_dir, start_time)
        elif has_text:
            return self.train_text_model(train_df, val_df, target_col, dataset_name, variant, model_dir, start_time)
        else:
            return self.train_tabular_model(train_df, val_df, target_col, dataset_name, variant, model_dir, start_time)

    def train_multimodal_model(self, train_df, val_df, target_col, dataset_name, variant, model_dir, start_time):
        """Train MultiModalPredictor for multimodal data"""
        logger.info(f"Using MultiModalPredictor for {dataset_name} ({variant})")

        try:
            # Create MultiModalPredictor
            predictor = MultiModalPredictor(
                label=target_col,
                path=model_dir
            )

            # Combine train and validation data for MultiModalPredictor
            # MultiModalPredictor handles train/val split internally
            combined_train_df = pd.concat([train_df, val_df], ignore_index=True)

            logger.info(f"Training MultiModalPredictor with {len(combined_train_df)} samples")
            logger.info(f"Columns: {combined_train_df.columns.tolist()}")

            # Train MultiModalPredictor with AutoGluon's settings
            predictor.fit(
                train_data=combined_train_df,
                time_limit=self.time_limit,
                presets=self.presets
            )

            training_time = time.time() - start_time
            logger.info(f"MultiModalPredictor training completed for {dataset_name} ({variant}) in {training_time:.2f} seconds")
            return predictor, training_time

        except Exception as e:
            logger.error(f"MultiModalPredictor failed: {e}")
            logger.warning("Falling back to TabularPredictor without images")

            # Prepare data for TabularPredictor (remove image column)
            train_df_tabular = train_df.copy()
            val_df_tabular = val_df.copy()

            # Remove image column for TabularPredictor as it can't handle PIL Images
            if 'image' in train_df_tabular.columns:
                logger.info("Removing image column for TabularPredictor fallback")
                train_df_tabular = train_df_tabular.drop(columns=['image'])
                val_df_tabular = val_df_tabular.drop(columns=['image'])

            return self.train_tabular_model(train_df_tabular, val_df_tabular, target_col, dataset_name, variant, model_dir, start_time)

    def train_text_model(self, train_df, val_df, target_col, dataset_name, variant, model_dir, start_time):
        """Train TabularPredictor for text data using AutoGluon defaults"""
        logger.info(f"Using TabularPredictor for text data for {dataset_name} ({variant})")

        # Determine problem type
        unique_targets = train_df[target_col].nunique()
        if unique_targets <= 10:  # Classification
            problem_type = 'multiclass' if unique_targets > 2 else 'binary'
            eval_metric = 'accuracy'
        else:  # Regression
            problem_type = 'regression'
            eval_metric = 'root_mean_squared_error'

        predictor = TabularPredictor(
            label=target_col,
            path=model_dir,
            problem_type=problem_type,
            eval_metric=eval_metric
        )

        # Use AutoGluon's default settings
        logger.info(f"Training TabularPredictor with problem_type={problem_type}, eval_metric={eval_metric}")
        
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
        logger.info(f"Text model training completed for {dataset_name} ({variant}) in {training_time:.2f} seconds")
        return predictor, training_time

    def train_tabular_model(self, train_df, val_df, target_col, dataset_name, variant, model_dir, start_time):
        """Train standard TabularPredictor using AutoGluon defaults"""
        logger.info(f"Using TabularPredictor for {dataset_name} ({variant})")

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
        logger.info(f"Tabular model training completed for {dataset_name} ({variant}) in {training_time:.2f} seconds")
        return predictor, training_time

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

    def evaluate_model(self, predictor, test_df, target_col, dataset_name, variant):
        """Evaluate the trained model with detailed metrics"""
        logger.info(f"Evaluating model for {dataset_name} ({variant})...")

        try:
            # Get model information (size, parameters)
            model_info = self.get_model_info(predictor, dataset_name, variant)

            # Save predictions and create confusion matrix first
            predictions, confusion_matrix = self.save_predictions_and_confusion_matrix(
                predictor, test_df, target_col, dataset_name, variant
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

    def run_comparison_for_dataset(self, dataset_path, dataset_name, target_col=None):
        """Run comparison between original and preprocessed version of a dataset"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*60}")
        
        dataset_results = {}
        
        # Define variants to process
        variants = ['train_subset', 'after']
        
        for variant in variants:
            variant_label = 'before' if variant == 'train_subset' else 'after'
            logger.info(f"\nProcessing {variant_label} preprocessing variant...")
            
            # Load dataset
            train_df, val_df, test_df, determined_target = self.load_dataset(
                dataset_path, dataset_name, variant, target_col
            )
            
            if train_df is None:
                logger.warning(f"Skipping {dataset_name} {variant_label} preprocessing due to loading error")
                continue
            
            # Use determined target if none was provided
            if target_col is None and determined_target is not None:
                target_col = determined_target
            
            # Train model
            predictor, training_time = self.train_model(
                train_df, val_df, target_col, dataset_name, variant_label
            )
            
            # Evaluate model
            results = self.evaluate_model(predictor, test_df, target_col, dataset_name, variant_label)
            results['total_training_time'] = training_time
            results['train_size'] = len(train_df)
            results['val_size'] = len(val_df)
            results['test_size'] = len(test_df)
            results['feature_count'] = len(train_df.columns) - 1
            
            # Store results
            dataset_results[variant_label] = results
        
        # Compare results between before and after preprocessing
        if 'before' in dataset_results and 'after' in dataset_results:
            logger.info(f"\nComparing results for {dataset_name}:")
            self.compare_results(dataset_results['before'], dataset_results['after'])
        
        return dataset_results
    
    def compare_results(self, before_results, after_results):
        """Compare results between before and after preprocessing"""
        comparison = {
            'dataset': before_results['dataset'],
            'before_accuracy': before_results['accuracy'],
            'after_accuracy': after_results['accuracy'],
            'before_test_score': before_results['test_score'],
            'after_test_score': after_results['test_score'],
            'before_train_size': before_results['train_size'],
            'after_train_size': after_results['train_size'],
            'before_feature_count': before_results['feature_count'],
            'after_feature_count': after_results['feature_count'],
            'before_best_model': before_results['best_model'],
            'after_best_model': after_results['best_model'],
            'before_training_time': before_results['total_training_time'],
            'after_training_time': after_results['total_training_time'],
        }
        
        # Calculate differences
        try:
            if isinstance(before_results['accuracy'], (int, float)) and isinstance(after_results['accuracy'], (int, float)):
                comparison['accuracy_diff'] = after_results['accuracy'] - before_results['accuracy']
                comparison['accuracy_diff_percent'] = (comparison['accuracy_diff'] / before_results['accuracy']) * 100 if before_results['accuracy'] > 0 else float('inf')
            else:
                comparison['accuracy_diff'] = 'N/A'
                comparison['accuracy_diff_percent'] = 'N/A'
        except:
            comparison['accuracy_diff'] = 'N/A'
            comparison['accuracy_diff_percent'] = 'N/A'
            
        # Log comparison
        logger.info(f"Comparison for {comparison['dataset']}:")
        logger.info(f"  Accuracy before: {comparison['before_accuracy']}")
        logger.info(f"  Accuracy after: {comparison['after_accuracy']}")
        logger.info(f"  Accuracy difference: {comparison['accuracy_diff']}")
        logger.info(f"  Best model before: {comparison['before_best_model']}")
        logger.info(f"  Best model after: {comparison['after_best_model']}")
        logger.info(f"  Training time before: {comparison['before_training_time']:.2f} seconds")
        logger.info(f"  Training time after: {comparison['after_training_time']:.2f} seconds")
        logger.info(f"  Train size before: {comparison['before_train_size']}")
        logger.info(f"  Train size after: {comparison['after_train_size']}")
        logger.info(f"  Feature count before: {comparison['before_feature_count']}")
        logger.info(f"  Feature count after: {comparison['after_feature_count']}")
        
        return comparison
    
    def run_all_comparisons(self, datasets_config):
        """Run comparison for all specified datasets"""
        all_results = {}
        all_comparisons = []
        
        for dataset_info in datasets_config:
            dataset_name = dataset_info['name']
            dataset_path = dataset_info['path']
            target_col = dataset_info.get('target_col', None)
            
            results = self.run_comparison_for_dataset(dataset_path, dataset_name, target_col)
            all_results[dataset_name] = results
            
            # Create comparison if both variants were processed
            if 'before' in results and 'after' in results:
                comparison = self.compare_results(results['before'], results['after'])
                all_comparisons.append(comparison)
        
        # Save all results and comparisons
        self.save_results_summary(all_results, all_comparisons)
        
        return all_results, all_comparisons
    
    def save_results_summary(self, all_results, all_comparisons):
        """Save experiment results to CSV and JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save all detailed results as JSON
        results_file_json = os.path.join(self.output_dir, f"detailed_results_{timestamp}.json")
        with open(results_file_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Detailed results saved to: {results_file_json}")
        
        # Save comparisons as CSV
        if all_comparisons:
            comparisons_df = pd.DataFrame(all_comparisons)
            comparisons_file = os.path.join(self.output_dir, f"comparisons_{timestamp}.csv")
            comparisons_df.to_csv(comparisons_file, index=False)
            
            logger.info(f"Comparisons saved to: {comparisons_file}")
            
            # Print summary table
            print("\n" + "="*80)
            print("COMPARISON SUMMARY")
            print("="*80)
            print(comparisons_df.to_string(index=False))
            print("="*80)
        else:
            logger.warning("No comparisons to save")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Run AutoGluon comparison on datasets before and after preprocessing")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--time_limit", type=int, default=3000, help="Time limit in seconds for each model training")
    parser.add_argument("--presets", type=str, default="medium_quality", help="AutoGluon preset quality level")
    parser.add_argument("--datasets", type=str, nargs='+', default=["anli_r1_noisy", "casehold_imbalanced", "scienceqa_outlier"], 
                        help="List of dataset names to process (default: all)")
    
    args = parser.parse_args()
    
    print("AutoGluon Comparison Tool")
    print("="*50)
    
    # Define dataset configurations
    all_datasets = {
        # "anli_r1_noisy": {
        #     "name": "anli_r1",
        #     "path": "/storage/nammt/autogluon/anli_r1_noisy",
        #     "target_col": "label"
        # },
        # "casehold_imbalanced": {
        #     "name": "casehold",
        #     "path": "/storage/nammt/autogluon/casehold_imbalanced",
        #     "target_col": "label"
        # },
        #  "anli_r1_full": {
        #     "name": "anli_r1_full",
        #     "path": "/storage/nammt/autogluon/anli_r1_full",
        #     "target_col": "label"
        # },
        "casehold_full": {
            "name": "casehold_full",
            "path": "/storage/nammt/autogluon/casehold_full",
            "target_col": "label"
        },
        # "scienceqa_outlier": {
        #     "name": "scienceqa",
        #     "path": "/storage/nammt/autogluon/scienceqa_outlier",
        #     "target_col": "answer"
        # }
    }
    
    # Filter datasets to process
    datasets_to_process = []
    for dataset_name in args.datasets:
        if dataset_name in all_datasets:
            datasets_to_process.append(all_datasets[dataset_name])
        else:
            logger.warning(f"Dataset {dataset_name} not found in configuration, skipping")
    
    if not datasets_to_process:
        logger.error("No valid datasets to process!")
        return
    
    # Create trainer and run comparisons
    trainer = AutoGluonComparisonTrainer(
        output_dir=args.output_dir,
        time_limit=args.time_limit,
        presets=args.presets
    )
    
    all_results, all_comparisons = trainer.run_all_comparisons(datasets_to_process)
    
    print(f"\nCompleted comparison on {len(datasets_to_process)} datasets")
    print("Check the output directory for:")
    print("  - Detailed results in JSON format")
    print("  - Comparison CSV file")
    print("  - Predictions CSV files in 'predictions/' subdirectory")
    print("  - Confusion matrices in 'confusion_matrices/' subdirectory")
    print("  - Trained models in 'models/' subdirectory")

if __name__ == "__main__":
    main()
