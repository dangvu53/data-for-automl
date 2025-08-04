#!/usr/bin/env python3
"""
Preprocessing Integration for AutoGluon

This script provides a CLI interface to preprocess datasets using various techniques:
1. Outlier Detection
2. Duplicate Removal
3. Imbalance Handling

Preprocessed datasets are saved and can be used by run_all_datasets_3000s.py
"""

import sys
import yaml
import pandas as pd
import logging
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any, Union

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loaders import DatasetLoader
from src.preprocessing.outlier_detection import OutlierDetectionPreprocessor
from src.preprocessing.duplicate_removal_enhanced import DuplicateRemovalPreprocessor
from src.preprocessing.imbalance_handling import ImbalanceHandlingPreprocessor
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_dataset_config(config_path: str = 'config/datasets.yaml') -> dict:
    """
    Load dataset configuration from YAML file.
    
    Args:
        config_path (str): Path to the dataset configuration file.
        
    Returns:
        dict: Dataset configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_preprocessing_config(config_path: str = 'config/preprocessing.yaml') -> dict:
    """
    Load preprocessing configuration from YAML file.
    
    Args:
        config_path (str): Path to the preprocessing configuration file.
        
    Returns:
        dict: Preprocessing configuration.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_preprocessing_config(config: dict, output_path: str = 'config/preprocessing_applied.yaml'):
    """
    Save preprocessing configuration to YAML file.
    
    Args:
        config (dict): Preprocessing configuration.
        output_path (str): Path to save the configuration.
    """
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def preprocess_dataset(
    dataset_name: str,
    dataset_config: dict,
    preprocessing_config: dict,
    output_dir: str = 'prepared_datasets'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Preprocess a dataset using specified techniques.
    
    Args:
        dataset_name (str): Name of the dataset.
        dataset_config (dict): Dataset configuration.
        preprocessing_config (dict): Preprocessing configuration.
        output_dir (str): Directory to save preprocessed datasets.
        
    Returns:
        Tuple: Preprocessed (train_data, val_data, test_data, preprocessing_stats)
    """
    logger.info(f"🚀 Starting preprocessing for {dataset_name}")
    
    # Load dataset
    loader = DatasetLoader(cache_dir='datasets/processed')
    train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
    
    logger.info(f"📊 Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Get dataset quality issues
    quality_issues = dataset_config.get('quality_issues', [])
    
    if not quality_issues:
        logger.warning(f"⚠️ No quality issues specified for {dataset_name}. Using default preprocessing.")
    else:
        logger.info(f"🏷️ Detected quality issues: {quality_issues}")
    
    # Map quality issues to preprocessing steps
    pipeline_steps = []
    
    # Define mapping from quality issues to preprocessing techniques
    quality_issue_mapping = {
        'outlier': {
            'type': 'outlier',
            'params': {
                'strategy': 'isolation-forest',
                'contamination': 0.03,
                'random_state': 42
            }
        },
        'redundant': {
            'type': 'duplicate',
            'params': {
                'strategy': 'approximate',
                'similarity_threshold': 0.9,
                'similarity_metric': 'cosine',
                'cross_label_duplicates': True,
                'random_state': 42
            }
        },
        'near-duplicate': {
            'type': 'duplicate',
            'params': {
                'strategy': 'approximate',
                'similarity_threshold': 0.9,
                'similarity_metric': 'cosine',
                'cross_label_duplicates': True,
                'random_state': 42
            }
        },
        'class_imbalance': {
            'type': 'imbalance',
            'params': {
                'strategy': 'smote',
                'sampling_strategy': 'not minority',
                'random_state': 42
            }
        },
        'fine_grained_classes': {
            'type': 'imbalance',
            'params': {
                'strategy': 'random-oversampling',
                'sampling_strategy': 'not minority',
                'random_state': 42
            }
        },
        'noisy': {
            'type': 'outlier',
            'params': {
                'strategy': 'lof',
                'contamination': 0.05,
                'n_neighbors': 20,
                'random_state': 42
            }
        },
        'small_dataset': {
            'type': 'imbalance',
            'params': {
                'strategy': 'eda',
                'eda_alpha': 0.1,
                'eda_num_aug': 4,
                'random_state': 42
            }
        },
        'context_dependency': {
            'type': 'outlier',
            'params': {
                'strategy': 'lof',
                'contamination': 0.05,
                'n_neighbors': 20,
                'random_state': 42
            }
        }
    }
    
    # Add steps based on quality issues
    for issue in quality_issues:
        if issue in quality_issue_mapping:
            step_config = quality_issue_mapping[issue]
            logger.info(f"🔧 Adding preprocessing step for issue '{issue}': {step_config['type']}")
            
            # Make a deep copy to avoid modifying the original mapping
            step = {
                'type': step_config['type'],
                'params': dict(step_config['params']),
                'enabled': True
            }
            
            pipeline_steps.append(step)
        else:
            logger.warning(f"⚠️ Unknown quality issue: {issue}. No corresponding preprocessing step available.")
    
    # If no quality issues matched, use default steps from config
    if not pipeline_steps and 'steps' in preprocessing_config:
        logger.info("📋 Using default preprocessing steps from configuration")
        pipeline_steps = preprocessing_config.get('steps', [])
    
    # Use predefined variation if 'noisy' quality issue is detected
    if 'noisy' in quality_issues and not pipeline_steps:
        logger.info("📋 Detected 'noisy' quality issue, using noisy_data variation")
        if 'variations' in preprocessing_config and 'noisy_data' in preprocessing_config['variations']:
            pipeline_steps = preprocessing_config['variations']['noisy_data'].get('steps', [])
            logger.info(f"📋 Using noisy_data variation with {len(pipeline_steps)} steps")
    
    # Use predefined variation for datasets with both class imbalance and near-duplicates
    if 'class_imbalance' in quality_issues and 'near-duplicate' in quality_issues and not pipeline_steps:
        logger.info("📋 Detected both 'class_imbalance' and 'near-duplicate' issues, using imbalanced_with_duplicates variation")
        if 'variations' in preprocessing_config and 'imbalanced_with_duplicates' in preprocessing_config['variations']:
            pipeline_steps = preprocessing_config['variations']['imbalanced_with_duplicates'].get('steps', [])
            logger.info(f"📋 Using imbalanced_with_duplicates variation with {len(pipeline_steps)} steps")
    
    # Get text and target columns
    text_column = dataset_config.get('text_column', dataset_config.get('text_columns', ['text'])[0])
    target_column = dataset_config.get('target_column', 'label')
    
    # Update text and target columns in each step
    updated_pipeline_steps = []
    for step in pipeline_steps:
        # Handle string-based step references from old format
        if isinstance(step, str):
            logger.warning(f"⚠️ Found string-based step reference: {step}. Converting to new format.")
            # Convert to new format
            updated_step = {
                'type': 'duplicate' if step in ['duplicate_removal', 'near_duplicate_removal'] else 
                         'imbalance' if step in ['smote', 'borderline_smote', 'random_undersampling', 'edited_nearest_neighbors'] else
                         'outlier' if step in ['confident_learning', 'isolation_forest', 'statistical_outliers', 'embedding_outliers'] else
                         'unknown',
                'params': {
                    'text_column': text_column,
                    'label_column': target_column
                },
                'enabled': True
            }
            updated_pipeline_steps.append(updated_step)
        else:
            # Handle dictionary-based steps (new format)
            params = step.get('params', {})
            
            # Add text and target columns if not specified
            if 'text_column' not in params:
                params['text_column'] = text_column
            if 'label_column' not in params and step['type'] != 'outlier':
                params['label_column'] = target_column
                
            updated_pipeline_steps.append(step)
            
    # Replace original steps with updated ones
    pipeline_steps = updated_pipeline_steps
    
    # Create pipeline
    pipeline = PreprocessingPipeline(steps=pipeline_steps, verbose=True)
    
    # First, fit the pipeline on training data only to learn parameters
    logger.info(f"⚙️ Fitting preprocessing pipeline on training data...")
    start_time = time.time()
    pipeline.fit(train_data)
    fit_time = time.time() - start_time
    logger.info(f"✅ Pipeline fitted on training data in {fit_time:.2f}s")
    
    # Apply preprocessing to train data
    logger.info(f"⚙️ Applying preprocessing pipeline to training data...")
    start_time = time.time()
    train_data_processed = pipeline.transform(train_data)
    train_time = time.time() - start_time
    logger.info(f"✅ Training data processed in {train_time:.2f}s")
    logger.info(f"📊 Original: {len(train_data)} samples → Processed: {len(train_data_processed)} samples")
    
    # Apply preprocessing to validation data if available
    if val_data is not None and len(val_data) > 0:
        logger.info(f"⚙️ Applying preprocessing pipeline to validation data (preserving sample count)...")
        start_time = time.time()
        val_data_processed = pipeline.transform(val_data, is_validation_or_test=True)
        val_time = time.time() - start_time
        logger.info(f"✅ Validation data processed in {val_time:.2f}s")
        logger.info(f"📊 Original: {len(val_data)} samples → Processed: {len(val_data_processed)} samples")
    else:
        val_data_processed = val_data
    
    # Apply preprocessing to test data if available
    if test_data is not None and len(test_data) > 0:
        logger.info(f"⚙️ Applying preprocessing pipeline to test data (preserving sample count)...")
        start_time = time.time()
        test_data_processed = pipeline.transform(test_data, is_validation_or_test=True)
        test_time = time.time() - start_time
        logger.info(f"✅ Test data processed in {test_time:.2f}s")
        logger.info(f"📊 Original: {len(test_data)} samples → Processed: {len(test_data_processed)} samples")
    else:
        test_data_processed = test_data
        
    # Get preprocessing statistics
    preprocessing_stats = pipeline.get_stats()
    
    return train_data_processed, val_data_processed, test_data_processed, preprocessing_stats

def save_preprocessed_dataset(
    dataset_name: str,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame],
    test_data: Optional[pd.DataFrame],
    preprocessing_stats: dict,
    output_dir: str = 'prepared_datasets'
) -> str:
    """
    Save preprocessed dataset and preprocessing statistics.
    
    Args:
        dataset_name (str): Name of the dataset.
        train_data (pd.DataFrame): Preprocessed training data.
        val_data (pd.DataFrame): Preprocessed validation data.
        test_data (pd.DataFrame): Preprocessed test data.
        preprocessing_stats (dict): Preprocessing statistics.
        output_dir (str): Directory to save preprocessed datasets.
        
    Returns:
        str: Path to the saved dataset directory.
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = Path(output_dir) / f"{dataset_name}_preprocessed"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure consistent column order across all data splits
    # Order columns with target column first, then all other columns
    target_column = None
    for col in train_data.columns:
        if col.lower() in ["label", "target", "class", "category", "sentiment"]:
            target_column = col
            break
    
    # Get common columns that exist in all datasets
    common_columns = set(train_data.columns)
    if val_data is not None and len(val_data) > 0:
        common_columns = common_columns.intersection(set(val_data.columns))
    if test_data is not None and len(test_data) > 0:
        common_columns = common_columns.intersection(set(test_data.columns))
    
    # Ensure target column is included in common columns
    if target_column and target_column not in common_columns:
        logger.warning(f"⚠️ Target column '{target_column}' not found in all datasets. Using available columns.")
    
    # Create ordered list of columns with target column first if available
    if target_column and target_column in common_columns:
        ordered_cols = [target_column] + [c for c in common_columns if c != target_column]
    else:
        ordered_cols = list(common_columns)
    
    # Apply column order to all datasets
    train_data = train_data[ordered_cols]
    if val_data is not None and len(val_data) > 0:
        val_data = val_data[ordered_cols]
    if test_data is not None and len(test_data) > 0:
        test_data = test_data[ordered_cols]
    
    # Save preprocessed data
    train_data.to_csv(dataset_dir / "train.csv", index=False)
    logger.info(f"💾 Saved preprocessed training data: {len(train_data)} samples")
    
    if val_data is not None and len(val_data) > 0:
        val_data.to_csv(dataset_dir / "val.csv", index=False)
        logger.info(f"💾 Saved preprocessed validation data: {len(val_data)} samples")
    
    if test_data is not None and len(test_data) > 0:
        test_data.to_csv(dataset_dir / "test.csv", index=False)
        logger.info(f"💾 Saved preprocessed test data: {len(test_data)} samples")
    
    # Save preprocessing statistics
    with open(dataset_dir / "preprocessing_stats.json", 'w') as f:
        json.dump(preprocessing_stats, f, indent=2, default=str)
    
    logger.info(f"💾 Saved preprocessing statistics")
    
    # Create metadata file with dataset info and preprocessing details
    metadata = {
        "dataset_name": dataset_name,
        "preprocessed_timestamp": timestamp,
        "train_samples": len(train_data),
        "val_samples": len(val_data) if val_data is not None else 0,
        "test_samples": len(test_data) if test_data is not None else 0,
        "preprocessing_applied": True,
        "preprocessing_steps": list(preprocessing_stats.keys())  # Just use the keys directly
    }
    
    with open(dataset_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"💾 Saved dataset metadata")
    
    return str(dataset_dir)

def update_datasets_config(
    datasets_config: dict,
    dataset_name: str,
    preprocessed_path: str,
    preprocessing_config: dict
) -> dict:
    """
    Update datasets configuration with preprocessed dataset information.
    
    Args:
        datasets_config (dict): Original datasets configuration.
        dataset_name (str): Name of the dataset.
        preprocessed_path (str): Path to the preprocessed dataset.
        preprocessing_config (dict): Preprocessing configuration used.
        
    Returns:
        dict: Updated datasets configuration.
    """
    # Create a deep copy of the original config
    updated_config = {**datasets_config}
    
    # Find the dataset in the config
    dataset_found = False
    for category, datasets in datasets_config.get('datasets', {}).items():
        if dataset_name in datasets:
            dataset_found = True
            # Get the dataset config safely
            original_dataset_config = datasets.get(dataset_name, {})
            
            # Ensure we're working with a dictionary
            if not isinstance(original_dataset_config, dict):
                logger.warning(f"⚠️ Dataset config for {dataset_name} is not a dictionary. Creating a new configuration.")
                updated_dataset_config = {}
            else:
                # Create a copy of the original config
                updated_dataset_config = {**original_dataset_config}
            
            # Add preprocessing information
            updated_dataset_config['preprocessed'] = True
            updated_dataset_config['preprocessed_path'] = preprocessed_path
            
            # Only add preprocessing_config if it's a dictionary
            if isinstance(preprocessing_config, dict):
                # We'll create a simplified version with just the steps to avoid potential circular references
                simplified_config = {
                    'steps': preprocessing_config.get('steps', []),
                    'variations': preprocessing_config.get('variations', {})
                }
                updated_dataset_config['preprocessing_config'] = simplified_config
            else:
                updated_dataset_config['preprocessing_config'] = {'applied': True}
            
            # Update the config
            if 'preprocessed_datasets' not in updated_config:
                updated_config['preprocessed_datasets'] = {}
            
            updated_config['preprocessed_datasets'][dataset_name] = updated_dataset_config
            break
    
    if not dataset_found:
        logger.warning(f"⚠️ Dataset '{dataset_name}' not found in configuration. Adding to preprocessed_datasets anyway.")
        if 'preprocessed_datasets' not in updated_config:
            updated_config['preprocessed_datasets'] = {}
        
        updated_config['preprocessed_datasets'][dataset_name] = {
            'preprocessed': True,
            'preprocessed_path': preprocessed_path,
            'preprocessing_config': {'applied': True}
        }
    
    return updated_config

def main():
    """Main function to preprocess datasets."""
    parser = argparse.ArgumentParser(description='Preprocess datasets for AutoGluon')
    parser.add_argument('--dataset', type=str, help='Dataset name to preprocess')
    parser.add_argument('--all', action='store_true', help='Preprocess all datasets')
    parser.add_argument('--config', type=str, default='config/preprocessing.yaml', 
                        help='Path to preprocessing configuration')
    parser.add_argument('--output-dir', type=str, default='prepared_datasets',
                        help='Directory to save preprocessed datasets')
    parser.add_argument('--datasets-config', type=str, default='config/datasets.yaml',
                        help='Path to datasets configuration')
    parser.add_argument('--save-config', type=str, default='config/preprocessing_applied.yaml',
                        help='Path to save applied preprocessing configuration')
    
    args = parser.parse_args()
    
    if not args.dataset and not args.all:
        parser.error("Either --dataset or --all must be specified")
    
    datasets_config = load_dataset_config(args.datasets_config)
    preprocessing_config = load_preprocessing_config(args.config)
    
    # Log important preprocessing configuration changes
    logger.info("📝 Important preprocessing configuration notes:")
    logger.info("  • Using cosine similarity instead of Levenshtein for duplicate detection to improve performance")
    logger.info("  • Preserving sample count in validation and test sets (no outlier removal/oversampling/duplicate removal)")
    
    datasets_to_process = []
    
    if args.all:
        for category, datasets in datasets_config['datasets'].items():
            for dataset_name, dataset_config in datasets.items():
                datasets_to_process.append((dataset_name, dataset_config, category))
        
        logger.info(f"📋 Found {len(datasets_to_process)} datasets to preprocess")
        
    else:
        # Find the specified dataset
        dataset_found = False
        
        for category, datasets in datasets_config['datasets'].items():
            if args.dataset in datasets:
                dataset_config = datasets[args.dataset]
                datasets_to_process.append((args.dataset, dataset_config, category))
                dataset_found = True
                break
        
        if not dataset_found:
            logger.error(f"❌ Dataset '{args.dataset}' not found in configuration")
            sys.exit(1)
    
    # Initialize updated config
    updated_datasets_config = {**datasets_config}
    if 'preprocessed_datasets' not in updated_datasets_config:
        updated_datasets_config['preprocessed_datasets'] = {}
    
    # Process each dataset
    for i, (dataset_name, dataset_config, category) in enumerate(datasets_to_process, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Dataset {i}/{len(datasets_to_process)}: {dataset_name} (Category: {category})")
        logger.info(f"📝 Description: {dataset_config.get('description', 'N/A')}")
        logger.info(f"🏷️  Issues: {dataset_config.get('quality_issues', [])}")
        logger.info(f"{'='*80}")
        
        try:
            # Preprocess dataset
            train_data, val_data, test_data, preprocessing_stats = preprocess_dataset(
                dataset_name,
                dataset_config,
                preprocessing_config,
                output_dir=args.output_dir
            )
            
            # Save preprocessed dataset
            preprocessed_path = save_preprocessed_dataset(
                dataset_name,
                train_data,
                val_data,
                test_data,
                preprocessing_stats,
                output_dir=args.output_dir
            )
            
            # Update datasets config
            try:
                updated_datasets_config = update_datasets_config(
                    updated_datasets_config,
                    dataset_name,
                    preprocessed_path,
                    preprocessing_config
                )
                logger.info(f"✅ Preprocessing completed for {dataset_name}")
            except Exception as e:
                logger.error(f"❌ Failed to update config for {dataset_name}: {str(e)}")
                logger.info(f"✅ Preprocessing completed for {dataset_name}, but config update failed.")
            
        except Exception as e:
            logger.error(f"❌ Failed to preprocess {dataset_name}: {str(e)}")
            # Print stack trace for debugging
            import traceback
            logger.error(traceback.format_exc())
    
    # Save updated datasets config
    save_preprocessing_config(updated_datasets_config, args.save_config)
    logger.info(f"💾 Saved updated datasets configuration to {args.save_config}")
    
    logger.info(f"\n🏁 PREPROCESSING COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"✅ Processed {len(datasets_to_process)} datasets")
    logger.info(f"💾 Preprocessed datasets saved to {args.output_dir}")
    logger.info(f"💾 Configuration saved to {args.save_config}")

if __name__ == "__main__":
    main()
