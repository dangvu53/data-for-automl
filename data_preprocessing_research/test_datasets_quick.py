#!/usr/bin/env python3
"""
Quick test script to verify each dataset works before running full experiments.
"""

import sys
import yaml
import pandas as pd
import logging
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data.loaders import DatasetLoader
from models.autogluon_pipeline import AutoGluonTextClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_dataset_loading(dataset_name: str, dataset_config: dict):
    """Test if a dataset can be loaded properly."""
    try:
        logger.info(f"🧪 Testing dataset loading: {dataset_name}")
        
        # Load dataset
        loader = DatasetLoader(cache_dir='datasets/processed')
        train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
        
        # Basic checks
        assert len(train_data) > 0, "Train data is empty"
        assert len(val_data) > 0, "Validation data is empty"
        assert len(test_data) > 0, "Test data is empty"
        
        # Check columns - use global settings if not in dataset config
        target_col = dataset_config.get("target_column", "label")  # Default from global_settings
        text_col = dataset_config.get("text_columns", ["text"])[0]  # Default from global_settings
        
        assert target_col in train_data.columns, f"Target column '{target_col}' not found"
        assert text_col in train_data.columns, f"Text column '{text_col}' not found"
        
        # Check 50K limit
        max_split = max(len(train_data), len(val_data), len(test_data))
        assert max_split <= 50000, f"Split size {max_split} exceeds 50K limit"
        
        # Check data quality
        unique_labels = train_data[target_col].nunique()
        text_lengths = train_data[text_col].astype(str).str.len()
        
        logger.info(f"  ✅ Data loaded successfully:")
        logger.info(f"    📈 Train: {len(train_data):,} samples")
        logger.info(f"    📊 Validation: {len(val_data):,} samples")
        logger.info(f"    📋 Test: {len(test_data):,} samples")
        logger.info(f"    🏷️  Unique labels: {unique_labels}")
        logger.info(f"    📝 Avg text length: {text_lengths.mean():.0f} chars")
        logger.info(f"    ✅ Max split size: {max_split:,} (≤ 50K)")
        
        return True, {
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "unique_labels": unique_labels,
            "avg_text_length": text_lengths.mean(),
            "max_split_size": max_split
        }
        
    except Exception as e:
        logger.error(f"  ❌ Failed to load {dataset_name}: {str(e)}")
        return False, {"error": str(e)}

def test_autogluon_quick(dataset_name: str, dataset_config: dict, time_limit: int = 60):
    """Test if AutoGluon can train on the dataset quickly."""
    try:
        logger.info(f"🤖 Testing AutoGluon training: {dataset_name} ({time_limit}s)")
        
        # Load dataset
        loader = DatasetLoader(cache_dir='datasets/processed')
        train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
        
        # Use small subset for quick test
        small_train = train_data.head(500)
        small_val = val_data.head(100)
        small_test = test_data.head(100)
        
        # Load AutoGluon config
        with open('config/autogluon_simple.yaml', 'r') as f:
            autogluon_config = yaml.safe_load(f)
        
        # Override time limit for quick test
        autogluon_config["autogluon"]["time_limit"] = time_limit
        
        # Create temporary model path
        model_path = f"temp_test_model_{dataset_name}"
        
        # Initialize classifier
        classifier = AutoGluonTextClassifier(
            config=autogluon_config["autogluon"],
            model_path=model_path,
            random_seed=42
        )
        
        # Train
        start_time = time.time()
        target_col = dataset_config.get("target_column", "label")  # Default from global_settings
        
        training_results = classifier.fit(small_train, target_col, validation_data=small_val)
        training_time = time.time() - start_time
        
        # Quick evaluation
        test_metrics = classifier.evaluate(small_test, target_col)
        
        # Clean up
        import shutil
        if Path(model_path).exists():
            shutil.rmtree(model_path)
        
        logger.info(f"  ✅ AutoGluon training successful:")
        logger.info(f"    ⏱️  Training time: {training_time:.1f}s")
        logger.info(f"    🏆 Best model: {training_results['best_model']}")
        logger.info(f"    📊 Models trained: {len(training_results.get('leaderboard', []))}")
        
        # Handle different metric types
        if 'accuracy' in test_metrics:
            primary_metric = test_metrics['accuracy']
            logger.info(f"    🎯 Test accuracy: {primary_metric:.4f}")
        elif 'root_mean_squared_error' in test_metrics:
            primary_metric = abs(test_metrics['root_mean_squared_error'])
            logger.info(f"    🎯 Test RMSE: {primary_metric:.4f}")
        else:
            primary_metric = list(test_metrics.values())[0] if test_metrics else 0
            logger.info(f"    🎯 Test metric: {primary_metric}")
        
        return True, {
            "training_time": training_time,
            "best_model": training_results["best_model"],
            "models_trained": len(training_results.get('leaderboard', [])),
            "test_metrics": test_metrics,
            "primary_metric": primary_metric
        }
        
    except Exception as e:
        logger.error(f"  ❌ AutoGluon training failed for {dataset_name}: {str(e)}")
        return False, {"error": str(e)}

def main():
    """Test all datasets."""
    logger.info("🧪 TESTING ALL DATASETS FOR READINESS")
    logger.info("=" * 60)
    
    # Load dataset config
    with open('config/datasets.yaml', 'r') as f:
        datasets_config = yaml.safe_load(f)
    
    # Collect all datasets
    all_datasets = []
    for category, datasets in datasets_config['datasets'].items():
        for dataset_name, dataset_config in datasets.items():
            all_datasets.append((dataset_name, dataset_config, category))
    
    logger.info(f"📋 Found {len(all_datasets)} datasets to test")
    
    # Test results
    loading_results = {}
    training_results = {}
    
    for i, (dataset_name, dataset_config, category) in enumerate(all_datasets, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"📊 Dataset {i}/{len(all_datasets)}: {dataset_name} (Category: {category})")
        logger.info(f"📝 Description: {dataset_config.get('description', 'N/A')}")
        logger.info(f"{'='*50}")
        
        # Test 1: Data loading
        loading_success, loading_info = test_dataset_loading(dataset_name, dataset_config)
        loading_results[dataset_name] = {
            "success": loading_success,
            "info": loading_info,
            "category": category
        }
        
        if not loading_success:
            logger.warning(f"⚠️  Skipping AutoGluon test for {dataset_name} due to loading failure")
            continue
        
        # Test 2: AutoGluon quick training
        training_success, training_info = test_autogluon_quick(dataset_name, dataset_config, time_limit=60)
        training_results[dataset_name] = {
            "success": training_success,
            "info": training_info,
            "category": category
        }
    
    # Summary
    logger.info(f"\n🏁 TESTING SUMMARY")
    logger.info("=" * 60)
    
    loading_passed = sum(1 for r in loading_results.values() if r["success"])
    training_passed = sum(1 for r in training_results.values() if r["success"])
    
    logger.info(f"📥 Data Loading: {loading_passed}/{len(all_datasets)} passed")
    logger.info(f"🤖 AutoGluon Training: {training_passed}/{len(all_datasets)} passed")
    
    # Detailed results
    logger.info(f"\n📊 READY FOR FULL EXPERIMENTS:")
    ready_datasets = []
    for dataset_name in loading_results:
        if (loading_results[dataset_name]["success"] and 
            training_results.get(dataset_name, {}).get("success", False)):
            ready_datasets.append(dataset_name)
            category = loading_results[dataset_name]["category"]
            samples = loading_results[dataset_name]["info"]["train_samples"]
            logger.info(f"  ✅ {dataset_name} ({category}): {samples:,} samples")
    
    logger.info(f"\n❌ FAILED DATASETS:")
    for dataset_name in loading_results:
        if not loading_results[dataset_name]["success"]:
            error = loading_results[dataset_name]["info"].get("error", "Unknown")
            logger.info(f"  ❌ {dataset_name}: Loading failed - {error}")
        elif not training_results.get(dataset_name, {}).get("success", False):
            error = training_results[dataset_name]["info"].get("error", "Unknown")
            logger.info(f"  ❌ {dataset_name}: Training failed - {error}")
    
    logger.info(f"\n🚀 RECOMMENDATION:")
    if len(ready_datasets) >= 3:
        logger.info(f"✅ {len(ready_datasets)} datasets are ready for full experiments!")
        logger.info(f"   You can proceed with run_all_datasets_3000s.py")
    else:
        logger.info(f"⚠️  Only {len(ready_datasets)} datasets ready. Consider fixing failed datasets first.")
    
    return ready_datasets

if __name__ == "__main__":
    ready_datasets = main()
    print(f"\n🎯 Ready datasets: {ready_datasets}")
