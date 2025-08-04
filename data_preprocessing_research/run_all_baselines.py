#!/usr/bin/env python3
"""
Run baseline experiments for all datasets without complex experiment tracking.
"""

import sys
import yaml
import pandas as pd
import logging
import json
import time
from pathlib import Path
from datetime import datetime

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

def run_baseline_for_dataset(dataset_name: str, dataset_config: dict, time_limit: int = 300):
    """Run baseline experiment for a single dataset."""
    logger.info(f"🚀 Starting baseline for {dataset_name}")
    
    # Load dataset
    loader = DatasetLoader(cache_dir='datasets/processed')
    train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
    
    logger.info(f"📊 Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/results/baselines") / f"{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load AutoGluon config
    with open('config/autogluon_simple.yaml', 'r') as f:
        autogluon_config = yaml.safe_load(f)
    
    # Override time limit
    autogluon_config["autogluon"]["time_limit"] = time_limit
    
    # Initialize classifier
    model_path = str(output_dir / "model")
    classifier = AutoGluonTextClassifier(
        config=autogluon_config["autogluon"],
        model_path=model_path,
        random_seed=42
    )
    
    # Train
    start_time = time.time()
    target_col = dataset_config["target_column"]
    
    try:
        training_results = classifier.fit(train_data, target_col, validation_data=val_data)
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.1f}s")
        logger.info(f"🏆 Best model: {training_results['best_model']}")
        
        # Evaluate
        test_metrics = classifier.evaluate(test_data, target_col)
        logger.info(f"🎯 Test accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
        
        # Save results
        results = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "timestamp": timestamp,
            "training_time": training_time,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
            "test_samples": len(test_data),
            "best_model": training_results["best_model"],
            "test_metrics": test_metrics,
            "time_limit": time_limit,
            "status": "completed"
        }
        
        # Save to JSON
        with open(output_dir / "results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save leaderboard if available
        if training_results.get("leaderboard") is not None:
            training_results["leaderboard"].to_csv(output_dir / "leaderboard.csv", index=False)
        
        logger.info(f"💾 Results saved to {output_dir}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Failed for {dataset_name}: {str(e)}")
        
        # Save failure info
        failure_results = {
            "dataset_name": dataset_name,
            "timestamp": timestamp,
            "status": "failed",
            "error": str(e),
            "time_limit": time_limit
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(failure_results, f, indent=2, default=str)
        
        return failure_results

def main():
    """Run baselines for all datasets."""
    logger.info("🧪 Running baseline experiments for all datasets")
    
    # Load dataset config
    with open('config/datasets.yaml', 'r') as f:
        datasets_config = yaml.safe_load(f)
    
    # Collect all datasets
    all_datasets = []
    for category, datasets in datasets_config['datasets'].items():
        for dataset_name, dataset_config in datasets.items():
            all_datasets.append((dataset_name, dataset_config))
    
    logger.info(f"📋 Found {len(all_datasets)} datasets to test")
    
    # Run experiments
    results_summary = []
    time_limit = 3000  # 5 minutes per dataset
    
    for i, (dataset_name, dataset_config) in enumerate(all_datasets, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Dataset {i}/{len(all_datasets)}: {dataset_name}")
        logger.info(f"📝 Description: {dataset_config.get('description', 'N/A')}")
        logger.info(f"🏷️  Issues: {dataset_config.get('quality_issues', [])}")
        logger.info(f"{'='*60}")
        
        result = run_baseline_for_dataset(dataset_name, dataset_config, time_limit)
        results_summary.append(result)
        
        # Print quick summary
        if result.get("status") == "completed":
            accuracy = result.get("test_metrics", {}).get("accuracy", 0)
            logger.info(f"✅ {dataset_name}: {accuracy:.4f} accuracy")
        else:
            logger.info(f"❌ {dataset_name}: Failed")
    
    # Save overall summary
    summary_dir = Path("experiments/results/baselines")
    summary_file = summary_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print final summary
    logger.info(f"\n🏁 BASELINE EXPERIMENTS COMPLETED")
    logger.info(f"{'='*60}")
    
    completed = [r for r in results_summary if r.get("status") == "completed"]
    failed = [r for r in results_summary if r.get("status") == "failed"]
    
    logger.info(f"✅ Completed: {len(completed)}/{len(all_datasets)}")
    logger.info(f"❌ Failed: {len(failed)}/{len(all_datasets)}")
    
    if completed:
        logger.info(f"\n📊 Results:")
        for result in completed:
            accuracy = result.get("test_metrics", {}).get("accuracy", 0)
            logger.info(f"  {result['dataset_name']}: {accuracy:.4f}")
    
    if failed:
        logger.info(f"\n❌ Failed datasets:")
        for result in failed:
            logger.info(f"  {result['dataset_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n💾 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
