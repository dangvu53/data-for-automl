#!/usr/bin/env python3
"""
Run baseline experiments for all datasets with 3000 seconds time limit.
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

def run_baseline_for_dataset(dataset_name: str, dataset_config: dict, time_limit: int = 3000):
    """Run baseline experiment for a single dataset."""
    logger.info(f"🚀 Starting baseline for {dataset_name} (time limit: {time_limit}s)")
    
    # Load dataset
    loader = DatasetLoader(cache_dir='datasets/processed')
    train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
    
    logger.info(f"📊 Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")
    
    # Check 50K limit
    max_split = max(len(train_data), len(val_data), len(test_data))
    if max_split <= 50000:
        logger.info(f"✅ All splits ≤ 50K (max: {max_split:,})")
    else:
        logger.warning(f"⚠️  Some splits > 50K (max: {max_split:,})")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/results/full_baseline_3000s") / f"{dataset_name}_{timestamp}"
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
    
    logger.info(f"🤖 Training with {len(autogluon_config['autogluon'].get('included_model_types', []))} model types")
    logger.info(f"📋 Model types: {autogluon_config['autogluon'].get('included_model_types', 'default')}")
    
    try:
        training_results = classifier.fit(train_data, target_col, validation_data=val_data)
        training_time = time.time() - start_time
        
        logger.info(f"✅ Training completed in {training_time:.1f}s")
        logger.info(f"🏆 Best model: {training_results['best_model']}")
        
        # Check how many models were actually trained
        if training_results.get("leaderboard") is not None:
            leaderboard = training_results["leaderboard"]
            num_models = len(leaderboard)
            logger.info(f"📊 Models trained: {num_models}")
            logger.info(f"📋 Top 5 models:")
            for i, row in leaderboard.head().iterrows():
                logger.info(f"  {i+1}. {row['model']} - Score: {row.get('score_val', 'N/A'):.4f}")
        
        # Evaluate
        test_metrics = classifier.evaluate(test_data, target_col)

        # Handle both classification and regression metrics
        if 'accuracy' in test_metrics:
            primary_metric = test_metrics['accuracy']
            logger.info(f"🎯 Test accuracy: {primary_metric:.4f}")
        elif 'root_mean_squared_error' in test_metrics:
            primary_metric = abs(test_metrics['root_mean_squared_error'])  # Make positive for logging
            logger.info(f"🎯 Test RMSE: {primary_metric:.4f}")
        else:
            primary_metric = list(test_metrics.values())[0] if test_metrics else 0
            metric_name = list(test_metrics.keys())[0] if test_metrics else 'unknown'
            logger.info(f"🎯 Test {metric_name}: {primary_metric}")
        
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
            "autogluon_config": autogluon_config["autogluon"],
            "models_trained": len(leaderboard) if training_results.get("leaderboard") is not None else 0,
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
    """Run baselines for all datasets with 3000s time limit."""
    logger.info("🧪 Running baseline experiments for all datasets (3000s each)")
    
    # Load dataset config
    with open('config/datasets.yaml', 'r') as f:
        datasets_config = yaml.safe_load(f)
    
    # Collect all datasets
    all_datasets = []
    for category, datasets in datasets_config['datasets'].items():
        for dataset_name, dataset_config in datasets.items():
            all_datasets.append((dataset_name, dataset_config, category))
    
    logger.info(f"📋 Found {len(all_datasets)} datasets to test")
    
    # Run experiments
    results_summary = []
    time_limit = 3000  # 50 minutes per dataset
    
    total_start_time = time.time()
    
    for i, (dataset_name, dataset_config, category) in enumerate(all_datasets, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Dataset {i}/{len(all_datasets)}: {dataset_name} (Category: {category})")
        logger.info(f"📝 Description: {dataset_config.get('description', 'N/A')}")
        logger.info(f"🏷️  Issues: {dataset_config.get('quality_issues', [])}")
        logger.info(f"⏱️  Time limit: {time_limit}s ({time_limit/60:.1f} minutes)")
        logger.info(f"{'='*80}")
        
        dataset_start_time = time.time()
        result = run_baseline_for_dataset(dataset_name, dataset_config, time_limit)
        dataset_time = time.time() - dataset_start_time
        
        result["actual_runtime"] = dataset_time
        results_summary.append(result)
        
        # Print quick summary
        if result.get("status") == "completed":
            test_metrics = result.get("test_metrics", {})
            models_trained = result.get("models_trained", 0)

            # Handle different metric types
            if 'accuracy' in test_metrics:
                metric_value = test_metrics['accuracy']
                metric_str = f"{metric_value:.4f} accuracy"
            elif 'root_mean_squared_error' in test_metrics:
                metric_value = abs(test_metrics['root_mean_squared_error'])
                metric_str = f"{metric_value:.4f} RMSE"
            else:
                metric_str = "completed"

            logger.info(f"✅ {dataset_name}: {metric_str}, {models_trained} models, {dataset_time/60:.1f}min")
        else:
            logger.info(f"❌ {dataset_name}: Failed after {dataset_time/60:.1f}min")
        
        # Estimate remaining time
        elapsed_total = time.time() - total_start_time
        avg_time_per_dataset = elapsed_total / i
        remaining_datasets = len(all_datasets) - i
        estimated_remaining = remaining_datasets * avg_time_per_dataset
        
        logger.info(f"⏱️  Progress: {i}/{len(all_datasets)} completed")
        logger.info(f"⏱️  Estimated remaining time: {estimated_remaining/3600:.1f} hours")
    
    # Save overall summary
    summary_dir = Path("experiments/results/full_baseline_3000s")
    summary_file = summary_dir / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(summary_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print final summary
    total_time = time.time() - total_start_time
    logger.info(f"\n🏁 ALL BASELINE EXPERIMENTS COMPLETED")
    logger.info(f"{'='*80}")
    logger.info(f"⏱️  Total runtime: {total_time/3600:.1f} hours")
    
    completed = [r for r in results_summary if r.get("status") == "completed"]
    failed = [r for r in results_summary if r.get("status") == "failed"]
    
    logger.info(f"✅ Completed: {len(completed)}/{len(all_datasets)}")
    logger.info(f"❌ Failed: {len(failed)}/{len(all_datasets)}")
    
    if completed:
        logger.info(f"\n📊 Results Summary:")
        for result in completed:
            test_metrics = result.get("test_metrics", {})
            models = result.get("models_trained", 0)
            runtime = result.get("actual_runtime", 0) / 60

            # Handle different metric types for summary
            if 'accuracy' in test_metrics:
                metric_str = f"{test_metrics['accuracy']:.4f} acc"
            elif 'root_mean_squared_error' in test_metrics:
                metric_str = f"{abs(test_metrics['root_mean_squared_error']):.4f} RMSE"
            else:
                metric_str = "completed"

            logger.info(f"  {result['dataset_name']}: {metric_str}, {models} models, {runtime:.1f}min")
    
    if failed:
        logger.info(f"\n❌ Failed datasets:")
        for result in failed:
            logger.info(f"  {result['dataset_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n💾 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
