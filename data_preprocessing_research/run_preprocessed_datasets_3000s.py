#!/usr/bin/env python3
"""
Run baseline experiments for all preprocessed datasets with 3000 seconds time limit.
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

def load_preprocessed_data(preprocessed_path: str) -> tuple:
    """
    Load preprocessed data from the given path.
    
    Args:
        preprocessed_path (str): Path to the preprocessed dataset.
        
    Returns:
        tuple: (train_data, val_data, test_data)
    """
    data_path = Path(preprocessed_path)
    
    # Load train data
    train_path = data_path / "train.csv"
    if train_path.exists():
        train_data = pd.read_csv(train_path)
        logger.info(f"📊 Loaded preprocessed train data: {len(train_data)} samples")
    else:
        raise FileNotFoundError(f"Preprocessed train data not found at {train_path}")
    
    # Load validation data
    val_path = data_path / "val.csv"
    if val_path.exists():
        val_data = pd.read_csv(val_path)
        logger.info(f"📊 Loaded preprocessed validation data: {len(val_data)} samples")
    else:
        val_data = None
        logger.warning(f"⚠️ No preprocessed validation data found at {val_path}")
    
    # Load test data
    test_path = data_path / "test.csv"
    if test_path.exists():
        test_data = pd.read_csv(test_path)
        logger.info(f"📊 Loaded preprocessed test data: {len(test_data)} samples")
    else:
        test_data = None
        logger.warning(f"⚠️ No preprocessed test data found at {test_path}")
    
    return train_data, val_data, test_data

def run_baseline_for_preprocessed_dataset(dataset_name: str, dataset_config: dict, time_limit: int = 3000):
    """Run baseline experiment for a preprocessed dataset."""
    logger.info(f"🚀 Starting baseline for preprocessed {dataset_name} (time limit: {time_limit}s)")
    
    # Check if dataset has been preprocessed
    if not dataset_config.get('preprocessed', False):
        logger.warning(f"⚠️ Dataset {dataset_name} has not been preprocessed. Using original data.")
        
        # Load dataset
        loader = DatasetLoader(cache_dir='datasets/processed')
        train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
    else:
        # Get preprocessed path
        preprocessed_path = dataset_config.get('preprocessed_path')
        if not preprocessed_path:
            logger.warning(f"⚠️ Preprocessed path not found for {dataset_name}. Using original data.")
            
            # Load original dataset
            loader = DatasetLoader(cache_dir='datasets/processed')
            train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
        else:
            # Load preprocessed data
            try:
                train_data, val_data, test_data = load_preprocessed_data(preprocessed_path)
                logger.info(f"✅ Loaded preprocessed data from {preprocessed_path}")
            except Exception as e:
                logger.error(f"❌ Failed to load preprocessed data: {str(e)}")
                
                # Fallback to original data
                logger.warning(f"⚠️ Falling back to original data.")
                loader = DatasetLoader(cache_dir='datasets/processed')
                train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)
    
    logger.info(f"📊 Data loaded: {len(train_data)} train, {len(val_data) if val_data is not None else 0} val, {len(test_data) if test_data is not None else 0} test")
    
    # Check 50K limit
    max_split = max(
        len(train_data), 
        len(val_data) if val_data is not None else 0, 
        len(test_data) if test_data is not None else 0
    )
    if max_split <= 50000:
        logger.info(f"✅ All splits ≤ 50K (max: {max_split:,})")
    else:
        logger.warning(f"⚠️  Some splits > 50K (max: {max_split:,})")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("experiments/results/preprocessed_baseline_3000s") / f"{dataset_name}_{timestamp}"
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
            "val_samples": len(val_data) if val_data is not None else 0,
            "test_samples": len(test_data) if test_data is not None else 0,
            "best_model": training_results["best_model"],
            "test_metrics": test_metrics,
            "time_limit": time_limit,
            "autogluon_config": autogluon_config["autogluon"],
            "models_trained": len(leaderboard) if training_results.get("leaderboard") is not None else 0,
            "status": "completed",
            "preprocessed": dataset_config.get('preprocessed', False),
            "preprocessing_info": dataset_config.get('preprocessing_config', {})
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
            "time_limit": time_limit,
            "preprocessed": dataset_config.get('preprocessed', False)
        }
        
        with open(output_dir / "results.json", 'w') as f:
            json.dump(failure_results, f, indent=2, default=str)
        
        return failure_results

def main():
    """Run baselines for all preprocessed datasets with 3000s time limit."""
    logger.info("🧪 Running baseline experiments for preprocessed datasets (3000s each)")
    
    # Load dataset config
    config_path = 'config/preprocessing_applied.yaml'
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            datasets_config = yaml.safe_load(f)
        logger.info(f"📋 Loaded preprocessed datasets configuration from {config_path}")
    else:
        logger.warning(f"⚠️ Preprocessed datasets configuration not found at {config_path}")
        logger.info("📋 Falling back to original datasets configuration")
        with open('config/datasets.yaml', 'r') as f:
            datasets_config = yaml.safe_load(f)
    
    # Collect all datasets
    all_datasets = []
    
    # First check for preprocessed datasets
    if 'preprocessed_datasets' in datasets_config:
        for dataset_name, dataset_config in datasets_config['preprocessed_datasets'].items():
            all_datasets.append((dataset_name, dataset_config, 'preprocessed'))
    
    # If no preprocessed datasets, use original datasets
    if not all_datasets:
        logger.warning("⚠️ No preprocessed datasets found. Using original datasets.")
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
        logger.info(f"💾 Preprocessed: {'Yes' if dataset_config.get('preprocessed', False) else 'No'}")
        logger.info(f"{'='*80}")
        
        dataset_start_time = time.time()
        result = run_baseline_for_preprocessed_dataset(dataset_name, dataset_config, time_limit)
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
    summary_dir = Path("experiments/results/preprocessed_baseline_3000s")
    summary_dir.mkdir(parents=True, exist_ok=True)
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
    
    # Calculate improvement from preprocessing
    preprocessed_results = [r for r in completed if r.get("preprocessed", False)]
    original_results = [r for r in completed if not r.get("preprocessed", False)]
    
    if preprocessed_results and original_results:
        logger.info(f"\n📊 Preprocessing Impact Summary:")
        logger.info(f"  Preprocessed datasets: {len(preprocessed_results)}")
        logger.info(f"  Original datasets: {len(original_results)}")
        
        # Compare metrics with baseline results
        baseline_path = Path("experiments/results/full_baseline_3000s")
        baseline_files = list(baseline_path.glob("summary_*.json"))
        
        if baseline_files:
            # Use the most recent baseline summary file
            latest_baseline = max(baseline_files, key=lambda x: x.stat().st_mtime)
            logger.info(f"  📈 Comparing with baseline results from: {latest_baseline.name}")
            
            try:
                with open(latest_baseline, 'r') as f:
                    baseline_results = json.load(f)
                
                # Create a lookup dictionary for baseline results
                baseline_lookup = {r["dataset_name"]: r for r in baseline_results if r.get("status") == "completed"}
                
                # Compare metrics for each preprocessed dataset
                comparison_table = []
                
                for prep_result in preprocessed_results:
                    dataset_name = prep_result["dataset_name"]
                    
                    if dataset_name in baseline_lookup:
                        baseline = baseline_lookup[dataset_name]
                        
                        # Get primary metric (accuracy or RMSE)
                        if 'accuracy' in prep_result.get("test_metrics", {}):
                            prep_metric = prep_result["test_metrics"]["accuracy"]
                            base_metric = baseline["test_metrics"].get("accuracy", 0)
                            metric_name = "accuracy"
                            # Higher is better for accuracy
                            is_improvement = prep_metric > base_metric
                            diff = prep_metric - base_metric
                            diff_percentage = (diff / base_metric) * 100 if base_metric > 0 else 0
                        elif 'root_mean_squared_error' in prep_result.get("test_metrics", {}):
                            prep_metric = abs(prep_result["test_metrics"]["root_mean_squared_error"])
                            base_metric = abs(baseline["test_metrics"].get("root_mean_squared_error", 0))
                            metric_name = "RMSE"
                            # Lower is better for RMSE
                            is_improvement = prep_metric < base_metric
                            diff = base_metric - prep_metric
                            diff_percentage = (diff / base_metric) * 100 if base_metric > 0 else 0
                        else:
                            # Skip if no comparable metric
                            continue
                        
                        # Add training time comparison
                        prep_time = prep_result.get("training_time", 0)
                        base_time = baseline.get("training_time", 0)
                        time_diff = base_time - prep_time
                        time_diff_percentage = (time_diff / base_time) * 100 if base_time > 0 else 0
                        
                        # Get preprocessing methods applied
                        preprocessing_methods = []
                        preprocessing_info = prep_result.get("preprocessing_info", {})
                        for step in preprocessing_info.get("steps", []):
                            if step.get("enabled", False):
                                step_type = step.get("type", "unknown")
                                strategy = step.get("params", {}).get("strategy", "")
                                preprocessing_methods.append(f"{step_type}-{strategy}")
                        
                        # Add to comparison table
                        comparison_table.append({
                            "dataset": dataset_name,
                            "metric": metric_name,
                            "baseline": base_metric,
                            "preprocessed": prep_metric,
                            "diff": diff,
                            "diff_percentage": diff_percentage,
                            "is_improvement": is_improvement,
                            "baseline_time": base_time,
                            "preprocessed_time": prep_time,
                            "time_diff_percentage": time_diff_percentage,
                            "preprocessing": ", ".join(preprocessing_methods) if preprocessing_methods else "N/A"
                        })
                
                # Output comparison results
                if comparison_table:
                    logger.info(f"\n  📊 Detailed Comparison:")
                    
                    # Sort by improvement percentage (descending)
                    comparison_table.sort(key=lambda x: x["diff_percentage"], reverse=True)
                    
                    for comp in comparison_table:
                        # Format improvement/degradation indicator
                        if comp["is_improvement"]:
                            change_symbol = "🔼"
                        else:
                            change_symbol = "🔽"
                        
                        logger.info(f"  {comp['dataset']}: {comp['metric']} {change_symbol}")
                        logger.info(f"    Baseline: {comp['baseline']:.4f}, Preprocessed: {comp['preprocessed']:.4f}")
                        logger.info(f"    Change: {comp['diff']:.4f} ({comp['diff_percentage']:.2f}%)")
                        logger.info(f"    Training time: {comp['time_diff_percentage']:.2f}% {comp['baseline_time']:.1f}s → {comp['preprocessed_time']:.1f}s")
                        logger.info(f"    Preprocessing applied: {comp['preprocessing']}")
                    
                    # Calculate overall statistics
                    improved_count = sum(1 for comp in comparison_table if comp["is_improvement"])
                    total_count = len(comparison_table)
                    improvement_rate = (improved_count / total_count) * 100 if total_count > 0 else 0
                    
                    # Calculate average improvement
                    avg_improvement = sum(comp["diff_percentage"] for comp in comparison_table) / total_count if total_count > 0 else 0
                    
                    logger.info(f"\n  📈 Overall Impact:")
                    logger.info(f"    Datasets improved: {improved_count}/{total_count} ({improvement_rate:.1f}%)")
                    logger.info(f"    Average metric change: {avg_improvement:.2f}%")
                    
                    # Save comparison results
                    comparison_file = summary_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(comparison_file, 'w') as f:
                        json.dump(comparison_table, f, indent=2, default=str)
                    
                    logger.info(f"  💾 Comparison details saved to: {comparison_file}")
                else:
                    logger.info("  ⚠️ No datasets could be compared with baseline results.")
            except Exception as e:
                logger.error(f"  ❌ Error comparing with baseline results: {str(e)}")
        else:
            logger.warning("  ⚠️ No baseline results found for comparison.")
    else:
        logger.info(f"  ⚠️ Either no preprocessed or no original datasets to compare.")
    
    if completed:
        logger.info(f"\n📊 Results Summary:")
        for result in completed:
            test_metrics = result.get("test_metrics", {})
            models = result.get("models_trained", 0)
            runtime = result.get("actual_runtime", 0) / 60
            preprocessed = "✓" if result.get("preprocessed", False) else "✗"

            # Handle different metric types for summary
            if 'accuracy' in test_metrics:
                metric_str = f"{test_metrics['accuracy']:.4f} acc"
            elif 'root_mean_squared_error' in test_metrics:
                metric_str = f"{abs(test_metrics['root_mean_squared_error']):.4f} RMSE"
            else:
                metric_str = "completed"

            logger.info(f"  {result['dataset_name']} [Preprocessed: {preprocessed}]: {metric_str}, {models} models, {runtime:.1f}min")
    
    if failed:
        logger.info(f"\n❌ Failed datasets:")
        for result in failed:
            logger.info(f"  {result['dataset_name']}: {result.get('error', 'Unknown error')}")
    
    logger.info(f"\n💾 Summary saved to: {summary_file}")

if __name__ == "__main__":
    main()
