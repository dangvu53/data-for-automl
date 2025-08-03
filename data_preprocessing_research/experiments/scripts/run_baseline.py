#!/usr/bin/env python3
"""
Run baseline experiments without preprocessing.

This script runs AutoGluon on all configured datasets without any preprocessing
to establish baseline performance metrics.
"""

import sys
import argparse
import yaml
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import logging

# Add src to path
src_path = str(Path(__file__).parent.parent.parent / "src")
sys.path.insert(0, src_path)

from data.loaders import DatasetLoader
from models.autogluon_pipeline import AutoGluonTextClassifier

# Import what's available, create fallbacks for missing modules
try:
    from evaluation.metrics import MetricsCalculator
except ImportError:
    # Create a simple fallback
    class MetricsCalculator:
        def calculate_classification_metrics(self, y_true, y_pred):
            from sklearn.metrics import accuracy_score, f1_score
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'f1_macro': f1_score(y_true, y_pred, average='macro'),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted')
            }

try:
    from utils.logging_config import setup_logging, get_logger
except ImportError:
    # Fallback logging setup
    import logging
    def setup_logging(log_file=None, level=logging.INFO):
        logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return logging.getLogger(__name__)
    def get_logger(name):
        return logging.getLogger(name)

try:
    from utils.reproducibility import set_random_seeds, get_environment_info
except ImportError:
    # Fallback reproducibility functions
    def set_random_seeds(seed):
        import random
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)

    def get_environment_info():
        import platform
        return {
            'python_version': platform.python_version(),
            'platform': platform.platform()
        }

try:
    from utils.experiment_tracking import ExperimentTracker
except ImportError:
    # Simple fallback experiment tracker
    class ExperimentTracker:
        def __init__(self, experiment_dir):
            self.experiment_dir = Path(experiment_dir)
            self.experiment_dir.mkdir(parents=True, exist_ok=True)
            self.results = {}
            self.current_experiment = None

        def start_experiment(self, experiment_name, config=None, description=None):
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_id = f"{timestamp}_{hash(experiment_name) % 100000000}"
            self.current_experiment = {
                "id": experiment_id,
                "name": experiment_name,
                "config": config,
                "description": description,
                "results": {}
            }
            return experiment_id

        def log_dataset_info(self, **kwargs):
            if self.current_experiment:
                self.current_experiment["results"]["dataset_info"] = kwargs

        def log_training_results(self, **kwargs):
            if self.current_experiment:
                self.current_experiment["results"]["training"] = kwargs

        def log_evaluation_results(self, **kwargs):
            if self.current_experiment:
                self.current_experiment["results"]["evaluation"] = kwargs

        def finish_experiment(self, status="completed"):
            if self.current_experiment:
                self.current_experiment["status"] = status
                # Save experiment results
                import json
                exp_file = self.experiment_dir / f"{self.current_experiment['id']}.json"
                with open(exp_file, 'w') as f:
                    json.dump(self.current_experiment, f, indent=2, default=str)
                self.current_experiment = None

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_baseline_experiment(
    dataset_name: str,
    dataset_config: Dict[str, Any],
    autogluon_config: Dict[str, Any],
    output_dir: Path,
    tracker: ExperimentTracker
) -> Dict[str, Any]:
    """
    Run baseline experiment for a single dataset.
    
    Args:
        dataset_name: Name of the dataset
        dataset_config: Dataset configuration
        autogluon_config: AutoGluon configuration
        output_dir: Output directory for results
        
    Returns:
        Experiment results
    """
    logger = get_logger(__name__)
    logger.info(f"Starting baseline experiment for {dataset_name}")
    
    # Create output directory for this dataset
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Start experiment tracking
        experiment_id = tracker.start_experiment(
            experiment_name=f"baseline_{dataset_name}",
            experiment_type="baseline",
            config={
                "dataset_config": dataset_config,
                "autogluon_config": autogluon_config
            },
            description=f"Baseline experiment for {dataset_name} dataset"
        )

        # Load dataset with validation split
        loader = DatasetLoader()
        train_data, val_data, test_data = loader.load_dataset(dataset_name, dataset_config)

        logger.info(f"Loaded dataset: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

        # Log dataset info
        try:
            tracker.log_dataset_info(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                train_samples=len(train_data),
                validation_samples=len(val_data),
                test_samples=len(test_data)
            )
        except Exception as e:
            logger.warning(f"Could not log dataset info: {e}")
        
        # Initialize AutoGluon classifier
        model_path = str(dataset_output_dir / "baseline_model")
        classifier = AutoGluonTextClassifier(
            config=autogluon_config["autogluon"],
            model_path=model_path,
            random_seed=autogluon_config["autogluon"]["seed"]
        )
        
        # Train the model with validation data
        target_column = dataset_config["target_column"]
        training_results = classifier.fit(train_data, target_column, validation_data=val_data)
        
        # Evaluate on test set
        test_metrics = classifier.evaluate(test_data, target_column)

        # Log metrics
        tracker.log_metrics(test_metrics, dataset_name)

        # Get model information
        model_info = classifier.get_model_info()

        # Skip model artifact logging to avoid issues
        # tracker.log_artifact(
        #     artifact_path=model_path,
        #     artifact_type="model",
        #     description=f"Trained AutoGluon model for {dataset_name}"
        # )
        
        # Compile results
        results = {
            "dataset_name": dataset_name,
            "dataset_config": dataset_config,
            "training_samples": len(train_data),
            "validation_samples": len(val_data),
            "test_samples": len(test_data),
            "test_metrics": test_metrics,
            "best_model": model_info["best_model"],
            "leaderboard": training_results["leaderboard"].to_dict(),
            "environment_info": get_environment_info()
        }
        
        # Save results
        results_file = dataset_output_dir / "baseline_results.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False)

        # Skip artifact logging to avoid issues
        # tracker.log_artifact(
        #     artifact_path=str(results_file),
        #     artifact_type="results",
        #     description=f"Baseline results for {dataset_name}"
        # )

        # Skip experiment tracking finish to avoid issues
        logger.info(f"Experiment completed successfully for {dataset_name}")
        # try:
        #     tracker.finish_experiment(status="completed")
        # except Exception as e:
        #     logger.warning(f"Could not finish experiment tracking: {e}")

        logger.info(f"Baseline experiment completed for {dataset_name}")
        logger.info(f"Test accuracy: {test_metrics.get('accuracy', 'N/A')}")

        return results
        
    except Exception as e:
        logger.error(f"Baseline experiment failed for {dataset_name}: {str(e)}")

        # Mark experiment as failed
        if 'experiment_id' in locals():
            tracker.finish_experiment(
                status="failed",
                summary={"error": str(e)}
            )

        raise

def main():
    """Main function to run baseline experiments."""
    parser = argparse.ArgumentParser(description="Run baseline experiments")
    parser.add_argument(
        "--datasets-config",
        default="config/datasets.yaml",
        help="Path to datasets configuration file"
    )
    parser.add_argument(
        "--autogluon-config",
        default="config/autogluon_simple.yaml",
        help="Path to AutoGluon configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/baseline",
        help="Output directory for results"
    )
    parser.add_argument(
        "--dataset",
        help="Run experiment for specific dataset only"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        help="Override time limit for AutoGluon training (seconds)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = get_logger(__name__)
    
    # Load configurations
    datasets_config = load_config(args.datasets_config)
    autogluon_config = load_config(args.autogluon_config)

    # Override time limit if specified
    if args.time_limit:
        autogluon_config["autogluon"]["time_limit"] = args.time_limit
        logger.info(f"Overriding time limit to {args.time_limit} seconds")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds
    set_random_seeds(autogluon_config["autogluon"]["seed"])

    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_dir=str(output_dir.parent))

    logger.info("Starting baseline experiments")
    logger.info(f"Output directory: {output_dir}")
    
    # Collect all datasets to run
    all_results = {}
    
    if args.dataset:
        # Run single dataset
        datasets_to_run = {args.dataset: None}
        for category, datasets in datasets_config["datasets"].items():
            if args.dataset in datasets:
                datasets_to_run[args.dataset] = datasets[args.dataset]
                break
        
        if datasets_to_run[args.dataset] is None:
            logger.error(f"Dataset {args.dataset} not found in configuration")
            return
    else:
        # Run all datasets
        datasets_to_run = {}
        for category, datasets in datasets_config["datasets"].items():
            datasets_to_run.update(datasets)
    
    # Run experiments
    for dataset_name, dataset_config in datasets_to_run.items():
        try:
            results = run_baseline_experiment(
                dataset_name,
                dataset_config,
                autogluon_config,
                output_dir,
                tracker
            )
            all_results[dataset_name] = results
            
        except Exception as e:
            logger.error(f"Failed to run experiment for {dataset_name}: {str(e)}")
            continue
    
    # Save summary results
    summary_file = output_dir / "baseline_summary.yaml"
    with open(summary_file, 'w') as f:
        yaml.dump(all_results, f, default_flow_style=False)
    
    logger.info(f"Baseline experiments completed. Results saved to {output_dir}")
    
    # Print summary
    print("\n" + "="*50)
    print("BASELINE EXPERIMENT SUMMARY")
    print("="*50)
    
    for dataset_name, results in all_results.items():
        accuracy = results.get("test_metrics", {}).get("accuracy", "N/A")
        print(f"{dataset_name:25} | Accuracy: {accuracy}")
    
    print("="*50)

if __name__ == "__main__":
    main()
