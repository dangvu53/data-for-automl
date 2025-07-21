#!/usr/bin/env python3
"""
Test Script for Meta-Learning Framework

This script demonstrates the meta-learning framework on CaseHold and ANLI R1 datasets.
It processes only the training data to discover optimal preprocessing sequences,
then evaluates the final pipeline on AutoGluon.
"""

import os
import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
from datasets import load_dataset
from meta_learning_framework import MetaLearningFramework, TextEmbedder
from operation_space import TextCleaner, TextLengthFilter, DifficultyBasedSelector, TextAugmenter

# Configure logging with timestamps
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_time(seconds):
    """Format seconds into human readable time"""
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

class MetaLearningExperiment:
    """Experiment runner for meta-learning framework"""
    
    def __init__(self, output_dir: str = "./meta_learning_experiments"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def load_casehold_dataset(self):
        """Load and prepare CaseHold dataset"""
        start_time = time.time()
        logger.info("Loading CaseHold dataset...")

        try:
            dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
            
            def prepare_casehold_data(split_data):
                data = []
                for item in split_data:
                    # Combine citing prompt with all holdings for context
                    text_features = item['citing_prompt']
                    for i in range(5):  # holdings 0-4
                        text_features += f" [HOLDING_{i}] " + item[f'holding_{i}']
                    
                    data.append({
                        'text': text_features,
                        'label': item['label']
                    })
                return pd.DataFrame(data)
            
            train_df = prepare_casehold_data(dataset['train'])
            val_df = prepare_casehold_data(dataset['validation'])
            test_df = prepare_casehold_data(dataset['test'])
            
            load_time = time.time() - start_time
            logger.info(f"CaseHold loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"CaseHold loading time: {format_time(load_time)}")
            return train_df, val_df, test_df, 'label'

        except Exception as e:
            logger.error(f"Error loading CaseHold: {e}")
            return None, None, None, None
    
    def load_anli_dataset(self):
        """Load and prepare ANLI R1 dataset"""
        start_time = time.time()
        logger.info("Loading ANLI R1 dataset...")

        try:
            dataset = load_dataset("facebook/anli")
            
            def prepare_anli_data(split_data):
                data = []
                for item in split_data:
                    # Combine premise and hypothesis for NLI
                    text_features = f"[PREMISE] {item['premise']} [HYPOTHESIS] {item['hypothesis']}"
                    
                    data.append({
                        'text': text_features,
                        'premise': item['premise'],
                        'hypothesis': item['hypothesis'],
                        'label': item['label']
                    })
                return pd.DataFrame(data)
            
            train_df = prepare_anli_data(dataset['train_r1'])
            val_df = prepare_anli_data(dataset['dev_r1'])
            test_df = prepare_anli_data(dataset['test_r1'])
            
            load_time = time.time() - start_time
            logger.info(f"ANLI R1 loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"ANLI R1 loading time: {format_time(load_time)}")
            return train_df, val_df, test_df, 'label'

        except Exception as e:
            logger.error(f"Error loading ANLI: {e}")
            return None, None, None, None
    
    def create_meta_learning_framework(self):
        """Create and configure the meta-learning framework"""
        logger.info("Creating meta-learning framework...")
        
        # Create text embedder
        embedder = TextEmbedder(model_name="all-MiniLM-L6-v2")
        
        # Create operation space
        operations = [
            TextCleaner(),
            TextLengthFilter(),
            DifficultyBasedSelector(),
            TextAugmenter()
        ]
        
        # Create framework with smaller population for faster testing
        framework = MetaLearningFramework(
            embedder=embedder,
            operations=operations,
            population_size=10,  # Small for testing
            generations=5,       # Small for testing
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_sequence_length=3,  # Keep sequences short
            output_dir=self.output_dir
        )
        
        return framework
    
    def run_experiment(self, dataset_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame,
                      test_df: pd.DataFrame, target_col: str):
        """Run meta-learning experiment on a dataset - ONLY processes training data"""
        experiment_start_time = time.time()
        logger.info(f"Starting meta-learning experiment on {dataset_name}")
        logger.info(f"Experiment start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Create framework
        framework_start_time = time.time()
        framework = self.create_meta_learning_framework()
        framework_creation_time = time.time() - framework_start_time
        logger.info(f"Framework creation time: {format_time(framework_creation_time)}")

        # Use only a subset of training data for faster experimentation
        subset_size = min(1000, len(train_df))
        train_subset = train_df.sample(n=subset_size, random_state=42)

        logger.info(f"Using training subset: {len(train_subset)} samples")
        logger.info("IMPORTANT: Only training data is used for meta-learning (no data leakage)")

        # Run meta-learning on TRAINING DATA ONLY
        meta_learning_start_time = time.time()
        best_sequence = framework.run_meta_learning(train_subset, target_col)
        meta_learning_time = time.time() - meta_learning_start_time
        logger.info(f"Meta-learning time: {format_time(meta_learning_time)}")

        if best_sequence:
            logger.info(f"Best sequence found for {dataset_name}:")
            logger.info(f"Fitness score: {best_sequence.fitness_score:.4f}")
            logger.info("Operations:")
            for i, op in enumerate(best_sequence.operations):
                logger.info(f"  {i+1}. {op['operation_class']} with params: {op['parameters']}")

            # Apply best sequence ONLY to training data for now
            processing_start_time = time.time()
            logger.info("Applying best sequence to full training data...")
            processed_train, train_metadata = framework.apply_sequence(train_df, best_sequence)
            processing_time = time.time() - processing_start_time
            logger.info(f"Data processing time: {format_time(processing_time)}")

            logger.info(f"Training data size after processing: {len(train_df)} -> {len(processed_train)}")

            # Save processed training dataset
            save_start_time = time.time()
            dataset_dir = os.path.join(self.output_dir, f"{dataset_name}_processed")
            os.makedirs(dataset_dir, exist_ok=True)

            processed_train.to_csv(os.path.join(dataset_dir, "train_processed.csv"), index=False)

            # Save the original val/test data unchanged (for later AutoGluon evaluation)
            val_df.to_csv(os.path.join(dataset_dir, "val_original.csv"), index=False)
            test_df.to_csv(os.path.join(dataset_dir, "test_original.csv"), index=False)

            # Save the best sequence for later application to val/test
            import json
            sequence_file = os.path.join(dataset_dir, "best_sequence.json")
            with open(sequence_file, 'w') as f:
                json.dump({
                    'operations': best_sequence.operations,
                    'fitness_score': best_sequence.fitness_score
                }, f, indent=2)

            save_time = time.time() - save_start_time
            logger.info(f"Data saving time: {format_time(save_time)}")
            logger.info(f"Processed training data and metadata saved to {dataset_dir}")

            total_experiment_time = time.time() - experiment_start_time
            logger.info(f"Total experiment time for {dataset_name}: {format_time(total_experiment_time)}")

            return best_sequence, processed_train, val_df, test_df

        else:
            total_experiment_time = time.time() - experiment_start_time
            logger.warning(f"No best sequence found for {dataset_name}")
            logger.info(f"Total experiment time for {dataset_name}: {format_time(total_experiment_time)}")
            return None, train_df, val_df, test_df
    
    def evaluate_with_autogluon(self, dataset_name: str, train_df: pd.DataFrame,
                               val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str):
        """Evaluate the processed data with AutoGluon"""
        autogluon_start_time = time.time()
        logger.info(f"Evaluating {dataset_name} with AutoGluon...")
        logger.info(f"AutoGluon evaluation start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        try:
            from autogluon.tabular import TabularPredictor
            import tempfile
            
            # Create temporary directory for AutoGluon
            with tempfile.TemporaryDirectory() as temp_dir:
                predictor = TabularPredictor(
                    label=target_col,
                    path=temp_dir,
                    problem_type='multiclass',
                    eval_metric='accuracy',
                    verbosity=1
                )
                
                # Train with processed data
                training_start_time = time.time()
                logger.info("Training AutoGluon model...")
                predictor.fit(
                    train_data=train_df,
                    tuning_data=val_df,
                    time_limit=300,  # 5 minutes
                    presets='medium_quality_faster_train'
                )
                training_time = time.time() - training_start_time
                logger.info(f"AutoGluon training time: {format_time(training_time)}")

                # Evaluate on test set
                eval_start_time = time.time()
                logger.info("Evaluating on test set...")
                performance = predictor.evaluate(test_df, silent=True)
                eval_time = time.time() - eval_start_time
                logger.info(f"AutoGluon evaluation time: {format_time(eval_time)}")

                logger.info(f"AutoGluon performance on {dataset_name}: {performance}")

                # Get leaderboard
                leaderboard = predictor.leaderboard(test_df, silent=True)
                logger.info(f"Best model: {leaderboard.iloc[0]['model']}")
                logger.info(f"Test score: {leaderboard.iloc[0]['score_test']}")

                total_autogluon_time = time.time() - autogluon_start_time
                logger.info(f"Total AutoGluon time for {dataset_name}: {format_time(total_autogluon_time)}")

                return performance, leaderboard
                
        except Exception as e:
            logger.error(f"Error evaluating with AutoGluon: {e}")
            return None, None
    
    def run_all_experiments(self):
        """Run experiments on all datasets"""
        total_start_time = time.time()
        logger.info("="*80)
        logger.info("STARTING META-LEARNING EXPERIMENTS")
        logger.info(f"Total experiment start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*80)

        datasets = [
            # ("casehold", self.load_casehold_dataset),
            ("anli_r1", self.load_anli_dataset)  # Start with just ANLI for testing
        ]

        results = {}

        for dataset_name, load_func in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"PROCESSING {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            
            try:
                # Load dataset
                train_df, val_df, test_df, target_col = load_func()
                
                if train_df is None:
                    logger.warning(f"Skipping {dataset_name} due to loading error")
                    continue
                
                # Run meta-learning experiment
                best_sequence, processed_train, processed_val, processed_test = self.run_experiment(
                    dataset_name, train_df, val_df, test_df, target_col
                )
                
                # Evaluate with AutoGluon
                performance, leaderboard = self.evaluate_with_autogluon(
                    dataset_name, processed_train, processed_val, processed_test, target_col
                )
                
                results[dataset_name] = {
                    'best_sequence': best_sequence,
                    'autogluon_performance': performance,
                    'leaderboard': leaderboard
                }
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                results[dataset_name] = {'error': str(e)}

        total_time = time.time() - total_start_time
        logger.info("="*80)
        logger.info("ALL EXPERIMENTS COMPLETED")
        logger.info(f"Total experiment end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time for all experiments: {format_time(total_time)}")
        logger.info("="*80)

        return results

def main():
    """Main function to run the meta-learning experiments"""
    main_start_time = time.time()
    logger.info("Starting Meta-Learning Framework Experiments")
    logger.info(f"Main start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create experiment runner
    experiment = MetaLearningExperiment(output_dir="./meta_learning_experiments")

    # Run all experiments
    results = experiment.run_all_experiments()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*60)
    
    for dataset_name, result in results.items():
        logger.info(f"\n{dataset_name.upper()}:")
        if 'error' in result:
            logger.info(f"  Error: {result['error']}")
        else:
            if result['best_sequence']:
                logger.info(f"  Best fitness: {result['best_sequence'].fitness_score:.4f}")
                logger.info(f"  Operations: {len(result['best_sequence'].operations)}")
            if result['autogluon_performance']:
                logger.info(f"  AutoGluon performance: {result['autogluon_performance']}")

    main_total_time = time.time() - main_start_time
    logger.info(f"\nExperiments completed!")
    logger.info(f"Total main execution time: {format_time(main_total_time)}")
    logger.info(f"Main end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
