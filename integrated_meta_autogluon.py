#!/usr/bin/env python3
"""
Integrated Meta-Learning + AutoGluon Training Script

This script combines the meta-learning framework with AutoGluon training:
1. Uses meta-learning to discover optimal preprocessing sequences (training data only)
2. Applies the best sequence to preprocess the data
3. Trains AutoGluon models on the processed data
4. Evaluates performance with comprehensive timing
"""

import os
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime
from datasets import load_dataset
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import our meta-learning framework
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

class MetaLearningAutoGluonTrainer:
    """Integrated meta-learning and AutoGluon trainer"""
    
    def __init__(self, output_dir="./meta_autogluon_results"):
        self.output_dir = output_dir
        self.results = {}
        self.predictions_dir = os.path.join(output_dir, "predictions")
        self.confusion_matrices_dir = os.path.join(output_dir, "confusion_matrices")
        self.meta_learning_dir = os.path.join(output_dir, "meta_learning")
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.confusion_matrices_dir, exist_ok=True)
        os.makedirs(self.meta_learning_dir, exist_ok=True)
    
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
        
        # Create framework with reasonable settings
        framework = MetaLearningFramework(
            embedder=embedder,
            operations=operations,
            population_size=15,  # Reasonable size for real experiments
            generations=8,       # Reasonable number of generations
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_sequence_length=3,
            output_dir=self.meta_learning_dir
        )
        
        return framework
    
    def run_meta_learning(self, dataset_name: str, train_df: pd.DataFrame, target_col: str):
        """Run meta-learning to discover optimal preprocessing sequence"""
        logger.info(f"Running meta-learning for {dataset_name}...")
        meta_start_time = time.time()
        
        # Create framework
        framework = self.create_meta_learning_framework()
        
        # Use subset for faster meta-learning
        subset_size = min(2000, len(train_df))  # Larger subset for better results
        train_subset = train_df.sample(n=subset_size, random_state=42)
        
        logger.info(f"Using training subset: {len(train_subset)} samples for meta-learning")
        logger.info("IMPORTANT: Only training data is used for meta-learning (no data leakage)")
        
        # Run meta-learning
        best_sequence = framework.run_meta_learning(train_subset, target_col)
        
        meta_time = time.time() - meta_start_time
        logger.info(f"Meta-learning completed in {format_time(meta_time)}")
        
        if best_sequence:
            logger.info(f"Best sequence found for {dataset_name}:")
            logger.info(f"Fitness score: {best_sequence.fitness_score:.4f}")
            logger.info("Operations:")
            for i, op in enumerate(best_sequence.operations):
                logger.info(f"  {i+1}. {op['operation_class']} with params: {op['parameters']}")
            
            # Save the best sequence
            import json
            sequence_file = os.path.join(self.meta_learning_dir, f"{dataset_name}_best_sequence.json")
            with open(sequence_file, 'w') as f:
                json.dump({
                    'dataset': dataset_name,
                    'operations': best_sequence.operations,
                    'fitness_score': best_sequence.fitness_score,
                    'meta_learning_time': meta_time
                }, f, indent=2)
            
            return best_sequence, framework
        else:
            logger.warning(f"No best sequence found for {dataset_name}")
            return None, framework
    
    def apply_preprocessing(self, framework, best_sequence, train_df, val_df, test_df, dataset_name):
        """Apply the discovered preprocessing sequence to all data splits"""
        logger.info(f"Applying preprocessing sequence to {dataset_name} data...")
        processing_start_time = time.time()
        
        # Apply to training data
        processed_train, train_metadata = framework.apply_sequence(train_df, best_sequence)
        
        # Apply to validation data (using same sequence discovered from training)
        processed_val, val_metadata = framework.apply_sequence(val_df, best_sequence)
        
        # Apply to test data (using same sequence discovered from training)
        processed_test, test_metadata = framework.apply_sequence(test_df, best_sequence)
        
        processing_time = time.time() - processing_start_time
        logger.info(f"Data preprocessing completed in {format_time(processing_time)}")
        
        logger.info(f"Data sizes after preprocessing:")
        logger.info(f"  Train: {len(train_df)} -> {len(processed_train)}")
        logger.info(f"  Val: {len(val_df)} -> {len(processed_val)}")
        logger.info(f"  Test: {len(test_df)} -> {len(processed_test)}")
        
        return processed_train, processed_val, processed_test, processing_time

    def train_text_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train TabularPredictor for text data using AutoGluon defaults"""
        logger.info(f"Using TabularPredictor for text data for {dataset_name}")

        predictor = TabularPredictor(
            label=target_col,
            path=model_dir,
            problem_type='multiclass',
            eval_metric='accuracy'
        )

        # Use AutoGluon's default settings - no custom model specifications
        logger.info("Using AutoGluon's default models and settings")

        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',  # Use AutoGluon's default preset
            time_limit=3000,
            verbosity=2,
            # Allow AutoGluon to use more memory
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
            # No included_model_types - let AutoGluon choose default models
        )

        training_time = time.time() - start_time
        logger.info(f"Text model training completed for {dataset_name} in {format_time(training_time)}")
        return predictor, training_time

    def train_tabular_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train standard TabularPredictor using AutoGluon defaults"""
        logger.info(f"Using TabularPredictor for {dataset_name}")

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
            presets='medium_quality',  # Use AutoGluon's default preset
            time_limit=3000,
            verbosity=2,
            # Allow AutoGluon to use more memory
            ag_args_fit={'ag.max_memory_usage_ratio': 5}
            # No included_model_types - let AutoGluon choose default models
        )

        training_time = time.time() - start_time
        logger.info(f"Tabular model training completed for {dataset_name} in {format_time(training_time)}")
        return predictor, training_time

    def save_predictions_and_confusion_matrix(self, predictor, test_df, target_col, dataset_name):
        """Save predictions as CSV and create confusion matrix for classification tasks"""
        logger.info(f"Saving predictions and creating confusion matrix for {dataset_name}...")

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
            pred_file = os.path.join(self.predictions_dir, f"{dataset_name}_predictions.csv")
            pred_df.to_csv(pred_file, index=False)
            logger.info(f"Predictions saved to: {pred_file}")

            # Create confusion matrix
            y_true = test_df[target_col]
            y_pred = predictions

            # Create confusion matrix
            cm = confusion_matrix(y_true, y_pred)

            # Get unique labels
            labels = sorted(list(set(y_true) | set(y_pred)))

            # Create confusion matrix plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - {dataset_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')

            # Save confusion matrix
            cm_file = os.path.join(self.confusion_matrices_dir, f"{dataset_name}_confusion_matrix.png")
            plt.savefig(cm_file, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Confusion matrix saved to: {cm_file}")

            # Save classification report
            report = classification_report(y_true, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            report_file = os.path.join(self.confusion_matrices_dir, f"{dataset_name}_classification_report.csv")
            report_df.to_csv(report_file)
            logger.info(f"Classification report saved to: {report_file}")

            return predictions, cm

        except Exception as e:
            logger.error(f"Error saving predictions/confusion matrix for {dataset_name}: {e}")
            return None, None

    def run_complete_experiment(self, dataset_name: str, load_func):
        """Run complete experiment: meta-learning + preprocessing + AutoGluon training"""
        experiment_start_time = time.time()
        logger.info(f"\n{'='*80}")
        logger.info(f"STARTING COMPLETE EXPERIMENT: {dataset_name.upper()}")
        logger.info(f"Experiment start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*80}")

        try:
            # 1. Load dataset
            train_df, val_df, test_df, target_col = load_func()
            if train_df is None:
                logger.error(f"Failed to load {dataset_name}")
                return None

            # 2. Run meta-learning to discover optimal preprocessing
            best_sequence, framework = self.run_meta_learning(dataset_name, train_df, target_col)

            if best_sequence is None:
                logger.warning(f"No preprocessing sequence found for {dataset_name}, using original data")
                processed_train, processed_val, processed_test = train_df, val_df, test_df
                processing_time = 0
            else:
                # 3. Apply preprocessing to all data splits
                processed_train, processed_val, processed_test, processing_time = self.apply_preprocessing(
                    framework, best_sequence, train_df, val_df, test_df, dataset_name
                )

            # 4. Train AutoGluon model on processed data
            model_dir = os.path.join(self.output_dir, f"{dataset_name}_model")
            os.makedirs(model_dir, exist_ok=True)

            autogluon_start_time = time.time()
            logger.info(f"Training AutoGluon model for {dataset_name}...")

            # Choose appropriate training method based on dataset
            if dataset_name == 'anli_r1':
                predictor, training_time = self.train_text_model(
                    processed_train, processed_val, target_col, dataset_name, model_dir, autogluon_start_time
                )
            elif dataset_name == 'casehold':
                predictor, training_time = self.train_tabular_model(
                    processed_train, processed_val, target_col, dataset_name, model_dir, autogluon_start_time
                )
            else:
                # Default to tabular
                predictor, training_time = self.train_tabular_model(
                    processed_train, processed_val, target_col, dataset_name, model_dir, autogluon_start_time
                )

            # 5. Evaluate model
            eval_start_time = time.time()
            logger.info(f"Evaluating {dataset_name} model...")

            # Get performance metrics
            performance = predictor.evaluate(processed_test, silent=True)
            leaderboard = predictor.leaderboard(processed_test, silent=True)

            eval_time = time.time() - eval_start_time
            logger.info(f"Evaluation completed in {format_time(eval_time)}")

            # 6. Save predictions and confusion matrix
            predictions, confusion_matrix = self.save_predictions_and_confusion_matrix(
                predictor, processed_test, target_col, dataset_name
            )

            # 7. Compile results
            total_experiment_time = time.time() - experiment_start_time

            results = {
                'dataset': dataset_name,
                'original_data_sizes': {
                    'train': len(train_df),
                    'val': len(val_df),
                    'test': len(test_df)
                },
                'processed_data_sizes': {
                    'train': len(processed_train),
                    'val': len(processed_val),
                    'test': len(processed_test)
                },
                'best_sequence': best_sequence.operations if best_sequence else None,
                'meta_learning_fitness': best_sequence.fitness_score if best_sequence else None,
                'autogluon_performance': performance,
                'best_model': leaderboard.iloc[0]['model'] if len(leaderboard) > 0 else 'Unknown',
                'test_score': leaderboard.iloc[0]['score_test'] if len(leaderboard) > 0 else None,
                'timing': {
                    'preprocessing_time': processing_time,
                    'autogluon_training_time': training_time,
                    'evaluation_time': eval_time,
                    'total_experiment_time': total_experiment_time
                }
            }

            logger.info(f"\n{'='*60}")
            logger.info(f"EXPERIMENT RESULTS FOR {dataset_name.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"Meta-learning fitness: {results['meta_learning_fitness']}")
            logger.info(f"AutoGluon performance: {results['autogluon_performance']}")
            logger.info(f"Best model: {results['best_model']}")
            logger.info(f"Test score: {results['test_score']}")
            logger.info(f"Preprocessing time: {format_time(processing_time)}")
            logger.info(f"AutoGluon training time: {format_time(training_time)}")
            logger.info(f"Total experiment time: {format_time(total_experiment_time)}")
            logger.info(f"{'='*60}")

            return results

        except Exception as e:
            logger.error(f"Error in experiment for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_all_experiments(self):
        """Run experiments on all datasets"""
        total_start_time = time.time()
        logger.info("="*100)
        logger.info("STARTING ALL META-LEARNING + AUTOGLUON EXPERIMENTS")
        logger.info(f"Total start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*100)

        datasets = [
            ("casehold", self.load_casehold_dataset),
            ("anli_r1", self.load_anli_dataset)
        ]

        all_results = []

        for dataset_name, load_func in datasets:
            result = self.run_complete_experiment(dataset_name, load_func)
            if result:
                all_results.append(result)
                self.results[dataset_name] = result

        total_time = time.time() - total_start_time
        logger.info("="*100)
        logger.info("ALL EXPERIMENTS COMPLETED")
        logger.info(f"Total end time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Total time for all experiments: {format_time(total_time)}")
        logger.info("="*100)

        # Save comprehensive results
        import json
        results_file = os.path.join(self.output_dir, 'complete_experiment_results.json')
        with open(results_file, 'w') as f:
            # Convert any non-serializable objects to strings
            serializable_results = {}
            for dataset, result in self.results.items():
                serializable_results[dataset] = {
                    k: str(v) if not isinstance(v, (dict, list, str, int, float, type(None))) else v
                    for k, v in result.items()
                }
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Complete results saved to: {results_file}")

        return all_results

def main():
    """Main function to run the integrated experiments"""
    main_start_time = time.time()
    logger.info("Starting Integrated Meta-Learning + AutoGluon Experiments")
    logger.info(f"Main start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Create trainer
    trainer = MetaLearningAutoGluonTrainer(output_dir="./meta_autogluon_results")

    # Run all experiments
    results = trainer.run_all_experiments()

    # Print final summary
    logger.info("\n" + "="*100)
    logger.info("FINAL EXPERIMENT SUMMARY")
    logger.info("="*100)

    for result in results:
        dataset = result['dataset']
        logger.info(f"\n{dataset.upper()}:")
        logger.info(f"  Meta-learning fitness: {result['meta_learning_fitness']}")
        logger.info(f"  AutoGluon performance: {result['autogluon_performance']}")
        logger.info(f"  Best model: {result['best_model']}")
        logger.info(f"  Total time: {format_time(result['timing']['total_experiment_time'])}")

    main_total_time = time.time() - main_start_time
    logger.info(f"\nAll experiments completed!")
    logger.info(f"Total execution time: {format_time(main_total_time)}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
