"""
Benchmarking framework for comparing baseline vs preprocessed performance.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import yaml
import time
import logging
from datetime import datetime

from .metrics import MetricsCalculator
from ..utils.logging_config import get_logger
from ..utils.reproducibility import set_random_seeds, get_environment_info
from ..utils.visualization import VisualizationUtils

logger = get_logger(__name__)

class BenchmarkRunner:
    """
    Comprehensive benchmarking system for preprocessing experiments.
    """
    
    def __init__(
        self,
        output_dir: str = "experiments/results/benchmarks",
        random_seed: int = 42
    ):
        """
        Initialize the benchmark runner.
        
        Args:
            output_dir: Directory to save benchmark results
            random_seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.random_seed = random_seed
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = VisualizationUtils(str(self.output_dir / "plots"))
        
        # Set random seeds
        set_random_seeds(random_seed)
        
        logger.info(f"BenchmarkRunner initialized with output dir: {output_dir}")
    
    def run_comprehensive_benchmark(
        self,
        baseline_results: Dict[str, Dict[str, Any]],
        preprocessing_results: Dict[str, Dict[str, Any]],
        experiment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run comprehensive benchmark comparing baseline vs preprocessing.
        
        Args:
            baseline_results: Results from baseline experiments
            preprocessing_results: Results from preprocessing experiments
            experiment_config: Experiment configuration
            
        Returns:
            Comprehensive benchmark results
        """
        logger.info("Starting comprehensive benchmark analysis")
        
        benchmark_results = {
            'experiment_info': {
                'timestamp': datetime.now().isoformat(),
                'random_seed': self.random_seed,
                'environment': get_environment_info(),
                'config': experiment_config
            },
            'dataset_comparisons': {},
            'overall_statistics': {},
            'statistical_tests': {},
            'visualizations': {}
        }
        
        # Compare each dataset
        for dataset_name in baseline_results.keys():
            if dataset_name in preprocessing_results:
                comparison = self._compare_dataset_results(
                    dataset_name,
                    baseline_results[dataset_name],
                    preprocessing_results[dataset_name]
                )
                benchmark_results['dataset_comparisons'][dataset_name] = comparison
        
        # Calculate overall statistics
        benchmark_results['overall_statistics'] = self._calculate_overall_statistics(
            benchmark_results['dataset_comparisons']
        )
        
        # Perform statistical significance tests
        benchmark_results['statistical_tests'] = self._perform_statistical_tests(
            baseline_results, preprocessing_results
        )
        
        # Generate visualizations
        self._generate_benchmark_visualizations(benchmark_results)
        
        # Save results
        self._save_benchmark_results(benchmark_results)
        
        logger.info("Comprehensive benchmark analysis completed")
        return benchmark_results
    
    def _compare_dataset_results(
        self,
        dataset_name: str,
        baseline_result: Dict[str, Any],
        preprocessing_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare results for a single dataset.
        
        Args:
            dataset_name: Name of the dataset
            baseline_result: Baseline experiment result
            preprocessing_result: Preprocessing experiment result
            
        Returns:
            Comparison results for the dataset
        """
        logger.info(f"Comparing results for dataset: {dataset_name}")
        
        baseline_metrics = baseline_result.get('test_metrics', {})
        preprocessing_metrics = preprocessing_result.get('test_metrics', {})
        
        # Calculate metric improvements
        comparison = self.metrics_calculator.compare_metrics(
            baseline_metrics, preprocessing_metrics
        )
        
        # Add dataset-specific information
        comparison.update({
            'dataset_name': dataset_name,
            'baseline_metrics': baseline_metrics,
            'preprocessing_metrics': preprocessing_metrics,
            'training_time_baseline': baseline_result.get('training_time', 0),
            'training_time_preprocessing': preprocessing_result.get('training_time', 0),
            'data_reduction': self._calculate_data_reduction(
                baseline_result, preprocessing_result
            )
        })
        
        # Calculate cost-benefit analysis
        comparison['cost_benefit'] = self._calculate_cost_benefit(comparison)
        
        return comparison
    
    def _calculate_data_reduction(
        self,
        baseline_result: Dict[str, Any],
        preprocessing_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate data reduction statistics."""
        baseline_samples = baseline_result.get('training_samples', 0)
        preprocessing_samples = preprocessing_result.get('training_samples', 0)
        
        if baseline_samples > 0:
            reduction_ratio = (baseline_samples - preprocessing_samples) / baseline_samples
            reduction_percentage = reduction_ratio * 100
        else:
            reduction_ratio = 0
            reduction_percentage = 0
        
        return {
            'original_samples': baseline_samples,
            'processed_samples': preprocessing_samples,
            'reduction_ratio': reduction_ratio,
            'reduction_percentage': reduction_percentage
        }
    
    def _calculate_cost_benefit(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate cost-benefit analysis.
        
        Args:
            comparison: Dataset comparison results
            
        Returns:
            Cost-benefit analysis
        """
        # Performance benefit (accuracy improvement)
        accuracy_improvement = comparison['improvements'].get('accuracy', 0)
        
        # Time cost (additional training time)
        time_baseline = comparison.get('training_time_baseline', 0)
        time_preprocessing = comparison.get('training_time_preprocessing', 0)
        time_overhead = time_preprocessing - time_baseline
        
        # Data cost (data reduction)
        data_reduction = comparison['data_reduction']['reduction_percentage']
        
        # Calculate benefit-to-cost ratio
        if time_overhead > 0:
            benefit_cost_ratio = accuracy_improvement / (time_overhead / 60)  # per minute
        else:
            benefit_cost_ratio = float('inf') if accuracy_improvement > 0 else 0
        
        return {
            'accuracy_improvement': accuracy_improvement,
            'time_overhead_seconds': time_overhead,
            'data_reduction_percentage': data_reduction,
            'benefit_cost_ratio': benefit_cost_ratio,
            'recommendation': self._get_recommendation(
                accuracy_improvement, time_overhead, data_reduction
            )
        }
    
    def _get_recommendation(
        self,
        accuracy_improvement: float,
        time_overhead: float,
        data_reduction: float
    ) -> str:
        """Generate recommendation based on cost-benefit analysis."""
        if accuracy_improvement > 0.05:  # 5% improvement
            if time_overhead < 300:  # Less than 5 minutes overhead
                return "Highly Recommended"
            elif time_overhead < 900:  # Less than 15 minutes overhead
                return "Recommended"
            else:
                return "Consider if time allows"
        elif accuracy_improvement > 0.01:  # 1% improvement
            if time_overhead < 60:  # Less than 1 minute overhead
                return "Recommended"
            else:
                return "Marginal benefit"
        else:
            return "Not recommended"
    
    def _calculate_overall_statistics(
        self,
        dataset_comparisons: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate overall statistics across all datasets."""
        logger.info("Calculating overall statistics")
        
        # Collect improvements across all datasets
        accuracy_improvements = []
        f1_improvements = []
        time_overheads = []
        data_reductions = []
        
        for comparison in dataset_comparisons.values():
            accuracy_improvements.append(comparison['improvements'].get('accuracy', 0))
            f1_improvements.append(comparison['improvements'].get('f1_weighted', 0))
            time_overheads.append(comparison['cost_benefit']['time_overhead_seconds'])
            data_reductions.append(comparison['data_reduction']['reduction_percentage'])
        
        # Calculate statistics
        stats = {
            'accuracy_improvement': {
                'mean': np.mean(accuracy_improvements),
                'std': np.std(accuracy_improvements),
                'median': np.median(accuracy_improvements),
                'min': np.min(accuracy_improvements),
                'max': np.max(accuracy_improvements),
                'positive_count': sum(1 for x in accuracy_improvements if x > 0),
                'total_count': len(accuracy_improvements)
            },
            'f1_improvement': {
                'mean': np.mean(f1_improvements),
                'std': np.std(f1_improvements),
                'median': np.median(f1_improvements),
                'min': np.min(f1_improvements),
                'max': np.max(f1_improvements)
            },
            'time_overhead': {
                'mean': np.mean(time_overheads),
                'std': np.std(time_overheads),
                'median': np.median(time_overheads),
                'total_minutes': sum(time_overheads) / 60
            },
            'data_reduction': {
                'mean': np.mean(data_reductions),
                'std': np.std(data_reductions),
                'median': np.median(data_reductions)
            }
        }
        
        # Overall recommendation
        positive_improvements = stats['accuracy_improvement']['positive_count']
        total_datasets = stats['accuracy_improvement']['total_count']
        success_rate = positive_improvements / total_datasets if total_datasets > 0 else 0
        
        stats['overall_recommendation'] = {
            'success_rate': success_rate,
            'recommendation': (
                "Preprocessing generally beneficial" if success_rate > 0.7 else
                "Preprocessing moderately beneficial" if success_rate > 0.5 else
                "Preprocessing benefits are dataset-specific"
            )
        }
        
        return stats
    
    def _perform_statistical_tests(
        self,
        baseline_results: Dict[str, Dict[str, Any]],
        preprocessing_results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Perform statistical significance tests."""
        logger.info("Performing statistical significance tests")
        
        # Collect scores for paired testing
        baseline_accuracies = []
        preprocessing_accuracies = []
        
        for dataset_name in baseline_results.keys():
            if dataset_name in preprocessing_results:
                baseline_acc = baseline_results[dataset_name]['test_metrics'].get('accuracy', 0)
                preprocessing_acc = preprocessing_results[dataset_name]['test_metrics'].get('accuracy', 0)
                
                baseline_accuracies.append(baseline_acc)
                preprocessing_accuracies.append(preprocessing_acc)
        
        # Perform tests
        tests = {}
        
        if len(baseline_accuracies) > 1:
            # Wilcoxon signed-rank test
            tests['wilcoxon'] = self.metrics_calculator.statistical_significance_test(
                baseline_accuracies, preprocessing_accuracies, test='wilcoxon'
            )
            
            # Effect size
            tests['effect_size'] = self.metrics_calculator.calculate_effect_size(
                baseline_accuracies, preprocessing_accuracies
            )
        
        return tests
    
    def _generate_benchmark_visualizations(self, benchmark_results: Dict[str, Any]) -> None:
        """Generate comprehensive visualizations."""
        logger.info("Generating benchmark visualizations")
        
        # Extract data for visualization
        comparison_data = {}
        for dataset_name, comparison in benchmark_results['dataset_comparisons'].items():
            comparison_data[dataset_name] = {
                'baseline_accuracy': comparison['baseline_metrics'].get('accuracy', 0),
                'preprocessed_accuracy': comparison['preprocessing_metrics'].get('accuracy', 0),
                'improvement': comparison['improvements'].get('accuracy', 0)
            }
        
        # Generate plots
        self.visualizer.plot_preprocessing_comparison(comparison_data)
        
        # Save visualization paths
        benchmark_results['visualizations'] = {
            'preprocessing_comparison': str(self.visualizer.output_dir / "preprocessing_comparison.png")
        }
    
    def _save_benchmark_results(self, benchmark_results: Dict[str, Any]) -> None:
        """Save benchmark results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"benchmark_results_{timestamp}.yaml"
        
        with open(results_file, 'w') as f:
            yaml.dump(benchmark_results, f, default_flow_style=False)
        
        logger.info(f"Benchmark results saved to {results_file}")
    
    def generate_summary_report(self, benchmark_results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary report.
        
        Args:
            benchmark_results: Benchmark results
            
        Returns:
            Summary report as string
        """
        stats = benchmark_results['overall_statistics']
        
        report = f"""
# Data Preprocessing Benchmark Report

## Experiment Overview
- **Timestamp**: {benchmark_results['experiment_info']['timestamp']}
- **Random Seed**: {benchmark_results['experiment_info']['random_seed']}
- **Datasets Analyzed**: {len(benchmark_results['dataset_comparisons'])}

## Overall Performance Impact
- **Mean Accuracy Improvement**: {stats['accuracy_improvement']['mean']:.4f} ± {stats['accuracy_improvement']['std']:.4f}
- **Median Accuracy Improvement**: {stats['accuracy_improvement']['median']:.4f}
- **Success Rate**: {stats['overall_recommendation']['success_rate']:.1%} of datasets showed improvement
- **Recommendation**: {stats['overall_recommendation']['recommendation']}

## Cost Analysis
- **Mean Time Overhead**: {stats['time_overhead']['mean']:.1f} seconds
- **Mean Data Reduction**: {stats['data_reduction']['mean']:.1f}%

## Statistical Significance
"""
        
        if 'statistical_tests' in benchmark_results and 'wilcoxon' in benchmark_results['statistical_tests']:
            test_result = benchmark_results['statistical_tests']['wilcoxon']
            report += f"- **Wilcoxon Test**: p-value = {test_result['p_value']:.4f} ({test_result['interpretation']})\n"
            
            if 'effect_size' in benchmark_results['statistical_tests']:
                effect = benchmark_results['statistical_tests']['effect_size']
                report += f"- **Effect Size (Cohen's d)**: {effect['cohens_d']:.3f} ({effect['interpretation']})\n"
        
        report += "\n## Dataset-Specific Results\n"
        
        for dataset_name, comparison in benchmark_results['dataset_comparisons'].items():
            improvement = comparison['improvements'].get('accuracy', 0)
            recommendation = comparison['cost_benefit']['recommendation']
            
            report += f"- **{dataset_name}**: {improvement:+.4f} accuracy improvement - {recommendation}\n"
        
        return report
