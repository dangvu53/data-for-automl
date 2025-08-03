"""
Comprehensive metrics calculation for text classification experiments.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats
import logging

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

class MetricsCalculator:
    """
    Calculate comprehensive evaluation metrics for text classification.
    """
    
    def __init__(self):
        """Initialize the metrics calculator."""
        self.supported_metrics = [
            'accuracy', 'precision_macro', 'precision_weighted',
            'recall_macro', 'recall_weighted', 'f1_macro', 'f1_weighted',
            'roc_auc_ovr', 'roc_auc_ovo'
        ]
        logger.info("MetricsCalculator initialized")
    
    def calculate_metrics(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        y_pred_proba: Optional[np.ndarray] = None,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculate comprehensive metrics for predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            metrics: List of metrics to calculate (default: all supported)
            
        Returns:
            Dictionary of metric names and values
        """
        if metrics is None:
            metrics = self.supported_metrics
        
        results = {}
        
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic classification metrics
        if 'accuracy' in metrics:
            results['accuracy'] = accuracy_score(y_true, y_pred)
        
        if 'precision_macro' in metrics:
            results['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        
        if 'precision_weighted' in metrics:
            results['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if 'recall_macro' in metrics:
            results['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        if 'recall_weighted' in metrics:
            results['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if 'f1_macro' in metrics:
            results['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        if 'f1_weighted' in metrics:
            results['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # ROC AUC metrics (require probabilities)
        if y_pred_proba is not None:
            try:
                if 'roc_auc_ovr' in metrics:
                    if len(np.unique(y_true)) == 2:
                        # Binary classification
                        results['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    else:
                        # Multiclass classification
                        results['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
                
                if 'roc_auc_ovo' in metrics and len(np.unique(y_true)) > 2:
                    results['roc_auc_ovo'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovo')
                    
            except Exception as e:
                logger.warning(f"Could not calculate ROC AUC: {e}")
        
        logger.info(f"Calculated metrics: {list(results.keys())}")
        return results
    
    def get_classification_report(
        self,
        y_true: Union[np.ndarray, pd.Series],
        y_pred: Union[np.ndarray, pd.Series],
        target_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            target_names: Names of target classes
            
        Returns:
            Classification report as dictionary
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Get classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        
        # Add confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        report['confusion_matrix'] = cm.tolist()
        
        return report
    
    def compare_metrics(
        self,
        baseline_metrics: Dict[str, float],
        treatment_metrics: Dict[str, float],
        significance_test: str = 'wilcoxon'
    ) -> Dict[str, Any]:
        """
        Compare metrics between baseline and treatment.
        
        Args:
            baseline_metrics: Baseline experiment metrics
            treatment_metrics: Treatment experiment metrics
            significance_test: Statistical test to use
            
        Returns:
            Comparison results with improvements and significance
        """
        comparison = {
            'improvements': {},
            'relative_improvements': {},
            'statistical_tests': {}
        }
        
        # Calculate improvements
        for metric in baseline_metrics:
            if metric in treatment_metrics:
                baseline_val = baseline_metrics[metric]
                treatment_val = treatment_metrics[metric]
                
                # Absolute improvement
                improvement = treatment_val - baseline_val
                comparison['improvements'][metric] = improvement
                
                # Relative improvement (percentage)
                if baseline_val != 0:
                    rel_improvement = (improvement / baseline_val) * 100
                    comparison['relative_improvements'][metric] = rel_improvement
                else:
                    comparison['relative_improvements'][metric] = float('inf') if improvement > 0 else 0
        
        logger.info(f"Metric comparison completed for {len(comparison['improvements'])} metrics")
        return comparison
    
    def calculate_effect_size(
        self,
        baseline_scores: List[float],
        treatment_scores: List[float]
    ) -> Dict[str, float]:
        """
        Calculate effect size (Cohen's d) between baseline and treatment.
        
        Args:
            baseline_scores: List of baseline scores
            treatment_scores: List of treatment scores
            
        Returns:
            Effect size statistics
        """
        baseline_scores = np.array(baseline_scores)
        treatment_scores = np.array(treatment_scores)
        
        # Calculate means and standard deviations
        mean_baseline = np.mean(baseline_scores)
        mean_treatment = np.mean(treatment_scores)
        std_baseline = np.std(baseline_scores, ddof=1)
        std_treatment = np.std(treatment_scores, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(baseline_scores), len(treatment_scores)
        pooled_std = np.sqrt(((n1 - 1) * std_baseline**2 + (n2 - 1) * std_treatment**2) / (n1 + n2 - 2))
        
        # Cohen's d
        cohens_d = (mean_treatment - mean_baseline) / pooled_std
        
        # Effect size interpretation
        if abs(cohens_d) < 0.2:
            interpretation = "negligible"
        elif abs(cohens_d) < 0.5:
            interpretation = "small"
        elif abs(cohens_d) < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return {
            'cohens_d': cohens_d,
            'interpretation': interpretation,
            'mean_difference': mean_treatment - mean_baseline,
            'pooled_std': pooled_std
        }
    
    def statistical_significance_test(
        self,
        baseline_scores: List[float],
        treatment_scores: List[float],
        test: str = 'wilcoxon',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform statistical significance test.
        
        Args:
            baseline_scores: Baseline scores
            treatment_scores: Treatment scores
            test: Statistical test ('wilcoxon', 'ttest', 'mannwhitney')
            alpha: Significance level
            
        Returns:
            Test results
        """
        baseline_scores = np.array(baseline_scores)
        treatment_scores = np.array(treatment_scores)
        
        if test == 'wilcoxon':
            # Wilcoxon signed-rank test (paired)
            if len(baseline_scores) == len(treatment_scores):
                statistic, p_value = stats.wilcoxon(baseline_scores, treatment_scores)
                test_name = "Wilcoxon Signed-Rank Test"
            else:
                # Fall back to Mann-Whitney U test
                statistic, p_value = stats.mannwhitneyu(baseline_scores, treatment_scores)
                test_name = "Mann-Whitney U Test"
                
        elif test == 'ttest':
            # Paired t-test
            if len(baseline_scores) == len(treatment_scores):
                statistic, p_value = stats.ttest_rel(baseline_scores, treatment_scores)
                test_name = "Paired T-Test"
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(baseline_scores, treatment_scores)
                test_name = "Independent T-Test"
                
        elif test == 'mannwhitney':
            # Mann-Whitney U test
            statistic, p_value = stats.mannwhitneyu(baseline_scores, treatment_scores)
            test_name = "Mann-Whitney U Test"
            
        else:
            raise ValueError(f"Unsupported test: {test}")
        
        is_significant = p_value < alpha
        
        return {
            'test_name': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'alpha': alpha,
            'is_significant': is_significant,
            'interpretation': 'significant' if is_significant else 'not significant'
        }
