"""
Visualization utilities for data preprocessing research.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .logging_config import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class VisualizationUtils:
    """
    Utilities for creating visualizations for the research framework.
    """
    
    def __init__(self, output_dir: str = "experiments/results/plots"):
        """
        Initialize visualization utilities.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        logger.info(f"VisualizationUtils initialized with output dir: {output_dir}")
    
    def plot_dataset_overview(
        self,
        datasets_info: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create overview visualization of all datasets.
        
        Args:
            datasets_info: Dictionary of dataset information
            save_path: Path to save the plot
        """
        # Prepare data for visualization
        data = []
        for dataset_name, info in datasets_info.items():
            data.append({
                'Dataset': dataset_name,
                'Max Samples': info.get('max_samples', 0),
                'Task Type': info.get('task_type', 'unknown'),
                'Source': info.get('source', 'unknown'),
                'Quality Issues': ', '.join(info.get('quality_issues', []))
            })
        
        df = pd.DataFrame(data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Dataset Overview', fontsize=16, fontweight='bold')
        
        # 1. Dataset sizes
        ax1 = axes[0, 0]
        bars = ax1.bar(df['Dataset'], df['Max Samples'], color='skyblue', alpha=0.7)
        ax1.set_title('Dataset Sizes')
        ax1.set_ylabel('Max Samples')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height):,}', ha='center', va='bottom')
        
        # 2. Task type distribution
        ax2 = axes[0, 1]
        task_counts = df['Task Type'].value_counts()
        ax2.pie(task_counts.values, labels=task_counts.index, autopct='%1.1f%%')
        ax2.set_title('Task Type Distribution')
        
        # 3. Source distribution
        ax3 = axes[1, 0]
        source_counts = df['Source'].value_counts()
        ax3.bar(source_counts.index, source_counts.values, color='lightcoral', alpha=0.7)
        ax3.set_title('Data Source Distribution')
        ax3.set_ylabel('Number of Datasets')
        
        # 4. Quality issues heatmap
        ax4 = axes[1, 1]
        quality_issues = ['redundancy', 'imbalance', 'noise', 'outliers']
        issue_matrix = []
        
        for _, row in df.iterrows():
            issues = row['Quality Issues'].lower()
            issue_row = [1 if issue in issues else 0 for issue in quality_issues]
            issue_matrix.append(issue_row)
        
        sns.heatmap(issue_matrix, 
                   xticklabels=quality_issues,
                   yticklabels=df['Dataset'],
                   cmap='Reds', 
                   cbar_kws={'label': 'Has Issue'},
                   ax=ax4)
        ax4.set_title('Quality Issues by Dataset')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "dataset_overview.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Dataset overview plot saved to {save_path}")
    
    def plot_baseline_results(
        self,
        results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize baseline experiment results.
        
        Args:
            results: Baseline experiment results
            save_path: Path to save the plot
        """
        # Extract metrics
        datasets = []
        accuracies = []
        f1_scores = []
        
        for dataset_name, result in results.items():
            metrics = result.get('test_metrics', {})
            datasets.append(dataset_name)
            accuracies.append(metrics.get('accuracy', 0))
            f1_scores.append(metrics.get('f1_weighted', 0))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Baseline Performance Results', fontsize=16, fontweight='bold')
        
        # Accuracy plot
        bars1 = ax1.bar(datasets, accuracies, color='steelblue', alpha=0.7)
        ax1.set_title('Accuracy by Dataset')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # F1 Score plot
        bars2 = ax2.bar(datasets, f1_scores, color='darkorange', alpha=0.7)
        ax2.set_title('F1 Score (Weighted) by Dataset')
        ax2.set_ylabel('F1 Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "baseline_results.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Baseline results plot saved to {save_path}")
    
    def plot_preprocessing_comparison(
        self,
        comparison_results: Dict[str, Dict[str, Any]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Visualize preprocessing vs baseline comparison.
        
        Args:
            comparison_results: Comparison results between baseline and preprocessing
            save_path: Path to save the plot
        """
        # Prepare data
        datasets = []
        baseline_acc = []
        preprocessed_acc = []
        improvements = []
        
        for dataset_name, result in comparison_results.items():
            datasets.append(dataset_name)
            baseline_acc.append(result.get('baseline_accuracy', 0))
            preprocessed_acc.append(result.get('preprocessed_accuracy', 0))
            improvements.append(result.get('improvement', 0))
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Preprocessing Impact Analysis', fontsize=16, fontweight='bold')
        
        # Comparison plot
        x = np.arange(len(datasets))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_acc, width, label='Baseline', 
                       color='lightcoral', alpha=0.7)
        bars2 = ax1.bar(x + width/2, preprocessed_acc, width, label='Preprocessed',
                       color='lightgreen', alpha=0.7)
        
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_xlabel('Dataset')
        ax1.set_xticks(x)
        ax1.set_xticklabels(datasets, rotation=45)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Improvement plot
        colors = ['green' if imp > 0 else 'red' for imp in improvements]
        bars3 = ax2.bar(datasets, improvements, color=colors, alpha=0.7)
        ax2.set_title('Performance Improvement')
        ax2.set_ylabel('Accuracy Improvement')
        ax2.set_xlabel('Dataset')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars3, improvements):
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (0.005 if imp > 0 else -0.015),
                    f'{imp:+.3f}', ha='center', va='bottom' if imp > 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / "preprocessing_comparison.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Preprocessing comparison plot saved to {save_path}")
    
    def create_interactive_dashboard(
        self,
        results_data: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> None:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            results_data: Combined results data
            save_path: Path to save the HTML dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Dataset Overview', 'Baseline Performance', 
                          'Preprocessing Impact', 'Statistical Significance'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Add plots (simplified for example)
        datasets = list(results_data.keys())
        baseline_scores = [results_data[d].get('baseline_accuracy', 0) for d in datasets]
        
        # Dataset overview
        fig.add_trace(
            go.Bar(x=datasets, y=baseline_scores, name="Baseline Accuracy"),
            row=1, col=1
        )
        
        # Make it interactive
        fig.update_layout(
            title_text="Data Preprocessing Research Dashboard",
            showlegend=True,
            height=800
        )
        
        # Save dashboard
        if save_path is None:
            save_path = self.output_dir / "interactive_dashboard.html"
        fig.write_html(save_path)
        
        logger.info(f"Interactive dashboard saved to {save_path}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        class_names: Optional[List[str]] = None,
        title: str = "Confusion Matrix",
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            title: Plot title
            save_path: Path to save the plot
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path is None:
            save_path = self.output_dir / f"{title.lower().replace(' ', '_')}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrix saved to {save_path}")
