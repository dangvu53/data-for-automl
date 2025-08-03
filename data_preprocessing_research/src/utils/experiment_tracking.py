"""
Experiment tracking utilities for reproducible research.
"""

import json
import yaml
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import hashlib
import logging

from .logging_config import get_logger
from .reproducibility import get_environment_info

logger = get_logger(__name__)

class ExperimentTracker:
    """
    Comprehensive experiment tracking for reproducible research.
    """
    
    def __init__(self, experiment_dir: str = "experiments/results"):
        """
        Initialize experiment tracker.
        
        Args:
            experiment_dir: Directory to store experiment results
        """
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_experiment = None
        self.experiments_log = self.experiment_dir / "experiments_log.json"
        
        # Load existing experiments log
        self.experiments = self._load_experiments_log()
        
        logger.info(f"ExperimentTracker initialized with dir: {experiment_dir}")
    
    def start_experiment(
        self,
        experiment_name: str,
        experiment_type: str,
        config: Dict[str, Any],
        description: Optional[str] = None
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            experiment_type: Type of experiment (baseline, preprocessing, etc.)
            config: Experiment configuration
            description: Optional description
            
        Returns:
            Experiment ID
        """
        timestamp = datetime.now()
        experiment_id = self._generate_experiment_id(experiment_name, timestamp)
        
        experiment_info = {
            'experiment_id': experiment_id,
            'experiment_name': experiment_name,
            'experiment_type': experiment_type,
            'description': description,
            'timestamp': timestamp.isoformat(),
            'config': config,
            'config_hash': self._hash_config(config),
            'environment': get_environment_info(),
            'status': 'running',
            'results': {},
            'metrics': {},
            'artifacts': []
        }
        
        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save experiment info
        with open(exp_dir / "experiment_info.yaml", 'w') as f:
            yaml.dump(experiment_info, f, default_flow_style=False)
        
        # Update experiments log
        self.experiments[experiment_id] = experiment_info
        self._save_experiments_log()
        
        self.current_experiment = experiment_id
        
        logger.info(f"Started experiment: {experiment_name} (ID: {experiment_id})")
        return experiment_id
    
    def log_dataset_info(
        self,
        dataset_name: str,
        dataset_config: Dict[str, Any],
        train_samples: int,
        test_samples: int,
        validation_samples: Optional[int] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        Log dataset information for the experiment.

        Args:
            dataset_name: Name of the dataset
            dataset_config: Dataset configuration
            train_samples: Number of training samples
            test_samples: Number of test samples
            validation_samples: Number of validation samples (optional)
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        dataset_info = {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'train_samples': train_samples,
            'test_samples': test_samples,
            'logged_at': datetime.now().isoformat()
        }

        # Add validation samples if provided
        if validation_samples is not None:
            dataset_info['validation_samples'] = validation_samples
            dataset_info['total_samples'] = train_samples + validation_samples + test_samples
        else:
            dataset_info['total_samples'] = train_samples + test_samples
        
        # Update experiment
        if 'datasets' not in self.experiments[exp_id]:
            self.experiments[exp_id]['datasets'] = {}
        
        self.experiments[exp_id]['datasets'][dataset_name] = dataset_info
        
        # Save to file
        exp_dir = self.experiment_dir / exp_id
        with open(exp_dir / "datasets_info.yaml", 'w') as f:
            yaml.dump(self.experiments[exp_id]['datasets'], f, default_flow_style=False)
        
        self._save_experiments_log()
        
        logger.info(f"Logged dataset info for {dataset_name} in experiment {exp_id}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        dataset_name: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        Log metrics for the experiment.
        
        Args:
            metrics: Dictionary of metrics
            dataset_name: Name of the dataset (optional)
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        metrics_entry = {
            'metrics': metrics,
            'dataset_name': dataset_name,
            'logged_at': datetime.now().isoformat()
        }
        
        # Update experiment
        if 'metrics' not in self.experiments[exp_id]:
            self.experiments[exp_id]['metrics'] = []
        
        self.experiments[exp_id]['metrics'].append(metrics_entry)
        
        # Save to file
        exp_dir = self.experiment_dir / exp_id
        with open(exp_dir / "metrics.json", 'w') as f:
            json.dump(self.experiments[exp_id]['metrics'], f, indent=2)
        
        self._save_experiments_log()
        
        logger.info(f"Logged metrics for experiment {exp_id}")
    
    def log_artifact(
        self,
        artifact_path: str,
        artifact_type: str,
        description: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        Log an artifact (file, model, plot, etc.) for the experiment.
        
        Args:
            artifact_path: Path to the artifact
            artifact_type: Type of artifact (model, plot, data, etc.)
            description: Optional description
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        artifact_info = {
            'path': artifact_path,
            'type': artifact_type,
            'description': description,
            'logged_at': datetime.now().isoformat(),
            'size_bytes': Path(artifact_path).stat().st_size if Path(artifact_path).exists() else 0
        }
        
        # Update experiment with better error handling
        if exp_id not in self.experiments:
            logger.warning(f"Experiment {exp_id} not found, skipping artifact logging")
            return

        if not isinstance(self.experiments[exp_id], dict):
            logger.warning(f"Experiment {exp_id} is not a dict, skipping artifact logging")
            return

        if 'artifacts' not in self.experiments[exp_id]:
            self.experiments[exp_id]['artifacts'] = []

        if not isinstance(self.experiments[exp_id]['artifacts'], list):
            self.experiments[exp_id]['artifacts'] = []

        self.experiments[exp_id]['artifacts'].append(artifact_info)
        
        self._save_experiments_log()
        
        logger.info(f"Logged artifact {artifact_path} for experiment {exp_id}")
    
    def finish_experiment(
        self,
        status: str = "completed",
        summary: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None
    ) -> None:
        """
        Finish the experiment.
        
        Args:
            status: Final status (completed, failed, cancelled)
            summary: Optional experiment summary
            experiment_id: Experiment ID (uses current if None)
        """
        exp_id = experiment_id or self.current_experiment
        if not exp_id:
            raise ValueError("No active experiment")
        
        # Update experiment status
        self.experiments[exp_id]['status'] = status
        self.experiments[exp_id]['finished_at'] = datetime.now().isoformat()
        
        if summary:
            self.experiments[exp_id]['summary'] = summary
        
        # Calculate experiment duration
        start_time = datetime.fromisoformat(self.experiments[exp_id]['timestamp'])
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        self.experiments[exp_id]['duration_seconds'] = duration
        
        # Save final experiment info
        exp_dir = self.experiment_dir / exp_id
        with open(exp_dir / "experiment_info.yaml", 'w') as f:
            yaml.dump(self.experiments[exp_id], f, default_flow_style=False)
        
        self._save_experiments_log()
        
        if exp_id == self.current_experiment:
            self.current_experiment = None
        
        logger.info(f"Finished experiment {exp_id} with status: {status}")
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get experiment information.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment information
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        return self.experiments[experiment_id]
    
    def list_experiments(
        self,
        experiment_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            experiment_type: Filter by experiment type
            status: Filter by status
            
        Returns:
            List of experiment information
        """
        experiments = list(self.experiments.values())
        
        if experiment_type:
            experiments = [exp for exp in experiments if exp.get('experiment_type') == experiment_type]
        
        if status:
            experiments = [exp for exp in experiments if exp.get('status') == status]
        
        # Sort by timestamp (newest first)
        experiments.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return experiments
    
    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to compare (all if None)
            
        Returns:
            Comparison DataFrame
        """
        comparison_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments:
                logger.warning(f"Experiment {exp_id} not found, skipping")
                continue
            
            exp = self.experiments[exp_id]
            
            # Extract metrics
            exp_metrics = {}
            if 'metrics' in exp and exp['metrics']:
                # Get the latest metrics entry
                latest_metrics = exp['metrics'][-1]['metrics']
                exp_metrics.update(latest_metrics)
            
            # Filter metrics if specified
            if metrics:
                exp_metrics = {k: v for k, v in exp_metrics.items() if k in metrics}
            
            # Add experiment info
            row = {
                'experiment_id': exp_id,
                'experiment_name': exp.get('experiment_name', ''),
                'experiment_type': exp.get('experiment_type', ''),
                'status': exp.get('status', ''),
                'timestamp': exp.get('timestamp', ''),
                'duration_seconds': exp.get('duration_seconds', 0),
                **exp_metrics
            }
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def _generate_experiment_id(self, experiment_name: str, timestamp: datetime) -> str:
        """Generate unique experiment ID."""
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        name_hash = hashlib.md5(experiment_name.encode()).hexdigest()[:8]
        return f"{timestamp_str}_{name_hash}"
    
    def _hash_config(self, config: Dict[str, Any]) -> str:
        """Generate hash of configuration for reproducibility."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()
    
    def _load_experiments_log(self) -> Dict[str, Any]:
        """Load experiments log from file."""
        if self.experiments_log.exists():
            try:
                with open(self.experiments_log, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load experiments log: {e}")
        
        return {}
    
    def _save_experiments_log(self) -> None:
        """Save experiments log to file."""
        try:
            with open(self.experiments_log, 'w') as f:
                json.dump(self.experiments, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save experiments log: {e}")
    
    def generate_experiment_report(self, experiment_id: str) -> str:
        """
        Generate a human-readable experiment report.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Report as string
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        
        report = f"""
# Experiment Report: {exp.get('experiment_name', 'Unknown')}

**Experiment ID:** {experiment_id}
**Type:** {exp.get('experiment_type', 'Unknown')}
**Status:** {exp.get('status', 'Unknown')}
**Started:** {exp.get('timestamp', 'Unknown')}
**Duration:** {exp.get('duration_seconds', 0):.1f} seconds

## Description
{exp.get('description', 'No description provided')}

## Configuration
```yaml
{yaml.dump(exp.get('config', {}), default_flow_style=False)}
```

## Datasets
"""
        
        if 'datasets' in exp:
            for dataset_name, dataset_info in exp['datasets'].items():
                report += f"- **{dataset_name}**: {dataset_info['total_samples']} samples\n"
        
        report += "\n## Metrics\n"
        
        if 'metrics' in exp and exp['metrics']:
            latest_metrics = exp['metrics'][-1]['metrics']
            for metric_name, value in latest_metrics.items():
                report += f"- **{metric_name}**: {value:.4f}\n"
        
        report += "\n## Artifacts\n"
        
        if 'artifacts' in exp:
            for artifact in exp['artifacts']:
                report += f"- **{artifact['type']}**: {artifact['path']}\n"
        
        return report
