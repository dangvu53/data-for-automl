#!/usr/bin/env python3
"""
AutoGluon Multi-Dataset Training Script

This script trains AutoGluon models on multiple datasets:
1. CaseHold (text classification)
2. ScienceQA (multimodal classification - text only due to AutoGluon limitations)
3. ANLI R1 (text classification)
4. Temperature Rain (time series - converted to tabular regression)

Uses default configurations for baseline performance comparison.
"""

import os
import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
from datasets import load_dataset
from autogluon.tabular import TabularPredictor
from autogluon.multimodal import MultiModalPredictor
from autogluon.timeseries import TimeSeriesPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def format_time(seconds):
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



def load_dataset_from_csv(base_path, dataset_name=None, label_column='label'):
    """Generic function to load any dataset from CSV files
    
    Args:
        base_path: Path to directory containing train_subset.csv, val_subset.csv, and test_subset.csv
        dataset_name: Optional name for logging purposes
        label_column: Name of the label column (defaults to 'label')
    
    Returns:
        train_df, val_df, test_df, label_column
    """
    start_time = time.time()
    if dataset_name:
        logger.info(f"Loading {dataset_name} dataset from {base_path}...")
    else:
        logger.info(f"Loading dataset from {base_path}...")

    try:
        train_df = pd.read_csv(f"{base_path}/train_subset.csv")
        val_df = pd.read_csv(f"{base_path}/val_subset.csv")
        test_df = pd.read_csv(f"{base_path}/test_subset.csv")
        
        # Ensure text column exists - construct if needed based on available columns
        if 'text' not in train_df.columns:
            # For ANLI
            if 'premise' in train_df.columns and 'hypothesis' in train_df.columns:
                for df in [train_df, val_df, test_df]:
                    df['text'] = df['premise'] + " [SEP] " + df['hypothesis']
            # For CaseHold
            elif 'citing_prompt' in train_df.columns and 'holding_0' in train_df.columns:
                for df in [train_df, val_df, test_df]:
                    # Create text column by concatenating citing_prompt with all holdings
                    text_parts = [df['citing_prompt'].astype(str)]
                    for i in range(5):
                        if f'holding_{i}' in df.columns:
                            text_parts.append(df[f'holding_{i}'].astype(str))
                    df['text'] = " [SEP] ".join(text_parts)
        
        load_time = time.time() - start_time
        logger.info(f"Dataset loaded from CSV in {format_time(load_time)}")
        logger.info(f"Sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        if label_column not in train_df.columns:
            available_columns = train_df.columns.tolist()
            logger.error(f"Label column '{label_column}' not found. Available columns: {available_columns}")
            return None, None, None, None
            
        return train_df, val_df, test_df, label_column

    except Exception as e:
        logger.error(f"Error loading dataset from CSV: {e}")
        return None, None, None, None

class AutoGluonMultiDatasetTrainer:

    def __init__(self, output_dir="./autogluon_multi_results"):
        self.output_dir = output_dir
        self.results = {}
        self.predictions_dir = os.path.join(output_dir, "predictions")
        self.confusion_matrices_dir = os.path.join(output_dir, "confusion_matrices")

        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.confusion_matrices_dir, exist_ok=True)
        
    def load_casehold_dataset(self):
        """Load and prepare CaseHold dataset for text classification"""
        logger.info("Loading CaseHold dataset...")
        
        try:
            dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "casehold")
            
            # Prepare data for text classification
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

            logger.info(f"CaseHold loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df, 'label'
            
        except Exception as e:
            logger.error(f"Error loading CaseHold: {e}")
            return None, None, None, None
    
    def load_scienceqa_dataset(self):
        """Load and prepare ScienceQA dataset for multimodal classification"""
        logger.info("Loading ScienceQA dataset...")

        try:
            dataset = load_dataset("MothMalone/SLMS-KD-Benchmarks", "scienceqa")

            def prepare_scienceqa_data(split_data):
                data = []
                for item in split_data:
                    # Prepare multimodal features
                    text_features = item['question']
                    if item['choices']:
                        text_features += " [CHOICES] " + " | ".join(item['choices'])
                    if item['hint']:
                        text_features += " [HINT] " + item['hint']
                    if item['lecture']:
                        text_features += " [LECTURE] " + item['lecture']

                    row = {
                        'text': text_features,
                        'task': item['task'],
                        'grade': item['grade'],
                        'subject': item['subject'],
                        'topic': item['topic'],
                        'category': item['category'],
                        'answer': item['answer']
                    }

                    # Handle image data properly for multimodal
                    if 'image' in item and item['image'] is not None:
                        try:
                            # Handle different image formats
                            if isinstance(item['image'], dict):
                                # If image is a dict with bytes, convert to PIL Image
                                if 'bytes' in item['image'] and item['image']['bytes']:
                                    image = Image.open(io.BytesIO(item['image']['bytes']))
                                    row['image'] = image
                                else:
                                    row['image'] = None
                            elif hasattr(item['image'], 'size'):
                                # Already a PIL Image
                                row['image'] = item['image']
                            else:
                                # Unknown format, skip image
                                row['image'] = None
                        except Exception as e:
                            logger.warning(f"Could not process image: {e}")
                            row['image'] = None
                    else:
                        row['image'] = None

                    data.append(row)
                return pd.DataFrame(data)

            # Use existing train/val/test splits
            train_df = prepare_scienceqa_data(dataset['train'])
            val_df = prepare_scienceqa_data(dataset['validation']) if 'validation' in dataset else None
            test_df = prepare_scienceqa_data(dataset['test']) if 'test' in dataset else None

            # If no validation/test splits, create them
            if val_df is None or test_df is None:
                logger.info("Creating validation/test splits from train data")
                temp_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42)
                train_df, val_df = train_test_split(temp_df, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

            logger.info(f"ScienceQA loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df, 'answer'

        except Exception as e:
            logger.error(f"Error loading ScienceQA: {e}")
            return None, None, None, None
    
    def load_anli_dataset(self):
        """Load and prepare ANLI R1 dataset for text classification"""
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

            logger.info(f"ANLI R1 loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            return train_df, val_df, test_df, 'label'
            
        except Exception as e:
            logger.error(f"Error loading ANLI: {e}")
            return None, None, None, None
    
    def load_temperature_rain_dataset(self):
        """Load and prepare Temperature Rain time series dataset from TSF file"""
        logger.info("Loading Temperature Rain dataset...")

        # Check if cached processed data exists
        cache_dir = './temp_data/temperature_rain_cache'
        train_cache = os.path.join(cache_dir, 'train_df.pkl')
        val_cache = os.path.join(cache_dir, 'val_df.pkl')
        test_cache = os.path.join(cache_dir, 'test_df.pkl')

        if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(test_cache):
            logger.info("Loading cached temperature_rain data...")
            try:
                train_df = pd.read_pickle(train_cache)
                val_df = pd.read_pickle(val_cache)
                test_df = pd.read_pickle(test_cache)

                logger.info(f"Cached Temperature Rain loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
                logger.info(f"Time series items: Train={train_df['item_id'].nunique()}, Val={val_df['item_id'].nunique()}, Test={test_df['item_id'].nunique()}")
                logger.info(f"Date range: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
                return train_df, val_df, test_df, 'target'
            except Exception as e:
                logger.warning(f"Error loading cached data: {e}. Will reload from source.")

        try:
            import zipfile
            import sys

            # Add monash_tsf to path for utils
            sys.path.append('./monash_tsf')
            from utils import convert_tsf_to_dataframe

            # Extract and load the TSF file
            zip_path = './monash_tsf/data/temperature_rain_dataset_with_missing_values.zip'
            tsf_file = 'temperature_rain_dataset_with_missing_values.tsf'

            if not os.path.exists(zip_path):
                raise Exception(f"Temperature rain zip file not found at {zip_path}")

            # Extract TSF file temporarily
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extract(tsf_file, './temp_data/')

            tsf_path = f'./temp_data/{tsf_file}'

            # Parse TSF file
            logger.info("Parsing TSF file...")
            df, frequency, forecast_horizon, contain_missing_values, contain_equal_length = convert_tsf_to_dataframe(
                tsf_path,
                replace_missing_vals_with=np.nan,
                value_column_name="target"
            )

            logger.info(f"TSF file parsed successfully:")
            logger.info(f"  - Shape: {df.shape}")
            logger.info(f"  - Frequency: {frequency}")
            logger.info(f"  - Forecast horizon: {forecast_horizon}")
            logger.info(f"  - Contains missing values: {contain_missing_values}")
            logger.info(f"  - Equal length series: {contain_equal_length}")
            logger.info(f"  - Columns: {df.columns.tolist()}")

            # Clean up temp file
            os.remove(tsf_path)
            os.rmdir('./temp_data/')

            # Convert to AutoGluon TimeSeriesPredictor format
            data_list = []

            # The TSF format typically has series_name and series_value columns
            # Group by series identifier and create time series records
            if 'series_name' in df.columns:
                item_id_col = 'series_name'
            elif 'station_id' in df.columns:
                item_id_col = 'station_id'
            else:
                # Find the first non-target column as item_id
                item_id_col = [col for col in df.columns if col != 'target'][0]

            logger.info(f"Using '{item_id_col}' as item_id column")

            # Process each time series
            for idx, row in df.iterrows():
                if idx % 100 == 0:
                    logger.info(f"Processing series {idx}/{len(df)}")

                # Get item_id
                item_id = row[item_id_col] if item_id_col in row else f'series_{idx}'

                # Get target values (this is an array/list of values)
                target_values = row['target']

                # Handle different target value formats
                if isinstance(target_values, (list, np.ndarray)):
                    target_array = target_values
                elif isinstance(target_values, str):
                    # Parse string representation of array
                    target_array = [float(x) for x in target_values.split() if x.strip()]
                else:
                    target_array = [target_values]

                # Create time series records starting from the dataset's start date
                start_date = pd.Timestamp('2015-05-02')  # As per dataset description

                for i, value in enumerate(target_array):
                    try:
                        if pd.notna(value) and value != '' and str(value).lower() != 'nan':
                            data_list.append({
                                'item_id': str(item_id),
                                'timestamp': start_date + pd.Timedelta(days=i),
                                'target': float(value)
                            })
                    except (ValueError, TypeError):
                        # Skip invalid values
                        continue

            # Create DataFrame
            full_data = pd.DataFrame(data_list)
            full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])
            full_data = full_data.sort_values(['item_id', 'timestamp'])

            logger.info(f"Created time series DataFrame with {len(full_data)} records")

            if len(full_data) == 0:
                raise Exception("No valid data found in temperature_rain dataset")

            # Split by time series items (stations)
            unique_items = full_data['item_id'].unique()
            logger.info(f"Found {len(unique_items)} unique weather stations")

            # Split stations: 70% train, 15% val, 15% test
            np.random.seed(42)
            shuffled_items = np.random.permutation(unique_items)

            n_train = int(0.7 * len(shuffled_items))
            n_val = int(0.15 * len(shuffled_items))

            train_items = shuffled_items[:n_train]
            val_items = shuffled_items[n_train:n_train + n_val]
            test_items = shuffled_items[n_train + n_val:]

            train_df = full_data[full_data['item_id'].isin(train_items)]
            val_df = full_data[full_data['item_id'].isin(val_items)]
            test_df = full_data[full_data['item_id'].isin(test_items)]

            logger.info(f"Temperature Rain loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
            logger.info(f"Time series items: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
            logger.info(f"Date range: {full_data['timestamp'].min()} to {full_data['timestamp'].max()}")

            # Cache the processed data for future use
            logger.info("Caching processed temperature_rain data...")
            os.makedirs(cache_dir, exist_ok=True)
            train_df.to_pickle(train_cache)
            val_df.to_pickle(val_cache)
            test_df.to_pickle(test_cache)
            logger.info("Data cached successfully!")

            return train_df, val_df, test_df, 'target'

        except Exception as e:
            logger.error(f"Error loading Temperature Rain: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to synthetic data if needed
            logger.warning("Falling back to synthetic temperature data")
            return self.create_synthetic_temperature_data()

    # def create_synthetic_temperature_data(self):
    #     """Create synthetic temperature time series data that mimics the real temperature_rain dataset"""
    #     logger.info("Creating synthetic temperature data...")

    #     # Generate synthetic temperature data in time series format
    #     np.random.seed(42)
    #     n_series = 50  # More series to better simulate real dataset
    #     n_points = 730  # Two years of daily data (like the real dataset: 2015-2017)

    #     data = []
    #     for series_id in range(n_series):
    #         # Generate realistic Australian weather station data
    #         # Different climate zones across Australia
    #         if series_id < 15:  # Tropical (Northern Australia)
    #             base_temp = np.random.normal(28, 3)
    #             seasonal_amplitude = 5
    #         elif series_id < 30:  # Temperate (Southern Australia)
    #             base_temp = np.random.normal(18, 4)
    #             seasonal_amplitude = 12
    #         else:  # Arid (Central Australia)
    #             base_temp = np.random.normal(22, 5)
    #             seasonal_amplitude = 15

    #         # Generate temperature-like time series with realistic patterns
    #         trend = np.random.normal(0, 0.005, n_points).cumsum()  # Slight warming trend
    #         seasonal = seasonal_amplitude * np.sin(np.linspace(0, 4*np.pi, n_points) + np.pi)  # Summer in Dec-Feb
    #         noise = np.random.normal(0, 3, n_points)  # Daily variation

    #         # Add some extreme weather events
    #         extreme_events = np.random.choice(n_points, size=int(0.02 * n_points), replace=False)
    #         extreme_values = np.zeros(n_points)
    #         extreme_values[extreme_events] = np.random.normal(0, 8, len(extreme_events))

    #         temps = base_temp + trend + seasonal + noise + extreme_values

    #         # Create time series format
    #         for i, temp in enumerate(temps):
    #             data.append({
    #                 'item_id': f'weather_station_{series_id:03d}',
    #                 'timestamp': pd.Timestamp('2015-05-02') + pd.Timedelta(days=i),  # Start date like real dataset
    #                 'target': round(temp, 2)
    #             })

    #     full_data = pd.DataFrame(data)
    #     full_data['timestamp'] = pd.to_datetime(full_data['timestamp'])

    #     # Split by time series items (not by time, to maintain full time series for each station)
    #     unique_items = full_data['item_id'].unique()
    #     np.random.shuffle(unique_items)  # Randomize station assignment

    #     train_items = unique_items[:int(0.7 * len(unique_items))]
    #     val_items = unique_items[int(0.7 * len(unique_items)):int(0.85 * len(unique_items))]
    #     test_items = unique_items[int(0.85 * len(unique_items)):]

    #     train_df = full_data[full_data['item_id'].isin(train_items)]
    #     val_df = full_data[full_data['item_id'].isin(val_items)]
    #     test_df = full_data[full_data['item_id'].isin(test_items)]

    #     logger.info(f"Synthetic Temperature data created: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    #     logger.info(f"Time series items: Train={len(train_items)}, Val={len(val_items)}, Test={len(test_items)}")
    #     logger.info(f"Date range: {full_data['timestamp'].min()} to {full_data['timestamp'].max()}")
    #     return train_df, val_df, test_df, 'target'
    
    def train_model(self, train_df, val_df, target_col, dataset_name, problem_type=None):
        """Train appropriate AutoGluon model based on dataset type"""
        logger.info(f"Training AutoGluon model for {dataset_name}...")

        start_time = time.time()

        # Create output directory for this dataset
        model_dir = os.path.join(self.output_dir, f"{dataset_name}_model")
        os.makedirs(model_dir, exist_ok=True)

        # Determine predictor type and problem type based on dataset
        if dataset_name == 'temperature_rain':
            return self.train_timeseries_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        elif dataset_name == 'scienceqa':
            return self.train_multimodal_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        elif dataset_name in ['casehold', 'anli_r1']:
            return self.train_text_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)
        else:
            return self.train_tabular_model(train_df, val_df, target_col, dataset_name, model_dir, start_time)

    def train_timeseries_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train TimeSeriesPredictor"""
        logger.info(f"Using TimeSeriesPredictor for {dataset_name}")

        # TimeSeriesPredictor expects specific format
        predictor = TimeSeriesPredictor(
            path=model_dir,
            target=target_col,
            prediction_length=24,  # Predict 24 time steps ahead
            eval_metric='MASE',
            freq='D'  # Daily frequency for temperature_rain dataset
        )

        # Use AutoGluon's default settings (TimeSeriesPredictor doesn't support ag_args_fit)
        predictor.fit(
            train_data=train_df,
            tuning_data=val_df,
            presets='medium_quality',  # Use AutoGluon's default preset
            time_limit=3000,
            verbosity=2
            # No included_model_types - let AutoGluon choose default models
        )

        training_time = time.time() - start_time
        logger.info(f"TimeSeriesPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
        return predictor, training_time

    def train_multimodal_model(self, train_df, val_df, target_col, dataset_name, model_dir, start_time):
        """Train MultiModalPredictor for multimodal data"""
        logger.info(f"Using MultiModalPredictor for {dataset_name}")

        try:
            # Create MultiModalPredictor with AutoGluon's default settings
            predictor = MultiModalPredictor(
                label=target_col,
                path=model_dir
            )

            # Combine train and validation data for MultiModalPredictor
            # MultiModalPredictor handles train/val split internally
            combined_train_df = pd.concat([train_df, val_df], ignore_index=True)

            logger.info(f"Training MultiModalPredictor with {len(combined_train_df)} samples")
            logger.info(f"Columns: {combined_train_df.columns.tolist()}")
            logger.info(f"Images available: {combined_train_df['image'].notna().sum()}/{len(combined_train_df)}")

            # Train MultiModalPredictor with AutoGluon's default settings
            predictor.fit(
                train_data=combined_train_df,
                time_limit=3000,
                presets='medium_quality'  # Use AutoGluon's default preset
                # No custom model specifications - let AutoGluon choose defaults
            )

            training_time = time.time() - start_time
            logger.info(f"MultiModalPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
            return predictor, training_time

        except Exception as e:
            logger.error(f"MultiModalPredictor failed: {e}")
            import traceback
            traceback.print_exc()

            logger.warning("Falling back to TabularPredictor without images")

            # Prepare data for TabularPredictor (remove image column)
            train_df_tabular = train_df.copy()
            val_df_tabular = val_df.copy()

            # Remove image column for TabularPredictor as it can't handle PIL Images
            if 'image' in train_df_tabular.columns:
                logger.info("Removing image column for TabularPredictor fallback")
                train_df_tabular = train_df_tabular.drop(columns=['image'])
                val_df_tabular = val_df_tabular.drop(columns=['image'])

            # Fallback to TabularPredictor with AutoGluon defaults
            predictor = TabularPredictor(
                label=target_col,
                path=model_dir,
                problem_type='multiclass',
                eval_metric='accuracy'
            )

            # Use AutoGluon's default settings with increased memory allowance
            predictor.fit(
                train_data=train_df_tabular,
                tuning_data=val_df_tabular,
                presets='medium_quality',  # Use AutoGluon's default preset
                time_limit=3000,
                verbosity=2,
                # Allow AutoGluon to use more memory
                ag_args_fit={'ag.max_memory_usage_ratio': 5}
                # No included_model_types - let AutoGluon choose default models
            )

            training_time = time.time() - start_time
            logger.info(f"Fallback TabularPredictor training completed for {dataset_name} in {training_time:.2f} seconds")
            return predictor, training_time

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
        logger.info(f"Text model training completed for {dataset_name} in {training_time:.2f} seconds")
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
        logger.info(f"Tabular model training completed for {dataset_name} in {training_time:.2f} seconds")
        return predictor, training_time

    def get_model_info(self, predictor, dataset_name):
        """Get model size and parameter information"""
        logger.info(f"Getting model information for {dataset_name}...")

        try:
            model_info = {
                'model_size_mb': 'N/A',
                'num_parameters': 'N/A',
                'model_type': 'Unknown'
            }

            # Get model type
            if hasattr(predictor, '__class__'):
                model_info['model_type'] = predictor.__class__.__name__

            # Try to get model size from disk
            if hasattr(predictor, 'path') and predictor.path:
                try:
                    import os
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(predictor.path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            if os.path.isfile(filepath):
                                total_size += os.path.getsize(filepath)
                    model_info['model_size_mb'] = round(total_size / (1024 * 1024), 2)
                except Exception as e:
                    logger.warning(f"Could not calculate model size: {e}")

            # Try to get parameter count for neural network models
            try:
                if hasattr(predictor, 'get_model_names'):
                    model_names = predictor.get_model_names()
                    total_params = 0
                    for model_name in model_names:
                        if 'NN' in model_name or 'Neural' in model_name or 'FASTAI' in model_name:
                            # Try to get model object and count parameters
                            try:
                                model_obj = predictor._trainer.load_model(model_name)
                                if hasattr(model_obj, 'model') and hasattr(model_obj.model, 'parameters'):
                                    params = sum(p.numel() for p in model_obj.model.parameters())
                                    total_params += params
                            except:
                                pass
                    if total_params > 0:
                        model_info['num_parameters'] = total_params
            except Exception as e:
                logger.warning(f"Could not count parameters: {e}")

            logger.info(f"Model info for {dataset_name}: {model_info}")
            return model_info

        except Exception as e:
            logger.error(f"Error getting model info for {dataset_name}: {e}")
            return {
                'model_size_mb': 'Error',
                'num_parameters': 'Error',
                'model_type': 'Error'
            }

    def save_predictions_and_confusion_matrix(self, predictor, test_df, target_col, dataset_name):
        """Save predictions as CSV and create confusion matrix for classification tasks"""
        logger.info(f"Saving predictions and creating confusion matrix for {dataset_name}...")

        try:
            # Make predictions
            if dataset_name == 'temperature_rain':
                # Time series predictions - TimeSeriesPredictor returns TimeSeriesDataFrame
                try:
                    # TimeSeriesPredictor needs the target column for prediction
                    predictions = predictor.predict(test_df)
                    logger.info(f"Time series predictions type: {type(predictions)}")
                    logger.info(f"Time series predictions shape: {predictions.shape if hasattr(predictions, 'shape') else 'No shape'}")

                    # TimeSeriesPredictor returns future predictions, not predictions for test data timestamps
                    # Convert to regular DataFrame for easier handling
                    if hasattr(predictions, 'reset_index'):
                        pred_df_clean = predictions.reset_index()
                        logger.info(f"Predictions columns after reset_index: {pred_df_clean.columns.tolist()}")
                        logger.info(f"Predictions shape: {pred_df_clean.shape}")
                        logger.info(f"Sample predictions:\n{pred_df_clean.head()}")

                        # Save predictions separately (these are future predictions)
                        pred_file_future = os.path.join(self.predictions_dir, f"{dataset_name}_future_predictions.csv")
                        pred_df_clean.to_csv(pred_file_future, index=False)
                        logger.info(f"Future predictions saved to: {pred_file_future}")

                        # Also save the test data for reference
                        test_file = os.path.join(self.predictions_dir, f"{dataset_name}_test_data.csv")
                        test_df.to_csv(test_file, index=False)
                        logger.info(f"Test data saved to: {test_file}")

                        # Create a combined file with both test data and prediction info
                        combined_df = test_df.copy()
                        combined_df['prediction_info'] = f"Future predictions saved in {dataset_name}_future_predictions.csv"
                        combined_df['num_future_predictions'] = len(pred_df_clean)
                        combined_df['prediction_length'] = 24  # As configured in TimeSeriesPredictor

                        pred_df = combined_df
                    else:
                        # If it's already a regular DataFrame
                        pred_df = predictions.copy()
                        if hasattr(pred_df, 'reset_index'):
                            pred_df = pred_df.reset_index()

                    pred_file = os.path.join(self.predictions_dir, f"{dataset_name}_predictions.csv")
                    pred_df.to_csv(pred_file, index=False)
                    logger.info(f"Time series predictions saved to: {pred_file}")
                    logger.info(f"Saved {len(pred_df)} rows with columns: {pred_df.columns.tolist()}")

                except Exception as e:
                    logger.error(f"Error saving time series predictions: {e}")
                    import traceback
                    traceback.print_exc()

                    # Create a simple fallback
                    pred_df = test_df.copy()
                    pred_df['prediction'] = 'Error'
                    pred_file = os.path.join(self.predictions_dir, f"{dataset_name}_predictions.csv")
                    pred_df.to_csv(pred_file, index=False)
                    predictions = None

                return predictions, None  # No confusion matrix for regression

            else:
                # Classification predictions
                test_features = test_df.drop(columns=[target_col])

                # Remove image column if it exists and contains PIL Images (for TabularPredictor)
                if 'image' in test_features.columns:
                    # Check if image column contains PIL Images
                    sample_image = test_features['image'].dropna().iloc[0] if test_features['image'].notna().any() else None
                    if sample_image is not None and hasattr(sample_image, 'size'):
                        logger.info("Removing image column for prediction (TabularPredictor fallback)")
                        test_features = test_features.drop(columns=['image'])

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

    def evaluate_model(self, predictor, test_df, target_col, dataset_name):
        """Evaluate the trained model with detailed metrics"""
        logger.info(f"Evaluating model for {dataset_name}...")

        try:
            # Get model information (size, parameters)
            model_info = self.get_model_info(predictor, dataset_name)

            # Save predictions and create confusion matrix first
            predictions, confusion_matrix = self.save_predictions_and_confusion_matrix(
                predictor, test_df, target_col, dataset_name
            )

            # Handle different predictor types
            if hasattr(predictor, 'predict'):
                if dataset_name == 'temperature_rain':
                    # TimeSeriesPredictor evaluation - different API
                    try:
                        predictions = predictor.predict(test_df)
                        # TimeSeriesPredictor.evaluate() doesn't take 'silent' parameter
                        performance = predictor.evaluate(test_df)

                        # Extract performance metrics
                        if isinstance(performance, dict):
                            mase_score = performance.get('MASE', performance.get('mean_absolute_scaled_error', 'N/A'))
                        else:
                            mase_score = performance

                        results = {
                            'dataset': dataset_name,
                            'performance': performance,
                            'best_model': 'TimeSeriesPredictor',
                            'best_val_score': mase_score,
                            'test_score': mase_score,
                            'accuracy': 'N/A',  # Not applicable for time series
                            'num_models_trained': 1,
                            'total_models_attempted': 1,
                            'leaderboard_top3': [{'model': 'TimeSeriesPredictor', 'score_val': mase_score}],
                            'model_training_times': {},
                            'problem_type': 'regression',
                            'eval_metric': 'MASE',
                            'predictions_saved': predictions is not None,
                            'confusion_matrix_saved': False,  # No confusion matrix for regression
                            'model_size_mb': model_info.get('model_size_mb', 'N/A'),
                            'num_parameters': model_info.get('num_parameters', 'N/A'),
                            'model_type': model_info.get('model_type', 'TimeSeriesPredictor')
                        }
                    except Exception as e:
                        logger.error(f"Error evaluating TimeSeriesPredictor: {e}")
                        results = {
                            'dataset': dataset_name,
                            'performance': 'Error',
                            'best_model': 'TimeSeriesPredictor',
                            'best_val_score': 'Error',
                            'test_score': 'Error',
                            'accuracy': 'Error',
                            'num_models_trained': 1,
                            'total_models_attempted': 1,
                            'leaderboard_top3': [{'model': 'TimeSeriesPredictor', 'score_val': 'Error'}],
                            'model_training_times': {},
                            'problem_type': 'regression',
                            'eval_metric': 'MASE',
                            'predictions_saved': False,
                            'confusion_matrix_saved': False,
                            'model_size_mb': model_info.get('model_size_mb', 'Error'),
                            'num_parameters': model_info.get('num_parameters', 'Error'),
                            'model_type': model_info.get('model_type', 'Error')
                        }
                else:
                    # TabularPredictor or MultiModalPredictor evaluation
                    test_features = test_df.drop(columns=[target_col])

                    # For MultiModalPredictor, we can keep the image column
                    # For TabularPredictor fallback, we need to remove it
                    if hasattr(predictor, '__class__') and 'MultiModal' not in predictor.__class__.__name__:
                        # This is TabularPredictor fallback, remove image column if it exists
                        if 'image' in test_features.columns:
                            logger.info("Removing image column for TabularPredictor evaluation")
                            test_features = test_features.drop(columns=['image'])

                    predictions = predictor.predict(test_features)

                    # Handle different evaluation APIs
                    try:
                        performance = predictor.evaluate(test_df, silent=True)
                    except TypeError:
                        # MultiModalPredictor might not accept silent parameter
                        performance = predictor.evaluate(test_df)

                    # Extract accuracy for classification tasks
                    accuracy = 'N/A'
                    if isinstance(performance, dict):
                        accuracy = performance.get('accuracy', performance.get('acc', 'N/A'))
                    elif isinstance(performance, (int, float)):
                        # If performance is a single number and it's classification, it's likely accuracy
                        problem_type = getattr(predictor, 'problem_type', 'unknown')
                        if problem_type in ['binary', 'multiclass']:
                            accuracy = performance

                    # Get leaderboard if available
                    try:
                        # Try TabularPredictor leaderboard first
                        leaderboard = predictor.leaderboard(test_df, silent=True)
                        best_model = leaderboard.iloc[0]['model']
                        best_val_score = leaderboard.iloc[0]['score_val']
                        test_score = leaderboard.iloc[0]['score_test'] if 'score_test' in leaderboard.columns else None

                        # Get training times for individual models
                        training_times = {}
                        if 'fit_time' in leaderboard.columns:
                            for _, row in leaderboard.iterrows():
                                training_times[row['model']] = row['fit_time']

                        leaderboard_top3 = leaderboard.head(3)[['model', 'score_val']].to_dict('records')
                        num_models = len(leaderboard)
                    except Exception as e:
                        logger.info(f"Leaderboard not available (likely MultiModalPredictor): {e}")
                        # Fallback for MultiModalPredictor which might not have leaderboard
                        predictor_type = 'MultiModalPredictor' if hasattr(predictor, '__class__') and 'MultiModal' in predictor.__class__.__name__ else 'TabularPredictor'
                        best_model = predictor_type
                        best_val_score = performance if isinstance(performance, (int, float)) else 'N/A'
                        test_score = performance if isinstance(performance, (int, float)) else 'N/A'
                        training_times = {}
                        leaderboard_top3 = [{'model': predictor_type, 'score_val': best_val_score}]
                        num_models = 1

                    results = {
                        'dataset': dataset_name,
                        'performance': performance,
                        'best_model': best_model,
                        'best_val_score': best_val_score,
                        'test_score': test_score,
                        'accuracy': accuracy,
                        'num_models_trained': num_models,
                        'total_models_attempted': num_models,
                        'leaderboard_top3': leaderboard_top3,
                        'model_training_times': training_times,
                        'problem_type': getattr(predictor, 'problem_type', 'unknown'),
                        'eval_metric': getattr(predictor, 'eval_metric', 'unknown'),
                        'predictions_saved': predictions is not None,
                        'confusion_matrix_saved': confusion_matrix is not None,
                        'model_size_mb': model_info.get('model_size_mb', 'N/A'),
                        'num_parameters': model_info.get('num_parameters', 'N/A'),
                        'model_type': model_info.get('model_type', 'Unknown')
                    }

                # Log detailed results
                logger.info(f"Evaluation completed for {dataset_name}")
                logger.info(f"Problem type: {results['problem_type']}")
                logger.info(f"Eval metric: {results['eval_metric']}")
                logger.info(f"Best model: {results['best_model']}")
                logger.info(f"Best validation score: {results['best_val_score']}")
                if results['test_score'] is not None:
                    logger.info(f"Test score: {results['test_score']}")
                if results['accuracy'] != 'N/A':
                    logger.info(f"Accuracy: {results['accuracy']}")
                logger.info(f"Total models trained: {results['num_models_trained']}")
                logger.info(f"Model size: {results['model_size_mb']} MB")
                logger.info(f"Model parameters: {results['num_parameters']}")

                # Log top models
                logger.info("Top models:")
                for i, model_info in enumerate(results['leaderboard_top3'], 1):
                    logger.info(f"  {i}. {model_info['model']}: {model_info['score_val']}")

                return results

        except Exception as e:
            logger.error(f"Error evaluating {dataset_name}: {e}")
            return {
                'dataset': dataset_name,
                'performance': 'Error',
                'best_model': 'Error',
                'best_val_score': 'Error',
                'test_score': 'Error',
                'accuracy': 'Error',
                'num_models_trained': 0,
                'total_models_attempted': 0,
                'leaderboard_top3': [],
                'model_training_times': {},
                'problem_type': 'Error',
                'eval_metric': 'Error',
                'predictions_saved': False,
                'confusion_matrix_saved': False,
                'model_size_mb': 'Error',
                'num_parameters': 'Error',
                'model_type': 'Error'
            }
    
    def run_all_experiments(self):
        """Run training on all datasets"""
        datasets = [
            # ("casehold", self.load_casehold_dataset),
            # ("scienceqa", self.load_scienceqa_dataset),
            # ("anli_r1", self.load_anli_dataset),
            # ("temperature_rain", self.load_temperature_rain_dataset)
            # ("casehold_imbalanced", lambda: load_dataset_from_csv("/storage/nammt/autogluon/casehold_imbalanced", "casehold_imbalanced", "label"))
            ("anli_r1_noisy", lambda: load_dataset_from_csv("/storage/nammt/autogluon/anli_r1_noisy", "anli_r1_noisy", "label"))
        ]
        
        all_results = []
        
        for dataset_name, load_func in datasets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {dataset_name.upper()}")
            logger.info(f"{'='*60}")

            try:
                # Load dataset
                train_df, val_df, test_df, target_col = load_func()
                
                if train_df is None:
                    logger.warning(f"Skipping {dataset_name} due to loading error")
                    continue
                

                # Train model
                predictor, training_time = self.train_model(
                    train_df, val_df, target_col, dataset_name
                )

                # Evaluate model
                results = self.evaluate_model(predictor, test_df, target_col, dataset_name)
                results['total_training_time'] = training_time
                results['train_size'] = len(train_df)
                results['val_size'] = len(val_df)
                results['test_size'] = len(test_df)
                results['feature_count'] = len(train_df.columns) - 1

                all_results.append(results)
                self.results[dataset_name] = results
                
            except Exception as e:
                logger.error(f"Error processing {dataset_name}: {e}")
                continue
        
        # Save results summary
        self.save_results_summary(all_results)
        
        return all_results
    
    def save_results_summary(self, results):
        """Save experiment results to CSV"""
        if not results:
            logger.warning("No results to save")
            return
        
        # Create summary DataFrame
        summary_data = []
        for result in results:
            summary_data.append({
                'dataset': result['dataset'],
                'problem_type': result.get('problem_type', 'Unknown'),
                'eval_metric': result.get('eval_metric', 'Unknown'),
                'best_model': result['best_model'],
                'best_val_score': result.get('best_val_score', result.get('best_score', 'N/A')),
                'test_score': result.get('test_score', 'N/A'),
                'accuracy': result.get('accuracy', 'N/A'),
                'total_training_time_seconds': result.get('total_training_time', result.get('training_time', 'N/A')),
                'model_size_mb': result.get('model_size_mb', 'N/A'),
                'num_parameters': result.get('num_parameters', 'N/A'),
                'model_type': result.get('model_type', 'Unknown'),
                'train_size': result['train_size'],
                'val_size': result.get('val_size', 'N/A'),
                'test_size': result['test_size'],
                'feature_count': result.get('feature_count', 'N/A'),
                'num_models_trained': result.get('num_models_trained', result.get('num_models', 'N/A')),
                'predictions_saved': result.get('predictions_saved', False),
                'confusion_matrix_saved': result.get('confusion_matrix_saved', False)
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.output_dir, f"autogluon_multi_dataset_results_{timestamp}.csv")
        summary_df.to_csv(results_file, index=False)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print("="*80)

def main():
    """Main execution function"""
    print("AutoGluon Multi-Dataset Training Script")
    print("="*50)
    
    # Create trainer
    trainer = AutoGluonMultiDatasetTrainer()
    
    # Run all experiments
    results = trainer.run_all_experiments()
    
    print(f"\nCompleted training on {len(results)} datasets")
    print("Check the output directory for:")
    print("  - Detailed results and trained models")
    print("  - Predictions CSV files in 'predictions/' subdirectory")
    print("  - Confusion matrices and classification reports in 'confusion_matrices/' subdirectory")

if __name__ == "__main__":
    main()
