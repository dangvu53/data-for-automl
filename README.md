# AutoGluon Multi-Dataset Training

This repository contains a training and evaluation script for AutoGluon across multiple datasets spanning text classification, multimodal classification, and time series forecasting. The goal is to establish baseline performance metrics using AutoGluon's default configurations.

## Datasets

| Dataset | Type | Task | Records | Description |
|---------|------|------|---------|-------------|
| **CaseHold** | Text | Classification | ~53K | Legal case holdings |
| **ScienceQA** | Multimodal | Classification | ~21K | Science Q&A with images |
| **ANLI R1** | Text | Classification | ~17K | Natural language inference |
| **Temperature Rain** | Time Series | Forecasting | 22.6M | Weather data (32K stations) |



### Prerequisites
- Python 3.10 or higher
- 21+ GB RAM recommended for full dataset processing


### Usage
```bash
python autogluon_multi_dataset_training.py
```

## Results


## Implementation Details




## Project Structure

```
autogluon-multi-dataset-training/
├── autogluon_multi_dataset_training.py  # Main training script
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
├── .gitignore                          # Git ignore rules
├── monash_tsf/                         # Monash TSF dataset loader
│   ├── monash_tsf.py                   # Dataset configuration
│   ├── utils.py                        # TSF parsing utilities
│   └── data/                           # Dataset files
└── autogluon_multi_results/                            # Generated results (gitignored)
    ├── predictions/                    # Model predictions
    ├── confusion_matrices/             # Classification matrices
    └── *.csv                          # Summary results
```

## Configuration

### Time Series Settings (Temperature Rain)
```python
# TimeSeriesPredictor configuration
predictor = TimeSeriesPredictor(
    path=model_dir,
    target='target',
    prediction_length=24,  # Predict 24 time steps ahead
    eval_metric='MASE',
    freq='D'  # Daily frequency for temperature_rain dataset
)

# Training settings
predictor.fit(
    train_data=train_df,
    tuning_data=val_df,
    presets='medium_quality',
    time_limit=3000,
    verbosity=2
)
```

### Tabular Settings (CaseHold, ANLI R1)
```python
# TabularPredictor configuration for text classification
predictor = TabularPredictor(
    label=target_col,
    path=model_dir,
    problem_type='multiclass',
    eval_metric='accuracy'
)

# Training settings with memory optimization
predictor.fit(
    train_data=train_df,
    tuning_data=val_df,
    presets='medium_quality',
    time_limit=3000,
    verbosity=2,
    ag_args_fit={'ag.max_memory_usage_ratio': 5}
)
```

### Multimodal Settings (ScienceQA)
```python
# MultiModalPredictor for datasets with images and text
predictor = MultiModalPredictor(
    label=target_col,
    path=model_dir
)

# Training settings (combines train and validation data internally)
predictor.fit(
    train_data=combined_train_df,
    time_limit=3000,
    presets='medium_quality'
)
```

## Advanced Usage

### Adding Custom Datasets
1. Add a dataset loader function following the existing pattern
2. Update the `run_all_experiments()` method to include the new dataset
3. Configure the appropriate predictor type for your data

### Performance Tuning
- Increase `time_limit` for longer training sessions
- Use `presets='high_quality'` for better model performance
- Adjust `prediction_length` for time series forecasting tasks

## License

This project is licensed under the MIT License.

## Acknowledgments

- [AutoGluon](https://auto.gluon.ai/) for the AutoML framework
- [Monash Time Series Forecasting Archive](https://forecastingdata.org/) for the temperature_rain dataset
- [Hugging Face Datasets](https://huggingface.co/datasets) for dataset hosting