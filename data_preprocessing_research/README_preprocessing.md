# Data Preprocessing Techniques for AutoGluon

This repository contains various data preprocessing techniques implemented for use with AutoGluon. These techniques are designed to enhance the quality of input data before feeding it into AutoGluon models.

## Overview

The repository implements three main categories of preprocessing techniques:

1. **Outlier Detection**
2. **Duplicate Removal**
3. **Imbalance Handling**

These techniques can be used individually or combined in a preprocessing pipeline.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd autogluon/data_preprocessing_research

# Install requirements
pip install -r requirements.txt
```

## Outlier Detection

The outlier detection module provides functionality to detect and remove outliers in datasets using various techniques:

- **Z-scores**: Identifies outliers based on the number of standard deviations from the mean
- **IQR (Interquartile Range)**: Identifies outliers based on the interquartile range
- **LOF (Local Outlier Factor)**: Identifies outliers based on local density deviations
- **Isolation Forest**: Identifies outliers using isolation forests, an ensemble method

### Usage

```python
from src.preprocessing.outlier_detection import OutlierDetectionPreprocessor

# Initialize the preprocessor
preprocessor = OutlierDetectionPreprocessor(
    strategy='z_score',  # Choose from: 'z_score', 'iqr', 'lof', 'isolation_forest'
    text_column='text',
    threshold=3.0,       # For z_score strategy
    contamination=0.1,   # For isolation_forest strategy
    verbose=True
)

# Apply to a dataframe
cleaned_df = preprocessor.transform(df)

# Get statistics
stats = preprocessor.get_stats()
```

## Duplicate Removal

The duplicate removal module provides functionality to detect and remove duplicates in datasets:

- **Exact Matching**: Identifies exact duplicates
- **Approximate Matching**: Identifies near-duplicates using various similarity metrics:
  - Levenshtein distance
  - Jaccard similarity
  - Cosine similarity
  - And more...

### Usage

```python
from src.preprocessing.duplicate_removal_enhanced import DuplicateRemovalPreprocessor

# Initialize the preprocessor
preprocessor = DuplicateRemovalPreprocessor(
    strategy='exact',    # Choose from: 'exact', 'approximate'
    text_column='text',
    label_column='label',
    similarity_threshold=0.9,  # For approximate strategy
    similarity_metric='levenshtein',  # For approximate strategy
    verbose=True
)

# Apply to a dataframe
cleaned_df = preprocessor.transform(df)

# Get statistics
stats = preprocessor.get_stats()
```

## Imbalance Handling

The imbalance handling module provides functionality to address class imbalance in datasets:

- **Random Oversampling**: Randomly duplicates samples from minority classes
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Creates synthetic samples from minority classes
- **ADASYN (Adaptive Synthetic Sampling)**: Creates synthetic samples with focus on difficult instances
- **EDA (Easy Data Augmentation)**: Text-specific augmentation techniques including:
  - Synonym replacement
  - Random insertion
  - Random swap
  - Random deletion

### Usage

```python
from src.preprocessing.imbalance_handling import ImbalanceHandlingPreprocessor

# Initialize the preprocessor
preprocessor = ImbalanceHandlingPreprocessor(
    strategy='random-oversampling',  # Choose from: 'random-oversampling', 'smote', 'adasyn', 'eda'
    text_column='text',
    label_column='label',
    sampling_strategy='auto',
    random_state=42,
    verbose=True
)

# Apply to a dataframe
balanced_df = preprocessor.transform(df)

# Get statistics
stats = preprocessor.get_stats()
```

## Integrated Pipeline

The integrated preprocessing pipeline allows combining multiple preprocessing techniques:

```python
from src.preprocessing.preprocessing_pipeline import PreprocessingPipeline

# Create a preprocessing pipeline
pipeline = PreprocessingPipeline(
    steps=[
        {
            'type': 'outlier',
            'params': {
                'strategy': 'isolation_forest',
                'text_column': 'text',
                'contamination': 0.1
            },
            'enabled': True
        },
        {
            'type': 'duplicate',
            'params': {
                'strategy': 'exact',
                'text_column': 'text',
                'label_column': 'label'
            },
            'enabled': True
        },
        {
            'type': 'imbalance',
            'params': {
                'strategy': 'random-oversampling',
                'text_column': 'text',
                'label_column': 'label'
            },
            'enabled': True
        }
    ],
    verbose=True
)

# Apply the pipeline
result_df = pipeline.fit_transform(df)

# Get statistics
stats = pipeline.get_stats()
```

## Integration with AutoGluon

The repository provides scripts to preprocess datasets and use them with AutoGluon:

### Preprocessing Datasets

Use the `preprocess_datasets.py` script to apply preprocessing techniques to datasets:

```bash
# Preprocess a specific dataset
python preprocess_datasets.py --dataset ag_news

# Preprocess all datasets
python preprocess_datasets.py --all

# Use a custom preprocessing configuration
python preprocess_datasets.py --all --config my_preprocessing.yaml
```

The script saves the preprocessed datasets to the `prepared_datasets` directory and updates the dataset configuration in `config/preprocessing_applied.yaml`.

### Running Experiments

Use the `run_preprocessed_datasets_3000s.py` script to run experiments with preprocessed datasets:

```bash
# Run experiments with preprocessed datasets
python run_preprocessed_datasets_3000s.py
```

This script is similar to `run_all_datasets_3000s.py` but it uses the preprocessed datasets if available.

### Configuration

The preprocessing pipeline is configured in `config/preprocessing.yaml`:

```yaml
# Preprocessing steps for the pipeline
steps:
  # Outlier Detection
  - type: outlier
    enabled: true
    params:
      strategy: isolation_forest
      contamination: 0.05
      
  # Duplicate Removal
  - type: duplicate
    enabled: true
    params:
      strategy: approximate
      similarity_threshold: 0.9
      
  # Imbalance Handling
  - type: imbalance
    enabled: true
    params:
      strategy: random-oversampling
      sampling_strategy: auto
```

You can define different preprocessing variations for experiments:

```yaml
# Preprocessing variations for experiments
variations:
  minimal:
    steps:
      - type: duplicate
        enabled: true
        params:
          strategy: exact
  
  medium:
    steps:
      - type: outlier
        enabled: true
        params:
          strategy: z_score
      - type: duplicate
        enabled: true
        params:
          strategy: exact
```

## Example Scripts

The repository includes example scripts to demonstrate the use of each preprocessing technique:

- `scripts/test_outlier_detection.py`: Demonstrates outlier detection techniques
- `scripts/test_duplicate_removal.py`: Demonstrates duplicate removal techniques
- `scripts/test_imbalance_handling.py`: Demonstrates imbalance handling techniques
- `scripts/test_preprocessing_pipeline.py`: Demonstrates the integrated preprocessing pipeline

## Dependencies

- numpy
- pandas
- scikit-learn
- scipy
- imbalanced-learn (optional, for SMOTE and ADASYN)
- fuzzywuzzy (optional, for approximate duplicate detection)
- textdistance (optional, for approximate duplicate detection)
- dedupe (optional, for advanced duplicate detection)

## License

[MIT License]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## References

- [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
