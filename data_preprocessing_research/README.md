---
configs:
- config_name: ag_news
  data_files:
  - path: ag_news/train.csv
    split: train
  - path: ag_news/validation.csv
    split: validation
  - path: ag_news/test.csv
    split: test
- config_name: amazon_polarity
  data_files:
  - path: amazon_polarity/train.csv
    split: train
  - path: amazon_polarity/validation.csv
    split: validation
  - path: amazon_polarity/test.csv
    split: test
- config_name: emotion
  data_files:
  - path: emotion/train.csv
    split: train
  - path: emotion/validation.csv
    split: validation
  - path: emotion/test.csv
    split: test
- config_name: imdb
  data_files:
  - path: imdb/train.csv
    split: train
  - path: imdb/validation.csv
    split: validation
  - path: imdb/test.csv
    split: test
- config_name: twenty_newsgroups
  data_files:
  - path: twenty_newsgroups/train.csv
    split: train
  - path: twenty_newsgroups/validation.csv
    split: validation
  - path: twenty_newsgroups/test.csv
    split: test
- config_name: yelp_polarity
  data_files:
  - path: yelp_polarity/train.csv
    split: train
  - path: yelp_polarity/validation.csv
    split: validation
  - path: yelp_polarity/test.csv
    split: test
dataset_info:
- config_name: ag_news
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 90000
  - name: validation
    num_examples: 30000
  - name: test
    num_examples: 7600
- config_name: amazon_polarity
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 2700000
  - name: validation
    num_examples: 900000
  - name: test
    num_examples: 400000
- config_name: emotion
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 250085
  - name: validation
    num_examples: 83362
  - name: test
    num_examples: 41681
- config_name: imdb
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 18750
  - name: validation
    num_examples: 6250
  - name: test
    num_examples: 25000
- config_name: twenty_newsgroups
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 8485
  - name: validation
    num_examples: 2829
  - name: test
    num_examples: 7532
- config_name: yelp_polarity
  features:
  - dtype: string
    name: text
  - dtype: int64
    name: label
  splits:
  - name: train
    num_examples: 420000
  - name: validation
    num_examples: 140000
  - name: test
    num_examples: 38000
language:
- en
license: apache-2.0
size_categories:
- 1K<n<10K
- 10K<n<100K
tags:
- data-preprocessing
- automl
- quality-issues
- benchmarks
task_categories:
- text-classification
---

# Data Preprocessing AutoML Benchmarks

This repository contains text classification datasets with known data quality issues for preprocessing research in AutoML.

## Dataset Categories

### Redundancy Issues
- **ag_news**: News categorization with topic overlap
- **twenty_newsgroups**: Newsgroup posts with cross-posting

### Class Imbalance Issues
- **yelp_polarity**: Sentiment analysis with rating bias

### Label Noise Issues
- **imdb**: Movie reviews with subjective labels
- **amazon_polarity**: Product reviews with rating inconsistencies

### Outlier Issues
- **emotion**: Twitter emotion with length outliers

## Dataset Structure

Each dataset contains:
- `train.csv`: Training split (~75% of original training data)
- `validation.csv`: Validation split (~25% of original training data)
- `test.csv`: Test split (original test set preserved)

All datasets have consistent columns:
- `text`: Input text
- `label`: Target label (integer encoded)

**Important**: Original test sets are preserved to maintain methodological integrity and enable comparison with published benchmarks.

## Usage

```python
from datasets import load_dataset

# Load a specific dataset
dataset = load_dataset("MothMalone/data-preprocessing-automl-benchmarks", "ag_news")

# Access splits
train_data = dataset["train"]
val_data = dataset["validation"]
test_data = dataset["test"]
```

## Dataset Details

ag_news:
  class_names:
  - World
  - Sports
  - Business
  - Technology
  description: News categorization with 4 classes, known for similar content across
    categories
  name: AG News Classification
  num_classes: 4
  original_test_samples: 7600
  original_train_samples: 120000
  quality_issues:
  - redundancy
  - similar_content
  - topic_overlap
  target_column: label
  task_type: multi_classification
  test_samples: 7600
  text_columns:
  - text
  total_samples: 127600
  train_samples: 90000
  validation_samples: 30000
amazon_polarity:
  class_names:
  - negative
  - positive
  description: Amazon reviews with noisy sentiment labels
  name: Amazon Product Reviews
  num_classes: 2
  original_test_samples: 400000
  original_train_samples: 3600000
  quality_issues:
  - label_noise
  - rating_inconsistency
  target_column: label
  task_type: binary_classification
  test_samples: 400000
  text_columns:
  - text
  total_samples: 4000000
  train_samples: 2700000
  validation_samples: 900000
emotion:
  class_names:
  - sadness
  - joy
  - love
  - anger
  - fear
  - surprise
  description: Twitter emotion classification with text length outliers
  name: Emotion Classification
  num_classes: 6
  original_test_samples: 41681
  original_train_samples: 333447
  quality_issues:
  - length_outliers
  - text_anomalies
  target_column: label
  task_type: multi_classification
  test_samples: 41681
  text_columns:
  - text
  total_samples: 375128
  train_samples: 250085
  validation_samples: 83362
imdb:
  class_names:
  - negative
  - positive
  description: Movie reviews with subjective sentiment labels and borderline cases
  name: IMDB Movie Reviews
  num_classes: 2
  original_test_samples: 25000
  original_train_samples: 25000
  quality_issues:
  - label_noise
  - subjective_labels
  - borderline_cases
  target_column: label
  task_type: binary_classification
  test_samples: 25000
  text_columns:
  - text
  total_samples: 50000
  train_samples: 18750
  validation_samples: 6250
twenty_newsgroups:
  class_names:
  - alt.atheism
  - comp.graphics
  - comp.os.ms-windows.misc
  - comp.sys.ibm.pc.hardware
  - comp.sys.mac.hardware
  - comp.windows.x
  - misc.forsale
  - rec.autos
  - rec.motorcycles
  - rec.sport.baseball
  - rec.sport.hockey
  - sci.crypt
  - sci.electronics
  - sci.med
  - sci.space
  - soc.religion.christian
  - talk.politics.guns
  - talk.politics.mideast
  - talk.politics.misc
  - talk.religion.misc
  description: Newsgroup posts with overlapping topics and cross-posting
  name: 20 Newsgroups
  num_classes: 20
  original_test_samples: 7532
  original_train_samples: 11314
  quality_issues:
  - redundancy
  - cross_posting
  - similar_topics
  target_column: label
  task_type: multi_classification
  test_samples: 7532
  text_columns:
  - text
  total_samples: 18846
  train_samples: 8485
  validation_samples: 2829
yelp_polarity:
  class_names:
  - negative
  - positive
  description: Yelp reviews with positive/negative sentiment, naturally imbalanced
  name: Yelp Review Polarity
  num_classes: 2
  original_test_samples: 38000
  original_train_samples: 560000
  quality_issues:
  - moderate_imbalance
  - rating_bias
  target_column: label
  task_type: binary_classification
  test_samples: 38000
  text_columns:
  - text
  total_samples: 598000
  train_samples: 420000
  validation_samples: 140000


## Citation

If you use these datasets in your research, please cite the original sources and this collection:

```bibtex
@misc{mothmalone2024preprocessing,
  title={Data Preprocessing AutoML Benchmarks},
  author={MothMalone},
  year={2024},
  url={https://huggingface.co/datasets/MothMalone/data-preprocessing-automl-benchmarks}
}
```
