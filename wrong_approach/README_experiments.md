# ML-Enhanced Meta-Learning Experiments

This directory contains the implementation of ML-enhanced meta-learning for automated data pipeline discovery across multiple datasets.

## 🚀 Quick Start

### Run Single Dataset Experiment

```bash
# Run ANLI experiment (subset mode for faster testing)
python3 run_all_datasets_experiment.py --single anli --mode subset

# Run CaseHOLD experiment (full dataset)
python3 run_all_datasets_experiment.py --single casehold --mode full

# Run ScienceQA experiment
python3 run_all_datasets_experiment.py --single scienceqa --mode subset

# Run Temperature Rain experiment
python3 run_all_datasets_experiment.py --single temperature_rain --mode subset
```

### Run Multi-Dataset Experiment

```bash
# Run all datasets (subset mode - recommended for testing)
python3 run_all_datasets_experiment.py --mode subset

# Run specific datasets
python3 run_all_datasets_experiment.py --datasets anli casehold --mode subset

# Run all datasets (full mode - takes much longer)
python3 run_all_datasets_experiment.py --mode full
```

## 📊 Supported Datasets

| Dataset | Type | Description | Subset Size | Full Size |
|---------|------|-------------|-------------|-----------|
| **ANLI** | Text Classification | Adversarial NLI | 5K/1K/1K | 17K/1K/1K |
| **CaseHOLD** | Legal Text | Legal case holding prediction | 3K/500/500 | 53K/3K/3K |
| **ScienceQA** | Multimodal QA | Science question answering | 2K/500/500 | 21K/1K/2K |
| **Temperature Rain** | Time Series | Synthetic weather data | 10K/2K/2K | Generated |

## 🧠 How It Works

### ML-Enhanced Meta-Learning Flow

```
Raw Data → Embeddings → ML Pipeline Prediction → Pipeline on Raw Data → Final Embeddings → AutoGluon
```

1. **Dataset Analysis**: Create embeddings to understand dataset characteristics
2. **ML Prediction**: Use trained ML models to predict optimal pipeline parameters
3. **Pipeline Application**: Apply predicted operations (clean, filter, select, augment) to raw data
4. **Final Embedding**: Create embeddings from processed data
5. **AutoGluon Training**: Train models on final embeddings

### Key Features

- ✅ **Embedding-Guided**: Uses dataset embeddings to predict optimal pipelines
- ✅ **Hybrid Approach**: Combines evolutionary search with ML prediction
- ✅ **Cross-Dataset Learning**: ML models learn from multiple dataset types
- ✅ **Automated Pipeline Discovery**: No manual parameter tuning required
- ✅ **Unified Framework**: Works across text, multimodal, and time series data

## 📈 Expected Results

### Performance Metrics

- **Meta-learning Fitness**: 0.55-0.65 (higher is better)
- **Data Retention**: 70-95% (percentage of data kept after pipeline)
- **AutoGluon Performance**: Varies by dataset complexity
- **Execution Time**: 5-15 minutes per dataset (subset mode)

### Example Output

```
✅ ANLI experiment completed successfully!
   Fitness: 0.5759
   Retention: 90.00%
   Time: 5m 38.7s
   Performance: {'accuracy': 0.31, 'balanced_accuracy': 0.32, 'mcc': -0.01}
```

## 🔧 Configuration

### Subset Sizes (for faster testing)

```python
subset_sizes = {
    'anli': {'train': 5000, 'val': 1000, 'test': 1000},
    'casehold': {'train': 3000, 'val': 500, 'test': 500},
    'scienceqa': {'train': 2000, 'val': 500, 'test': 500},
    'temperature_rain': {'train': 10000, 'val': 2000, 'test': 2000}
}
```

### Meta-Learning Parameters

```python
MLEnhancedMetaLearner(
    population_size=8,    # Population size for evolution
    generations=3,        # Number of evolutionary generations
    ml_candidates=5       # Number of ML-predicted candidates per generation
)
```

## 📁 Output Files

Each experiment generates:

- `results_{dataset}/` - Directory with all results
- `{dataset}_complete_results.json` - Detailed experiment results
- `{dataset}_model/` - Trained AutoGluon model
- `multi_dataset_results_*.json` - Combined results (for multi-dataset runs)

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Warnings**: Normal TensorFlow/PyTorch warnings, can be ignored
2. **Memory Issues**: Use subset mode or reduce population_size
3. **Dataset Loading Errors**: Check internet connection for HuggingFace datasets
4. **Long Execution Times**: Use subset mode for faster testing

### Performance Tips

- Use `--mode subset` for faster experimentation
- Run single datasets first to test setup
- Monitor GPU memory usage for large datasets
- Use `Ctrl+C` to interrupt long-running experiments

## 🔬 Research Applications

This framework enables research in:

- **Automated ML Pipeline Discovery**
- **Cross-Dataset Transfer Learning**
- **Embedding-Based Meta-Learning**
- **Multi-Modal Data Processing**
- **Time Series Pipeline Optimization**

## 📚 Dependencies

- Python 3.8+
- AutoGluon
- HuggingFace Datasets
- Sentence Transformers
- Scikit-learn
- Pandas, NumPy
- CUDA (optional, for GPU acceleration)

## 🎯 Next Steps

1. **Scale Up**: Run full experiments on all datasets
2. **Cross-Dataset Learning**: Train ML models on multiple datasets
3. **Model Persistence**: Save/load trained ML models
4. **Custom Datasets**: Add your own datasets to the framework
5. **Hyperparameter Tuning**: Optimize meta-learning parameters

---

**Happy Experimenting!** 🚀
