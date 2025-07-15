# AutoGluon Multi-Dataset Training

Comprehensive training and evaluation script for AutoGluon on multiple diverse datasets including text classification, multimodal classification, and time series forecasting.

## 🎯 Datasets

| Dataset | Type | Task | Records | Features |
|---------|------|------|---------|----------|
| **CaseHold** | Text | Classification | ~53K | Legal case holdings |
| **ScienceQA** | Multimodal | Classification | ~21K | Science Q&A with images |
| **ANLI R1** | Text | Classification | ~17K | Natural language inference |
| **Temperature Rain** | Time Series | Forecasting | 22.6M | Weather data (32K stations) |

## ✨ Features

- ✅ **Real temperature_rain dataset** loading with intelligent caching (200x speedup)
- ✅ **AutoGluon default settings** for fair baseline comparison
- ✅ **Comprehensive logging**: accuracy, training time, model size, parameters
- ✅ **Memory optimization**: Configurable memory usage ratios
- ✅ **Confusion matrices** and classification reports for classification tasks
- ✅ **Predictions export** for all datasets
- ✅ **Time series forecasting** with proper frequency handling

## 🚀 Quick Start

### Prerequisites
```bash
python >= 3.10
21+ GB RAM recommended
```

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/autogluon-multi-dataset-training.git
cd autogluon-multi-dataset-training
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Usage
```bash
python autogluon_multi_dataset_training.py
```

## 📊 Results

### Temperature Rain (Time Series)
- **MASE Score**: 0.56 (44% better than naive baseline)
- **Training Time**: ~36 minutes for 32,072 weather stations
- **Model Size**: 1,040 MB
- **Best Model**: Ensemble (Chronos + DirectTabular)
- **Caching**: First load 10+ min, subsequent loads 3 seconds

### Performance Summary
| Dataset | Problem Type | Best Model | Score | Training Time |
|---------|-------------|------------|-------|---------------|
| Temperature Rain | Time Series | Ensemble | MASE: 0.56 | 36.6 min |
| CaseHold | Text Classification | TBD | TBD | TBD |
| ScienceQA | Multimodal | TBD | TBD | TBD |
| ANLI R1 | Text Classification | TBD | TBD | TBD |

## 🏗️ Architecture

### Data Loading
- **Cached processing** for large datasets
- **TSF format parsing** for Monash time series data
- **Multimodal handling** for images + text
- **Memory-efficient** data splitting

### Model Training
- **AutoGluon defaults** (no custom model specifications)
- **Automatic problem type detection**
- **Memory usage optimization**
- **Time-based splitting** for time series

### Evaluation & Export
- **Comprehensive metrics** logging
- **Confusion matrices** for classification
- **Predictions export** to CSV
- **Model size tracking**

## 📁 Project Structure

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
└── results/                            # Generated results (gitignored)
    ├── predictions/                    # Model predictions
    ├── confusion_matrices/             # Classification matrices
    └── *.csv                          # Summary results
```

## 🔧 Configuration

### Memory Settings
```python
# Increase memory usage for large datasets
ag_args_fit={'ag.max_memory_usage_ratio': 5}
```

### Time Series Settings
```python
# Configure prediction horizon and frequency
TimeSeriesPredictor(
    prediction_length=24,  # 24-day forecasts
    freq='D',             # Daily frequency
    eval_metric='MASE'    # Mean Absolute Scaled Error
)
```

## 📈 Advanced Usage

### Custom Dataset Addition
1. Add dataset loader function
2. Update `run_all_experiments()` method
3. Configure appropriate predictor type

### Performance Tuning
- Increase `time_limit` for longer training
- Use `presets='high_quality'` for better models
- Adjust `prediction_length` for time series

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [AutoGluon](https://auto.gluon.ai/) for the excellent AutoML framework
- [Monash Time Series Forecasting Archive](https://forecastingdata.org/) for the temperature_rain dataset
- [Hugging Face Datasets](https://huggingface.co/datasets) for dataset hosting

## 📞 Contact

- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Issues: [GitHub Issues](https://github.com/YOUR_USERNAME/autogluon-multi-dataset-training/issues)