# Learn2Clean Setup Instructions

## Issue: ModuleNotFoundError: No module named 'learn2clean'

Learn2Clean is an older package (circa 2018-2019) with dependencies that may conflict with modern Python environments. Here are several approaches to get it working:

## Option 1: Add to Python Path (Recommended for Development)

Add this to the beginning of your notebook cells:

```python
# Add Learn2Clean to Python path
import sys
import os
sys.path.append(os.path.abspath('../python-package'))

# Now import Learn2Clean modules
import learn2clean.loading.reader as rd
import learn2clean.normalization.normalizer as nl
# ... other imports
```

## Option 2: Install Missing Dependencies

Learn2Clean requires several older packages. Install them individually:

```bash
# Core dependencies that usually work
pip install jellyfish joblib matplotlib seaborn statsmodels tdda

# Potentially problematic dependencies (try if needed)
pip install fancyimpute impyute py_stringmatching py_stringsimjoin
pip install sklearn_contrib_py_earth  # May require compilation
```

## Option 3: Create a Conda Environment (Most Reliable)

```bash
# Create a new conda environment with Python 3.6-3.8
conda create -n learn2clean python=3.8
conda activate learn2clean

# Install dependencies
conda install pandas=0.23.0 numpy=1.14.3 scipy=1.2.1 matplotlib=2.2.2
conda install scikit-learn joblib seaborn statsmodels
pip install jellyfish tdda fancyimpute impyute
```

## Option 4: Docker Container

Create a Dockerfile with the exact environment:

```dockerfile
FROM python:3.8-slim
WORKDIR /app
COPY Learn2Clean/python-package/requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

## Option 5: Use Alternative Libraries

If Learn2Clean proves too difficult to install, consider modern alternatives:

- **AutoML Libraries**: AutoGluon, H2O AutoML, TPOT
- **Data Cleaning**: pandas-profiling, missingno, dataprep
- **Feature Engineering**: feature-engine, sklearn preprocessing
- **Automated Preprocessing**: auto-sklearn, FLAML

## Troubleshooting Common Issues

### 1. Compilation Errors
Some packages like `fancyimpute` and `sklearn_contrib_py_earth` require compilation. Install build tools:

```bash
# Ubuntu/Debian
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Windows
# Install Microsoft C++ Build Tools
```

### 2. Version Conflicts
Learn2Clean was built for older package versions. If you get version conflicts:

```bash
# Force install specific versions (use with caution)
pip install pandas==0.23.0 --force-reinstall
pip install numpy==1.14.3 --force-reinstall
```

### 3. Import Errors
If imports fail, check the Python path:

```python
import sys
print(sys.path)
# Make sure the Learn2Clean path is included
```

## Testing the Installation

Use the provided test script:

```bash
cd Learn2Clean/examples
python3 test_learn2clean_import.py
```

## Alternative: Conceptual Implementation

If installation fails, you can still understand Learn2Clean concepts and implement similar functionality using modern libraries:

```python
# Modern equivalent using sklearn and pandas
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import GridSearchCV
import pandas as pd

# This provides similar functionality to Learn2Clean
# but with modern, maintained libraries
```

## For AutoGluon Integration

The key principle remains the same regardless of the preprocessing library:

1. **Train preprocessing on training data only**
2. **Apply same preprocessing to validation/test data**
3. **Avoid data leakage**

```python
# Correct approach for any preprocessing library
# 1. Fit on training data
preprocessor.fit(X_train)

# 2. Transform all splits with same parameters
X_train_clean = preprocessor.transform(X_train)
X_val_clean = preprocessor.transform(X_val)  # Same parameters!
X_test_clean = preprocessor.transform(X_test)  # Same parameters!

# 3. Use with AutoGluon
predictor = TabularPredictor(label='target')
predictor.fit(X_train_clean, validation_data=X_val_clean)
```

This ensures no data leakage and valid model evaluation with AutoGluon.
