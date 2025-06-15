# 🚀 Quick Start Guide

## Getting Started in 5 Minutes

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

### 2. Validate Setup
```bash
python3 validate_project.py
```

### 3. Run Notebooks (in order)
1. **01_data_exploration.ipynb** - Understand the data
2. **02_feature_engineering.ipynb** - Create predictive features  
3. **03_model_development.ipynb** - Train and compare models
4. **04_model_evaluation.ipynb** - Evaluate and explain results

### 4. Quick Test (Python)
```python
# Load and test the pipeline
import sys
sys.path.append('src')

from data_preprocessing import load_and_preprocess_data
from feature_engineering import engineer_features_for_dataset

# Load data
train_data, _, _ = load_and_preprocess_data("data/train.csv")
print(f"Data loaded: {train_data.shape}")

# Engineer features
engineered_data, features, descriptions = engineer_features_for_dataset(train_data)
print(f"Features engineered: {len(features)} selected")

print("✅ Pipeline working!")
```

## 📊 Key Files

| File | Purpose |
|------|---------|
| `data/train.csv` | Training dataset (25K customers) |
| `src/data_preprocessing.py` | Data cleaning and validation |
| `src/feature_engineering.py` | Feature creation and selection |
| `src/model_training.py` | Model training and comparison |
| `src/evaluation.py` | Model evaluation and metrics |
| `PROJECT_SUMMARY.md` | Comprehensive project overview |

## 🎯 Expected Results

- **F2 Score**: >0.75 (primary metric)
- **Recall**: >80% (catching defaults)
- **Precision**: >60% (minimizing false alarms)
- **ROI**: 3x improvement over baseline

## 🔧 Troubleshooting

**Import errors?** 
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Data not found?**
```bash
# Check if train.csv is in data/ folder
ls data/train.csv
```

**Module not found?**
```python
import sys
sys.path.append('src')  # Add this line before imports
```

## 🎉 Success!

If you can run the notebooks successfully, you have a working credit card default prediction system that's ready for business use!

---
💡 **Tip**: Start with notebook 01 for a complete walkthrough of the data and insights.
