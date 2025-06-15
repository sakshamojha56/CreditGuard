#!/usr/bin/env python3
"""
Project validation script for Credit Card Default Prediction.

This script validates that all components are properly set up and working.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def validate_project_structure():
    """Validate that all required directories and files exist."""
    print("🔍 Validating project structure...")
    
    required_dirs = ['data', 'notebooks', 'src', 'models', 'results']
    required_files = [
        'README.md',
        'requirements.txt', 
        'PROJECT_SUMMARY.md',
        '__init__.py',
        'src/__init__.py',
        'src/data_preprocessing.py',
        'src/feature_engineering.py', 
        'src/model_training.py',
        'src/evaluation.py'
    ]
    
    missing_dirs = []
    missing_files = []
    
    # Check directories
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            missing_dirs.append(dir_name)
    
    # Check files
    for file_name in required_files:
        if not os.path.exists(file_name):
            missing_files.append(file_name)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ Project structure validated successfully!")
    return True

def validate_data():
    """Validate that the training data exists and is readable."""
    print("📊 Validating data...")
    
    data_file = 'data/train.csv'
    if not os.path.exists(data_file):
        print(f"❌ Training data not found: {data_file}")
        return False
    
    try:
        df = pd.read_csv(data_file)
        print(f"✅ Training data loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        # Check for required columns
        required_cols = ['Customer_ID', 'next_month_default', 'LIMIT_BAL', 'age']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Check target distribution
        target_dist = df['next_month_default'].value_counts(normalize=True)
        print(f"📈 Target distribution: {target_dist.to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return False

def validate_modules():
    """Validate that all Python modules can be imported."""
    print("🐍 Validating Python modules...")
    
    try:
        sys.path.append('src')
        
        from data_preprocessing import DataPreprocessor
        from feature_engineering import FeatureEngineer
        from model_training import ModelTrainer
        from evaluation import ModelEvaluator
        
        print("✅ All modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

def validate_notebooks():
    """Validate that all notebooks exist."""
    print("📓 Validating notebooks...")
    
    notebook_dir = 'notebooks'
    expected_notebooks = [
        '01_data_exploration.ipynb',
        '02_feature_engineering.ipynb', 
        '03_model_development.ipynb',
        '04_model_evaluation.ipynb'
    ]
    
    missing_notebooks = []
    for notebook in expected_notebooks:
        notebook_path = os.path.join(notebook_dir, notebook)
        if not os.path.exists(notebook_path):
            missing_notebooks.append(notebook)
    
    if missing_notebooks:
        print(f"❌ Missing notebooks: {missing_notebooks}")
        return False
    
    print("✅ All notebooks found!")
    return True

def run_quick_test():
    """Run a quick end-to-end test of the main pipeline."""
    print("🧪 Running quick pipeline test...")
    
    try:
        sys.path.append('src')
        from data_preprocessing import load_and_preprocess_data
        from feature_engineering import engineer_features_for_dataset
        
        # Load and preprocess a small sample
        print("  Loading data...")
        train_data, _, metadata = load_and_preprocess_data("data/train.csv")
        
        # Sample for quick test
        sample_data = train_data.sample(n=min(1000, len(train_data)), random_state=42)
        
        print("  Engineering features...")
        engineered_data, selected_features, descriptions = engineer_features_for_dataset(sample_data)
        
        print(f"  ✅ Pipeline test completed!")
        print(f"  Original features: {sample_data.shape[1]}")
        print(f"  Engineered features: {engineered_data.shape[1]}")
        print(f"  Selected features: {len(selected_features)}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Pipeline test failed: {str(e)}")
        return False

def print_project_summary():
    """Print a summary of the project."""
    print("\n" + "="*60)
    print("📋 CREDIT CARD DEFAULT PREDICTION PROJECT")
    print("="*60)
    print("🎯 Objective: Predict credit card default risk")
    print("📊 Dataset: 25,249 customers with 26 features")
    print("🔧 Methods: Feature engineering + ML classification")
    print("📈 Target: F2 Score >0.75 (emphasizes recall)")
    print("🏗️ Architecture: Modular Python codebase")
    print("📓 Documentation: 4 comprehensive notebooks")
    print("🚀 Status: Production-ready implementation")
    print("="*60)
    
    print("\n📁 Project Structure:")
    print("├── data/                  # Training dataset")
    print("├── notebooks/             # Analysis & modeling")
    print("├── src/                   # Source code modules")
    print("├── models/                # Trained model artifacts")
    print("├── results/               # Predictions & reports")
    print("├── README.md             # Project documentation")
    print("├── PROJECT_SUMMARY.md    # Comprehensive summary")
    print("└── requirements.txt      # Dependencies")
    
    print("\n🎯 Next Steps:")
    print("1. Run notebooks in order (01 → 02 → 03 → 04)")
    print("2. Explore data patterns and feature engineering")
    print("3. Train and compare multiple models")
    print("4. Evaluate performance and explainability")
    print("5. Deploy to production environment")
    print("="*60)

def main():
    """Main validation function."""
    print("🔍 CREDIT CARD DEFAULT PREDICTION - PROJECT VALIDATION")
    print("="*60)
    
    all_valid = True
    
    # Run all validation checks
    all_valid &= validate_project_structure()
    all_valid &= validate_data()
    all_valid &= validate_modules()
    all_valid &= validate_notebooks()
    all_valid &= run_quick_test()
    
    print("\n" + "="*60)
    if all_valid:
        print("✅ ALL VALIDATIONS PASSED!")
        print("🚀 Project is ready for use!")
    else:
        print("❌ SOME VALIDATIONS FAILED!")
        print("⚠️ Please fix issues before proceeding.")
    
    print_project_summary()

if __name__ == "__main__":
    main()
