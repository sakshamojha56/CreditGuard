"""
Credit Card Default Prediction Project

A comprehensive machine learning project for predicting credit card default risk
using customer payment history, demographic data, and financial behavior patterns.

Author: GitHub Copilot
Date: December 2024
"""

__version__ = "1.0.0"
__author__ = "GitHub Copilot"
__email__ = "github-copilot@github.com"
__description__ = "Credit Card Default Risk Prediction using Machine Learning"

# Project structure
PROJECT_STRUCTURE = {
    "data/": "Raw and processed datasets",
    "notebooks/": "Jupyter notebooks for analysis and modeling",
    "src/": "Source code modules",
    "models/": "Trained model artifacts",
    "results/": "Model outputs and predictions",
    "requirements.txt": "Python dependencies"
}

# Key features to engineer
FEATURE_CATEGORIES = {
    "demographic": ["Customer_ID", "sex", "education", "marriage", "age"],
    "financial": ["LIMIT_BAL", "AVG_Bill_amt", "PAY_TO_BILL_ratio"],
    "payment_status": ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"],
    "bill_amounts": ["Bill_amt1", "Bill_amt2", "Bill_amt3", "Bill_amt4", "Bill_amt5", "Bill_amt6"],
    "payment_amounts": ["pay_amt1", "pay_amt2", "pay_amt3", "pay_amt4", "pay_amt5", "pay_amt6"],
    "target": ["next_month_default"]
}

# Model evaluation metrics
EVALUATION_METRICS = [
    "accuracy", "precision", "recall", "f1_score", "f2_score", "roc_auc"
]

print(f"Credit Card Default Prediction Project v{__version__}")
print(f"by {__author__}")
print("\n📊 Key Statistics:")
print("- Dataset: 25,249 customers")
print("- Features: 26 variables")
print("- Task: Binary classification")
print("- Primary Metric: F2 Score (emphasizes Recall)")
print("\n🎯 Project Goals:")
print("1. Predict customers likely to default next month")
print("2. Identify key risk factors")
print("3. Provide interpretable predictions")
print("4. Optimize for business impact")
