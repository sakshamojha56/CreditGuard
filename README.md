# 💳 CreditGuard: Credit Card Fraud Detection System

**CreditGuard** is a machine learning-based system designed to detect fraudulent credit card transactions using supervised learning techniques. The project addresses the critical issue of class imbalance in fraud detection and aims to enhance financial security and risk mitigation in digital payment systems.

## 📌 Problem Statement

Credit card fraud is a major concern for financial institutions and consumers alike. Given the rarity of fraudulent transactions in real-world data, the main challenge lies in building accurate models that can detect fraud despite severe class imbalance.

## 🧠 ML Approach

- **Models Used**: Logistic Regression, Random Forest, XGBoost, Support Vector Machines (SVM)
- **Key Techniques**:
  - Handling **severely imbalanced data** using **SMOTE** (Synthetic Minority Oversampling Technique)
  - **Feature scaling** and **data preprocessing** for improved model performance
  - **Hyperparameter tuning** to optimize model metrics

## ⚙️ Workflow

1. **Data Cleaning**: Remove nulls, scale numerical features
2. **Exploratory Data Analysis (EDA)**: Understand feature distributions and correlations
3. **SMOTE Oversampling**: Balance the dataset for improved learning
4. **Model Training**: Train and evaluate multiple classification models
5. **Evaluation**: Use metrics like **Precision**, **Recall**, **F1-score**, **AUC**, and **confusion matrix**

## 📊 Evaluation Metrics

- **Confusion Matrix**  
- **Precision-Recall Curve**
- **ROC-AUC Score**
- Focused on minimizing **false negatives** to reduce undetected fraud

## 💻 Tech Stack

- Python
- pandas, numpy
- scikit-learn
- imbalanced-learn (SMOTE)
- matplotlib, seaborn

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/sakshamojha56/CreditGuard.git
   cd CreditGuard

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   
  ## ✅ Results
    Achieved high precision and recall in fraud detection
    Robust model performance on highly imbalanced datasets
    Useful for real-world deployment in financial fraud monitoring systems

  ## 📄 License
  This project is licensed under the MIT License.

