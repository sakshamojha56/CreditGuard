"""
Data preprocessing utilities for credit card default prediction.

This module contains functions for:
- Data loading and validation
- Missing value handling
- Categorical encoding
- Data type conversions
- Timeline validation for payment data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

class DataPreprocessor:
    """Main class for data preprocessing operations."""
    
    def __init__(self):
        self.categorical_mappings = {}
        self.scaler = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset from CSV file with basic validation.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with loaded data
        """
        try:
            df = pd.read_csv(file_path)
            print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
    
    def validate_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate expected columns are present and identify column types.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with column categorization
        """
        expected_cols = {
            'demographic': ['Customer_ID', 'sex', 'education', 'marriage', 'age'],
            'financial': ['LIMIT_BAL', 'AVG_Bill_amt', 'PAY_TO_BILL_ratio'],
            'payment_status': [f'pay_{i}' for i in range(7)],  # pay_0 to pay_6
            'bill_amounts': [f'Bill_amt{i}' for i in range(1, 7)],  # Bill_amt1 to 6
            'payment_amounts': [f'pay_amt{i}' for i in range(1, 7)],  # pay_amt1 to 6
            'target': ['next_month_default']
        }
        
        missing_cols = []
        present_cols = {}
        
        for category, cols in expected_cols.items():
            present = [col for col in cols if col in df.columns]
            missing = [col for col in cols if col not in df.columns]
            
            present_cols[category] = present
            if missing:
                missing_cols.extend(missing)
        
        if missing_cols:
            warnings.warn(f"Missing expected columns: {missing_cols}")
        
        return present_cols
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        df_clean = df.copy()
        
        # Check for missing values
        missing_summary = df_clean.isnull().sum()
        missing_cols = missing_summary[missing_summary > 0]
        
        if len(missing_cols) > 0:
            print("Missing values found:")
            print(missing_cols)
            
            # Handle missing values based on column type
            for col in missing_cols.index:
                if col in ['sex', 'education', 'marriage']:
                    # For categorical: use mode
                    mode_val = df_clean[col].mode()[0]
                    df_clean.loc[:, col] = df_clean[col].fillna(mode_val)
                    print(f"Filled {col} missing values with mode: {mode_val}")
                    
                elif col in ['age', 'LIMIT_BAL']:
                    # For numerical: use median
                    median_val = df_clean[col].median()
                    df_clean.loc[:, col] = df_clean[col].fillna(median_val)
                    print(f"Filled {col} missing values with median: {median_val}")
                    
                elif 'pay_' in col or 'Bill_amt' in col or 'pay_amt' in col:
                    # For payment data: use 0 (no activity)
                    df_clean.loc[:, col] = df_clean[col].fillna(0)
                    print(f"Filled {col} missing values with 0")
        else:
            print("No missing values found in the dataset.")
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical variables with proper mappings.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit new encodings or use existing ones
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df_encoded = df.copy()
        
        # Define categorical mappings based on data dictionary
        categorical_mappings = {
            'sex': {0: 'Female', 1: 'Male'},
            'education': {1: 'Graduate_School', 2: 'University', 3: 'High_School', 4: 'Others'},
            'marriage': {1: 'Married', 2: 'Single', 3: 'Others'}
        }
        
        if fit:
            self.categorical_mappings = categorical_mappings
        
        # Create dummy variables for categorical columns
        for col, mapping in categorical_mappings.items():
            if col in df_encoded.columns:
                # Create descriptive names for categories
                df_encoded[f'{col}_category'] = df_encoded[col].map(mapping)
                
                # Create dummy variables
                dummies = pd.get_dummies(df_encoded[f'{col}_category'], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                
                # Drop original categorical column and temporary category column
                df_encoded.drop([col, f'{col}_category'], axis=1, inplace=True)
        
        return df_encoded
    
    def validate_payment_timeline(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Validate payment timeline consistency and detect anomalies.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with validation results
        """
        validation_results = {
            'total_records': len(df),
            'anomalies_found': 0,
            'invalid_payment_status': 0,
            'negative_limits': 0,
            'extreme_ratios': 0
        }
        
        # Check for invalid payment status values
        pay_cols = [col for col in df.columns if col.startswith('pay_') and col != 'pay_amt']
        for col in pay_cols:
            if col in df.columns:
                invalid_vals = df[~df[col].isin([-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])][col]
                if len(invalid_vals) > 0:
                    validation_results['invalid_payment_status'] += len(invalid_vals)
                    print(f"Found {len(invalid_vals)} invalid values in {col}")
        
        # Check for negative credit limits
        if 'LIMIT_BAL' in df.columns:
            negative_limits = df[df['LIMIT_BAL'] < 0]
            validation_results['negative_limits'] = len(negative_limits)
            if len(negative_limits) > 0:
                print(f"Found {len(negative_limits)} records with negative credit limits")
        
        # Check for extreme payment ratios
        if 'PAY_TO_BILL_ratio' in df.columns:
            extreme_ratios = df[df['PAY_TO_BILL_ratio'] > 5]  # More than 5x payments vs bills
            validation_results['extreme_ratios'] = len(extreme_ratios)
            if len(extreme_ratios) > 0:
                print(f"Found {len(extreme_ratios)} records with extreme payment ratios (>5)")
        
        validation_results['anomalies_found'] = (
            validation_results['invalid_payment_status'] + 
            validation_results['negative_limits'] + 
            validation_results['extreme_ratios']
        )
        
        return validation_results
    
    def preprocess_dataset(self, df: pd.DataFrame, fit: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            fit: Whether to fit preprocessing steps or use existing
            
        Returns:
            Tuple of (processed DataFrame, preprocessing metadata)
        """
        print("Starting data preprocessing...")
        
        # 1. Validate columns
        column_info = self.validate_columns(df)
        print(f"Column validation completed. Found {sum(len(v) for v in column_info.values())} expected columns.")
        
        # 2. Handle missing values
        df_clean = self.handle_missing_values(df)
        
        # 3. Validate payment timeline
        validation_results = self.validate_payment_timeline(df_clean)
        
        # 4. Encode categorical variables
        df_processed = self.encode_categorical_variables(df_clean, fit=fit)
        
        # 5. Basic data type conversions
        # Ensure payment status columns are integers
        pay_cols = [col for col in df_processed.columns if col.startswith('pay_') and '_' not in col[4:]]
        for col in pay_cols:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(int)
        
        # Ensure target is binary
        if 'next_month_default' in df_processed.columns:
            df_processed['next_month_default'] = df_processed['next_month_default'].astype(int)
        
        metadata = {
            'original_shape': df.shape,
            'processed_shape': df_processed.shape,
            'column_info': column_info,
            'validation_results': validation_results,
            'categorical_mappings': self.categorical_mappings
        }
        
        print(f"Preprocessing completed. Shape: {df.shape} -> {df_processed.shape}")
        
        return df_processed, metadata


def load_and_preprocess_data(train_path: str, val_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict]:
    """
    Convenience function to load and preprocess training and validation data.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data (optional)
        
    Returns:
        Tuple of (train_df, val_df, preprocessing_metadata)
    """
    preprocessor = DataPreprocessor()
    
    # Load and preprocess training data
    print("Loading training data...")
    train_df = preprocessor.load_data(train_path)
    train_processed, metadata = preprocessor.preprocess_dataset(train_df, fit=True)
    
    val_processed = None
    if val_path:
        print("Loading validation data...")
        val_df = preprocessor.load_data(val_path)
        val_processed, _ = preprocessor.preprocess_dataset(val_df, fit=False)
        print(f"Validation data shape: {val_processed.shape}")
    
    return train_processed, val_processed, metadata


if __name__ == "__main__":
    # Example usage
    train_data, val_data, meta = load_and_preprocess_data("../data/train.csv")
    print("\nPreprocessing Summary:")
    print(f"Training data: {meta['processed_shape']}")
    print(f"Anomalies found: {meta['validation_results']['anomalies_found']}")
    print(f"Categorical mappings: {list(meta['categorical_mappings'].keys())}")
