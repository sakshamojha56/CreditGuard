"""
Feature engineering module for credit card default prediction.

This module creates meaningful financial and behavioral features from raw data:
- Credit utilization metrics
- Payment behavior patterns  
- Delinquency analysis
- Financial stability indicators
- Temporal trends
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings

class FeatureEngineer:
    """Main class for feature engineering operations."""
    
    def __init__(self):
        self.feature_names = []
        self.feature_descriptions = {}
    
    def create_credit_utilization_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create credit utilization and balance-related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new utilization features
        """
        df_new = df.copy()
        
        # Credit utilization ratio
        if 'AVG_Bill_amt' in df.columns and 'LIMIT_BAL' in df.columns:
            df_new['credit_utilization_ratio'] = df_new['AVG_Bill_amt'] / (df_new['LIMIT_BAL'] + 1e-8)
            df_new['credit_utilization_ratio'] = np.clip(df_new['credit_utilization_ratio'], 0, 5)  # Cap at 500%
            
            # High utilization flag (>80%)
            df_new['high_utilization_flag'] = (df_new['credit_utilization_ratio'] > 0.8).astype(int)
            
            # Credit headroom (available credit)
            df_new['credit_headroom'] = np.maximum(0, df_new['LIMIT_BAL'] - df_new['AVG_Bill_amt'])
            df_new['credit_headroom_ratio'] = df_new['credit_headroom'] / (df_new['LIMIT_BAL'] + 1e-8)
        
        # Bill amount volatility
        bill_cols = [f'Bill_amt{i}' for i in range(1, 7) if f'Bill_amt{i}' in df.columns]
        if len(bill_cols) >= 3:
            df_new['bill_amount_std'] = df[bill_cols].std(axis=1)
            df_new['bill_amount_cv'] = df_new['bill_amount_std'] / (df[bill_cols].mean(axis=1) + 1e-8)
            df_new['bill_volatility_high'] = (df_new['bill_amount_cv'] > 1.0).astype(int)
        
        return df_new
    
    def create_payment_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment behavior and pattern features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with payment behavior features
        """
        df_new = df.copy()
        
        # Payment gap analysis
        bill_cols = [f'Bill_amt{i}' for i in range(1, 7) if f'Bill_amt{i}' in df.columns]
        pay_cols = [f'pay_amt{i}' for i in range(1, 7) if f'pay_amt{i}' in df.columns]
        
        if len(bill_cols) >= 3 and len(pay_cols) >= 3:
            # Calculate payment gaps (how much less was paid vs bill)
            payment_gaps = []
            for i in range(min(len(bill_cols), len(pay_cols))):
                gap = df[bill_cols[i]] - df[pay_cols[i]]
                payment_gaps.append(gap)
            
            payment_gap_df = pd.DataFrame(payment_gaps).T
            
            # Average payment gap
            df_new['avg_payment_gap'] = payment_gap_df.mean(axis=1)
            
            # Payment gap trend (is gap increasing?)
            df_new['payment_gap_trend'] = payment_gap_df.iloc[:, -1] - payment_gap_df.iloc[:, 0]
            
            # Consistent underpayment flag
            df_new['consistent_underpayment'] = (payment_gap_df > 0).sum(axis=1)
            df_new['chronic_underpayment_flag'] = (df_new['consistent_underpayment'] >= 4).astype(int)
        
        # Payment amount analysis
        if len(pay_cols) >= 3:
            # Payment consistency
            df_new['payment_amount_std'] = df[pay_cols].std(axis=1)
            df_new['payment_amount_mean'] = df[pay_cols].mean(axis=1)
            df_new['payment_consistency'] = df_new['payment_amount_std'] / (df_new['payment_amount_mean'] + 1e-8)
            
            # Zero payment months
            df_new['zero_payment_months'] = (df[pay_cols] == 0).sum(axis=1)
            df_new['frequent_zero_payments'] = (df_new['zero_payment_months'] >= 3).astype(int)
            
            # Payment trend
            recent_payments = df[pay_cols[-3:]].mean(axis=1)  # Last 3 months
            early_payments = df[pay_cols[:3]].mean(axis=1)    # First 3 months
            df_new['payment_trend'] = recent_payments - early_payments
            df_new['declining_payments'] = (df_new['payment_trend'] < -1000).astype(int)
        
        # Enhanced PAY_TO_BILL_ratio features
        if 'PAY_TO_BILL_ratio' in df.columns:
            df_new['underpayment_severe'] = (df_new['PAY_TO_BILL_ratio'] < 0.3).astype(int)
            df_new['overpayment_flag'] = (df_new['PAY_TO_BILL_ratio'] > 1.2).astype(int)
            df_new['balanced_payment'] = ((df_new['PAY_TO_BILL_ratio'] >= 0.8) & 
                                         (df_new['PAY_TO_BILL_ratio'] <= 1.2)).astype(int)
        
        return df_new
    
    def create_delinquency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create delinquency and payment status features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with delinquency features
        """
        df_new = df.copy()
        
        # Find payment status columns
        pay_status_cols = [f'pay_{i}' for i in range(7) if f'pay_{i}' in df.columns]
        
        if len(pay_status_cols) >= 3:
            pay_status_df = df[pay_status_cols]
            
            # Count of delinquent months (payment status >= 1)
            df_new['delinquency_count'] = (pay_status_df >= 1).sum(axis=1)
            df_new['high_delinquency'] = (df_new['delinquency_count'] >= 3).astype(int)
            
            # Maximum delinquency level
            df_new['max_delinquency'] = pay_status_df.max(axis=1)
            df_new['severe_delinquency'] = (df_new['max_delinquency'] >= 3).astype(int)
            
            # Recent delinquency (last 3 months)
            recent_pay_cols = pay_status_cols[:3]  # pay_0, pay_2, pay_3 are most recent
            df_new['recent_delinquency_count'] = (df[recent_pay_cols] >= 1).sum(axis=1)
            df_new['recent_delinquency_flag'] = (df_new['recent_delinquency_count'] >= 2).astype(int)
            
            # Delinquency streak analysis
            def calculate_longest_streak(row):
                streak = 0
                max_streak = 0
                for val in row:
                    if val >= 1:
                        streak += 1
                        max_streak = max(max_streak, streak)
                    else:
                        streak = 0
                return max_streak
            
            df_new['longest_delinquency_streak'] = pay_status_df.apply(calculate_longest_streak, axis=1)
            df_new['long_streak_flag'] = (df_new['longest_delinquency_streak'] >= 3).astype(int)
            
            # Delinquency pattern - worsening or improving
            early_delinq = df[pay_status_cols[-3:]].mean(axis=1)  # Earlier months
            recent_delinq = df[pay_status_cols[:3]].mean(axis=1)  # Recent months
            df_new['delinquency_trend'] = recent_delinq - early_delinq
            df_new['worsening_delinquency'] = (df_new['delinquency_trend'] > 0.5).astype(int)
            
            # No-bill vs full payment pattern
            df_new['no_bill_months'] = (pay_status_df == -2).sum(axis=1)
            df_new['full_payment_months'] = (pay_status_df == -1).sum(axis=1)
            df_new['responsible_payment_ratio'] = (df_new['full_payment_months'] / 
                                                 np.maximum(1, 7 - df_new['no_bill_months']))
        
        return df_new
    
    def create_financial_stability_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features indicating financial stability and risk.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with financial stability features
        """
        df_new = df.copy()
        
        # Age-based risk profiling
        if 'age' in df.columns:
            # Create numeric age categories instead of string labels
            df_new['age_risk_category'] = pd.cut(df_new['age'], 
                                               bins=[0, 25, 35, 50, 100], 
                                               labels=[0, 1, 2, 3])  # 0=very_young, 1=young, 2=middle, 3=mature
            df_new['age_risk_category'] = df_new['age_risk_category'].astype(float)
            df_new['high_risk_age'] = ((df_new['age'] < 25) | (df_new['age'] > 65)).astype(int)
        
        # Credit limit analysis
        if 'LIMIT_BAL' in df.columns:
            # Create numeric credit limit categories instead of string labels
            df_new['credit_limit_category'] = pd.cut(df_new['LIMIT_BAL'], 
                                                   bins=[0, 50000, 200000, 500000, np.inf],
                                                   labels=[0, 1, 2, 3])  # 0=low, 1=medium, 2=high, 3=premium
            df_new['credit_limit_category'] = df_new['credit_limit_category'].astype(float)
            df_new['low_credit_limit'] = (df_new['LIMIT_BAL'] < 100000).astype(int)
        
        # Financial behavior stability score
        stability_components = []
        
        # Component 1: Payment consistency
        if 'payment_consistency' in df_new.columns:
            stability_components.append(1 / (1 + df_new['payment_consistency']))
        
        # Component 2: Low utilization
        if 'credit_utilization_ratio' in df_new.columns:
            stability_components.append(1 - np.clip(df_new['credit_utilization_ratio'], 0, 1))
        
        # Component 3: Responsible payment ratio
        if 'responsible_payment_ratio' in df_new.columns:
            stability_components.append(df_new['responsible_payment_ratio'])
        
        # Component 4: Low delinquency
        if 'delinquency_count' in df_new.columns:
            stability_components.append(1 - np.clip(df_new['delinquency_count'] / 7, 0, 1))
        
        if stability_components:
            df_new['financial_stability_score'] = np.mean(stability_components, axis=0)
            df_new['high_stability_flag'] = (df_new['financial_stability_score'] > 0.7).astype(int)
        
        # Risk concentration score
        risk_components = []
        
        if 'high_utilization_flag' in df_new.columns:
            risk_components.append(df_new['high_utilization_flag'])
        if 'chronic_underpayment_flag' in df_new.columns:
            risk_components.append(df_new['chronic_underpayment_flag'])
        if 'high_delinquency' in df_new.columns:
            risk_components.append(df_new['high_delinquency'])
        if 'worsening_delinquency' in df_new.columns:
            risk_components.append(df_new['worsening_delinquency'])
        
        if risk_components:
            df_new['risk_concentration_score'] = np.sum(risk_components, axis=0)
            df_new['high_risk_concentration'] = (df_new['risk_concentration_score'] >= 3).astype(int)
        
        return df_new
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal trend and seasonality features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        df_new = df.copy()
        
        # Bill amount trends
        bill_cols = [f'Bill_amt{i}' for i in range(1, 7) if f'Bill_amt{i}' in df.columns]
        if len(bill_cols) >= 4:
            # Linear trend in bill amounts
            def calculate_trend(row):
                x = np.arange(len(row))
                y = row.values
                if np.std(y) > 1e-8:  # Avoid division by zero
                    slope = np.polyfit(x, y, 1)[0]
                    return slope
                return 0
            
            df_new['bill_amount_trend'] = df[bill_cols].apply(calculate_trend, axis=1)
            df_new['increasing_bills'] = (df_new['bill_amount_trend'] > 1000).astype(int)
            
            # Bill amount acceleration (second derivative)
            recent_bills = df[bill_cols[-3:]].mean(axis=1)
            middle_bills = df[bill_cols[1:4]].mean(axis=1)
            early_bills = df[bill_cols[:3]].mean(axis=1)
            
            df_new['bill_acceleration'] = recent_bills - 2*middle_bills + early_bills
            df_new['accelerating_debt'] = (df_new['bill_acceleration'] > 5000).astype(int)
        
        # Payment amount trends
        pay_cols = [f'pay_amt{i}' for i in range(1, 7) if f'pay_amt{i}' in df.columns]
        if len(pay_cols) >= 4:
            df_new['payment_trend'] = df[pay_cols].apply(lambda row: np.polyfit(range(len(row)), row, 1)[0], axis=1)
            df_new['declining_payments'] = (df_new['payment_trend'] < -500).astype(int)
        
        return df_new
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different aspects.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        df_new = df.copy()
        
        # Age and credit utilization interaction
        if 'age' in df.columns and 'credit_utilization_ratio' in df_new.columns:
            df_new['young_high_utilization'] = ((df_new['age'] < 30) & 
                                               (df_new['credit_utilization_ratio'] > 0.8)).astype(int)
        
        # Education and delinquency interaction
        if 'education' in df.columns and 'delinquency_count' in df_new.columns:
            # Note: This assumes education is still numerical. Adjust if categorical encoded.
            df_new['educated_delinquent'] = ((df['education'] <= 2) & 
                                           (df_new['delinquency_count'] >= 2)).astype(int)
        
        # Credit limit and payment behavior
        if 'LIMIT_BAL' in df.columns and 'PAY_TO_BILL_ratio' in df.columns:
            df_new['high_limit_poor_payment'] = ((df_new['LIMIT_BAL'] > 200000) & 
                                                (df_new['PAY_TO_BILL_ratio'] < 0.5)).astype(int)
        
        # Marriage and financial stability
        if 'marriage' in df.columns and 'financial_stability_score' in df_new.columns:
            df_new['single_unstable'] = ((df['marriage'] == 2) & 
                                        (df_new['financial_stability_score'] < 0.5)).astype(int)
        
        return df_new
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (DataFrame with engineered features, feature descriptions)
        """
        print("Starting feature engineering...")
        
        df_features = df.copy()
        
        # Apply all feature engineering steps
        print("Creating credit utilization features...")
        df_features = self.create_credit_utilization_features(df_features)
        
        print("Creating payment behavior features...")
        df_features = self.create_payment_behavior_features(df_features)
        
        print("Creating delinquency features...")
        df_features = self.create_delinquency_features(df_features)
        
        print("Creating financial stability features...")
        df_features = self.create_financial_stability_features(df_features)
        
        print("Creating temporal features...")
        df_features = self.create_temporal_features(df_features)
        
        print("Creating interaction features...")
        df_features = self.create_interaction_features(df_features)
        
        # Document new features
        new_features = [col for col in df_features.columns if col not in df.columns]
        
        feature_descriptions = {
            'credit_utilization_ratio': 'Average bill amount divided by credit limit',
            'high_utilization_flag': 'Flag for credit utilization > 80%',
            'credit_headroom': 'Available credit (limit - average bill)',
            'avg_payment_gap': 'Average difference between bills and payments',
            'delinquency_count': 'Number of months with payment delays',
            'longest_delinquency_streak': 'Maximum consecutive months of payment delays',
            'financial_stability_score': 'Composite score of payment reliability (0-1)',
            'risk_concentration_score': 'Count of high-risk behaviors',
            'bill_amount_trend': 'Linear trend in bill amounts over time',
            'payment_trend': 'Linear trend in payment amounts over time',
            'zero_payment_months': 'Number of months with zero payments',
            'chronic_underpayment_flag': 'Flag for consistent underpayment pattern'
        }
        
        print(f"Feature engineering completed. Created {len(new_features)} new features.")
        print(f"Total features: {df.shape[1]} -> {df_features.shape[1]}")
        
        return df_features, feature_descriptions
    
    def select_features(self, df: pd.DataFrame, target_col: str = 'next_month_default', 
                       method: str = 'correlation') -> List[str]:
        """
        Select most important features based on different criteria.
        
        Args:
            df: Input DataFrame with engineered features
            target_col: Target column name
            method: Selection method ('correlation', 'variance', 'all')
            
        Returns:
            List of selected feature names
        """
        feature_cols = [col for col in df.columns if col not in ['Customer_ID', target_col]]
        
        if method == 'all':
            return feature_cols
        
        if method == 'correlation' and target_col in df.columns:
            # Select features based on correlation with target
            correlations = df[feature_cols].corrwith(df[target_col]).abs()
            top_features = correlations.nlargest(50).index.tolist()
            print(f"Selected top 50 features by correlation with target")
            return top_features
        
        if method == 'variance':
            # Remove low variance features
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(df[feature_cols])
            selected_features = [feature_cols[i] for i in range(len(feature_cols)) if selector.get_support()[i]]
            print(f"Selected {len(selected_features)} features after variance filtering")
            return selected_features
        
        return feature_cols


def engineer_features_for_dataset(df: pd.DataFrame, target_col: str = 'next_month_default') -> Tuple[pd.DataFrame, List[str], Dict[str, str]]:
    """
    Convenience function to engineer features for a dataset.
    
    Args:
        df: Input DataFrame
        target_col: Target column name
        
    Returns:
        Tuple of (engineered DataFrame, selected features, feature descriptions)
    """
    engineer = FeatureEngineer()
    
    # Engineer features
    df_engineered, descriptions = engineer.engineer_features(df)
    
    # Select features
    selected_features = engineer.select_features(df_engineered, target_col, method='correlation')
    
    return df_engineered, selected_features, descriptions


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_preprocessing import load_and_preprocess_data
    
    # Load preprocessed data
    train_data, _, _ = load_and_preprocess_data("../data/train.csv")
    
    # Engineer features
    train_engineered, selected_features, descriptions = engineer_features_for_dataset(train_data)
    
    print(f"\nFeature Engineering Summary:")
    print(f"Original features: {train_data.shape[1]}")
    print(f"Engineered features: {train_engineered.shape[1]}")
    print(f"Selected features: {len(selected_features)}")
    print(f"\nTop 10 selected features:")
    for i, feature in enumerate(selected_features[:10]):
        desc = descriptions.get(feature, "Original feature")
        print(f"{i+1:2d}. {feature}: {desc}")
