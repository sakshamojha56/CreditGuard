"""
Model training pipeline for credit card default prediction.

This module handles:
- Model selection and training
- Hyperparameter tuning
- Cross-validation
- Class imbalance handling
- Model comparison and selection
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
import warnings
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, fbeta_score
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

warnings.filterwarnings('ignore')

class ModelTrainer:
    """Main class for model training and evaluation."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.scalers = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'next_month_default', 
                    feature_cols: Optional[List[str]] = None, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting and scaling.
        
        Args:
            df: Input DataFrame
            target_col: Target column name
            feature_cols: List of feature columns to use
            test_size: Proportion for test split
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Select features
        if feature_cols is None:
            feature_cols = [col for col in df.columns if col not in ['Customer_ID', target_col]]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y
        )
        
        print(f"Data prepared: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Default rate - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_test, y_train, y_test
    
    def define_models(self) -> Dict[str, Any]:
        """
        Define different models to train and compare.
        
        Returns:
            Dictionary of model configurations
        """
        models = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=self.random_state, max_iter=1000),
                'params': {
                    'classifier__C': [0.1, 1, 10, 100],
                    'classifier__penalty': ['l1', 'l2'],
                    'classifier__solver': ['liblinear']
                },
                'scaling': True
            },
            
            'decision_tree': {
                'model': DecisionTreeClassifier(random_state=self.random_state),
                'params': {
                    'classifier__max_depth': [5, 10, 15, 20],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__criterion': ['gini', 'entropy']
                },
                'scaling': False
            },
            
            'random_forest': {
                'model': RandomForestClassifier(random_state=self.random_state, n_jobs=-1),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [10, 15, 20, None],
                    'classifier__min_samples_split': [2, 5],
                    'classifier__min_samples_leaf': [1, 2],
                    'classifier__max_features': ['sqrt', 'log2']
                },
                'scaling': False
            },
            
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=self.random_state, eval_metric='logloss'),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__subsample': [0.8, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                    'classifier__scale_pos_weight': [1, 2, 3]  # For class imbalance
                },
                'scaling': False
            },
            
            'lightgbm': {
                'model': lgb.LGBMClassifier(random_state=self.random_state, verbosity=-1),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 5, 7],
                    'classifier__learning_rate': [0.05, 0.1, 0.2],
                    'classifier__num_leaves': [31, 50, 100],
                    'classifier__subsample': [0.8, 0.9, 1.0],
                    'classifier__colsample_bytree': [0.8, 0.9, 1.0],
                    'classifier__class_weight': ['balanced']
                },
                'scaling': False
            }
        }
        
        return models
    
    def create_pipeline(self, model: Any, scaling: bool = False, 
                       sampling_strategy: str = 'smote') -> ImbPipeline:
        """
        Create a pipeline with preprocessing, sampling, and classification.
        
        Args:
            model: The classifier model
            scaling: Whether to include feature scaling
            sampling_strategy: Sampling strategy for imbalanced data
            
        Returns:
            Pipeline object
        """
        steps = []
        
        # Add scaling if needed
        if scaling:
            steps.append(('scaler', RobustScaler()))
        
        # Add sampling strategy
        if sampling_strategy == 'smote':
            steps.append(('sampler', SMOTE(random_state=self.random_state)))
        elif sampling_strategy == 'smote_tomek':
            steps.append(('sampler', SMOTETomek(random_state=self.random_state)))
        elif sampling_strategy == 'undersample':
            steps.append(('sampler', RandomUnderSampler(random_state=self.random_state)))
        
        # Add classifier
        steps.append(('classifier', model))
        
        return ImbPipeline(steps)
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray,
                          X_test: np.ndarray, y_test: np.ndarray, 
                          param_search: bool = True) -> Dict[str, Any]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            X_test: Test features  
            y_test: Test target
            param_search: Whether to perform hyperparameter search
            
        Returns:
            Dictionary with model results
        """
        print(f"\nTraining {model_name}...")
        
        models_config = self.define_models()
        model_config = models_config[model_name]
        
        # Create pipeline
        pipeline = self.create_pipeline(
            model_config['model'], 
            scaling=model_config['scaling']
        )
        
        if param_search:
            # Hyperparameter tuning with cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Use RandomizedSearchCV for faster searching
            search = RandomizedSearchCV(
                pipeline,
                model_config['params'],
                cv=cv,
                n_iter=20,  # Reduced for faster execution
                scoring='f1',  # Can also use fbeta_score with beta=2
                n_jobs=-1,
                random_state=self.random_state
            )
            
            search.fit(X_train, y_train)
            best_pipeline = search.best_estimator_
            best_params = search.best_params_
            cv_score = search.best_score_
            
        else:
            # Train with default parameters
            pipeline.fit(X_train, y_train)
            best_pipeline = pipeline
            best_params = {}
            cv_score = 0
        
        # Make predictions
        y_pred = best_pipeline.predict(X_test)
        y_pred_proba = best_pipeline.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'f2': fbeta_score(y_test, y_pred, beta=2),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'cv_score': cv_score
        }
        
        results = {
            'model': best_pipeline,
            'params': best_params,
            'metrics': metrics,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        print(f"{model_name} - F2 Score: {metrics['f2']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return results
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all defined models and compare performance.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with results for all models
        """
        print("Training multiple models for comparison...")
        
        models_config = self.define_models()
        all_results = {}
        
        for model_name in models_config.keys():
            try:
                results = self.train_single_model(
                    model_name, X_train, y_train, X_test, y_test
                )
                all_results[model_name] = results
                
                # Track best model based on F2 score
                f2_score = results['metrics']['f2']
                if f2_score > self.best_score:
                    self.best_score = f2_score
                    self.best_model = results['model']
                    self.best_model_name = model_name
                    
            except Exception as e:
                print(f"Error training {model_name}: {str(e)}")
                continue
        
        # Print comparison
        self._print_model_comparison(all_results)
        
        return all_results
    
    def _print_model_comparison(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print a comparison table of model performances."""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'F2':<10} {'ROC-AUC':<10}")
        print("-"*80)
        
        for model_name, result in results.items():
            metrics = result['metrics']
            print(f"{model_name:<20} "
                  f"{metrics['accuracy']:<10.4f} "
                  f"{metrics['precision']:<10.4f} "
                  f"{metrics['recall']:<10.4f} "
                  f"{metrics['f1']:<10.4f} "
                  f"{metrics['f2']:<10.4f} "
                  f"{metrics['roc_auc']:<10.4f}")
        
        print("-"*80)
        print(f"Best Model: {self.best_model_name} (F2 Score: {self.best_score:.4f})")
        print("="*80)
    
    def optimize_threshold(self, model: Any, X_val: np.ndarray, y_val: np.ndarray, 
                          metric: str = 'f2') -> Tuple[float, Dict[str, float]]:
        """
        Optimize classification threshold for business impact.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            metric: Metric to optimize ('f1', 'f2', 'recall', 'precision')
            
        Returns:
            Tuple of (optimal threshold, metrics at optimal threshold)
        """
        print(f"\nOptimizing threshold for {metric} score...")
        
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Test different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_metric_score = 0
        
        threshold_results = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            try:
                metrics = {
                    'threshold': threshold,
                    'accuracy': accuracy_score(y_val, y_pred_thresh),
                    'precision': precision_score(y_val, y_pred_thresh, zero_division=0),
                    'recall': recall_score(y_val, y_pred_thresh, zero_division=0),
                    'f1': f1_score(y_val, y_pred_thresh, zero_division=0),
                    'f2': fbeta_score(y_val, y_pred_thresh, beta=2, zero_division=0)
                }
                
                threshold_results.append(metrics)
                
                # Check if this is the best threshold
                if metrics[metric] > best_metric_score:
                    best_metric_score = metrics[metric]
                    best_threshold = threshold
                    
            except Exception as e:
                continue
        
        # Get metrics at best threshold
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)
        optimal_metrics = {
            'accuracy': accuracy_score(y_val, y_pred_optimal),
            'precision': precision_score(y_val, y_pred_optimal, zero_division=0),
            'recall': recall_score(y_val, y_pred_optimal, zero_division=0),
            'f1': f1_score(y_val, y_pred_optimal, zero_division=0),
            'f2': fbeta_score(y_val, y_pred_optimal, beta=2, zero_division=0)
        }
        
        print(f"Optimal threshold: {best_threshold:.3f}")
        print(f"Optimized {metric} score: {best_metric_score:.4f}")
        
        return best_threshold, optimal_metrics
    
    def save_model(self, model: Any, model_name: str, save_path: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model to save
            model_name: Name for the saved model
            save_path: Directory to save the model
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        model_file = os.path.join(save_path, f"{model_name}.joblib")
        joblib.dump(model, model_file)
        print(f"Model saved: {model_file}")
    
    def load_model(self, model_path: str) -> Any:
        """
        Load saved model from disk.
        
        Args:
            model_path: Path to the saved model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        print(f"Model loaded: {model_path}")
        return model


def train_models_pipeline(df: pd.DataFrame, feature_cols: List[str], 
                         target_col: str = 'next_month_default') -> Tuple[Any, float, Dict[str, Any]]:
    """
    Complete model training pipeline.
    
    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name
        
    Returns:
        Tuple of (best model, optimal threshold, training results)
    """
    trainer = ModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, target_col, feature_cols)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Optimize threshold for best model
    optimal_threshold, optimal_metrics = trainer.optimize_threshold(
        trainer.best_model, X_test, y_test, metric='f2'
    )
    
    # Save best model
    trainer.save_model(trainer.best_model, trainer.best_model_name, "../models")
    
    training_summary = {
        'all_results': results,
        'best_model_name': trainer.best_model_name,
        'best_f2_score': trainer.best_score,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics': optimal_metrics
    }
    
    return trainer.best_model, optimal_threshold, training_summary


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.append('../src')
    from data_preprocessing import load_and_preprocess_data
    from feature_engineering import engineer_features_for_dataset
    
    print("Loading and preprocessing data...")
    train_data, _, _ = load_and_preprocess_data("../data/train.csv")
    
    print("Engineering features...")
    train_engineered, selected_features, _ = engineer_features_for_dataset(train_data)
    
    print("Training models...")
    best_model, threshold, summary = train_models_pipeline(
        train_engineered, selected_features
    )
    
    print(f"\nTraining completed!")
    print(f"Best model: {summary['best_model_name']}")
    print(f"Best F2 score: {summary['best_f2_score']:.4f}")
    print(f"Optimal threshold: {threshold:.3f}")
