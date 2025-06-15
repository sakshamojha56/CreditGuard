"""
Model evaluation and explainability module for credit card default prediction.

This module provides:
- Comprehensive model evaluation metrics
- Visualizations for model performance
- SHAP and LIME explanations
- Business impact analysis
- Report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import warnings
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc, 
    precision_recall_curve, fbeta_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import shap
import lime
import lime.lime_tabular

warnings.filterwarnings('ignore')

class ModelEvaluator:
    """Main class for model evaluation and explainability."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.feature_importance = {}
        self.shap_explainer = None
        self.lime_explainer = None
    
    def calculate_comprehensive_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                      y_pred_proba: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'f2_score': fbeta_score(y_true, y_pred, beta=2, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba),
        }
        
        # Additional business metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'negative_predictive_value': tn / (tn + fn) if (tn + fn) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
        })
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix") -> plt.Figure:
        """
        Plot confusion matrix with annotations.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xticklabels(['No Default', 'Default'])
        ax.set_yticklabels(['No Default', 'Default'])
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / total * 100
                ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                       ha='center', va='center', fontsize=10, color='gray')
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str = "ROC Curve") -> plt.Figure:
        """
        Plot ROC curve with AUC score.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   title: str = "Precision-Recall Curve") -> plt.Figure:
        """
        Plot Precision-Recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        ax.plot(recall, precision, color='blue', lw=2,
               label=f'PR Curve (AUC = {pr_auc:.3f})')
        
        # Baseline (random classifier)
        baseline = np.mean(y_true)
        ax.axhline(y=baseline, color='red', linestyle='--', 
                  label=f'Baseline (Random) = {baseline:.3f}')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance(self, model: Any, feature_names: List[str], 
                              top_n: int = 20) -> plt.Figure:
        """
        Plot feature importance from tree-based models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'named_steps') and hasattr(model.named_steps['classifier'], 'feature_importances_'):
            importances = model.named_steps['classifier'].feature_importances_
        else:
            print("Model does not have feature_importances_ attribute")
            return fig
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        sns.barplot(data=importance_df, y='feature', x='importance', ax=ax, palette='viridis')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Features', fontsize=12)
        
        plt.tight_layout()
        return fig
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> plt.Figure:
        """
        Plot metrics vs threshold to find optimal threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        thresholds = np.arange(0.1, 0.9, 0.02)
        metrics_over_threshold = {
            'threshold': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'f2': []
        }
        
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            metrics_over_threshold['threshold'].append(threshold)
            metrics_over_threshold['precision'].append(precision_score(y_true, y_pred, zero_division=0))
            metrics_over_threshold['recall'].append(recall_score(y_true, y_pred, zero_division=0))
            metrics_over_threshold['f1'].append(f1_score(y_true, y_pred, zero_division=0))
            metrics_over_threshold['f2'].append(fbeta_score(y_true, y_pred, beta=2, zero_division=0))
        
        # Plot metrics
        ax.plot(metrics_over_threshold['threshold'], metrics_over_threshold['precision'], 
               label='Precision', linewidth=2)
        ax.plot(metrics_over_threshold['threshold'], metrics_over_threshold['recall'], 
               label='Recall', linewidth=2)
        ax.plot(metrics_over_threshold['threshold'], metrics_over_threshold['f1'], 
               label='F1 Score', linewidth=2)
        ax.plot(metrics_over_threshold['threshold'], metrics_over_threshold['f2'], 
               label='F2 Score', linewidth=2, linestyle='--')
        
        # Find and mark optimal F2 threshold
        f2_scores = metrics_over_threshold['f2']
        optimal_idx = np.argmax(f2_scores)
        optimal_threshold = metrics_over_threshold['threshold'][optimal_idx]
        optimal_f2 = f2_scores[optimal_idx]
        
        ax.axvline(x=optimal_threshold, color='red', linestyle=':', alpha=0.7,
                  label=f'Optimal F2 Threshold ({optimal_threshold:.3f})')
        ax.scatter([optimal_threshold], [optimal_f2], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Classification Threshold', fontsize=12)
        ax.set_ylabel('Metric Score', fontsize=12)
        ax.set_title('Metrics vs Classification Threshold', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        return fig
    
    def create_shap_explanations(self, model: Any, X_train: pd.DataFrame, 
                                X_test: pd.DataFrame, max_samples: int = 1000) -> Dict[str, Any]:
        """
        Create SHAP explanations for model interpretability.
        
        Args:
            model: Trained model
            X_train: Training features for background
            X_test: Test features to explain
            max_samples: Maximum samples for SHAP background
            
        Returns:
            Dictionary with SHAP values and plots
        """
        print("Creating SHAP explanations...")
        
        # Sample data for faster computation
        if len(X_train) > max_samples:
            sample_idx = np.random.choice(len(X_train), max_samples, replace=False)
            X_background = X_train.iloc[sample_idx]
        else:
            X_background = X_train
        
        if len(X_test) > max_samples:
            sample_idx = np.random.choice(len(X_test), max_samples, replace=False)
            X_explain = X_test.iloc[sample_idx]
        else:
            X_explain = X_test
        
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                self.shap_explainer = shap.Explainer(model.predict_proba, X_background)
            else:
                self.shap_explainer = shap.Explainer(model, X_background)
            
            # Calculate SHAP values
            shap_values = self.shap_explainer(X_explain)
            
            # If binary classification, take values for positive class
            if len(shap_values.shape) == 3:
                shap_values_pos = shap_values[:, :, 1]
            else:
                shap_values_pos = shap_values
            
            shap_results = {
                'shap_values': shap_values_pos,
                'explainer': self.shap_explainer,
                'X_explain': X_explain
            }
            
            print("SHAP explanations created successfully")
            return shap_results
            
        except Exception as e:
            print(f"Error creating SHAP explanations: {str(e)}")
            return {}
    
    def plot_shap_summary(self, shap_results: Dict[str, Any], top_n: int = 20) -> plt.Figure:
        """
        Create SHAP summary plot.
        
        Args:
            shap_results: Results from create_shap_explanations
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
        """
        if not shap_results:
            return plt.figure()
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_results['shap_values'], 
            shap_results['X_explain'],
            max_display=top_n,
            show=False
        )
        plt.title(f'SHAP Summary Plot - Top {top_n} Features', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return plt.gcf()
    
    def create_business_impact_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                      cost_fn: float = 100, cost_fp: float = 10) -> Dict[str, Any]:
        """
        Analyze business impact of model predictions.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            cost_fn: Cost of false negative (missing a default)
            cost_fp: Cost of false positive (wrong default prediction)
            
        Returns:
            Dictionary with business impact metrics
        """
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Calculate costs
        total_cost = (fn * cost_fn) + (fp * cost_fp)
        cost_per_sample = total_cost / len(y_true)
        
        # Calculate savings compared to naive strategies
        baseline_all_positive_cost = len(y_true) * cost_fp  # Predict all as default
        baseline_all_negative_cost = np.sum(y_true) * cost_fn  # Predict none as default
        
        savings_vs_all_positive = baseline_all_positive_cost - total_cost
        savings_vs_all_negative = baseline_all_negative_cost - total_cost
        
        business_metrics = {
            'total_cost': total_cost,
            'cost_per_sample': cost_per_sample,
            'cost_false_negatives': fn * cost_fn,
            'cost_false_positives': fp * cost_fp,
            'savings_vs_predict_all_default': savings_vs_all_positive,
            'savings_vs_predict_no_default': savings_vs_all_negative,
            'defaults_caught': tp,
            'defaults_missed': fn,
            'false_alarms': fp,
            'correct_no_default': tn
        }
        
        return business_metrics
    
    def generate_evaluation_report(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray,
                                 feature_names: List[str], model_name: str = "Model") -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            feature_names: List of feature names
            model_name: Name of the model
            
        Returns:
            Dictionary with complete evaluation results
        """
        print(f"Generating evaluation report for {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(y_test, y_pred, y_pred_proba)
        
        # Business impact
        business_impact = self.create_business_impact_analysis(y_test, y_pred)
        
        # Create visualizations
        plots = {}
        plots['confusion_matrix'] = self.plot_confusion_matrix(y_test, y_pred, 
                                                              f"{model_name} - Confusion Matrix")
        plots['roc_curve'] = self.plot_roc_curve(y_test, y_pred_proba, 
                                                 f"{model_name} - ROC Curve")
        plots['pr_curve'] = self.plot_precision_recall_curve(y_test, y_pred_proba,
                                                            f"{model_name} - Precision-Recall Curve")
        plots['threshold_analysis'] = self.plot_threshold_analysis(y_test, y_pred_proba)
        
        # Feature importance (if available)
        try:
            plots['feature_importance'] = self.plot_feature_importance(model, feature_names)
        except:
            plots['feature_importance'] = None
        
        evaluation_report = {
            'model_name': model_name,
            'metrics': metrics,
            'business_impact': business_impact,
            'plots': plots,
            'predictions': {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
        }
        
        # Print summary
        self._print_evaluation_summary(evaluation_report)
        
        return evaluation_report
    
    def _print_evaluation_summary(self, report: Dict[str, Any]) -> None:
        """Print evaluation summary."""
        print("\n" + "="*60)
        print(f"EVALUATION SUMMARY - {report['model_name']}")
        print("="*60)
        
        metrics = report['metrics']
        print(f"Accuracy:     {metrics['accuracy']:.4f}")
        print(f"Precision:    {metrics['precision']:.4f}")
        print(f"Recall:       {metrics['recall']:.4f}")
        print(f"F1 Score:     {metrics['f1_score']:.4f}")
        print(f"F2 Score:     {metrics['f2_score']:.4f}")
        print(f"ROC-AUC:      {metrics['roc_auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(f"True Positives:  {metrics['true_positives']:,}")
        print(f"True Negatives:  {metrics['true_negatives']:,}")
        print(f"False Positives: {metrics['false_positives']:,}")
        print(f"False Negatives: {metrics['false_negatives']:,}")
        
        business = report['business_impact']
        print(f"\nBusiness Impact:")
        print(f"Defaults Caught: {business['defaults_caught']:,}")
        print(f"Defaults Missed: {business['defaults_missed']:,}")
        print(f"False Alarms:    {business['false_alarms']:,}")
        print(f"Total Cost:      ${business['total_cost']:,.2f}")
        print("="*60)


def evaluate_model_comprehensive(model: Any, X_test: pd.DataFrame, y_test: np.ndarray,
                               feature_names: List[str], model_name: str = "Model") -> Dict[str, Any]:
    """
    Convenience function for comprehensive model evaluation.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        feature_names: List of feature names
        model_name: Name of the model
        
    Returns:
        Complete evaluation report
    """
    evaluator = ModelEvaluator()
    return evaluator.generate_evaluation_report(model, X_test, y_test, feature_names, model_name)


if __name__ == "__main__":
    # Example usage
    print("Model evaluation module ready for use.")
    print("Use evaluate_model_comprehensive() for complete evaluation.")
