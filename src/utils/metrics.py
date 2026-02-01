"""
Metrics calculation utilities for model evaluation.

This module provides comprehensive metrics calculation including
accuracy, precision, recall, F1-score, confusion matrix, and 
classification reports for wafer defect detection.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class MetricsCalculator:
    """
    Comprehensive metrics calculator for classification tasks.
    """
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes in classification task
            class_names: Optional list of class names
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
    
    def calculate_metrics(self, 
                         true_labels: List[int], 
                         predicted_labels: List[int],
                         predicted_probabilities: Optional[List[List[float]]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive classification metrics.
        
        Args:
            true_labels: Ground truth labels
            predicted_labels: Predicted labels
            predicted_probabilities: Predicted class probabilities (optional)
            
        Returns:
            Dictionary containing all calculated metrics
        """
        true_labels = np.array(true_labels)
        predicted_labels = np.array(predicted_labels)
        
        metrics = {}
        
        # Basic accuracy
        metrics['accuracy'] = accuracy_score(true_labels, predicted_labels) * 100
        
        # Precision, Recall, F1-Score
        metrics['precision_macro'] = precision_score(true_labels, predicted_labels, average='macro', zero_division=0)
        metrics['recall_macro'] = recall_score(true_labels, predicted_labels, average='macro', zero_division=0)
        metrics['f1_score'] = f1_score(true_labels, predicted_labels, average='macro', zero_division=0)
        
        metrics['precision_micro'] = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
        metrics['recall_micro'] = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
        metrics['f1_micro'] = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
        
        metrics['precision_weighted'] = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(true_labels, predicted_labels, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predicted_labels, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predicted_labels, average=None, zero_division=0)
        
        metrics['per_class_precision'] = dict(zip(self.class_names, precision_per_class))
        metrics['per_class_recall'] = dict(zip(self.class_names, recall_per_class))
        metrics['per_class_f1'] = dict(zip(self.class_names, f1_per_class))
        
        # Confusion Matrix
        cm = confusion_matrix(true_labels, predicted_labels)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Classification Report
        report = classification_report(
            true_labels, predicted_labels, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Calculate additional metrics if probabilities are provided
        if predicted_probabilities is not None:
            predicted_probabilities = np.array(predicted_probabilities)
            
            # ROC AUC (for multi-class, use ovr strategy)
            try:
                if self.num_classes > 2:
                    metrics['roc_auc_ovr'] = roc_auc_score(
                        true_labels, predicted_probabilities, 
                        multi_class='ovr', average='macro'
                    )
                    metrics['roc_auc_ovo'] = roc_auc_score(
                        true_labels, predicted_probabilities, 
                        multi_class='ovo', average='macro'
                    )
                else:
                    metrics['roc_auc'] = roc_auc_score(true_labels, predicted_probabilities[:, 1])
            except ValueError:
                # Handle case where only one class is present
                metrics['roc_auc_ovr'] = 0.0
                metrics['roc_auc_ovo'] = 0.0
            
            # Average Precision Score
            try:
                if self.num_classes > 2:
                    metrics['avg_precision_macro'] = average_precision_score(
                        true_labels, predicted_probabilities, average='macro'
                    )
                else:
                    metrics['avg_precision'] = average_precision_score(
                        true_labels, predicted_probabilities[:, 1]
                    )
            except ValueError:
                metrics['avg_precision_macro'] = 0.0
        
        # Class distribution
        unique, counts = np.unique(true_labels, return_counts=True)
        metrics['class_distribution'] = dict(zip([self.class_names[i] for i in unique], counts.tolist()))
        
        # Prediction distribution  
        unique_pred, counts_pred = np.unique(predicted_labels, return_counts=True)
        metrics['prediction_distribution'] = dict(zip([self.class_names[i] for i in unique_pred], counts_pred.tolist()))
        
        return metrics
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """
        Print a formatted summary of metrics.
        
        Args:
            metrics: Dictionary of calculated metrics
        """
        print("\n" + "="*60)
        print("                    METRICS SUMMARY")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy:           {metrics['accuracy']:.2f}%")
        print(f"  Macro F1-Score:     {metrics['f1_score']:.4f}")
        print(f"  Macro Precision:    {metrics['precision_macro']:.4f}")
        print(f"  Macro Recall:       {metrics['recall_macro']:.4f}")
        
        print(f"\nWeighted Averages:")
        print(f"  Weighted F1-Score:  {metrics['f1_weighted']:.4f}")
        print(f"  Weighted Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Weighted Recall:    {metrics['recall_weighted']:.4f}")
        
        if 'roc_auc_ovr' in metrics:
            print(f"\nROC AUC Scores:")
            print(f"  One-vs-Rest:        {metrics['roc_auc_ovr']:.4f}")
            print(f"  One-vs-One:         {metrics['roc_auc_ovo']:.4f}")
        
        print(f"\nPer-Class Performance:")
        for i, class_name in enumerate(self.class_names):
            if class_name in metrics['per_class_f1']:
                f1 = metrics['per_class_f1'][class_name]
                precision = metrics['per_class_precision'][class_name]
                recall = metrics['per_class_recall'][class_name]
                print(f"  {class_name:15s} - F1: {f1:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        print(f"\nClass Distribution:")
        for class_name, count in metrics['class_distribution'].items():
            print(f"  {class_name:15s}: {count:4d} samples")
    
    def plot_confusion_matrix(self, 
                            confusion_matrix: np.ndarray,
                            title: str = "Confusion Matrix",
                            normalize: bool = False,
                            figsize: Tuple[int, int] = (10, 8),
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            confusion_matrix: Confusion matrix to plot
            title: Title for the plot
            normalize: Whether to normalize the matrix
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        if normalize:
            cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title += ' (Normalized)'
        else:
            cm = confusion_matrix
            fmt = 'd'
        
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            ax=ax
        )
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_per_class_metrics(self, 
                              metrics: Dict[str, Any],
                              metric_types: List[str] = ['f1', 'precision', 'recall'],
                              figsize: Tuple[int, int] = (12, 6),
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot per-class metrics comparison.
        
        Args:
            metrics: Dictionary of calculated metrics
            metric_types: Types of metrics to plot
            figsize: Figure size
            save_path: Optional path to save the plot
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.class_names))
        width = 0.25
        
        for i, metric_type in enumerate(metric_types):
            if metric_type == 'f1':
                values = [metrics['per_class_f1'].get(name, 0) for name in self.class_names]
                label = 'F1-Score'
            elif metric_type == 'precision':
                values = [metrics['per_class_precision'].get(name, 0) for name in self.class_names]
                label = 'Precision'
            elif metric_type == 'recall':
                values = [metrics['per_class_recall'].get(name, 0) for name in self.class_names]
                label = 'Recall'
            else:
                continue
            
            ax.bar(x + i * width, values, width, label=label, alpha=0.8)
        
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(self.class_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


def calculate_classification_metrics(true_labels: List[int],
                                   predicted_labels: List[int],
                                   predicted_probabilities: Optional[List[List[float]]] = None,
                                   num_classes: Optional[int] = None,
                                   class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to calculate classification metrics.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels  
        predicted_probabilities: Predicted class probabilities (optional)
        num_classes: Number of classes (inferred if None)
        class_names: Class names (auto-generated if None)
        
    Returns:
        Dictionary containing calculated metrics
    """
    if num_classes is None:
        num_classes = max(max(true_labels), max(predicted_labels)) + 1
    
    calculator = MetricsCalculator(num_classes, class_names)
    return calculator.calculate_metrics(true_labels, predicted_labels, predicted_probabilities)