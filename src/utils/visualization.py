"""
Visualization utilities for training progress and model evaluation.

This module provides tools for visualizing training curves, model
predictions, confusion matrices, and other evaluation metrics.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import torch


class TrainingVisualizer:
    """
    Visualization tools for training progress and results.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer with matplotlib style.
        
        Args:
            style: Matplotlib style to use
        """
        try:
            plt.style.use(style)
        except OSError:
            plt.style.use('default')
        
        # Set color palette
        self.colors = plt.cm.Set1(np.linspace(0, 1, 10))
    
    def plot_training_curves(self, 
                            history: Dict[str, List[float]],
                            save_path: Optional[str] = None,
                            figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot training and validation curves.
        
        Args:
            history: Training history with loss and accuracy curves
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot Loss curves
        axes[0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot Accuracy curves
        axes[1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot Learning Rate
        if 'learning_rate' in history:
            axes[2].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
            axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_examples(self,
                                model: torch.nn.Module,
                                dataloader: torch.utils.data.DataLoader,
                                class_names: List[str],
                                num_examples: int = 16,
                                device: str = 'cuda',
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (16, 16)) -> plt.Figure:
        """
        Plot prediction examples with true and predicted labels.
        
        Args:
            model: Trained model
            dataloader: Data loader for examples
            class_names: List of class names
            num_examples: Number of examples to show
            device: Device for inference
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        model.eval()
        model.to(device)
        
        # Get batch of data
        images, labels = next(iter(dataloader))
        images = images.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
        
        # Convert to numpy for plotting
        images = images.cpu()
        labels = labels.cpu().numpy()
        predicted = predicted.cpu().numpy()
        probabilities = probabilities.cpu().numpy()
        
        # Create grid plot
        cols = 4
        rows = (num_examples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_examples, len(images))):
            row = i // cols
            col = i % cols
            
            # Denormalize image for display
            img = images[i]
            img = img * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = img + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            img = torch.clamp(img, 0, 1)
            
            # Convert to HWC format
            img = img.permute(1, 2, 0)
            
            axes[row, col].imshow(img)
            
            true_label = class_names[labels[i]]
            pred_label = class_names[predicted[i]]
            confidence = probabilities[i][predicted[i]] * 100
            
            # Color code: green for correct, red for incorrect
            color = 'green' if labels[i] == predicted[i] else 'red'
            
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            axes[row, col].set_title(title, fontsize=10, color=color, fontweight='bold')
            axes[row, col].axis('off')
        
        # Hide unused subplots
        for i in range(num_examples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Model Predictions on Sample Images', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_class_distribution(self,
                               class_distribution: Dict[str, int],
                               title: str = "Class Distribution",
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot class distribution as a bar chart.
        
        Args:
            class_distribution: Dictionary mapping class names to counts
            title: Plot title
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        classes = list(class_distribution.keys())
        counts = list(class_distribution.values())
        
        bars = ax.bar(classes, counts, color=self.colors[:len(classes)], alpha=0.8)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Classes', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if too many classes
        if len(classes) > 6:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_training_metrics_comparison(self,
                                        metrics_history: Dict[str, Dict[str, List[float]]],
                                        save_path: Optional[str] = None,
                                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Compare training metrics across multiple experiments.
        
        Args:
            metrics_history: Dictionary of experiment names to their training history
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        metrics = ['train_loss', 'val_loss', 'train_acc', 'val_acc']
        titles = ['Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            for exp_name, history in metrics_history.items():
                if metric in history:
                    epochs = range(1, len(history[metric]) + 1)
                    axes[idx].plot(epochs, history[metric], label=exp_name, linewidth=2)
            
            axes[idx].set_title(title, fontsize=14, fontweight='bold')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss' if 'loss' in metric else 'Accuracy (%)')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_architecture_summary(self,
                                       model_info: Dict[str, Any],
                                       save_path: Optional[str] = None,
                                       figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Create a summary visualization of model architecture.
        
        Args:
            model_info: Dictionary containing model information
            save_path: Optional path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Create text summary
        summary_text = f"""
        Model Architecture Summary
        {'='*50}
        
        Model Name: {model_info.get('model_name', 'Unknown')}
        Number of Classes: {model_info.get('num_classes', 'Unknown')}
        Feature Dimension: {model_info.get('feature_dim', 'Unknown')}
        
        Parameters:
        - Total Parameters: {model_info.get('total_parameters', 0):,}
        - Trainable Parameters: {model_info.get('trainable_parameters', 0):,}
        - Frozen Parameters: {model_info.get('total_parameters', 0) - model_info.get('trainable_parameters', 0):,}
        
        Configuration:
        - Dropout Rate: {model_info.get('dropout', 'Unknown')}
        - Backbone Frozen: {model_info.get('frozen_backbone', 'Unknown')}
        """
        
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_evaluation_report(self,
                                metrics: Dict[str, Any],
                                confusion_matrix: np.ndarray,
                                class_names: List[str],
                                save_dir: str,
                                model_name: str = "model") -> Dict[str, str]:
        """
        Create a comprehensive evaluation report with multiple visualizations.
        
        Args:
            metrics: Calculated metrics dictionary
            confusion_matrix: Confusion matrix
            class_names: List of class names
            save_dir: Directory to save plots
            model_name: Name of the model for file naming
            
        Returns:
            Dictionary mapping plot types to saved file paths
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_plots = {}
        
        # 1. Confusion Matrix
        from .metrics import MetricsCalculator
        metrics_calc = MetricsCalculator(len(class_names), class_names)
        
        cm_path = save_dir / f"{model_name}_confusion_matrix.png"
        metrics_calc.plot_confusion_matrix(confusion_matrix, save_path=str(cm_path))
        saved_plots['confusion_matrix'] = str(cm_path)
        
        # 2. Normalized Confusion Matrix
        cm_norm_path = save_dir / f"{model_name}_confusion_matrix_normalized.png"
        metrics_calc.plot_confusion_matrix(confusion_matrix, normalize=True, save_path=str(cm_norm_path))
        saved_plots['confusion_matrix_normalized'] = str(cm_norm_path)
        
        # 3. Per-class metrics
        per_class_path = save_dir / f"{model_name}_per_class_metrics.png"
        metrics_calc.plot_per_class_metrics(metrics, save_path=str(per_class_path))
        saved_plots['per_class_metrics'] = str(per_class_path)
        
        # 4. Class distribution
        if 'class_distribution' in metrics:
            dist_path = save_dir / f"{model_name}_class_distribution.png"
            self.plot_class_distribution(
                metrics['class_distribution'], 
                title="True Label Distribution",
                save_path=str(dist_path)
            )
            saved_plots['class_distribution'] = str(dist_path)
        
        return saved_plots


def save_training_plots(history: Dict[str, List[float]],
                       save_dir: str,
                       model_name: str = "model"):
    """
    Convenience function to save all training plots.
    
    Args:
        history: Training history
        save_dir: Directory to save plots
        model_name: Model name for file naming
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = TrainingVisualizer()
    
    # Training curves
    curves_path = save_dir / f"{model_name}_training_curves.png"
    visualizer.plot_training_curves(history, save_path=str(curves_path))
    
    print(f"Training plots saved to: {save_dir}")