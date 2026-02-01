"""
Training utilities and pipeline for wafer defect classification.

This module provides comprehensive training functionality including
loss functions, optimizers, schedulers, and the main training loop
with logging and checkpointing capabilities.
"""

import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

from ..utils.metrics import MetricsCalculator
from ..utils.visualization import TrainingVisualizer


class EarlyStopping:
    """Early stopping to stop training when validation loss stops improving."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum improvement required to reset patience
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        
        self.best_loss = float('inf')
        self.best_weights = None
        self.wait = 0
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
            
        return False


class WaferDefectTrainer:
    """
    Comprehensive trainer for wafer defect classification models.
    
    Handles training loop, validation, checkpointing, logging, and early stopping.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: torch.utils.data.DataLoader,
                 val_loader: torch.utils.data.DataLoader,
                 test_loader: torch.utils.data.DataLoader,
                 config: Dict[str, Any],
                 device: str = 'cuda',
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader  
            config: Training configuration
            device: Device to train on
            log_dir: Directory for logging
            checkpoint_dir: Directory for checkpoints
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup directories
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.writer = SummaryWriter(self.log_dir)
        
        # Setup training components
        self.criterion = self._create_loss_function()
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.early_stopping = self._create_early_stopping()
        
        # Setup metrics and visualization
        self.metrics_calculator = MetricsCalculator(
            num_classes=model.num_classes,
            class_names=self._get_class_names()
        )
        self.visualizer = TrainingVisualizer()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': [],
            'epoch_times': []
        }
        
        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.start_epoch = 0
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config.get('loss', {})
        loss_type = loss_config.get('type', 'cross_entropy')
        
        if loss_type == 'cross_entropy':
            # Handle class imbalance with weights
            if loss_config.get('use_class_weights', False):
                # Calculate class weights from training data
                class_weights = self._calculate_class_weights()
                return nn.CrossEntropyLoss(weight=class_weights)
            else:
                return nn.CrossEntropyLoss()
                
        elif loss_type == 'focal':
            # Focal loss for handling class imbalance
            return FocalLoss(
                alpha=loss_config.get('focal_alpha', 1.0),
                gamma=loss_config.get('focal_gamma', 2.0)
            )
            
        elif loss_type == 'label_smoothing':
            # Label smoothing for regularization
            return nn.CrossEntropyLoss(
                label_smoothing=loss_config.get('smoothing', 0.1)
            )
        
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def _calculate_class_weights(self) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        # Get class counts from dataset
        if hasattr(self.train_loader.dataset, 'get_class_weights'):
            weights = self.train_loader.dataset.get_class_weights()
        else:
            # Fallback: uniform weights
            weights = torch.ones(self.model.num_classes)
        
        return weights.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        optim_config = self.config.get('optimizer', {})
        optim_type = optim_config.get('type', 'adam')
        lr = optim_config.get('learning_rate', 0.001)
        weight_decay = optim_config.get('weight_decay', 0.0001)
        
        if optim_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=optim_config.get('betas', (0.9, 0.999))
            )
        elif optim_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=optim_config.get('momentum', 0.9),
                nesterov=optim_config.get('nesterov', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optim_type}")
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'cosine')
        
        if scheduler_type == 'step':
            return StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100),
                eta_min=scheduler_config.get('eta_min', 0.0001)
            )
        elif scheduler_type == 'plateau':
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=scheduler_config.get('factor', 0.5),
                patience=scheduler_config.get('patience', 10),
                verbose=True
            )
        elif scheduler_type == 'none':
            return None
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    def _create_early_stopping(self) -> Optional[EarlyStopping]:
        """Create early stopping based on configuration."""
        early_stopping_config = self.config.get('early_stopping', {})
        
        if early_stopping_config.get('enabled', True):
            return EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=early_stopping_config.get('min_delta', 0.001),
                restore_best_weights=early_stopping_config.get('restore_best_weights', True)
            )
        
        return None
    
    def _get_class_names(self) -> List[str]:
        """Get class names from dataset."""
        if hasattr(self.train_loader.dataset, 'class_mapping'):
            # Reverse mapping from id to name
            id_to_name = {v: k for k, v in self.train_loader.dataset.class_mapping.items()}
            return [id_to_name.get(i, f"class_{i}") for i in range(self.model.num_classes)]
        else:
            return [f"class_{i}" for i in range(self.model.num_classes)]
    
    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct_predictions / total_predictions
            progress_bar.set_postfix({
                'Loss': f'{running_loss / (batch_idx + 1):.4f}',
                'Acc': f'{current_acc:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct_predictions / total_predictions
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self) -> Tuple[float, float, Dict[str, Any]]:
        """
        Validate for one epoch.
        
        Returns:
            Tuple of (average_loss, accuracy, detailed_metrics)
        """
        self.model.eval()
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            
            for images, labels in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * np.sum(np.array(all_predictions) == np.array(all_labels)) / len(all_labels)
        
        # Calculate detailed metrics
        detailed_metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions
        )
        
        return epoch_loss, epoch_acc, detailed_metrics
    
    def save_checkpoint(self, epoch: int, val_acc: float, val_loss: float, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch
            val_acc: Current validation accuracy
            val_loss: Current validation loss
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'current_val_acc': val_acc,
            'current_val_loss': val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save checkpoint for each epoch with detailed naming
        epoch_checkpoint_path = self.checkpoint_dir / f'epoch_{epoch+1:03d}_acc_{val_acc:.2f}_loss_{val_loss:.4f}.pth'
        torch.save(checkpoint, epoch_checkpoint_path)
        
        # Save latest checkpoint (for resuming)
        latest_checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_checkpoint_path = self.checkpoint_dir / 'best_checkpoint.pth'
            torch.save(checkpoint, best_checkpoint_path)
            print(f"✓ Best model saved: epoch_{epoch+1:03d}_acc_{val_acc:.2f}_loss_{val_loss:.4f}.pth")
        
        print(f"✓ Model saved: epoch_{epoch+1:03d}_acc_{val_acc:.2f}_loss_{val_loss:.4f}.pth")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', self.history)
        
        print(f"Checkpoint loaded. Resuming from epoch {self.start_epoch}")
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training history and final metrics
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Validation phase
            val_loss, val_acc, val_metrics = self.validate_epoch()
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Record metrics
            epoch_time = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            self.history['epoch_times'].append(epoch_time)
            
            # Tensorboard logging
            self.writer.add_scalar('Loss/Train', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
            self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
            self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
            
            # Log detailed validation metrics
            for metric_name, metric_value in val_metrics.items():
                if isinstance(metric_value, (int, float)):
                    self.writer.add_scalar(f'Validation/{metric_name}', metric_value, epoch)
            
            # Print detailed epoch results
            self.print_detailed_metrics(epoch, train_loss, train_acc, val_loss, val_acc, 
                                      val_metrics, epoch_time, current_lr)
            
            # Check for improvement
            is_best = False
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                is_best = True
                print(f"🎉 New best validation accuracy: {val_acc:.2f}%!")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if not is_best:  # Only print if not already printed above
                    print(f"🔥 New best validation loss: {val_loss:.6f}!")
            
            # Save checkpoint (every epoch or when specified)
            save_frequency = self.config.get('logging', {}).get('save_frequency', 1)
            save_every_epoch = self.config.get('logging', {}).get('save_every_epoch', True)
            
            if save_every_epoch or (epoch + 1) % save_frequency == 0 or is_best:
                self.save_checkpoint(epoch, val_acc, val_loss, is_best)
            
            # Early stopping
            if self.early_stopping is not None:
                if self.early_stopping(val_loss, self.model):
                    print(f"\nEarly stopping triggered after epoch {epoch + 1}")
                    break
        
        print(f"\nTraining completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        
        # Final evaluation on test set
        test_metrics = self.evaluate_test_set()
        
        self.writer.close()
        
        return {
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics
        }
    
    def evaluate_test_set(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.
        
        Returns:
            Test metrics
        """
        print("\nEvaluating on test set...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Testing"):
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate comprehensive test metrics
        test_metrics = self.metrics_calculator.calculate_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        print(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
        print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
        
        return test_metrics
    
    def print_detailed_metrics(self, epoch: int, train_loss: float, train_acc: float, 
                             val_loss: float, val_acc: float, val_metrics: Dict[str, Any], 
                             epoch_time: float, current_lr: float):
        """
        Print detailed training metrics to console.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            train_acc: Training accuracy
            val_loss: Validation loss
            val_acc: Validation accuracy
            val_metrics: Detailed validation metrics
            epoch_time: Time taken for epoch
            current_lr: Current learning rate
        """
        print("\n" + "="*80)
        print(f"EPOCH {epoch + 1} RESULTS")
        print("="*80)
        
        # Basic metrics
        print(f"Training   - Loss: {train_loss:.6f} | Accuracy: {train_acc:.2f}%")
        print(f"Validation - Loss: {val_loss:.6f} | Accuracy: {val_acc:.2f}%")
        
        # Performance info
        print(f"Learning Rate: {current_lr:.8f}")
        print(f"Epoch Time: {epoch_time:.2f}s")
        
        # Detailed validation metrics
        if self.config.get('logging', {}).get('print_detailed_metrics', True):
            print("\nDetailed Validation Metrics:")
            print("-" * 40)
            
            # Classification metrics
            if 'precision' in val_metrics:
                print(f"Precision: {val_metrics['precision']:.4f}")
            if 'recall' in val_metrics:
                print(f"Recall: {val_metrics['recall']:.4f}")
            if 'f1_score' in val_metrics:
                print(f"F1-Score: {val_metrics['f1_score']:.4f}")
            
            # Per-class metrics (if available)
            if 'per_class_precision' in val_metrics:
                print("\nPer-Class Precision:")
                for i, prec in enumerate(val_metrics['per_class_precision']):
                    class_name = self._get_class_names()[i] if i < len(self._get_class_names()) else f"Class_{i}"
                    print(f"  {class_name}: {prec:.4f}")
            
            if 'per_class_recall' in val_metrics:
                print("\nPer-Class Recall:")
                for i, recall in enumerate(val_metrics['per_class_recall']):
                    class_name = self._get_class_names()[i] if i < len(self._get_class_names()) else f"Class_{i}"
                    print(f"  {class_name}: {recall:.4f}")
        
        # Best metrics so far
        print(f"\nBest So Far:")
        print(f"  Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Best Validation Loss: {self.best_val_loss:.6f}")
        
        print("="*80)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Reference: Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
    Focal loss for dense object detection.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss