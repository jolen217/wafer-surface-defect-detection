"""
CNN models for wafer defect classification using transfer learning.

This module implements various CNN architectures with pretrained weights
from ImageNet for transfer learning on wafer defect detection tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Dict, Optional, Any
import timm


class WaferDefectClassifier(nn.Module):
    """
    Base classifier for wafer defect detection with configurable backbone.
    
    Supports various pretrained CNN architectures via transfer learning.
    """
    
    def __init__(self,
                 model_name: str = 'efficientnet_b0',
                 num_classes: int = 6,
                 pretrained: bool = True,
                 dropout: float = 0.2,
                 freeze_backbone: bool = False):
        """
        Initialize the classifier.
        
        Args:
            model_name: Name of the backbone model
            num_classes: Number of defect classes to classify
            pretrained: Whether to use pretrained ImageNet weights
            dropout: Dropout probability for regularization
            freeze_backbone: Whether to freeze backbone weights during training
        """
        super(WaferDefectClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Create backbone model
        self.backbone = self._create_backbone(model_name, pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get feature dimension from backbone
        self.feature_dim = self._get_feature_dim()
        
        # Create classifier head
        self.classifier = self._create_classifier_head()
        
    def _create_backbone(self, model_name: str, pretrained: bool) -> nn.Module:
        """
        Create the backbone CNN model.
        
        Args:
            model_name: Name of the model architecture
            pretrained: Whether to load pretrained weights
            
        Returns:
            Backbone model without classifier
        """
        if model_name.startswith('resnet'):
            if model_name == 'resnet18':
                backbone = models.resnet18(pretrained=pretrained)
            elif model_name == 'resnet34':
                backbone = models.resnet34(pretrained=pretrained)
            elif model_name == 'resnet50':
                backbone = models.resnet50(pretrained=pretrained)
            elif model_name == 'resnet101':
                backbone = models.resnet101(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported ResNet variant: {model_name}")
            
            # Remove the final classification layer
            backbone = nn.Sequential(*list(backbone.children())[:-1])
            
        elif model_name.startswith('efficientnet'):
            # Use timm for EfficientNet models
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
                global_pool='avg'  # Use global average pooling
            )
            
        elif model_name.startswith('mobilenet'):
            if model_name == 'mobilenet_v2':
                backbone = models.mobilenet_v2(pretrained=pretrained)
                backbone.classifier = nn.Identity()  # Remove classifier
            elif model_name == 'mobilenet_v3_large':
                backbone = models.mobilenet_v3_large(pretrained=pretrained)
                backbone.classifier = nn.Identity()  # Remove classifier
            elif model_name == 'mobilenet_v3_small':
                backbone = models.mobilenet_v3_small(pretrained=pretrained)
                backbone.classifier = nn.Identity()  # Remove classifier
            else:
                raise ValueError(f"Unsupported MobileNet variant: {model_name}")
                
        elif model_name.startswith('vit'):
            # Vision Transformer models
            backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Remove classifier
            )
            
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        return backbone
    
    def _get_feature_dim(self) -> int:
        """
        Get the feature dimension of the backbone model.
        
        Returns:
            Number of features from backbone
        """
        # Create a dummy input to infer feature dimension
        dummy_input = torch.randn(1, 3, 224, 224)
        
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # Handle different output shapes
            if len(features.shape) == 4:  # Conv features [B, C, H, W]
                features = F.adaptive_avg_pool2d(features, 1).flatten(1)
            elif len(features.shape) == 3:  # Transformer features [B, N, C]
                features = features.mean(dim=1)  # Global average pooling
            
            feature_dim = features.shape[1]
        
        return feature_dim
    
    def _create_classifier_head(self) -> nn.Module:
        """
        Create the classification head.
        
        Returns:
            Classification head module
        """
        classifier = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout / 2),
            nn.Linear(256, self.num_classes)
        )
        
        return classifier
    
    def _freeze_backbone(self):
        """Freeze backbone parameters to prevent training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters to allow fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape [B, C, H, W]
            
        Returns:
            Logits tensor of shape [B, num_classes]
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Handle different feature shapes
        if len(features.shape) == 4:  # Conv features [B, C, H, W]
            features = F.adaptive_avg_pool2d(features, 1)
            features = features.flatten(1)
        elif len(features.shape) == 3:  # Transformer features [B, N, C]
            features = features.mean(dim=1)  # Global average pooling
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def get_feature_extractor(self) -> nn.Module:
        """
        Get a feature extractor (backbone + pooling) without classification.
        
        Returns:
            Feature extractor model
        """
        class FeatureExtractor(nn.Module):
            def __init__(self, backbone, feature_dim):
                super().__init__()
                self.backbone = backbone
                self.feature_dim = feature_dim
            
            def forward(self, x):
                features = self.backbone(x)
                
                if len(features.shape) == 4:
                    features = F.adaptive_avg_pool2d(features, 1)
                    features = features.flatten(1)
                elif len(features.shape) == 3:
                    features = features.mean(dim=1)
                
                return features
        
        return FeatureExtractor(self.backbone, self.feature_dim)


class EnsembleClassifier(nn.Module):
    """
    Ensemble classifier that combines multiple models for improved performance.
    """
    
    def __init__(self, models: list, weights: Optional[list] = None):
        """
        Initialize ensemble classifier.
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (uniform if None)
        """
        super(EnsembleClassifier, self).__init__()
        
        self.models = nn.ModuleList(models)
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Ensure all models have same number of classes
        num_classes = [model.num_classes for model in models]
        if len(set(num_classes)) != 1:
            raise ValueError("All models must have the same number of classes")
        
        self.num_classes = num_classes[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            Weighted average of model predictions
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad() if not self.training else torch.enable_grad():
                pred = F.softmax(model(x), dim=1)
                predictions.append(weight * pred)
        
        # Weighted average
        ensemble_pred = torch.stack(predictions).sum(dim=0)
        
        return ensemble_pred


def create_model(model_name: str = 'efficientnet_b0',
                num_classes: int = 6,
                pretrained: bool = True,
                dropout: float = 0.2,
                freeze_backbone: bool = False) -> WaferDefectClassifier:
    """
    Factory function to create a wafer defect classifier.
    
    Args:
        model_name: Name of the backbone model
        num_classes: Number of defect classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        freeze_backbone: Whether to freeze backbone weights
        
    Returns:
        Configured model instance
    """
    model = WaferDefectClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        freeze_backbone=freeze_backbone
    )
    
    return model


def get_model_info(model: WaferDefectClassifier) -> Dict[str, Any]:
    """
    Get information about a model.
    
    Args:
        model: Model to analyze
        
    Returns:
        Dictionary containing model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        'model_name': model.model_name,
        'num_classes': model.num_classes,
        'feature_dim': model.feature_dim,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_backbone': total_params != trainable_params,
        'dropout': model.dropout
    }
    
    return info


def load_model_checkpoint(checkpoint_path: str, 
                         model_name: str = 'efficientnet_b0',
                         num_classes: int = 6,
                         device: str = 'cpu') -> WaferDefectClassifier:
    """
    Load a model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        model_name: Name of the model architecture
        num_classes: Number of classes
        device: Device to load model on
        
    Returns:
        Loaded model
    """
    # Create model with same architecture
    model = create_model(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False  # Don't load pretrained weights
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model_checkpoint(model: WaferDefectClassifier, 
                         checkpoint_path: str,
                         optimizer: Optional[torch.optim.Optimizer] = None,
                         epoch: Optional[int] = None,
                         loss: Optional[float] = None,
                         metrics: Optional[Dict] = None):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        checkpoint_path: Path to save checkpoint
        optimizer: Optional optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'model_name': model.model_name,
            'num_classes': model.num_classes,
            'dropout': model.dropout
        }
    }
    
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if epoch is not None:
        checkpoint['epoch'] = epoch
    
    if loss is not None:
        checkpoint['loss'] = loss
    
    if metrics is not None:
        checkpoint['metrics'] = metrics
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Model checkpoint saved to: {checkpoint_path}")


# Predefined model configurations
MODEL_CONFIGS = {
    'resnet18': {
        'model_name': 'resnet18',
        'feature_dim': 512,
        'description': 'Lightweight ResNet model, good for quick experiments'
    },
    'resnet50': {
        'model_name': 'resnet50', 
        'feature_dim': 2048,
        'description': 'Standard ResNet model with good accuracy/speed tradeoff'
    },
    'efficientnet_b0': {
        'model_name': 'efficientnet_b0',
        'feature_dim': 1280,
        'description': 'Efficient model optimized for mobile deployment'
    },
    'efficientnet_b2': {
        'model_name': 'efficientnet_b2',
        'feature_dim': 1408,
        'description': 'Larger EfficientNet with better accuracy'
    },
    'mobilenet_v2': {
        'model_name': 'mobilenet_v2',
        'feature_dim': 1280,
        'description': 'Very lightweight model for mobile/edge deployment'
    }
}