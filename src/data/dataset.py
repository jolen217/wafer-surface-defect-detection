"""
PyTorch Dataset and DataLoader implementations for wafer defect detection.

This module provides custom Dataset classes with data augmentation,
normalization, and efficient data loading for training and inference.
"""

import os
from pathlib import Path
from typing import Tuple, Optional, Callable, List, Dict, Any
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2


class WaferDefectDataset(Dataset):
    """
    Custom Dataset class for wafer defect images.
    
    Supports data augmentation, normalization, and efficient loading
    for both training and inference.
    """
    
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[Callable] = None,
                 augment: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to processed dataset directory
            split: Data split ('train', 'val', or 'test')
            image_size: Target image size (height, width)
            transform: Custom transform function (overrides default transforms)
            augment: Whether to apply data augmentation (for training only)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augment = augment and (split == 'train')
        
        # Load class mapping
        self.class_mapping = self._load_class_mapping()
        self.num_classes = len(self.class_mapping)
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Set up transforms
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transforms()
    
    def _load_class_mapping(self) -> Dict[str, int]:
        """Load class mapping from metadata."""
        metadata_path = self.data_dir / "metadata" / "class_mapping.csv"
        
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            return dict(zip(df['class_name'], df['class_id']))
        else:
            # Fallback: infer from directory structure
            split_dir = self.data_dir / self.split
            if split_dir.exists():
                class_names = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
                return {name: idx for idx, name in enumerate(class_names)}
            else:
                raise FileNotFoundError(f"Cannot find class mapping or split directory: {split_dir}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load image paths and corresponding labels."""
        samples = []
        split_dir = self.data_dir / self.split
        
        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")
        
        # Collect samples from each class directory
        for class_name, class_id in self.class_mapping.items():
            class_dir = split_dir / class_name
            if class_dir.exists():
                # Find all image files in class directory
                image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
                
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in image_extensions:
                        samples.append((str(img_path), class_id))
        
        if not samples:
            raise ValueError(f"No images found in {split_dir}")
        
        return samples
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default image transforms based on split and augmentation settings."""
        
        if self.augment and self.split == 'train':
            # Training transforms with augmentation
            transforms_list = [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Rotate(limit=15, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.Blur(blur_limit=3, p=0.1),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],  # ImageNet means
                    std=[0.229, 0.224, 0.225]   # ImageNet stds
                ),
                ToTensorV2()
            ]
        else:
            # Validation/Test transforms (no augmentation)
            transforms_list = [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ]
        
        return A.Compose(transforms_list)
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path, label = self.samples[idx]
        
        try:
            # Load image
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            return image_tensor, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            black_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            transformed = self.transform(image=black_image)
            return transformed['image'], label
    
    def get_class_weights(self) -> torch.Tensor:
        """
        Calculate class weights for handling imbalanced datasets.
        
        Returns:
            Tensor of class weights
        """
        from collections import Counter
        
        # Count samples per class
        labels = [label for _, label in self.samples]
        class_counts = Counter(labels)
        
        # Calculate weights (inverse frequency)
        total_samples = len(self.samples)
        weights = []
        
        for class_id in range(self.num_classes):
            if class_id in class_counts:
                weight = total_samples / (self.num_classes * class_counts[class_id])
            else:
                weight = 1.0  # Default weight for classes with no samples
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)
    
    def get_class_distribution(self) -> Dict[str, int]:
        """
        Get the distribution of samples across classes.
        
        Returns:
            Dictionary mapping class names to sample counts
        """
        from collections import Counter
        
        labels = [label for _, label in self.samples]
        class_counts = Counter(labels)
        
        # Map class IDs back to names
        id_to_name = {v: k for k, v in self.class_mapping.items()}
        distribution = {}
        
        for class_id in range(self.num_classes):
            class_name = id_to_name.get(class_id, f"class_{class_id}")
            distribution[class_name] = class_counts.get(class_id, 0)
        
        return distribution


def create_data_loaders(data_dir: str,
                       image_size: Tuple[int, int] = (224, 224),
                       batch_size: int = 32,
                       num_workers: int = 4,
                       pin_memory: bool = True,
                       augment_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_dir: Path to processed dataset directory
        image_size: Target image size (height, width)
        batch_size: Batch size for training
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster GPU transfer
        augment_train: Whether to apply augmentation to training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Create datasets
    train_dataset = WaferDefectDataset(
        data_dir=data_dir,
        split='train',
        image_size=image_size,
        augment=augment_train
    )
    
    val_dataset = WaferDefectDataset(
        data_dir=data_dir,
        split='val',
        image_size=image_size,
        augment=False
    )
    
    test_dataset = WaferDefectDataset(
        data_dir=data_dir,
        split='test',
        image_size=image_size,
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent training
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader


def analyze_dataset_statistics(data_dir: str) -> Dict[str, Any]:
    """
    Analyze and print dataset statistics.
    
    Args:
        data_dir: Path to processed dataset directory
        
    Returns:
        Dictionary containing dataset statistics
    """
    stats = {}
    
    for split in ['train', 'val', 'test']:
        try:
            dataset = WaferDefectDataset(data_dir=data_dir, split=split, augment=False)
            
            split_stats = {
                'num_samples': len(dataset),
                'num_classes': dataset.num_classes,
                'class_distribution': dataset.get_class_distribution(),
                'class_mapping': dataset.class_mapping
            }
            
            stats[split] = split_stats
            
            print(f"\n{split.upper()} SET STATISTICS:")
            print(f"  Total samples: {len(dataset)}")
            print(f"  Number of classes: {dataset.num_classes}")
            print(f"  Class distribution:")
            for class_name, count in dataset.get_class_distribution().items():
                print(f"    {class_name}: {count}")
                
        except FileNotFoundError:
            print(f"Warning: {split} split not found")
            stats[split] = None
    
    return stats


class InferenceDataset(Dataset):
    """
    Dataset class for inference on single images or batches of images.
    """
    
    def __init__(self, 
                 image_paths: List[str],
                 image_size: Tuple[int, int] = (224, 224),
                 transform: Optional[Callable] = None):
        """
        Initialize inference dataset.
        
        Args:
            image_paths: List of paths to images for inference
            image_size: Target image size (height, width)
            transform: Custom transform function
        """
        self.image_paths = image_paths
        self.image_size = image_size
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = A.Compose([
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Get image tensor and path for inference.
        
        Returns:
            Tuple of (image_tensor, image_path)
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            return image_tensor, img_path
            
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return black image as fallback
            black_image = np.zeros((self.image_size[0], self.image_size[1], 3), dtype=np.uint8)
            transformed = self.transform(image=black_image)
            return transformed['image'], img_path


def create_inference_loader(image_paths: List[str],
                          image_size: Tuple[int, int] = (224, 224),
                          batch_size: int = 32) -> DataLoader:
    """
    Create DataLoader for inference.
    
    Args:
        image_paths: List of image paths for inference
        image_size: Target image size
        batch_size: Batch size for inference
        
    Returns:
        DataLoader for inference
    """
    dataset = InferenceDataset(image_paths, image_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Single worker for inference
        pin_memory=True
    )