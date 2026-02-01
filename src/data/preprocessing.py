"""
Data preprocessing and dataset preparation utilities.

This module provides functions for organizing the raw dataset, 
creating train/validation/test splits, and preprocessing images
for the wafer defect detection model.
"""

import os
import shutil
import random
from pathlib import Path
from typing import Dict
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split


class DataPreprocessor:
    """Class for preprocessing and organizing wafer defect dataset."""
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        """
        Initialize the data preprocessor.
        
        Args:
            raw_data_path: Path to raw dataset
            processed_data_path: Path to save processed dataset
        """
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        self.class_mapping = {}
        self.dataset_stats = {}
    
    def analyze_dataset_structure(self) -> Dict:
        """
        Analyze the structure of the raw dataset.
        
        Returns:
            Dictionary containing dataset statistics
        """
        print("Analyzing dataset structure...")
        
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data path does not exist: {self.raw_data_path}")
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(list(self.raw_data_path.rglob(f"*{ext}")))
            image_files.extend(list(self.raw_data_path.rglob(f"*{ext.upper()}")))
        
        # Analyze class distribution
        classes = {}
        for img_path in image_files:
            # Assume class is the parent directory name
            class_name = img_path.parent.name
            if class_name not in classes:
                classes[class_name] = []
            classes[class_name].append(str(img_path))
        
        # Create class mapping
        self.class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(classes.keys()))}
        
        # Calculate statistics
        total_images = len(image_files)
        num_classes = len(classes)
        class_distribution = {cls: len(files) for cls, files in classes.items()}
        
        self.dataset_stats = {
            'total_images': total_images,
            'num_classes': num_classes,
            'classes': list(classes.keys()),
            'class_distribution': class_distribution,
            'class_mapping': self.class_mapping,
            'image_paths_by_class': classes
        }
        
        print(f"Dataset Analysis:")
        print(f"  Total images: {total_images}")
        print(f"  Number of classes: {num_classes}")
        print(f"  Classes: {list(classes.keys())}")
        print(f"  Class distribution: {class_distribution}")
        
        return self.dataset_stats
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                     test_ratio: float = 0.15, random_state: int = 42) -> Dict:
        """
        Create train/validation/test splits while maintaining class balance.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary containing split information
        """
        print("Creating train/validation/test splits...")
        
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
            raise ValueError("Split ratios must sum to 1.0")
        
        if not self.dataset_stats:
            self.analyze_dataset_structure()
        
        splits = {'train': [], 'val': [], 'test': []}
        
        # Create stratified splits for each class
        for class_name, image_paths in self.dataset_stats['image_paths_by_class'].items():
            if len(image_paths) < 3:
                print(f"Warning: Class {class_name} has only {len(image_paths)} samples")
                # Put all samples in training set if too few
                for path in image_paths:
                    splits['train'].append((path, class_name))
                continue
            
            # First split: train vs temp (val + test)
            train_paths, temp_paths = train_test_split(
                image_paths, 
                test_size=(val_ratio + test_ratio),
                random_state=random_state,
                stratify=None  # Can't stratify single class
            )
            
            # Second split: val vs test
            if len(temp_paths) >= 2:
                val_paths, test_paths = train_test_split(
                    temp_paths,
                    test_size=test_ratio / (val_ratio + test_ratio),
                    random_state=random_state
                )
            else:
                # If only one sample for val+test, put it in val
                val_paths = temp_paths
                test_paths = []
            
            # Add to splits with labels
            for path in train_paths:
                splits['train'].append((path, class_name))
            for path in val_paths:
                splits['val'].append((path, class_name))
            for path in test_paths:
                splits['test'].append((path, class_name))
        
        # Shuffle each split
        random.seed(random_state)
        for split in splits.values():
            random.shuffle(split)
        
        print(f"Split sizes:")
        print(f"  Train: {len(splits['train'])}")
        print(f"  Validation: {len(splits['val'])}")
        print(f"  Test: {len(splits['test'])}")
        
        return splits
    
    def organize_dataset(self, splits: Dict, copy_files: bool = True):
        """
        Organize dataset into train/val/test folders.
        
        Args:
            splits: Dictionary containing train/val/test splits
            copy_files: Whether to copy files or create symlinks
        """
        print("Organizing dataset into folders...")
        
        # Create directory structure
        for split_name in ['train', 'val', 'test']:
            for class_name in self.dataset_stats['classes']:
                split_dir = self.processed_data_path / split_name / class_name
                split_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy/link files to organized structure
        for split_name, file_list in splits.items():
            for img_path, class_name in file_list:
                src_path = Path(img_path)
                dst_dir = self.processed_data_path / split_name / class_name
                dst_path = dst_dir / src_path.name
                
                # Handle duplicate filenames
                counter = 1
                original_dst = dst_path
                while dst_path.exists():
                    stem = original_dst.stem
                    suffix = original_dst.suffix
                    dst_path = dst_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                
                if copy_files:
                    shutil.copy2(src_path, dst_path)
                else:
                    os.symlink(src_path, dst_path)
        
        print(f"Dataset organized in: {self.processed_data_path}")
    
    def create_dataset_metadata(self, splits: Dict):
        """
        Create metadata files for the dataset.
        
        Args:
            splits: Dictionary containing train/val/test splits
        """
        metadata_dir = self.processed_data_path / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        
        # Save class mapping
        class_mapping_df = pd.DataFrame([
            {'class_name': name, 'class_id': idx} 
            for name, idx in self.class_mapping.items()
        ])
        class_mapping_df.to_csv(metadata_dir / "class_mapping.csv", index=False)
        
        # Save dataset statistics
        stats_df = pd.DataFrame([
            {'metric': 'total_images', 'value': self.dataset_stats['total_images']},
            {'metric': 'num_classes', 'value': self.dataset_stats['num_classes']}
        ])
        stats_df.to_csv(metadata_dir / "dataset_stats.csv", index=False)
        
        # Save split information
        for split_name, file_list in splits.items():
            split_df = pd.DataFrame([
                {
                    'image_path': img_path,
                    'class_name': class_name,
                    'class_id': self.class_mapping[class_name]
                }
                for img_path, class_name in file_list
            ])
            split_df.to_csv(metadata_dir / f"{split_name}_split.csv", index=False)
        
        print(f"Metadata saved in: {metadata_dir}")
    
    def validate_images(self, max_samples_per_class: int = 10):
        """
        Validate a sample of images to ensure they can be loaded properly.
        
        Args:
            max_samples_per_class: Maximum number of samples to validate per class
        """
        print("Validating sample images...")
        
        corrupted_files = []
        valid_count = 0
        
        for class_name, image_paths in self.dataset_stats['image_paths_by_class'].items():
            sample_paths = random.sample(image_paths, min(len(image_paths), max_samples_per_class))
            
            for img_path in sample_paths:
                try:
                    with Image.open(img_path) as img:
                        img.verify()  # Verify image integrity
                        valid_count += 1
                except Exception as e:
                    corrupted_files.append((img_path, str(e)))
        
        if corrupted_files:
            print(f"Found {len(corrupted_files)} corrupted files:")
            for file_path, error in corrupted_files[:5]:  # Show first 5
                print(f"  {file_path}: {error}")
        
        print(f"Validated {valid_count} images successfully")
        return corrupted_files
    
    def preprocess_dataset(self, train_ratio: float = 0.7, val_ratio: float = 0.15, 
                          test_ratio: float = 0.15, copy_files: bool = True,
                          validate_images: bool = True) -> Dict:
        """
        Complete preprocessing pipeline.
        
        Args:
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation  
            test_ratio: Proportion of data for testing
            copy_files: Whether to copy files or create symlinks
            validate_images: Whether to validate image integrity
            
        Returns:
            Dictionary containing preprocessing results
        """
        print("Starting complete dataset preprocessing...")
        
        # Step 1: Analyze dataset structure
        self.analyze_dataset_structure()
        
        # Step 2: Validate images (optional)
        corrupted_files = []
        if validate_images:
            corrupted_files = self.validate_images()
        
        # Step 3: Create splits
        splits = self.create_splits(train_ratio, val_ratio, test_ratio)
        
        # Step 4: Organize dataset
        self.organize_dataset(splits, copy_files)
        
        # Step 5: Create metadata
        self.create_dataset_metadata(splits)
        
        results = {
            'dataset_stats': self.dataset_stats,
            'splits': splits,
            'corrupted_files': corrupted_files,
            'processed_path': str(self.processed_data_path)
        }
        
        print("Dataset preprocessing completed successfully!")
        return results


def preprocess_wafer_dataset(raw_data_path: str, processed_data_path: str, 
                           train_ratio: float = 0.7, val_ratio: float = 0.15,
                           test_ratio: float = 0.15) -> Dict:
    """
    Convenience function to preprocess the wafer defect dataset.
    
    Args:
        raw_data_path: Path to raw dataset
        processed_data_path: Path to save processed dataset
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        Dictionary containing preprocessing results
    """
    preprocessor = DataPreprocessor(raw_data_path, processed_data_path)
    return preprocessor.preprocess_dataset(train_ratio, val_ratio, test_ratio)