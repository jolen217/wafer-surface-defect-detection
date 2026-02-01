#!/usr/bin/env python3
"""
Script to download and setup the Mixed-type Wafer Defect Dataset from Kaggle.

This script automatically downloads the dataset, extracts it to the correct
directory structure, and performs initial validation.

Requirements:
- Kaggle API credentials configured (~/.kaggle/kaggle.json)
- Install: pip install kaggle
"""

import os
import sys
import zipfile
import shutil
from pathlib import Path
import subprocess
import argparse


def check_kaggle_setup():
    """Check if Kaggle API is properly configured."""
    print("Checking Kaggle API setup...")
    
    # Check for environment variable authentication first
    kaggle_token = os.getenv('KAGGLE_API_TOKEN')
    
    if kaggle_token:
        print(f"✓ Using KAGGLE_API_TOKEN environment variable")
        
        # Create kaggle.json for token authentication
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"
        
        kaggle_dir.mkdir(exist_ok=True)
        
        # Create config for token
        import json
        temp_config = {
            "username": "token-user",
            "key": kaggle_token
        }
        
        try:
            with open(kaggle_json, 'w') as f:
                json.dump(temp_config, f)
            
            # Set proper permissions
            kaggle_json.chmod(0o600)
            
            # Now test authentication
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            api = KaggleApi()
            api.authenticate()
            print("✓ Kaggle API authentication successful (via token)")
            
            return True
            
        except Exception as e:
            print(f"✗ Kaggle API authentication failed with token: {e}")
            
            # Clean up file on failure
            if kaggle_json.exists():
                kaggle_json.unlink()
            
            print("Please check your KAGGLE_API_TOKEN value")
            return False
    
    # Fallback to checking kaggle package availability
    try:
        import kaggle
        print("✓ Kaggle package installed")
    except ImportError:
        print("✗ Kaggle package not found")
        print("Install with: pip install kaggle")
        return False
    
    # Check for kaggle.json file authentication
    kaggle_config_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_config_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("✗ Kaggle API credentials not found")
        print("Please either:")
        print("1. Set KAGGLE_API_TOKEN environment variable, or")
        print(f"2. Create {kaggle_json} with your API credentials:")
        print("   - Go to https://www.kaggle.com/account")
        print("   - Click 'Create New API Token'")
        print("   - Save kaggle.json to ~/.kaggle/")
        print("   - Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Check if file is empty
    if kaggle_json.stat().st_size == 0:
        print("✗ Kaggle credentials file is empty")
        print("Please either:")
        print("1. Set KAGGLE_API_TOKEN environment variable, or")
        print("2. Add proper credentials to ~/.kaggle/kaggle.json")
        return False
    
    # Check permissions
    stat_info = kaggle_json.stat()
    if stat_info.st_mode & 0o077:
        print("✗ Kaggle credentials file has wrong permissions")
        print(f"Fix with: chmod 600 {kaggle_json}")
        return False
    
    print("✓ Kaggle API credentials configured (via file)")
    
    # Test API access
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("✓ Kaggle API authentication successful (via file)")
        return True
    except Exception as e:
        print(f"✗ Kaggle API authentication failed: {e}")
        return False


def download_dataset(dataset_name, download_path, force=False):
    """
    Download dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier
        download_path: Path to download the dataset
        force: Force re-download even if exists
    """
    print(f"Downloading dataset: {dataset_name}")
    print(f"Download path: {download_path}")
    
    download_path = Path(download_path)
    download_path.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if not force and any(download_path.iterdir()):
        print("Dataset files already exist. Use --force to re-download.")
        return True
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        # Download dataset
        print("Starting download...")
        api.dataset_download_files(
            dataset_name,
            path=str(download_path),
            unzip=True,
            quiet=False
        )
        
        print("✓ Dataset downloaded successfully")
        return True
        
    except Exception as e:
        print(f"✗ Download failed: {e}")
        return False


def organize_dataset(download_path, target_path):
    """
    Organize downloaded dataset into the expected structure.
    Handles both NPZ files and image directories.
    
    Args:
        download_path: Path where dataset was downloaded
        target_path: Target path for organized dataset
    """
    print("Organizing dataset structure...")
    
    download_path = Path(download_path)
    target_path = Path(target_path)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    try:
        import numpy as np
        
        # Look for NPZ files first (common format for wafer datasets)
        npz_files = list(download_path.glob("*.npz"))
        if npz_files:
            return organize_npz_dataset(npz_files[0], target_path)
        
        # Fall back to image files
        return organize_image_dataset(download_path, target_path)
        
    except ImportError:
        print("✗ NumPy not available for NPZ processing")
        print("Install with: pip install numpy")
        return False
    except Exception as e:
        print(f"✗ Error organizing dataset: {e}")
        return False


def organize_npz_dataset(npz_file: Path, output_path: Path) -> bool:
    """Organize NPZ dataset into train/val splits with class folders."""
    try:
        import numpy as np
        
        print(f"Processing NPZ file: {npz_file.name}")
        
        # Load the NPZ file
        data = np.load(npz_file, allow_pickle=True)
        
        print(f"NPZ file keys: {list(data.keys())}")
        
        # Assume first array is images, second is labels
        arrays = [data[key] for key in data.keys()]
        images = arrays[0]  # Should be (N, H, W) or (N, H, W, C)
        labels = arrays[1]  # Should be (N,) or (N, num_classes)
        
        print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")
        
        # Handle different label formats
        if labels.ndim > 1:
            # One-hot encoded or multi-class - take argmax
            labels = np.argmax(labels, axis=1)
        
        # Create class mapping
        unique_labels = np.unique(labels)
        class_names = [f"class_{i}" for i in unique_labels]
        
        print(f"Found {len(unique_labels)} classes: {class_names}")
        
        # Create directory structure
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        
        for class_name in class_names:
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Split dataset (80% train, 20% validation)
        from sklearn.model_selection import train_test_split
        
        indices = np.arange(len(images))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=labels
        )
        
        print(f"Train samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
        
        # Save images to respective directories
        def save_images(indices, base_dir, split_name):
            for i, idx in enumerate(indices):
                image_data = images[idx]
                label = labels[idx]
                class_name = f"class_{label}"
                
                # Normalize image data to 0-255 range
                if image_data.dtype != np.uint8:
                    # Scale to 0-255 range
                    img_min, img_max = image_data.min(), image_data.max()
                    if img_max > img_min:
                        image_data = ((image_data - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        image_data = np.zeros_like(image_data, dtype=np.uint8)
                
                # Save as PNG using numpy
                filename = f"{split_name}_{idx:06d}.png"
                filepath = base_dir / class_name / filename
                
                # Simple PNG saving without PIL for now
                import imageio
                imageio.imwrite(str(filepath), image_data)
                
                if (i + 1) % 1000 == 0:
                    print(f"  Saved {i + 1}/{len(indices)} {split_name} images...")
        
        print("Saving training images...")
        save_images(train_idx, train_dir, "train")
        
        print("Saving validation images...")
        save_images(val_idx, val_dir, "val")
        
        # Create dataset info file
        info = {
            "dataset_name": "Mixed-type Wafer Defect Dataset",
            "total_samples": len(images),
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            "image_size": list(images.shape[1:]),
            "num_classes": len(unique_labels),
            "class_names": class_names,
            "source_file": npz_file.name
        }
        
        import json
        with open(output_path / "dataset_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"✓ Successfully organized {len(images)} samples")
        print(f"  - Training: {len(train_idx)} samples")
        print(f"  - Validation: {len(val_idx)} samples")
        print(f"  - Classes: {len(unique_labels)}")
        print(f"  - Image size: {images.shape[1:]}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Missing required package: {e}")
        print("Install with: pip install scikit-learn imageio")
        return False
    except Exception as e:
        print(f"✗ Error processing NPZ file: {e}")
        return False


def organize_image_dataset(download_path: Path, target_path: Path) -> bool:
    """Organize image files from download directory."""
    # Find all files in download directory
    all_files = list(download_path.rglob("*"))
    image_files = [f for f in all_files if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}]
    
    if not image_files:
        print("✗ No image files found in downloaded dataset")
        return False
    
    print(f"Found {len(image_files)} image files")
    
    # Analyze directory structure to understand class organization
    class_dirs = set()
    for img_file in image_files:
        # Get the parent directory name as potential class
        relative_path = img_file.relative_to(download_path)
        if len(relative_path.parts) > 1:
            class_name = relative_path.parts[-2]  # Parent directory
        else:
            class_name = "unknown"
        class_dirs.add(class_name)
    
    print(f"Detected classes: {sorted(class_dirs)}")
    
    # Copy files to organized structure
    copied_count = 0
    for img_file in image_files:
        try:
            relative_path = img_file.relative_to(download_path)
            
            # Determine class name
            if len(relative_path.parts) > 1:
                class_name = relative_path.parts[-2]
            else:
                class_name = "unknown"
            
            # Create class directory
            class_dir = target_path / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Copy file
            target_file = class_dir / img_file.name
            
            # Handle duplicate names
            counter = 1
            original_target = target_file
            while target_file.exists():
                stem = original_target.stem
                suffix = original_target.suffix
                target_file = class_dir / f"{stem}_{counter}{suffix}"
                counter += 1
            
            shutil.copy2(img_file, target_file)
            copied_count += 1
            
            if copied_count % 100 == 0:
                print(f"Copied {copied_count}/{len(image_files)} files...")
                
        except Exception as e:
            print(f"Error copying {img_file}: {e}")
    
    print(f"✓ Organized {copied_count} images into {len(class_dirs)} classes")
    
    # Create summary
    summary = {}
    for class_dir in target_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            summary[class_dir.name] = count
    
    print("\nClass distribution:")
    for class_name, count in sorted(summary.items()):
        print(f"  {class_name}: {count} images")
    
    return True


def validate_dataset(dataset_path):
    """
    Validate the organized dataset.
    
    Args:
        dataset_path: Path to organized dataset
    """
    print("\nValidating dataset...")
    
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print("✗ Dataset path does not exist")
        return False
    
    # Count classes and images
    class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print("✗ No class directories found")
        return False
    
    total_images = 0
    min_images_per_class = float('inf')
    max_images_per_class = 0
    
    for class_dir in class_dirs:
        image_files = [f for f in class_dir.iterdir() if f.is_file()]
        count = len(image_files)
        total_images += count
        min_images_per_class = min(min_images_per_class, count)
        max_images_per_class = max(max_images_per_class, count)
        
        # Validate a few images
        for img_file in image_files[:3]:
            try:
                from PIL import Image
                with Image.open(img_file) as img:
                    img.verify()
            except Exception as e:
                print(f"✗ Corrupted image found: {img_file} - {e}")
                return False
    
    print(f"✓ Dataset validation successful")
    print(f"  Classes: {len(class_dirs)}")
    print(f"  Total images: {total_images}")
    print(f"  Images per class: {min_images_per_class} - {max_images_per_class}")
    
    if min_images_per_class < 10:
        print("⚠️  Warning: Some classes have very few images")
    
    return True


def cleanup_download(download_path):
    """Clean up temporary download files."""
    download_path = Path(download_path)
    if download_path.exists() and download_path.name == "temp_download":
        print("Cleaning up temporary files...")
        shutil.rmtree(download_path)
        print("✓ Cleanup completed")


def main():
    parser = argparse.ArgumentParser(description='Download and setup Kaggle wafer defect dataset')
    
    # Dataset selection
    dataset_group = parser.add_mutually_exclusive_group(required=False)
    dataset_group.add_argument('--mixed-type', action='store_true',
                              help='Download Mixed-type Wafer Defect Dataset (default)')
    dataset_group.add_argument('--wm811k', action='store_true',
                              help='Download WM-811K Wafer Map Dataset')
    dataset_group.add_argument('--custom-dataset', type=str,
                              help='Custom Kaggle dataset identifier (username/dataset-name)')
    
    # Paths
    parser.add_argument('--output-dir', type=str, default='data/raw',
                       help='Output directory for organized dataset (default: data/raw)')
    parser.add_argument('--download-dir', type=str, default='temp_download',
                       help='Temporary download directory (default: temp_download)')
    
    # Options
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if files exist')
    parser.add_argument('--keep-download', action='store_true',
                       help='Keep temporary download files')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate existing dataset, skip download')
    
    args = parser.parse_args()
    
    # Determine dataset to download
    if args.custom_dataset:
        dataset_name = args.custom_dataset
    elif args.wm811k:
        dataset_name = "qingyi/wm811k-wafer-map"
    else:  # Default to mixed-type
        dataset_name = "co1d7era/mixedtype-wafer-defect-datasets"
    
    print("="*60)
    print("KAGGLE WAFER DEFECT DATASET DOWNLOADER")
    print("="*60)
    print(f"Dataset: {dataset_name}")
    print(f"Output directory: {args.output_dir}")
    
    # Validation only mode
    if args.validate_only:
        success = validate_dataset(args.output_dir)
        return 0 if success else 1
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n❌ Kaggle setup incomplete. Please fix the issues above.")
        return 1
    
    try:
        # Download dataset
        success = download_dataset(dataset_name, args.download_dir, args.force)
        if not success:
            return 1
        
        # Organize dataset
        success = organize_dataset(args.download_dir, args.output_dir)
        if not success:
            return 1
        
        # Validate organized dataset
        success = validate_dataset(args.output_dir)
        if not success:
            return 1
        
        # Cleanup
        if not args.keep_download:
            cleanup_download(args.download_dir)
        
        print(f"\n🎉 Dataset setup completed successfully!")
        print(f"Dataset location: {Path(args.output_dir).absolute()}")
        print(f"\nNext steps:")
        print(f"1. Run training: python train.py --data-path {args.output_dir}")
        print(f"2. Or test setup: python test_setup.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Download interrupted by user")
        cleanup_download(args.download_dir)
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        cleanup_download(args.download_dir)
        return 1


if __name__ == '__main__':
    sys.exit(main())