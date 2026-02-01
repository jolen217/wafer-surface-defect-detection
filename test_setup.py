#!/usr/bin/env python3
"""
Simple test script to validate the project setup.

This script tests basic functionality without requiring a trained model
or dataset, useful for initial setup validation.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("✗ PyTorch not found - install with: pip install torch")
        return False
    
    try:
        import torchvision
        print(f"✓ torchvision {torchvision.__version__}")
    except ImportError:
        print("✗ torchvision not found - install with: pip install torchvision")
        return False
    
    try:
        import fastapi
        print(f"✓ FastAPI {fastapi.__version__}")
    except ImportError:
        print("✗ FastAPI not found - install with: pip install fastapi")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy {np.__version__}")
    except ImportError:
        print("✗ NumPy not found - install with: pip install numpy")
        return False
    
    try:
        import pandas as pd
        print(f"✓ Pandas {pd.__version__}")
    except ImportError:
        print("✗ Pandas not found - install with: pip install pandas")
        return False
    
    try:
        import albumentations
        print(f"✓ Albumentations {albumentations.__version__}")
    except ImportError:
        print("✗ Albumentations not found - install with: pip install albumentations")
        return False
    
    try:
        import timm
        print(f"✓ timm {timm.__version__}")
    except ImportError:
        print("✗ timm not found - install with: pip install timm")
        return False
    
    return True


def test_project_structure():
    """Test that the project structure is correct."""
    print("\nTesting project structure...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        'src',
        'src/data',
        'src/models', 
        'src/utils',
        'config',
        'api',
        'data',
        'models',
        'logs',
        'notebooks'
    ]
    
    required_files = [
        'requirements.txt',
        'config/config.yaml',
        'src/config.py',
        'src/data/dataset.py',
        'src/models/classifier.py',
        'api/main.py',
        'train.py',
        'Dockerfile'
    ]
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ Directory: {dir_path}")
        else:
            print(f"✗ Missing directory: {dir_path}")
            all_good = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ File: {file_path}")
        else:
            print(f"✗ Missing file: {file_path}")
            all_good = False
    
    return all_good


def test_config_loading():
    """Test configuration loading."""
    print("\nTesting configuration loading...")
    
    try:
        # Add src to path
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.config import get_config
        config = get_config()
        
        print(f"✓ Config loaded successfully")
        print(f"  Model: {config.model.name}")
        print(f"  Batch size: {config.data.batch_size}")
        print(f"  Epochs: {config.training.epochs}")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_model_creation():
    """Test model creation."""
    print("\nTesting model creation...")
    
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.models import create_model
        
        model = create_model(
            model_name='efficientnet_b0',
            num_classes=6,
            pretrained=False,  # Don't download weights for testing
            dropout=0.2
        )
        
        print(f"✓ Model created successfully")
        print(f"  Model type: {type(model).__name__}")
        
        # Test forward pass with dummy data
        import torch
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        
        print(f"  Output shape: {output.shape}")
        print(f"  Expected: [1, 6]")
        
        if output.shape == torch.Size([1, 6]):
            print("✓ Model forward pass successful")
            return True
        else:
            print("✗ Unexpected output shape")
            return False
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_data_preprocessing():
    """Test data preprocessing utilities."""
    print("\nTesting data preprocessing...")
    
    try:
        sys.path.append(str(Path(__file__).parent / 'src'))
        
        from src.data.preprocessing import DataPreprocessor
        
        # Test with dummy paths (won't actually process)
        preprocessor = DataPreprocessor(
            raw_data_path="dummy/path",
            processed_data_path="dummy/processed"
        )
        
        print("✓ DataPreprocessor created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Data preprocessing test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("WAFER DEFECT DETECTION - SETUP VALIDATION")
    print("="*60)
    
    tests = [
        ("Import Test", test_imports),
        ("Project Structure", test_project_structure),
        ("Configuration", test_config_loading),
        ("Model Creation", test_model_creation),
        ("Data Preprocessing", test_data_preprocessing)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-'*20} {test_name} {'-'*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*60}")
    print("VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        emoji = "✓" if result else "✗"
        print(f"{emoji} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The project setup is ready.")
        print("\nNext steps:")
        print("1. Download dataset to data/raw/")
        print("2. Run: python train.py --data-path data/raw")
        print("3. Start API: cd api && uvicorn main:app --reload")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix the issues above.")
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())