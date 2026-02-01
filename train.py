"""
Main training script for wafer defect classification.

This script provides a complete training pipeline with configuration
management, data loading, model training, and evaluation.
"""

import sys
import argparse
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from src.config import get_config
from src.data import create_data_loaders, analyze_dataset_statistics, preprocess_wafer_dataset
from src.models import create_model, get_model_info, save_model_checkpoint
from src.utils import WaferDefectTrainer


def setup_environment(config):
    """Setup training environment and device."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
    
    # Setup device
    if config.device.use_cuda and torch.cuda.is_available():
        device = f'cuda:{config.device.device_id}'
        cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(config.device.device_id)}")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Wafer Defect Classification Model')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data-path', type=str, required=True,
                       help='Path to raw dataset')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run without training (for testing setup)')
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = get_config()
    
    # Setup environment
    device = setup_environment(config)
    
    # Setup paths
    project_root = Path(__file__).parent
    raw_data_path = Path(args.data_path)
    processed_data_path = project_root / config.data.processed_path
    
    print(f"Project root: {project_root}")
    print(f"Raw data path: {raw_data_path}")
    print(f"Processed data path: {processed_data_path}")
    
    # Check if we need to preprocess data
    if not processed_data_path.exists() or len(list(processed_data_path.iterdir())) == 0:
        print("\nPreprocessing dataset...")
        preprocess_results = preprocess_wafer_dataset(
            str(raw_data_path),
            str(processed_data_path),
            config.data.train_split,
            config.data.val_split,
            config.data.test_split
        )
        
        # Update config with actual number of classes
        actual_num_classes = preprocess_results['dataset_stats']['num_classes']
        config.update_num_classes(actual_num_classes)
        config.save_config()  # Save updated config
        
        print(f"Dataset preprocessing completed!")
        print(f"Found {actual_num_classes} classes")
        
    else:
        print("Using existing processed dataset...")
    
    # Analyze dataset
    print("\nAnalyzing dataset...")
    dataset_stats = analyze_dataset_statistics(str(processed_data_path))
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=str(processed_data_path),
        image_size=tuple(config.data.image_size),
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        augment_train=config.augmentation.enabled
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\nCreating model: {config.model.name}")
    model = create_model(
        model_name=config.model.name,
        num_classes=config.model.num_classes,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout
    )
    
    # Print model information
    model_info = get_model_info(model)
    print(f"Model created successfully!")
    print(f"Total parameters: {model_info['total_parameters']:,}")
    print(f"Trainable parameters: {model_info['trainable_parameters']:,}")
    
    if args.dry_run:
        print("\nDry run completed successfully!")
        return
    
    # Setup training configuration
    training_config = {
        'epochs': config.training.epochs,
        'optimizer': {
            'type': config.training.optimizer,
            'learning_rate': config.training.learning_rate,
            'weight_decay': config.training.weight_decay
        },
        'scheduler': {
            'type': config.training.scheduler
        },
        'early_stopping': {
            'enabled': True,
            'patience': config.training.early_stopping_patience
        },
        'save_frequency': config.logging.save_frequency,
        'loss': {
            'type': 'cross_entropy',
            'use_class_weights': True  # Handle class imbalance
        }
    }
    
    # Create trainer
    trainer = WaferDefectTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=training_config,
        device=device,
        log_dir=str(project_root / config.logging.log_dir),
        checkpoint_dir=str(project_root / 'checkpoints')
    )
    
    # Start training
    print(f"\nStarting training for {config.training.epochs} epochs...")
    results = trainer.train(
        num_epochs=config.training.epochs,
        resume_from=args.resume
    )
    
    # Save final model
    final_model_path = project_root / 'models' / 'final_model.pth'
    final_model_path.parent.mkdir(exist_ok=True)
    
    save_model_checkpoint(
        model=model,
        checkpoint_path=str(final_model_path),
        metrics=results['test_metrics']
    )
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
    print(f"Test accuracy: {results['test_metrics']['accuracy']:.2f}%")
    print(f"Test F1-score: {results['test_metrics']['f1_score']:.4f}")
    print(f"Final model saved to: {final_model_path}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"- Model: {config.model.name}")
    print(f"- Dataset: {len(train_loader.dataset)} training samples")
    print(f"- Classes: {config.model.num_classes}")
    print(f"- Training time: {sum(results['history']['epoch_times']):.2f} seconds")
    print(f"- Final validation loss: {results['best_val_loss']:.4f}")


if __name__ == '__main__':
    main()