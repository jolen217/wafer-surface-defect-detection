"""
Example script for inference with the trained wafer defect detection model.

This script demonstrates how to use the trained model for inference
on new images, both single image and batch processing.
"""

import sys
import argparse
from pathlib import Path
import torch
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.models import load_model_checkpoint


def predict_single_image(model, image_path, class_names, device='cpu'):
    """
    Predict defect class for a single image.
    
    Args:
        model: Trained model
        image_path: Path to image file
        class_names: List of class names
        device: Device for inference
        
    Returns:
        Prediction results dictionary
    """
    from src.data.dataset import InferenceDataset
    
    # Create dataset for single image
    dataset = InferenceDataset([str(image_path)])
    image_tensor, _ = dataset[0]
    
    # Add batch dimension and move to device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Get all class probabilities
    all_probs = probabilities[0].cpu().numpy()
    
    result = {
        'image_path': str(image_path),
        'predicted_class': class_names[predicted_class],
        'predicted_class_id': predicted_class,
        'confidence': float(confidence),
        'all_probabilities': {
            class_name: float(prob)
            for class_name, prob in zip(class_names, all_probs)
        }
    }
    
    return result


def predict_batch(model, image_paths, class_names, device='cpu', batch_size=32):
    """
    Predict defect classes for multiple images.
    
    Args:
        model: Trained model
        image_paths: List of image paths
        class_names: List of class names
        device: Device for inference
        batch_size: Batch size for processing
        
    Returns:
        List of prediction results
    """
    from torch.utils.data import DataLoader
    from src.data.dataset import InferenceDataset
    
    # Create dataset and loader
    dataset = InferenceDataset(image_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for batch_idx, (images, paths) in enumerate(dataloader):
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            
            # Process each image in the batch
            for i in range(len(images)):
                predicted_class = predicted_classes[i].item()
                confidence = probabilities[i][predicted_class].item()
                all_probs = probabilities[i].cpu().numpy()
                
                result = {
                    'image_path': paths[i],
                    'predicted_class': class_names[predicted_class],
                    'predicted_class_id': predicted_class,
                    'confidence': float(confidence),
                    'all_probabilities': {
                        class_name: float(prob)
                        for class_name, prob in zip(class_names, all_probs)
                    }
                }
                results.append(result)
    
    return results


def load_class_names(processed_data_path):
    """Load class names from metadata."""
    try:
        import pandas as pd
        metadata_path = Path(processed_data_path) / "metadata" / "class_mapping.csv"
        
        if metadata_path.exists():
            df = pd.read_csv(metadata_path)
            class_mapping = dict(zip(df['class_id'], df['class_name']))
            return [class_mapping[i] for i in range(len(class_mapping))]
        else:
            # Fallback
            return [f"class_{i}" for i in range(6)]  # Default number of classes
    except Exception:
        return [f"class_{i}" for i in range(6)]


def main():
    parser = argparse.ArgumentParser(description='Wafer Defect Detection Inference')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--image-dir', type=str,
                       help='Directory containing images for batch prediction')
    parser.add_argument('--output', type=str, default='predictions.json',
                       help='Output file for predictions')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference (auto, cpu, cuda)')
    parser.add_argument('--processed-data-path', type=str, 
                       default='data/processed',
                       help='Path to processed data for class names')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from: {args.model_path}")
    try:
        # Try to load checkpoint info first
        checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
        
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
            model = load_model_checkpoint(
                args.model_path,
                model_name=model_config['model_name'],
                num_classes=model_config['num_classes'],
                device=device
            )
        else:
            # Fallback with default config
            model = load_model_checkpoint(
                args.model_path,
                model_name='efficientnet_b0',
                num_classes=6,
                device=device
            )
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    print("Model loaded successfully!")
    
    # Load class names
    class_names = load_class_names(args.processed_data_path)
    print(f"Classes: {class_names}")
    
    # Perform inference
    if args.image:
        # Single image prediction
        print(f"Predicting for single image: {args.image}")
        
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}")
            return
        
        result = predict_single_image(model, args.image, class_names, device)
        
        print(f"\nPrediction Results:")
        print(f"Image: {result['image_path']}")
        print(f"Predicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nAll Probabilities:")
        for class_name, prob in result['all_probabilities'].items():
            print(f"  {class_name}: {prob:.4f}")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    elif args.image_dir:
        # Batch prediction
        print(f"Predicting for images in directory: {args.image_dir}")
        
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            print(f"Error: Directory not found: {args.image_dir}")
            return
        
        # Find all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(image_dir.glob(f"*{ext}")))
            image_paths.extend(list(image_dir.glob(f"*{ext.upper()}")))
        
        if not image_paths:
            print(f"Error: No images found in {args.image_dir}")
            return
        
        print(f"Found {len(image_paths)} images")
        
        results = predict_batch(
            model, [str(p) for p in image_paths], 
            class_names, device, args.batch_size
        )
        
        print(f"\nBatch Prediction Results:")
        for result in results:
            print(f"{Path(result['image_path']).name}: {result['predicted_class']} ({result['confidence']:.3f})")
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {args.output}")
        
    else:
        print("Error: Either --image or --image-dir must be specified")


if __name__ == '__main__':
    main()