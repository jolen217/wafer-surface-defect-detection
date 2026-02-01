"""
FastAPI application for wafer defect classification inference.

This module provides a REST API for serving the trained wafer defect
classification model with endpoints for single image prediction
and batch inference.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import uvicorn
from pathlib import Path
import sys
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from src.models import load_model_checkpoint
from src.config import get_config


class WaferDefectPredictor:
    """
    Wafer defect prediction service.
    """
    
    def __init__(self, model_path: str, config_path: str = None, device: str = 'cpu'):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device for inference
        """
        self.device = device
        self.model_path = model_path
        
        # Load configuration
        if config_path:
            self.config = get_config()
        else:
            self.config = get_config()
        
        # Load model
        self.model = self._load_model()
        self.model.eval()
        
        # Setup image preprocessing
        self.transform = A.Compose([
            A.Resize(height=224, width=224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Load class names
        self.class_names = self._load_class_names()
        
        print(f"Model loaded successfully!")
        print(f"Number of classes: {len(self.class_names)}")
        print(f"Device: {self.device}")
    
    def _load_model(self):
        """Load the trained model from checkpoint."""
        try:
            # Try to load as a full checkpoint first
            # Note: Using weights_only=False because we trust our checkpoint files
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            
            if 'model_config' in checkpoint:
                # Load from full checkpoint
                model_config = checkpoint['model_config']
                model = load_model_checkpoint(
                    checkpoint_path=self.model_path,
                    model_name=model_config['model_name'],
                    num_classes=model_config['num_classes'],
                    device=self.device
                )
            else:
                # Load from state dict only
                model = load_model_checkpoint(
                    checkpoint_path=self.model_path,
                    model_name=self.config.model.name,
                    num_classes=self.config.model.num_classes,
                    device=self.device
                )
            
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def _load_class_names(self) -> List[str]:
        """Load class names from metadata."""
        try:
            # Try to load from processed data metadata
            metadata_path = Path(self.config.get_processed_data_path()) / "metadata" / "class_mapping.csv"
            
            if metadata_path.exists():
                import pandas as pd
                df = pd.read_csv(metadata_path)
                class_mapping = dict(zip(df['class_id'], df['class_name']))
                return [class_mapping[i] for i in range(len(class_mapping))]
            else:
                # Fallback to generic names
                return [f"class_{i}" for i in range(self.config.model.num_classes)]
                
        except Exception:
            # Final fallback
            return [f"class_{i}" for i in range(self.config.model.num_classes)]
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image tensor
        """
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Apply transforms
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image']
        
        # Add batch dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image: PIL Image
            
        Returns:
            Prediction results dictionary
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get all class probabilities
            all_probabilities = probabilities[0].cpu().numpy()
            
            # Create result
            result = {
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_id': predicted_class,
                'confidence': float(confidence),
                'all_probabilities': {
                    class_name: float(prob)
                    for class_name, prob in zip(self.class_names, all_probabilities)
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


# Initialize FastAPI app
app = FastAPI(
    title="Wafer Defect Classification API",
    description="API for classifying defects in wafer surface images using deep learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor: Optional[WaferDefectPredictor] = None


@app.on_event("startup")
async def load_model():
    """Load model on startup."""
    global predictor
    
    # Model path - adjust as needed
    model_path = Path(__file__).parent.parent / "models" / "final_model.pth"
    
    if not model_path.exists():
        # Try checkpoint directory
        model_path = Path(__file__).parent.parent / "checkpoints" / "best_checkpoint.pth"
    
    if not model_path.exists():
        raise RuntimeError(f"Model not found at {model_path}")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        predictor = WaferDefectPredictor(
            model_path=str(model_path),
            device=device
        )
        print("Model loaded successfully!")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize predictor: {e}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Wafer Defect Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "POST - Predict defect class for uploaded image",
            "/predict/batch": "POST - Predict defect classes for multiple images",
            "/health": "GET - Health check",
            "/info": "GET - Model information"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": predictor is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info")
async def model_info():
    """Get model information."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": predictor.model_path,
        "device": predictor.device,
        "num_classes": len(predictor.class_names),
        "class_names": predictor.class_names,
        "input_size": [224, 224],
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict defect class for uploaded image.
    
    Args:
        file: Uploaded image file
        
    Returns:
        Prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Make prediction
        result = predictor.predict(image)
        
        # Add file info
        result['file_info'] = {
            'filename': file.filename,
            'size': len(image_data),
            'format': image.format
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(files: List[UploadFile] = File(...)):
    """
    Predict defect classes for multiple images.
    
    Args:
        files: List of uploaded image files
        
    Returns:
        Batch prediction results
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                'filename': file.filename,
                'error': 'File is not an image'
            })
            continue
        
        try:
            # Read and process image
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data))
            
            # Make prediction
            result = predictor.predict(image)
            result['filename'] = file.filename
            
            results.append(result)
            
        except Exception as e:
            results.append({
                'filename': file.filename,
                'error': str(e)
            })
    
    return {
        'batch_results': results,
        'total_files': len(files),
        'successful_predictions': len([r for r in results if 'error' not in r]),
        'timestamp': datetime.now().isoformat()
    }


# Development server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )