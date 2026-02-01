# Wafer Surface Defect Detection

A comprehensive deep learning system for classifying defects in semiconductor wafer surface images using PyTorch and FastAPI.

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)

## Project Overview

This project implements a production-ready computer vision system for detecting and classifying defects on semiconductor wafer surfaces. The system uses transfer learning with pre-trained CNN models (EfficientNet, ResNet) to achieve high accuracy in defect classification tasks.

### Key Features

- **Transfer Learning**: Leverages pre-trained CNN architectures (EfficientNet, ResNet, MobileNet)
- **Data Augmentation**: Comprehensive augmentation pipeline for robust training
- **Class Imbalance Handling**: Weighted loss functions and sampling strategies
- **Production API**: FastAPI-based REST API for inference
- **Monitoring**: TensorBoard integration for training visualization
- **Containerization**: Docker support for easy deployment
- **Comprehensive Evaluation**: Accuracy, F1-score, confusion matrices, and per-class metrics

### Dataset Selection

After evaluating both suggested Kaggle datasets, I selected the **Mixed-type Wafer Defect Dataset** for the following reasons:

1. **Real Image Data**: Contains actual wafer surface images rather than binary maps
2. **Multiple Defect Types**: Diverse defect categories for robust classification
3. **Industrial Relevance**: Closer to real-world manufacturing inspection scenarios
4. **CNN Compatibility**: Image format works optimally with CNN architectures

## Project Architecture

```
wafer-surface-defect-detection/
├── src/                    # Source code
│   ├── data/              # Data loading and preprocessing
│   ├── models/            # Model architectures
│   ├── utils/             # Training, metrics, visualization
│   └── config.py          # Configuration management
├── api/                   # FastAPI application
├── config/                # Configuration files
├── data/                  # Dataset storage
│   ├── raw/              # Original dataset
│   └── processed/        # Preprocessed data
├── models/                # Saved model checkpoints
├── logs/                  # Training logs and TensorBoard
├── notebooks/             # Jupyter notebooks for experiments
├── tests/                 # Unit tests
├── train.py               # Main training script
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Multi-container setup
└── requirements.txt       # Python dependencies
```

## Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)
- Docker (for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd wafer-surface-defect-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Kaggle API and download dataset**
   ```bash
   # Method 1: Using environment variable (recommended)
   export KAGGLE_API_TOKEN=your_api_token_here
   python download_dataset.py
   
   # Method 2: Using kaggle.json file
   # Get API token from https://www.kaggle.com/account
   # Save to ~/.kaggle/kaggle.json and chmod 600
   python download_dataset.py
   ```

5. **Train the model**
   ```bash
   python train.py --data-path data/raw
   ```

6. **Start API server**
   ```bash
   cd api
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   # For production
   docker-compose up -d

   # For development (includes Jupyter)
   docker-compose --profile development up -d

   # For monitoring (includes TensorBoard)
   docker-compose --profile monitoring up -d
   ```

2. **Access services**
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Jupyter (dev): http://localhost:8888 (token: wafer-defect-detection)
   - TensorBoard (monitoring): http://localhost:6006

## Model Performance

### Architecture Details

- **Backbone**: EfficientNet-B0 (configurable)
- **Pre-training**: ImageNet weights
- **Input Size**: 224×224×3
- **Classes**: 6 defect types (varies by dataset)
- **Parameters**: ~5.3M total, ~1.2M trainable

### Training Configuration

- **Optimizer**: Adam (lr=0.001, weight_decay=0.0001)
- **Scheduler**: Cosine Annealing
- **Batch Size**: 32
- **Data Augmentation**: Rotation, flip, brightness/contrast, blur, noise
- **Early Stopping**: Patience=10 epochs

### Expected Performance

*Note: Actual results will vary based on your specific dataset*

- **Training Accuracy**: 95-98%
- **Validation Accuracy**: 92-95%
- **Test Accuracy**: 90-93%
- **Macro F1-Score**: 0.88-0.92
- **Training Time**: ~2-4 hours (RTX 3080)

## Configuration

The system uses YAML configuration files in the `config/` directory:

```yaml
# config/config.yaml
data:
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  batch_size: 32
  image_size: [224, 224]

model:
  name: "efficientnet_b0"
  pretrained: true
  dropout: 0.2

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "cosine"
```

## API Usage

### Single Image Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@wafer_image.jpg"
```

```json
{
  "predicted_class": "scratch",
  "predicted_class_id": 2,
  "confidence": 0.94,
  "all_probabilities": {
    "normal": 0.02,
    "crack": 0.01,
    "scratch": 0.94,
    "stain": 0.02,
    "spot": 0.01
  },
  "timestamp": "2024-12-06T10:30:45.123456"
}
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/predict/batch" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "files=@image1.jpg" \
     -F "files=@image2.jpg"
```

### Python Client Example

```python
import requests

# Single prediction
with open("wafer_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )
    result = response.json()
    print(f"Prediction: {result['predicted_class']} ({result['confidence']:.2f})")
```

## Development & Experimentation

### Jupyter Notebooks

#### Important note: This is currently WIP!

The `notebooks/` directory contains example notebooks:

- `01_data_exploration.ipynb`: Dataset analysis and visualization
- `02_model_training.ipynb`: Interactive training and evaluation
- `03_inference_examples.ipynb`: Model inference examples
- `04_performance_analysis.ipynb`: Detailed performance analysis

### Experiment with Different Models

```python
# Train with different architectures
python train.py --data-path data/raw --config config/resnet50_config.yaml
python train.py --data-path data/raw --config config/mobilenet_config.yaml
```

### Custom Configuration

Create custom configuration files for different experiments:

```yaml
# config/experiment_config.yaml
model:
  name: "resnet50"
  freeze_backbone: true  # Feature extraction only

training:
  epochs: 30
  learning_rate: 0.0001  # Lower LR for frozen backbone
```

## Testing

#### Important note: Tests are still WIP!

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_models.py
python -m pytest tests/test_data_loading.py

# Run with coverage
python -m pytest --cov=src tests/
```

## Deployment

### Production Deployment

1. **Configure environment variables**
   ```bash
   export MODEL_PATH=/path/to/best_model.pth
   export API_HOST=0.0.0.0
   export API_PORT=8000
   ```

2. **Deploy with Docker Compose**
   ```bash
   docker-compose --profile production up -d
   ```

3. **Health monitoring**
   ```bash
   curl http://localhost:8000/health
   ```

### Scaling Considerations

- **Load Balancing**: Use nginx upstream for multiple API instances
- **GPU Optimization**: Configure CUDA_VISIBLE_DEVICES for multi-GPU setups
- **Model Optimization**: Consider ONNX/TensorRT for faster inference
- **Caching**: Implement Redis for prediction caching

## Monitoring & Logging

### TensorBoard

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

### Production Monitoring

- **Health Checks**: Built-in `/health` endpoint
- **Metrics Collection**: Prometheus-compatible metrics (optional)
- **Log Aggregation**: Structured JSON logging
- **Error Tracking**: Sentry integration (configurable)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Style

- Use Black for code formatting: `black src/ api/ tests/`
- Follow PEP 8 guidelines
- Add type hints for new functions
- Write docstrings for public methods

## Performance Optimization

### Training Optimizations

1. **Mixed Precision Training**: Enable AMP for faster training
2. **Data Loading**: Increase `num_workers` based on CPU cores
3. **Batch Size**: Maximize based on GPU memory
4. **Model Architecture**: Try different EfficientNet variants

### Inference Optimizations

1. **Model Quantization**: Post-training quantization for mobile deployment
2. **ONNX Export**: Convert to ONNX for cross-platform inference
3. **TensorRT**: NVIDIA TensorRT for GPU acceleration
4. **Batch Inference**: Process multiple images simultaneously

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Reduce batch size in config.yaml
   data:
     batch_size: 16  # Reduce from 32
   ```

2. **Slow Data Loading**
   ```python
   # Increase number of workers
   data:
     num_workers: 8  # Increase based on CPU cores
   ```

3. **Model Not Loading**
   ```bash
   # Check model path and compatibility
   python -c "import torch; print(torch.load('models/final_model.pth', map_location='cpu'))"
   ```

4. **API Startup Issues**
   ```bash
   # Check model availability
   docker logs wafer-api
   ```

## Requirements

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU-only training possible
- **Recommended**: 16GB RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB RAM, RTX 3080/4080 or Tesla V100+

### Software Dependencies

See `requirements.txt` for complete list. Key dependencies:

- PyTorch 2.0+
- torchvision 0.15+
- FastAPI 0.100+
- Albumentations 1.3+
- Pandas, NumPy, Matplotlib
- Docker (for containerization)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- FastAPI creators for the modern Python web framework
- Kaggle community for providing datasets
- EfficientNet authors for the efficient model architecture

## References

1. Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition.
3. Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection.
4. Howard, A., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.