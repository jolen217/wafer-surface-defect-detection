# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A PyTorch deep learning system for classifying defects on semiconductor wafer surfaces. Transfer learning from ImageNet (EfficientNet, ResNet, MobileNet, ViT) feeds into a custom 3-layer MLP head. A FastAPI server wraps the trained model for production inference. The dataset is the Mixed-type Wafer Defect Dataset from Kaggle — real RGB wafer surface photos organized by defect class.

## Commands

**Environment setup**
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python test_setup.py          # validates imports, structure, config loading, and model creation
```

**Dataset download** (requires Kaggle API token)
```bash
export KAGGLE_API_TOKEN=your_token_here
python download_dataset.py
```

**Training**
```bash
python train.py --data-path data/raw                          # standard run
python train.py --data-path data/raw --config config/config.yaml
python train.py --data-path data/raw --resume checkpoints/latest_checkpoint.pth
python train.py --data-path data/raw --dry-run               # setup check, no training
```

**Standalone inference** (no API server needed)
```bash
python inference.py --model-path checkpoints/best_checkpoint.pth --image path/to/image.jpg
python inference.py --model-path checkpoints/best_checkpoint.pth --image-dir path/to/dir/ --output results.json
```

**API server**
```bash
cd api && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
# Docs: http://localhost:8000/docs
```

**Monitoring**
```bash
tensorboard --logdir logs/ --host 0.0.0.0 --port 6006
```

**Docker**
```bash
docker-compose up -d                          # API only
docker-compose --profile development up -d    # + Jupyter on :8888 (token: wafer-defect-detection)
docker-compose --profile monitoring up -d     # + TensorBoard on :6006
docker-compose --profile production up -d     # + nginx reverse proxy
```

**Code style**
```bash
black src/ api/
flake8 src/ api/
```

## Architecture

### End-to-end data flow

```
data/raw/{class_name}/*.jpg
        ↓  preprocess_wafer_dataset() on first train.py run
data/processed/
  train/{class_name}/   val/{class_name}/   test/{class_name}/
  metadata/class_mapping.csv   ← used by dataset, API, and inference script
        ↓  WaferDefectDataset + Albumentations
   224×224 tensors (ImageNet-normalized)
        ↓  WaferDefectClassifier
   backbone features → 3-layer MLP → class logits
        ↓  WaferDefectTrainer
   checkpoints/best_checkpoint.pth   checkpoints/latest_checkpoint.pth
   checkpoints/epoch_NNN_acc_XX.XX_loss_X.XXXX.pth   (one per epoch)
        ↓  api/main.py or inference.py
   JSON predictions
```

### `src/config.py` — Config

`Config` loads `config/config.yaml` into six typed dataclasses: `DataConfig`, `ModelConfig`, `TrainingConfig`, `AugmentationConfig`, `LoggingConfig`, `DeviceConfig`. After preprocessing, `train.py` calls `config.update_num_classes(n)` and `config.save_config()` to write the real class count back to the YAML file.

### `src/data/preprocessing.py` — DataPreprocessor

`preprocess_wafer_dataset()` (called by `train.py`) runs the full pipeline: scans raw directory for images grouped by parent folder name → stratified train/val/test split via sklearn → copies files into `data/processed/split/class/` → writes `metadata/class_mapping.csv` (columns: `class_name`, `class_id`). Class IDs are assigned alphabetically by class name.

### `src/data/dataset.py` — Datasets and loaders

`WaferDefectDataset` reads `metadata/class_mapping.csv` (falls back to scanning the split directory). Training split gets Albumentations augmentation (horizontal/vertical flip, rotation, brightness/contrast, blur, Gaussian noise); val/test get only resize + ImageNet normalization. `get_class_weights()` returns inverse-frequency weights used by the loss function.

`InferenceDataset` accepts an arbitrary list of image paths without labels — used by `inference.py` for standalone batch prediction.

`create_data_loaders()` returns `(train_loader, val_loader, test_loader)` as a tuple; training loader uses `drop_last=True`.

### `src/models/classifier.py` — Model

`WaferDefectClassifier` wraps a configurable backbone:
- ResNet variants → via `torchvision.models`, final FC removed
- EfficientNet/ViT → via `timm.create_model(..., num_classes=0)`
- MobileNet → via `torchvision.models`, classifier replaced with `nn.Identity()`

Feature dimension is auto-detected by running a dummy `[1, 3, 224, 224]` tensor through the backbone. The classifier head is always `dropout → Linear(feat, 512) → ReLU → dropout → Linear(512, 256) → ReLU → dropout/2 → Linear(256, num_classes)`.

`EnsembleClassifier` wraps a list of trained `WaferDefectClassifier` instances and returns their weighted-average softmax outputs.

Checkpoints store `model_config` dict (`model_name`, `num_classes`, `dropout`) so the architecture can be reconstructed at load time without knowing it in advance.

### `src/utils/training.py` — Trainer

`WaferDefectTrainer` supports three loss types: `cross_entropy` (with optional inverse-frequency class weights), `focal` (`FocalLoss` with configurable alpha/gamma), and `label_smoothing`. Optimizer choices: `adam`, `adamw`, `sgd`. Scheduler choices: `cosine` (CosineAnnealingLR), `step` (StepLR), `plateau` (ReduceLROnPlateau).

Each epoch saves a named checkpoint (`epoch_NNN_acc_XX.XX_loss_X.XXXX.pth`), overwrites `latest_checkpoint.pth`, and overwrites `best_checkpoint.pth` when val accuracy improves. `EarlyStopping` monitors val loss and optionally restores best weights on trigger.

TensorBoard scalars logged: `Loss/Train`, `Loss/Validation`, `Accuracy/Train`, `Accuracy/Validation`, `Learning_Rate`, plus all scalar entries from the validation metrics dict.

### `src/utils/metrics.py` — MetricsCalculator

Computes accuracy, macro/micro/weighted precision/recall/F1, per-class P/R/F1, confusion matrix, sklearn classification report, ROC-AUC (OVR + OVO for multiclass), and average precision. Optionally plots confusion matrices and per-class bar charts.

### `src/utils/visualization.py` — TrainingVisualizer

Plots training/validation loss and accuracy curves (3-panel: loss, accuracy, LR schedule), grid of prediction examples with color-coded correct/incorrect labels, class distribution bars, and multi-experiment metric comparison. `create_evaluation_report()` saves confusion matrix, normalized confusion matrix, per-class metrics, and class distribution PNGs to a directory.

### `api/main.py` — FastAPI inference server

`WaferDefectPredictor` loads a checkpoint on startup. Model path priority: `models/final_model.pth` → `checkpoints/best_checkpoint.pth`. If the checkpoint contains `model_config`, architecture is reconstructed from it; otherwise falls back to `config.yaml`.

Endpoints:
- `POST /predict` — single image upload, returns `predicted_class`, `predicted_class_id`, `confidence`, `all_probabilities`, `timestamp`
- `POST /predict/batch` — up to 50 files, returns per-file results with aggregate counts
- `GET /health` — liveness check
- `GET /info` — model metadata and class names

## Key behaviors to know

**`num_classes` coupling**: `config.yaml` is rewritten with the actual discovered class count after the first preprocessing run. If you change datasets or delete `data/processed/`, the next `train.py` run will re-preprocess and update the config automatically.

**Class imbalance**: training always uses inverse-frequency class weights (via `get_class_weights()`) passed to `CrossEntropyLoss`. This is hardcoded in `train.py`'s `training_config` dict, not controllable from `config.yaml`.

**Backbone feature dim inference**: creating a `WaferDefectClassifier` always runs one dummy forward pass to detect the backbone output size. This happens at import time of the model — expect a brief delay and ensure the correct device is set beforehand for large ViT models.

**Notebooks and tests are WIP**: the `notebooks/` and `tests/` directories do not yet contain content. `test_setup.py` at the project root is the current smoke-test.
