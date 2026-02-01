# Kaggle Dataset Setup Guide

## Quick Start

1. **Install Kaggle API**:
   ```bash
   pip install kaggle
   ```

2. **Setup Kaggle API credentials** (choose one method):

   **Method 1: Environment Variable (Recommended)**
   ```bash
   # Set your Kaggle API token as environment variable
   export KAGGLE_API_TOKEN=your_api_token_here
   
   # Add to your shell profile for persistence
   echo 'export KAGGLE_API_TOKEN=your_api_token_here' >> ~/.bashrc
   # or for zsh:
   echo 'export KAGGLE_API_TOKEN=your_api_token_here' >> ~/.zshrc
   ```
   
   **Method 2: Configuration File**
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
   - Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. **Download the dataset**:
   ```bash
   # Download Mixed-type Wafer Defect Dataset (recommended)
   python download_dataset.py
   
   # Or download WM-811K Wafer Map Dataset
   python download_dataset.py --wm811k
   
   # Custom dataset
   python download_dataset.py --custom-dataset username/dataset-name
   ```

## Usage Examples

```bash
# Basic download to data/raw (default)
python download_dataset.py

# Download to custom directory
python download_dataset.py --output-dir my_data/

# Force re-download even if exists
python download_dataset.py --force

# Keep temporary download files
python download_dataset.py --keep-download

# Only validate existing dataset
python download_dataset.py --validate-only
```

## Troubleshooting

### 1. Kaggle API not found
```bash
pip install kaggle
```

### 2. API credentials error
**For Environment Variable method:**
- Check if variable is set: `echo $KAGGLE_API_TOKEN`
- Ensure token is valid and not expired
- Get token from https://www.kaggle.com/account

**For File method:**
- Ensure `~/.kaggle/kaggle.json` exists
- Check file permissions: `chmod 600 ~/.kaggle/kaggle.json`
- Verify JSON format:
  ```json
  {"username":"your_username","key":"your_api_key"}
  ```

### 3. Dataset not found
- Verify the dataset exists on Kaggle
- Check dataset name spelling
- Ensure you have access to the dataset

### 4. Download timeout
- Check internet connection
- Try again later
- Use `--force` to restart interrupted downloads

### 5. Token vs File priority
- Script checks `KAGGLE_API_TOKEN` environment variable first
- Falls back to `~/.kaggle/kaggle.json` if no token found
- Both methods work, use whichever is more convenient

## Supported Datasets

1. **Mixed-type Wafer Defect Dataset** (Default)
   - ID: `co1d7era/mixedtype-wafer-defect-datasets`
   - Real wafer surface images with multiple defect types
   - Best for CNN classification

2. **WM-811K Wafer Map Dataset**
   - ID: `qingyi/wm811k-wafer-map`
   - Wafer maps with defect patterns
   - Binary/categorical data format

## What the script does

1. **Downloads** the dataset from Kaggle
2. **Extracts** ZIP files automatically  
3. **Organizes** images into class directories
4. **Validates** image integrity
5. **Reports** class distribution
6. **Cleans up** temporary files

## Output Structure

```
data/raw/
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class_2/
│   ├── image3.jpg
│   └── ...
└── ...
```

This structure is ready for the training pipeline!