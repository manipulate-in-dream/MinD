# MinD Setup Guide

This guide provides instructions for setting up the MinD project with proper path configuration for reproducible research.

## Overview

MinD is a dual-system world model for robotics that combines video generation and action prediction. This setup guide ensures all paths are properly configured for your environment.

## Prerequisites

- Python >= 3.8
- PyTorch >= 2.0
- CUDA Toolkit >= 12.1
- Git

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/manipulate-in-dream/MinD.git
cd MinD
```

### 2. Create Python Environment

```bash
# Using conda
conda create -n mind python=3.8
conda activate mind

# Or using venv
python -m venv mind_env
source mind_env/bin/activate  # On Windows: mind_env\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Paths

#### Option A: Using Environment Variables (Recommended)

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` file with your paths:
```bash
# Base paths
MIND_DATA_ROOT=/path/to/your/data
MIND_MODEL_ROOT=/path/to/your/models
MIND_OUTPUT_ROOT=/path/to/your/outputs
MIND_DATASET_ROOT=/path/to/your/datasets

# Model checkpoints
DYNAMICRAFTER_CHECKPOINT=/path/to/dynamicrafter/checkpoint.ckpt
VPP_MODEL_PATH=/path/to/vpp/model
ACTION_MODEL_FOLDER=/path/to/action/models

# Dataset paths
RTX_DATASET_PATH=/path/to/rtx/dataset
CALVIN_DATASET_PATH=/path/to/calvin/dataset
```

3. Load environment variables:
```bash
source .env  # On Unix/Linux/Mac
# Or use python-dotenv in your scripts
```

#### Option B: Using config.yaml

Edit `config.yaml` to set default paths:
```yaml
data_root: ./data
model_root: ./checkpoints
output_root: ./outputs
dataset_root: ./data/datasets
```

### 5. Download Model Checkpoints

Create the checkpoint directories:
```bash
mkdir -p checkpoints/dynamicrafter
mkdir -p checkpoints/action_models
mkdir -p checkpoints/vla_models
```

Download required checkpoints:
- DynamiCrafter checkpoint
- Action model checkpoint
- VLA model checkpoint
- Dataset statistics JSON

Place them in the appropriate directories or update your `.env` file with their locations.

### 6. Prepare Datasets

Set up your datasets according to the structure expected by the models:
```
$MIND_DATASET_ROOT/
├── rtx_dataset/
├── calvin/
├── rlbench/
└── custom_datasets/
```

## Running the Code

### Remote Inference Server

```bash
# Set environment variables
export DYNAMICRAFTER_CHECKPOINT=/path/to/checkpoint.ckpt
export MIND_FULL_CHECKPOINT=/path/to/full_checkpoint.pt
export DATASET_STATISTICS_JSON=/path/to/stats.json

# Run the server
python remote_infer.py
```

### Training

Use the improved training script with configurable paths:
```bash
# Set your paths
export MIND_DATASET_ROOT=/path/to/datasets
export DYNAMICRAFTER_CHECKPOINT=/path/to/checkpoint.ckpt

# Run training
bash scripts/train_vdmvla_franka_oxe_improved.sh
```

### Using VLA Models

```bash
# Set model paths
export ACTION_MODEL_FOLDER=/path/to/action/models
export VIDEO_MODEL_PATH=/path/to/video/model

# Run VLA
python vla/vppvla.py --input_image_path /path/to/image.png
```

## Path Configuration System

The project uses a centralized path configuration system (`config.py`) that:

1. **Priority Order**: Environment variables > config.yaml > defaults
2. **Automatic Directory Creation**: Required directories are created automatically
3. **Path Validation**: Validates paths exist before use
4. **Cross-Platform**: Works on Windows, macOS, and Linux

### Using in Your Code

```python
from config import get_path_config

# Get configuration instance
config = get_path_config()

# Access paths
model_path = config.get_model_path('video', 'dynamicrafter')
dataset_path = config.get_dataset_path('calvin')

# Resolve relative paths
absolute_path = config.resolve_path('./relative/path')

# Validate paths
config.validate_path(model_path, must_exist=True)
```

## Troubleshooting

### Common Issues

1. **Path not found errors**
   - Check your `.env` file or environment variables
   - Ensure all required files are downloaded
   - Use absolute paths instead of relative paths

2. **CUDA out of memory**
   - Reduce batch size in config
   - Set `CUDA_VISIBLE_DEVICES` to limit GPU usage

3. **Import errors**
   - Ensure all dependencies are installed
   - Check that `PYTHONPATH` includes the project root

4. **Permission denied**
   - Ensure you have write permissions for output directories
   - Check file ownership and permissions

### Debug Mode

Enable detailed logging:
```bash
export LOG_LEVEL=DEBUG
python your_script.py
```

## Project Structure

```
MinD/
├── config.py              # Centralized path configuration
├── config.yaml            # Default configuration
├── .env.example          # Example environment variables
├── remote_infer.py       # Inference server (fixed)
├── vla/
│   ├── vppvla.py        # VPP VLA model (fixed)
│   └── vgmactvla.py     # VGM ACT VLA model (fixed)
├── scripts/
│   └── train_vdmvla_franka_oxe_improved.sh  # Training script (fixed)
├── DynamiCrafter/
│   └── configs/
│       └── ww_training_128_4frame_v1.0/
│           └── config_improved.yaml  # DynamiCrafter config (fixed)
└── checkpoints/          # Model checkpoints (create this)
```

## Contributing

When adding new scripts or modules:
1. Use the `config.py` module for all path operations
2. Support environment variable overrides
3. Provide sensible defaults
4. Add path validation
5. Update this documentation

## Support

For issues related to path configuration:
1. Check this documentation first
2. Verify your environment variables
3. Look for error messages in logs
4. Open an issue with details about your setup

## Paper Repository Standards

This setup follows best practices for research reproducibility:
- **No hardcoded paths**: All paths are configurable
- **Environment isolation**: Use virtual environments
- **Clear documentation**: Step-by-step setup instructions
- **Version control**: Track configuration files (except `.env`)
- **Portable across systems**: Works on different OS and hardware
- **Reproducible results**: Same configuration yields same results