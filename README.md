# MaFIR - Fisheye Image Rectification via Manhattan Attention

<p>
    <a href='https://pytorch.org/' target="_blank"><img src="https://img.shields.io/badge/PyTorch-1.10.0-red"></a>
    <a href='https://www.python.org/' target="_blank"><img src="https://img.shields.io/badge/Python-3.8+-blue"></a>
</p>


> **MaFIR: High-Fidelity Fisheye Image Rectification via Manhattan Attention and Dynamic Feature Optimization**  

## Overview

MaFIR is a deep learning-based fisheye image rectification system that employs a two-stage training approach:
- **Pre-training Stage**: Learns distortion-aware representations from fisheye images
- **Fine-tuning Stage**: Learns pixel-wise flow mapping for image rectification


![总体框架](https://github.com/user-attachments/assets/c17c5b67-95d5-4e8c-92a8-133040b1d282)





## Project Structure

```
MaFIR/
├── core/                   # Core modules
│   ├── dataset.py          # Dataset processing
│   ├── loss.py             # Loss functions
│   ├── metric.py           # Evaluation metrics
│   ├── trainer.py          # Trainer
│   └── utils.py            # Utility functions
├── pre_training/           # Pre-training module
│   ├── main.py             # Main training script
│   ├── models_pre.py       # Pre-training models
│   └── dataset.py          # Pre-training dataset
├── fine-tuning/            # Fine-tuning module
│   ├── main.py             # Main training script
│   ├── models_fine.py      # Fine-tuning models
│   └── dataset.py          # Fine-tuning dataset
├── data_prepare/           # Data preparation
├── dataset1/               # Dataset 1
├── dataset2/               # Dataset 2
└── requirements            # Dependencies list
```

## Requirements

- Python 3.8+
- PyTorch 1.10.0+ (CUDA 11.3)
- torchvision 0.11.1+
- Other dependencies listed in `core/requirements`

### Install Dependencies

```bash
pip install -r requirements
```

## Quick Start

### 1. Data Preparation

Prepare training data including:
- Source images (without distortion)
- Synthetic fisheye images
- Corresponding distortion labels

### 2. Pre-training

Start pre-training with the following command:

```bash
cd pre_training
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1285 main.py
```

Parameters:
- `--nproc_per_node`: Number of GPUs
- `--master_port`: Master port (can be randomly selected)

Pre-trained models will be saved in `pre_training/save/net/` directory.

### 3. Fine-tuning

Place pre-trained models in `fine-tuning/pretrain/` directory, then run:

```bash
cd fine-tuning
python -m torch.distributed.launch --nproc_per_node=2 --master_port 1285 main.py
```

Fine-tuned models will be saved in `fine-tuning/save/net/` directory.

### 4. Testing

Evaluate model performance using test scripts:

```bash
cd fine-tuning
python test.py    # Synthetic fisheye image testing
python test_for_real.py   # Real fisheye image testing
```

## Model Architecture

This project is based on Vision Transformer (ViT) architecture, including:

- **Patch Embedding**: Image patch embedding
- **Transformer Blocks**: Self-attention mechanism
- **Positional Encoding**: Position encoding
- **Decoder**: For generating rectification results

## Evaluation Metrics

Use the following metrics to evaluate model performance:

- **PSNR** (Peak Signal-to-Noise Ratio)
- **SSIM** (Structural Similarity Index)

Run evaluation script:

```bash
cd core
python compare.py
```

## Datasets

The project supports various dataset formats, including:
- Synthetic fisheye image datasets
- Real fisheye image datasets
- Custom datasets

### Dataset Structure

```
dataset1/
├── data/           # Fisheye images
└── ddm/            # Distortion labels (pre-training)

dataset2/
├── data/           # Fisheye images
└── flow/           # Flow mapping labels (fine-tuning)
```

## Custom Training

### Parameter Configuration

Customize training process by modifying parameters in training scripts:

```python
# Main parameters
parser.add_argument('--batchSize', type=int, default=32)    # Batch size
parser.add_argument('--lr', type=float, default=0.0001)     # Learning rate
parser.add_argument('--num_epochs', type=int, default=65)   # Training epochs
```

### Custom Dataset

Create custom dataset class by inheriting from `torch.utils.data.Dataset`:

```python
class CustomDataset(Dataset):
    def __init__(self, data_args):
        # Initialization logic
        pass
    
    def __getitem__(self, index):
        # Return image and label
        pass
```


## License

This project is for academic research use only. For commercial use, please contact the authors.
