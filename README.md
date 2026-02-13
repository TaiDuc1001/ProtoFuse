# ViFE: Learning Fine-Grained Features via Self-Supervised Distillation for Few-Shot Image Classification

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F16-green.svg)](https://github.com/openai/CLIP)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official implementation** of *"Learning Fine-Grained Features via Self-Supervised Distillation for Few-Shot Image Classification"*

ViFE (Visual Fine-grained Extractor) extends APT (Adapted Prompt Tuning) by introducing a self-supervised distillation pipeline that learns fine-grained visual features from unlabeled data. The method combines CLIP-based cross-attention prompt tuning with DINO-style self-supervised learning and a learned fusion mechanism, achieving state-of-the-art performance on fine-grained few-shot classification tasks.

## Installation

```bash
git clone https://github.com/TaiDuc1001/Visual-Finegrained-Extractor.git
cd Visual-Finegrained-Extractor
uv sync
```

## Dataset Preparation

### CUB-200-2011

1. Download the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset
2. Extract and place it under `datasets/cub-200-2011/`
3. The directory should follow this structure:

```
datasets/cub-200-2011/
├── class_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_002/
│   └── ...
└── ...
```

## Usage

### Training

```bash
# ViFE (proposed method)
uv run apt.py --config configs/vife.yaml

# APT (baseline)
uv run apt.py --config configs/apt.yaml

# CoOp
uv run coop.py --config configs/coop.yaml

# MaPLe
uv run maple.py --config configs/maple.yaml
```

### Configuration Overrides

All config values can be overridden from the command line:

```bash
uv run apt.py --config configs/vife.yaml \
    --data.kshot 8 \
    --training.epochs 20 \
    --training.batch_size 16 \
    --training.device cuda:1
```

### Zero-Shot Evaluation

```bash
uv run test/zeroshot_clip.py
```

### Computational Analysis

Compare learnable parameters, GFLOPs, FPS, and latency across all methods:

```bash
uv run test/compare_methods.py
```

## Project Structure

```
Visual-Finegrained-Extractor/
├── apt.py              # APT & ViFE training pipeline
├── apt_ssl.py          # SSL modules (DINO, TransformerAdapter, FusionWeightLearner)
├── coop.py             # CoOp implementation
├── cocoop.py           # CoCoOp implementation
├── maple.py            # MaPLe implementation
├── utils.py            # Config, checkpointing, metrics, visualization
├── logger.py           # Logging utilities
├── configs/
│   ├── apt.yaml        # APT configuration
│   ├── vife.yaml       # ViFE configuration
│   ├── coop.yaml       # CoOp configuration
│   ├── cocoop.yaml     # CoCoOp configuration
│   └── maple.yaml      # MaPLe configuration
├── fixmatch/
│   ├── fixmatch_apt.py             # FixMatch + APT
│   ├── fixmatch_coop.py            # FixMatch + CoOp
│   ├── fixmatch_cocoop.py          # FixMatch + CoCoOp
│   ├── fixmatch_maple.py           # FixMatch + MaPLe
│   ├── fixmatch_utils.py           # FixMatch utilities
│   └── configs/
│       ├── _fixmatch_apt_config.yaml
│       ├── _fixmatch_coop_config.yaml
│       ├── _fixmatch_cocoop_config.yaml
│       └── _fixmatch_maple_config.yaml
├── test/
│   ├── compare_methods.py          # Computational analysis (params, GFLOPs, FPS)
│   └── zeroshot_clip.py            # Zero-shot CLIP evaluation
├── checkpoints/                    # Cached training checkpoints
├── outputs/                        # Training logs and results
├── models/                         # Pre-trained CLIP weights
├── datasets/                       # Dataset root
└── pyproject.toml                  # Project dependencies
```

## Supported Methods

| Method | Script | Description |
|--------|--------|-------------|
| **ViFE** | `apt.py` | APT + self-supervised distillation with learned fusion (proposed) |
| **APT** | `apt.py` | Cross-attention prompt tuning on CLIP |
| **CoOp** | `coop.py` | Context Optimization for CLIP |
| **MaPLe** | `maple.py` | Multi-modal Prompt Learning for CLIP |

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
