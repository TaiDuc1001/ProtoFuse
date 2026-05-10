# ProtoFuse

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9+-ee4c2c.svg)](https://pytorch.org/)
[![CLIP](https://img.shields.io/badge/CLIP-ViT--B%2F16-green.svg)](https://github.com/openai/CLIP)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/)

Few-shot CLIP adaptation experiments for fine-grained image classification. The repository includes training and evaluation pipelines for ProtoFuse, APT, CoOp, MaPLe, APE, TIMO, TIMOS, and Tip-Adapter.

## Installation

```bash
git clone https://github.com/TaiDuc1001/protofuse.git
cd protofuse
uv sync
```

## Dataset Preparation

### CUB-200-2011

1. Download the [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/) dataset.
2. Extract and place it under `datasets/cub-200-2011/`.
3. The directory should follow this structure:

```text
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
# ProtoFuse
uv run protofuse.py --config configs/protofuse.yaml

# APT
uv run apt.py --config configs/apt.yaml

# CoOp
uv run coop.py --config configs/coop.yaml

# MaPLe
uv run maple.py --config configs/maple.yaml

# Tip-Adapter / Tip-Adapter-F
uv run tip_adapter.py --config configs/tip_adapter.yaml

# TIMO
uv run timo.py --config configs/timo.yaml

# TIMOS
uv run timos.py --config configs/timos.yaml
```

### Configuration Overrides

All config values can be overridden from the command line:

```bash
uv run protofuse.py --config configs/protofuse.yaml \
    --data.kshot 8 \
    --training.batch_size 16 \
    --training.device cuda:1
```

### Zero-Shot Evaluation

```bash
uv run test/zeroshot_clip.py
```

### Computational Analysis

Compare learnable parameters, GFLOPs, FPS, and latency across the supported baseline methods:

```bash
uv run test/compare_methods.py
```

## Project Structure

```text
protofuse/
├── protofuse.py        # ProtoFuse pipeline entry point
├── apt.py              # APT training entry point
├── coop.py             # CoOp training entry point
├── maple.py            # MaPLe training entry point
├── tip_adapter.py      # Tip-Adapter training entry point
├── timo.py             # TIMO training-free entry point
├── timos.py            # TIMOS training-free entry point
├── utils.py            # Config, checkpointing, metrics, and visualization helpers
├── logger.py           # Logging utilities
├── configs/
│   ├── protofuse.yaml  # ProtoFuse configuration
│   ├── apt.yaml        # APT configuration
│   ├── coop.yaml       # CoOp configuration
│   ├── maple.yaml      # MaPLe configuration
│   ├── tip_adapter.yaml
│   ├── timo.yaml       # TIMO configuration
│   └── timos.yaml      # TIMOS configuration
├── src/
│   ├── models/         # Model implementations
│   └── pipelines/      # Training and evaluation pipelines
├── test/
│   ├── compare_methods.py
│   └── zeroshot_clip.py
├── checkpoints/
├── outputs/
├── datasets/
└── pyproject.toml
```

## Supported Methods

| Method | Script | Description |
|--------|--------|-------------|
| **ProtoFuse** | `protofuse.py` | Prototype fusion for few-shot CLIP adaptation |
| **APT** | `apt.py` | Cross-attention prompt tuning on CLIP |
| **CoOp** | `coop.py` | Context Optimization for CLIP |
| **MaPLe** | `maple.py` | Multi-modal Prompt Learning for CLIP |
| **Tip-Adapter** | `tip_adapter.py` | Cache-based CLIP adaptation with optional Tip-Adapter-F fine-tuning |
| **TIMO** | `timo.py` | Text-image mutual guidance optimization |
| **TIMOS** | `timos.py` | TIMO search variant |
