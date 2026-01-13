# TinyVLA

Minimal Vision-Language-Action model. Train a robot policy in **one file**.

Based on [VLA-0](https://arxiv.org/abs/2510.13054) — achieves state-of-the-art on LIBERO with zero architecture modifications.

## Quick Start

```bash
# Setup (H100 optimized, CUDA 12.4 + Flash Attention)
./setup.sh
source .venv/bin/activate

# Train
python tinyvla.py                           # Full training
python tinyvla.py --test                    # Smoke test

# Multi-GPU (4x H100)
accelerate launch --num_processes 4 tinyvla.py --batch-size 16

# Evaluate on LIBERO
python tinyvla.py --eval checkpoints/final --suite libero_spatial
```

## How It Works

```
Image + Instruction  →  Qwen3-VL-2B  →  "500 234 789 120..."  →  Robot Actions
```

1. **Input**: Two camera images (agentview + wrist) + natural language instruction
2. **VLM outputs**: Space-separated integers (0-1000) for 8 timesteps × 7 DOF = 56 numbers
3. **Decode**: Map integers back to continuous actions using learned min/max bounds

No special heads, no custom tokens, no architecture changes. Just text.

## Key Techniques from VLA-0

| Technique | What it does |
|-----------|--------------|
| **Constrained Decoding** | Only allow digits + spaces during generation → no parsing errors |
| **Masked Action Augmentation** | Replace digits with `?` during training → forces visual reasoning |
| **Action Ensembling** | Average overlapping predictions → smoother robot control |
| **Per-task Normalization** | Learn action bounds from data → better discretization |

## Configuration

All VLA-0 defaults are tuned for LIBERO:

```bash
python tinyvla.py \
  --model qwen3-2b \        # Qwen3-VL-2B (default)
  --epochs 24 \             # Training epochs
  --batch-size 8 \          # Batch size
  --lr 5e-6 \               # Learning rate (full finetune)
  --horizon 8 \             # Action prediction horizon
  --mask-prob 0.4           # Masked augmentation probability
```

### Models

| Model | Size | VRAM | H100 batch size |
|-------|------|------|-----------------|
| `qwen3-2b` | 2.2B | ~12GB | 32+ |
| `qwen3-4b` | 4.0B | ~18GB | 24 |
| `qwen3-8b` | 8.0B | ~32GB | 16 |
| `qwen2.5-3b` | 3.9B | ~24GB | 20 |
| `qwen2.5-7b` | 7.6B | ~40GB | 12 |

H100 80GB fits all models without LoRA. Use `--use-lora` on smaller GPUs.

## Dataset

TinyVLA uses the [LeRobot](https://github.com/huggingface/lerobot) LIBERO dataset from HuggingFace:

```bash
# Automatic download on first run
python tinyvla.py  # Downloads from physical-intelligence/libero
```

Or use a different dataset:
```bash
python tinyvla.py --data-repo your-org/your-dataset
```

## Evaluation

### Setup

```bash
# Install eval dependencies
uv pip install -e ".[eval]"

# Install LIBERO (HuggingFace fork, works with modern torch)
pip install git+https://github.com/huggingface/lerobot-libero.git --no-deps
```

### Run Evaluation

```bash
python tinyvla.py --eval checkpoints/final --suite libero_spatial
python tinyvla.py --eval checkpoints/final --suite libero_object
python tinyvla.py --eval checkpoints/final --suite libero_goal
python tinyvla.py --eval checkpoints/final --suite libero_10
```

## Project Structure

```
tinyvla/
├── tinyvla.py      # Everything in one file (~600 lines)
├── setup.sh        # One-command setup
├── pyproject.toml  # Dependencies
└── vla0/           # Reference VLA-0 implementation
```

## Performance

Expected results on LIBERO (from VLA-0 paper):

| Suite | Success Rate |
|-------|--------------|
| libero_spatial | ~85% |
| libero_object | ~82% |
| libero_goal | ~78% |
| libero_10 | ~75% |

## Citation

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and others},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}
```

---

*Inspired by [nanoGPT](https://github.com/karpathy/nanoGPT) — making complex ideas simple.*
