# TinyVLA

Train VLA-0 in **one file**. (~500 lines)

Based on [VLA-0](https://arxiv.org/abs/2510.13054) using TRL's SFTTrainer.

## Install

```bash
uv venv --python 3.11.6 && source .venv/bin/activate

# Core
uv pip install -e .

# Eval + LeRobot (LeRobot is pinned; zarr<3 keeps numpy==1.26.4 for LIBERO/robosuite)
# Note: `GIT_LFS_SKIP_SMUDGE=1` prevents downloading large LFS assets during install.
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[eval,lerobot]"
```

## Train

```bash
# Single GPU (uses TRL defaults)
python tinyvla.py --output_dir runs/vla0

# Train on one LIBERO suite only (e.g. LIBERO-Object)
python tinyvla.py --train_suite libero_object --output_dir runs/vla0-object

# Run eval automatically after training (defaults to 5 episodes per task)
python tinyvla.py --train_suite libero_object --eval_after_train true --output_dir runs/vla0-object

# LoRA (parameter-efficient)
uv pip install -e ".[lora]"
python tinyvla.py --use_lora true --output_dir runs/vla0-lora

# With config file
python tinyvla.py --config config.yaml

# Multi-GPU
accelerate launch --num_processes=4 tinyvla.py --output_dir runs/vla0
```

## Evaluate

```bash
# Install eval dependencies
uv pip install -e ".[eval]"

# Run evaluation
python tinyvla.py --eval runs/vla0/final --suite libero_spatial
```

## Config

Create `config.yaml`:

```yaml
# Model
model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
use_flash_attention: true

# Data
repo_id: "physical-intelligence/libero"
horizon: 8
img_size: 224
crop_ratio: 0.875
tile_images: true

# Augmentation
brightness_aug: 0.2
contrast_aug: 0.2
saturation_aug: 0.2
hue_aug: 0.05
action_mask_aug_pct: 0.4

# Training (SFTConfig)
output_dir: "./runs/vla0"
num_train_epochs: 32
per_device_train_batch_size: 8
learning_rate: 4.0e-5
lr_scheduler_type: "constant"
bf16: true
logging_steps: 10
save_steps: 10000
dataloader_num_workers: 8
report_to: ["wandb"]
```

## How It Works

```
Image + Instruction → Qwen2.5-VL → "500 234 789..." → Robot Actions
```

1. Two camera images tiled horizontally + task instruction
2. VLM outputs space-separated integers (0-1000) for 8 timesteps × 7 DOF
3. Decode integers back to continuous actions

Key techniques:
- **Constrained decoding**: Only allow digits + spaces
- **Action mask augmentation**: Replace tokens with '?' to force visual reasoning
- **Per-task normalization**: Learn action bounds from data

## References

- [VLA-0 Paper](https://arxiv.org/abs/2510.13054)
- [vla0-trl](https://github.com/MilkClouds/vla0-trl)
