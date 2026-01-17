# tinyvla (wip): Qwen2.5-VL + LoRA on LIBERO-10

This repo provides a **single-file** entrypoint: `train.py`.

- **Train**: Qwen2.5-VL with **LoRA**, predicting discretized actions as text.
- **Eval**: **local-only** evaluation restricted to **`libero_10`**.

---

## Commands (run in order)

### 0) Clone + checkout the branch

```bash
git clone <your-repo-url>
cd tinyvla
git checkout wip
```

### 1) Create and activate a virtualenv

Use Python 3.11 (recommended).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

### 2) Install training dependencies

```bash
# LeRobot install is a git dependency; skipping LFS avoids pulling large assets during install.
export GIT_LFS_SKIP_SMUDGE=1

pip install -r requirements.txt
```

### 3) (Optional) Enable Weights & Biases logging

```bash
# Option A: interactive login
wandb login

# Option B: non-interactive
export WANDB_API_KEY="..."

# Recommended
export WANDB_PROJECT="tinyvla"
```

W&B is enabled automatically when `wandb` is installed (unless you pass `--report_to` yourself).

### 4) Run a quick smoke-train (recommended first run)

This verifies:
- your HF dataset access works
- the `libero_10` index cache builds
- training steps run end-to-end

```bash
python train.py \
  --mode train \
  --output_dir runs/smoke_lora \
  --max_train_samples 256 \
  --step_budget 50
```

### 5) Run full training (step-budgeted)

```bash
python train.py \
  --mode train \
  --output_dir runs/libero10_lora \
  --step_budget 13000 \
  --use_flash_attention
```

Notes:
- Training is **step-budgeted** (default `--step_budget 13000`).
- LoRA is enabled by default (`--use_lora` is on by default in `train.py`).

---

## Local evaluation on `libero_10` (optional)

### 6) Install eval dependencies

`libero` evaluation requires MuJoCo and robosuite. `mujoco==3.3.2` is recommended due to known rendering issues on newer versions.

```bash
pip install mujoco==3.3.2 numpy==1.26.4 robosuite imageio
pip install "libero @ git+https://github.com/MilkClouds/LIBERO.git@minor-fix"
```

### 7) Run eval on a checkpoint (no video by default)

```bash
python train.py \
  --mode eval \
  --model_path runs/libero10_lora/checkpoint-XXXX \
  --episodes_per_task 5 \
  --no_video
```

To evaluate a single task:

```bash
python train.py \
  --mode eval \
  --model_path runs/libero10_lora/checkpoint-XXXX \
  --task_name "<libero_task_name>" \
  --episodes_per_task 5 \
  --no_video
```

If `dataset_stats.json` can’t be auto-found, pass it explicitly:

```bash
python train.py \
  --mode eval \
  --model_path runs/libero10_lora/checkpoint-XXXX \
  --stats_path runs/libero10_lora/dataset_stats.json \
  --no_video
```

---

## Notes / troubleshooting

- **Cache location**: `./cache/` (contains `libero_10` index JSON). Rebuild with `--rebuild_cache`.
- **Flash attention**: `--use_flash_attention` uses `kernels-community/flash-attn3` via `kernels`. This is typically for Linux GPU environments.
- **Nested reference repo**: `vla0-trl/` in this workspace is not tracked by git and is not required to run `train.py`.
