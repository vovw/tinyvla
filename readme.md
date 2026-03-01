# SimVLA: A Simple VLA Baseline for Robotic Manipulation

| **Paper** | **Website** | **Model & Data** |
| :------------------: | :-----------------------: | :---------------------: |
| [![Paper](https://img.shields.io/badge/Paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2602.18224) | [![Website](https://img.shields.io/badge/Project%20Page-181717?style=for-the-badge&logo=githubpages&logoColor=white)](https://frontierrobo.github.io/SimVLA/) | [![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFBA00?style=for-the-badge&logo=huggingface&logoColor=white)](https://huggingface.co/collections/YuankaiLuo/simvla) |

A simple and efficient Vision-Language-Action (VLA) model for robot manipulation tasks.

<img width="506" height="796" alt="image" src="https://github.com/user-attachments/assets/7ffb8969-aa4f-4bcc-8c38-33d5e7da4b25" />

## Installation

### Option A: Fast setup with `uv` (recommended)

```bash
# install uv once if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create .venv and sync dependencies from pyproject.toml
uv sync

# optional extras
uv sync --extra flash-attn
uv sync --extra tensorflow
uv sync --extra libero-client
```

Run commands inside the environment:
```bash
uv run python e2e_train.py --help
```

> `pyproject.toml` already points `torch`/`torchvision` to CUDA 12.4 wheels.

### Option B: Manual pip/conda setup

```bash
conda create -n simvla python=3.10 -y
conda activate simvla

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers>=4.57.0
pip install peft accelerate fastapi tensorboard uvicorn json_numpy safetensors scipy einops timm mmengine pyarrow h5py mediapy num2words av wandb websockets msgpack_numpy
pip install flash-attn==2.5.6 --no-build-isolation
pip install tensorflow tensorflow-datasets
```

> Important: Use `transformers>=4.57.0`.

## Training (LIBERO Dataset)

### 1. Prepare LIBERO Dataset

Download [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) dataset, and place it in `./datasets/metas/`.

### 2. Create Training Metadata

```bash
python create_libero_meta.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./datasets/metas/libero_train.json
```

### 3. Compute Normalization Statistics

```bash
python compute_libero_norm_stats.py \
    --data_dir ./datasets/metas \
    --subsets libero_10 libero_goal libero_object libero_spatial \
    --output ./norm_stats/libero_norm.json
```

### 4. Start Training

**Small Model Configuration:**
```bash
uv run bash train_smolvlm_small.sh
```

**Large Model Configuration:**
```bash
uv run bash train_smolvlm_large.sh
```

**Single-command E2E training (prep + train):**
```bash
uv run python e2e_train.py \
    --size small \
    --data_dir ./datasets/metas \
    --gpus "0,1,2,3"
```

### 5. Evaluation

```bash
cd evaluation/libero
```

Unified wrapper from repo root:

```bash
# 1) start server (SimVLA env)
uv run python e2e_eval.py serve \
    --checkpoint ./runs/simvla_libero_small/ckpt-20000 \
    --norm_stats ./norm_stats/libero_norm.json \
    --port 8102

# 2) run one suite (LIBERO env)
python e2e_eval.py client \
    --python-bin python \
    --host 127.0.0.1 \
    --port 8102 \
    --task_suite libero_spatial \
    --num_trials 10

# 3) run all 4 suites in parallel
python e2e_eval.py all \
    --port 8102 \
    --num_trials 10 \
    --output_prefix eval_simvla_20k \
    --gpus "0 1 2 3"
```

### 6. Results

<img width="506" height="1220" alt="image" src="https://github.com/user-attachments/assets/6ee1cd5e-42c5-4cf7-9cce-6dc04c1a215f" />

## Model Architecture

- **Vision-Language Backbone**: SmolVLM-500M-Instruct (576 hidden dim)
- **Action Transformer**: Configurable depth and width
  - Small: 768 hidden, 12 layers, 12 heads
  - Large: 1024 hidden, 24 layers, 16 heads

## Reference

If you find our codes useful, please consider citing our work

```
@article{luo2026simvla,
  title={SimVLA: A Simple VLA Baseline for Robotic Manipulation},
  author={Luo, Yuankai and Chen, Woping and Liang, Tong and Wang, Baiqiao and Li, Zhenguo},
  journal={arXiv preprint arXiv:2602.18224},
  year={2026}
}
```
