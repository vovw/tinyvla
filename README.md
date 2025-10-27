**TinyVLA** is a minimal vision-language-action model built on NVIDIA's VLA-0 Architecture.

Most VLAs modify the base vision-language model—adding action tokens, custom heads, or new architectures. VLA-0 does none of this. It represents actions as text and generates them like any other output.

This approach is surprisingly powerful, outperforming methods with specialized architectures and even those pretrained on massive robot datasets.

TinyVLA is meant to help people understand why this simple method works.

**Table of Contents**
- [Getting Started](#getting-started)
- [Overview](#overview)
- [Building Blocks](#building-blocks)
  - [Masked Action Augmentation](#masked-action-augmentation)
  - [Action Ensembling](#action-ensembling)
  - [LoRA Fine-tuning](#lora-fine-tuning)
- [Architecture](#architecture)
  - [Text-Based Actions](#text-based-actions)
  - [Vision-Language Backbone](#vision-language-backbone)
  - [Action Prediction](#action-prediction)
- [TinyVLA Inference](#tinyvla-inference)
- [Data](#data)
- [Training/Inference Acceleration](#traininginference-acceleration)
- [Results](#results)
- [Next Steps](#next-steps)

## Getting Started
```bash
# Installation
git clone --recursive https://github.com/vovw/tinyvla.git
cd tinyvla
python3.10 -m venv .venv
source .venv/bin/activate
uv pip install -e .

# Training
# 1. download LIBERO datasets
cd LIBERO && uv pip install -e .
python libero/scripts/download_libero_datasets.py --suite spatial
# 2. run training
python train.py --data-dir LIBERO/libero/datasets/libero_spatial --epochs 64
```

## Overview

**Why Vision-Language-Action Models?**

Vision-Language-Action models adapt pretrained VLMs for robot control. But how should we build them?

Current approaches fall into three families:

**1. Discrete Token VLAs** (RT-2, OpenVLA)
- Discretize actions into bins
- Assign each bin to a token in the VLM vocabulary
- Problem: Limited action resolution, corrupts pretrained language understanding

**2. Generative Action Head VLAs** (π0, SmolVLA, GR00T-N1)
- Add a diffusion or flow-matching head on top of the VLM
- VLM outputs latent, head decodes to continuous actions
- Problem: New untrained component, degrades language grounding

**3. Custom Architecture VLAs** (OpenVLA-OFT, π-FAST)
- Specialized heads, custom tokenizers, architectural modifications
- Problem: Complex, requires intricate training pipelines

**VLA-0: The Simple Alternative**

What if we just generate actions as text? No vocabulary changes, no new heads, no architecture modifications.

The VLM already knows how to generate numbers. Continuous actions [−1, 1] become integers [0, 1000]. A 7-DOF action over 10 timesteps is 70 space-separated integers:

```
"500 234 789 120 0 500 1000 501 235 790 ..."
```

This preserves the VLM's integrity while allowing arbitrary action resolution.

**Why VLA-0 Works**

The paper identifies three critical components:

**1. Masked Action Augmentation**
- Problem: VLMs auto-complete number sequences without using visual input
- Solution: Randomly mask digits during training ("500" → "5_0")
- Forces model to reason from vision, not pattern matching

**2. Action Ensembling**
- Predict 10 future actions at each timestep
- Average overlapping predictions across time
- Reduces jitter, improves stability (from ACT)

**3. Proper Training Recipe**
- Full fine-tuning (not just LoRA in this impl)
- Careful hyperparameters
- System prompt design

These three components unlock state-of-the-art performance from the simplest possible VLA architecture.

## Building Blocks

### Masked Action Augmentation

Masked Action Augmentation is the critical training technique that makes text-based actions viable.

**The Problem: Autoregressive Collapse**

VLMs generate text autoregressively—each token depends on previous tokens. When generating numerical sequences, the model can pattern-match and auto-complete without ever looking at the visual input.

Example: After generating "500 123", the model might predict "789" based purely on statistical patterns in the action distribution, completely ignoring what the robot camera sees.

This is catastrophic. The model appears to work during training (low loss) but fails at test time because it never learned visual reasoning.

**The Solution: Character-Level Masking**

During training only, we randomly mask individual characters in the target action string with underscores:

```
Original: "500 123 789 0 50 ..."
Masked:   "5_0 1_3 _89 _ 50 ..."
```

The model must now reconstruct the masked characters. Since it can't rely on auto-completion, it's forced to use the visual observation to infer the correct digits.

**Implementation**
- Each digit independently has probability `mask_prob` (default 0.3) of being masked
- Spaces are never masked (preserve structure)
- Only during training; inference uses normal generation
- Ablation: Removes this → performance drops 1.2 points

This simple augmentation is the difference between a model that auto-completes and one that actually sees.

### Action Ensembling

Action Ensembling provides temporal smoothing for stable robot control. This technique is borrowed from ACT (Action Chunking Transformers) and is used by state-of-the-art VLAs like OpenVLA-OFT.

**How It Works**

At timestep `t`, the model predicts a sequence of `n=10` future actions:
```
Predict at t:   [a_t, a_{t+1}, a_{t+2}, ..., a_{t+9}]
```

At timestep `t+1`, it predicts another sequence:
```
Predict at t+1: [a_{t+1}, a_{t+2}, a_{t+3}, ..., a_{t+10}]
```

Notice that action `a_{t+1}` appears in both predictions. We can exploit this overlap.

**Temporal Averaging**

For any action at time `t`, we have multiple predictions from different timesteps:
- Prediction made at time `t` (immediate)
- Prediction made at time `t-1` (1-step ahead)
- Prediction made at time `t-2` (2-step ahead)
- ...
- Prediction made at time `t-n+1` (n-1 steps ahead)

We average all available predictions:
```python
a_t = mean([pred_{t}[0], pred_{t-1}[1], pred_{t-2}[2], ..., pred_{t-n+1}[n-1]])
```

**Why It Works**

Near-term predictions are more accurate (the model sees more recent observations), while far-future predictions are noisy. By averaging, we:
- Smooth out single-step jitter
- Implicitly regularize across time
- Improve stability of robot trajectories

**Results**
- Ablation: Remove ensembling → performance drops 2.0 points
- Critical for real robot deployment (reduces control instability)

### System Prompt Design

The system prompt is critical for guiding the VLM to generate well-formed action sequences.

**Paper's System Prompt:**
```
Analyze the input image and predict robot actions for the next H timesteps.
Each action has D dimensions. Output a single sequence of H × D integers
(0 - B each), representing the H timesteps sequentially. Provide only
space-separated numbers. Nothing else.
```

Where:
- `H` = action horizon (typically 10 timesteps)
- `D` = action dimensions (7-DOF for LIBERO: position, rotation, gripper)
- `B` = discretization bins (1000 for LIBERO)

The prompt explicitly constrains output format: space-separated integers only, no other text.

**Why This Matters**

Without careful prompting, VLMs might:
- Add explanatory text before/after actions
- Use inconsistent formatting (commas, brackets)
- Generate out-of-range values

The system prompt eliminates these failure modes.

**Training Details**


## Architecture

### Text-Based Actions

**Why Text Actions?**

Three families of VLAs all struggle with the same problem: how to represent continuous actions in a discrete model.

- Discrete Token VLAs: Limited to vocabulary size, corrupt language understanding
- Generative Head VLAs: Add untrained components, degrade grounding
- Custom VLAs: Complex pipelines, specialized architectures

Text-based actions sidestep all of these issues. Numbers are already in the vocabulary. The VLM already knows how to generate them. We need zero modifications.

**Action Discretization**

For LIBERO, actions are continuous 7-DOF vectors (3D position, 3D rotation, 1D gripper) sampled at 10 Hz.

Discretization process:
1. Normalize each dimension: `[−1, 1] → [0, 1000]`
2. Round to nearest integer: `0.234 → 234`
3. Serialize as space-separated text: `"500 234 789 120 0 500 1000"`

For action horizon `H=10` and dimensions `D=7`:
```
"500 234 789 120 0 500 1000 501 235 790 121 1 501 999 ..."
 └─────── timestep 1 ──────┘ └─────── timestep 2 ──────┘ ...
```

Total: `H × D = 70` integers per prediction.

**Why 1000 Bins?**

The paper ablates action resolution:
- 250 bins: Performance drops 1.5 points (insufficient precision)
- 1000 bins: Optimal for LIBERO
- 4000 bins: No improvement (diminishing returns)

Unlike discrete token VLAs, we can tune this freely without touching the vocabulary.

**Inference: Text to Actions**

At inference, reverse the process:
1. Parse text to integers: `"500 234" → [500, 234]`
2. Denormalize: `[500, 234] → [0.0, -0.532]`
3. Send to robot controller

Uses only existing tokens: digits 0-9 and spaces. Zero vocabulary modifications.

### Vision-Language Backbone

**Base Model: Qwen2.5-VL-3B-Instruct**

The paper uses Qwen2.5-VL-3B for several reasons:
- 3.9B parameters (efficient, fast training/inference)
- State-of-the-art performance for model size
- Open weights (reproducible)
- Native multi-image support

**Architecture Components**
- **Vision Encoder**: Vision Transformer (ViT) extracts visual features
- **Language Model**: LLM processes text and visual tokens jointly
- **Zero modifications**: No new layers, no vocabulary changes

**Input Format**

The model takes three inputs:

1. **System Prompt**: Specifies output format (see System Prompt Design above)
2. **Images**: One or more camera views (LIBERO: third-person + wrist)
3. **Task Instruction**: Natural language (e.g., "pick up the red block")

**Output Format**

Pure text:
```
"500 234 789 120 0 500 1000 501 235 790 ..."
```

The VLM's text generation capabilities handle this natively. Standard cross-entropy loss over the vocabulary.

### Training with Masked Actions

**Training Format (without masking):**
```
System: [system prompt describing format]
User: [image] "pick up the red block"
Assistant: "500 234 789 120 0 500 1000 501 235 ..."
```

**Training Format (with Masked Action Augmentation):**
```
System: [system prompt describing format]
User: [image] "pick up the red block"
Assistant: "5_0 _34 789 1_0 _ 5_0 100_ 50_ 2__ ..."
```

The model is trained with standard language modeling loss (cross-entropy) to predict the target sequence character-by-character.

Crucially, with masking, the model cannot auto-complete—it must use visual features to reconstruct masked digits. This forces the VLM to ground actions in visual observations.

## TinyVLA Inference

**Single-Step Inference (without ensembling)**

At each timestep:

1. **Input**: System prompt + images + task instruction
2. **VLM Forward**: Vision encoder extracts features, LLM generates action text
3. **Parse**: Convert text to integers: `"500 234 ..." → [500, 234, ...]`
4. **Denormalize**: Map [0, 1000] → [−1, 1] for each dimension
5. **Execute**: Send action to robot controller
6. **Loop**: Get next observation, repeat

**With Action Ensembling**

At timestep `t`:

1. **Predict**: Generate sequence of 10 future actions: `[a_t, a_{t+1}, ..., a_{t+9}]`
2. **Buffer**: Store predictions in temporal buffer
3. **Ensemble**: Average all predictions for action `a_t` from previous timesteps
4. **Execute**: Send ensembled action to robot
5. **Loop**: Advance buffer, repeat

The ensembling window provides temporal smoothing, critical for stable robot control.

**Inference Speed**
- Paper (real robot): 4 Hz on desktop GPU
- Bottleneck: VLM forward pass, not action representation
- Improvements possible: quantization, distillation, speculative decoding

## Data

**LIBERO: Benchmark for Lifelong Robot Learning**

LIBERO provides 90 diverse manipulation tasks across 4 evaluation suites:

| Suite | Tasks | Focus |
|-------|-------|-------|
| `libero_spatial` | 10 | Spatial reasoning ("pick up X and place it on Y") |
| `libero_object` | 10 | Object manipulation (different objects, same task) |
| `libero_goal` | 10 | Goal-conditioned (achieving specific configurations) |
| `libero_long` | 10 | Long-horizon (multi-step tasks requiring planning) |

Each task has ~50 demonstrations for training.

**Data Format**
- **Actions**: 7-DOF continuous (3D position, 3D rotation, 1D gripper)
- **Observations**: RGB images (128×128) from third-person + wrist cameras
- **Instructions**: Natural language task descriptions

**Download**
```bash
cd LIBERO
uv pip install -e .
python libero/scripts/download_libero_datasets.py --suite spatial
python libero/scripts/download_libero_datasets.py --suite object
python libero/scripts/download_libero_datasets.py --suite goal
python libero/scripts/download_libero_datasets.py --suite long
```

Each demonstration is a sequence of `(observation, action, instruction)` tuples.

## Training/Inference Acceleration

**Paper Configuration (Full Fine-tuning)**
- Hardware: 8× A100 GPUs
- Batch size: 192 (total across GPUs)
- Learning rate: 5e-6
- Optimizer: Adam
- Training time: 32 hours
- All 3.9B parameters updated

**This Implementation (LoRA)**

For consumer hardware, we use memory-efficient techniques:

**LoRA Fine-tuning**: Updates only 3.8% of parameters (148.6M / 3.9B)
- Massively reduces optimizer state memory
- Enables training on single consumer GPU
- Minimal performance degradation vs full fine-tuning

**Gradient Accumulation**: Simulates large batch sizes
- Example: batch_size=4, grad_accum=16 → effective batch=64
- Trains with large batches on memory-constrained GPUs

**Mixed Precision (BF16)**: Reduces memory, increases throughput
- Automatic mixed precision training
- Works on Ampere+ GPUs (RTX 30XX, 40XX, A100)

**8-bit AdamW**: Quantized optimizer (via bitsandbytes)
- ~50% memory reduction for optimizer states
- Negligible performance impact

**Gradient Checkpointing**: Trades compute for memory
- Recomputes activations during backward pass
- Enables larger models on smaller GPUs

Training configuration:
```bash
python train.py \
  --data-dir LIBERO/libero/datasets/libero_spatial \
  --epochs 64 \
  --batch-size 4 \
  --grad-accum 16 \
  --mask-prob 0.3 \
  --lr 5e-5
```

Key arguments:
- `--batch-size 4`: Per-GPU batch size
- `--grad-accum 16`: Gradient accumulation (effective batch = 64)
- `--mask-prob 0.3`: Masked action augmentation probability (from paper)
- `--lr 5e-5`: Learning rate (higher than paper's 5e-6 due to LoRA)

Estimated training time: ~127 hours on RTX 4090

## Results

**Real Robot Performance**

## Next Steps

1. **Large-scale pretraining**: How does VLA-0 perform when pretrained on massive robot datasets (Open X-Embodiment, DROID, etc.)? The current results are without pretraining.

2. **Inference speedup**: 4 Hz is acceptable but not ideal. Optimization techniques to explore:
   - Model quantization (INT8, INT4)
   - Knowledge distillation to smaller VLMs
   - Speculative decoding for faster generation

**Additional Research Directions**

- **Masking strategies**: Token-level vs digit-level, adaptive masking schedules
- **Base VLM comparison**: LLaVA, InternVL, PaliGemma, Molmo—which works best?
- **Action resolution**: Find optimal discretization for different tasks (fine manipulation vs gross motion)
- **Action horizons**: Longer prediction windows (20 steps, 50 steps) for planning
- **Multi-task training**: Joint training across all LIBERO suites
- **Scaling laws**: Performance vs model size (7B, 14B, 30B parameters)
- **Full fine-tuning**: Match paper's setup with FSDP on 8× GPUs
- **Alternative architectures**: Test on other robot platforms (Mobile ALOHA, Franka, UR5)

**Ablations to Explore**

The paper's ablations suggest room for improvement:
- Better image tiling strategies for multi-view inputs
- Curriculum learning over task difficulty
- Alternative ensembling schedules (weighted averaging, learned weights)

---

**Citation**

If you use this code, please cite:

```bibtex
@article{goyal2025vla0,
  title={VLA-0: Building State-of-the-Art VLAs with Zero Modification},
  author={Goyal, Ankit and Hadfield, Hugo and Yang, Xuning and Blukis, Valts and Ramos, Fabio},
  journal={arXiv preprint arXiv:2510.13054},
  year={2025}
}

@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={NeurIPS},
  year={2023}
}
```

aesthetic inspired by [Tinygrad](https://github.com/tinygrad/tinygrad) and [TinyWorlds](https://github.com/AlmondGod/tinyworlds)
