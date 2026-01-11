"""
TinyVLA: Train VLA-0 in one file. Inspired by nanoGPT.

Usage:
    python tinyvla.py                           # Train with VLA-0 defaults
    python tinyvla.py --resume                  # Resume from checkpoint
    python tinyvla.py --eval out/final          # Evaluate checkpoint

Single A100 80GB command:
    python tinyvla.py --batch_size=16 --compile

References:
    - VLA-0 Paper: https://arxiv.org/abs/2510.13054
    - Official VLA-0: https://github.com/NVlabs/vla0
    - vla0-trl: https://github.com/MilkClouds/vla0-trl
"""

import json
import math
import os
import pickle
import random
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Default config values - VLA-0 paper settings for LIBERO
# -----------------------------------------------------------------------------

# I/O
out_dir = "out"
eval_interval = 2  # evaluate every N epochs
log_interval = 10  # log every N steps
eval_only = False
always_save_checkpoint = True
init_from = "scratch"  # 'scratch' or 'resume'

# wandb logging
wandb_log = True
wandb_project = "tinyvla"
wandb_run_name = "vla0"

# data
data_repo = "physical-intelligence/libero"
cam_list = ("image", "wrist_image")

# model
model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
horizon = 8  # predict 8 future timesteps
action_dim = 7  # 7-DoF actions
num_bins = 1000  # discretize to [0, 1000]
use_lora = True  # default to LoRA for lower VRAM; disable via CLI
use_flash_attention = True  # try FlashAttention2/3 when available
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
lora_target_modules = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")

# training
num_epochs = 24  # VLA-0 paper default (single-GPU in this script)
batch_size = 8  # per GPU, use 48 on B200, 16 on A100 80GB
learning_rate = 5e-6  # VLA-0 default (scaled by num_gpus in multi-gpu)
weight_decay = 1e-10
grad_clip = 1.0
grad_accum_steps = 1  # gradient accumulation steps (effective_batch = batch_size * grad_accum_steps)
num_workers = 8

# augmentation (VLA-0 defaults)
img_size = 224
crop_ratio = 0.875
tile_images = True
brightness_aug = 0.2
contrast_aug = 0.2
saturation_aug = 0.2
hue_aug = 0.05
action_mask_aug_pct = 0.4  # masked action augmentation

# system
device = "cuda"
dtype = "bfloat16"
compile = False  # torch.compile - faster but slower startup

# -----------------------------------------------------------------------------
config_keys = [
    k for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str, tuple))
]
exec(open("configurator.py").read()) if os.path.exists("configurator.py") else None
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = "cuda" if "cuda" in device else "cpu"
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = nullcontext() if device_type == "cpu" else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# =============================================================================
# Constrained Decoding
# =============================================================================

class NumberSpaceLogitsProcessor:
    """Only allow digits (0-9), spaces, and EOS during generation."""

    def __init__(self, tokenizer):
        self.allowed = set()
        for i in range(10):
            self.allowed.add(tokenizer.encode(str(i), add_special_tokens=False)[0])
        self.allowed.add(tokenizer.encode(" ", add_special_tokens=False)[0])
        if tokenizer.eos_token_id:
            self.allowed.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for tok in self.allowed:
            mask[:, tok] = 0
        return scores + mask


# =============================================================================
# Dataset
# =============================================================================

class LiberoDataset(Dataset):
    """LIBERO dataset with VLA-0 preprocessing."""

    def __init__(self, repo_id, horizon, img_size, crop_ratio, tile_images,
                 brightness_aug, contrast_aug, saturation_aug, hue_aug,
                 cam_list, action_dim, num_bins, train=True):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        self.horizon = horizon
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.train = train
        self.num_bins = num_bins
        self.action_dim = action_dim

        # Augmentation params
        self.brightness = brightness_aug
        self.contrast = contrast_aug
        self.saturation = saturation_aug
        self.hue = hue_aug

        # Build delta timestamps (matches VLA-0)
        fps = 10
        delta_timestamps = {
            "state": [0.0],
            "actions": [x / fps for x in range(-1, horizon)],  # history + future
        }
        for cam in cam_list:
            delta_timestamps[cam] = [0.0]

        print(f"Loading {repo_id}...")
        self.dataset = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)
        self.cam_list = cam_list

        # Get action stats for normalization
        stats = self.dataset.meta.stats["actions"]
        self.action_min = np.array(stats["min"])
        self.action_max = np.array(stats["max"])

        # Save stats for inference
        self.stats = {"out_ori_act": {"min": self.action_min.tolist(), "max": self.action_max.tolist()}}

        # System prompt (VLA-0 format)
        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
            f"Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers (0-{num_bins} each), "
            f"representing the {horizon} timesteps sequentially. "
            f"Provide only space separated numbers. Nothing else."
        )

    def __len__(self):
        return len(self.dataset)

    def _process_image(self, img_tensor):
        """Process image tensor to numpy array with augmentation."""
        img = img_tensor
        if img.ndim == 4:
            img = img[0]  # Remove batch dim
        img = (img * 255).byte()  # [C, H, W]

        # Random crop (train) or center crop (eval)
        if self.crop_ratio < 1.0:
            h, w = img.shape[-2:]
            ch, cw = int(h * self.crop_ratio), int(w * self.crop_ratio)
            if self.train:
                top = random.randint(0, h - ch)
                left = random.randint(0, w - cw)
            else:
                top, left = (h - ch) // 2, (w - cw) // 2
            img = img[:, top:top+ch, left:left+cw]

        # Resize
        import torchvision.transforms.functional as TF
        img = TF.resize(img, [self.img_size, self.img_size])

        # Color augmentation (train only)
        if self.train:
            img = img.float() / 255.0
            if self.brightness > 0:
                img = TF.adjust_brightness(img, 1 + random.uniform(-self.brightness, self.brightness))
            if self.contrast > 0:
                img = TF.adjust_contrast(img, 1 + random.uniform(-self.contrast, self.contrast))
            if self.saturation > 0:
                img = TF.adjust_saturation(img, 1 + random.uniform(-self.saturation, self.saturation))
            if self.hue > 0:
                img = TF.adjust_hue(img, random.uniform(-self.hue, self.hue))
            img = (img * 255).byte()

        # [C, H, W] -> [H, W, C]
        return img.permute(1, 2, 0).numpy()

    def _actions_to_text(self, actions):
        """Discretize actions to text."""
        normalized = (actions - self.action_min) / (self.action_max - self.action_min + 1e-8)
        discretized = np.clip(np.round(normalized * self.num_bins), 0, self.num_bins).astype(int)
        return " ".join(map(str, discretized.flatten()))

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Process images
        images = [self._process_image(sample[cam]) for cam in self.cam_list]
        if self.tile_images and len(images) > 1:
            image = np.concatenate(images, axis=1)  # Tile horizontally
        else:
            image = images[0]

        # Get instruction and actions
        instruction = sample["task"]
        actions = sample["actions"][1:].numpy()  # Skip history, keep horizon
        action_text = self._actions_to_text(actions)

        return {
            "image": image,
            "instruction": instruction,
            "action_text": action_text,
        }


# =============================================================================
# Data Collator with Action Masking
# =============================================================================

class VLACollator:
    """Collate batch with action mask augmentation."""

    def __init__(self, processor, action_mask_aug_pct):
        self.processor = processor
        self.aug_pct = action_mask_aug_pct
        q_ids = self.processor.tokenizer("?", add_special_tokens=False)["input_ids"]
        self.question_id = q_ids[0] if q_ids else self.processor.tokenizer.pad_token_id
        self.pad_id = self.processor.tokenizer.pad_token_id

    def __call__(self, examples):
        from qwen_vl_utils import process_vision_info

        texts = []
        image_inputs = []
        action_texts = []

        for ex in examples:
            img = Image.fromarray(ex["image"])

            messages = [
                {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": ex["instruction"]}]},
                {"role": "assistant", "content": [{"type": "text", "text": ex["action_text"]}]},
            ]

            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, add_vision_id=False)
            texts.append(text)
            image_inputs.append(process_vision_info(messages)[0])
            action_texts.append(ex["action_text"])

        # Tokenize batch
        inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # Create labels (mask system + user tokens)
        labels = inputs["input_ids"].clone()

        for i, action_text in enumerate(action_texts):
            action_tokens = self.processor.tokenizer(action_text, add_special_tokens=False)["input_ids"]
            action_len = len(action_tokens)
            nonpad_len = inputs["attention_mask"][i].sum().item()
            sysuser_len = int(nonpad_len - action_len - 2)

            # Mask system + user
            labels[i, :sysuser_len] = -100

            # Action mask augmentation (VLA-0 key technique)
            if random.random() < 0.1:
                aug = 0.0
            else:
                aug = random.uniform(0.0, self.aug_pct)

            mask_len = int(action_len * aug)
            if mask_len > 0:
                indices = random.sample(range(action_len), mask_len)
                indices = [x + sysuser_len for x in indices if x + sysuser_len < labels.size(1)]
                if indices:
                    labels[i, indices] = -100
                    inputs["input_ids"][i, indices] = self.question_id  # '?' token

        # Mask padding
        labels[labels == self.pad_id] = -100
        inputs["labels"] = labels

        return inputs


# =============================================================================
# Model Loading
# =============================================================================

def load_model(model_id, compile_model=False, use_lora_flag=False):
    """Load Qwen2.5-VL for training (supports full FT or LoRA)."""
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    import warnings

    print(f"Loading {model_id}...")
    kwargs = {"torch_dtype": ptdtype, "device_map": device}
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    except ValueError as e:
        warnings.warn(f"Flash attention not available, falling back to default attention: {e}")
        kwargs.pop("attn_implementation", None)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False

    if use_lora_flag:
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=list(lora_target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

    # Processor with correct pixel settings
    pixel_count = img_size * img_size * (2 if tile_images else 1)
    processor = Qwen2_5_VLProcessor.from_pretrained(
        model_id,
        min_pixels=pixel_count,
        max_pixels=pixel_count,
    )

    if compile_model:
        print("Compiling model...")
        model = torch.compile(model)

    return model, processor


# =============================================================================
# Training
# =============================================================================

def train():
    global system_prompt

    # Load dataset
    dataset = LiberoDataset(
        repo_id=data_repo,
        horizon=horizon,
        img_size=img_size,
        crop_ratio=crop_ratio,
        tile_images=tile_images,
        brightness_aug=brightness_aug,
        contrast_aug=contrast_aug,
        saturation_aug=saturation_aug,
        hue_aug=hue_aug,
        cam_list=cam_list,
        action_dim=action_dim,
        num_bins=num_bins,
        train=True,
    )
    system_prompt = dataset.system_prompt

    # Save stats for inference
    with open(f"{out_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset.stats, f, indent=2)

    # Load model
    model, processor = load_model(model_id, compile, use_lora)

    # Collator
    collator = VLACollator(processor, action_mask_aug_pct)

    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=True,
    )

    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches")
    print(f"Batch size: {batch_size}")

    # Resume from checkpoint
    epoch_start = 0
    best_loss = float("inf")

    if init_from == "resume":
        ckpt_path = f"{out_dir}/ckpt.pt"
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=device)

            # Load model weights FIRST
            model_path = f"{out_dir}/model_last"
            if os.path.exists(model_path):
                from transformers import Qwen2_5_VLForConditionalGeneration

                if use_lora:
                    from peft import PeftModel

                    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_id, torch_dtype=ptdtype, device_map=device
                    )
                    model = PeftModel.from_pretrained(base, model_path)
                else:
                    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        model_path, torch_dtype=ptdtype, device_map=device
                    )
                if compile:
                    model = torch.compile(model)

            epoch_start = ckpt["epoch"] + 1
            best_loss = ckpt.get("best_loss", float("inf"))
            print(f"Resumed at epoch {epoch_start}, best_loss={best_loss:.4f}")
        else:
            print(f"No checkpoint found at {ckpt_path}, starting from scratch")

    # Optimizer - create AFTER model is finalized (important for resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    if init_from == "resume" and os.path.exists(f"{out_dir}/ckpt.pt"):
        optimizer.load_state_dict(ckpt["optimizer"])

    # Wandb
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config, resume="allow")

    # Training loop
    model.train()
    t0 = time.time()
    step_t0 = time.time()
    global_step = epoch_start * len(loader) // grad_accum_steps
    optimizer.zero_grad()
    tokens_processed = 0

    print(f"Gradient accumulation steps: {grad_accum_steps}")
    print(f"Effective batch size: {batch_size * grad_accum_steps}")

    for epoch in range(epoch_start, num_epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        epoch_loss = 0.0
        n_batches = 0
        accum_loss = 0.0

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            with ctx:
                outputs = model(**batch)
                loss = outputs.loss / grad_accum_steps  # Scale for accumulation

            loss.backward()
            accum_loss += loss.item()
            tokens_processed += batch["input_ids"].numel()

            # Step optimizer after accumulating gradients
            if (batch_idx + 1) % grad_accum_steps == 0:
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log to wandb with throughput metrics
                if global_step % log_interval == 0 and wandb_log:
                    import wandb
                    step_dt = time.time() - step_t0
                    samples_per_sec = (batch_size * grad_accum_steps * log_interval) / step_dt
                    tokens_per_sec = tokens_processed / step_dt

                    wandb.log({
                        "train/loss": accum_loss * grad_accum_steps,
                        "train/epoch": epoch + (batch_idx + 1) / len(loader),
                        "train/lr": learning_rate,
                        "train/global_step": global_step,
                        "perf/samples_per_sec": samples_per_sec,
                        "perf/tokens_per_sec": tokens_per_sec,
                        "perf/gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
                    }, step=global_step)

                    step_t0 = time.time()
                    tokens_processed = 0

                accum_loss = 0.0

            epoch_loss += loss.item() * grad_accum_steps
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss/n_batches:.4f}")

        avg_loss = epoch_loss / n_batches
        dt = time.time() - t0
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, time={dt/60:.1f}min")

        if wandb_log:
            import wandb
            wandb.log({"train/epoch_loss": avg_loss}, step=global_step)

        # Save checkpoint (always save for resume capability)
        if always_save_checkpoint or avg_loss < best_loss:
            best_loss = min(best_loss, avg_loss)

            # Save model
            model_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_save.save_pretrained(f"{out_dir}/model_last")
            processor.save_pretrained(f"{out_dir}/model_last")

            # Save optimizer state for resume
            ckpt = {
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_loss": best_loss,
                "config": config,
            }
            torch.save(ckpt, f"{out_dir}/ckpt.pt")
            print(f"Saved checkpoint at epoch {epoch+1}")

        # Save numbered checkpoint
        if (epoch + 1) % eval_interval == 0:
            model_save = model._orig_mod if hasattr(model, "_orig_mod") else model
            model_save.save_pretrained(f"{out_dir}/model_epoch{epoch+1}")
            processor.save_pretrained(f"{out_dir}/model_epoch{epoch+1}")

    # Save final
    model_save = model._orig_mod if hasattr(model, "_orig_mod") else model
    model_save.save_pretrained(f"{out_dir}/final")
    processor.save_pretrained(f"{out_dir}/final")

    if wandb_log:
        import wandb
        wandb.finish()

    print("Training complete!")


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(checkpoint_path, suite="libero_spatial", n_episodes=50):
    """Evaluate on LIBERO benchmark."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print("LIBERO not installed. Run: pip install libero")
        return

    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    from qwen_vl_utils import process_vision_info

    MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
    }

    # Load model
    print(f"Loading {checkpoint_path}...")
    if use_lora:
        from peft import PeftModel

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=ptdtype, device_map=device
        )
        model = PeftModel.from_pretrained(base, checkpoint_path)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            checkpoint_path, torch_dtype=ptdtype, device_map=device
        )
    processor = Qwen2_5_VLProcessor.from_pretrained(checkpoint_path)
    model.eval()

    # Load stats
    stats_path = Path(checkpoint_path).parent / "dataset_stats.json"
    if not stats_path.exists():
        stats_path = Path(checkpoint_path) / "dataset_stats.json"

    with open(stats_path) as f:
        stats = json.load(f)
    action_min = np.array(stats["out_ori_act"]["min"])
    action_max = np.array(stats["out_ori_act"]["max"])

    logits_processor = NumberSpaceLogitsProcessor(processor.tokenizer)

    # System prompt
    sys_prompt = (
        f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
        f"Each action has {action_dim} dimensions. "
        f"Output a single sequence of {horizon * action_dim} integers (0-{num_bins} each), "
        f"representing the {horizon} timesteps sequentially. "
        f"Provide only space separated numbers. Nothing else."
    )

    def predict(image, instruction):
        """Get action prediction from model."""
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
        image_input = process_vision_info(messages)[0]
        inputs = processor(text=[text], images=[image_input], return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=256, logits_processor=[logits_processor], do_sample=False)

        generated = output[0, inputs["input_ids"].shape[1]:]
        action_text = processor.decode(generated, skip_special_tokens=True)

        # Decode to actions
        try:
            tokens = [int(x) for x in action_text.strip().split()]
            if len(tokens) < horizon * action_dim:
                print(f"Warning: Expected {horizon * action_dim} tokens, got {len(tokens)}: '{action_text[:50]}...'")
                tokens = tokens + [500] * (horizon * action_dim - len(tokens))  # Pad with neutral
            actions = np.array(tokens[:horizon * action_dim]).reshape(horizon, action_dim)
            actions = (actions / num_bins) * (action_max - action_min) + action_min
        except (ValueError, IndexError) as e:
            print(f"Warning: Failed to parse action text: {e}, text='{action_text[:50]}...'")
            actions = np.zeros((horizon, action_dim))

        return actions

    def preprocess_obs(obs):
        """Preprocess LIBERO observation."""
        img1 = obs["agentview_image"][::-1, ::-1]
        img2 = obs["robot0_eye_in_hand_image"][::-1, ::-1]

        # Resize and center crop
        from PIL import Image as PILImage
        pil1 = PILImage.fromarray(img1).resize((img_size, img_size))
        pil2 = PILImage.fromarray(img2).resize((img_size, img_size))

        if tile_images:
            tiled = np.concatenate([np.array(pil1), np.array(pil2)], axis=1)
            return PILImage.fromarray(tiled)
        return pil1

    # Run evaluation
    task_suite = benchmark.get_benchmark_dict()[suite]()
    max_steps = MAX_STEPS.get(suite, 300)

    print(f"\nEvaluating {suite}: {task_suite.n_tasks} tasks, {n_episodes} episodes each")

    results = {}
    total_success = 0
    total_episodes = 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)

        print(f"\nTask {task_id}: {task.name[:50]}")
        successes = []

        for ep in tqdm(range(min(n_episodes, len(init_states))), desc="Episodes"):
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(ep)
            env.reset()
            obs = env.set_init_state(init_states[ep])

            action_buffer = []  # List of (start_step, actions) tuples
            action_i = 0
            action_horizon = horizon  # Match VLA-0: execute full horizon before re-query
            done = False

            # Skip initial frames
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

            for step in range(max_steps):
                if action_i >= action_horizon or step == 0:
                    image = preprocess_obs(obs)
                    actions = predict(image, instruction)
                    action_buffer.append((step, actions))  # Track when prediction was made
                    if len(action_buffer) > 8:
                        action_buffer.pop(0)
                    action_i = 0

                # Ensemble: average overlapping predictions for current step
                ensemble_actions = []
                for pred_step, pred_actions in action_buffer:
                    idx = step - pred_step  # Index into this prediction
                    if 0 <= idx < len(pred_actions):
                        ensemble_actions.append(pred_actions[idx])

                action = np.mean(ensemble_actions, axis=0) if ensemble_actions else np.zeros(action_dim)
                action[-1] = 1.0 if action[-1] > 0 else -1.0  # Gripper binary

                obs, _, done, _ = env.step(action.tolist())
                action_i += 1

                if done:
                    break

            successes.append(float(done))
            env.close()

        rate = np.mean(successes) * 100
        results[task.name] = rate
        total_success += sum(successes)
        total_episodes += len(successes)
        print(f"  Success: {rate:.1f}%")

    mean_success = np.mean(list(results.values()))

    print(f"\n{'='*60}")
    print(f"Results - {suite}")
    print(f"{'='*60}")
    for name, rate in results.items():
        print(f"  {name[:45]:45s} {rate:5.1f}%")
    print(f"{'='*60}")
    print(f"Mean: {mean_success:.1f}%")

    # Log to wandb
    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=f"{wandb_run_name}-eval-{suite}", config={
            "checkpoint": checkpoint_path,
            "suite": suite,
            "n_episodes": n_episodes,
        })

        # Log per-task results
        for name, rate in results.items():
            safe_name = name.replace(" ", "_").replace("/", "-")[:40]
            wandb.log({f"eval/{suite}/{safe_name}": rate})

        # Log summary metrics
        wandb.log({
            f"eval/{suite}/mean_success": mean_success,
            f"eval/{suite}/total_episodes": total_episodes,
            f"eval/{suite}/total_successes": total_success,
        })

        # Log results table
        table = wandb.Table(columns=["task", "success_rate"])
        for name, rate in results.items():
            table.add_data(name, rate)
        wandb.log({f"eval/{suite}/results_table": table})

        wandb.finish()
        print(f"\nResults logged to wandb: {wandb_project}/{wandb_run_name}-eval-{suite}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default=None, help="Evaluate checkpoint")
    parser.add_argument("--suite", type=str, default="libero_spatial")
    parser.add_argument("--n_episodes", type=int, default=50)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Override config via CLI
    for k in config_keys:
        v = globals()[k]
        if isinstance(v, bool):
            parser.add_argument(f"--{k}", action="store_true" if not v else "store_false")
        elif isinstance(v, tuple):
            parser.add_argument(f"--{k}", type=str, default=str(v))
        else:
            parser.add_argument(f"--{k}", type=type(v), default=v)

    args = parser.parse_args()

    # Update globals from args
    for k in config_keys:
        if hasattr(args, k):
            v = getattr(args, k)
            if isinstance(globals()[k], tuple) and isinstance(v, str):
                v = eval(v)
            globals()[k] = v

    if args.resume:
        globals()["init_from"] = "resume"

    config = {k: globals()[k] for k in config_keys}

    if args.eval:
        evaluate(args.eval, suite=args.suite, n_episodes=args.n_episodes)
    elif not eval_only:
        train()
