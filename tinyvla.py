"""
TinyVLA: Minimal Vision-Language-Action model in one file.

A clean implementation inspired by VLA-0 and vla0-trl for robot control.
Train a robot policy that outputs actions as space-separated integers.

Usage:
    python tinyvla.py --test                      # 2-min smoke test
    python tinyvla.py --epochs 32                 # Full training
    python tinyvla.py --eval checkpoints/best     # Evaluate on LIBERO

Key techniques from VLA-0:
    - Image Tiling: Concatenate 2 camera views into 1 wide image
    - Constrained Decoding: Only allow digits, spaces, and EOS tokens
    - Masked Action Augmentation: Randomly mask digits → force visual reasoning
    - Data Augmentation: Random crop + color jitter
    - Action Ensembling: Average overlapping predictions for smoother control
    - Per-task Normalization: Learn action bounds per task for better discretization
"""

import argparse
import json
import os
import sys
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add LIBERO to path if available
if Path("/tmp/LIBERO").exists():
    sys.path.insert(0, "/tmp/LIBERO")

# =============================================================================
# Configuration
# =============================================================================

MODELS = {
    "qwen2.5-3b": "Qwen/Qwen2.5-VL-3B-Instruct",  # VLA-0 original
    "qwen2.5-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-2b": "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3-4b": "Qwen/Qwen3-VL-4B-Instruct",
}

@dataclass
class Config:
    """All hyperparameters in one place."""
    # Model
    model: str = "qwen2.5-3b"         # VLA-0 uses Qwen2.5-VL-3B
    use_lora: bool = False            # Full finetuning by default
    lora_r: int = 8
    lora_alpha: int = 16

    # Training (vla0-trl defaults)
    epochs: int = 32                  # vla0-trl default
    batch_size: int = 8               # per GPU
    lr: float = 4e-5                  # vla0-trl: 5e-6 * 8 GPUs
    grad_clip: float = 1.0            # Enable gradient clipping
    weight_decay: float = 1e-10
    num_workers: int = 8

    # VLA-0 specific
    horizon: int = 8                  # Action prediction horizon
    num_bins: int = 1000              # Discretization bins [0, 1000]
    mask_prob: float = 0.4            # Masked action augmentation
    img_size: int = 224               # Image size per camera
    tile_images: bool = True          # Tile 2 cameras → 1 wide image

    # Data augmentation
    crop_ratio: float = 0.875         # Random crop ratio
    brightness_aug: float = 0.2
    contrast_aug: float = 0.2
    saturation_aug: float = 0.2
    hue_aug: float = 0.05

    # Data
    data_repo: str = "lerobot/libero_spatial_image"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 2

    # Logging
    log_every: int = 10
    eval_every: int = 0               # Eval every N epochs (0 = only at end)
    eval_episodes: int = 5
    eval_suite: str = "libero_spatial"

    # Wandb
    wandb_project: str = "tinyvla"
    wandb_run: Optional[str] = None

    # Testing
    test: bool = False
    test_samples: int = 100
    test_batch_size: int = 2

# =============================================================================
# Constrained Decoding: Only allow numbers, spaces, and EOS
# =============================================================================

class NumberSpaceOnlyProcessor:
    """Constrains generation to numbers (0-9), spaces, and EOS."""

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        for i in range(10):
            self.allowed_tokens.add(tokenizer.encode(str(i), add_special_tokens=False)[0])
        self.allowed_tokens.add(tokenizer.encode(" ", add_special_tokens=False)[0])
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask

# =============================================================================
# ActionTokenizer: Convert between actions and text
# =============================================================================

class ActionTokenizer:
    """Convert between continuous actions and discretized text.

    Supports per-task normalization: each task can have its own action bounds.
    """

    def __init__(self, num_bins: int = 1000, mask_prob: float = 0.4,
                 horizon: int = 8, action_dim: int = 7):
        self.num_bins = num_bins
        self.mask_prob = mask_prob
        self.horizon = horizon
        self.action_dim = action_dim
        self.total_values = horizon * action_dim
        # Per-task stats: {task_key: {"min": np.array, "max": np.array}}
        self.task_stats: Dict[str, Dict[str, np.ndarray]] = {}
        # Fallback global stats
        self.global_min = None
        self.global_max = None

    def set_stats(self, action_min: np.ndarray, action_max: np.ndarray):
        """Set global fallback stats."""
        self.global_min = action_min.astype(np.float32)
        self.global_max = action_max.astype(np.float32)

    def set_task_stats(self, task_key: str, action_min: np.ndarray, action_max: np.ndarray):
        """Set per-task normalization bounds."""
        self.task_stats[task_key] = {
            "min": action_min.astype(np.float32),
            "max": action_max.astype(np.float32),
        }

    def _get_bounds(self, task_key: Optional[str] = None):
        """Get min/max bounds for a task, falling back to global."""
        if task_key and task_key in self.task_stats:
            stats = self.task_stats[task_key]
            return stats["min"], stats["max"]
        if self.global_min is not None:
            return self.global_min, self.global_max
        return None, None

    def encode(self, actions: np.ndarray, mask: bool = False, task_key: Optional[str] = None) -> str:
        """Encode actions to text using per-task normalization."""
        action_min, action_max = self._get_bounds(task_key)

        if action_min is None:
            normalized = (actions + 1) / 2
        else:
            normalized = (actions - action_min) / (action_max - action_min + 1e-8)

        binned = np.round(normalized * self.num_bins).astype(int)
        binned = np.clip(binned, 0, self.num_bins)
        text = ' '.join(map(str, binned.flatten()))

        if mask and self.mask_prob > 0:
            text = self._apply_mask(text)

        return text

    def decode(self, text: str, task_key: Optional[str] = None) -> np.ndarray:
        """Decode text to actions using per-task normalization."""
        ints = []
        for token in text.strip().split():
            try:
                val = int(token)
                if 0 <= val <= self.num_bins:
                    ints.append(val)
            except ValueError:
                continue
            if len(ints) >= self.total_values:
                break

        neutral = self.num_bins // 2
        while len(ints) < self.total_values:
            ints.append(neutral)

        normalized = np.array(ints[:self.total_values]) / self.num_bins
        normalized = normalized.reshape(self.horizon, self.action_dim)

        action_min, action_max = self._get_bounds(task_key)

        if action_min is None:
            actions = normalized * 2 - 1
        else:
            actions = normalized * (action_max - action_min) + action_min

        return actions.astype(np.float32)

    def _apply_mask(self, text: str) -> str:
        """Masked Action Augmentation - randomly mask digits with '?'."""
        if random.random() < 0.1:
            return text
        actual_prob = random.uniform(0.0, self.mask_prob)
        chars = list(text)
        for i in range(len(chars)):
            if chars[i].isdigit() and random.random() < actual_prob:
                chars[i] = '?'
        return ''.join(chars)

# =============================================================================
# ActionEnsemble: Temporal smoothing
# =============================================================================

class ActionEnsemble:
    """Temporal ensemble of action predictions for smoother control."""

    def __init__(self, horizon: int = 8):
        self.horizon = horizon
        self.buffer: List[np.ndarray] = []

    def add(self, actions: np.ndarray):
        self.buffer.append(actions)
        if len(self.buffer) > self.horizon:
            self.buffer.pop(0)

    def get_action(self) -> np.ndarray:
        if len(self.buffer) == 0:
            return np.zeros(7, dtype=np.float32)
        current_actions = []
        for i, pred in enumerate(self.buffer):
            timestep = len(self.buffer) - 1 - i
            if timestep < len(pred):
                current_actions.append(pred[timestep])
        if not current_actions:
            return np.zeros(7, dtype=np.float32)
        return np.mean(current_actions, axis=0).astype(np.float32)

    def reset(self):
        self.buffer.clear()

# =============================================================================
# Image Augmentation
# =============================================================================

def augment_image(img: np.ndarray, cfg: Config) -> np.ndarray:
    """Apply random crop and color augmentation to image."""
    h, w = img.shape[:2]

    # Random crop
    if cfg.crop_ratio < 1.0:
        crop_h = int(h * cfg.crop_ratio)
        crop_w = int(w * cfg.crop_ratio)
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        img = img[top:top+crop_h, left:left+crop_w]

    # Resize to target size
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((cfg.img_size, cfg.img_size), Image.BILINEAR)

    # Color augmentation using PIL
    from PIL import ImageEnhance

    if cfg.brightness_aug > 0:
        factor = 1 + random.uniform(-cfg.brightness_aug, cfg.brightness_aug)
        pil_img = ImageEnhance.Brightness(pil_img).enhance(factor)

    if cfg.contrast_aug > 0:
        factor = 1 + random.uniform(-cfg.contrast_aug, cfg.contrast_aug)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(factor)

    if cfg.saturation_aug > 0:
        factor = 1 + random.uniform(-cfg.saturation_aug, cfg.saturation_aug)
        pil_img = ImageEnhance.Color(pil_img).enhance(factor)

    return np.array(pil_img)

def tile_images(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Tile two images horizontally into one wide image."""
    return np.concatenate([img1, img2], axis=1)

# =============================================================================
# Dataset
# =============================================================================

class LiberoDataset(Dataset):
    """LIBERO dataset with image tiling and augmentation."""

    def __init__(
        self,
        repo_id: str,
        action_tokenizer: ActionTokenizer,
        config: Config,
        split: str = "train",
        max_samples: Optional[int] = None,
    ):
        self.tokenizer = action_tokenizer
        self.config = config
        self.horizon = config.horizon
        self.split = split
        self.augment = (split == "train")

        print(f"Loading dataset from {repo_id}...")
        from datasets import load_dataset

        self.dataset = load_dataset(repo_id, split="train")
        columns = self.dataset.column_names
        print(f"Dataset columns: {columns}")

        # Find image keys
        camera_keys = [k for k in columns if 'image' in k.lower() or 'rgb' in k.lower()]
        if not camera_keys:
            raise ValueError(f"No image keys found. Available: {columns}")
        self.img_key1 = camera_keys[0]
        self.img_key2 = camera_keys[1] if len(camera_keys) > 1 else camera_keys[0]

        # Action key
        action_keys = [k for k in columns if 'action' in k.lower()]
        self.action_key = action_keys[0] if action_keys else "action"

        # Task key
        self.task_key = None
        for key in ["task", "language_instruction", "instruction", "task_description"]:
            if key in columns:
                self.task_key = key
                break

        # Task descriptions from tasks.jsonl
        self.task_index_key = "task_index" if "task_index" in columns else None
        self.task_descriptions = {}
        if self.task_index_key and not self.task_key:
            self._load_task_descriptions(repo_id)

        # Episode index
        self.episode_key = None
        for key in ["episode_index", "episode_id", "episode"]:
            if key in columns:
                self.episode_key = key
                break

        print(f"Using images: {self.img_key1}, {self.img_key2} (tiled: {config.tile_images})")
        print(f"Using actions: {self.action_key}")

        self._build_episode_index()
        self._compute_action_stats()

        if max_samples:
            self.indices = list(range(min(max_samples, len(self.valid_indices))))
        else:
            self.indices = list(range(len(self.valid_indices)))

        print(f"{split.upper()}: {len(self.indices)} samples")

    def _load_task_descriptions(self, repo_id: str):
        from huggingface_hub import hf_hub_download
        try:
            tasks_file = hf_hub_download(
                repo_id=repo_id,
                filename="meta/tasks.jsonl",
                repo_type="dataset",
            )
            with open(tasks_file) as f:
                for line in f:
                    item = json.loads(line)
                    self.task_descriptions[item["task_index"]] = item["task"]
            print(f"Loaded {len(self.task_descriptions)} task descriptions")
        except Exception as e:
            print(f"Could not load task descriptions: {e}")

    def _build_episode_index(self):
        if self.episode_key is None:
            self.valid_indices = list(range(len(self.dataset) - self.horizon))
            self.episode_starts = {0: 0}
            self.episode_ends = {0: len(self.dataset)}
            return

        episodes = {}
        for i in range(len(self.dataset)):
            ep_idx = self.dataset[i][self.episode_key]
            if ep_idx not in episodes:
                episodes[ep_idx] = []
            episodes[ep_idx].append(i)

        self.valid_indices = []
        self.episode_starts = {}
        self.episode_ends = {}

        for ep_idx, indices in episodes.items():
            indices = sorted(indices)
            self.episode_starts[ep_idx] = indices[0]
            self.episode_ends[ep_idx] = indices[-1] + 1
            for i in indices[:-self.horizon + 1]:
                self.valid_indices.append(i)

    def _compute_action_stats(self):
        """Compute per-task action statistics for normalization."""
        # Group actions by task
        task_actions: Dict[str, List[np.ndarray]] = {}
        all_actions = []

        print("Computing per-task action statistics...")
        for i in tqdm(range(len(self.dataset)), desc="Scanning actions", leave=False):
            sample = self.dataset[i]
            action = sample[self.action_key]
            if isinstance(action, list):
                action = np.array(action)
            all_actions.append(action)

            # Get task key
            if self.task_key:
                task = sample[self.task_key]
                if isinstance(task, bytes):
                    task = task.decode()
            elif self.task_index_key and self.task_descriptions:
                task_idx = sample[self.task_index_key]
                task = self.task_descriptions.get(task_idx, "default")
            else:
                task = "default"

            if task not in task_actions:
                task_actions[task] = []
            task_actions[task].append(action)

        # Compute per-task stats
        for task, actions in task_actions.items():
            actions_arr = np.stack(actions)
            task_min = actions_arr.min(axis=0)
            task_max = actions_arr.max(axis=0)
            self.tokenizer.set_task_stats(task, task_min, task_max)

        # Also set global stats as fallback
        all_actions = np.stack(all_actions)
        global_min = all_actions.min(axis=0)
        global_max = all_actions.max(axis=0)
        self.tokenizer.set_stats(global_min, global_max)

        print(f"Per-task stats: {len(task_actions)} tasks")
        print(f"Global action range: [{global_min.min():.3f}, {global_max.max():.3f}]")

    def __len__(self):
        return len(self.indices)

    def _get_image(self, sample, key) -> np.ndarray:
        img = sample[key]
        if isinstance(img, Image.Image):
            return np.array(img)
        if torch.is_tensor(img):
            img = img.numpy()
        if isinstance(img, list):
            img = np.array(img)
        if isinstance(img, dict) and "bytes" in img:
            from io import BytesIO
            img = Image.open(BytesIO(img["bytes"]))
            return np.array(img)
        while img.ndim > 3:
            img = img.squeeze(0)
        if img.dtype in (np.float32, np.float64):
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
        if img.ndim == 3 and img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        return img

    def __getitem__(self, idx):
        real_idx = self.valid_indices[self.indices[idx]]
        sample = self.dataset[real_idx]

        # Get images
        img1 = self._get_image(sample, self.img_key1)
        img2 = self._get_image(sample, self.img_key2)

        # Apply augmentation
        if self.augment:
            img1 = augment_image(img1, self.config)
            img2 = augment_image(img2, self.config)
        else:
            img1 = np.array(Image.fromarray(img1).resize((self.config.img_size, self.config.img_size)))
            img2 = np.array(Image.fromarray(img2).resize((self.config.img_size, self.config.img_size)))

        # Tile images if enabled
        if self.config.tile_images:
            image = tile_images(img1, img2)  # 224x448
        else:
            image = (img1, img2)

        # Get instruction
        if self.task_key:
            instruction = sample[self.task_key]
        elif self.task_index_key and self.task_descriptions:
            task_idx = sample[self.task_index_key]
            instruction = self.task_descriptions.get(task_idx, "Complete the task")
        else:
            instruction = "Complete the task"

        if isinstance(instruction, bytes):
            instruction = instruction.decode()

        # Get action chunk
        actions = []
        current_ep = sample.get(self.episode_key, 0) if self.episode_key else 0
        ep_end = self.episode_ends.get(current_ep, len(self.dataset))

        for i in range(self.horizon):
            future_idx = min(real_idx + i, ep_end - 1)
            future_sample = self.dataset[future_idx]
            action = future_sample[self.action_key]
            if isinstance(action, list):
                action = np.array(action)
            if torch.is_tensor(action):
                action = action.numpy()
            actions.append(action)

        actions = np.stack(actions)
        action_text = self.tokenizer.encode(actions, mask=self.augment, task_key=instruction)

        return image, instruction, action_text, actions

# =============================================================================
# TinyVLA Model
# =============================================================================

class TinyVLA:
    """Vision-Language-Action model using Qwen2.5-VL."""

    def __init__(
        self,
        model_name: str = "qwen2.5-3b",
        horizon: int = 8,
        action_dim: int = 7,
        num_bins: int = 1000,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
        tile_images: bool = True,
        device: str = "cuda",
    ):
        self.device = device
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.tile_images = tile_images
        self.model_path = MODELS.get(model_name, model_name)

        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
            f"Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers (0-{num_bins} each), "
            f"representing the {horizon} timesteps sequentially. "
            f"Provide only space separated numbers. Nothing else."
        )

        print(f"Loading {self.model_path}...")

        if "Qwen3" in self.model_path:
            from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
            ModelClass = Qwen3VLForConditionalGeneration
        else:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            ModelClass = Qwen2_5_VLForConditionalGeneration

        self.model = ModelClass.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        # Processor with correct pixel settings for tiled images
        img_pixels = 224 * 224 * (2 if tile_images else 1)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=img_pixels,
            max_pixels=img_pixels,
        )

        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        if use_lora:
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        self.action_tokenizer = ActionTokenizer(
            num_bins=num_bins,
            horizon=horizon,
            action_dim=action_dim,
        )

        print(f"Loaded {self.model_path}")

    def _format_message(self, image: Image.Image, instruction: str, action_txt: Optional[str] = None):
        """Format a single example as chat messages."""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        if action_txt is not None:
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": action_txt}],
            })

        return messages

    def _prepare_inputs(
        self,
        images: List[Image.Image],
        instructions: List[str],
        action_texts: Optional[List[str]] = None,
    ):
        """Prepare batch inputs for training/inference."""
        from qwen_vl_utils import process_vision_info

        batch_size = len(images)
        messages_batch = []

        for i in range(batch_size):
            action_txt = action_texts[i] if action_texts else None
            messages = self._format_message(images[i], instructions[i], action_txt)
            messages_batch.append(messages)

        texts = [
            self.processor.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=(action_texts is None),
                add_vision_id=False,  # Match vla0-trl
            )
            for msg in messages_batch
        ]

        image_inputs = [process_vision_info(msg)[0] for msg in messages_batch]

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        )

        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    def forward(self, images, instructions, action_texts):
        """Training forward pass. Returns loss."""
        batch_size = len(images)

        inputs = self._prepare_inputs(images, instructions, action_texts)
        labels = inputs["input_ids"].clone()

        # Compute label mask (only train on action tokens)
        for i in range(batch_size):
            action_tokens = self.processor.tokenizer(
                action_texts[i], add_special_tokens=False
            )["input_ids"]
            action_len = len(action_tokens)

            nonpad_len = inputs["attention_mask"][i].sum().item()
            sysuser_len = int(nonpad_len - action_len - 2)  # -2 for end tokens

            labels[i, :sysuser_len] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            labels=labels,
        )

        return outputs.loss

    @torch.no_grad()
    def generate(self, images: List[Image.Image], instructions: List[str]) -> tuple:
        """Generate action predictions with constrained decoding."""
        inputs = self._prepare_inputs(images, instructions)

        generated_ids = self.model.generate(
            **inputs,
            logits_processor=[self.logits_processor],
            max_new_tokens=256,
            do_sample=False,
        )

        generated_ids = [
            out[len(inp):] for inp, out in zip(inputs["input_ids"], generated_ids)
        ]

        texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # Decode with per-task normalization
        actions = [
            self.action_tokenizer.decode(text, task_key=instr)
            for text, instr in zip(texts, instructions)
        ]
        return actions, texts

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.processor.save_pretrained(path)

        # Convert per-task stats to serializable format
        task_stats_serializable = {}
        for task_key, task_data in self.action_tokenizer.task_stats.items():
            task_stats_serializable[task_key] = {
                "min": task_data["min"].tolist(),
                "max": task_data["max"].tolist(),
            }

        stats = {
            "global_min": self.action_tokenizer.global_min.tolist() if self.action_tokenizer.global_min is not None else None,
            "global_max": self.action_tokenizer.global_max.tolist() if self.action_tokenizer.global_max is not None else None,
            "task_stats": task_stats_serializable,
            "horizon": self.horizon,
            "action_dim": self.action_dim,
            "num_bins": self.num_bins,
            "tile_images": self.tile_images,
        }
        with open(path / "action_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        print(f"Saved to {path} ({len(task_stats_serializable)} task stats)")

    def load(self, path: str):
        path = Path(path)

        stats_path = path / "action_stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)

            # Load global stats (check both old and new key names for backwards compat)
            global_min = stats.get("global_min") or stats.get("action_min")
            global_max = stats.get("global_max") or stats.get("action_max")
            if global_min:
                self.action_tokenizer.set_stats(
                    np.array(global_min),
                    np.array(global_max),
                )

            # Load per-task stats
            task_stats = stats.get("task_stats", {})
            for task_key, task_data in task_stats.items():
                self.action_tokenizer.set_task_stats(
                    task_key,
                    np.array(task_data["min"]),
                    np.array(task_data["max"]),
                )

            self.tile_images = stats.get("tile_images", True)
            print(f"Loaded {len(task_stats)} per-task action stats")

        adapter_config = path / "adapter_config.json"
        if adapter_config.exists():
            from peft import PeftModel
            base_model = self.model
            if hasattr(base_model, "base_model"):
                base_model = base_model.base_model
            self.model = PeftModel.from_pretrained(base_model, str(path))
            print(f"Loaded LoRA weights from {path}")
        else:
            config_path = path / "config.json"
            is_qwen3 = False
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                is_qwen3 = "qwen3" in config.get("model_type", "").lower()

            if is_qwen3:
                from transformers import Qwen3VLForConditionalGeneration
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    str(path), torch_dtype=torch.bfloat16, device_map="auto",
                )
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    str(path), torch_dtype=torch.bfloat16, device_map="auto",
                )
            print(f"Loaded full model from {path}")

# =============================================================================
# Training
# =============================================================================

def train(config: Config):
    """Main training loop."""
    torch.set_float32_matmul_precision('medium')

    # Initialize wandb
    try:
        import wandb
        run_name = config.wandb_run or f"{config.model}-{config.epochs}ep"
        wandb.init(project=config.wandb_project, name=run_name, config=vars(config))
        print(f"Logging to wandb: {config.wandb_project}/{run_name}")
    except ImportError:
        wandb = None
        print("wandb not installed, skipping logging")

    action_tokenizer = ActionTokenizer(
        num_bins=config.num_bins,
        mask_prob=config.mask_prob,
        horizon=config.horizon,
    )

    dataset = LiberoDataset(
        repo_id=config.data_repo,
        action_tokenizer=action_tokenizer,
        config=config,
        split="train",
        max_samples=config.test_samples if config.test else None,
    )

    def collate_fn(batch):
        images, instructions, action_texts, actions = zip(*batch)
        # Convert to PIL images
        if config.tile_images:
            pil_images = [Image.fromarray(img) for img in images]
        else:
            # If not tiling, we'd have tuples - but we tile by default
            pil_images = [Image.fromarray(img) for img in images]
        return pil_images, list(instructions), list(action_texts), actions

    batch_size = config.test_batch_size if config.test else config.batch_size
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Train: {len(dataset)} samples, {len(loader)} batches")
    print(f"Batch size: {batch_size}")

    vla = TinyVLA(
        model_name=config.model,
        horizon=config.horizon,
        num_bins=config.num_bins,
        use_lora=config.use_lora,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        tile_images=config.tile_images,
    )

    vla.action_tokenizer = action_tokenizer

    optimizer = torch.optim.AdamW(
        vla.model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    Path(config.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    vla.model.train()

    epochs = 1 if config.test else config.epochs
    global_step = 0
    best_success_rate = 0.0

    for epoch in range(epochs):
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0.0
        n_batches = 0

        for images, instructions, action_texts, _ in pbar:
            loss = vla.forward(images, instructions, action_texts)

            optimizer.zero_grad()
            loss.backward()

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(vla.model.parameters(), config.grad_clip)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if wandb and global_step % config.log_every == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/epoch": epoch + n_batches / len(loader),
                    "train/lr": config.lr,
                }, step=global_step)

            pbar.set_postfix(loss=f"{epoch_loss/n_batches:.4f}", step=global_step)

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}")

        if wandb:
            wandb.log({"train/epoch_loss": avg_loss, "train/epoch": epoch + 1}, step=global_step)

        # Evaluation
        if config.eval_every > 0 and (epoch + 1) % config.eval_every == 0:
            print(f"\nRunning evaluation on {config.eval_suite}...")
            vla.model.eval()
            results = quick_eval(vla, config.eval_suite, config.eval_episodes)
            vla.model.train()

            if wandb:
                wandb.log({
                    "eval/success_rate": results["mean_success"],
                    "eval/epoch": epoch + 1,
                }, step=global_step)

            print(f"Eval success rate: {results['mean_success']:.1f}%")

            if results["mean_success"] > best_success_rate:
                best_success_rate = results["mean_success"]
                vla.save(Path(config.checkpoint_dir) / "best")
                print(f"New best: {best_success_rate:.1f}%")

        # Save checkpoint
        if (epoch + 1) % config.save_every == 0 or (epoch + 1) == epochs:
            vla.save(Path(config.checkpoint_dir) / f"epoch_{epoch+1}")

    vla.save(Path(config.checkpoint_dir) / "final")

    if wandb:
        wandb.finish()

    print(f"Training complete!")

def quick_eval(vla, suite: str = "libero_spatial", n_episodes: int = 5) -> dict:
    """Quick evaluation during training."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print("LIBERO not installed, skipping eval")
        return {"mean_success": 0.0, "per_task": {}}

    MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }

    task_suite = benchmark.get_benchmark_dict()[suite]()
    max_steps = MAX_STEPS.get(suite, 300)

    results = {}
    total_success = 0
    total_episodes = 0

    print(f"Evaluating {task_suite.n_tasks} tasks, {n_episodes} episodes each...")

    for task_id in tqdm(range(task_suite.n_tasks), desc="Tasks"):
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)

        successes = []

        for episode in range(min(n_episodes, len(init_states))):
            task_bddl = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )

            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(episode)
            env.reset()
            obs = env.set_init_state(init_states[episode])

            ensemble = ActionEnsemble(horizon=vla.horizon)
            action_horizon = 5
            action_i = action_horizon
            done = False

            for step in range(max_steps):
                if action_i >= action_horizon:
                    img1 = obs['agentview_image'][::-1, ::-1]
                    img2 = obs['robot0_eye_in_hand_image'][::-1, ::-1]

                    # Resize images
                    pil1 = Image.fromarray(img1).resize((224, 224))
                    pil2 = Image.fromarray(img2).resize((224, 224))

                    # Tile images if model expects tiled input
                    if vla.tile_images:
                        tiled = np.concatenate([np.array(pil1), np.array(pil2)], axis=1)
                        input_image = Image.fromarray(tiled)
                    else:
                        input_image = pil1  # Fallback

                    with torch.no_grad():
                        actions, _ = vla.generate([input_image], [instruction])

                    ensemble.add(actions[0])
                    action_i = 0

                action = ensemble.get_action()
                action[-1] = 1.0 if action[-1] > 0 else -1.0

                obs, reward, done, info = env.step(action.tolist())
                action_i += 1

                if done:
                    break

            successes.append(float(done))
            env.close()

        success_rate = np.mean(successes) * 100
        results[task.name] = success_rate
        total_success += sum(successes)
        total_episodes += len(successes)
        tqdm.write(f"  Task {task_id}: {success_rate:.0f}%")

    mean_success = total_success / max(total_episodes, 1) * 100
    print(f"Overall: {mean_success:.1f}%")

    return {"mean_success": mean_success, "per_task": results}

# =============================================================================
# Full Evaluation
# =============================================================================

def evaluate(checkpoint_path: str, suite: str = "libero_spatial", n_episodes: int = 50):
    """Full evaluation on LIBERO benchmark."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print("LIBERO not installed.")
        return

    MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }

    checkpoint_path = Path(checkpoint_path)

    # Detect model type
    config_path = checkpoint_path / "config.json"
    model_name = "qwen2.5-3b"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        if "qwen3" in config.get("model_type", "").lower():
            model_name = "qwen3-2b"

    print(f"Loading {checkpoint_path}...")
    vla = TinyVLA(model_name=model_name)
    vla.load(str(checkpoint_path))
    vla.model.eval()

    task_suite = benchmark.get_benchmark_dict()[suite]()
    max_steps = MAX_STEPS.get(suite, 300)

    print(f"\nEvaluating on {suite}: {task_suite.n_tasks} tasks, {n_episodes} episodes each")

    results = {}
    total_success = 0
    total_episodes = 0

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        instruction = task.language
        init_states = task_suite.get_task_init_states(task_id)

        print(f"\nTask {task_id}: {task.name}")

        successes = []

        for episode in tqdm(range(min(n_episodes, len(init_states))), desc=f"Task {task_id}"):
            task_bddl = os.path.join(
                get_libero_path("bddl_files"),
                task.problem_folder,
                task.bddl_file,
            )

            env = OffScreenRenderEnv(
                bddl_file_name=task_bddl,
                camera_heights=256,
                camera_widths=256,
            )
            env.seed(episode)
            env.reset()
            obs = env.set_init_state(init_states[episode])

            ensemble = ActionEnsemble(horizon=vla.horizon)
            action_horizon = 5
            action_i = action_horizon
            done = False

            for step in range(max_steps):
                if action_i >= action_horizon:
                    img1 = obs['agentview_image'][::-1, ::-1]
                    img2 = obs['robot0_eye_in_hand_image'][::-1, ::-1]

                    pil1 = Image.fromarray(img1).resize((224, 224))
                    pil2 = Image.fromarray(img2).resize((224, 224))

                    if vla.tile_images:
                        tiled = np.concatenate([np.array(pil1), np.array(pil2)], axis=1)
                        input_image = Image.fromarray(tiled)
                    else:
                        input_image = pil1

                    actions, _ = vla.generate([input_image], [instruction])
                    ensemble.add(actions[0])
                    action_i = 0

                action = ensemble.get_action()
                action[-1] = 1.0 if action[-1] > 0 else -1.0

                obs, reward, done, info = env.step(action.tolist())
                action_i += 1

                if done:
                    break

            successes.append(float(done))
            env.close()

        success_rate = np.mean(successes) * 100
        results[task.name] = success_rate
        total_success += sum(successes)
        total_episodes += len(successes)
        print(f"{task.name}: {success_rate:.1f}%")

    print(f"\n{'='*60}")
    print(f"Results - {suite}")
    print(f"{'='*60}")
    for name, rate in results.items():
        print(f"{name[:40]:40s}: {rate:5.1f}%")
    print(f"\nMean: {np.mean(list(results.values())):5.1f}%")

# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TinyVLA: Train robot policies with VLMs")

    parser.add_argument('--eval', type=str, default=None, help='Checkpoint path for evaluation')
    parser.add_argument('--test', action='store_true', help='Quick smoke test')

    parser.add_argument('--model', type=str, default='qwen2.5-3b', choices=list(MODELS.keys()))
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA')

    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=4e-5)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    parser.add_argument('--horizon', type=int, default=8)
    parser.add_argument('--mask-prob', type=float, default=0.4)
    parser.add_argument('--no-tile', action='store_true', help='Disable image tiling')

    parser.add_argument('--data-repo', type=str, default='lerobot/libero_spatial_image')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')

    parser.add_argument('--wandb-project', type=str, default='tinyvla')
    parser.add_argument('--wandb-run', type=str, default=None)

    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--eval-episodes', type=int, default=5)
    parser.add_argument('--suite', type=str, default='libero_spatial')
    parser.add_argument('--n-episodes', type=int, default=50)

    args = parser.parse_args()

    if args.eval:
        evaluate(args.eval, suite=args.suite, n_episodes=args.n_episodes)
    else:
        config = Config(
            model=args.model,
            use_lora=args.use_lora,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            grad_clip=args.grad_clip,
            horizon=args.horizon,
            mask_prob=args.mask_prob,
            tile_images=not args.no_tile,
            data_repo=args.data_repo,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project,
            wandb_run=args.wandb_run,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
            eval_suite=args.suite,
            test=args.test,
        )
        train(config)

if __name__ == '__main__':
    main()
