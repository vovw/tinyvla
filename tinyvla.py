#!/usr/bin/env python3
"""
TinyVLA: Train VLA-0 in one file.

Usage:
    # Install
    uv venv && source .venv/bin/activate
    uv pip install -e .
    GIT_LFS_SKIP_SMUDGE=1 uv pip install git+https://github.com/huggingface/lerobot.git

    # Train (single GPU)
    python tinyvla.py

    # Train (multi-GPU with accelerate)
    accelerate launch --num_processes=4 tinyvla.py

    # Train with config file
    python tinyvla.py --config config.yaml

    # Evaluate
    uv pip install -e ".[eval]"
    python tinyvla.py --eval runs/vla0/final --suite libero_spatial

References:
    - VLA-0: https://arxiv.org/abs/2510.13054
    - vla0-trl: https://github.com/MilkClouds/vla0-trl
"""

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import LogitsProcessor, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from trl import SFTConfig, SFTTrainer, TrlParser


# =============================================================================
# Constrained Decoding
# =============================================================================


class NumberSpaceOnlyProcessor(LogitsProcessor):
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
    """LIBERO dataset for VLA training with SFTTrainer."""

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        train_suite: Optional[str] = None,
        max_episodes: Optional[int] = None,
        history: int = 1,
        horizon: int = 8,
        action_key: str = "action",
        state_key: str = "observation.state",
        cam_list: Tuple[str, ...] = ("observation.images.image", "observation.images.image2"),
        img_size: int = 224,
        crop_ratio: float = 0.875,
        tile_images: bool = True,
        brightness_aug: float = 0.2,
        contrast_aug: float = 0.2,
        saturation_aug: float = 0.2,
        hue_aug: float = 0.05,
        num_bins: int = 1000,
        action_dim: int = 7,
    ):
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        self.history = history
        self.horizon = horizon
        self.cam_list = cam_list
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.brightness_aug = brightness_aug
        self.contrast_aug = contrast_aug
        self.saturation_aug = saturation_aug
        self.hue_aug = hue_aug

        fps = 10
        delta_timestamps = {
            state_key: [-x / fps for x in range(history - 1, -1, -1)],
            action_key: [x / fps for x in range(-history, horizon)],
        }
        for cam in cam_list:
            delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]

        print(f"Loading {repo_id}...")
        episodes = None
        if train_suite is not None:
            try:
                from libero.libero import benchmark
            except Exception as e:  # pragma: no cover
                raise RuntimeError("`train_suite` requires LIBERO installed (use `uv pip install -e '.[eval]'`).") from e

            meta = LeRobotDatasetMetadata(repo_id=repo_id)
            benchmark_dict = benchmark.get_benchmark_dict()
            if train_suite not in benchmark_dict:
                raise ValueError(f"Unknown train_suite={train_suite!r}. Expected one of: {sorted(benchmark_dict.keys())}")

            ts = benchmark_dict[train_suite]()
            suite_task_prompts = [t.language for t in ts.tasks]
            suite_task_indices = {meta.get_task_index(prompt) for prompt in suite_task_prompts}
            suite_task_indices.discard(None)

            episodes = [
                ep_idx
                for ep_idx, ep in meta.episodes.items()
                if ep.get("task_index", None) in suite_task_indices
            ]
            if max_episodes is not None:
                episodes = episodes[:max_episodes]

            print(f"Filtered to {len(episodes)} episodes for suite {train_suite}")

        self.dataset = LeRobotDataset(repo_id=repo_id, episodes=episodes, delta_timestamps=delta_timestamps)
        self.action_key = action_key

        act_stats = self.dataset.meta.stats[action_key]
        self.stats = {
            "out_ori_act": {
                "min": act_stats["min"].tolist() if hasattr(act_stats["min"], "tolist") else act_stats["min"],
                "max": act_stats["max"].tolist() if hasattr(act_stats["max"], "tolist") else act_stats["max"],
            }
        }

        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
            f"Each action has {action_dim} dimensions. Output a single sequence of {horizon * action_dim} "
            f"integers (0-{num_bins} each), representing the {horizon} timesteps sequentially. "
            f"Provide only space separated numbers. Nothing else."
        )

    def __len__(self):
        return len(self.dataset)

    def _process_images(self, sample) -> List[Image.Image]:
        images = []
        for cam in self.cam_list:
            img = sample[cam]
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).byte()

            if self.crop_ratio < 1.0:
                h, w = img.shape[-2:]
                ch, cw = int(h * self.crop_ratio), int(w * self.crop_ratio)
                top = np.random.randint(0, h - ch + 1)
                left = np.random.randint(0, w - cw + 1)
                img = TF.crop(img, top, left, ch, cw)

            if self.img_size > 0:
                img = TF.resize(img, [self.img_size, self.img_size])

            img_f = img.float() / 255.0
            if self.brightness_aug > 0:
                img_f = TF.adjust_brightness(img_f, 1 + np.random.uniform(-self.brightness_aug, self.brightness_aug))
            if self.contrast_aug > 0:
                img_f = TF.adjust_contrast(img_f, 1 + np.random.uniform(-self.contrast_aug, self.contrast_aug))
            if self.saturation_aug > 0:
                img_f = TF.adjust_saturation(img_f, 1 + np.random.uniform(-self.saturation_aug, self.saturation_aug))
            if self.hue_aug > 0:
                img_f = TF.adjust_hue(img_f, np.random.uniform(-self.hue_aug, self.hue_aug))

            img = (img_f * 255).byte()
            img = rearrange(img, "c h w -> h w c").numpy()
            images.append(img)

        if self.tile_images and len(images) > 1:
            return [Image.fromarray(np.concatenate(images, axis=1))]
        return [Image.fromarray(img) for img in images]

    def _action_to_text(self, actions: np.ndarray) -> str:
        stats = self.stats["out_ori_act"]
        min_act, max_act = np.array(stats["min"]), np.array(stats["max"])
        normalized = (actions - min_act) / (max_act - min_act + 1e-8)
        discretized = np.clip(np.round(normalized * self.num_bins), 0, self.num_bins).astype(int)
        return " ".join(map(str, discretized.flatten()))

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        images = self._process_images(sample)
        instruction = sample["task"]
        actions = sample[self.action_key].numpy()[self.history:]
        action_text = self._action_to_text(actions)

        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]},
            {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        ]
        return {"messages": messages, "images": images}


# =============================================================================
# Collator
# =============================================================================


@dataclass
class VLACollator:
    """Collator with action mask augmentation."""

    processor: Any
    action_mask_aug_pct: float = 0.4

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts, image_inputs, action_texts = [], [], []

        for ex in examples:
            messages, images = ex["messages"], ex["images"]
            action_texts.append(messages[-1]["content"][0]["text"])

            formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    content = [{"type": "image", "image": images[0]} if c["type"] == "image" else c for c in msg["content"]]
                    formatted.append({"role": "user", "content": content})
                else:
                    formatted.append(msg)

            texts.append(self.processor.apply_chat_template(formatted, tokenize=False, add_generation_prompt=False, add_vision_id=False))
            image_inputs.append(process_vision_info(formatted)[0])

        inputs = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)
        labels = inputs["input_ids"].clone()

        for i, action_text in enumerate(action_texts):
            action_tokens = self.processor.tokenizer(action_text, add_special_tokens=False)["input_ids"]
            nonpad_len = inputs["attention_mask"][i].sum().item()
            sysuser_len = int(nonpad_len - len(action_tokens) - 2)
            labels[i, :sysuser_len] = -100

            aug_pct = 0.0 if random.random() < 0.1 else random.uniform(0.0, self.action_mask_aug_pct)
            mask_len = int(len(action_text) * aug_pct)
            if mask_len > 0:
                indices = [x + sysuser_len for x in random.sample(range(len(action_text)), mask_len) if x + sysuser_len < labels.size(1)]
                if indices:
                    labels[i, indices] = -100
                    inputs["input_ids"][i, indices] = 30  # '?'

        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs


# =============================================================================
# Model Loading
# =============================================================================


def load_model(model_id: str, use_flash_attention: bool = False):
    """Load Qwen2.5-VL for training."""
    kwargs = {"torch_dtype": torch.bfloat16}
    if use_flash_attention:
        try:
            kwargs["attn_implementation"] = "kernels-community/flash-attn3"
            return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        except Exception:
            try:
                kwargs["attn_implementation"] = "flash_attention_2"
                return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
            except Exception:
                kwargs.pop("attn_implementation", None)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model


def load_processor(model_id: str, img_size: int = 224, tile_images: bool = True):
    """Load processor with correct pixel settings."""
    pixels = img_size * img_size * (2 if tile_images else 1)
    return Qwen2_5_VLProcessor.from_pretrained(model_id, min_pixels=pixels, max_pixels=pixels)


# =============================================================================
# Actor for Inference
# =============================================================================


class QwenVLActor:
    """Inference wrapper for trained VLA."""

    def __init__(self, model_path: str, stats_path: Optional[str] = None, horizon: int = 8,
                 action_dim: int = 7, num_bins: int = 1000, device: str = "cuda"):
        self.horizon, self.action_dim, self.num_bins, self.device = horizon, action_dim, num_bins, device

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device)
        self.model.eval()
        self.processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        self.stats = json.load(open(stats_path)) if stats_path else None
        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next {horizon} timesteps. "
            f"Each action has {action_dim} dimensions. Output a single sequence of {horizon * action_dim} "
            f"integers (0-{num_bins} each). Provide only space separated numbers. Nothing else."
        )

    @torch.no_grad()
    def predict(self, image, instruction: str):
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=False)
        inputs = self.processor(text=[text], images=[process_vision_info(messages)[0]], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        output = self.model.generate(**inputs, max_new_tokens=256, logits_processor=[self.logits_processor], do_sample=False)
        action_text = self.processor.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        stats = self.stats["out_ori_act"]
        min_act, max_act = torch.tensor(stats["min"]), torch.tensor(stats["max"])
        try:
            tokens = [int(x) for x in action_text.strip().split()]
            actions = torch.tensor(tokens, dtype=torch.float32).reshape(-1, self.action_dim)
            if actions.shape[0] < self.horizon:
                actions = torch.cat([actions, actions[-1:].repeat(self.horizon - actions.shape[0], 1)])
            actions = (actions[:self.horizon] / self.num_bins) * (max_act - min_act) + min_act
        except Exception:
            actions = ((min_act + max_act) / 2).repeat(self.horizon, 1)
        return actions


# =============================================================================
# Evaluation
# =============================================================================


def evaluate(model_path: str, suite: str = "libero_spatial", n_episodes: int = 50,
             stats_path: Optional[str] = None, horizon: int = 8, action_horizon: int = 8,
             img_size: int = 224, crop_ratio: float = 0.875, tile_images: bool = True):
    """Evaluate on LIBERO benchmark."""
    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        print("LIBERO not installed. Run: uv pip install -e '.[eval]'")
        return None

    MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}

    if stats_path is None:
        for p in [Path(model_path) / "dataset_stats.json", Path(model_path).parent / "dataset_stats.json"]:
            if p.exists():
                stats_path = str(p)
                break

    print(f"Loading model: {model_path}")
    model = QwenVLActor(model_path, stats_path, horizon=horizon)

    def preprocess_obs(obs):
        cams = ["agentview_image", "robot0_eye_in_hand_image"]
        images = []
        for cam in cams:
            img = np.ascontiguousarray(obs[cam][::-1, ::-1])
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            if crop_ratio < 1.0:
                h, w = img.shape[-2:]
                ch, cw = int(h * crop_ratio), int(w * crop_ratio)
                img = TF.crop(img, (h - ch) // 2, (w - cw) // 2, ch, cw)
            img = TF.resize(img, [img_size, img_size])
            img = (img * 255).byte().permute(1, 2, 0).numpy()
            images.append(img)
        if tile_images:
            return Image.fromarray(np.concatenate(images, axis=1))
        return Image.fromarray(images[0])

    task_suite = benchmark.get_benchmark_dict()[suite]()
    max_steps = MAX_STEPS.get(suite, 300)
    results = {}

    print(f"\nEvaluating {suite}: {task_suite.n_tasks} tasks, {n_episodes} episodes each")

    for task_id in range(task_suite.n_tasks):
        task = task_suite.get_task(task_id)
        init_states = task_suite.get_task_init_states(task_id)
        instruction = task.language
        successes = []

        print(f"\nTask {task_id}: {task.name[:50]}")

        for ep in tqdm(range(min(n_episodes, len(init_states))), desc="Episodes"):
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
            env.seed(ep)
            env.reset()
            obs = env.set_init_state(init_states[ep])

            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

            action_i, action_chunk = action_horizon, None
            for step in range(max_steps):
                if action_i >= action_horizon:
                    action_chunk = model.predict(preprocess_obs(obs), instruction).numpy()
                    action_i = 0
                act = action_chunk[action_i].tolist()
                act[-1] = 1.0 if act[-1] > 0 else -1.0
                obs, _, done, _ = env.step(act)
                action_i += 1
                if done:
                    break

            successes.append(float(done))
            env.close()

        rate = np.mean(successes) * 100
        results[task.name] = rate
        print(f"  Success: {rate:.1f}%")

    print(f"\n{'='*60}\nResults - {suite}\n{'='*60}")
    for name, rate in results.items():
        print(f"  {name[:45]:45s} {rate:5.1f}%")
    mean_rate = float(np.mean(list(results.values()))) if results else 0.0
    print(f"{'='*60}\nMean: {mean_rate:.1f}%")
    return {"suite": suite, "n_episodes": int(n_episodes), "per_task_success": results, "mean_success": mean_rate}


# =============================================================================
# Training Arguments
# =============================================================================


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    use_flash_attention: bool = field(default=True)


@dataclass
class DataArguments:
    repo_id: str = field(default="physical-intelligence/libero")
    train_suite: Optional[str] = field(
        default=None,
        metadata={"help": "Train on a LIBERO suite only (e.g. libero_object). Requires `libero` installed."},
    )
    max_episodes: Optional[int] = field(
        default=None,
        metadata={"help": "Optionally cap the number of episodes (after filtering) for quick runs."},
    )
    history: int = field(default=1)
    horizon: int = field(default=8)
    img_size: int = field(default=224)
    crop_ratio: float = field(default=0.875)
    tile_images: bool = field(default=True)
    brightness_aug: float = field(default=0.2)
    contrast_aug: float = field(default=0.2)
    saturation_aug: float = field(default=0.2)
    hue_aug: float = field(default=0.05)


@dataclass
class VLAArguments:
    action_mask_aug_pct: float = field(default=0.4)

@dataclass
class EvalAfterTrainArguments:
    eval_after_train: bool = field(default=False, metadata={"help": "Run evaluation after training and log to W&B if enabled"})
    eval_suite: Optional[str] = field(
        default=None,
        metadata={"help": "Suite to evaluate after training (default: train_suite if set, else libero_object)"},
    )
    eval_n_episodes: int = field(default=5, metadata={"help": "Episodes per task for eval-after-train"})
    eval_action_horizon: int = field(default=8, metadata={"help": "action_horizon for eval-after-train"})

@dataclass
class LoraArguments:
    """LoRA/PEFT parameters (optional)."""

    use_lora: bool = field(default=False, metadata={"help": "Enable LoRA fine-tuning via PEFT"})
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated module names to apply LoRA to"},
    )


# =============================================================================
# Main
# =============================================================================


def train():
    """Train VLA using TRL's SFTTrainer."""
    parser = TrlParser(dataclass_types=[ModelArguments, DataArguments, VLAArguments, EvalAfterTrainArguments, LoraArguments, SFTConfig])
    model_args, data_args, vla_args, eval_args, lora_args, training_args = parser.parse_args_and_config()

    print(f"Loading model: {model_args.model_id}")
    model = load_model(model_args.model_id, model_args.use_flash_attention)
    processor = load_processor(model_args.model_id, data_args.img_size, data_args.tile_images)

    if lora_args.use_lora:
        try:
            from peft import LoraConfig, TaskType, get_peft_model
        except Exception as e:  # pragma: no cover
            raise RuntimeError("LoRA requested but `peft` is not installed. Install with: uv pip install -e '.[lora]'") from e

        target_modules = [m.strip() for m in lora_args.lora_target_modules.split(",") if m.strip()]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            lora_dropout=lora_args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    print("Loading dataset...")
    dataset = LiberoDataset(
        repo_id=data_args.repo_id,
        train_suite=data_args.train_suite,
        max_episodes=data_args.max_episodes,
        history=data_args.history,
        horizon=data_args.horizon,
        img_size=data_args.img_size,
        crop_ratio=data_args.crop_ratio,
        tile_images=data_args.tile_images,
        brightness_aug=data_args.brightness_aug,
        contrast_aug=data_args.contrast_aug,
        saturation_aug=data_args.saturation_aug,
        hue_aug=data_args.hue_aug,
    )

    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{training_args.output_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset.stats, f, indent=2)

    collator = VLACollator(processor=processor, action_mask_aug_pct=vla_args.action_mask_aug_pct)

    training_args.max_length = None
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=processor,
    )

    print("Starting training...")
    trainer.train()
    trainer.save_model(f"{training_args.output_dir}/final")
    processor.save_pretrained(f"{training_args.output_dir}/final")
    print(f"Model saved to {training_args.output_dir}/final")

    if eval_args.eval_after_train:
        suite = eval_args.eval_suite or data_args.train_suite or "libero_object"
        final_dir = f"{training_args.output_dir}/final"
        stats_path = f"{training_args.output_dir}/dataset_stats.json"
        print(f"Running eval-after-train: suite={suite}, episodes={eval_args.eval_n_episodes}")
        out = evaluate(
            final_dir,
            suite=suite,
            n_episodes=eval_args.eval_n_episodes,
            stats_path=stats_path if Path(stats_path).exists() else None,
            horizon=data_args.horizon,
            action_horizon=eval_args.eval_action_horizon,
            img_size=data_args.img_size,
            crop_ratio=data_args.crop_ratio,
            tile_images=data_args.tile_images,
        )

        # Log to W&B if enabled
        report_to = getattr(training_args, "report_to", None) or []
        if out is not None and ("wandb" in report_to or report_to == "all"):
            try:
                import wandb
            except Exception:
                wandb = None
            if wandb is not None and getattr(wandb, "run", None) is not None:
                metrics = {
                    f"eval/{suite}/mean_success": out["mean_success"],
                    f"eval/{suite}/n_episodes": out["n_episodes"],
                }
                for task_name, rate in out["per_task_success"].items():
                    metrics[f"eval/{suite}/task/{task_name}"] = float(rate)
                wandb.log(metrics)


if __name__ == "__main__":
    import argparse
    import sys

    # Check if running evaluation
    if "--eval" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--eval", type=str, required=True)
        parser.add_argument("--suite", type=str, default="libero_spatial")
        parser.add_argument("--n_episodes", type=int, default=50)
        parser.add_argument("--stats_path", type=str, default=None)
        parser.add_argument("--horizon", type=int, default=8)
        parser.add_argument("--action_horizon", type=int, default=8)
        args = parser.parse_args()
        evaluate(args.eval, args.suite, args.n_episodes, args.stats_path, args.horizon, args.action_horizon)
    else:
        train()
