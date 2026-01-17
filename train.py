import argparse
import dataclasses
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image
from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import LogitsProcessor, Qwen2_5_VLProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration,
)
from trl import SFTConfig, SFTTrainer, TrlParser


@dataclass
class ModelArguments:
    model_id: str = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    use_flash_attention: bool = field(default=False)

    # LoRA
    use_lora: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = field(default="none")
    lora_target_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": "Comma-separated target module name suffixes. If unset, inferred from model."
        },
    )

    # Inference
    torch_compile: bool = field(default=False)


@dataclass
class DataArguments:
    repo_id: str = field(default="physical-intelligence/libero")
    suite_name: str = field(default="libero_10")
    history: int = field(default=1)
    horizon: int = field(default=8)
    action_key: str = field(default="actions")
    state_key: str = field(default="state")
    cam_list: Tuple[str, ...] = field(default_factory=lambda: ("image", "wrist_image"))

    img_size: int = field(default=224)
    crop_ratio: float = field(default=0.875)
    tile_images: bool = field(default=True)
    brightness_aug: float = field(default=0.2)
    contrast_aug: float = field(default=0.2)
    saturation_aug: float = field(default=0.2)
    hue_aug: float = field(default=0.05)

    num_bins: int = field(default=1000)
    action_dim: int = field(default=7)

    # Subset cache
    cache_dir: str = field(default="./cache")
    rebuild_cache: bool = field(default=False)
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Optional cap for smoke tests."}
    )


@dataclass
class BudgetArguments:
    # Step-budgeted by default to keep within ~24 single-H100 hours.
    step_budget: int = field(default=13000)
    seed: int = field(default=7)


@dataclass
class EvalArguments:
    # Local eval only on libero_10
    model_path: Optional[str] = field(default=None, metadata={"help": "Checkpoint/final dir."})
    stats_path: Optional[str] = field(default=None, metadata={"help": "dataset_stats.json path."})
    task_name: Optional[str] = field(default=None)

    episodes_per_task: int = field(default=5)
    max_tasks: Optional[int] = field(default=None)
    save_video: bool = field(default=False)
    frame_skip: int = field(default=10)
    action_horizon: int = field(default=8)
    ensemble_prediction: int = field(default=1)
    ensemble_version: int = field(default=1)
    ensemble_weight: float = field(default=0.5)
    log_dir: Optional[str] = field(default=None)


@dataclass
class VLACollator:
    """Collator for VLA training that handles image + text batching.

    This collator:
    1. Applies chat template to format messages
    2. Processes images through Qwen's vision encoder
    3. Creates labels with masked system/user tokens
    4. Applies action mask augmentation (masking random action tokens)
    """

    processor: Any  # Qwen2_5_VLProcessor
    action_mask_aug_pct: float = 0.4

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_size = len(examples)

        # Apply chat template
        texts = []
        image_inputs = []
        action_texts = []

        for example in examples:
            messages = example["messages"]
            images = example["images"]

            # Extract action text (assistant content is [{"type": "text", "text": ...}])
            action_text = messages[-1]["content"][0]["text"]
            action_texts.append(action_text)

            # Format for Qwen processor - inject actual images
            formatted = []
            for msg in messages:
                if msg["role"] == "user":
                    content = []
                    for item in msg["content"]:
                        if item["type"] == "image":
                            content.append({"type": "image", "image": images[0]})
                        else:
                            content.append(item)
                    formatted.append({"role": "user", "content": content})
                else:
                    # system and assistant already in correct format
                    formatted.append(msg)

            text = self.processor.apply_chat_template(
                formatted,
                tokenize=False,
                add_generation_prompt=False,
                add_vision_id=False,
            )
            texts.append(text)
            image_inputs.append(process_vision_info(formatted)[0])

        # Tokenize batch
        model_inputs = self.processor(
            text=texts, images=image_inputs, return_tensors="pt", padding=True
        )

        # Create labels (mask system + user tokens)
        labels = model_inputs["input_ids"].clone()

        for i in range(batch_size):
            # Compute length of system + user portion
            action_tokens = self.processor.tokenizer(
                action_texts[i], add_special_tokens=False
            )["input_ids"]
            action_len = len(action_tokens)

            # Total non-pad tokens
            nonpad_len = model_inputs["attention_mask"][i].sum().item()
            # System + user length = total - action - 2 (assistant end tokens)
            sysuser_len = int(nonpad_len - action_len - 2)

            # Mask system + user
            labels[i, :sysuser_len] = -100

            # Apply action mask augmentation (matches original QwenActor logic)
            # BUG: original implementation mask not only action token but also spacebar but I maintain the original logic for now.
            seq_len = labels.size(1)
            if random.random() < 0.1:
                aug_pct = 0.0
            else:
                aug_pct = random.uniform(0.0, self.action_mask_aug_pct)

            mask_len = int(len(action_texts[i]) * aug_pct)
            if mask_len > 0:
                mask_indices = random.sample(range(len(action_texts[i])), mask_len)
                mask_indices = [x + sysuser_len for x in mask_indices]
                mask_indices = [idx for idx in mask_indices if idx < seq_len]
                if mask_indices:
                    labels[i, mask_indices] = -100
                    model_inputs["input_ids"][i, mask_indices] = 30  # '?' token

        # Mask pad tokens (151643 = <|endoftext|>)
        # Note: EOS is 151645 (<|im_end|>), which should NOT be masked for training
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        model_inputs["labels"] = labels
        return model_inputs


class LiberoDataset(Dataset):
    """LIBERO dataset formatted for VLA-0 training with SFTTrainer.

    Returns samples in the format expected by SFTTrainer for VLMs:
    - messages: conversation format with system, user (with images), assistant
    - images: list of PIL images
    """

    def __init__(
        self,
        repo_id: str = "physical-intelligence/libero",
        history: int = 1,
        horizon: int = 8,
        action_key: str = "actions",
        state_key: str = "state",
        cam_list: Tuple[str, ...] = ("image", "wrist_image"),
        img_size: int = 224,
        crop_ratio: float = 0.875,
        tile_images: bool = True,
        brightness_aug: float = 0.2,
        contrast_aug: float = 0.2,
        saturation_aug: float = 0.2,
        hue_aug: float = 0.05,
        num_bins: int = 1000,
        action_dim: int = 7,
        episodes: Optional[List[int]] = None,
    ):
        self.history = history
        self.horizon = horizon
        self.cam_list = cam_list
        self.img_size = img_size
        self.crop_ratio = crop_ratio
        self.tile_images = tile_images
        self.num_bins = num_bins
        self.action_dim = action_dim

        # Augmentation params
        self.brightness_aug = brightness_aug
        self.contrast_aug = contrast_aug
        self.saturation_aug = saturation_aug
        self.hue_aug = hue_aug

        # Build delta_timestamps for history and horizon
        # Matches original RoboVerse: actions from -history to horizon-1
        fps = 10  # LIBERO dataset fps
        delta_timestamps = {
            state_key: [-x / fps for x in range(history - 1, -1, -1)],
            action_key: [x / fps for x in range(-history, horizon)],
        }
        for cam in cam_list:
            delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            delta_timestamps=delta_timestamps,
            episodes=episodes,
        )

        self.action_key = action_key
        self.state_key = state_key

        # Compute stats (convert to lists for JSON serialization)
        act_stats = self.dataset.meta.stats[self.action_key]
        self.stats = {
            "out_ori_act": {
                "min": act_stats["min"].tolist()
                if hasattr(act_stats["min"], "tolist")
                else act_stats["min"],
                "max": act_stats["max"].tolist()
                if hasattr(act_stats["max"], "tolist")
                else act_stats["max"],
            },
        }

        # System prompt
        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next "
            f"{horizon} timesteps. Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers "
            f"(0-{num_bins} each), representing the {horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _process_images(self, sample: Dict) -> List[Image.Image]:
        """Extract and process images from sample."""
        images = []
        for cam in self.cam_list:
            img = sample[cam]  # (C, H, W) or (1, C, H, W)
            if img.ndim == 4:
                img = img[0]
            img = (img * 255).byte()

            # Apply augmentations
            if self.crop_ratio < 1.0:
                h, w = img.shape[-2:]
                crop_h, crop_w = int(h * self.crop_ratio), int(w * self.crop_ratio)
                top = np.random.randint(0, h - crop_h + 1)
                left = np.random.randint(0, w - crop_w + 1)
                img = TF.crop(img, top, left, crop_h, crop_w)

            if self.img_size > 0:
                img = TF.resize(img, [self.img_size, self.img_size])

            # Color augmentation
            img_float = img.float() / 255.0
            if self.brightness_aug > 0:
                img_float = TF.adjust_brightness(
                    img_float,
                    1 + np.random.uniform(-self.brightness_aug, self.brightness_aug),
                )
            if self.contrast_aug > 0:
                img_float = TF.adjust_contrast(
                    img_float,
                    1 + np.random.uniform(-self.contrast_aug, self.contrast_aug),
                )
            if self.saturation_aug > 0:
                img_float = TF.adjust_saturation(
                    img_float,
                    1 + np.random.uniform(-self.saturation_aug, self.saturation_aug),
                )
            if self.hue_aug > 0:
                img_float = TF.adjust_hue(
                    img_float, np.random.uniform(-self.hue_aug, self.hue_aug)
                )

            img = (img_float * 255).byte()
            img = rearrange(img, "c h w -> h w c").numpy()
            images.append(img)

        if self.tile_images and len(images) > 1:
            # Tile images horizontally
            tiled = np.concatenate(images, axis=1)
            return [Image.fromarray(tiled)]

        return [Image.fromarray(img) for img in images]

    def _action_to_text(self, actions: np.ndarray) -> str:
        """Convert actions to discretized text."""
        stats = self.stats["out_ori_act"]
        min_act = np.array(stats["min"])
        max_act = np.array(stats["max"])

        normalized = (actions - min_act) / (max_act - min_act + 1e-8)
        discretized = np.round(normalized * self.num_bins).astype(int)
        discretized = np.clip(discretized, 0, self.num_bins)

        return " ".join(map(str, discretized.flatten().tolist()))

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]

        images = self._process_images(sample)
        instruction = sample["task"]
        # Actions include history, take only future actions (matches original)
        all_actions = sample[self.action_key].numpy()
        actions = all_actions[self.history :]  # Skip history, keep horizon
        action_text = self._action_to_text(actions)

        # Format for SFTTrainer VLM - matches original QwenActor.format_data()
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "image"}]
                + [{"type": "text", "text": instruction}],
            },
            {"role": "assistant", "content": [{"type": "text", "text": action_text}]},
        ]

        return {"messages": messages, "images": images}


class IndexSubsetDataset(Dataset):
    """Index-based subset wrapper with stable underlying indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]):
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.dataset[self.indices[i]]


class NumberSpaceOnlyProcessor(LogitsProcessor):
    """Constrains generation to numbers (0-9), spaces, and EOS."""

    def __init__(self, tokenizer):
        self.allowed_tokens = set()
        for i in range(10):
            tok = tokenizer.encode(str(i), add_special_tokens=False)
            if tok:
                self.allowed_tokens.add(tok[0])
        sp = tokenizer.encode(" ", add_special_tokens=False)
        if sp:
            self.allowed_tokens.add(sp[0])
        if tokenizer.eos_token_id is not None:
            self.allowed_tokens.add(tokenizer.eos_token_id)

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        for token_id in self.allowed_tokens:
            mask[:, token_id] = 0
        return scores + mask


class QwenVLActor:
    """Inference wrapper that supports LoRA adapters saved by PEFT."""

    def __init__(
        self,
        *,
        base_model_id: str,
        model_path: str,
        stats_path: str,
        horizon: int = 8,
        action_dim: int = 7,
        num_bins: int = 1000,
        device: str = "cuda",
        use_flash_attention: bool = False,
        torch_compile: bool = False,
        img_size: int = 224,
        num_cams: int = 2,
        tile_images: bool = True,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.device = device

        kwargs = {"torch_dtype": torch.bfloat16, "device_map": device}
        if use_flash_attention:
            kwargs["attn_implementation"] = "kernels-community/flash-attn3"

        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(base_model_id, **kwargs)

        # If this is a PEFT adapter dir, load it; otherwise assume full model.
        adapter_loaded = False
        try:
            from peft import PeftModel

            if (Path(model_path) / "adapter_config.json").exists() or (
                Path(model_path) / "adapter_model.safetensors"
            ).exists():
                base = PeftModel.from_pretrained(base, model_path)
                adapter_loaded = True
        except Exception:
            adapter_loaded = False

        self.model = base
        self.model.eval()
        if torch_compile:
            self.model = torch.compile(self.model)

        pixel_count = img_size * img_size
        if tile_images:
            pixel_count *= num_cams
        self.processor = Qwen2_5_VLProcessor.from_pretrained(
            base_model_id, min_pixels=pixel_count, max_pixels=pixel_count
        )
        self.logits_processor = NumberSpaceOnlyProcessor(self.processor.tokenizer)

        with open(stats_path, "r") as f:
            self.stats = json.load(f)

        self.system_prompt = (
            f"Analyze the input image and predict robot actions for the next "
            f"{horizon} timesteps. Each action has {action_dim} dimensions. "
            f"Output a single sequence of {horizon * action_dim} integers "
            f"(0-{num_bins} each), representing the {horizon} timesteps "
            f"sequentially. Provide only space separated numbers. Nothing else."
        )

        if adapter_loaded:
            # Useful breadcrumb when debugging inference paths.
            print(f"[eval] Loaded PEFT adapter from: {model_path}")
        else:
            print(f"[eval] Loaded full model from: {model_path}")

    @torch.no_grad()
    def predict(self, image: Image.Image, instruction: str, temperature: float = 0.0) -> torch.Tensor:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": instruction}]},
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
        )
        image_inputs = process_vision_info(messages)[0]
        inputs = self.processor(text=[text], images=[image_inputs], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        gen_kwargs = {"max_new_tokens": 256, "logits_processor": [self.logits_processor]}
        if temperature and temperature > 0:
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        output_ids = self.model.generate(**inputs, **gen_kwargs)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        action_text = self.processor.decode(generated, skip_special_tokens=True)
        return self._text_to_action(action_text)

    def _text_to_action(self, text: str) -> torch.Tensor:
        stats = self.stats["out_ori_act"]
        min_act = torch.tensor(stats["min"])
        max_act = torch.tensor(stats["max"])

        try:
            tokens = [int(x) for x in text.strip().split()]
            actions = torch.tensor(tokens, dtype=torch.float32)
            actions = actions.reshape(-1, self.action_dim)

            if actions.shape[0] < self.horizon:
                pad = actions[-1:].repeat(self.horizon - actions.shape[0], 1)
                actions = torch.cat([actions, pad], dim=0)
            actions = actions[: self.horizon]

            actions = (actions / self.num_bins) * (max_act - min_act) + min_act
        except Exception:
            actions = ((min_act + max_act) / 2).repeat(self.horizon, 1)

        return actions


def _hash_dict(d: Dict[str, Any]) -> str:
    blob = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _build_delta_timestamps(
    *, history: int, horizon: int, cam_list: Sequence[str], state_key: str, action_key: str
) -> Dict[str, List[float]]:
    fps = 10
    delta_timestamps = {
        state_key: [-x / fps for x in range(history - 1, -1, -1)],
        action_key: [x / fps for x in range(-history, horizon)],
    }
    for cam in cam_list:
        delta_timestamps[cam] = [-x / fps for x in range(history - 1, -1, -1)]
    return delta_timestamps


def get_libero_instruction_set(suite_name: str) -> set[str]:
    # Membership comes from simulator benchmark definitions.
    from libero.libero import benchmark

    bench = benchmark.get_benchmark_dict()[suite_name]()
    return {t.language for t in bench.tasks}


def build_or_load_subset_indices(
    *,
    repo_id: str,
    suite_name: str,
    history: int,
    horizon: int,
    cam_list: Sequence[str],
    state_key: str,
    action_key: str,
    cache_dir: str,
    rebuild_cache: bool,
) -> List[int]:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    instructions = sorted(get_libero_instruction_set(suite_name))

    key = {
        "repo_id": repo_id,
        "suite_name": suite_name,
        "history": history,
        "horizon": horizon,
        "cam_list": list(cam_list),
        "state_key": state_key,
        "action_key": action_key,
        "instructions_sha16": _hash_dict({"instructions": instructions}),
    }
    cache_path = Path(cache_dir) / f"{suite_name}_indices_{_hash_dict(key)}.json"

    if cache_path.exists() and not rebuild_cache:
        with open(cache_path, "r") as f:
            payload = json.load(f)
        return list(map(int, payload["indices"]))

    instruction_set = set(instructions)
    delta_timestamps = _build_delta_timestamps(
        history=history,
        horizon=horizon,
        cam_list=cam_list,
        state_key=state_key,
        action_key=action_key,
    )
    raw = LeRobotDataset(repo_id=repo_id, delta_timestamps=delta_timestamps)

    indices: List[int] = []
    for i in tqdm(range(len(raw)), desc=f"indexing {suite_name}", ncols=100):
        sample = raw[i]
        if sample.get("task") in instruction_set:
            indices.append(i)

    with open(cache_path, "w") as f:
        json.dump({"key": key, "num_total": len(raw), "num_subset": len(indices), "indices": indices}, f, indent=2)
    return indices


def load_processor(*, model_id: str, img_size: int, num_cams: int, tile_images: bool) -> Qwen2_5_VLProcessor:
    pixel_count = img_size * img_size
    if tile_images:
        pixel_count *= num_cams
    return Qwen2_5_VLProcessor.from_pretrained(model_id, min_pixels=pixel_count, max_pixels=pixel_count)


def load_model_for_training(*, model_id: str, use_flash_attention: bool) -> Qwen2_5_VLForConditionalGeneration:
    kwargs = {"torch_dtype": torch.bfloat16}
    if use_flash_attention:
        kwargs["attn_implementation"] = "kernels-community/flash-attn3"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    model.config.use_cache = False
    return model


def infer_lora_target_modules(model: torch.nn.Module) -> List[str]:
    # Common Qwen/LLaMA-like projection suffixes.
    candidates = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    found = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        for c in candidates:
            if name.endswith(c):
                found.add(c)
    if not found:
        return candidates
    # Stable ordering
    return [c for c in candidates if c in found]


def apply_lora(model: Qwen2_5_VLForConditionalGeneration, model_args: ModelArguments) -> torch.nn.Module:
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "LoRA requested but `peft` is not available. Install `peft` (and `accelerate`) in your env."
        ) from e

    if model_args.lora_target_modules:
        target_modules = [x.strip() for x in model_args.lora_target_modules.split(",") if x.strip()]
    else:
        target_modules = infer_lora_target_modules(model)

    lora_cfg = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Helpful printout for sanity.
    try:
        model.print_trainable_parameters()
    except Exception:
        pass
    return model


def _auto_find_stats_path(model_path: str) -> Optional[str]:
    p = Path(model_path)
    candidates = [
        p / "dataset_stats.json",
        p.parent / "dataset_stats.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def build_log_dir(model_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    p = Path(model_path)
    model_name = (Path(p.parent.name) / p.name) if "checkpoint-" in p.name else p.name
    return str(Path("eval_logs") / model_name / timestamp)


def flip_image(img: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(img[::-1, ::-1])


def preprocess_obs(
    obs: Dict,
    *,
    img_size: int = 224,
    crop_ratio: float = 0.875,
    tile_images: bool = True,
) -> Image.Image:
    cams = ["agentview_image", "robot0_eye_in_hand_image"]
    images = []
    for cam in cams:
        img = flip_image(obs[cam])
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        if crop_ratio < 1.0:
            h, w = img.shape[-2:]
            crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            img = TF.crop(img, top, left, crop_h, crop_w)
        if img_size > 0:
            img = TF.resize(img, [img_size, img_size])
        img = (img * 255).byte()
        img = img.permute(1, 2, 0).numpy()
        images.append(img)
    if tile_images:
        tiled = np.concatenate(images, axis=1)
        return Image.fromarray(tiled)
    return Image.fromarray(images[0])


def init_libero_env(task, *, seed: int = 7, env_resolution: int = 256):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=env_resolution,
        camera_widths=env_resolution,
        camera_depths=True,
    )
    env.seed(seed)
    return env


def get_libero_10_tasks():
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()["libero_10"]()
    return suite


def evaluate_libero_10(
    *,
    actor: QwenVLActor,
    eval_args: EvalArguments,
    data_args: DataArguments,
    budget_args: BudgetArguments,
):
    suite = get_libero_10_tasks()
    # Keep original suite indices so init-states line up even after filtering.
    indexed_tasks = list(enumerate(suite.tasks))
    tasks = indexed_tasks
    if eval_args.task_name is not None:
        tasks = [(i, t) for (i, t) in tasks if t.name == eval_args.task_name]
        if not tasks:  # type: ignore[truthy-bool]
            raise ValueError(f"Task not found in libero_10: {eval_args.task_name}")
    if eval_args.max_tasks is not None:
        tasks = tasks[: eval_args.max_tasks]

    # Suite-specific max steps (aligned to vla0-trl defaults)
    max_steps = 520
    dummy_action = [0.0] * 6 + [-1.0]

    log_dir = eval_args.log_dir or build_log_dir(eval_args.model_path or "model")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    suite_dir = Path(log_dir) / "libero_10"
    suite_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(budget_args.seed)
    results = {}
    total_succ, total_total = 0, 0

    for suite_task_i, task in tasks:
        env = init_libero_env(task, seed=budget_args.seed)
        init_states = suite.get_task_init_states(suite_task_i)
        instruction = task.language

        # Pick a small set of episodes per task for cheap local eval.
        n = len(init_states)
        k = min(eval_args.episodes_per_task, n)
        chosen = list(range(n)) if k == n else rng.sample(range(n), k=k)

        task_succ = 0
        task_frames_written = 0
        for run_idx in tqdm(chosen, desc=f"eval {task.name}", ncols=100):
            env.reset()
            obs = env.set_init_state(init_states[run_idx])

            frames: List[np.ndarray] = []
            action_chunk = None
            action_i = 0
            action_horizon = eval_args.action_horizon

            old_action_chunks = [] if eval_args.ensemble_prediction and eval_args.ensemble_prediction > 1 else None

            done = False
            for t in range(max_steps + eval_args.frame_skip):
                if t < eval_args.frame_skip:
                    obs, _, done, _ = env.step(dummy_action)
                    continue

                if action_chunk is None or action_i >= action_horizon:
                    image = preprocess_obs(
                        obs,
                        img_size=data_args.img_size,
                        crop_ratio=data_args.crop_ratio,
                        tile_images=data_args.tile_images,
                    )
                    action_chunk = actor.predict(image, instruction).cpu().numpy()

                    if old_action_chunks is not None:
                        old_action_chunks.append(action_chunk.copy())
                        if len(old_action_chunks) > eval_args.ensemble_prediction:
                            old_action_chunks.pop(0)

                        if len(old_action_chunks) > 1:
                            ensemble_chunk = np.zeros_like(action_chunk)
                            ensemble_count = np.zeros_like(action_chunk)
                            new_old_chunks = []
                            num_old = len(old_action_chunks)
                            for i, old_chunk in enumerate(old_action_chunks[:-1]):
                                if len(old_chunk) <= action_horizon:
                                    continue
                                old_chunk = old_chunk[action_horizon:]
                                new_old_chunks.append(old_chunk)
                                if eval_args.ensemble_version == 1:
                                    weight = eval_args.ensemble_weight
                                else:
                                    weight = eval_args.ensemble_weight ** (num_old - i - 1)
                                ensemble_chunk[: len(old_chunk)] += weight * old_chunk
                                ensemble_count[: len(old_chunk)] += weight

                            new_old_chunks.append(old_action_chunks[-1])
                            ensemble_chunk += old_action_chunks[-1]
                            ensemble_count += 1
                            old_action_chunks = new_old_chunks
                            action_chunk = ensemble_chunk / ensemble_count

                    action_i = 0
                    action_horizon = min(eval_args.action_horizon, len(action_chunk))

                act = action_chunk[action_i].tolist()
                act[-1] = 1.0 if act[-1] > 0 else -1.0
                obs, _, done, _ = env.step(act)
                if eval_args.save_video:
                    frames.append(flip_image(obs["agentview_image"]))
                action_i += 1
                if done:
                    break

            success = bool(done)
            task_succ += int(success)
            total_succ += int(success)
            total_total += 1

            if eval_args.save_video and frames:
                try:
                    import imageio

                    suffix = "success" if success else "failure"
                    out = suite_dir / f"run{run_idx}__{suffix}__{task.name}.mp4"
                    # Keep videos bounded if user evals many tasks.
                    if task_frames_written < 3:
                        imageio.mimwrite(str(out), frames, fps=20)
                        task_frames_written += 1
                except Exception:
                    pass

        env.close()
        results[task.name] = {"success": task_succ, "total": len(chosen)}
        print(f"[eval] {task.name}: {task_succ}/{len(chosen)} ({(task_succ/len(chosen)*100):.1f}%)")

    overall = (total_succ / total_total * 100) if total_total else 0.0
    print(f"[eval] libero_10 TOTAL: {total_succ}/{total_total} ({overall:.1f}%)")

    with open(Path(log_dir) / "summary.json", "w") as f:
        json.dump({"suite": "libero_10", "results": results, "total": [total_succ, total_total]}, f, indent=2)
    print(f"[eval] wrote: {Path(log_dir) / 'summary.json'}")


def main():
    parser = argparse.ArgumentParser(description="tinyvla: single-file train/eval for libero_10")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional TRL config file (YAML/JSON) for dataclass args + SFTConfig.",
    )

    # Parse just mode/config first, then delegate full parsing to TrlParser for consistency.
    orig_argv = sys.argv[1:].copy()
    prelim, remaining = parser.parse_known_args()

    trl_parser = TrlParser(dataclass_types=[ModelArguments, DataArguments, BudgetArguments, EvalArguments, SFTConfig])
    if prelim.config:
        sys.argv = [sys.argv[0]] + remaining + ["--config", prelim.config]
    else:
        sys.argv = [sys.argv[0]] + remaining

    model_args, data_args, budget_args, eval_args, training_args = trl_parser.parse_args_and_config()

    # Inject mode from prelim into run logic (TrlParser doesn't handle subcommands cleanly).
    mode = prelim.mode

    random.seed(budget_args.seed)
    np.random.seed(budget_args.seed)
    torch.manual_seed(budget_args.seed)

    if mode == "eval":
        if not eval_args.model_path:
            raise ValueError("--model_path is required for --mode eval")

        stats_path = eval_args.stats_path or _auto_find_stats_path(eval_args.model_path)
        if stats_path is None:
            raise ValueError("Could not find dataset_stats.json; pass --stats_path explicitly.")

        actor = QwenVLActor(
            base_model_id=model_args.model_id,
            model_path=eval_args.model_path,
            stats_path=stats_path,
            horizon=data_args.horizon,
            action_dim=data_args.action_dim,
            num_bins=data_args.num_bins,
            use_flash_attention=model_args.use_flash_attention,
            torch_compile=model_args.torch_compile,
            img_size=data_args.img_size,
            num_cams=len(data_args.cam_list),
            tile_images=data_args.tile_images,
        )

        evaluate_libero_10(actor=actor, eval_args=eval_args, data_args=data_args, budget_args=budget_args)
        return

    # ---- TRAIN ----
    if data_args.suite_name != "libero_10":
        raise ValueError("This script is intentionally restricted to suite_name=libero_10 for local-only training/eval.")

    indices = build_or_load_subset_indices(
        repo_id=data_args.repo_id,
        suite_name="libero_10",
        history=data_args.history,
        horizon=data_args.horizon,
        cam_list=data_args.cam_list,
        state_key=data_args.state_key,
        action_key=data_args.action_key,
        cache_dir=data_args.cache_dir,
        rebuild_cache=data_args.rebuild_cache,
    )
    if data_args.max_train_samples is not None:
        indices = indices[: data_args.max_train_samples]

    print(f"[data] libero_10 subset: {len(indices)} samples")

    model = load_model_for_training(model_id=model_args.model_id, use_flash_attention=model_args.use_flash_attention)
    if model_args.use_lora:
        model = apply_lora(model, model_args)

    # Enable checkpointing after LoRA wrap for best memory behavior.
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    except Exception:
        pass

    processor = load_processor(
        model_id=model_args.model_id,
        img_size=data_args.img_size,
        num_cams=len(data_args.cam_list),
        tile_images=data_args.tile_images,
    )

    print("[data] loading base dataset...")
    dataset = LiberoDataset(
        repo_id=data_args.repo_id,
        history=data_args.history,
        horizon=data_args.horizon,
        action_key=data_args.action_key,
        state_key=data_args.state_key,
        cam_list=data_args.cam_list,
        img_size=data_args.img_size,
        crop_ratio=data_args.crop_ratio,
        tile_images=data_args.tile_images,
        brightness_aug=data_args.brightness_aug,
        contrast_aug=data_args.contrast_aug,
        saturation_aug=data_args.saturation_aug,
        hue_aug=data_args.hue_aug,
        num_bins=data_args.num_bins,
        action_dim=data_args.action_dim,
    )
    train_dataset: Dataset = IndexSubsetDataset(dataset, indices)

    # Save stats for inference
    Path(training_args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{training_args.output_dir}/dataset_stats.json", "w") as f:
        json.dump(dataset.stats, f, indent=2)

    collator = VLACollator(processor=processor)

    # VLM-specific settings
    training_args.max_length = None  # Don't truncate images
    training_args.remove_unused_columns = False
    training_args.dataset_kwargs = {"skip_prepare_dataset": True}

    # Single-H100-friendly defaults (only if not explicitly provided on CLI).
    def _flag_present(prefix: str) -> bool:
        return any(a == prefix or a.startswith(prefix + "=") for a in orig_argv)

    if not _flag_present("--per_device_train_batch_size"):
        training_args.per_device_train_batch_size = 1
    if not _flag_present("--gradient_accumulation_steps"):
        training_args.gradient_accumulation_steps = 8
    if not _flag_present("--learning_rate"):
        training_args.learning_rate = 5e-5
    if not _flag_present("--logging_steps"):
        training_args.logging_steps = 10
    if not _flag_present("--save_steps"):
        training_args.save_steps = 500
    if not _flag_present("--bf16") and hasattr(training_args, "bf16"):
        training_args.bf16 = True
    if not _flag_present("--fp16") and hasattr(training_args, "fp16"):
        training_args.fp16 = False

    # W&B integration: enable by default if available and not overridden.
    if not _flag_present("--report_to"):
        try:
            import wandb  # noqa: F401

            training_args.report_to = ["wandb"]
            if not getattr(training_args, "run_name", None):
                training_args.run_name = Path(training_args.output_dir).name
        except Exception:
            # If wandb isn't installed (or import fails), fall back to no-op logging.
            training_args.report_to = []

    # Step-budgeting: avoid epochs by default.
    training_args.max_steps = budget_args.step_budget
    training_args.num_train_epochs = 1  # ignored when max_steps is set

    # Print rough projection (user can refine by observing early throughput).
    print(f"[budget] max_steps={training_args.max_steps} (default sized for ~24 single-H100 hours)")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        processing_class=processor,
    )

    t0 = time.time()
    trainer.train()
    t1 = time.time()
    if trainer.state.global_step:
        print(f"[train] wall={((t1 - t0)/3600):.2f}h, step={trainer.state.global_step}, sec/step={((t1-t0)/trainer.state.global_step):.3f}")

    print("[train] saving final...")
    trainer.save_model(f"{training_args.output_dir}/final")
    processor.save_pretrained(f"{training_args.output_dir}/final")


if __name__ == "__main__":
    main()
