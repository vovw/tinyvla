"""VLA-0: Minimal implementation for LIBERO robot control via text-based actions."""

import argparse
import glob
import random
from pathlib import Path
from typing import List, Tuple, Optional
import json

import h5py
import numpy as np
import torch
torch.backends.cuda.matmul.fp32_precision = 'high'
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
import wandb
import bitsandbytes as bnb
from peft import LoraConfig, get_peft_model
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=UserWarning, message='None of the inputs have requires_grad=True')


# ============================================================================
# Dataset
# ============================================================================

class LIBERODataset(Dataset):
    """LIBERO HDF5 dataset with lazy loading."""

    def __init__(
        self,
        data_dir: str = "~/libero_data",
        horizon: int = 10,
        split: str = "train",
        val_ratio: float = 0.1,
        seed: int = 42,
        mask_prob: float = 0.3
    ):
        self.data_dir = Path(data_dir).expanduser()
        self.horizon = horizon
        self.split = split
        self.mask_prob = mask_prob if split == "train" else 0.0  # Only mask during training
        
        # Build index: (file_path, demo_key, timestep)
        files = sorted(glob.glob(str(self.data_dir / "*.hdf5")))
        print(f"Found {len(files)} HDF5 files")
        
        all_indices = []
        for filepath in files:
            with h5py.File(filepath, 'r') as f:
                for demo_key in f['data'].keys():
                    demo = f['data'][demo_key]
                    n_steps = len(demo['actions'])
                    for t in range(n_steps - horizon):
                        all_indices.append((filepath, demo_key, t))
        
        # Train/val split
        random.seed(seed)
        random.shuffle(all_indices)
        
        n_val = int(len(all_indices) * val_ratio)
        if split == "train":
            self.indices = all_indices[n_val:]
        else:
            self.indices = all_indices[:n_val]
        
        print(f"{split.upper()} samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        filepath, demo_key, t = self.indices[idx]
        
        with h5py.File(filepath, 'r') as f:
            demo = f['data'][demo_key]
            
            # Get image key names
            if 'agentview_image' in demo['obs']:
                agentview = np.array(demo['obs']['agentview_image'][t])
                eyeinhand = np.array(demo['obs']['eye_in_hand_image'][t])
            else:
                agentview = np.array(demo['obs']['agentview_rgb'][t])
                eyeinhand = np.array(demo['obs']['eye_in_hand_rgb'][t])
            
            # Get instruction
            instruction = demo.attrs.get('task_description', demo.attrs.get('task', b'Complete the task'))
            if isinstance(instruction, bytes):
                instruction = instruction.decode('utf-8')
            
            # Get actions
            actions = np.array(demo['actions'][t:t + self.horizon], dtype=np.float32)
        
        # Keep as numpy arrays for faster processing, convert to PIL in collate_fn
        img1 = agentview
        img2 = eyeinhand
        
        # Normalize actions: [-1, 1] → [0, 1000]
        normalized = ((actions + 1) / 2 * 1000).astype(int).clip(0, 1000)
        action_text = ' '.join(map(str, normalized.flatten()))

        # Apply Masked Action Augmentation (critical component from VLA-0 paper)
        if self.mask_prob > 0:
            action_text = self._mask_action_text(action_text)

        return img1, img2, instruction, action_text, actions

    def _mask_action_text(self, action_text: str) -> str:
        """
        Masked Action Augmentation from VLA-0 paper (Section III.B).
        Randomly masks individual digits in the action string to force the model
        to reason about actions based on visual observations rather than just
        auto-completing numerical sequences.
        """
        chars = list(action_text)
        for i in range(len(chars)):
            # Only mask digits, not spaces
            if chars[i].isdigit() and random.random() < self.mask_prob:
                # Replace digit with '_' mask token
                chars[i] = '_'
        return ''.join(chars)


# ============================================================================
# Model
# ============================================================================

class VLA0:
    """Wrapper around Qwen2.5-VL-3B for robot action prediction."""

    SYSTEM_PROMPT = "Predict robot actions for the next 10 timesteps (7 dimensions each). Output 70 space-separated integers (0-1000)."

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct", device: str = "cuda", use_lora: bool = True):
        self.device = device

        # Load model with optimizations
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16,   # Changed from bfloat16 to float16 for 4090
            device_map="auto",
        )
        
        # Enable gradient checkpointing before PEFT
        self.model.gradient_checkpointing_enable()
        
        # Disable KV cache for training
        if hasattr(self.model.config, 'use_cache'):
            self.model.config.use_cache = False
        
        # Apply LoRA for parameter-efficient training
        if use_lora:
            lora_config = LoraConfig(
                r=32,          # LoRA rank - reduced for 4090
                lora_alpha=64, # Scaled proportionally
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            # Re-enable for PEFT model
            self.model.enable_input_require_grads()

        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"Loaded {model_name}")

    def prepare_inputs(
        self, 
        images1: List[Image.Image], 
        images2: List[Image.Image],
        instructions: List[str], 
        action_texts: Optional[List[str]] = None
    ):
        """Prepare batch inputs for training/inference."""
        messages_batch = []

        for i in range(len(images1)):
            messages = [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": images1[i]},
                        {"type": "image", "image": images2[i]},
                        {"type": "text", "text": f"Task: {instructions[i]}"}
                    ]
                }
            ]

            # Add assistant response for training
            if action_texts is not None:
                messages.append({"role": "assistant", "content": action_texts[i]})

            messages_batch.append(messages)

        # Process with Qwen processor
        texts = [
            self.processor.apply_chat_template(
                msg, 
                tokenize=False, 
                add_generation_prompt=(action_texts is None)
            )
            for msg in messages_batch
        ]

        image_inputs, video_inputs = process_vision_info(messages_batch)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            max_pixels=128*128,  # Reduced for faster processing on 4090
        )

        return inputs.to(self.device)

    def forward_train(self, images1, images2, instructions, action_texts):
        """Training forward pass."""
        inputs = self.prepare_inputs(images1, images2, instructions, action_texts)
        inputs["labels"] = inputs["input_ids"].clone()
        outputs = self.model(**inputs)
        return outputs.loss

    @torch.no_grad()
    def predict(self, images1, images2, instructions):
        """Inference: generate action sequences."""
        inputs = self.prepare_inputs(images1, images2, instructions)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

        # Trim input tokens
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode
        texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        # Parse to actions
        actions_batch = []
        for text in texts:
            actions = self._parse_action_text(text)
            actions_batch.append(actions)

        return actions_batch, texts

    def _parse_action_text(self, text: str) -> np.ndarray:
        """Parse generated text to action array."""
        try:
            # Extract integers
            tokens = text.strip().split()
            ints = []
            for t in tokens:
                try:
                    val = int(t)
                    if 0 <= val <= 1000:
                        ints.append(val)
                except ValueError:
                    continue
                if len(ints) >= 70:
                    break
            
            # Pad if needed
            while len(ints) < 70:
                ints.append(500)  # Neutral value

            # Denormalize: [0, 1000] → [-1, 1]
            actions = (np.array(ints[:70]) / 1000.0) * 2 - 1
            actions = actions.reshape(10, 7)

            return actions.astype(np.float32)
        except Exception as e:
            print(f"Parse error: {e}, text: {text[:100]}")
            return np.zeros((10, 7), dtype=np.float32)


# ============================================================================
# Training
# ============================================================================

def validate(vla, dataloader, max_batches=20, split_name="val"):
    """Validation using cross-entropy loss."""
    vla.model.eval()
    
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for batch_idx, (img1, img2, instructions, action_texts, _) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            loss = vla.forward_train(img1, img2, instructions, action_texts)
            total_loss += loss.item()
            n_batches += 1
    
    vla.model.train()
    
    avg_loss = total_loss / max(n_batches, 1)
    return {f'{split_name}/loss': avg_loss}


def validate_generation(vla, dataset, n_samples=20, split_name="val"):
    """Generation-based validation with detailed metrics."""
    vla.model.eval()
    
    parse_success = 0
    total_mse = 0.0
    total_mae = 0.0
    per_dim_errors = np.zeros(7)
    sample_generations = []
    
    indices = np.random.choice(len(dataset), min(n_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in tqdm(indices, desc=f"Eval {split_name}"):
            img1, img2, instruction, action_text_gt, actions_gt = dataset[idx]
            
            try:
                pred_actions, generated_texts = vla.predict([img1], [img2], [instruction])
                pred_actions = pred_actions[0]
                
                # Compute metrics
                mse = np.mean((pred_actions - actions_gt) ** 2)
                mae = np.mean(np.abs(pred_actions - actions_gt))
                total_mse += mse
                total_mae += mae
                
                # Per-dimension errors (average across timesteps)
                dim_errors = np.mean(np.abs(pred_actions - actions_gt), axis=0)
                per_dim_errors += dim_errors
                
                parse_success += 1
                
                # Save first 3 examples
                if len(sample_generations) < 3:
                    sample_generations.append({
                        'instruction': instruction,
                        'generated': generated_texts[0][:150],
                        'mse': float(mse),
                        'mae': float(mae)
                    })
                    
            except Exception as e:
                print(f"Error in {split_name} generation: {e}")
    
    vla.model.train()
    
    n_success = max(parse_success, 1)
    per_dim_errors = per_dim_errors / n_success
    
    metrics = {
        f'{split_name}/parse_rate': parse_success / n_samples,
        f'{split_name}/mse': total_mse / n_success,
        f'{split_name}/mae': total_mae / n_success,
    }
    
    # Add per-dimension errors
    for i in range(7):
        metrics[f'{split_name}/dim{i}_mae'] = float(per_dim_errors[i])
    
    # Print examples
    if sample_generations:
        print(f"\n{split_name.upper()} Examples:")
        for i, ex in enumerate(sample_generations, 1):
            print(f"  {i}. Task: {ex['instruction']}")
            print(f"     Gen: {ex['generated']}")
            print(f"     MSE: {ex['mse']:.4f}, MAE: {ex['mae']:.4f}")
    
    return metrics, sample_generations


def train(args):
    """Main training loop."""

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project="vla0-libero",
            name=args.run_name,
            config=vars(args),
        )

    # Setup
    torch.set_float32_matmul_precision('medium')

    # Dataset
    train_dataset = LIBERODataset(
        args.data_dir,
        horizon=args.horizon,
        split="train",
        val_ratio=args.val_ratio,
        mask_prob=args.mask_prob
    )
    val_dataset = LIBERODataset(
        args.data_dir,
        horizon=args.horizon,
        split="val",
        val_ratio=args.val_ratio,
        mask_prob=0.0  # No masking during validation
    )

    # Collate function
    def collate_fn(batch):
        images1, images2, instructions, action_texts, actions = zip(*batch)
        # Convert numpy arrays to PIL images as expected by the processor
        pil_images1 = [Image.fromarray(img) for img in images1]
        pil_images2 = [Image.fromarray(img) for img in images2]
        return pil_images1, pil_images2, list(instructions), list(action_texts), actions

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    print(f"Effective batch size: {args.batch_size * args.grad_accum}")

    # Model
    vla = VLA0()

    # Count parameters
    n_params = sum(p.numel() for p in vla.model.parameters())
    n_trainable = sum(p.numel() for p in vla.model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params/1e6:.1f}M total, {n_trainable/1e6:.1f}M trainable")

    # Optimizer & Scheduler (8-bit for memory efficiency)
    optimizer = bnb.optim.AdamW8bit(
        vla.model.parameters(), 
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            vla.model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt.get('epoch', 0) + 1
            global_step = ckpt.get('step', 0)
            print(f"Resumed from epoch {start_epoch}, step {global_step}")

    # Training loop
    best_val_loss = float('inf')
    vla.model.train()

    Path(args.checkpoint_dir).mkdir(exist_ok=True, parents=True)
    
    # Save config
    config_path = Path(args.checkpoint_dir) / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    for epoch in range(start_epoch, args.epochs):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (img1, img2, instructions, action_texts, _) in enumerate(pbar):
            # Forward
            loss = vla.forward_train(img1, img2, instructions, action_texts)
            loss = loss / args.grad_accum

            # Backward
            loss.backward()

            # Update weights
            if (batch_idx + 1) % args.grad_accum == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(vla.model.parameters(), args.grad_clip)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

                # Logging
                epoch_loss += loss.item() * args.grad_accum
                n_batches += 1

                if global_step % args.log_every == 0:
                    avg_loss = epoch_loss / n_batches
                    current_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': global_step
                    })

                    if not args.no_wandb:
                        wandb.log({
                            'train/loss': avg_loss,
                            'train/epoch': epoch,
                            'train/lr': current_lr,
                        }, step=global_step)

            # Quick test mode
            if args.test and global_step >= args.test_steps:
                print(f"Test mode: stopping after {args.test_steps} steps")
                return

        # Validation (loss-based on both train and val)
        if (epoch + 1) % args.val_every == 0 or epoch == 0:
            print(f"\n=== Epoch {epoch+1} Validation ===")
            
            # Val loss
            val_metrics = validate(vla, val_loader, max_batches=args.val_batches, split_name="val")
            print(f"Val loss: {val_metrics['val/loss']:.4f}")
            
            # Train loss (for comparison)
            train_metrics = validate(vla, train_loader, max_batches=args.val_batches, split_name="train")
            print(f"Train loss: {train_metrics['train/loss']:.4f}")
            
            all_metrics = {**val_metrics, **train_metrics}

            if not args.no_wandb:
                wandb.log(all_metrics, step=global_step)

            # Save best model
            if val_metrics['val/loss'] < best_val_loss:
                best_val_loss = val_metrics['val/loss']
                best_path = Path(args.checkpoint_dir) / "best.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': vla.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': global_step,
                    'val_loss': best_val_loss,
                }, best_path)
                print(f"✓ Saved best model: {best_path}")
        
        # Generation validation (expensive, do less often)
        if (epoch + 1) % args.gen_val_every == 0:
            print(f"\n=== Generation Validation ===")
            
            # Val set
            val_gen_metrics, val_examples = validate_generation(
                vla, val_dataset, n_samples=args.gen_val_samples, split_name="val"
            )
            
            # Train set
            train_gen_metrics, train_examples = validate_generation(
                vla, train_dataset, n_samples=args.gen_val_samples, split_name="train"
            )
            
            all_gen_metrics = {**val_gen_metrics, **train_gen_metrics}
            
            print(f"\nSummary:")
            print(f"  Val:   MSE={val_gen_metrics['val/mse']:.4f}, MAE={val_gen_metrics['val/mae']:.4f}, Parse={val_gen_metrics['val/parse_rate']:.2%}")
            print(f"  Train: MSE={train_gen_metrics['train/mse']:.4f}, MAE={train_gen_metrics['train/mae']:.4f}, Parse={train_gen_metrics['train/parse_rate']:.2%}")
            
            if not args.no_wandb:
                wandb.log(all_gen_metrics, step=global_step)
                
                # Log example generations as table
                if val_examples:
                    table = wandb.Table(columns=["split", "instruction", "generated", "mse", "mae"])
                    for ex in val_examples:
                        table.add_data("val", ex['instruction'], ex['generated'], ex['mse'], ex['mae'])
                    for ex in train_examples[:2]:  # Just 2 train examples
                        table.add_data("train", ex['instruction'], ex['generated'], ex['mse'], ex['mae'])
                    wandb.log({"examples": table}, step=global_step)

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = Path(args.checkpoint_dir) / f"epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': vla.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'step': global_step,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Save final
    final_path = Path(args.checkpoint_dir) / "final.pt"
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': vla.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'step': global_step,
    }, final_path)
    print(f"Training complete. Saved: {final_path}")

    if not args.no_wandb:
        wandb.finish()


def main():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--data-dir', type=str, default='LIBERO/libero/datasets/libero_spatial')
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--val-ratio', type=float, default=0.1)
    parser.add_argument('--mask-prob', type=float, default=0.3,
                        help='Probability of masking each digit in action text (VLA-0 paper: Masked Action Augmentation)')

    # Training
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)  # Increased for 4090
    parser.add_argument('--grad-accum', type=int, default=16) # Reduced from 96 for faster epochs
    parser.add_argument('--lr', type=float, default=5e-5)     # Increased for LoRA
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.05)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=8) # Increased for faster data loading

    # Validation
    parser.add_argument('--val-every', type=int, default=1, help='Loss validation every N epochs')
    parser.add_argument('--val-batches', type=int, default=50, help='Batches for loss validation')
    parser.add_argument('--gen-val-every', type=int, default=5, help='Generation validation every N epochs')
    parser.add_argument('--gen-val-samples', type=int, default=30, help='Samples for generation validation')

    # Checkpointing
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--save-every', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Logging
    parser.add_argument('--run-name', type=str, default=None)
    parser.add_argument('--log-every', type=int, default=1, help='Log every N weight updates')
    parser.add_argument('--no-wandb', action='store_true')

    # Testing
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test-steps', type=int, default=5)

    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()

