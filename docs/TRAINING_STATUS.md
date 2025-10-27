# VLA LIBERO Training Status

## ✅ Current Run
**Run Name**: `vla-libero-lora-r64-v3`  
**WandB Link**: https://wandb.ai/voidz7447-ksagar-site/vla0-libero/runs/xr6svwnc

**Started**: Oct 23, 2025 00:24 IST

## Dataset
- **Source**: LIBERO-Spatial (10 tasks)
- **Location**: `/home/sra/tinyvla/LIBERO/libero/datasets/libero_spatial/`
- **Train samples**: 51,525
- **Val samples**: 5,725

## Model Configuration
- **Base model**: Qwen2.5-VL-3B-Instruct
- **Total parameters**: 3.9B
- **Trainable parameters**: 148.6M (3.8%) via LoRA
  - LoRA rank: 64
  - LoRA alpha: 128
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Training Configuration
- **Batch size**: 2
- **Gradient accumulation**: 96
- **Effective batch size**: 192
- **Epochs**: 64
- **Learning rate**: 5e-6 (cosine schedule with 5% warmup)
- **Optimizer**: AdamW 8-bit (memory efficient)
- **Gradient clipping**: 1.0

## Validation & Metrics

### WandB Logging
- **Frequency**: Every weight update (every 96 batches ≈ 26 seconds)
- Metrics logged:
  - `train/loss`: Training loss
  - `train/lr`: Current learning rate
  - `train/epoch`: Current epoch

### Every Epoch (val-every=1):
- Train/Val loss on 50 batches each
- Logged as `train/loss` and `val/loss`

### Every 5 Epochs (gen-val-every=5):
- **Generation-based validation** on 30 samples from train & val
- Metrics logged:
  - `{split}/mse`: Mean squared error
  - `{split}/mae`: Mean absolute error  
  - `{split}/parse_rate`: Success rate of parsing model outputs
  - `{split}/dim{0-6}_mae`: Per-dimension action errors
- **Example generations table** with:
  - Task instructions
  - Generated action sequences
  - MSE and MAE per example

## Checkpointing
- **Best model**: Saved when val loss improves → `checkpoints/best.pt`
- **Periodic**: Every 5 epochs → `checkpoints/epoch_{N}.pt`
- **Final**: After 64 epochs → `checkpoints/final.pt`

## Monitoring

### Check WandB (recommended):
https://wandb.ai/voidz7447-ksagar-site/vla0-libero/runs/xr6svwnc

### Check training progress:
```bash
tail -f /home/sra/tinyvla/training.log
```

### Check GPU usage:
```bash
nvidia-smi
```

### Check process:
```bash
ps aux | grep train.py
```

### Kill training (if needed):
```bash
pkill -f "python3 train.py"
```

## Expected Timeline
- **Speed**: ~3.6 iterations/second
- **Batches per epoch**: 25,763
- **Time per epoch**: ~1.99 hours
- **Total time for 64 epochs**: ~127 hours (5.3 days)

## Metrics Now Visible in WandB
✅ Loss and learning rate should appear in WandB within ~1 minute of training start
✅ First validation metrics after epoch 1 completes (~2 hours)
✅ Generation examples every 5 epochs

## Safe to Close Computer
Training is running with `nohup` in the background. Logs being written to:
- `/home/sra/tinyvla/training.log`
- `/home/sra/tinyvla/wandb/run-20251023_002412-xr6svwnc/`
- WandB cloud: https://wandb.ai/voidz7447-ksagar-site/vla0-libero/runs/xr6svwnc

**You can safely close your computer.** The training will continue on the server and metrics will sync to WandB automatically.
