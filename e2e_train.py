#!/usr/bin/env python3
"""
Single-entry end-to-end LIBERO training launcher for SimVLA.

This script collapses:
1) metadata creation
2) normalization stats computation
3) multi-GPU training launch

into one command.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

from compute_libero_norm_stats import compute_norm_stats
from create_libero_meta import create_libero_meta


DEFAULT_SUBSETS = ["libero_10", "libero_goal", "libero_object", "libero_spatial"]


def parse_gpu_ids(raw: str | None) -> list[str]:
    if not raw:
        return []
    parts = [p for p in re.split(r"[\s,]+", raw.strip()) if p]
    return parts


def profile_defaults(size: str) -> dict[str, int | float]:
    if size == "large":
        return {
            "batch_size": 64,
            "learning_rate": 2e-4,
            "hidden_size": 1024,
            "depth": 24,
            "num_heads": 16,
        }
    return {
        "batch_size": 64,
        "learning_rate": 1e-4,
        "hidden_size": 768,
        "depth": 12,
        "num_heads": 12,
    }


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    cmd_str = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n[run] {cmd_str}\n", flush=True)
    subprocess.run(cmd, check=True, env=env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SimVLA E2E LIBERO training")

    # End-to-end control
    parser.add_argument("--skip_prepare", action="store_true", help="Skip metadata/statistics generation")
    parser.add_argument("--prepare_only", action="store_true", help="Only generate metadata/statistics")

    # Data prep
    parser.add_argument("--data_dir", type=str, required=True, help="LIBERO root containing subset folders")
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--meta_path", type=str, default="./datasets/metas/libero_train.json")
    parser.add_argument("--norm_stats_path", type=str, default="./norm_stats/libero_norm.json")

    # Training profile
    parser.add_argument("--size", choices=["small", "large"], default="small")
    parser.add_argument("--output_dir", type=str, default="./runs/simvla_libero_e2e")
    parser.add_argument("--resume_ckpt", type=str, default=None)

    # Core hyperparameters
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--learning_coef", type=float, default=0.1)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--iters", type=int, default=200000)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--image_size", type=int, default=384)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Model / training options
    parser.add_argument("--smolvlm_model_path", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    parser.add_argument("--use_adaln", action="store_true")
    parser.add_argument("--use_cosine_decay", action="store_true")
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)

    # Launch config
    parser.add_argument("--gpus", type=str, default=None, help='Examples: "0,1,2,3" or "0 1 2 3"')
    parser.add_argument("--num_processes", type=int, default=None, help="Defaults to number of GPUs or 1")
    parser.add_argument("--main_process_port", type=int, default=29504)
    parser.add_argument("--mixed_precision", type=str, default="bf16")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Resolve defaults by size profile.
    profile = profile_defaults(args.size)
    batch_size = args.batch_size if args.batch_size is not None else int(profile["batch_size"])
    learning_rate = args.learning_rate if args.learning_rate is not None else float(profile["learning_rate"])
    hidden_size = int(profile["hidden_size"])
    depth = int(profile["depth"])
    num_heads = int(profile["num_heads"])

    # Step 1/2: prepare data artifacts.
    if not args.skip_prepare:
        meta_path = Path(args.meta_path)
        norm_path = Path(args.norm_stats_path)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        norm_path.parent.mkdir(parents=True, exist_ok=True)

        print("[prep] creating metadata...", flush=True)
        create_libero_meta(
            data_dir=args.data_dir,
            subsets=args.subsets,
            output_path=str(meta_path),
        )

        print("[prep] computing normalization stats...", flush=True)
        compute_norm_stats(
            data_dir=args.data_dir,
            subsets=args.subsets,
            output_path=str(norm_path),
        )
    else:
        print("[prep] skipped", flush=True)

    if args.prepare_only:
        print("[done] preparation complete (prepare_only=true)", flush=True)
        return

    # Step 3: launch training.
    gpu_ids = parse_gpu_ids(args.gpus)
    num_processes = args.num_processes if args.num_processes is not None else (len(gpu_ids) if gpu_ids else 1)

    env = os.environ.copy()
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if gpu_ids:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)

    train_cmd = [
        "accelerate",
        "launch",
        "--num_processes",
        str(num_processes),
        "--main_process_port",
        str(args.main_process_port),
        "--mixed_precision",
        args.mixed_precision,
        "train_smolvlm.py",
        "--output_dir",
        args.output_dir,
        "--train_metas_path",
        args.meta_path,
        "--smolvlm_model_path",
        args.smolvlm_model_path,
        "--action_mode",
        "libero_joint",
        "--batch_size",
        str(batch_size),
        "--learning_rate",
        str(learning_rate),
        "--learning_coef",
        str(args.learning_coef),
        "--num_actions",
        str(args.num_actions),
        "--iters",
        str(args.iters),
        "--warmup_steps",
        str(args.warmup_steps),
        "--freeze_steps",
        str(args.freeze_steps),
        "--hidden_size",
        str(hidden_size),
        "--depth",
        str(depth),
        "--num_heads",
        str(num_heads),
        "--num_workers",
        str(args.num_workers),
        "--save_interval",
        str(args.save_interval),
        "--log_interval",
        str(args.log_interval),
        "--image_size",
        str(args.image_size),
        "--norm_stats_path",
        args.norm_stats_path,
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--min_lr_ratio",
        str(args.min_lr_ratio),
    ]

    if args.use_adaln:
        train_cmd.append("--use_adaln")
    if args.use_cosine_decay:
        train_cmd.append("--use_cosine_decay")
    if args.wandb_project:
        train_cmd.extend(["--wandb_project", args.wandb_project])
    if args.wandb_api_key:
        train_cmd.extend(["--wandb_api_key", args.wandb_api_key])
    if args.resume_ckpt:
        train_cmd.extend(["--models", args.resume_ckpt, "--resume"])

    run(train_cmd, env=env)
    print("[done] training finished", flush=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"[error] command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
