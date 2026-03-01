#!/usr/bin/env python3
"""
Unified LIBERO evaluation entrypoint for SimVLA.

Subcommands:
  - serve  : start policy WebSocket server
  - client : run one LIBERO task-suite evaluation
  - all    : run 4-suite parallel evaluation via existing shell script
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EVAL_DIR = ROOT / "evaluation" / "libero"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    shown = " ".join(shlex.quote(x) for x in cmd)
    if cwd is not None:
        print(f"[run] (cd {cwd} && {shown})", flush=True)
    else:
        print(f"[run] {shown}", flush=True)
    subprocess.run(cmd, check=True, cwd=str(cwd) if cwd else None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("SimVLA E2E evaluation launcher")
    sub = parser.add_subparsers(dest="mode", required=True)

    # serve
    p_serve = sub.add_parser("serve", help="Start SimVLA LIBERO policy server")
    p_serve.add_argument("--python-bin", type=str, default="python")
    p_serve.add_argument("--checkpoint", type=str, required=True)
    p_serve.add_argument("--norm_stats", type=str, default="./norm_stats/libero_norm.json")
    p_serve.add_argument("--smolvlm_model", type=str, default="HuggingFaceTB/SmolVLM-500M-Instruct")
    p_serve.add_argument("--host", type=str, default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8102)

    # client
    p_client = sub.add_parser("client", help="Run LIBERO evaluation client for one suite")
    p_client.add_argument("--python-bin", type=str, default="python")
    p_client.add_argument("--host", type=str, default="127.0.0.1")
    p_client.add_argument("--port", type=int, default=8102)
    p_client.add_argument("--client_type", type=str, choices=["websocket", "http"], default="websocket")
    p_client.add_argument(
        "--task_suite",
        type=str,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        default="libero_spatial",
    )
    p_client.add_argument("--num_trials", type=int, default=10)
    p_client.add_argument("--seed", type=int, default=7)
    p_client.add_argument("--replan_steps", type=int, default=5)
    p_client.add_argument("--video_out", type=str, default="./eval_results")
    p_client.add_argument("--no_video", action="store_true")
    p_client.add_argument("--connection_info", type=str, default=None)

    # all
    p_all = sub.add_parser("all", help="Run 4-suite parallel evaluation")
    p_all.add_argument("--bash-bin", type=str, default="bash")
    p_all.add_argument("--port", type=int, default=8102)
    p_all.add_argument("--num_trials", type=int, default=10)
    p_all.add_argument("--output_prefix", type=str, default="eval_simvla")
    p_all.add_argument("--gpus", type=str, default="0 1 2 3")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.mode == "serve":
        cmd = [
            args.python_bin,
            str(EVAL_DIR / "serve_smolvlm_libero.py"),
            "--checkpoint",
            args.checkpoint,
            "--norm_stats",
            args.norm_stats,
            "--smolvlm_model",
            args.smolvlm_model,
            "--host",
            args.host,
            "--port",
            str(args.port),
        ]
        run(cmd)
        return

    if args.mode == "client":
        cmd = [
            args.python_bin,
            "libero_client.py",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--client_type",
            args.client_type,
            "--task_suite",
            args.task_suite,
            "--num_trials",
            str(args.num_trials),
            "--seed",
            str(args.seed),
            "--replan_steps",
            str(args.replan_steps),
            "--video_out",
            args.video_out,
        ]
        if args.no_video:
            cmd.append("--no_video")
        if args.connection_info:
            cmd.extend(["--connection_info", args.connection_info])
        run(cmd, cwd=EVAL_DIR)
        return

    if args.mode == "all":
        cmd = [
            args.bash_bin,
            "run_eval_all.sh",
            str(args.port),
            str(args.num_trials),
            args.output_prefix,
            args.gpus,
        ]
        run(cmd, cwd=EVAL_DIR)
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
