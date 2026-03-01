#!/bin/bash
# Thin wrapper around e2e_train.py (large profile).

set -euo pipefail

BATCH_SIZE=${1:-64}
LEARNING_COEF=${2:-0.1}
OUTPUT_DIR=${3:-./runs/simvla_libero_large}
RESUME_CKPT=${4:-""}

DATA_DIR=${LIBERO_DATA_DIR:-./datasets/metas}
META_PATH=${TRAIN_METAS_PATH:-./datasets/metas/libero_train.json}
NORM_PATH=${NORM_STATS_PATH:-./norm_stats/libero_norm.json}
GPU_IDS=${CUDA_VISIBLE_DEVICES:-4,5,6,7}
SUBSETS_STR=${LIBERO_SUBSETS:-"libero_10 libero_goal libero_object libero_spatial libero_90"}
SMOLVLM_MODEL=${SMOLVLM_MODEL:-HuggingFaceTB/SmolVLM-500M-Instruct}

read -r -a SUBSETS <<< "$SUBSETS_STR"

CMD=(
  python e2e_train.py
  --size large
  --data_dir "$DATA_DIR"
  --subsets "${SUBSETS[@]}"
  --meta_path "$META_PATH"
  --norm_stats_path "$NORM_PATH"
  --output_dir "$OUTPUT_DIR"
  --batch_size "$BATCH_SIZE"
  --learning_coef "$LEARNING_COEF"
  --smolvlm_model_path "$SMOLVLM_MODEL"
  --gpus "$GPU_IDS"
)

if [ -n "$RESUME_CKPT" ]; then
  CMD+=(--resume_ckpt "$RESUME_CKPT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
