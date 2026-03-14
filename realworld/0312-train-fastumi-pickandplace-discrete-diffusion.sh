#!/bin/bash
# FastUMI real pick-and-place training (Discrete Diffusion / MaskGIT-style)
# Uses: starVLA/config/training/starvla_train_discrete_diffusion_real.yaml
#
# 0312.md reminder:
# - Retrain from scratch after normalization fixes (don't resume old checkpoints).

set -euo pipefail

REPO_DIR=${REPO_DIR:-/scratch/wangpc/starVLA}

# Dataset selection
DATASET_NAME=${DATASET_NAME:-pickandplace-real-0307}
DATA_MIX=${DATA_MIX:-fastumi_pickandplace_real_0307}

# GPUs
NUM_GPUS=${1:-8}

# Attention implementation
ATTN_IMPL=${ATTN_IMPL:-flash_attention_2}

cd "${REPO_DIR}"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation "${ATTN_IMPL}" \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_discrete_diffusion_real_0312 \
  --wandb_project starVLA_FastUMI \
  --wandb_entity 2200011093-peking-university

