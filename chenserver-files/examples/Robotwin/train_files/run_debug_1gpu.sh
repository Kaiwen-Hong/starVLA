#!/usr/bin/env bash
# 1-GPU debug run: single process, debugpy on port 10092, minimal steps.
# Usage: bash examples/Robotwin/train_files/run_debug_1gpu.sh
#
# Attach debugger: In VS Code/Cursor, add "Python: Remote Attach" with host 127.0.0.1 port 10092

set -euo pipefail

export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"

# Base config
CONFIG_YAML="${CONFIG_YAML:-starVLA/config/training/starvla_train_discrete_diffusion.yaml}"
BASE_VLM="${BASE_VLM:-playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action}"
DATA_MIX="${DATA_MIX:-robotwin_task1}"

echo "Debug mode: 1 GPU, is_debug=true, max_train_steps=5"
echo "Attach debugger to port 10092 when prompted"

accelerate launch \
  --config_file starVLA/config/deepseeds/accelerate_debug.yaml \
  starVLA/training/train_starvla.py \
  --config_yaml "$CONFIG_YAML" \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm "$BASE_VLM" \
  --framework.qwenvl.vl_hidden_dim 4096 \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix "$DATA_MIX" \
  --datasets.vla_data.per_device_batch_size 1 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 5 \
  --trainer.save_interval 2 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 2 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id debug_1gpu \
  --is_debug true \
  --wandb_project starVLA_Robotwin
