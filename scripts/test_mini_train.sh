#!/usr/bin/env bash
# Mini training test: 4×H100, 10 steps, 1 task only (robotwin_task1)
# Purpose: verify all components (model, dataloader, video backend, deepspeed) work end-to-end

set -euo pipefail

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── Environment ──
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
export HF_HOME=$LAB_ROOT/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

# Load CUDA module (Kempner HPC)
module load cuda/12.2.0-fasrc01 2>/dev/null || module load cuda 2>/dev/null || true

# ── Run ──
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.per_device_batch_size 4 \
  --datasets.vla_data.data_mix robotwin_task1 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 999999 \
  --trainer.logging_frequency 1 \
  --trainer.eval_interval 999999 \
  --run_root_dir ./results/Checkpoints \
  --run_id mini_test_4gpu
