#!/usr/bin/env bash
# Local / non-Kempner training script — no InfiniBand NCCL vars, uses flash_attention_2
# Run from repo root: bash examples/Robotwin/train_files/run_robotwin_train_local.sh

set -euo pipefail

# Use first 4 GPUs (0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Optional NCCL for multi-GPU (skip Kempner-specific bond0, mlx5)
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

###########################################################################################
# === Modify paths / config for your machine ===
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"

Framework_name=QwenOFT
freeze_module_list=''
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin_task1
run_id=local_${data_mix}_qwenOFT

# GPU count (4 when using CUDA_VISIBLE_DEVICES=0,1,2,3)
NUM_GPUS=4

# Use flash_attention_2 on machines with GLIBC >= 2.32; use sdpa if flash-attn fails
ATTN_IMPL=flash_attention_2

# 1-GPU mode: set RUN_1GPU=1 to use batch=1, grad_accum=16 to fit in ~96GB
# Use GPU_ID=1 or GPU_ID=3 etc. to pick which physical GPU (default 0)
RUN_1GPU="${RUN_1GPU:-0}"
if [ "${RUN_1GPU}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
  NUM_GPUS=1
  BATCH_PER_GPU=1
  GRAD_ACC=16
  run_id=local_1gpu_${data_mix}_qwenOFT
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  # Gradient accumulation to keep effective batch ~64 (8 GPUs x 8 batch x 1 = 64)
  BATCH_PER_GPU=8
  TARGET_GLOBAL=64
  GRAD_ACC=$(( (TARGET_GLOBAL + NUM_GPUS * BATCH_PER_GPU - 1) / (NUM_GPUS * BATCH_PER_GPU) ))
  [ "${GRAD_ACC}" -lt 1 ] && GRAD_ACC=1
fi
# === End config ===
###########################################################################################

export WANDB_MODE=disabled

output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}
cp "$0" ${output_dir}/

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.attn_implementation ${ATTN_IMPL} \
  --datasets.vla_data.per_device_batch_size ${BATCH_PER_GPU} \
  --datasets.vla_data.data_mix ${data_mix} \
  --trainer.freeze_modules ${freeze_module_list} \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps ${GRAD_ACC} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Robotwin
