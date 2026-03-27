#!/usr/bin/env bash
# ============================================================
# Dynamic-324 training (Discrete Diffusion / MaskGIT-style)
# 4x H100 local training
# Uses: starVLA/config/training/starvla_train_discrete_diffusion_real.yaml
#
# 用法：
#   bash realworld/0324-dynamic-training.sh
# ============================================================
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "============================================"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:   $HF_HOME"
echo "============================================"

NUM_GPUS=${1:-4}

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.representation bin \
  --framework.action_model.num_bins 256 \
  --framework.action_model.action_low -1.0 \
  --framework.action_model.action_high 1.0 \
  --framework.action_model.num_inference_steps 8 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix dynamic-324 \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.is_resume false \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 25000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id dynamic_first_0324 \
  --wandb_project starVLA_FastUMI \
  --wandb_entity hca
