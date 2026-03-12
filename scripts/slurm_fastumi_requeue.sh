#!/bin/bash
#SBATCH --job-name=starVLA_fastumi
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h200
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=1440G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# StarVLA FastUMI (pickandplace) on kempner_requeue
# 1 node x 4 H200 80GB, DeepSpeed ZeRO-2, QwenOFT
#
# Key differences from RoboTwin/Custom scripts:
#   - action_dim = 10 (not 14): 3 pos + 6 rot6d + 1 gripper
#   - state_dim = 10 (not 14): same layout
#   - data_root_dir = playground/Datasets/FastUMI
#   - data_mix = fastumi_pickandplace
#   - Single wrist camera (not 3 cameras)
#
# (change --constraint to h100 and --cpus-per-task to 96 for H100)
# ============================================================

# ── Environment setup ──
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── NCCL ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Diagnostics ──
echo "============================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:   $HF_HOME"
echo "============================================"

# ── Training ──
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix fastumi_pickandplace \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenOFT_requeue \
  --wandb_project starVLA_FastUMI \
  --wandb_entity kaiwenh-17-uiuc
