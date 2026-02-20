#!/bin/bash
#SBATCH --job-name=starVLA_custom_2n
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h200
#SBATCH --requeue
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=1440G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# StarVLA Custom Dataset Training on Kempner kempner_requeue
# 2 nodes × 4 H200 80GB = 8 GPUs, DeepSpeed ZeRO-2, QwenOFT
# (change --constraint to h100 and --cpus-per-task to 96 for H100)
# ============================================================

# ── Environment setup ──
# Source lab environment (conda, HF cache, wandb, etc.)
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

# Load CUDA (required for DeepSpeed nvcc detection)
module load cuda/12.2.0-fasrc01

# Activate starVLA conda env from lab miniforge
conda activate starVLA

# Move to repo root
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── Multi-node communication ──
# MASTER_ADDR: first node in the allocation
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

# ── NCCL (multi-node settings) ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Use IB if available on Kempner, fall back to TCP
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# ── Diagnostics ──
echo "============================================"
echo "Job:        $SLURM_JOB_ID"
echo "Nodes:      $SLURM_JOB_NODELIST"
echo "MASTER_ADDR:$MASTER_ADDR"
echo "MASTER_PORT:$MASTER_PORT"
echo "============================================"

# ── Training ──
# srun launches one task per node (--ntasks-per-node=1).
# Each task runs accelerate launch, which spawns 4 GPU processes locally.
# --machine_rank is derived from SLURM_PROCID (0 on first node, 1 on second).
srun bash -c '
echo "Node $(hostname): rank=$SLURM_PROCID, GPUs=$CUDA_VISIBLE_DEVICES"
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2_2node.yaml \
  --num_machines 2 \
  --num_processes 8 \
  --machine_rank $SLURM_PROCID \
  --main_process_ip '"$MASTER_ADDR"' \
  --main_process_port '"$MASTER_PORT"' \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules "" \
  --trainer.max_train_steps 250000 \
  --trainer.save_interval 25000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_qwenOFT_2node \
  --wandb_project starVLA_Custom \
  --wandb_entity kaiwenh-17-uiuc
'
