#!/bin/bash
# Smoke test: 7-GPU version of 0403-dynamic-0403-0-pick_to_moved.sh
# Purpose: verify the QwenDiscreteDiffusion training pipeline can run on the
#          7 healthy GPUs (excluding GPU 3 with uncorrectable DRAM ECC).
#
# Safety: separate run_id + run_root_dir, is_resume=false, save_interval=999999,
#         eval_interval=999999, max_train_steps=5, WANDB_MODE=disabled, no hf push.
#         Will NOT touch the real checkpoint at
#         results/Checkpoints/fastumi_pickandplace_qwenDiscreteDiffusion_0403_0_pick_to_moved.

set -e

cd /scratch/wangpc/starVLA

# Activate the starVLA conda env (the script may be launched from base)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# Use the 7 healthy GPUs only (skip GPU 3 — uncorrectable ECC).
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7
# Disable wandb so the smoke test doesn't pollute the real wandb project.
export WANDB_MODE=disabled

echo "[smoke] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[smoke] nvidia-smi -L:"
nvidia-smi -L | sed 's/^/    /'
echo "[smoke] python: $(which python)"
echo "[smoke] torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

############# QwenDiscreteDiffusion smoke (5 steps) #############
accelerate launch      \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml      \
    --num_processes 7      \
    starVLA/training/train_starvla.py      \
    --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml      \
    --framework.name QwenDiscreteDiffusion      \
    --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action      \
    --framework.qwenvl.vl_hidden_dim 4096    \
    --framework.qwenvl.attn_implementation flash_attention_2       \
    --framework.action_model.representation bin        \
    --framework.action_model.num_bins 256       \
    --framework.action_model.action_low -1.0       \
    --framework.action_model.action_high 1.0      \
    --framework.action_model.num_inference_steps 8       \
    --datasets.vla_data.data_root_dir playground/Datasets/FastUMI      \
    --datasets.vla_data.data_mix dynamic-0403-0-pick_to_moved       \
    --datasets.vla_data.include_state false       \
    --datasets.vla_data.per_device_batch_size 8        \
    --datasets.vla_data.video_backend torchvision_av        \
    --trainer.freeze_modules ''        \
    --trainer.is_resume false \
    --trainer.max_train_steps 5 \
    --trainer.save_interval 999999 \
    --trainer.logging_frequency 1       \
    --trainer.eval_interval 999999       \
    --trainer.gradient_accumulation_steps 1      \
    --run_root_dir ./results/Checkpoints_smoke_temp        \
    --run_id smoke7gpu_0403_0_pick_to_moved_QDD \
    --wandb_project starVLA_smoke \
    --wandb_entity 2200011093-peking-university

echo
echo "[smoke] === Training launch returned ==="
