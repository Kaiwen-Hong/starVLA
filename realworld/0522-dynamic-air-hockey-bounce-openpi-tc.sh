#!/bin/bash
# 7-GPU training: QwenPI + DiT-S on 3airhockey_dynamic_bounce.
# State/action: 2-D left-arm TCP world XY (not 10-D EEF).
# HORIZON=32, N_OBS_STEPS=1, fps=50.
# Runs on the 7 healthy GPUs (excluding GPU 3 — uncorrectable ECC).

set -e

cd /scratch/wangpc/starVLA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# Skip GPU 3 (uncorrectable DRAM ECC).
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7

echo "[tc] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[tc] python: $(which python)"
echo "[tc] torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 7 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 2 \
  --framework.action_model.state_dim 2 \
  --framework.action_model.future_action_window_size 31 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-S \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix 3airhockey_dynamic_bounce \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.is_resume false \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 2000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_airhockey_bounce_qwenPI_0522_DiT-S \
  --wandb_project starVLA_FastUMI_airhockey_bounce_0522_DiTS \
  --wandb_entity 2200011093-peking-university

echo
echo "[tc] === Training launch returned ==="
