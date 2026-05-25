#!/bin/bash
############# QwenPI training-time RTC finetune #############
# Finetunes the QwenPI checkpoint with per-position prefix conditioning
# (training-time RTC). Mirrors the QwenPI block in
# 0403-dynamic-0403-0-pick_to_moved.sh but adds --framework.action_model.simulated_delay
# and uses a distinct run_id so the original checkpoint is not overwritten.
# Resumes from the existing QwenPI run by setting --trainer.is_resume true.
#
# NOTE: this is an early draft that does NOT pass --trainer.pretrained_checkpoint,
# so it starts from scratch in the new dir. For the corrected finetune that
# loads the existing 30k-step checkpoint, see 0524-qwenpi-rtc-finetune-both.sh.
#
# Runs on the 7 healthy GPUs (excluding GPU 3 — uncorrectable ECC).

set -e

cd /scratch/wangpc/starVLA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# Use the last 4 GPUs only (skip GPU 3 — uncorrectable DRAM ECC; the first
# three are reserved for other work).
export CUDA_VISIBLE_DEVICES=4,5,6,7

HF_REPO_ID="outsider86/DiscreteRTC"
HF_TOKEN_VAL="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN_VAL" ]]; then
  echo "[rtc-ft] ERROR: HF token not found. Run 'huggingface-cli login' or export HF_TOKEN." >&2
  exit 1
fi
echo "[rtc-ft] HF auto-upload target: ${HF_REPO_ID}/<run_id>/"
echo "[rtc-ft] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[rtc-ft] python: $(which python)"
echo "[rtc-ft] torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --framework.action_model.simulated_delay 4 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix dynamic-0403-0-pick_to_moved \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI_0403_0_pick_to_moved_rtc_ft \
  --wandb_project starVLA_FastUMI_dynamic_0403_0_pick_to_moved \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"
