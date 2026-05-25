#!/bin/bash
# QwenPI training-time RTC finetune for the air hockey checkpoints.
# Two blocks:
#   1) fastumi_airhockey_bounce_qwenPI_0522_DiT-S   (2-D action, fwsize=31, bounce)
#   2) fastumi_airhockey_qwenPI_0520_DiT-S          (10-D action, fwsize=15, strike)
#
# Each block:
#   - loads weights from the existing QwenPI checkpoint via
#     --trainer.pretrained_checkpoint, NOT from is_resume's auto-pick.
#   - writes to a fresh run_id "<original>_rtc_ft" so the original
#     checkpoint directory is left untouched.
#   - sets --framework.action_model.simulated_delay 6 to activate the
#     training-time RTC loss (see chenserver-files/ref_codes/IMPLEMENTATION.md).
#   - keeps --trainer.is_resume true so an interrupted run picks up from
#     the new _rtc_ft dir on relaunch (the trainer prefers existing
#     checkpoints in output_dir over pretrained_checkpoint).
#
# Runs on the 7 healthy GPUs (excluding GPU 3 — uncorrectable ECC).

set -e

cd /scratch/wangpc/starVLA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# Use the last 4 GPUs only (skip GPU 3 — uncorrectable DRAM ECC; the first
# three are reserved for other work).
export CUDA_VISIBLE_DEVICES=4,5,6,7

# HF auto-upload target + token (same pattern as 0523.sh).
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

CKPT_ROOT=/scratch/wangpc/starVLA/results/Checkpoints

############# Block 1: airhockey bounce (DiT-S, 2-D action) #############
PRETRAINED_PT_BOUNCE=$CKPT_ROOT/fastumi_airhockey_bounce_qwenPI_0522_DiT-S/checkpoints/steps_20000_pytorch_model.pt

echo "[rtc-ft] === Block 1: airhockey bounce ==="
echo "[rtc-ft] pretrained_checkpoint=$PRETRAINED_PT_BOUNCE"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
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
  --framework.action_model.simulated_delay 6 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix 3airhockey_dynamic_bounce \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint $PRETRAINED_PT_BOUNCE \
  --trainer.is_resume true \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_airhockey_bounce_qwenPI_0522_DiT-S_rtc_ft \
  --wandb_project starVLA_FastUMI_airhockey_bounce_0522_DiTS \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"

echo
echo "[rtc-ft] === Block 1 returned ==="
echo


############# Block 2: airhockey strike (DiT-S, 10-D action) #############
PRETRAINED_PT_STRIKE=$CKPT_ROOT/fastumi_airhockey_qwenPI_0520_DiT-S/checkpoints/steps_14000_pytorch_model.pt

echo "[rtc-ft] === Block 2: airhockey strike ==="
echo "[rtc-ft] pretrained_checkpoint=$PRETRAINED_PT_STRIKE"

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
  --framework.action_model.action_model_type DiT-S \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --framework.action_model.simulated_delay 6 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix 1airhockey_strike_combined \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint $PRETRAINED_PT_STRIKE \
  --trainer.is_resume true \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_airhockey_qwenPI_0520_DiT-S_rtc_ft \
  --wandb_project starVLA_FastUMI_airhockey_0520_DiTS \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"

echo
echo "[rtc-ft] === Block 2 returned ==="
