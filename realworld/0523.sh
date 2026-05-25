#!/bin/bash
# 7-GPU training: QwenDiscreteDiffusion (MaskGIT-style, 256-bin) on 3airhockey_dynamic_bounce.
# Discrete-version counterpart of realworld/0522-dynamic-air-hockey-bounce-openpi-tc.sh.
# (Pool dataset is intentionally NOT included here — only dynamic airhockey bounce.)
# State/action: 2-D left-arm TCP world XY (not 10-D EEF).
# HORIZON=32 (future_action_window_size=31, past=0), fps=50.
# Runs on the 7 healthy GPUs (excluding GPU 3 — uncorrectable ECC).
#
# Notes:
#   - QwenDiscreteDiffusion auto-overrides vl_hidden_dim and cross_attention_dim
#     from the base VLM (see starVLA/model/framework/QwenDiscreteDiffusion.py:55-60),
#     so we do NOT pass --framework.qwenvl.vl_hidden_dim.
#   - action_horizon is auto-computed as future_action_window_size + 1
#     (see DiscreteDiffusion_ActionHeader.py:48).
#   - Auto-uploads to outsider86/DiscreteRTC/<run_id>/ on every save_interval and
#     at training end (see train_starvla.py:270-294, 459-472). The token is sourced
#     from $HF_TOKEN or ~/.cache/huggingface/token (huggingface-cli login).

set -e

cd /scratch/wangpc/starVLA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# Skip GPU 3 (uncorrectable DRAM ECC).
export CUDA_VISIBLE_DEVICES=0,1,2,4,5,6,7

# HF auto-upload destination + token. Token is read from cached login if HF_TOKEN
# is not already exported. The training code strips hf_token from the saved
# config.yaml (train_starvla.py:255-260), so it won't be persisted to disk.
HF_REPO_ID="outsider86/DiscreteRTC"
HF_TOKEN_VAL="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN_VAL" ]]; then
  echo "[discrete] ERROR: HF token not found. Run 'huggingface-cli login' or export HF_TOKEN." >&2
  exit 1
fi
echo "[discrete] HF auto-upload target: ${HF_REPO_ID}/<run_id>/"

echo "[discrete] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "[discrete] python: $(which python)"
echo "[discrete] torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 7 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion_real.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 2 \
  --framework.action_model.state_dim 2 \
  --framework.action_model.future_action_window_size 31 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.representation bin \
  --framework.action_model.num_bins 256 \
  --framework.action_model.action_low -1.0 \
  --framework.action_model.action_high 1.0 \
  --framework.action_model.num_inference_steps 8 \
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
  --run_id fastumi_airhockey_bounce_qwenDiscreteDiffusion_0523 \
  --wandb_project starVLA_FastUMI_airhockey_bounce_0523_Discrete \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"

echo
echo "[discrete] === Training launch returned ==="
