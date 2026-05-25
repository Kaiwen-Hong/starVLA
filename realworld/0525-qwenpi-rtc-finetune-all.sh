#!/bin/bash
# Combined training-time RTC finetune script — runs all RTC finetune blocks
# sequentially in one launch. Mirrors the per-block configs from:
#   - realworld/0524-qwenpi-rtc-finetune-airhockey.sh
#   - realworld/0524-qwenpi-rtc-finetune-both.sh
#
# Blocks (in run order):
#   1) airhockey bounce        (DiT-S, 2-D action,  fwsize=31, simulated_delay=6)
#   2) pickandplace pick_to    (DiT-B, 10-D action, fwsize=15, simulated_delay=4)
#   3) pickandplace pick_from  (DiT-B, 10-D action, fwsize=15, simulated_delay=4)
#
# Each block is independent. Failure of one block does NOT abort the others
# — all three runs attempt to complete. A pass/fail summary is printed at
# the end. (No `set -e` for this reason.)
#
# Excluded: realworld/0403-dynamic-0403-0-pick_to_moved-rtc-ft.sh — that's
# an early draft that lacks --trainer.pretrained_checkpoint and would train
# pick_to_moved from scratch. Block 3 below supersedes it.
#
# All blocks use 4 GPUs via accelerate --gpu_ids 0,5,6,7.

cd /scratch/wangpc/starVLA

source ~/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

# GPU subset: 0,5,6,7 (4 GPUs, includes GPU 0).
#
# The original "last 4 GPUs" choice (4,5,6,7) crashed at the first NCCL
# broadcast inside DeepSpeed._broadcast_model with "CUDA error: an
# illegal memory access was encountered", regardless of transport:
# bare / +NCCL_P2P_DISABLE / +NCCL_SHM_DISABLE / socket-only all crashed
# the same way. Isolation test confirmed the issue is specifically about
# excluding GPU 0 — any 4-GPU subset INCLUDING GPU 0 trains fine. This
# is a hardware/PCIe-topology limitation of this machine, not a config
# issue we can resolve in env vars. So we borrow GPU 0 for ~10h.

# HF auto-upload target + token (same pattern as 0523.sh).
HF_REPO_ID="outsider86/DiscreteRTC"
HF_TOKEN_VAL="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN_VAL" ]]; then
  echo "[rtc-ft] ERROR: HF token not found. Run 'huggingface-cli login' or export HF_TOKEN." >&2
  exit 1
fi
echo "[rtc-ft] HF auto-upload target: ${HF_REPO_ID}/<run_id>/"
echo "[rtc-ft] GPU selection: accelerate --gpu_ids 0,5,6,7 (set per-block)"
echo "[rtc-ft] python: $(which python)"
echo "[rtc-ft] torch.cuda.device_count(): $(python -c 'import torch; print(torch.cuda.device_count())')"
echo

CKPT_ROOT=/scratch/wangpc/starVLA/results/Checkpoints
SUMMARY=()

run_block() {
  local name=$1
  local rc=$2
  if [ "$rc" -eq 0 ]; then
    SUMMARY+=("OK    $name")
  else
    SUMMARY+=("FAIL  $name (exit $rc)")
  fi
}


############# Block 1: airhockey bounce (DiT-S, 2-D, simulated_delay=6) #############
NAME1="airhockey_bounce"
PRETRAINED_PT_BOUNCE=$CKPT_ROOT/fastumi_airhockey_bounce_qwenPI_0522_DiT-S/checkpoints/steps_20000_pytorch_model.pt
echo
echo "============================================================"
echo "[rtc-ft] === START: $NAME1 ==="
echo "[rtc-ft] pretrained_checkpoint=$PRETRAINED_PT_BOUNCE"
echo "============================================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --gpu_ids 0,5,6,7 \
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
run_block "$NAME1" $?


############# Block 2: pick_to_moved (DiT-B, 10-D, simulated_delay=4) #############
NAME2="pickandplace_pick_to_moved"
PRETRAINED_PT_P0=$CKPT_ROOT/fastumi_pickandplace_qwenPI_0403_0_pick_to_moved/checkpoints/steps_30000_pytorch_model.pt
echo
echo "============================================================"
echo "[rtc-ft] === START: $NAME2 ==="
echo "[rtc-ft] pretrained_checkpoint=$PRETRAINED_PT_P0"
echo "============================================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --gpu_ids 0,5,6,7 \
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
  --trainer.pretrained_checkpoint $PRETRAINED_PT_P0 \
  --trainer.is_resume true \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI_0403_0_pick_to_moved_rtc_ft \
  --wandb_project starVLA_FastUMI_dynamic_0403_0_pick_to_moved \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"
run_block "$NAME2" $?


############# Block 3: pick_from_moved (DiT-B, 10-D, simulated_delay=4) #############
NAME3="pickandplace_pick_from_moved"
PRETRAINED_PT_P1=$CKPT_ROOT/fastumi_pickandplace_qwenPI_0403_1_pick_from_moved/checkpoints/steps_30000_pytorch_model.pt
echo
echo "============================================================"
echo "[rtc-ft] === START: $NAME3 ==="
echo "[rtc-ft] pretrained_checkpoint=$PRETRAINED_PT_P1"
echo "============================================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  --gpu_ids 0,5,6,7 \
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
  --datasets.vla_data.data_mix dynamic-0403-1-pick_from_moved \
  --datasets.vla_data.include_state false \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.pretrained_checkpoint $PRETRAINED_PT_P1 \
  --trainer.is_resume true \
  --trainer.max_train_steps 5000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI_0403_1_pick_from_moved_rtc_ft \
  --wandb_project starVLA_FastUMI_dynamic_0403_1_pick_from_moved \
  --wandb_entity 2200011093-peking-university \
  --hf_repo_id "$HF_REPO_ID" \
  --hf_token "$HF_TOKEN_VAL"
run_block "$NAME3" $?


############# Summary #############
echo
echo "============================================================"
echo "[rtc-ft] === ALL BLOCKS COMPLETE ==="
echo "============================================================"
for line in "${SUMMARY[@]}"; do
  echo "[rtc-ft] $line"
done

# Exit non-zero if any block failed (useful for CI/cron).
for line in "${SUMMARY[@]}"; do
  case "$line" in FAIL*) exit 1;; esac
done
exit 0
