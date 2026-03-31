#!/usr/bin/env bash
# ============================================================
# v52 OFT finetune v1 — Training only (data must be ready)
#
# From Qwen3-VL-OFT-RoboTwin2-All checkpoint (steps_140000)
# Framework: QwenOFT (L1 MLP regression), VLM: Qwen3-VL-4B-Instruct
# Action space: 14D joint-space (not EE)
#
# Data: place_cup5_tray1 (clean1, wp5) + place_stapler_stand (clean1)
#       3 variants, 250 episodes each, 750 total
#
# Prerequisite: run 0331-v4252-data-processing.sh first!
#
# Prerequisites:
#   salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
#     -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-08:00:00
#
# Usage:
#   bash r-preference/0331-v52-training-finetune-version2.sh 2>&1 | tee v52_finetune_v2.log
# ============================================================

set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── Config ────────────────────────────────────────────────────────
LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

LEROBOT_REPO="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0320_v52_repo"

SPLIT_TASKS=(
  place_cup5_tray1_clean1
  place_cup5_tray1_wp5
  place_stapler_stand_clean1
)
SPLIT_DST="./playground/Datasets/Custom"
EPISODE_NUM=250

PRETRAINED_CKPT="./checkpoints/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
DATA_MIX="custom_v0320_v52"
RUN_ID="v0320_v52_qwenOFT_finetune_v1"
WANDB_PROJECT="starVLA_v52_finetune"

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

STEP_LOG() {
  echo ""
  echo "########################################################################"
  echo "# $1"
  echo "# $(date '+%Y-%m-%d %H:%M:%S')"
  echo "########################################################################"
  echo ""
}

echo "============================================"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-all}"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Pipeline:  v52 OFT finetune v1 (training only)"
echo "============================================"

# ============================================================================
# Step 1/4: Verify LeRobot dataset
# ============================================================================
STEP_LOG "Step 1/4: Verifying LeRobot dataset"

if [ ! -f "${LEROBOT_REPO}/meta/info.json" ]; then
  echo "ERROR: LeRobot repo not found at ${LEROBOT_REPO}"
  echo "Run 0331-v4252-data-processing.sh first!"
  exit 1
fi

python3 -c "
import json, pathlib
info = json.loads(pathlib.Path('${LEROBOT_REPO}/meta/info.json').read_text())
ep = info['total_episodes']
print(f'  Episodes: {ep}')
print(f'  Frames:   {info[\"total_frames\"]}')
assert ep == 750, f'Expected 750 episodes, got {ep}. Re-run data processing.'
print('  OK.')
"

# Verify pretrained checkpoint
if [ ! -f "${PRETRAINED_CKPT}" ]; then
  echo "ERROR: Pretrained checkpoint not found at ${PRETRAINED_CKPT}"
  exit 1
fi
echo "  Pretrained checkpoint: OK"

# ============================================================================
# Step 2/4: Split into per-task directories + verify 14D
# ============================================================================
STEP_LOG "Step 2/4: Splitting LeRobot dataset into per-task directories"

$STAR_PYTHON ./scripts/split_custom_lerobot.py \
    --src "$LEROBOT_REPO" \
    --dst "$SPLIT_DST" \
    --tasks "${SPLIT_TASKS[@]}" \
    --episodes-per-task "$EPISODE_NUM" \
    --workers 0

MISSING=0
for task in "${SPLIT_TASKS[@]}"; do
  dir="${SPLIT_DST}/${task}"
  if [ -d "$dir" ] && [ -f "$dir/meta/modality.json" ]; then
    echo "  OK: ${task}"
  else
    echo "  MISSING: ${task}"
    MISSING=$((MISSING + 1))
  fi
done
if [ $MISSING -ne 0 ]; then
  echo "ERROR: ${MISSING} task directories missing"
  exit 1
fi

echo ""
echo "--- Verifying 14D joint-space ---"
$STAR_PYTHON -c "
import json
m = json.load(open('${SPLIT_DST}/${SPLIT_TASKS[0]}/meta/modality.json'))
action_keys = list(m['action'].keys())
print(f'  Action keys: {action_keys}')
total_dim = max(v['end'] for v in m['action'].values())
print(f'  Total action dim: {total_dim}')
assert total_dim == 14, f'Expected 14D, got {total_dim}D'
print('  OK: 14D confirmed')
"

# ============================================================================
# Step 3/4: Sanity check (20 steps, no wandb)
# ============================================================================
STEP_LOG "Step 3/4: Sanity check — 20 training steps"

WANDB_MODE=disabled accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 10 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume false \
  --run_root_dir ./results/Checkpoints \
  --run_id "${RUN_ID}_sanity" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity hca

echo "Sanity check PASSED."
rm -rf "./results/Checkpoints/${RUN_ID}_sanity"

# ============================================================================
# Step 4/4: Full training
# ============================================================================
STEP_LOG "Step 4/4: Full training — 60000 steps"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix "${DATA_MIX}" \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.pretrained_checkpoint "${PRETRAINED_CKPT}" \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 60000 \
  --trainer.save_interval 15000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity hca

STEP_LOG "ALL DONE"
echo "Checkpoints at: ./results/Checkpoints/${RUN_ID}/"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
