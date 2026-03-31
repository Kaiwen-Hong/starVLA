#!/usr/bin/env bash
# ============================================================
# v52 OFT finetune v1 — All-in-one pipeline
#
# From Qwen3-VL-OFT-RoboTwin2-All checkpoint (steps_140000)
# Framework: QwenOFT (L1 MLP regression), VLM: Qwen3-VL-4B-Instruct
# Action space: 14D joint-space (not EE)
#
# Data: place_cup5_tray1 (clean1, wp5) + place_stapler_stand (clean1)
#       3 variants, 250 episodes each, 750 total
#
# Steps:
#   1. Verify raw data in ar-research-kempner
#   2. Process raw → joint-space HDF5 (process_data.py, NOT _ee)
#   3. Generate LeRobot dataset (convert_..._robotwin.py, NOT _ee)
#   4. Split into per-task directories + verify 14D
#   5. Sanity check (20 steps, no wandb)
#   6. Full training (50000 steps)
#
# Prerequisites:
#   salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
#     -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-12:00:00
#
# Usage:
#   bash r-preference/0320-v52-training-finetune-version1.sh 2>&1 | tee v52_finetune_v1.log
# ============================================================

set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01

# ── Paths ─────────────────────────────────────────────────────────
LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
KEMPNER_REPO="${LAB_ROOT}/ar-research-kempner"
KEMPNER_CACHE="${LAB_ROOT}/.cache"
STARVLA_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

# uv binary (for ar-research-kempner steps)
export PATH="${LAB_ROOT}/.local/bin:${PATH}"

# ── Data generation config ────────────────────────────────────────
DATA_CUSTOM="${KEMPNER_REPO}/data_custom_0320"
DATA_COMPAT="${KEMPNER_REPO}/data"
VARIANTS=(
  "place_cup5_tray1:clean1"
  "place_cup5_tray1:wp5"
  "place_stapler_stand:clean1"
)
EPISODE_NUM=250
TRAINING_DATA_DIR="training_data/custom_v0320_v52"
REPO_ID="custom_v0320_v52_repo"
LEROBOT_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${REPO_ID}"

# ── Split config ──────────────────────────────────────────────────
SPLIT_TASKS=(
  place_cup5_tray1_clean1
  place_cup5_tray1_wp5
  place_stapler_stand_clean1
)
SPLIT_DST="${STARVLA_ROOT}/playground/Datasets/Custom"

# ── Training config ───────────────────────────────────────────────
PRETRAINED_CKPT="./checkpoints/Qwen3-VL-OFT-RoboTwin2-All/checkpoints/steps_140000_pytorch_model.pt"
DATA_MIX="custom_v0320_v52"
RUN_ID="v0320_v52_qwenOFT_finetune_v1"
WANDB_PROJECT="starVLA_v52_finetune"

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
echo "Pipeline:  v52 OFT finetune v1 (joint-space, RoboTwin2-All)"
echo "============================================"

# ============================================================================
# Step 1/6: Verify raw data exists
# ============================================================================
STEP_LOG "Step 1/6: Verifying raw data in ar-research-kempner"

for entry in "${VARIANTS[@]}"; do
  task="${entry%%:*}"
  setting="${entry##*:}"
  variant_dir="${task}_${setting}"
  if [ ! -d "${DATA_CUSTOM}/${variant_dir}" ]; then
    echo "ERROR: ${DATA_CUSTOM}/${variant_dir} not found"
    echo "Make sure data_custom_0320/ has been extracted."
    exit 1
  fi
  echo "  OK: ${variant_dir}"
done
echo "All ${#VARIANTS[@]} raw data variants found."

# ============================================================================
# Step 2/6: Process raw → joint-space HDF5
# ============================================================================
STEP_LOG "Step 2/6: Processing raw data → HDF5 (joint-space)"

cd "${KEMPNER_REPO}/policy/pi05"
mkdir -p processed_data "${TRAINING_DATA_DIR}"

echo "--- Creating symlinks for process_data.py compatibility ---"
for entry in "${VARIANTS[@]}"; do
  task="${entry%%:*}"
  setting="${entry##*:}"
  variant_dir="${task}_${setting}"

  if [ ! -d "${DATA_CUSTOM}/${variant_dir}" ]; then
    echo "WARNING: ${DATA_CUSTOM}/${variant_dir} not found, skipping"
    continue
  fi

  mkdir -p "${DATA_COMPAT}/${task}"
  if [ ! -e "${DATA_COMPAT}/${task}/${setting}" ]; then
    ln -s "${DATA_CUSTOM}/${variant_dir}" "${DATA_COMPAT}/${task}/${setting}"
    echo "  Linked: data/${task}/${setting} → data_custom_0320/${variant_dir}"
  fi
done

echo ""
echo "--- Processing HDF5 (joint-space, using process_data.py) ---"
for entry in "${VARIANTS[@]}"; do
  task="${entry%%:*}"
  setting="${entry##*:}"
  output_name="${task}-${setting}-${EPISODE_NUM}"

  if [ -d "${TRAINING_DATA_DIR}/${output_name}" ]; then
    echo "  Already done: ${output_name}, skipping"
    continue
  fi

  if [ -d "processed_data/${output_name}" ]; then
    echo "  Already processed: ${output_name}, moving to training_data/"
    mv "processed_data/${output_name}" "${TRAINING_DATA_DIR}/${output_name}"
    continue
  fi

  echo "  Processing: ${task} ${setting} ${EPISODE_NUM}"
  uv run python scripts/process_data.py "$task" "$setting" "$EPISODE_NUM"
  mv "processed_data/${output_name}" "${TRAINING_DATA_DIR}/${output_name}"
done

echo ""
echo "--- Cleaning up symlinks ---"
for entry in "${VARIANTS[@]}"; do
  task="${entry%%:*}"
  setting="${entry##*:}"
  link="${DATA_COMPAT}/${task}/${setting}"
  [ -L "$link" ] && rm "$link"
  rmdir "${DATA_COMPAT}/${task}" 2>/dev/null || true
done

PROCESS_COUNT=$(ls "${TRAINING_DATA_DIR}" | wc -l)
echo "Processed variants: ${PROCESS_COUNT} (expected ${#VARIANTS[@]})"
if [ "$PROCESS_COUNT" -ne "${#VARIANTS[@]}" ]; then
  echo "ERROR: expected ${#VARIANTS[@]} variants, got ${PROCESS_COUNT}"
  exit 1
fi

# ============================================================================
# Step 3/6: Generate LeRobot dataset
# ============================================================================
STEP_LOG "Step 3/6: Generating LeRobot dataset (joint-space)"

if [ -f "${LEROBOT_DEST}/meta/info.json" ]; then
  echo "LeRobot dataset already exists at ${LEROBOT_DEST}, skipping generation."
else
  LOCAL_CACHE="/scratch/lerobot_tmp_v52"
  mkdir -p "$LOCAL_CACHE"
  export HF_LEROBOT_HOME="${LOCAL_CACHE}/huggingface/lerobot"

  NUM_WORKERS="${NUM_WORKERS:-64}"
  echo "Using ${NUM_WORKERS} workers"
  echo "Local SSD:  ${HF_LEROBOT_HOME}/${REPO_ID}/"
  echo "Final dest: ${LEROBOT_DEST}"

  uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
      --raw_dir "./${TRAINING_DATA_DIR}/" \
      --repo_id "${REPO_ID}" \
      --num-workers "$NUM_WORKERS"

  LOCAL_OUTPUT="${HF_LEROBOT_HOME}/${REPO_ID}/"
  echo "Local dataset size: $(du -sh "$LOCAL_OUTPUT" | cut -f1)"

  echo "Copying to isilon..."
  mkdir -p "$LEROBOT_DEST"
  rsync -a --info=progress2 "${LOCAL_OUTPUT}" "${LEROBOT_DEST}/"
  echo "Isilon dataset size: $(du -sh "$LEROBOT_DEST" | cut -f1)"

  rm -rf "$LOCAL_CACHE"
  unset HF_LEROBOT_HOME
fi

# Verify LeRobot dataset
if [ ! -f "${LEROBOT_DEST}/meta/info.json" ]; then
  echo "ERROR: LeRobot dataset info.json not found at ${LEROBOT_DEST}/meta/info.json"
  exit 1
fi
python3 -c "
import json, pathlib
info = json.loads(pathlib.Path('${LEROBOT_DEST}/meta/info.json').read_text())
ep = info['total_episodes']
fr = info['total_frames']
print(f'  Episodes: {ep}')
print(f'  Frames:   {fr}')
assert ep == 750, f'Expected 750 episodes, got {ep}'
print('  Episode count OK.')
"

# Cleanup intermediate HDF5
if [ -d "${TRAINING_DATA_DIR}" ]; then
  echo "Cleaning up intermediate HDF5: ${TRAINING_DATA_DIR}"
  rm -rf "${TRAINING_DATA_DIR}"
fi

# ============================================================================
# Step 4/6: Split into per-task directories + verify 14D
# ============================================================================
STEP_LOG "Step 4/6: Splitting LeRobot dataset into per-task directories"

cd "${STARVLA_ROOT}"

if [ ! -d "${LEROBOT_DEST}/meta" ]; then
  echo "ERROR: source repo not found at ${LEROBOT_DEST}"
  exit 1
fi

$STAR_PYTHON "${STARVLA_ROOT}/scripts/split_custom_lerobot.py" \
    --src "$LEROBOT_DEST" \
    --dst "$SPLIT_DST" \
    --tasks "${SPLIT_TASKS[@]}" \
    --episodes-per-task "$EPISODE_NUM" \
    --workers 0

echo ""
echo "--- Verifying split ---"
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
echo "All ${#SPLIT_TASKS[@]} task directories created."

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
# Step 5/6: Sanity check (20 steps, no wandb)
# ============================================================================
STEP_LOG "Step 5/6: Sanity check — 20 training steps"

conda activate starVLA

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"

WANDB_MODE=disabled accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
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

echo "Sanity check PASSED (20 steps completed without error)."

# Cleanup sanity check artifacts
rm -rf "./results/Checkpoints/${RUN_ID}_sanity"

# ============================================================================
# Step 6/6: Full training
# ============================================================================
STEP_LOG "Step 6/6: Full training — 50000 steps"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
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
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id "${RUN_ID}" \
  --wandb_project "${WANDB_PROJECT}" \
  --wandb_entity hca

# ============================================================================
# Done
# ============================================================================
STEP_LOG "ALL DONE"
echo "Checkpoints at: ./results/Checkpoints/${RUN_ID}/"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
