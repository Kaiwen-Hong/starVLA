#!/usr/bin/env bash
# ============================================================
# Data processing for v42 + v52 (joint-space, 14D)
#
# Processes both datasets SEQUENTIALLY to avoid symlink race
# conditions on the shared place_stapler_stand variant.
#
# v42: place_cup_tray (clean1, wp5) + place_stapler_stand (clean1) = 750 ep
# v52: place_cup5_tray1 (clean1, wp5) + place_stapler_stand (clean1) = 750 ep
#
# Steps per dataset:
#   1. Process raw → joint-space HDF5 (process_data.py, NOT _ee)
#   2. Generate LeRobot dataset (convert_..._robotwin.py, NOT _ee)
#   3. Verify episode count = 750
#
# After this script finishes, run the training scripts separately:
#   bash r-preference/0331-v42-training-finetune-version2.sh
#   bash r-preference/0331-v52-training-finetune-version1.sh
#
# Prerequisites:
#   salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
#     -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-04:00:00
#
# Usage:
#   bash r-preference/0331-v4252-data-processing.sh 2>&1 | tee v4252_data.log
# ============================================================

set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01

# ── Paths ─────────────────────────────────────────────────────────
LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
KEMPNER_REPO="${LAB_ROOT}/ar-research-kempner"
KEMPNER_CACHE="${LAB_ROOT}/.cache"

export PATH="${LAB_ROOT}/.local/bin:${PATH}"

DATA_CUSTOM="${KEMPNER_REPO}/data_custom_0320"
DATA_COMPAT="${KEMPNER_REPO}/data"
EPISODE_NUM=250

STEP_LOG() {
  echo ""
  echo "########################################################################"
  echo "# $1"
  echo "# $(date '+%Y-%m-%d %H:%M:%S')"
  echo "########################################################################"
  echo ""
}

# ── Helper: process one dataset ───────────────────────────────────
# Usage: process_dataset <name> <training_data_dir> <variant1> <variant2> ...
process_dataset() {
  local NAME="$1"
  local TDATA_DIR="$2"
  shift 2
  local VARIANTS=("$@")

  echo "--- Processing ${NAME}: ${#VARIANTS[@]} variants ---"
  mkdir -p processed_data "${TDATA_DIR}"

  # Create symlinks
  for entry in "${VARIANTS[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local variant_dir="${task}_${setting}"

    if [ ! -d "${DATA_CUSTOM}/${variant_dir}" ]; then
      echo "ERROR: ${DATA_CUSTOM}/${variant_dir} not found"
      exit 1
    fi

    mkdir -p "${DATA_COMPAT}/${task}"
    if [ ! -e "${DATA_COMPAT}/${task}/${setting}" ]; then
      ln -s "${DATA_CUSTOM}/${variant_dir}" "${DATA_COMPAT}/${task}/${setting}"
      echo "  Linked: data/${task}/${setting} → data_custom_0320/${variant_dir}"
    fi
  done

  # Process each variant
  for entry in "${VARIANTS[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local output_name="${task}-${setting}-${EPISODE_NUM}"

    # Check if already done (and complete)
    if [ -d "${TDATA_DIR}/${output_name}" ]; then
      local actual_count
      actual_count=$(ls -d "${TDATA_DIR}/${output_name}"/episode_* 2>/dev/null | wc -l)
      if [ "$actual_count" -eq "$EPISODE_NUM" ]; then
        echo "  OK: ${output_name} (${actual_count} episodes), skipping"
        continue
      else
        echo "  Partial: ${output_name} (${actual_count}/${EPISODE_NUM}), removing and reprocessing"
        rm -rf "${TDATA_DIR}/${output_name}"
      fi
    fi

    # Check processed_data
    if [ -d "processed_data/${output_name}" ]; then
      local pc
      pc=$(ls -d "processed_data/${output_name}"/episode_* 2>/dev/null | wc -l)
      if [ "$pc" -eq "$EPISODE_NUM" ]; then
        echo "  Moving complete processed_data/${output_name} → ${TDATA_DIR}/"
        mv "processed_data/${output_name}" "${TDATA_DIR}/${output_name}"
        continue
      else
        echo "  Partial processed_data (${pc}/${EPISODE_NUM}), removing"
        rm -rf "processed_data/${output_name}"
      fi
    fi

    echo "  Processing: ${task} ${setting} ${EPISODE_NUM}"
    uv run python scripts/process_data.py "$task" "$setting" "$EPISODE_NUM"
    mv "processed_data/${output_name}" "${TDATA_DIR}/${output_name}"
  done

  # Clean up symlinks
  for entry in "${VARIANTS[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local link="${DATA_COMPAT}/${task}/${setting}"
    [ -L "$link" ] && rm "$link"
    rmdir "${DATA_COMPAT}/${task}" 2>/dev/null || true
  done

  # Verify all variants complete
  for entry in "${VARIANTS[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local output_name="${task}-${setting}-${EPISODE_NUM}"
    local count
    count=$(ls -d "${TDATA_DIR}/${output_name}"/episode_* 2>/dev/null | wc -l)
    if [ "$count" -ne "$EPISODE_NUM" ]; then
      echo "ERROR: ${output_name} has ${count}/${EPISODE_NUM} episodes"
      exit 1
    fi
  done
  echo "  All ${#VARIANTS[@]} variants verified (${EPISODE_NUM} episodes each)."
}

# ── Helper: generate LeRobot dataset ─────────────────────────────
# Usage: generate_lerobot <name> <training_data_dir> <repo_id> <expected_episodes>
generate_lerobot() {
  local NAME="$1"
  local TDATA_DIR="$2"
  local REPO_ID="$3"
  local EXPECTED_EP="$4"
  local LEROBOT_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${REPO_ID}"

  # Check if already done and correct
  if [ -f "${LEROBOT_DEST}/meta/info.json" ]; then
    local ep
    ep=$(python3 -c "import json; print(json.load(open('${LEROBOT_DEST}/meta/info.json'))['total_episodes'])")
    if [ "$ep" -eq "$EXPECTED_EP" ]; then
      echo "  LeRobot ${REPO_ID} already has ${ep} episodes, skipping."
      return 0
    else
      echo "  LeRobot ${REPO_ID} has ${ep}/${EXPECTED_EP} episodes, removing and regenerating."
      rm -rf "${LEROBOT_DEST}"
    fi
  fi

  local LOCAL_CACHE="/scratch/lerobot_tmp_${NAME}"
  mkdir -p "$LOCAL_CACHE"
  export HF_LEROBOT_HOME="${LOCAL_CACHE}/huggingface/lerobot"

  local NUM_WORKERS="${NUM_WORKERS:-64}"
  echo "  Generating LeRobot (${NUM_WORKERS} workers)..."
  echo "  Local SSD:  ${HF_LEROBOT_HOME}/${REPO_ID}/"
  echo "  Final dest: ${LEROBOT_DEST}"

  uv run examples/aloha_real/convert_aloha_data_to_lerobot_robotwin.py \
      --raw_dir "./${TDATA_DIR}/" \
      --repo_id "${REPO_ID}" \
      --num-workers "$NUM_WORKERS"

  local LOCAL_OUTPUT="${HF_LEROBOT_HOME}/${REPO_ID}/"
  echo "  Local size: $(du -sh "$LOCAL_OUTPUT" | cut -f1)"

  mkdir -p "$LEROBOT_DEST"
  rsync -a --info=progress2 "${LOCAL_OUTPUT}" "${LEROBOT_DEST}/"
  echo "  Isilon size: $(du -sh "$LEROBOT_DEST" | cut -f1)"

  rm -rf "$LOCAL_CACHE"
  unset HF_LEROBOT_HOME

  # Verify
  local ep
  ep=$(python3 -c "import json; print(json.load(open('${LEROBOT_DEST}/meta/info.json'))['total_episodes'])")
  if [ "$ep" -ne "$EXPECTED_EP" ]; then
    echo "ERROR: Expected ${EXPECTED_EP} episodes, got ${ep}"
    exit 1
  fi
  echo "  Verified: ${ep} episodes."
}

echo "============================================"
echo "Node:      $(hostname)"
echo "Pipeline:  v42 + v52 data processing (sequential)"
echo "============================================"

cd "${KEMPNER_REPO}/policy/pi05"

# ============================================================================
# v42: 3 variants, 750 episodes
# ============================================================================
STEP_LOG "v42: Processing raw → HDF5 (joint-space)"

V42_VARIANTS=(
  "place_cup_tray:clean1"
  "place_cup_tray:wp5"
  "place_stapler_stand:clean1"
)
V42_TDATA="training_data/custom_v0320_v42"
V42_REPO="custom_v0320_v42_repo"

process_dataset "v42" "$V42_TDATA" "${V42_VARIANTS[@]}"

STEP_LOG "v42: Generating LeRobot dataset"

generate_lerobot "v42" "$V42_TDATA" "$V42_REPO" 750

# ============================================================================
# v52: 3 variants, 750 episodes
# ============================================================================
STEP_LOG "v52: Processing raw → HDF5 (joint-space)"

V52_VARIANTS=(
  "place_cup5_tray1:clean1"
  "place_cup5_tray1:wp5"
  "place_stapler_stand:clean1"
)
V52_TDATA="training_data/custom_v0320_v52"
V52_REPO="custom_v0320_v52_repo"

process_dataset "v52" "$V52_TDATA" "${V52_VARIANTS[@]}"

STEP_LOG "v52: Generating LeRobot dataset"

generate_lerobot "v52" "$V52_TDATA" "$V52_REPO" 750

# ============================================================================
# Cleanup intermediate HDF5
# ============================================================================
STEP_LOG "Cleaning up intermediate HDF5"

rm -rf "$V42_TDATA" "$V52_TDATA"
echo "Removed training_data dirs."

# ============================================================================
# Final summary
# ============================================================================
STEP_LOG "ALL DONE — Data processing complete"

V42_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${V42_REPO}"
V52_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${V52_REPO}"

python3 -c "
import json, pathlib
for name, path in [('v42', '${V42_DEST}'), ('v52', '${V52_DEST}')]:
    info = json.loads(pathlib.Path(f'{path}/meta/info.json').read_text())
    print(f'  {name}: {info[\"total_episodes\"]} episodes, {info[\"total_frames\"]} frames')
"

echo ""
echo "Ready for training. Run on separate GPU nodes:"
echo "  bash r-preference/0331-v42-training-finetune-version2.sh"
echo "  bash r-preference/0331-v52-training-finetune-version1.sh"
echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
