#!/usr/bin/env bash
# ============================================================
# Data processing for v62 + v61 + v51 (joint-space, 14D)
#
# Processes unique variants ONCE, then assembles per-version
# training_data dirs via hardlinks to avoid duplicate work.
#
# Unique variants (5):
#   place_cup5_tray5:clean1   (shared by v62, v61)
#   place_cup5_tray5:wp5      (v62 only)
#   place_cup5_tray5:wp4      (v61 only)
#   place_cup5_tray1:clean1   (v51 only)
#   place_cup5_tray1:wp4      (v51 only)
#   place_stapler_stand:clean1 (shared by v62, v61, v51)
#
# v62: place_cup5_tray5 (clean1, wp5) + place_stapler_stand (clean1) = 750 ep
# v61: place_cup5_tray5 (clean1, wp4) + place_stapler_stand (clean1) = 750 ep
# v51: place_cup5_tray1 (clean1, wp4) + place_stapler_stand (clean1) = 750 ep
#
# After this script finishes, run the training scripts separately:
#   bash r-preference/0331-v62-training-finetune-version2.sh
#   bash r-preference/0331-v61-training-finetune-version2.sh
#   bash r-preference/0331-v51-training-finetune-version2.sh
#
# Prerequisites:
#   salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
#     -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-04:00:00
#
# Usage:
#   bash r-preference/0331-v626151-data-processing.sh 2>&1 | tee v626151_data.log
# ============================================================

set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01

# -- Paths --
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

# -- Helper: process a single variant (task:setting) --
# Outputs to SHARED_POOL/<task>-<setting>-<EPISODE_NUM>/
# Skips if already complete.
process_variant() {
  local POOL_DIR="$1"
  local ENTRY="$2"
  local task="${ENTRY%%:*}"
  local setting="${ENTRY##*:}"
  local output_name="${task}-${setting}-${EPISODE_NUM}"

  # Already done in pool?
  if [ -d "${POOL_DIR}/${output_name}" ]; then
    local count
    count=$(ls -d "${POOL_DIR}/${output_name}"/episode_* 2>/dev/null | wc -l)
    if [ "$count" -eq "$EPISODE_NUM" ]; then
      echo "  OK: ${output_name} (${count} episodes), skipping"
      return 0
    else
      echo "  Partial: ${output_name} (${count}/${EPISODE_NUM}), removing and reprocessing"
      rm -rf "${POOL_DIR}/${output_name}"
    fi
  fi

  # Check processed_data (default output of process_data.py)
  if [ -d "processed_data/${output_name}" ]; then
    local pc
    pc=$(ls -d "processed_data/${output_name}"/episode_* 2>/dev/null | wc -l)
    if [ "$pc" -eq "$EPISODE_NUM" ]; then
      echo "  Moving complete processed_data/${output_name} -> pool"
      mv "processed_data/${output_name}" "${POOL_DIR}/${output_name}"
      return 0
    else
      echo "  Partial processed_data (${pc}/${EPISODE_NUM}), removing"
      rm -rf "processed_data/${output_name}"
    fi
  fi

  # Create symlink for data compat
  local variant_dir="${task}_${setting}"
  if [ ! -d "${DATA_CUSTOM}/${variant_dir}" ]; then
    echo "ERROR: ${DATA_CUSTOM}/${variant_dir} not found"
    exit 1
  fi
  mkdir -p "${DATA_COMPAT}/${task}"
  if [ ! -e "${DATA_COMPAT}/${task}/${setting}" ]; then
    ln -s "${DATA_CUSTOM}/${variant_dir}" "${DATA_COMPAT}/${task}/${setting}"
    echo "  Linked: data/${task}/${setting} -> data_custom_0320/${variant_dir}"
  fi

  # Process
  echo "  Processing: ${task} ${setting} ${EPISODE_NUM}"
  uv run python scripts/process_data.py "$task" "$setting" "$EPISODE_NUM"
  mv "processed_data/${output_name}" "${POOL_DIR}/${output_name}"

  # Clean up symlink
  local link="${DATA_COMPAT}/${task}/${setting}"
  [ -L "$link" ] && rm "$link"
  rmdir "${DATA_COMPAT}/${task}" 2>/dev/null || true
}

# -- Helper: assemble a version's training_data dir from pool via hardlinks --
# Usage: assemble_version <name> <tdata_dir> <pool_dir> <entry1> <entry2> ...
assemble_version() {
  local NAME="$1"
  local TDATA_DIR="$2"
  local POOL_DIR="$3"
  shift 3
  local ENTRIES=("$@")

  echo "--- Assembling ${NAME}: ${#ENTRIES[@]} variants ---"
  mkdir -p "${TDATA_DIR}"

  for entry in "${ENTRIES[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local output_name="${task}-${setting}-${EPISODE_NUM}"

    if [ -d "${TDATA_DIR}/${output_name}" ]; then
      local count
      count=$(ls -d "${TDATA_DIR}/${output_name}"/episode_* 2>/dev/null | wc -l)
      if [ "$count" -eq "$EPISODE_NUM" ]; then
        echo "  OK: ${output_name} already in ${NAME}"
        continue
      fi
      rm -rf "${TDATA_DIR}/${output_name}"
    fi

    echo "  Hardlinking: ${output_name} -> ${NAME}"
    cp -al "${POOL_DIR}/${output_name}" "${TDATA_DIR}/${output_name}"
  done

  # Verify
  for entry in "${ENTRIES[@]}"; do
    local task="${entry%%:*}"
    local setting="${entry##*:}"
    local output_name="${task}-${setting}-${EPISODE_NUM}"
    local count
    count=$(ls -d "${TDATA_DIR}/${output_name}"/episode_* 2>/dev/null | wc -l)
    if [ "$count" -ne "$EPISODE_NUM" ]; then
      echo "ERROR: ${output_name} has ${count}/${EPISODE_NUM} episodes in ${NAME}"
      exit 1
    fi
  done
  echo "  ${NAME}: all ${#ENTRIES[@]} variants verified."
}

# -- Helper: generate LeRobot dataset --
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
echo "Pipeline:  v62 + v61 + v51 data processing (deduplicated)"
echo "============================================"

cd "${KEMPNER_REPO}/policy/pi05"

# ============================================================================
# Step 1/3: Process unique variants (5 total, each processed once)
# ============================================================================
STEP_LOG "Step 1/3: Processing 5 unique variants -> shared pool"

SHARED_POOL="training_data/_shared_pool"
mkdir -p processed_data "${SHARED_POOL}"

UNIQUE_VARIANTS=(
  "place_cup5_tray5:clean1"
  "place_cup5_tray5:wp5"
  "place_cup5_tray5:wp4"
  "place_cup5_tray1:clean1"
  "place_cup5_tray1:wp4"
  "place_stapler_stand:clean1"
)

for entry in "${UNIQUE_VARIANTS[@]}"; do
  process_variant "${SHARED_POOL}" "$entry"
done

echo ""
echo "All ${#UNIQUE_VARIANTS[@]} unique variants processed."

# ============================================================================
# Step 2/3: Assemble per-version training_data dirs + generate LeRobot
# ============================================================================

# -- v62 --
STEP_LOG "v62: Assembling training_data + generating LeRobot"

V62_VARIANTS=("place_cup5_tray5:clean1" "place_cup5_tray5:wp5" "place_stapler_stand:clean1")
V62_TDATA="training_data/custom_v0320_v62"
V62_REPO="custom_v0320_v62_repo"

assemble_version "v62" "$V62_TDATA" "$SHARED_POOL" "${V62_VARIANTS[@]}"
generate_lerobot "v62" "$V62_TDATA" "$V62_REPO" 750

# -- v61 --
STEP_LOG "v61: Assembling training_data + generating LeRobot"

V61_VARIANTS=("place_cup5_tray5:clean1" "place_cup5_tray5:wp4" "place_stapler_stand:clean1")
V61_TDATA="training_data/custom_v0320_v61"
V61_REPO="custom_v0320_v61_repo"

assemble_version "v61" "$V61_TDATA" "$SHARED_POOL" "${V61_VARIANTS[@]}"
generate_lerobot "v61" "$V61_TDATA" "$V61_REPO" 750

# -- v51 --
STEP_LOG "v51: Assembling training_data + generating LeRobot"

V51_VARIANTS=("place_cup5_tray1:clean1" "place_cup5_tray1:wp4" "place_stapler_stand:clean1")
V51_TDATA="training_data/custom_v0320_v51"
V51_REPO="custom_v0320_v51_repo"

assemble_version "v51" "$V51_TDATA" "$SHARED_POOL" "${V51_VARIANTS[@]}"
generate_lerobot "v51" "$V51_TDATA" "$V51_REPO" 750

# ============================================================================
# Step 3/3: Cleanup intermediate HDF5
# ============================================================================
STEP_LOG "Cleaning up intermediate HDF5"

rm -rf "$SHARED_POOL" "$V62_TDATA" "$V61_TDATA" "$V51_TDATA"
echo "Removed shared pool and training_data dirs."

# ============================================================================
# Final summary
# ============================================================================
STEP_LOG "ALL DONE -- Data processing complete"

V62_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${V62_REPO}"
V61_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${V61_REPO}"
V51_DEST="${KEMPNER_CACHE}/huggingface/lerobot/${V51_REPO}"

python3 -c "
import json, pathlib
for name, path in [('v62', '${V62_DEST}'), ('v61', '${V61_DEST}'), ('v51', '${V51_DEST}')]:
    info = json.loads(pathlib.Path(f'{path}/meta/info.json').read_text())
    print(f'  {name}: {info[\"total_episodes\"]} episodes, {info[\"total_frames\"]} frames')
"

echo ""
echo "Ready for training. Run on separate GPU nodes:"
echo "  bash r-preference/0331-v62-training-finetune-version2.sh"
echo "  bash r-preference/0331-v61-training-finetune-version2.sh"
echo "  bash r-preference/0331-v51-training-finetune-version2.sh"
echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
