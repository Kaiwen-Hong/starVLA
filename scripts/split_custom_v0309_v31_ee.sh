#!/usr/bin/env bash
# Split custom_v0309_v31_ee_repo (merged LeRobot dataset, 16D EE-space) into per-task directories
#
# Input:  $LAB_ROOT/.cache/huggingface/lerobot/custom_v0309_v31_ee_repo/
# Output: playground/Datasets/CustomEE_v31/ (3 task variant directories)
#
# Episode order in the merged repo (alphabetical by HDF5 dir name):
#   ep   0- 99: place_container_plate_clean1_ee
#   ep 100-199: place_container_plate_wp4_ee
#   ep 200-299: place_object_stand_clean1_ee
#
# Usage: bash split_custom_v0309_v31_ee.sh [workers]

set -euo pipefail

LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
REPO_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

SRC="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0309_v31_ee_repo"
DST="${REPO_ROOT}/playground/Datasets/CustomEE_v31"
MODALITY_FILE="${REPO_ROOT}/examples/RobotwinEE/train_files/modality_ee_16d.json"

# 3 task variants in episode order (must match merged repo order exactly)
TASKS=(
    place_container_plate_clean1_ee
    place_container_plate_wp4_ee
    place_object_stand_clean1_ee
)

WORKERS="${1:-0}"

echo "============================================"
echo " Splitting: ${SRC}"
echo " Into:      ${DST}"
echo " Tasks:     ${#TASKS[@]}"
echo " Workers:   ${WORKERS}"
echo " Modality:  ${MODALITY_FILE}"
echo "============================================"

if [ ! -d "$SRC/meta" ]; then
    echo "ERROR: source repo not found at ${SRC}"
    exit 1
fi

if [ ! -f "$MODALITY_FILE" ]; then
    echo "ERROR: modality file not found at ${MODALITY_FILE}"
    exit 1
fi

$STAR_PYTHON "${REPO_ROOT}/scripts/split_custom_lerobot.py" \
    --src "$SRC" \
    --dst "$DST" \
    --tasks "${TASKS[@]}" \
    --episodes-per-task 100 \
    --workers "$WORKERS" \
    --modality-file "$MODALITY_FILE"

echo ""
echo "===== Verifying ====="

MISSING=0
for task in "${TASKS[@]}"; do
    dir="${DST}/${task}"
    if [ -d "$dir" ] && [ -f "$dir/meta/modality.json" ]; then
        echo "  OK: ${task}"
    else
        echo "  MISSING: ${task}"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "All ${#TASKS[@]} task directories created successfully."
fi

echo ""
echo "Done. Ready to train with data_mix=custom_v0309_v31_ee"
