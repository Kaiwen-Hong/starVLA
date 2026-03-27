#!/usr/bin/env bash
# Split custom_v0320_v50_ee_repo (merged LeRobot dataset, 16D EE-space) into per-task directories
#
# Input:  $LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v50_ee_repo/
# Output: playground/Datasets/CustomEE/ (4 task variant directories)
#
# Episode order in the merged repo (alphabetical by HDF5 dir name):
#   ep    0-249: place_cup5_tray1_clean1_ee
#   ep  250-499: place_cup5_tray1_wp5_ee
#   ep  500-749: place_stapler_stand_clean1_ee
#   ep  750-999: place_stapler_stand_wp5_ee
#
# Usage: bash split_custom_v0320_v50_ee.sh [workers]

set -euo pipefail

LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
REPO_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

SRC="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0320_v50_ee_repo"
DST="${REPO_ROOT}/playground/Datasets/CustomEE"
MODALITY_FILE="${REPO_ROOT}/examples/RobotwinEE/train_files/modality_ee_16d.json"

TASKS=(
    place_cup5_tray1_clean1_ee
    place_cup5_tray1_wp5_ee
    place_stapler_stand_clean1_ee
    place_stapler_stand_wp5_ee
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
    echo "Run the ar-research-kempner pipeline first to generate the merged EE dataset."
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
    --episodes-per-task 250 \
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
echo "Verifying modality is 16D EE-space:"
python3 -c "
import json
m = json.load(open('${DST}/place_cup5_tray1_clean1_ee/meta/modality.json'))
action_keys = list(m['action'].keys())
print(f'  Action keys: {action_keys}')
total_dim = max(v['end'] for v in m['action'].values())
print(f'  Total action dim: {total_dim}')
assert total_dim == 16, f'Expected 16D, got {total_dim}D'
print('  OK: 16D confirmed')
"

echo ""
echo "Done. Ready to train with data_mix=custom_v0320_v50_ee"
