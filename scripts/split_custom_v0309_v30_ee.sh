#!/usr/bin/env bash
# Split custom_v0309_v30_ee_repo (merged LeRobot dataset, 16D EE-space) into per-task directories
#
# Input:  $LAB_ROOT/.cache/huggingface/lerobot/custom_v0309_v30_ee_repo/
# Output: playground/Datasets/CustomEE/ (4 task variant directories)
#
# Episode order in the merged repo (alphabetical by HDF5 dir name):
#   ep   0- 99: place_container_plate_clean1_ee
#   ep 100-199: place_container_plate_wp4_ee
#   ep 200-299: place_object_stand_clean1_ee
#   ep 300-399: place_object_stand_wp4_ee
#
# Usage: bash split_custom_v0309_v30_ee.sh [workers]
#   workers: number of parallel workers (default 0 = one per task = 4)

set -euo pipefail

LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
REPO_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

SRC="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0309_v30_ee_repo"
DST="${REPO_ROOT}/playground/Datasets/CustomEE"
MODALITY_FILE="${REPO_ROOT}/examples/RobotwinEE/train_files/modality_ee_16d.json"

# 4 task variants in episode order (must match merged repo order exactly)
TASKS=(
    place_container_plate_clean1_ee
    place_container_plate_wp4_ee
    place_object_stand_clean1_ee
    place_object_stand_wp4_ee
)

WORKERS="${1:-0}"  # default 0 = one worker per task (4), override with: bash split_custom_v0309_v30_ee.sh 1

echo "============================================"
echo " Splitting: ${SRC}"
echo " Into:      ${DST}"
echo " Tasks:     ${#TASKS[@]}"
echo " Workers:   ${WORKERS}"
echo " Modality:  ${MODALITY_FILE}"
echo "============================================"

# Verify source exists
if [ ! -d "$SRC/meta" ]; then
    echo "ERROR: source repo not found at ${SRC}"
    echo "Run the ar-research-kempner pipeline first to generate the merged EE dataset."
    exit 1
fi

# Verify modality file exists
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
echo "Sample info.json:"
python3 -c "
import json
for name in ['place_container_plate_clean1_ee', 'place_object_stand_clean1_ee']:
    d = json.load(open('${DST}/' + name + '/meta/info.json'))
    print(f'  {name}: {d[\"total_episodes\"]} episodes, {d[\"total_frames\"]} frames')
"

echo ""
echo "Verifying modality is 16D EE-space:"
python3 -c "
import json
m = json.load(open('${DST}/place_container_plate_clean1_ee/meta/modality.json'))
action_keys = list(m['action'].keys())
print(f'  Action keys: {action_keys}')
total_dim = max(v['end'] for v in m['action'].values())
print(f'  Total action dim: {total_dim}')
assert total_dim == 16, f'Expected 16D, got {total_dim}D'
print('  OK: 16D confirmed')
"

echo ""
echo "Total task dirs in ${DST}: $(ls "${DST}" | wc -l)"
echo ""
echo "Done. Ready to train: sbatch scripts/slurm_v0309_v30_ee_requeue.sh"
