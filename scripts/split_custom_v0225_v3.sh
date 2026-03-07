#!/usr/bin/env bash
# Split custom_v0225_v3_repo (merged LeRobot dataset) into per-task directories for StarVLA
#
# Input:  $KEMPNER_BASE/.cache/huggingface/lerobot/custom_v0225_v3_repo/
# Output: playground/Datasets/Custom/ (20 task variant directories)
#
# Episode order in the merged repo (alphabetical by HDF5 dir name):
#   ep   0- 99: adjust_bottle (clean)
#   ep 100-199: adjust_bottle_hv
#   ep 200-299: adjust_bottle_lv
#   ep 300-399: beat_block_hammer (clean)
#   ep 400-499: beat_block_hammer_hv
#   ep 500-599: beat_block_hammer_lv
#   ep 600-699: blocks_ranking_rgb (clean)
#   ep 700-799: blocks_ranking_rgb_hv
#   ep 800-899: blocks_ranking_rgb_lv
#   ep 900-999: blocks_ranking_size (clean)
#   ep 1000-1099: handover_block (clean)
#   ep 1100-1199: handover_block_hv
#   ep 1200-1299: handover_block_lv
#   ep 1300-1399: handover_mic (clean)
#   ep 1400-1499: move_can_pot (clean)
#   ep 1500-1599: move_can_pot_hv
#   ep 1600-1699: move_can_pot_lv
#   ep 1700-1799: move_pillbottle_pad (clean)
#   ep 1800-1899: move_stapler_pad (clean)
#   ep 1900-1999: place_empty_cup (clean)
#
# Note: Clean dirs already exist in playground/Datasets/Custom/ from the custom_all split.
#       This script will overwrite them with data from the same source HDF5 (harmless).
#
# Usage: bash split_custom_v0225_v3.sh [workers]
#   workers: number of parallel workers (default 0 = one per task = 20)

set -euo pipefail

LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
REPO_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

SRC="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0225_v3_repo"
DST="${REPO_ROOT}/playground/Datasets/Custom"

# 20 task variants in episode order (must match merged repo order exactly)
# Order: alphabetical by ar-research HDF5 dir name, with clean→hv→lv mapping
TASKS=(
    adjust_bottle adjust_bottle_hv adjust_bottle_lv
    beat_block_hammer beat_block_hammer_hv beat_block_hammer_lv
    blocks_ranking_rgb blocks_ranking_rgb_hv blocks_ranking_rgb_lv
    blocks_ranking_size
    handover_block handover_block_hv handover_block_lv
    handover_mic
    move_can_pot move_can_pot_hv move_can_pot_lv
    move_pillbottle_pad
    move_stapler_pad
    place_empty_cup
)

WORKERS="${1:-0}"  # default 0 = one worker per task (20), override with: bash split_custom_v0225_v3.sh 16

echo "============================================"
echo " Splitting: ${SRC}"
echo " Into:      ${DST}"
echo " Tasks:     ${#TASKS[@]}"
echo " Workers:   ${WORKERS}"
echo "============================================"

# Verify source exists
if [ ! -d "$SRC/meta" ]; then
    echo "ERROR: source repo not found at ${SRC}"
    echo "Run the ar-research-kempner pipeline first (Kempner_generate_custom_v0225_v3.sh)"
    exit 1
fi

$STAR_PYTHON "${REPO_ROOT}/scripts/split_custom_lerobot.py" \
    --src "$SRC" \
    --dst "$DST" \
    --tasks "${TASKS[@]}" \
    --episodes-per-task 100 \
    --workers "$WORKERS"

echo ""
echo "===== Verifying ====="

# Check hv/lv dirs (the new ones)
MISSING=0
for task in adjust_bottle beat_block_hammer blocks_ranking_rgb handover_block move_can_pot; do
    for variant in hv lv; do
        dir="${DST}/${task}_${variant}"
        if [ -d "$dir" ] && [ -f "$dir/meta/modality.json" ]; then
            echo "  OK: ${task}_${variant}"
        else
            echo "  MISSING: ${task}_${variant}"
            MISSING=$((MISSING + 1))
        fi
    done
done

if [ $MISSING -eq 0 ]; then
    echo ""
    echo "All 10 hv/lv directories created successfully."
fi

echo ""
echo "Sample info.json:"
python3 -c "
import json
for name in ['adjust_bottle', 'adjust_bottle_hv', 'adjust_bottle_lv']:
    d = json.load(open('${DST}/' + name + '/meta/info.json'))
    print(f'  {name}: {d[\"total_episodes\"]} episodes, {d[\"total_frames\"]} frames')
"

echo ""
echo "Total task dirs in ${DST}: $(ls "${DST}" | wc -l)"
echo ""
echo "Done. Ready to train: sbatch scripts/slurm_custom_v0225_v3_requeue.sh"
