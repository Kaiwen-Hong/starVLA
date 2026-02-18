#!/usr/bin/env bash
# Split custom_all_repo (merged LeRobot dataset) into per-task directories for StarVLA
#
# Input:  $KEMPNER_BASE/.cache/huggingface/lerobot/custom_all_repo/
# Output: playground/Datasets/Custom/ (33 task variant directories)

set -euo pipefail

LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
REPO_ROOT="${LAB_ROOT}/starVLA"
STAR_PYTHON="${LAB_ROOT}/miniforge3/envs/starVLA/bin/python"

SRC="${LAB_ROOT}/.cache/huggingface/lerobot/custom_all_repo"
DST="${REPO_ROOT}/playground/Datasets/Custom"

# 33 task variants: 11 tasks × 3 variants (clean, wp1, wp2), alphabetically sorted
TASKS=(
    adjust_bottle adjust_bottle_wp1 adjust_bottle_wp2
    beat_block_hammer beat_block_hammer_wp1 beat_block_hammer_wp2
    blocks_ranking_rgb blocks_ranking_rgb_wp1 blocks_ranking_rgb_wp2
    blocks_ranking_size blocks_ranking_size_wp1 blocks_ranking_size_wp2
    click_alarmclock click_alarmclock_wp1 click_alarmclock_wp2
    handover_block handover_block_wp1 handover_block_wp2
    handover_mic handover_mic_wp1 handover_mic_wp2
    move_can_pot move_can_pot_wp1 move_can_pot_wp2
    move_pillbottle_pad move_pillbottle_pad_wp1 move_pillbottle_pad_wp2
    move_stapler_pad move_stapler_pad_wp1 move_stapler_pad_wp2
    place_empty_cup place_empty_cup_wp1 place_empty_cup_wp2
)

WORKERS="${1:-0}"  # default 0 = one worker per task (33), override with: bash split_custom_all.sh 16

echo "============================================"
echo " Splitting: ${SRC}"
echo " Into:      ${DST}"
echo " Tasks:     ${#TASKS[@]}"
echo " Workers:   ${WORKERS}"
echo "============================================"

$STAR_PYTHON "${REPO_ROOT}/scripts/split_custom_lerobot.py" \
    --src "$SRC" \
    --dst "$DST" \
    --tasks "${TASKS[@]}" \
    --episodes-per-task 100 \
    --workers "$WORKERS"

echo ""
echo "Verifying..."
echo "Task directories: $(ls "$DST" | wc -l) (expected 33)"
echo "Sample info.json:"
python3 -c "
import json
d = json.load(open('${DST}/adjust_bottle/meta/info.json'))
print(f'  adjust_bottle: {d[\"total_episodes\"]} episodes, {d[\"total_frames\"]} frames, fps={d[\"fps\"]}')
"
echo ""
echo "Done. Now register in mixtures.py and set data_root_dir to playground/Datasets/Custom"
