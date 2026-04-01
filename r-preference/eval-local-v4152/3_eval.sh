#!/usr/bin/env bash
# ============================================================
# Step 3: Run RoboTwin evaluation (joint-space, 14D)
#
# Run this in Terminal 2 with the RoboTwin conda env.
# Make sure the policy server is running first (step 2).
#
# Usage:
#   conda activate robotwin   (or your RoboTwin env name)
#   bash r-preference/eval-local-v4152/3_eval.sh <version> <task_name> <task_config> [seed] [gpu_id]
#
# Examples:
#   # v42: trained on place_cup_tray (clean1, wp5) + place_stapler_stand (clean1)
#   bash r-preference/eval-local-v4152/3_eval.sh v42 place_cup_tray demo_clean 0 0
#   bash r-preference/eval-local-v4152/3_eval.sh v42 place_stapler_stand demo_clean 0 0
#
#   # v41: trained on place_cup_tray (clean1, wp4) + place_stapler_stand (clean1)
#   bash r-preference/eval-local-v4152/3_eval.sh v41 place_cup_tray demo_clean 0 0
#
#   # v52: trained on place_cup5_tray1 (clean1, wp5) + place_stapler_stand (clean1)
#   bash r-preference/eval-local-v4152/3_eval.sh v52 place_cup5_tray1 demo_clean 0 0
#
# Environment variables:
#   STARVLA_ROOT   -- starVLA repo path (auto-detected)
#   AR_ROOT        -- ar-research-kempner repo path (has assets/, script/, etc.)
#   PORT           -- policy server port (default: 5694)
#   STEP           -- checkpoint step for deploy yml (default: 60000)
# ============================================================
set -euo pipefail

VERSION="${1:?Usage: 3_eval.sh <version> <task_name> <task_config> [seed] [gpu_id]}"
TASK_NAME="${2:?}"
TASK_CONFIG="${3:?}"
SEED="${4:-0}"
GPU_ID="${5:-0}"

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PORT="${PORT:-5694}"
STEP="${STEP:-60000}"

# Map version to run_id
declare -A RUN_IDS=(
    ["v41"]="v0320_v41_qwenOFT_finetune_v2"
    ["v42"]="v0320_v42_qwenOFT_finetune_v2"
    ["v52"]="v0320_v52_qwenOFT_finetune_v1"
)

RUN_ID="${RUN_IDS[$VERSION]:-}"
if [ -z "$RUN_ID" ]; then
    echo "ERROR: Unknown version '${VERSION}'. Use one of: v41, v42, v52"
    exit 1
fi

CKPT_PATH="${STARVLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_${STEP}_pytorch_model.pt"
DEPLOY_YML="${STARVLA_ROOT}/r-preference/eval-local-v4152/deploy_policy_${VERSION}.yml"

if [ ! -f "$DEPLOY_YML" ]; then
    echo "ERROR: Deploy config not found: $DEPLOY_YML"
    exit 1
fi

# Detect AR_ROOT (ar-research-kempner — contains assets/, script/, etc.)
if [ -z "${AR_ROOT:-}" ]; then
    for candidate in \
        "${HOME}/Desktop/research/ar-research-kempner" \
        "${HOME}/ar-research-kempner" \
        "${HOME}/code/ar-research-kempner"; do
        if [ -d "$candidate/assets" ] && [ -f "$candidate/script/eval_policy.py" ]; then
            AR_ROOT="$candidate"
            break
        fi
    done
fi

if [ -z "${AR_ROOT:-}" ]; then
    echo "ERROR: Cannot find ar-research-kempner (needs assets/ and script/eval_policy.py)."
    echo "Set AR_ROOT env var, e.g.:"
    echo "  AR_ROOT=~/Desktop/research/ar-research-kempner bash ..."
    exit 1
fi

CKPT_SETTING="${VERSION}_step${STEP}"
EVAL_FILES_DIR="${STARVLA_ROOT}/examples/Robotwin/eval_files"

echo "============================================"
echo "RoboTwin Evaluation (joint-space, 14D)"
echo "  Version:     ${VERSION}"
echo "  Task:        ${TASK_NAME}"
echo "  Config:      ${TASK_CONFIG}"
echo "  Seed:        ${SEED}"
echo "  GPU:         ${GPU_ID}"
echo "  Checkpoint:  ${CKPT_PATH}"
echo "  Deploy YML:  ${DEPLOY_YML}"
echo "  AR_ROOT:     ${AR_ROOT}"
echo "============================================"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${AR_ROOT}:${STARVLA_ROOT}:${EVAL_FILES_DIR}:${PYTHONPATH:-}"

cd "$AR_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "$DEPLOY_YML" \
    --overrides \
    --task_name "$TASK_NAME" \
    --task_config "$TASK_CONFIG" \
    --ckpt_setting "$CKPT_SETTING" \
    --seed "$SEED" \
    --policy_name model2robotwin_interface \
    --policy_ckpt_path "$CKPT_PATH" \
    --save_as_policy pi05_ee \
    --exp_idx "${VERSION}w-starvla" \
    --test_num 75
