#!/usr/bin/env bash
# ============================================================
# Step 3: Run RoboTwin evaluation (joint-space, 14D)
#
# Run this with the RoboTwin conda env.
# Make sure the policy server is running first (step 2).
#
# Usage:
#   conda activate robotwin
#   bash r-preference/eval-local-v6162-vast/3_eval.sh <version> <task_name> <task_config> [seed] [gpu_id]
#
# Examples:
#   # v61: place_stapler_stand (clean + wp4)
#   bash r-preference/eval-local-v6162-vast/3_eval.sh v61 place_stapler_stand place_stapler_stand_clean1 0 0
#   bash r-preference/eval-local-v6162-vast/3_eval.sh v61 place_stapler_stand place_stapler_stand_wp4 0 1
#
#   # v62: place_stapler_stand (clean + wp5)
#   bash r-preference/eval-local-v6162-vast/3_eval.sh v62 place_stapler_stand place_stapler_stand_clean1 0 2
#   bash r-preference/eval-local-v6162-vast/3_eval.sh v62 place_stapler_stand place_stapler_stand_wp5 0 3
#
# Environment variables:
#   STARVLA_ROOT   -- starVLA repo path (default: /home/user/starVLA)
#   AR_ROOT        -- ar-research-kempner repo path (default: /home/user/ar-research-kempner)
#   PORT           -- policy server port (default: 5694)
#   STEP           -- checkpoint step (default: 60000)
# ============================================================
set -euo pipefail

VERSION="${1:?Usage: 3_eval.sh <version> <task_name> <task_config> [seed] [gpu_id]}"
TASK_NAME="${2:?}"
TASK_CONFIG="${3:?}"
SEED="${4:-0}"
GPU_ID="${5:-0}"

STARVLA_ROOT="${STARVLA_ROOT:-/home/user/starVLA}"
AR_ROOT="${AR_ROOT:-/home/user/ar-research-kempner}"
PORT="${PORT:-5694}"
STEP="${STEP:-60000}"

# Map version to run_id
declare -A RUN_IDS=(
    ["v61"]="v0320_v61_qwenOFT_finetune_v2"
    ["v62"]="v0320_v62_qwenOFT_finetune_v2"
)

RUN_ID="${RUN_IDS[$VERSION]:-}"
if [ -z "$RUN_ID" ]; then
    echo "ERROR: Unknown version '${VERSION}'. Use one of: v61, v62"
    exit 1
fi

CKPT_PATH="${STARVLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_${STEP}_pytorch_model.pt"
DEPLOY_YML="${STARVLA_ROOT}/r-preference/eval-local-v6162-vast/deploy_policy_${VERSION}.yml"

if [ ! -f "$DEPLOY_YML" ]; then
    echo "ERROR: Deploy config not found: $DEPLOY_YML"
    exit 1
fi

if [ ! -d "$AR_ROOT" ] || [ ! -f "$AR_ROOT/script/eval_policy.py" ]; then
    echo "ERROR: Cannot find ar-research-kempner at: $AR_ROOT"
    echo "Set AR_ROOT env var, e.g.:"
    echo "  AR_ROOT=/home/user/ar-research-kempner bash ..."
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
echo "  Port:        ${PORT}"
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
    --port "$PORT" \
    --policy_name model2robotwin_interface \
    --policy_ckpt_path "$CKPT_PATH" \
    --save_as_policy pi05_ee \
    --exp_idx "${VERSION}w-starvla" \
    --test_num 75
