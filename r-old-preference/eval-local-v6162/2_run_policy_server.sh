#!/usr/bin/env bash
# ============================================================
# Step 2: Start the StarVLA policy server (joint-space, 14D)
#
# Run this in Terminal 1 with the starVLA conda env.
#
# Usage:
#   conda activate starVLA
#   bash r-preference/eval-local-v6162/2_run_policy_server.sh <version>
#
# Examples:
#   bash r-preference/eval-local-v6162/2_run_policy_server.sh v61
#   bash r-preference/eval-local-v6162/2_run_policy_server.sh v62
#
# Environment variables (override as needed):
#   STARVLA_ROOT   -- starVLA repo path (auto-detected)
#   STARVLA_PYTHON -- Python binary (default: conda env python)
#   PORT           -- WebSocket port (default: 5694)
#   GPU_ID         -- GPU device (default: 0)
#   STEP           -- checkpoint step (default: 60000)
# ============================================================
set -euo pipefail

VERSION="${1:?Usage: 2_run_policy_server.sh <version>  (v61|v62)}"

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
PORT="${PORT:-5694}"
GPU_ID="${GPU_ID:-0}"
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

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: Checkpoint not found: $CKPT_PATH"
    echo "Run step 1 first: bash r-preference/eval-local-v6162/1_download_checkpoints.sh"
    exit 1
fi

echo "============================================"
echo "StarVLA Policy Server (joint-space, 14D)"
echo "  Version:      ${VERSION}"
echo "  Run ID:       ${RUN_ID}"
echo "  Checkpoint:   ${CKPT_PATH}"
echo "  Port:         ${PORT}"
echo "  GPU:          ${GPU_ID}"
echo "  STARVLA_ROOT: ${STARVLA_ROOT}"
echo "============================================"

export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
cd "$STARVLA_ROOT"

CUDA_VISIBLE_DEVICES=$GPU_ID \
    "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT_PATH" \
    --port "$PORT" \
    --use_bf16
