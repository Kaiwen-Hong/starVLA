#!/usr/bin/env bash
# ============================================================
# One-command attention comparison: v41 vs v42 vs v61 vs v62
#
# Sequentially evaluates 4 models on 1 GPU (4090), same seed,
# capturing attention every N steps. Then generates comparison
# visualizations.
#
# Usage:
#   bash r-preference/eval-local-v41426162-analysis/run_all.sh
#   (no need to activate any conda env — the script uses explicit Python paths)
#
# Environment variables (override as needed):
#   SEED              -- random seed for all 4 evals (default: 0)
#   GPU_ID            -- GPU device (default: 0)
#   PORT              -- WebSocket port (default: 5694)
#   STEP              -- checkpoint step (default: 60000)
#   TEST_NUM          -- episodes per model (default: 1)
#   ATTN_INTERVAL     -- save attention every N inference steps (default: 20)
#   STARVLA_ROOT      -- starVLA repo path (auto-detected)
#   AR_ROOT           -- ar-research-kempner path (default: sibling dir)
#   SERVER_PYTHON     -- Python for policy server (default: fastur5 env)
#   EVAL_PYTHON       -- Python for RoboTwin eval  (default: RoboTwin env)
#   VIZ_PYTHON        -- Python for visualization  (default: fastur5 env)
#   SKIP_EVAL         -- set to 1 to skip eval and only run visualization
# ============================================================
set -euo pipefail

# ── Paths ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
AR_ROOT="${AR_ROOT:-/home/kaiwen/Desktop/research/ar-research-kempner}"
EVAL_FILES_DIR="${STARVLA_ROOT}/examples/Robotwin/eval_files"

# ── Python interpreters (server needs fastur5/starVLA, eval needs RoboTwin) ─
CONDA_BASE="/home/kaiwen/miniforge3/envs"
SERVER_PYTHON="${SERVER_PYTHON:-${CONDA_BASE}/fastur5/bin/python}"
EVAL_PYTHON="${EVAL_PYTHON:-${CONDA_BASE}/RoboTwin/bin/python}"
VIZ_PYTHON="${VIZ_PYTHON:-${CONDA_BASE}/fastur5/bin/python}"

# ── Parameters ──────────────────────────────────────────────
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
STEP="${STEP:-60000}"
TEST_NUM="${TEST_NUM:-1}"
ATTN_INTERVAL="${ATTN_INTERVAL:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# ── Output directory ────────────────────────────────────────
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_BASE="${STARVLA_ROOT}/results/attention_analysis/${TIMESTAMP}_seed${SEED}"
mkdir -p "$OUTPUT_BASE"

# ── Model definitions ──────────────────────────────────────
# Format: VERSION  RUN_ID  TASK_NAME  TASK_CONFIG  LABEL
#   v41: avoid-obstacle (wp4), cup_tray
#   v42: non-avoid (wp5),      cup_tray
#   v61: avoid-obstacle (wp4), cup5_tray5
#   v62: non-avoid (wp5),      cup5_tray5
declare -a VERSIONS=(v41 v42 v61 v62)
declare -A RUN_IDS=(
    ["v41"]="v0320_v41_qwenOFT_finetune_v2"
    ["v42"]="v0320_v42_qwenOFT_finetune_v2"
    ["v61"]="v0320_v61_qwenOFT_finetune_v2"
    ["v62"]="v0320_v62_qwenOFT_finetune_v2"
)
declare -A TASK_NAMES=(
    ["v41"]="place_stapler_stand"
    ["v42"]="place_stapler_stand"
    ["v61"]="place_stapler_stand"
    ["v62"]="place_stapler_stand"
)
declare -A TASK_CONFIGS=(
    ["v41"]="place_stapler_stand_wp4"
    ["v42"]="place_stapler_stand_wp5"
    ["v61"]="place_stapler_stand_wp4"
    ["v62"]="place_stapler_stand_wp5"
)
declare -A LABELS=(
    ["v41"]="v41 (cup_tray, avoid-obstacle)"
    ["v42"]="v42 (cup_tray, standard)"
    ["v61"]="v61 (cup5_tray5, avoid-obstacle)"
    ["v62"]="v62 (cup5_tray5, standard)"
)

# ── Preflight checks ───────────────────────────────────────
echo "============================================"
echo "Attention Comparison: v41 vs v42 vs v61 vs v62"
echo "============================================"
echo "  Output:       ${OUTPUT_BASE}"
echo "  Seed:         ${SEED}"
echo "  GPU:          ${GPU_ID}"
echo "  Port:         ${PORT}"
echo "  Interval:     every ${ATTN_INTERVAL} steps"
echo "  Episodes:     ${TEST_NUM}"
echo "  Server Python: ${SERVER_PYTHON}"
echo "  Eval Python:   ${EVAL_PYTHON}"
echo "  Viz Python:    ${VIZ_PYTHON}"
echo "============================================"

ALL_OK=true
for V in "${VERSIONS[@]}"; do
    CKPT="${STARVLA_ROOT}/results/Checkpoints/${RUN_IDS[$V]}/checkpoints/steps_${STEP}_pytorch_model.pt"
    if [ -f "$CKPT" ]; then
        echo "  OK: $V  $(du -h "$CKPT" | cut -f1)"
    else
        echo "  MISSING: $V  $CKPT"
        ALL_OK=false
    fi
done

if [ ! -d "$AR_ROOT" ] || [ ! -f "$AR_ROOT/script/eval_policy.py" ]; then
    echo "  MISSING: ar-research-kempner at $AR_ROOT"
    ALL_OK=false
fi

if ! $ALL_OK; then
    echo "ABORT: Prerequisites missing."
    exit 1
fi
echo ""

# ── Helper: run one model ──────────────────────────────────
run_one_model() {
    local VERSION="$1"
    local RUN_ID="${RUN_IDS[$VERSION]}"
    local TASK_NAME="${TASK_NAMES[$VERSION]}"
    local TASK_CONFIG="${TASK_CONFIGS[$VERSION]}"
    local LABEL="${LABELS[$VERSION]}"

    local CKPT_PATH="${STARVLA_ROOT}/results/Checkpoints/${RUN_ID}/checkpoints/steps_${STEP}_pytorch_model.pt"
    local DEPLOY_YML="${SCRIPT_DIR}/deploy_policy_${VERSION}.yml"
    local ATTN_DIR="${OUTPUT_BASE}/${VERSION}"
    local CKPT_SETTING="${VERSION}_step${STEP}"

    mkdir -p "$ATTN_DIR"

    echo ""
    echo "========================================"
    echo "  [$VERSION] ${LABEL}"
    echo "  Task:   ${TASK_NAME} / ${TASK_CONFIG}"
    echo "  Attn:   ${ATTN_DIR}"
    echo "========================================"

    # --- Start policy server in background ---
    echo "  [1/3] Starting policy server..."
    export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

    STARVLA_SAVE_ATTENTION=1 \
    STARVLA_ATTENTION_DIR="$ATTN_DIR" \
    STARVLA_ATTENTION_INTERVAL="$ATTN_INTERVAL" \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$SERVER_PYTHON" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "$CKPT_PATH" \
        --port "$PORT" \
        --use_bf16 \
        --idle_timeout -1 \
        > "${ATTN_DIR}/server.log" 2>&1 &
    local SERVER_PID=$!

    # Wait for server to be ready (check port)
    echo "  Waiting for server (PID ${SERVER_PID}) on port ${PORT}..."
    local WAIT_COUNT=0
    local MAX_WAIT=300  # 300 seconds max (model loading can be slow)
    while true; do
        if "$SERVER_PYTHON" -c "
import socket, sys
s = socket.socket()
s.settimeout(1)
try:
    s.connect(('127.0.0.1', ${PORT}))
    s.close()
except Exception:
    sys.exit(1)
" 2>/dev/null; then
            break
        fi
        sleep 2
        WAIT_COUNT=$((WAIT_COUNT + 2))
        if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
            echo "  ERROR: Server did not start within ${MAX_WAIT}s"
            echo "  Server log tail:"
            tail -20 "${ATTN_DIR}/server.log"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
        fi
        # Check if server process died
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "  ERROR: Server process died."
            echo "  Server log tail:"
            tail -20 "${ATTN_DIR}/server.log"
            return 1
        fi
    done
    echo "  Server ready (waited ${WAIT_COUNT}s)."

    # --- Run evaluation ---
    echo "  [2/3] Running closed-loop eval (${TEST_NUM} episode(s), seed=${SEED})..."
    export PYTHONPATH="${AR_ROOT}:${STARVLA_ROOT}:${EVAL_FILES_DIR}:${PYTHONPATH:-}"

    cd "$AR_ROOT"
    PYTHONWARNINGS=ignore::UserWarning \
    CUDA_VISIBLE_DEVICES="$GPU_ID" \
        "$EVAL_PYTHON" script/eval_policy.py --config "$DEPLOY_YML" \
        --overrides \
        --task_name "$TASK_NAME" \
        --task_config "$TASK_CONFIG" \
        --ckpt_setting "$CKPT_SETTING" \
        --seed "$SEED" \
        --port "$PORT" \
        --policy_name model2robotwin_interface \
        --policy_ckpt_path "$CKPT_PATH" \
        --save_as_policy pi05_ee \
        --exp_idx "attn_analysis_${VERSION}" \
        --test_num "$TEST_NUM" \
        > "${ATTN_DIR}/eval.log" 2>&1 || {
            echo "  WARNING: Eval returned non-zero exit code. Check ${ATTN_DIR}/eval.log"
        }

    cd "$STARVLA_ROOT"

    # --- Kill server ---
    echo "  [3/3] Stopping server (PID ${SERVER_PID})..."
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true

    # Report
    local NUM_ATTN=$(ls "$ATTN_DIR"/step_*.npz 2>/dev/null | wc -l)
    echo "  Done: ${NUM_ATTN} attention snapshots saved."
}

# ── Main loop: evaluate all 4 models sequentially ──────────
if [ "$SKIP_EVAL" != "1" ]; then
    for V in "${VERSIONS[@]}"; do
        run_one_model "$V"
    done
else
    echo "SKIP_EVAL=1: Skipping evaluation, running visualization only."
fi

# ── Generate comparison visualizations ──────────────────────
echo ""
echo "========================================"
echo "Generating comparison visualizations..."
echo "========================================"

export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

"$VIZ_PYTHON" "${SCRIPT_DIR}/compare_attention.py" \
    --base_dir "$OUTPUT_BASE" \
    --v41_dir "${OUTPUT_BASE}/v41" \
    --v42_dir "${OUTPUT_BASE}/v42" \
    --v61_dir "${OUTPUT_BASE}/v61" \
    --v62_dir "${OUTPUT_BASE}/v62" \
    --output_dir "${OUTPUT_BASE}/comparison_figures"

echo ""
echo "============================================"
echo "ALL DONE"
echo "============================================"
echo "  Attention data: ${OUTPUT_BASE}/{v41,v42,v61,v62}/"
echo "  Figures:        ${OUTPUT_BASE}/comparison_figures/"
echo "============================================"
