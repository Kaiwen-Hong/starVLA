#!/usr/bin/env bash
# ============================================================
# Batch evaluation: auto start server + eval + stop server
#
# Runs all v61/v62 models on their trained tasks.
# This script manages the policy server lifecycle automatically.
#
# Prerequisites:
#   1. Checkpoints downloaded (step 1)
#   2. starVLA conda env available
#   3. RoboTwin conda env is the CURRENT env (run this from robotwin env)
#
# Usage:
#   conda activate robotwin
#   bash r-preference/eval-local-v6162/run_eval_batch.sh [seed] [step]
#
# Environment variables:
#   STARVLA_ROOT   -- starVLA repo path (auto-detected)
#   STARVLA_PYTHON -- starVLA env Python (default: ~/miniconda3/envs/starVLA/bin/python)
#   AR_ROOT        -- ar-research-kempner repo path
#   GPU_ID         -- GPU device (default: 0)
#   SERVER_WAIT    -- seconds to wait for server (default: 120)
# ============================================================
set -euo pipefail

SEED="${1:-0}"
STEP="${2:-60000}"

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${HOME}/miniconda3/envs/starVLA/bin/python}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"
SERVER_WAIT="${SERVER_WAIT:-120}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Evaluation configurations ────────────────────────────────
# Format: version|task_name|task_config
EVAL_CONFIGS=(
    "v61|place_cup5_tray5|demo_clean"
    "v61|place_stapler_stand|demo_clean"
    "v62|place_cup5_tray5|demo_clean"
    "v62|place_stapler_stand|demo_clean"
)

# ── Run ID mapping ───────────────────────────────────────────
declare -A RUN_IDS=(
    ["v61"]="v0320_v61_qwenOFT_finetune_v2"
    ["v62"]="v0320_v62_qwenOFT_finetune_v2"
)

# ── Helpers ──────────────────────────────────────────────────
STEP_LOG() {
    echo ""
    echo "########################################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "########################################################################"
    echo ""
}

wait_for_server() {
    local port=$1 max_wait=$2
    echo "  Waiting for server on port ${port}..."
    for i in $(seq 1 $max_wait); do
        if python3 -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${port})); s.close()" 2>/dev/null; then
            echo "  Server ready after ${i}s."
            return 0
        fi
        sleep 1
    done
    echo "  ERROR: Server did not start within ${max_wait}s."
    return 1
}

kill_server() {
    local port=$1
    local pids
    pids=$(lsof -ti tcp:${port} 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Stopping server on port ${port} (pids: ${pids})"
        kill $pids 2>/dev/null || true
        sleep 3
    fi
}

# ── Preflight ────────────────────────────────────────────────
STEP_LOG "Preflight checks"

if [ ! -f "$STARVLA_PYTHON" ]; then
    echo "ERROR: STARVLA_PYTHON not found: $STARVLA_PYTHON"
    echo "Set STARVLA_PYTHON to the starVLA conda env Python binary."
    exit 1
fi
echo "  STARVLA_PYTHON: $STARVLA_PYTHON"
echo "  STARVLA_ROOT:   $STARVLA_ROOT"

for version in v61 v62; do
    run_id="${RUN_IDS[$version]}"
    ckpt="${STARVLA_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${STEP}_pytorch_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  OK: ${version} checkpoint found"
    else
        echo "  MISSING: ${ckpt}"
        echo "  Run step 1 first!"
        exit 1
    fi
done

# ── Main evaluation loop ────────────────────────────────────
TOTAL=${#EVAL_CONFIGS[@]}
EVAL_NUM=0
EVAL_FAILED=0
CURRENT_SERVER_VERSION=""
PHASE_START=$(date +%s)

for entry in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r version task_name task_config <<< "$entry"
    EVAL_NUM=$((EVAL_NUM + 1))
    run_id="${RUN_IDS[$version]}"
    ckpt="${STARVLA_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${STEP}_pytorch_model.pt"

    echo ""
    echo "======================================================"
    echo "[${EVAL_NUM}/${TOTAL}] ${version}: ${task_name} / ${task_config}"
    echo "======================================================"

    # Start new server if version changed
    if [ "$version" != "$CURRENT_SERVER_VERSION" ]; then
        kill_server $PORT

        echo "  Starting server for ${version}..."
        PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
        CUDA_VISIBLE_DEVICES=$GPU_ID \
            "$STARVLA_PYTHON" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
            --ckpt_path "$ckpt" \
            --port "$PORT" \
            --use_bf16 \
            --idle_timeout -1 \
            > "/tmp/starvla_server_${version}.log" 2>&1 &
        SERVER_PID=$!
        echo "  Server PID: ${SERVER_PID}"

        if ! wait_for_server $PORT $SERVER_WAIT; then
            echo "  Server failed to start. Log:"
            tail -30 "/tmp/starvla_server_${version}.log" || true
            kill $SERVER_PID 2>/dev/null || true
            EVAL_FAILED=$((EVAL_FAILED + 1))
            continue
        fi
        CURRENT_SERVER_VERSION="$version"
    fi

    # Run eval
    EVAL_START=$(date +%s)
    if PORT=$PORT STEP=$STEP STARVLA_ROOT="$STARVLA_ROOT" \
       bash "${EVAL_DIR}/3_eval.sh" "$version" "$task_name" "$task_config" "$SEED" "$GPU_ID"; then
        echo "  [${version}/${task_name}] Evaluation succeeded."
    else
        echo "  [${version}/${task_name}] Evaluation FAILED."
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi

    EVAL_END=$(date +%s)
    EVAL_ELAPSED=$((EVAL_END - EVAL_START))
    TOTAL_ELAPSED=$((EVAL_END - PHASE_START))
    REMAINING=$((TOTAL - EVAL_NUM))
    printf ">>> [%d/%d] This: %dm%ds | Total: %dm%ds | Remaining: %d\n\n" \
        "$EVAL_NUM" "$TOTAL" \
        $((EVAL_ELAPSED / 60)) $((EVAL_ELAPSED % 60)) \
        $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60)) \
        "$REMAINING"
done

# Cleanup
kill_server $PORT

# ── Summary ──────────────────────────────────────────────────
STEP_LOG "Results Summary"

echo "  Total:  $TOTAL"
echo "  Failed: $EVAL_FAILED"
echo "  Seed:   $SEED  |  Step: $STEP"
echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
