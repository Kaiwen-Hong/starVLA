#!/usr/bin/env bash
# ============================================================================
# Overnight batch evaluation: starVLA (joint-space, 14D)
#
# Automatically starts/stops policy server per version, runs eval, frees GPU.
#
# Evaluations:
#   v41: place_stapler_stand / place_stapler_stand_wp4
#   v42: place_stapler_stand / place_stapler_stand_wp5
#   v52: place_stapler_stand / place_stapler_stand_wp5
#
# Prerequisites:
#   1. Checkpoints downloaded (step 1)
#   2. Run from the robotwin conda env
#
# Usage:
#   conda activate robotwin
#   bash r-preference/eval-local-v4152/run_overnight.sh [test_num] [gpu_id] [step]
#
# Examples:
#   bash r-preference/eval-local-v4152/run_overnight.sh 1          # trial run
#   bash r-preference/eval-local-v4152/run_overnight.sh 150        # full run
#   bash r-preference/eval-local-v4152/run_overnight.sh 150 0 60000
# ============================================================================
set -euo pipefail

TEST_NUM="${1:-150}"
GPU_ID="${2:-0}"
STEP="${3:-60000}"
SEED=0   # st_seed = 100000 * (1 + seed) = 100000

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${HOME}/miniconda3/envs/starVLA/bin/python}"
PORT="${PORT:-5694}"
SERVER_WAIT="${SERVER_WAIT:-180}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Evaluation configurations ───────────────────────────────
# Format: version|task_name|task_config
EVAL_CONFIGS=(
    "v41|place_stapler_stand|place_stapler_stand_wp4"
    "v42|place_stapler_stand|place_stapler_stand_wp5"
    "v52|place_stapler_stand|place_stapler_stand_wp5"
)

# ── Run ID mapping ──────────────────────────────────────────
declare -A RUN_IDS=(
    ["v41"]="v0320_v41_qwenOFT_finetune_v2"
    ["v42"]="v0320_v42_qwenOFT_finetune_v2"
    ["v52"]="v0320_v52_qwenOFT_finetune_v1"
)

# ── Helpers ─────────────────────────────────────────────────
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
    echo "  Waiting for server on port ${port} (max ${max_wait}s)..."
    for i in $(seq 1 "$max_wait"); do
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
    pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Stopping server on port ${port} (pids: ${pids})"
        kill $pids 2>/dev/null || true
        sleep 5
        # Force kill if still alive
        pids=$(lsof -ti "tcp:${port}" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "  Force killing remaining pids: ${pids}"
            kill -9 $pids 2>/dev/null || true
            sleep 2
        fi
    fi
    echo "  Server stopped, GPU memory freed."
}

# ── Preflight checks ───────────────────────────────────────
STEP_LOG "Preflight checks"

if [ ! -f "$STARVLA_PYTHON" ]; then
    echo "ERROR: STARVLA_PYTHON not found: $STARVLA_PYTHON"
    echo "Set STARVLA_PYTHON to the starVLA conda env Python binary."
    exit 1
fi

echo "  STARVLA_ROOT:   $STARVLA_ROOT"
echo "  STARVLA_PYTHON: $STARVLA_PYTHON"
echo "  GPU_ID:         $GPU_ID"
echo "  PORT:           $PORT"
echo "  STEP:           $STEP"
echo "  SEED:           $SEED (st_seed=100000)"
echo "  TEST_NUM:       $TEST_NUM"
echo ""

# Check all checkpoints exist
for entry in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r version task_name task_config <<< "$entry"
    run_id="${RUN_IDS[$version]}"
    ckpt="${STARVLA_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${STEP}_pytorch_model.pt"
    if [ -f "$ckpt" ]; then
        echo "  OK: ${version} checkpoint found"
    else
        echo "  MISSING: ${ckpt}"
        echo "  Run: bash r-preference/eval-local-v4152/1_download_checkpoints.sh"
        exit 1
    fi
done

# Kill any existing server on the port
kill_server $PORT

# ── Main evaluation loop ───────────────────────────────────
STEP_LOG "Starting evaluations (${#EVAL_CONFIGS[@]} total, test_num=${TEST_NUM})"

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
    echo "  Checkpoint: ${ckpt}"
    echo "  Seed: ${SEED} (st_seed=100000)"
    echo "  Rollouts: ${TEST_NUM}"
    echo "======================================================"

    # Start new server if version changed
    if [ "$version" != "$CURRENT_SERVER_VERSION" ]; then
        kill_server $PORT

        echo "  Starting policy server for ${version}..."
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
            echo "  Server failed to start. Last 30 lines of log:"
            tail -30 "/tmp/starvla_server_${version}.log" || true
            kill $SERVER_PID 2>/dev/null || true
            EVAL_FAILED=$((EVAL_FAILED + 1))
            continue
        fi
        CURRENT_SERVER_VERSION="$version"
    fi

    # Run evaluation
    EVAL_START=$(date +%s)
    if TEST_NUM=$TEST_NUM PORT=$PORT STEP=$STEP STARVLA_ROOT="$STARVLA_ROOT" \
       bash "${EVAL_DIR}/3_eval.sh" "$version" "$task_name" "$task_config" "$SEED" "$GPU_ID"; then
        echo "  [${version}/${task_config}] Evaluation succeeded."
    else
        echo "  [${version}/${task_config}] Evaluation FAILED (exit code $?)."
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi

    EVAL_END=$(date +%s)
    EVAL_ELAPSED=$((EVAL_END - EVAL_START))
    TOTAL_ELAPSED=$((EVAL_END - PHASE_START))
    REMAINING=$((TOTAL - EVAL_NUM))

    printf "\n>>> [%d/%d] This: %dm%ds | Total: %dm%ds | Remaining: %d" \
        "$EVAL_NUM" "$TOTAL" \
        $((EVAL_ELAPSED / 60)) $((EVAL_ELAPSED % 60)) \
        $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60)) \
        "$REMAINING"
    if [ $REMAINING -gt 0 ]; then
        AVG=$((TOTAL_ELAPSED / EVAL_NUM))
        ETA=$((AVG * REMAINING))
        printf " | ETA: ~%dm%ds" $((ETA / 60)) $((ETA % 60))
    fi
    echo -e "\n"
done

# ── Cleanup ─────────────────────────────────────────────────
kill_server $PORT

# ── Summary ─────────────────────────────────────────────────
STEP_LOG "Results Summary"

echo "  Total:    $TOTAL"
echo "  Failed:   $EVAL_FAILED"
echo "  Seed:     $SEED (st_seed=100000)"
echo "  Step:     $STEP"
echo "  Test Num: $TEST_NUM"
echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
