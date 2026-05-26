#!/bin/bash
# Evaluate StarVLA v30-ee and v31-ee on place_object_stand_wp4.
#
# Prerequisites (Desktop):
#   1. Checkpoints downloaded to ~/Desktop/research/starVLA/results/Checkpoints/
#      (run r-preference/temp/0317-download_checkpoint_3031_from_hf.py)
#   2. starVLA_ee/ policy module copied to ar-research-kempner/policy/starVLA_ee/
#      (cp -r r-preference/eval-local/starVLA_ee ar-research-kempner/policy/)
#   3. StarVLA conda env with server dependencies installed
#   4. RoboTwin conda env active for the sim
#
# Usage:
#   conda activate RoboTwin
#   bash r-preference/eval-local/run_eval_v3031.sh [seed] [step]
#     seed: random seed (default: 0)
#     step: checkpoint step (default: 50000)
#
# Environment variables (override as needed):
#   AR_ROOT        -- ar-research-kempner repo (default: ~/Desktop/research/ar-research-kempner)
#   STARVLA_ROOT   -- starVLA repo (default: ~/Desktop/research/starVLA)
#   STARVLA_PYTHON -- Python for policy server (default: ~/miniconda3/envs/starVLA/bin/python)
#   GPU_ID         -- GPU device (default: 0)
#   SERVER_WAIT    -- seconds to wait for server startup (default: 90)

set -euo pipefail

SEED="${1:-0}"
STEP="${2:-50000}"

AR_ROOT="${AR_ROOT:-${HOME}/Desktop/research/ar-research-kempner}"
STARVLA_ROOT="${STARVLA_ROOT:-${HOME}/Desktop/research/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${HOME}/miniconda3/envs/starVLA/bin/python}"
GPU_ID="${GPU_ID:-0}"
SERVER_WAIT="${SERVER_WAIT:-90}"
PORT=5695

EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT_BASE="${STARVLA_ROOT}/results/Checkpoints"

# v30 and v31 configs: version|run_name|deploy_yml|exp_idx
VERSIONS=(
    "v30-ee|v0309_v30_ee_qwenPI_requeue|${EVAL_DIR}/deploy_policy_ee_v30.yml|30"
    "v31-ee|v0309_v31_ee_qwenPI_requeue|${EVAL_DIR}/deploy_policy_ee_v31.yml|31"
)
TASK_NAME="place_object_stand"
TASK_CONFIG="place_object_stand_wp4"
TOTAL=${#VERSIONS[@]}

# ── Helpers ──────────────────────────────────────────────────────────────────

STEP_LOG() {
    echo ""
    echo "########################################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "########################################################################"
    echo ""
}

wait_for_server() {
    local port=$1
    local max_wait=$2
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
        sleep 2
    fi
}

# ── Preflight checks ─────────────────────────────────────────────────────────

STEP_LOG "Preflight checks"

# Check policy module
if [ ! -f "${AR_ROOT}/policy/starVLA_ee/__init__.py" ]; then
    echo "ERROR: policy module not found at ${AR_ROOT}/policy/starVLA_ee/"
    echo "Run: cp -r ${EVAL_DIR}/starVLA_ee ${AR_ROOT}/policy/"
    exit 1
fi
echo "  OK: policy/starVLA_ee/ found"

# Check STARVLA_PYTHON
if [ ! -f "$STARVLA_PYTHON" ]; then
    echo "ERROR: Python not found: $STARVLA_PYTHON"
    echo "Set STARVLA_PYTHON to the starVLA conda env Python binary."
    exit 1
fi
echo "  OK: STARVLA_PYTHON = $STARVLA_PYTHON"

# Check checkpoints
for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r version run_name deploy_yml exp_idx <<< "$entry"
    ckpt="${CKPT_BASE}/${run_name}/checkpoints/steps_${STEP}_pytorch_model.pt"
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: checkpoint not found: $ckpt"
        echo "Run: python r-preference/temp/0317-download_checkpoint_3031_from_hf.py --steps $STEP"
        exit 1
    fi
    echo "  OK: $version checkpoint found (step $STEP)"
done

# ── Evaluation loop ───────────────────────────────────────────────────────────

STEP_LOG "Evaluations (seed=${SEED}, step=${STEP})"

EVAL_NUM=0
EVAL_FAILED=0
PHASE_START=$(date +%s)

for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r version run_name deploy_yml exp_idx <<< "$entry"
    EVAL_NUM=$((EVAL_NUM + 1))
    CKPT_PATH="${CKPT_BASE}/${run_name}/checkpoints/steps_${STEP}_pytorch_model.pt"

    echo ""
    echo "======================================================"
    echo "[${EVAL_NUM}/${TOTAL}] ${version}: ${TASK_NAME} x ${TASK_CONFIG}"
    echo "  Checkpoint: ${CKPT_PATH}"
    echo "  exp_idx:    ${exp_idx}"
    echo "======================================================"

    # Kill any leftover server
    kill_server $PORT

    # Start policy server in background
    echo "  Starting policy server (port ${PORT})..."
    STARVLA_ROOT="$STARVLA_ROOT" \
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        "$STARVLA_PYTHON" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "$CKPT_PATH" \
        --port "$PORT" \
        --use_bf16 \
        --idle_timeout -1 \
        > /tmp/starvla_server_${version}.log 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: ${SERVER_PID} (log: /tmp/starvla_server_${version}.log)"

    # Wait for server to be ready
    if ! wait_for_server $PORT $SERVER_WAIT; then
        echo "  Server log:"
        tail -20 /tmp/starvla_server_${version}.log || true
        kill $SERVER_PID 2>/dev/null || true
        EVAL_FAILED=$((EVAL_FAILED + 1))
        continue
    fi

    # Run evaluation (sim in RoboTwin env)
    VERSION_START=$(date +%s)
    if STARVLA_ROOT="$STARVLA_ROOT" \
       bash "${EVAL_DIR}/eval_starvla_ee.sh" \
           "$TASK_NAME" "$TASK_CONFIG" "$deploy_yml" "$SEED" "$GPU_ID" "$exp_idx"; then
        echo "[${version}] Evaluation succeeded."
    else
        echo "[${version}] Evaluation FAILED."
        EVAL_FAILED=$((EVAL_FAILED + 1))
    fi
    VERSION_END=$(date +%s)
    VERSION_ELAPSED=$((VERSION_END - VERSION_START))

    # Stop server
    kill_server $PORT

    # Progress
    TOTAL_ELAPSED=$((VERSION_END - PHASE_START))
    REMAINING=$((TOTAL - EVAL_NUM))
    if [ $EVAL_NUM -gt 0 ]; then
        AVG=$((TOTAL_ELAPSED / EVAL_NUM))
        ETA=$((AVG * REMAINING))
        printf "\n>>> Progress: %d/%d done | This: %dm %ds | Total elapsed: %dm %ds" \
            "$EVAL_NUM" "$TOTAL" \
            $((VERSION_ELAPSED / 60)) $((VERSION_ELAPSED % 60)) \
            $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60))
        if [ $REMAINING -gt 0 ]; then
            printf " | ETA: ~%dm %ds (%d remaining)" \
                $((ETA / 60)) $((ETA % 60)) "$REMAINING"
        fi
        echo -e "\n"
    fi
done

# ── Summary ───────────────────────────────────────────────────────────────────

STEP_LOG "Results Summary"

echo "======================================================"
echo "  Version  | Task Config              | Results"
echo "======================================================"
for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r version run_name deploy_yml exp_idx <<< "$entry"
    result_dir="${AR_ROOT}/policy/starVLA_ee/eval_logs/exp-idx${exp_idx}/eval_results/${TASK_NAME}/${TASK_CONFIG}"
    result_file="${result_dir}/_result.txt"
    if [ -f "$result_file" ]; then
        sr_line=$(grep -i "combined success rate" "$result_file" 2>/dev/null | head -1 || echo "N/A")
        printf "  %-8s | %-24s | %s\n" "$version" "$TASK_CONFIG" "$sr_line"
    else
        printf "  %-8s | %-24s | %s\n" "$version" "$TASK_CONFIG" "(no results)"
    fi
done
echo "======================================================"
echo ""
echo "Detailed results at:"
for entry in "${VERSIONS[@]}"; do
    IFS='|' read -r version run_name deploy_yml exp_idx <<< "$entry"
    echo "  ${version}: ${AR_ROOT}/policy/starVLA_ee/eval_logs/exp-idx${exp_idx}/eval_results/"
done
echo ""
echo "======================================================"
echo "All evaluations complete!"
echo "  Total:  $TOTAL"
echo "  Failed: $EVAL_FAILED"
echo "  Seed:   $SEED  |  Step: $STEP"
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"
