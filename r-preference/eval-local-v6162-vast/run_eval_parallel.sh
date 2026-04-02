#!/usr/bin/env bash
# ============================================================
# Parallel evaluation on 4×5090 (vast machine)
#
# Runs 4 eval configs simultaneously — one per GPU.
# Each GPU loads its own policy server and runs one evaluation.
#
# GPU assignment:
#   GPU 0 (port 5694): v61 — place_stapler_stand | place_stapler_stand (clean)
#   GPU 1 (port 5695): v61 — place_stapler_stand | place_stapler_stand_wp4
#   GPU 2 (port 5696): v62 — place_stapler_stand | place_stapler_stand (clean)
#   GPU 3 (port 5697): v62 — place_stapler_stand | place_stapler_stand_wp5
#
# Prerequisites:
#   1. Checkpoints downloaded (step 1)
#   2. conda activate robotwin (or RoboTwin env)
#   3. STARVLA_PYTHON set to starVLA env python (for server)
#
# Usage:
#   conda activate robotwin
#   bash r-preference/eval-local-v6162-vast/run_eval_parallel.sh [seed] [step]
#
# Environment variables:
#   STARVLA_ROOT     -- starVLA repo (default: /home/user/starVLA)
#   STARVLA_PYTHON   -- starVLA env Python for server (default: python)
#   AR_ROOT          -- ar-research-kempner repo (default: /home/user/ar-research-kempner)
#   SERVER_WAIT      -- seconds to wait for server startup (default: 120)
#
# NOTE on wp4/wp5 environments:
#   If place_stapler_stand_wp4 / place_stapler_stand_wp5 are not in
#   ar-research-kempner's _eval_step_limit.yml, add them:
#     place_stapler_stand_wp4: 400
#     place_stapler_stand_wp5: 400
# ============================================================
set -uo pipefail

SEED="${1:-0}"
STEP="${2:-60000}"

STARVLA_ROOT="${STARVLA_ROOT:-/home/user/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
AR_ROOT="${AR_ROOT:-/home/user/ar-research-kempner}"
SERVER_WAIT="${SERVER_WAIT:-120}"

EVAL_DIR="$(cd "$(dirname "$0")" && pwd)"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${STARVLA_ROOT}/results/eval_logs/v6162_vast_seed${SEED}_step${STEP}_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

# ── RUN_IDs ───────────────────────────────────────────────────
declare -A RUN_IDS=(
    ["v61"]="v0320_v61_qwenOFT_finetune_v2"
    ["v62"]="v0320_v62_qwenOFT_finetune_v2"
)

# ── 4 eval configs: one per GPU ──────────────────────────────
# Format: "label|version|task_name|task_config|gpu_id|port"
EVAL_CONFIGS=(
    "v61_clean|v61|place_stapler_stand|place_stapler_stand|0|5694"
    "v61_wp4|v61|place_stapler_stand|place_stapler_stand_wp4|1|5695"
    "v62_clean|v62|place_stapler_stand|place_stapler_stand|2|5696"
    "v62_wp5|v62|place_stapler_stand|place_stapler_stand_wp5|3|5697"
)

# ── Helpers ───────────────────────────────────────────────────
wait_for_server() {
    local port=$1 max_wait=$2
    for i in $(seq 1 "$max_wait"); do
        if python -c "import socket; s=socket.socket(); s.settimeout(1); s.connect(('127.0.0.1', ${port})); s.close()" 2>/dev/null; then
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
    pids=$(lsof -ti tcp:"${port}" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "  Stopping server on port ${port} (pids: ${pids})"
        kill $pids 2>/dev/null || true
        sleep 3
    fi
}

# ── Run one eval config (server + eval) on one GPU ───────────
run_one_eval() {
    local label=$1 version=$2 task_name=$3 task_config=$4 gpu_id=$5 port=$6
    local run_id=${RUN_IDS[$version]}
    local ckpt="${STARVLA_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${STEP}_pytorch_model.pt"
    local log="${LOG_DIR}/${label}.log"
    local server_log="${LOG_DIR}/${label}_server.log"

    echo "[${label}] GPU=${gpu_id} PORT=${port} — ${version} ${task_name} ${task_config}" | tee -a "$log"

    # Check checkpoint
    if [ ! -f "$ckpt" ]; then
        echo "[${label}] SKIP: Checkpoint not found: $ckpt" | tee -a "$log"
        return 1
    fi

    # Kill any existing server on this port
    kill_server "$port" >> "$log" 2>&1

    # Start policy server
    echo "[${label}] Starting policy server..." | tee -a "$log"
    PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}" \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
        "$STARVLA_PYTHON" "${STARVLA_ROOT}/deployment/model_server/server_policy.py" \
        --ckpt_path "$ckpt" \
        --port "$port" \
        --use_bf16 \
        --idle_timeout -1 \
        > "$server_log" 2>&1 &
    local server_pid=$!
    echo "[${label}] Server PID: ${server_pid}" >> "$log"

    if ! wait_for_server "$port" "$SERVER_WAIT" >> "$log" 2>&1; then
        echo "[${label}] FAILED: Server did not start. See ${server_log}" | tee -a "$log"
        kill "$server_pid" 2>/dev/null || true
        return 1
    fi

    # Run evaluation
    local eval_start
    eval_start=$(date +%s)

    if PORT="$port" STEP="$STEP" STARVLA_ROOT="$STARVLA_ROOT" AR_ROOT="$AR_ROOT" \
       bash "${EVAL_DIR}/3_eval.sh" "$version" "$task_name" "$task_config" "$SEED" "$gpu_id" \
       >> "$log" 2>&1; then
        echo "[${label}] OK" | tee -a "$log"
    else
        echo "[${label}] FAILED" | tee -a "$log"
    fi

    local eval_end
    eval_end=$(date +%s)
    local elapsed=$((eval_end - eval_start))
    printf "[%s] Took %dm%ds\n" "$label" $((elapsed / 60)) $((elapsed % 60)) | tee -a "$log"

    # Cleanup server
    kill_server "$port" >> "$log" 2>&1
    kill "$server_pid" 2>/dev/null || true
}

# ── Preflight ─────────────────────────────────────────────────
echo ""
echo "########################################################################"
echo "# Parallel Evaluation: v61/v62 on 4×5090 (vast)"
echo "# $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"
echo ""
echo "  STARVLA_ROOT:   $STARVLA_ROOT"
echo "  STARVLA_PYTHON: $STARVLA_PYTHON"
echo "  AR_ROOT:        $AR_ROOT"
echo "  SEED:           $SEED"
echo "  STEP:           $STEP"
echo "  LOG_DIR:        $LOG_DIR"
echo ""

echo "Preflight: checking checkpoints..."
for version in v61 v62; do
    run_id="${RUN_IDS[$version]}"
    ckpt="${STARVLA_ROOT}/results/Checkpoints/${run_id}/checkpoints/steps_${STEP}_pytorch_model.pt"
    if [ -f "$ckpt" ]; then
        size=$(du -h "$ckpt" | cut -f1)
        echo "  OK:   ${version} (${size})"
    else
        echo "  MISSING: ${version} — ${ckpt}"
    fi
done

echo ""
echo "Eval configs:"
for entry in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r label version task_name task_config gpu_id port <<< "$entry"
    echo "  GPU ${gpu_id} (port ${port}): ${label} — ${task_name} / ${task_config}"
done
echo ""

# ── Launch all 4 evals in parallel ───────────────────────────
PHASE_START=$(date +%s)
declare -A PIDS

for entry in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r label version task_name task_config gpu_id port <<< "$entry"
    echo "Launching ${label}..."
    run_one_eval "$label" "$version" "$task_name" "$task_config" "$gpu_id" "$port" &
    PIDS[$label]=$!
done

echo ""
echo "All 4 evals launched. Waiting for completion..."
echo "  Logs: $LOG_DIR"
echo ""

# ── Wait and collect results ─────────────────────────────────
declare -A RESULTS
TOTAL_FAILED=0

for label in "${!PIDS[@]}"; do
    pid=${PIDS[$label]}
    if wait "$pid"; then
        RESULTS[$label]="OK"
    else
        RESULTS[$label]="FAILED"
        TOTAL_FAILED=$((TOTAL_FAILED + 1))
    fi
done

PHASE_END=$(date +%s)
TOTAL_ELAPSED=$((PHASE_END - PHASE_START))

# ── Summary ──────────────────────────────────────────────────
echo ""
echo "########################################################################"
echo "# Results Summary"
echo "# $(date '+%Y-%m-%d %H:%M:%S')"
echo "########################################################################"
echo ""

for entry in "${EVAL_CONFIGS[@]}"; do
    IFS='|' read -r label version task_name task_config gpu_id port <<< "$entry"
    status="${RESULTS[$label]:-UNKNOWN}"
    echo "  [GPU ${gpu_id}] ${label}: ${status}  (${task_name} / ${task_config})"
done

echo ""
printf "Total time: %dm%ds\n" $((TOTAL_ELAPSED / 60)) $((TOTAL_ELAPSED % 60))
echo "Failed: ${TOTAL_FAILED} / ${#EVAL_CONFIGS[@]}"
echo "Seed: ${SEED}  |  Step: ${STEP}"
echo "Logs: ${LOG_DIR}"
echo ""
