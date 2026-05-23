#!/usr/bin/env bash
# Watchdog v2 for the preference-VLA Stage A training run.
#
# Differentiates:
#   STABLE_LOW_FREE  — GPU min free is low but training is still progressing (informational, NOT alert)
#   GPU_LOW_FREE     — first time we see low free, expected to stabilize (1 grace cycle, NOT alert)
#   OOM_CRASH        — CUDA OOM in train.log (alert)
#   PROCESS_DEAD     — train PID gone (alert)
#   STEP_STUCK       — step number unchanged across multiple intervals (alert)
#   STEP_DEGRADED    — step rate dropped >50% vs initial (alert)
#
# Writes one line per cycle to $LOG_FILE. Touches $ALERT_FILE only on real
# problems (OOM, dead, stuck, degraded). The user can `cat` either to see
# situation when they wake up.

# NB: NOT using `set -e` because (( $(bc) )) returns 1 when bc emits "0",
# which would kill the script on every normal "free > threshold" check.
set -uo pipefail

TRAIN_PID="${1:?usage: watchdog.sh <train_pid>}"
INTERVAL="${INTERVAL:-1800}"
RUN_DIR=/home/kaiwenh/starVLA/results/Checkpoints/pref_baseline_stage_a_v1_noVQA
LOG_FILE=${RUN_DIR}/watchdog.log
ALERT_FILE=${RUN_DIR}/WATCHDOG_ALERT
TRAIN_LOG=${RUN_DIR}/train.log

mkdir -p "$RUN_DIR"
echo "[$(date -u +'%Y-%m-%dT%H:%M:%SZ')] watchdog v2 start  TRAIN_PID=$TRAIN_PID  INTERVAL=${INTERVAL}s" >> "$LOG_FILE"

prev_step=-1
prev_step_ts=$(date +%s)
init_step_per_sec=""

while true; do
  ts=$(date -u +'%Y-%m-%dT%H:%M:%SZ')

  # 1) process alive?
  if ! ps -p "$TRAIN_PID" > /dev/null 2>&1; then
    echo "[$ts] PROCESS_DEAD pid=$TRAIN_PID" >> "$LOG_FILE"
    {
      echo "PROCESS_DEAD at $ts"
      echo "PID: $TRAIN_PID"
      echo "Last 30 lines of train.log:"
      tail -30 "$TRAIN_LOG" 2>&1
    } >> "$ALERT_FILE"
    exit 1
  fi

  # 2) CUDA OOM in train log?
  oom_hit=$(grep -iE "out of memory|CUDA error" "$TRAIN_LOG" 2>/dev/null | tail -1)
  if [ -n "$oom_hit" ]; then
    echo "[$ts] OOM_CRASH" >> "$LOG_FILE"
    {
      echo "OOM_CRASH at $ts"
      echo "Match: $oom_hit"
      tail -30 "$TRAIN_LOG"
    } >> "$ALERT_FILE"
  fi

  # 3) GPU min-free
  min_free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  min_free_gb=$(awk "BEGIN{printf \"%.1f\", $min_free_mib / 1024}")

  # 4) last step
  last_step=$(grep -oE '[[:space:]][0-9]+/50000[[:space:]]\[' "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -oE '^[[:space:]]*[0-9]+' | tr -d ' ' || echo "")
  last_step="${last_step:-?}"

  status="OK"

  # known stable low-free is not an alert anymore
  if (( $(echo "$min_free_gb < 0.1" | bc -l) )); then
    # really at the brink — only alert
    status="GPU_CRITICAL_FREE"
    {
      echo "GPU_CRITICAL_FREE at $ts (min free ${min_free_gb} GB)"
      nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
    } >> "$ALERT_FILE"
  elif (( $(echo "$min_free_gb < 1.5" | bc -l) )); then
    # known steady-state low, log informational, do not alert
    status="STABLE_LOW_FREE_${min_free_gb}GB"
  fi

  # step progress check
  now=$(date +%s)
  if [[ "$last_step" != "?" ]]; then
    if [[ "$last_step" == "$prev_step" ]]; then
      stuck_s=$((now - prev_step_ts))
      if (( stuck_s > 3 * INTERVAL )); then
        status="STEP_STUCK_${stuck_s}s"
        {
          echo "STEP_STUCK at $ts (step=$last_step held for ${stuck_s}s)"
          tail -20 "$TRAIN_LOG"
        } >> "$ALERT_FILE"
      fi
    else
      # step advanced — record initial rate after first 3 cycles
      if [ -z "$init_step_per_sec" ] && [ "$prev_step" -gt 0 ]; then
        dt=$((now - prev_step_ts))
        if (( dt > 0 )); then
          delta=$((last_step - prev_step))
          init_step_per_sec=$(awk "BEGIN{printf \"%.4f\", $delta / $dt}")
        fi
      fi
      # rate degradation check
      if [ -n "$init_step_per_sec" ]; then
        dt=$((now - prev_step_ts))
        if (( dt > 0 )); then
          delta=$((last_step - prev_step))
          cur_rate=$(awk "BEGIN{printf \"%.4f\", $delta / $dt}")
          slowdown_ratio=$(awk "BEGIN{printf \"%.2f\", $cur_rate / $init_step_per_sec}")
          if (( $(echo "$slowdown_ratio < 0.5" | bc -l) )); then
            status="${status},STEP_DEGRADED_${slowdown_ratio}x"
            {
              echo "STEP_DEGRADED at $ts (current rate $cur_rate vs initial $init_step_per_sec)"
            } >> "$ALERT_FILE"
          fi
        fi
      fi
      prev_step="$last_step"
      prev_step_ts=$now
    fi
  fi

  # latest checkpoint
  latest_ckpt=$(ls -t "$RUN_DIR/checkpoints" 2>/dev/null | head -1 || echo "(none yet)")

  echo "[$ts] step=${last_step} min_gpu_free=${min_free_gb}GB latest_ckpt=${latest_ckpt} status=${status}" >> "$LOG_FILE"

  sleep "$INTERVAL"
done
