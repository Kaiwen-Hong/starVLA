#!/usr/bin/env bash
# ============================================================
# temp-h100-continue.sh — resume H100 pref-VLA training where temp-h100.sh
# left off (after user paused for other scripts).
#
# Picks up from latest saved ckpt for the in-flight run, then runs the
# remaining fresh run.
#
# Runs in order:
#   01-hvlv-baseline-RESUME  IS_RESUME=1 → reads latest steps_N from
#                            results/Checkpoints/pref_baseline_stage_a_v1_noVQA_hvlv/checkpoints/
#                            loads model weights + completed_steps=N + lr scheduler
#                            fast-forward. Trains until 25000.
#                            Caveat: Adam optimizer momentum resets (only
#                            model weights are saved). Negligible for a
#                            25k-step run — momentum re-accumulates in
#                            ~100 steps.
#   02-height-vqa-FRESH      Normal fresh launch from Qwen3-VL-4B base.
#
# Behavior:
#   - Continues to next run even if one fails (same as temp-h100.sh).
#   - Cleanup orphan procs between runs, 30s sleep for GPU release.
#   - Per-run logs at results/.temp_h100_runs/<tag>.log
#   - Top-level summary at results/.temp_h100_runs/_summary.log
#
# Usage (in tmux):
#   tmux new -s pref_continue
#   bash examples/preference/temp-h100-continue.sh
# ============================================================

set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_h100_runs"
LOG_FILE="$LOG_DIR/_summary.log"
mkdir -p "$LOG_DIR"

cd "$REPO"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log_top() { echo "$@" | tee -a "$LOG_FILE"; }

cleanup_orphans() {
  if pgrep -af "starvla_pref|train_starvla.py|accelerate\.launch|pt_elastic" > /dev/null 2>&1; then
    log_top "  [$(ts)] cleanup: pkill any orphan train/accelerate/pt_elastic procs"
    pkill -9 -f "starvla_pref"      2>/dev/null || true
    pkill -9 -f "train_starvla.py"  2>/dev/null || true
    pkill -9 -f "accelerate.launch" 2>/dev/null || true
    pkill -9 -f "pt_elastic"        2>/dev/null || true
    sleep 30
    log_top "  [$(ts)] cleanup: done, GPU should be free"
  fi
}

# Args: <tag> <script> [<key=value> ...]   — KV pairs become env vars for the launcher.
run_one() {
  local tag="$1"; shift
  local script="$1"; shift
  local env_assigns=("$@")
  local rlog="$LOG_DIR/$tag.log"

  log_top "============================================================"
  log_top "[$(ts)] BEGIN  $tag"
  log_top "  script: $script"
  log_top "  envs:   ${env_assigns[*]:-(none)}"
  log_top "  rlog:   $rlog"
  log_top "============================================================"

  cleanup_orphans

  local rc=0
  env "${env_assigns[@]}" bash "$script" 2>&1 | tee "$rlog"
  rc=${PIPESTATUS[0]}

  if [ "$rc" -eq 0 ]; then
    log_top "[$(ts)] DONE   $tag"
  else
    log_top "[$(ts)] FAILED $tag  (exit=$rc)"
    log_top "  last 10 lines of $rlog:"
    tail -10 "$rlog" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG_FILE"
    log_top "  -> CONTINUING to next run anyway"
  fi
}

log_top ""
log_top "############################################################"
log_top "# [$(ts)] H100 CONTINUE — 2 runs left"
log_top "# host: $(hostname) user: $(whoami)"
log_top "# logs: $LOG_DIR/"
log_top "############################################################"

run_one "02-hvlv-baseline-resume" examples/preference/launch_pref_stage_a_baseline_hvlv.sh \
        IS_RESUME=1
run_one "03-height-vqa"           examples/preference/launch_pref_stage_a_vqa_height.sh

log_top "############################################################"
log_top "# [$(ts)] H100 CONTINUE ALL DONE (check $LOG_FILE for per-run status)"
log_top "############################################################"
