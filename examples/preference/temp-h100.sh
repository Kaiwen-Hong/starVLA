#!/usr/bin/env bash
# ============================================================
# temp-h100.sh — sequential pref-VLA training on H100 (kaiwenh)
#
# 3 runs back-to-back:
#   01  height baseline (~8h)
#   02  hvlv baseline   (~8h)
#   03  height VQA      (~9h)
#
# Behavior:
#   - Continues to next run even if one fails (does NOT abort whole script).
#   - Between runs, kills any orphan pref-VLA processes and waits 30s for
#     GPU memory to release.
#   - tees each run's full stdout/stderr to a separate per-run log.
#   - Top-level log at $LOG_FILE captures BEGIN/DONE/FAILED markers only.
#
# Usage (in tmux):
#   tmux new -s pref_h100
#   bash examples/preference/temp-h100.sh
#   # Ctrl+B D to detach; `tmux attach -t pref_h100` to resume.
# ============================================================

# NOTE: deliberately NO `set -e` — we want failures to advance to next run.
set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_h100_runs"
LOG_FILE="$LOG_DIR/_summary.log"
mkdir -p "$LOG_DIR"

cd "$REPO"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

log_top() {
  # Plain echo: tee-via-fd to avoid duplicating in stdout AND the per-run log.
  echo "$@" | tee -a "$LOG_FILE"
}

cleanup_orphans() {
  # Only matches OUR project names (don't touch other users' jobs).
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

run_one() {
  local tag="$1"; shift
  local script="$1"
  local rlog="$LOG_DIR/$tag.log"

  log_top "============================================================"
  log_top "[$(ts)] BEGIN  $tag"
  log_top "  script: $script"
  log_top "  rlog:   $rlog"
  log_top "============================================================"

  cleanup_orphans

  # pipefail makes `bash $script 2>&1 | tee` return bash's exit code if non-zero.
  local rc=0
  bash "$script" 2>&1 | tee "$rlog"
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
log_top "# [$(ts)] H100 sequential pref-VLA: 3 runs"
log_top "# host: $(hostname) user: $(whoami)"
log_top "# logs: $LOG_DIR/"
log_top "############################################################"

run_one "01-height-baseline" examples/preference/launch_pref_stage_a_baseline_height.sh
run_one "02-hvlv-baseline"   examples/preference/launch_pref_stage_a_baseline_hvlv.sh
run_one "03-height-vqa"      examples/preference/launch_pref_stage_a_vqa_height.sh

log_top "############################################################"
log_top "# [$(ts)] H100 ALL DONE (check $LOG_FILE for per-run status)"
log_top "############################################################"
