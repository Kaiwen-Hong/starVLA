#!/usr/bin/env bash
# ============================================================
# temp-stageb-contact.sh — sequential Stage B (main + B0) on H100
#
# 2 runs back-to-back:
#   01  pref_main_stage_b_v1_contact   (1500 step, ETA 30 min - 1.5 h)
#   02  pref_b0_stage_b_v1_contact     (1500 step, ETA 30 min - 1.5 h)
#
# Behavior (same pattern as temp-h100.sh):
#   - Continues to next run even if one fails (does NOT abort whole script)
#   - Between runs: kills any orphan pref-VLA processes; sleeps 30s for
#     GPU memory release
#   - Tees each run's full stdout/stderr to a per-run log
#   - Top-level summary log captures BEGIN/DONE/FAILED markers only
#
# Usage (in tmux):
#   tmux new -s pref_stageb
#   bash examples/preference/temp-stageb-contact.sh
#   # Ctrl+B D to detach; `tmux attach -t pref_stageb` to resume
#
# Total ETA: ~1-3 h depending on per-step settling (see Stage B doc §6).
# ============================================================

# NOTE: deliberately NO `set -e` — we want failures to advance to next run.
set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_stageb_runs"
LOG_FILE="$LOG_DIR/_summary.log"
mkdir -p "$LOG_DIR"

cd "$REPO"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }

log_top() {
  echo "$@" | tee -a "$LOG_FILE"
}

cleanup_orphans() {
  # Only matches OUR project names (don't touch other users' jobs).
  if pgrep -af "starvla_pref|train_starvla.py|accelerate\.launch|pt_elastic" > /dev/null 2>&1; then
    log_top "  [$(ts)] cleanup: pkill orphan train/accelerate/pt_elastic procs"
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
log_top "# [$(ts)] Stage B contact: sequential main + B0 on H100"
log_top "# host: $(hostname) user: $(whoami)"
log_top "# logs: $LOG_DIR/"
log_top "############################################################"

run_one "01-main"  examples/preference/launch_pref_stage_b_main_contact.sh
run_one "02-b0"    examples/preference/launch_pref_stage_b_b0_contact.sh

log_top "############################################################"
log_top "# [$(ts)] Stage B ALL DONE (check $LOG_FILE for per-run status)"
log_top "############################################################"
log_top ""
log_top "Next: run sanity_pref_flip on main's final ckpt to verify:"
log_top "  python -m examples.preference.stage_b.sanity_pref_flip \\"
log_top "    --policy_yaml examples/preference/train_files/starvla_pref_stage_b_main_contact.yaml \\"
log_top "    --policy_ckpt $REPO/results/Checkpoints/pref_main_stage_b_v1_contact/final_model/pytorch_model.pt \\"
log_top "    --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB"
