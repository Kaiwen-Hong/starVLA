#!/usr/bin/env bash
# ============================================================
# temp-523-h100.sh — H100 followup batch (Stage B contact + place baseline).
#
# Context: 2026-05-24. After the original temp-h100.sh + temp-h100-continue.sh
# completed (height baseline + hvlv baseline + height VQA{OOM-failed}), the
# only Stage A baseline still missing across the 4 + 1 = 5 categories is
# the new `place` category (registered 2026-05-24 — see
# r-preference/doc/0524-place-category.md).
#
# Stage B contact (added 2026-05-24): pseudo-label cache built off
# 35k VQA-Stage-A; main + B0 continue-train on contact taskB (100 ep).
# Smoke (MAX_STEPS=2 NO_SAVE=1) passed cleanly on both 2026-05-24.
# Pseudo-label launch gate also passed (acc 1.000, 92/100 kept).
# See r-preference/doc/0524-stageB-contact-plan.md.
#
# Order (per user 2026-05-24): Stage B FIRST (prioritize results),
# THEN place baseline.
#
# All remaining VQA training (height VQA redo + place VQA fresh +
# orient VQA resume) is queued on H200 in temp-523-h200.sh because
# H100 80GB cannot fit VQA at bs=8 (verified twice: prod height VQA OOM
# 04:34 UTC + smoke place VQA OOM 08:11 UTC — both 79.14/79.18 GB rank OOM).
#
# Runs in order:
#   01-stageb-main-contact  Stage B main, warm-start VQA 35k, 1500 step ~1.5h
#   02-stageb-b0-contact    Stage B B0,   warm-start no-VQA 35k, 1500 step ~1.5h
#   03-place-baseline       Fresh place Stage A baseline, 25k steps, ~8h, 5 ckpts
#
# Behavior:
#   - cleanup_orphans + run_one pattern from temp-h100.sh / -continue.sh.
#   - Continues to next run on failure (single-job here so moot).
#   - Per-run logs at results/.temp_523_h100_runs/<tag>.log
#   - Top-level summary at results/.temp_523_h100_runs/_summary.log
#
# Usage (in tmux):
#   tmux new -s pref_523_h100
#   bash examples/preference/temp-523-h100.sh
# ============================================================

set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_523_h100_runs"
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
log_top "# [$(ts)] H100 temp-523: 3 runs (Stage B contact main + B0, then place baseline)"
log_top "# host: $(hostname) user: $(whoami)"
log_top "# logs: $LOG_DIR/"
log_top "############################################################"

run_one "01-stageb-main-contact" examples/preference/launch_pref_stage_b_main_contact.sh
run_one "02-stageb-b0-contact"   examples/preference/launch_pref_stage_b_b0_contact.sh
run_one "03-place-baseline"      examples/preference/launch_pref_stage_a_baseline_place.sh

log_top "############################################################"
log_top "# [$(ts)] H100 temp-523 ALL DONE (check $LOG_FILE)"
log_top "############################################################"
log_top ""
log_top "After Stage B main finishes (~step 1500 final ckpt), run sanity (§7.5):"
log_top "  python -m examples.preference.stage_b.sanity_pref_flip \\"
log_top "    --policy_yaml examples/preference/train_files/starvla_pref_stage_b_main_contact.yaml \\"
log_top "    --policy_ckpt \$REPO/results/Checkpoints/pref_main_stage_b_v1_contact/final_model/pytorch_model.pt \\"
log_top "    --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \\"
log_top "    --out r-preference/eval/stageb_sanity_main_contact.json"
