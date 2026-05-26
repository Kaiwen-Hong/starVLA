#!/usr/bin/env bash
# ============================================================
# temp-523-h200.sh — H200 followup batch.
#
# Context: 2026-05-24. After killing the previous H200 orient-VQA run
# (`pref_main_stage_a_v1_VQA_orient`) at step ~13480 to free GPUs for the
# missing-VQA backlog. Latest saved ckpt for orient VQA = step 10000 (will
# resume from there; 10k → 13480 in-memory progress is gone).
#
# Why ALL VQA on H200: H100 80GB cannot fit VQA at bs=8 — verified twice
# (height VQA OOM 04:34 UTC + place VQA smoke OOM 08:11 UTC, both rank
# 79.14/79.18 GB). H200 141GB has ~50 GB headroom per GPU.
#
# Runs in order:
#   01-orient-vqa-resume   IS_RESUME=1, picks up from steps_10000.
#                          Trains to 25000 (~5h 30m at 1.3 s/step).
#                          Adam momentum resets — negligible per
#                          ~100-step re-accumulation (per training-runbook §8).
#   02-height-vqa-fresh    Fresh from Qwen3-VL-4B base (the H100 attempt
#                          at 04:34 UTC produced 0 ckpts before OOM).
#                          25000 steps, ~9h, eff_batch=64.
#   03-place-vqa-fresh     New `place` category VQA cotrain.
#                          25000 steps, ~9h. Requires:
#                            - examples/preference/train_files/starvla_pref_stage_a_vqa_place.yaml
#                            - examples/preference/launch_pref_stage_a_vqa_place.sh
#                            - examples/preference/dataset/stats_place_v1.json
#                          All 3 scp'd to H200 on 2026-05-24.
#
# Total wall estimate: ~23h 30m (5h 30m + 9h + 9h).
#
# Behavior: same as temp-h{100,200}.sh — continue to next run on failure;
# cleanup orphan procs between runs; per-run + summary log.
#
# Usage (in tmux on H200):
#   tmux new -s pref_523_h200
#   bash examples/preference/temp-523-h200.sh
# ============================================================

set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_523_h200_runs"
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
log_top "# [$(ts)] H200 temp-523: 3 runs (orient VQA resume + height VQA + place VQA)"
log_top "# host: $(hostname) user: $(whoami)"
log_top "# logs: $LOG_DIR/"
log_top "############################################################"

run_one "01-orient-vqa-resume" examples/preference/launch_pref_stage_a_vqa_orient.sh IS_RESUME=1
run_one "02-height-vqa-fresh"  examples/preference/launch_pref_stage_a_vqa_height.sh
run_one "03-place-vqa-fresh"   examples/preference/launch_pref_stage_a_vqa_place.sh

log_top "############################################################"
log_top "# [$(ts)] H200 temp-523 ALL DONE (check $LOG_FILE)"
log_top "############################################################"
