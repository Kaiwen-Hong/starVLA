#!/usr/bin/env bash
# ============================================================
# temp-524-retrain-orient-vqa.sh — H200 overnight queue (2026-05-25)
#
# Single-launch queue designed for user to "fire-and-forget":
#   - Recovers orient VQA (failed resume → fresh retrain)
#   - Then queues every Stage B job that's reasonable to try
#   - Finally runs a diagnostic frame_window_test for place (in case mid_8
#     rescues place taskB VQA acc like 0524 contact analysis)
#
# Total ETA: ~14h. Run before sleeping; check in the morning.
#
# Sequence:
#   01-orient-vqa-fresh      Fresh retrain orient VQA 25k from base (~9h)
#                            Replaces failed-resume ckpts (dir was deleted).
#   02-gate-eval-orient      Re-run stage_a_gate.py for orient with NEW
#                            VQA ckpt (~10 min). Result tells whether retrain
#                            actually fixed it.
#   03-pseudolabel-orient    Generate orient taskB pseudo-labels via
#                            pseudo_label_offline.py (~5 min). Launch-gate may
#                            pass or fail — if it fails, Stage B main job
#                            below will abort cleanly via launcher pre-check.
#   04-stageb-orient-main    Stage B main, warm-start NEW orient VQA 25k
#                            (~1.5h, 3000 steps, 6 ckpts + final).
#   05-stageb-orient-b0      Stage B B0, warm-start orient baseline (~1.5h).
#                            No pseudo-label cache needed (B0 doesn't read it).
#   06-stageb-place-b0       Stage B B0 for place (~1.5h). Place VQA + cache
#                            already failed launch-gate; Stage B main for
#                            place is skipped on purpose (would no-op).
#                            B0 works without cache.
#   07-fw-test-place         frame_window_test for place: uniform_8 vs mid_8
#                            vs dense_16 vs mid_dense_16 (~15 min, diagnostic).
#                            If any strategy rescues place taskB acc ≥ 0.9,
#                            you can re-pseudo-label + Stage B place main
#                            tomorrow.
#
# Behavior:
#   - Continue to next job on failure (so a flaky step doesn't waste sleep).
#   - Logs at results/.temp_524_orient_retrain/<tag>.log per job.
#   - Top-level summary at results/.temp_524_orient_retrain/_summary.log.
#
# Usage (in tmux on H200):
#   ssh kevin@34.34.93.23
#   tmux new -s pref_524_orient
#   cd ~/starVLA
#   bash examples/preference/temp-524-retrain-orient-vqa.sh
#   # Ctrl-B D to detach; tmux a -t pref_524_orient to reattach
# ============================================================
set -uo pipefail

REPO="${REPO:-$HOME/starVLA}"
LOG_DIR="$REPO/results/.temp_524_orient_retrain"
LOG_FILE="$LOG_DIR/_summary.log"
mkdir -p "$LOG_DIR"
cd "$REPO"

ts() { date -u +'%Y-%m-%dT%H:%M:%SZ'; }
log_top() { echo "$@" | tee -a "$LOG_FILE"; }

cleanup_orphans() {
  if pgrep -af "train_starvla.py|accelerate\.launch|pt_elastic|pseudo_label_offline|frame_window_test|stage_a_gate" > /dev/null 2>&1; then
    log_top "  [$(ts)] cleanup orphans"
    pkill -9 -f "train_starvla.py"        2>/dev/null || true
    pkill -9 -f "accelerate.launch"       2>/dev/null || true
    pkill -9 -f "pt_elastic"              2>/dev/null || true
    pkill -9 -f "pseudo_label_offline"    2>/dev/null || true
    pkill -9 -f "frame_window_test"       2>/dev/null || true
    pkill -9 -f "stage_a_gate"            2>/dev/null || true
    sleep 30
  fi
}

run_one() {
  local tag="$1"; shift
  local rlog="$LOG_DIR/${tag}.log"
  log_top "============================================================"
  log_top "[$(ts)] BEGIN  $tag"
  log_top "============================================================"
  cleanup_orphans
  ( "$@" ) 2>&1 | tee "$rlog"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -eq 0 ]; then
    log_top "[$(ts)] DONE   $tag"
  else
    log_top "[$(ts)] FAILED $tag (exit=$rc)"
    log_top "  last 10 lines of $rlog:"
    tail -10 "$rlog" 2>/dev/null | sed 's/^/    /' | tee -a "$LOG_FILE"
    log_top "  -> CONTINUING to next run anyway"
  fi
}

# Sanity: pre-existing orient VQA ckpt would auto-resume → defensive abort
ORIENT_VQA_DIR="$REPO/results/Checkpoints/pref_main_stage_a_v1_VQA_orient"
if [ -d "$ORIENT_VQA_DIR/checkpoints" ] && [ -n "$(ls -A $ORIENT_VQA_DIR/checkpoints 2>/dev/null)" ]; then
  log_top "[$(ts)] WARN: $ORIENT_VQA_DIR/checkpoints/ already has ckpts:"
  ls "$ORIENT_VQA_DIR/checkpoints/" | tee -a "$LOG_FILE"
  log_top "  Aborting to avoid accidental resume. Manually 'rm -rf $ORIENT_VQA_DIR' to retry."
  exit 1
fi

source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh
conda activate starVLA

log_top ""
log_top "############################################################"
log_top "# [$(ts)] H200 temp-524 OVERNIGHT QUEUE (~14h)"
log_top "# 7 jobs: orient retrain + post-retrain Stage B + place diagnostic"
log_top "# host: $(hostname) user: $(whoami)"
log_top "############################################################"

# ---------- 01: orient VQA fresh retrain (9h) ----------
# DO NOT pass IS_RESUME — fresh from base
run_one "01-orient-vqa-fresh" \
  bash "examples/preference/launch_pref_stage_a_vqa_orient.sh"

# ---------- 02: re-eval orient gate (verify retrain worked) ----------
run_one "02-gate-eval-orient" \
  python -m examples.preference.eval.stage_a_gate \
    --category orient \
    --baseline_yaml examples/preference/train_files/starvla_pref_stage_a_baseline_orient.yaml \
    --vqa_yaml      examples/preference/train_files/starvla_pref_stage_a_vqa_orient.yaml \
    --baseline_ckpt results/Checkpoints/pref_baseline_stage_a_v1_noVQA_orient/checkpoints/steps_25000_pytorch_model.pt \
    --vqa_ckpt      results/Checkpoints/pref_main_stage_a_v1_VQA_orient/checkpoints/steps_25000_pytorch_model.pt \
    --taskA_n_eps_per_task 10 \
    --taskB_data_root /mnt/localssd/kevin/pref/data/orient/taskB \
    --out r-preference/eval/stage_a_gate_orient_25k_retrained.json

# ---------- 03: pseudo-label orient ----------
run_one "03-pseudolabel-orient" \
  python -m examples.preference.stage_b.pseudo_label_offline \
    --category orient \
    --vqa_yaml examples/preference/train_files/starvla_pref_stage_a_vqa_orient.yaml \
    --vqa_ckpt results/Checkpoints/pref_main_stage_a_v1_VQA_orient/checkpoints/steps_25000_pytorch_model.pt \
    --taskB_data_root /mnt/localssd/kevin/pref/data/orient/taskB \
    --task_groups move_can5_away \
    --filter_threshold 10.0 \
    --out r-preference/eval/pref_pseudo_labels_orient_B.json

# ---------- 04: Stage B orient main ----------
run_one "04-stageb-orient-main" \
  bash "examples/preference/launch_pref_stage_b_main_orient.sh"

# ---------- 05: Stage B orient b0 ----------
run_one "05-stageb-orient-b0" \
  bash "examples/preference/launch_pref_stage_b_b0_orient.sh"

# ---------- 06: Stage B place b0 ----------
run_one "06-stageb-place-b0" \
  bash "examples/preference/launch_pref_stage_b_b0_place.sh"

# ---------- 07: frame_window_test for place (diagnostic) ----------
run_one "07-fw-test-place" \
  python -m examples.preference.eval.frame_window_test \
    --category place \
    --vqa_yaml examples/preference/train_files/starvla_pref_stage_a_vqa_place.yaml \
    --vqa_ckpt results/Checkpoints/pref_main_stage_a_v1_VQA_place/checkpoints/steps_25000_pytorch_model.pt \
    --taskB_data_root /mnt/localssd/kevin/pref/data/place/taskB \
    --strategies "uniform_8,mid_8,dense_16,mid_dense_16" \
    --taskA_n_eps_per_task 10 \
    --out r-preference/eval/frame_window_place_25k.json

log_top "############################################################"
log_top "# [$(ts)] H200 temp-524 ALL DONE"
log_top "# Check $LOG_FILE for per-job pass/fail"
log_top "############################################################"
