#!/usr/bin/env bash
# ============================================================
# DD Dynamic-325: run discrete diffusion on dynamic-324 checkpoint (0325)
#
# Usage:
#   bash realworld/dd-dynamic-325.sh sync
#   bash realworld/dd-dynamic-325.sh async
#   bash realworld/dd-dynamic-325.sh rtc
# ============================================================
set -euo pipefail

# ── Edit these before running ──────────────────────────────
CHECKPOINT="checkpoints/discreteRTC/0325/checkpoints/steps_100000_pytorch_model.pt"
INSTRUCTION="dynamic-324"
CAMERA_DEV=0
ARM="left"
N_ACTIONS=8
INFERENCE_DELAY=8
# ───────────────────────────────────────────────────────────

MODE="${1:-rtc}"

COMMON_ARGS=(
  --checkpoint "$CHECKPOINT"
  --instruction "$INSTRUCTION"
  --camera_dev "$CAMERA_DEV"
  --arm "$ARM"
  --n_actions "$N_ACTIONS"
  --save_rollout
)

case "$MODE" in
  sync)
    python ur5/scripts/run_dd.py --mode sync "${COMMON_ARGS[@]}"
    ;;
  async)
    python ur5/scripts/run_dd.py --mode async "${COMMON_ARGS[@]}"
    ;;
  rtc)
    python ur5/scripts/run_dd.py --mode rtc --inference_delay "$INFERENCE_DELAY" "${COMMON_ARGS[@]}"
    ;;
  *)
    echo "Unknown mode: $MODE (choose: sync, async, rtc)"
    exit 1
    ;;
esac
