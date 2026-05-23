#!/usr/bin/env bash
# ============================================================
# Self-contained manual run for Pref-VLA Stage A baseline.
# This is the SAME command Claude was running -- nohup/wrapping
# adds zero overhead, but feel free to verify yourself.
#
# What's inside:
#   1. Show current GPU + process state (so you know if you need to clean)
#   2. (Optional, set CLEAN=1) Kill existing training + watchdog, wait for GPUs
#   3. Set env vars (HF_HOME, NCCL, tokenizers parallelism)
#   4. Source conda + activate starVLA env
#   5. accelerate launch in FOREGROUND (you see tqdm live; Ctrl-C kills cleanly)
#
# Usage:
#   bash examples/preference/manual_run.sh           # just launch (assumes GPUs free)
#   CLEAN=1 bash examples/preference/manual_run.sh   # kill existing first, then launch
#   STEPS=200 bash examples/preference/manual_run.sh # override max_train_steps (smoke / fast bench)
#
# To exit:
#   Ctrl-C   -- SIGINT propagates to accelerate -> deepspeed workers
# ============================================================

# Strict mode AFTER conda activate (conda hooks use unbound vars)
set -e

REPO_ROOT=/home/kaiwenh/starVLA
RUN_DIR=${REPO_ROOT}/results/Checkpoints/pref_baseline_stage_a_v1_noVQA
CLEAN="${CLEAN:-0}"
STEPS_OVERRIDE="${STEPS:-}"
# Default to 29501 instead of accelerate's default 29500 so we don't collide
# with another concurrent training using port 29500.
MAIN_PORT="${MAIN_PORT:-29501}"

cd "$REPO_ROOT"

# ---- 0. Storage sanity (see r-preference/doc/training-runbook.md §4) ----
# results/ is a symlink to SSD. If SSD didn't mount, writes silently fall
# back onto root (194G) and we OOM the disk mid-training. Hard-fail here.
if ! mountpoint -q /mnt/localssd; then
  echo "FATAL: /mnt/localssd is not mounted. Refusing to launch." >&2
  exit 1
fi
if [ ! -L "$REPO_ROOT/results" ]; then
  echo "FATAL: $REPO_ROOT/results is not a symlink." >&2
  echo "       Expected -> /mnt/localssd/kaiwenh/starVLA_runs/results (H100) or /mnt/localssd/kevin/starVLA_runs/results (H200)" >&2
  echo "       See r-preference/doc/training-runbook.md §4 to set up." >&2
  exit 1
fi

# ---- 1. Show current state ----
echo "============================================"
echo "Node:        $(hostname)"
echo "User:        $(whoami)"
echo "CWD:         $(pwd)"
echo "Time(UTC):   $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "Will use:    accelerate launch with deepspeed_zero2.yaml, 8 processes"
echo "============================================"
echo
echo "=== Current GPU state ==="
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
echo
echo "=== Any existing training process? ==="
pgrep -af "train_starvla|starvla_pref|watchdog.sh" 2>&1 | head -10 || echo "(none)"
echo

# ---- 1b. GPU availability check (refuse to launch if GPUs are full and CLEAN!=1) ----
if [ "$CLEAN" != "1" ]; then
  max_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$max_used" -gt 5000 ]; then
    echo "ERROR: a GPU is already using ${max_used} MiB (existing training holds memory)."
    echo "Either:"
    echo "  - rerun with CLEAN=1 to kill existing training first, OR"
    echo "  - wait for the existing run to finish."
    echo
    echo "Existing processes on GPUs:"
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | head -10
    exit 1
  fi
fi

# ---- 2. Optional cleanup ----
if [ "$CLEAN" = "1" ]; then
  echo "CLEAN=1 -> killing existing training + watchdog..."
  pkill -9 -f "starvla_pref_stage_a"   2>&1 || true
  pkill -9 -f "train_starvla"           2>&1 || true
  pkill -9 -f "accelerate.launch"       2>&1 || true
  pkill -9 -f "pt_elastic"              2>&1 || true
  pkill -9 -f "watchdog.sh"             2>&1 || true
  echo "Sent SIGKILL. Waiting 30s for GPU memory release..."
  sleep 30
  echo
  echo "=== GPU state after cleanup ==="
  nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits
  echo
  # Sanity-check GPUs are now free
  worst_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
  if [ "$worst_used" -gt 1000 ]; then
    echo "WARN: max GPU usage is ${worst_used} MiB -- some process still holding memory."
    echo "Inspect: fuser /dev/nvidia0 ; nvidia-smi"
    echo "Continuing anyway. Ctrl-C in 10s to abort."
    sleep 10
  fi
fi

# ---- 3. Env vars ----
export HF_HOME=/mnt/localssd/kaiwenh/cache/hf
export TOKENIZERS_PARALLELISM=false
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
# Helpful for memory fragmentation on long runs:
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# ---- 4. Conda ----
# conda activate hooks use unbound vars; disable strict before sourcing.
set +eu
if [ -f /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh
else
  source ~/.bashrc 2>/dev/null || true
fi
conda activate starVLA
set -e  # don't enable -u; deepspeed hooks have unbound vars

echo "=== Env after activate ==="
echo "Python:       $(which python)"
echo "Torch:        $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:   $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:      $HF_HOME"
echo "PYTORCH_CUDA: $PYTORCH_CUDA_ALLOC_CONF"
echo

# ---- 5. accelerate launch (FOREGROUND) ----
mkdir -p "$RUN_DIR"

# Build optional override args
EXTRA_ARGS=()
if [ -n "$STEPS_OVERRIDE" ]; then
  EXTRA_ARGS+=("--trainer.max_train_steps" "$STEPS_OVERRIDE")
  echo "OVERRIDE: max_train_steps = $STEPS_OVERRIDE (default 50000)"
fi

echo "Launching..."
echo "main_process_port: $MAIN_PORT"
echo "Cmd: accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \\"
echo "       --num_processes 8 --main_process_port $MAIN_PORT \\"
echo "       starVLA/training/train_starvla.py \\"
echo "       --config_yaml examples/preference/train_files/starvla_pref_stage_a_baseline.yaml \\"
echo "       --trainer.gradient_accumulation_steps 1 ${EXTRA_ARGS[*]:-}"
echo

# Press Ctrl-C in next 5s to abort
echo "5s grace period before launch (Ctrl-C to abort)..."
sleep 5

exec accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  --main_process_port "$MAIN_PORT" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_baseline.yaml \
  --trainer.gradient_accumulation_steps 1 \
  "${EXTRA_ARGS[@]}"
