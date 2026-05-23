#!/usr/bin/env bash
# ============================================================
# Pref-VLA Stage A baseline (no VQA) — HEIGHT category
#   Data:    /mnt/localssd/$USER/pref/data/height
#   Spec:    25k steps, save every 5k, eff batch 64
#   Run on:  H100 (per user plan 2026-05-23)
#
# Usage:
#   bash examples/preference/launch_pref_stage_a_baseline_height.sh
#   NUM_GPUS=4 bash examples/preference/launch_pref_stage_a_baseline_height.sh
#   MAX_STEPS=200 NO_SAVE=1 bash examples/preference/launch_pref_stage_a_baseline_height.sh   # smoke
# ============================================================
set +u

# Conda activate (host-aware)
if [ -f /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh
elif [ -f /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh
else
  source ~/.bashrc 2>/dev/null
fi
conda activate starVLA
set -eo pipefail

# Storage sanity (see r-preference/doc/training-runbook.md §4)
if ! mountpoint -q /mnt/localssd; then
  echo "FATAL: /mnt/localssd is not mounted." >&2; exit 1
fi
if [ ! -L "$HOME/starVLA/results" ]; then
  echo "FATAL: $HOME/starVLA/results is not a symlink. See training-runbook.md §4." >&2
  exit 1
fi

cd "$HOME/starVLA"

# Host-aware DATA_ROOT
DATA_ROOT=""
for c in /mnt/localssd/kaiwenh/pref/data/height /mnt/localssd/kevin/pref/data/height; do
  [ -d "$c" ] && DATA_ROOT="$c" && break
done
[ -z "$DATA_ROOT" ] && { echo "ERROR: pref/data/height not found on /mnt/localssd/{kaiwenh,kevin}"; exit 1; }

NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -ge 8 ]; then GRAD_ACCUM=1; else GRAD_ACCUM=2; fi
EFF_BATCH=$((8 * NUM_GPUS * GRAD_ACCUM))

# Profile mode overrides
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS+=(--trainer.max_train_steps "${MAX_STEPS}")
  EXTRA_ARGS+=(--trainer.eval_interval "$((MAX_STEPS + 1))")
fi
if [ -n "${NO_SAVE:-}" ]; then
  SAVE_GUARD="${MAX_STEPS:-1000000}"
  EXTRA_ARGS+=(--trainer.save_interval "$((SAVE_GUARD + 1))")
fi

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================"
echo "Run:         pref_baseline_stage_a_v1_noVQA_height"
echo "Node:        $(hostname) ($(whoami))"
echo "DATA_ROOT:   $DATA_ROOT"
echo "NUM_GPUS:    ${NUM_GPUS}  GRAD_ACCUM=${GRAD_ACCUM}  eff_batch=${EFF_BATCH}"
echo "MAX_STEPS:   ${MAX_STEPS:-(yaml 25000)}"
echo "NO_SAVE:     ${NO_SAVE:-(off)}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS[*]:-(none)}"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_baseline_height.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
