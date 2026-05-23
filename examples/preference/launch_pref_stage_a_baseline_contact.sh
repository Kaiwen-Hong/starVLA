#!/usr/bin/env bash
# ============================================================
# Pref-VLA Stage A baseline (no VQA) — CONTACT category
#   Contact = canonical name for legacy "giveobj" (same data, renamed
#   /mnt/.../pref/data/giveobj -> /mnt/.../pref/data/contact; symlink kept).
#   Data:    /mnt/localssd/$USER/pref/data/contact   (16 task × 100 ep)
#   Spec:    50k steps, save every 5k, eff batch 64
# ============================================================
set +u
if [ -f /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh
elif [ -f /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh
else
  source ~/.bashrc 2>/dev/null
fi
conda activate starVLA
set -eo pipefail

if ! mountpoint -q /mnt/localssd; then
  echo "FATAL: /mnt/localssd is not mounted." >&2; exit 1
fi
if [ ! -L "$HOME/starVLA/results" ]; then
  echo "FATAL: $HOME/starVLA/results is not a symlink. See training-runbook.md §4." >&2
  exit 1
fi

cd "$HOME/starVLA"

DATA_ROOT=""
for c in /mnt/localssd/kaiwenh/pref/data/contact /mnt/localssd/kevin/pref/data/contact; do
  [ -d "$c" ] && DATA_ROOT="$c" && break
done
[ -z "$DATA_ROOT" ] && { echo "ERROR: pref/data/contact not found"; exit 1; }

NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -ge 8 ]; then GRAD_ACCUM=1; else GRAD_ACCUM=2; fi
EFF_BATCH=$((8 * NUM_GPUS * GRAD_ACCUM))

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
echo "Run:         pref_baseline_stage_a_v1_noVQA_contact"
echo "Node:        $(hostname) ($(whoami))"
echo "DATA_ROOT:   $DATA_ROOT"
echo "NUM_GPUS:    ${NUM_GPUS}  GRAD_ACCUM=${GRAD_ACCUM}  eff_batch=${EFF_BATCH}"
echo "MAX_STEPS:   ${MAX_STEPS:-(yaml 50000)}"
echo "NO_SAVE:     ${NO_SAVE:-(off)}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS[*]:-(none)}"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_baseline_contact.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
