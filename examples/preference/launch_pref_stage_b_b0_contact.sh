#!/usr/bin/env bash
# ============================================================
# Pref-VLA Stage B B0 (baseline) — contact (put_boxdrink3_plate, 100 ep)
#   Warm-start from NO-VQA Stage A 35k (only ckpt available).
#   No pseudo-label cache (B0 doesn't read it); NO "Preference: ..." suffix.
#   Action loss only; framework = plain QwenPI.
#   Same hyperparams as main (1500 steps, save every 250) for clean
#   comparison against main.
#
# Usage:
#   bash examples/preference/launch_pref_stage_b_b0_contact.sh
#   MAX_STEPS=2 NO_SAVE=1 bash examples/preference/launch_pref_stage_b_b0_contact.sh   # smoke
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
  echo "FATAL: $HOME/starVLA/results is not a symlink." >&2; exit 1
fi

cd "$HOME/starVLA"

DATA_ROOT=""
for c in /mnt/localssd/kaiwenh/pref/data/contact/taskB /mnt/localssd/kevin/pref/data/contact/taskB; do
  [ -d "$c" ] && DATA_ROOT="$c" && break
done
[ -z "$DATA_ROOT" ] && { echo "ERROR: pref/data/contact/taskB not found"; exit 1; }

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
echo "Run:           pref_b0_stage_b_v1_contact"
echo "Node:          $(hostname) ($(whoami))"
echo "DATA_ROOT:     $DATA_ROOT"
echo "CACHE:         (B0 — no cache)"
echo "WARM-START:    no-VQA Stage A 35k (contact baseline)"
echo "NUM_GPUS:      ${NUM_GPUS}  GRAD_ACCUM=${GRAD_ACCUM}  eff_batch=${EFF_BATCH}"
echo "MAX_STEPS:     ${MAX_STEPS:-(yaml 1500)}"
echo "NO_SAVE:       ${NO_SAVE:-(off)}"
echo "EXTRA_ARGS:    ${EXTRA_ARGS[*]:-(none)}"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_b_b0_contact.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
