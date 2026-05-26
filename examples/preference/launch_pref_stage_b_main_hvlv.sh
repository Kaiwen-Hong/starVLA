#!/usr/bin/env bash
# ============================================================
# Pref-VLA Stage B MAIN — hvlv (stamp_seal6, 100 ep)
#   Warm-start from VQA-Stage-A 35k.
#   Pseudo-labels from r-preference/eval/pref_pseudo_labels_hvlv_B.json
#   (mid_8 + |logit_gap|>=10 filter (per-cat acc to be verified at launch-gate))
#   Action loss only (no VQA loss); framework = plain QwenPI.
#   Spec:    1500 steps, save every 250 → 6 ckpts grid
#
# Usage:
#   bash examples/preference/launch_pref_stage_b_main_contact.sh
#   MAX_STEPS=2 NO_SAVE=1 bash examples/preference/launch_pref_stage_b_main_contact.sh   # smoke
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
for c in /mnt/localssd/kaiwenh/pref/data/hvlv/taskB /mnt/localssd/kevin/pref/data/hvlv/taskB; do
  [ -d "$c" ] && DATA_ROOT="$c" && break
done
[ -z "$DATA_ROOT" ] && { echo "ERROR: pref/data/contact/taskB not found"; exit 1; }

CACHE_PATH="r-preference/eval/pref_pseudo_labels_hvlv_B.json"
[ ! -f "$CACHE_PATH" ] && { echo "FATAL: pseudo-label cache missing: $CACHE_PATH"; exit 1; }

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
echo "Run:           pref_main_stage_b_v1_hvlv"
echo "Node:          $(hostname) ($(whoami))"
echo "DATA_ROOT:     $DATA_ROOT"
echo "PSEUDO CACHE:  $CACHE_PATH"
echo "WARM-START:    VQA-Stage-A 25k (hvlv)"
echo "NUM_GPUS:      ${NUM_GPUS}  GRAD_ACCUM=${GRAD_ACCUM}  eff_batch=${EFF_BATCH}"
echo "MAX_STEPS:     ${MAX_STEPS:-(yaml 1500)}"
echo "NO_SAVE:       ${NO_SAVE:-(off)}"
echo "EXTRA_ARGS:    ${EXTRA_ARGS[*]:-(none)}"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_b_main_hvlv.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
