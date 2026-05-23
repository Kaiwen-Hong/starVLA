#!/usr/bin/env bash
# ============================================================
# Preference-conditioned VLA — Stage A MAIN-METHOD (VQA cotrain)
#
# L = L_action + lambda_vqa * L_vqa. Aggregation inside Qwen_PI_VQA.forward.
# Action stream byte-identical to baseline (starvla_pref_stage_a_baseline).
#
# Hardware:
#   NUM_GPUS=8  -> grad_accum=1 (eff batch 64). Recommended on 8x H200 141GB.
#   NUM_GPUS=4  -> grad_accum=2.
#
# Profile mode:
#   MAX_STEPS=100 NO_SAVE=1 NUM_GPUS=8 bash examples/preference/launch_pref_stage_a_vqa.sh
#   -> caps trainer.max_train_steps + bumps eval_interval to inf so per-step
#      timing isn't polluted by eval. Use this BEFORE committing the 50k run
#      (spec gotcha §5: don't eyeball 50k from estimates).
#
# Full run:
#   NUM_GPUS=8 bash examples/preference/launch_pref_stage_a_vqa.sh
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
cd "$(dirname "$0")/../.."

# Host-aware data root (lf=/mnt/localssd/kaiwenh, H200=/mnt/localssd/kevin).
# YAML default is kaiwenh; override here if running as kevin.
DATA_ROOT=""
for c in /mnt/localssd/kaiwenh/pref/data/giveobj /mnt/localssd/kevin/pref/data/giveobj; do
  if [ -d "$c" ]; then
    DATA_ROOT="$c"; break
  fi
done
if [ -z "$DATA_ROOT" ]; then
  echo "ERROR: pref data not found on either /mnt/localssd/kaiwenh or /mnt/localssd/kevin"
  exit 1
fi

NUM_GPUS="${NUM_GPUS:-8}"
if [ "${NUM_GPUS}" -ge 8 ]; then
  GRAD_ACCUM=1
else
  GRAD_ACCUM=2
fi
EFF_BATCH=$((8 * NUM_GPUS * GRAD_ACCUM))

# Profile mode overrides.
EXTRA_ARGS=()
if [ -n "${MAX_STEPS:-}" ]; then
  EXTRA_ARGS+=(--trainer.max_train_steps "${MAX_STEPS}")
  # Push eval_interval beyond MAX_STEPS so it never fires (clean per-step
  # timing). Same for save_interval if NO_SAVE.
  EXTRA_ARGS+=(--trainer.eval_interval "$((MAX_STEPS + 1))")
fi
if [ -n "${NO_SAVE:-}" ]; then
  # Use save_interval > MAX_STEPS or > max_train_steps default to suppress.
  SAVE_GUARD="${MAX_STEPS:-1000000}"
  EXTRA_ARGS+=(--trainer.save_interval "$((SAVE_GUARD + 1))")
fi

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false

echo "============================================"
echo "Node:        $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "NUM_GPUS:    ${NUM_GPUS}"
echo "GRAD_ACCUM:  ${GRAD_ACCUM}  (eff batch = 8 * ${NUM_GPUS} * ${GRAD_ACCUM} = ${EFF_BATCH})"
echo "MAX_STEPS:   ${MAX_STEPS:-(yaml default 50000)}"
echo "NO_SAVE:     ${NO_SAVE:-(off)}"
echo "EXTRA_ARGS:  ${EXTRA_ARGS[*]:-(none)}"
echo "Python:      $(which python)"
echo "Torch:       $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# Same stats JSON as baseline -- DO NOT regenerate (would shift the split).
STATS_JSON=examples/preference/dataset/stats_giveobj_v1.json
if [ ! -f "${STATS_JSON}" ]; then
  echo "[setup] Stats JSON missing, pre-computing..."
  python -m examples.preference.dataset.precompute_stats \
    --data_root /mnt/localssd/kaiwenh/pref/data/giveobj \
    --out "${STATS_JSON}"
fi

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${NUM_GPUS}" \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_vqa.yaml \
  --datasets.vla_data.data_root_dir "${DATA_ROOT}" \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${EXTRA_ARGS[@]}"
