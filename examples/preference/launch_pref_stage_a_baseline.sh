#!/usr/bin/env bash
# ============================================================
# Preference-conditioned VLA — Stage A baseline (no VQA)
#
# This is the "B0 baseline stage 1" / "main-method Stage A ablation control":
#   - data: /mnt/localssd/kaiwenh/pref/data/giveobj (16 task dirs, 1600 demos)
#   - backbone: Qwen3-VL-4B-Instruct base (NOT a StarVLA action ckpt)
#   - prompt: stripped paraphrase + " Preference: low/high contact"
#   - loss:   L_action only (flow-matching MSE). NO VQA cotrain.
#
# Hardware:
#   NUM_GPUS=4  -> grad_accum=2, ~10-20h
#   NUM_GPUS=8  -> grad_accum=1, ~5-10h  (recommended for overnight)
#
# Usage:
#   bash examples/preference/launch_pref_stage_a_baseline.sh
#   NUM_GPUS=8 bash examples/preference/launch_pref_stage_a_baseline.sh
# ============================================================
# conda activate hooks have unbound vars; keep `set -u` OFF until after activate.
set +u

# ----- Storage sanity (H100 path; H200 mirrors this with /mnt/localssd/kevin/...)
# `results/` is a symlink to /mnt/localssd/.../starVLA_runs/results so ckpts land
# on the 5.9T SSD instead of the 194G root. If the SSD didn't mount, writes would
# silently fall back onto root and OOM the disk halfway through training. See
# r-preference/doc/training-runbook.md.
if ! mountpoint -q /mnt/localssd; then
  echo "FATAL: /mnt/localssd is not mounted. Refusing to launch (would fill root disk)." >&2
  exit 1
fi
RESULTS_LINK=/home/kaiwenh/starVLA/results
if [ ! -L "$RESULTS_LINK" ]; then
  echo "FATAL: $RESULTS_LINK is not a symlink. Expected -> /mnt/localssd/kaiwenh/starVLA_runs/results" >&2
  echo "       See r-preference/doc/training-runbook.md §4 to repair." >&2
  exit 1
fi

if [ -f /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh ]; then
  source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh
else
  source ~/.bashrc 2>/dev/null
fi

conda activate starVLA

set -eo pipefail   # don't enable -u; conda activate scripts use unbound vars

# Run from repo root
cd "$(dirname "$0")/../.."

NUM_GPUS="${NUM_GPUS:-4}"
if [ "${NUM_GPUS}" -ge 8 ]; then
  GRAD_ACCUM=1
else
  GRAD_ACCUM=2
fi
EFF_BATCH=$((8 * NUM_GPUS * GRAD_ACCUM))

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export TOKENIZERS_PARALLELISM=false   # silence warning under num_workers > 0
# CRITICAL: expandable_segments gives a 4x step-time speedup on this stack
# (1.17 s/step vs 4.4 s/step on H100). Without it, the allocator hits a
# cudaMalloc slow path frequently on high-rank GPUs (Zero2 sharded state
# fragments). Verified 2026-05-22.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "============================================"
echo "Node:        $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-all}"
echo "NUM_GPUS:    ${NUM_GPUS}"
echo "GRAD_ACCUM:  ${GRAD_ACCUM}  (eff batch = 8 * ${NUM_GPUS} * ${GRAD_ACCUM} = ${EFF_BATCH})"
echo "Python:      $(which python)"
echo "Torch:       $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:  $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "============================================"

# Pre-compute action norm stats if missing (idempotent).
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
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_baseline.yaml \
  --trainer.gradient_accumulation_steps "${GRAD_ACCUM}"
