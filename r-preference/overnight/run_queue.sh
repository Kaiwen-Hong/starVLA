#!/usr/bin/env bash
# ============================================================
# Pref-VLA Stage-A overnight queue runner (host-agnostic, H100 or H200).
# Runs the given jobs SEQUENTIALLY, one full 8-GPU run at a time.
# Each job arg is "<variant>:<cat>" where variant in {token, base}:
#   token -> starvla_pref_stage_a_oftvqa_token_<cat>.yaml  (OFT + token-VQA cotrain)
#   base  -> starvla_pref_stage_a_oft_baseline_<cat>.yaml  (OFT, no VQA)
# Continues to the next job if one fails. Logs + ckpts go to SSD.
#
# Usage:  bash run_queue.sh token:height token:orient base:place ...
# ============================================================
set +u
CONDA_U=""
for u in kaiwenh kevin; do
  if [ -f "/mnt/localssd/$u/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/mnt/localssd/$u/miniconda3/etc/profile.d/conda.sh"; CONDA_U="$u"; break
  fi
done
[ -z "$CONDA_U" ] && { echo "FATAL: no conda found on this host" >&2; exit 1; }
conda activate starVLA
export CUDA_HOME="$CONDA_PREFIX"

disc() { for p in "$@"; do [ -e "$p" ] && { echo "$p"; return 0; }; done; return 1; }
CKPT=$(disc /mnt/localssd/kaiwenh/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt \
            /mnt/localssd/kevin/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt)
VLM=$(disc /mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct \
           /mnt/localssd/kevin/pref/Qwen3-VL-4B-Instruct)
DATA_BASE=$(disc /mnt/localssd/kaiwenh/pref/data/0526 /mnt/localssd/kevin/pref/data/0526)
LOGDIR=$(disc /mnt/localssd/kaiwenh/logs /mnt/localssd/kevin/logs) || LOGDIR="/mnt/localssd/$CONDA_U/logs"
mkdir -p "$LOGDIR"
[ -f "/mnt/localssd/$CONDA_U/env.sh" ] && source "/mnt/localssd/$CONDA_U/env.sh"

cd "$HOME/starVLA" || exit 1
if [ ! -e results ]; then echo "FATAL: $HOME/starVLA/results missing (need SSD symlink)" >&2; exit 1; fi
[ -z "$CKPT" ] && { echo "FATAL: OFT ckpt not found" >&2; exit 1; }
[ -z "$VLM" ] && { echo "FATAL: base VLM not found" >&2; exit 1; }
[ -z "$DATA_BASE" ] && { echo "FATAL: 0526 data not found" >&2; exit 1; }

export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_BLOCKING_WAIT=1 \
       NCCL_ASYNC_ERROR_HANDLING=1 TOKENIZERS_PARALLELISM=false \
       PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_GPUS="${NUM_GPUS:-8}"
QLOG="$LOGDIR/overnight_queue_$(hostname).log"

echo "==== queue start $(date) on $(hostname) ($CONDA_U) ====" | tee -a "$QLOG"
echo "CKPT=$CKPT" | tee -a "$QLOG"
echo "VLM=$VLM  DATA_BASE=$DATA_BASE  GPUS=$NUM_GPUS  jobs: $*" | tee -a "$QLOG"

for job in "$@"; do
  variant="${job%%:*}"; cat="${job##*:}"
  if [ "$variant" = "token" ]; then
    yaml="starvla_pref_stage_a_oftvqa_token_${cat}.yaml"; rid="pref_oftvqa_token_${cat}_10k"
  else
    yaml="starvla_pref_stage_a_oft_baseline_${cat}.yaml"; rid="pref_oft_baseline_${cat}_10k"
  fi
  log="$LOGDIR/overnight_${rid}.log"
  echo "[$(date)] START $rid  ($yaml)" | tee -a "$QLOG"
  if accelerate launch \
        --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes "$NUM_GPUS" \
        starVLA/training/train_starvla.py \
        --config_yaml "examples/preference/train_files/$yaml" \
        --run_id "$rid" \
        --datasets.vla_data.data_root_dir "$DATA_BASE/$cat" \
        --trainer.pretrained_checkpoint "$CKPT" \
        --framework.qwenvl.base_vlm "$VLM" \
        > "$log" 2>&1; then
    echo "[$(date)] DONE  $rid" | tee -a "$QLOG"
  else
    echo "[$(date)] FAILED $rid (rc=$?) — see $log — continuing" | tee -a "$QLOG"
  fi
done
echo "==== QUEUE COMPLETE $(date) on $(hostname) ====" | tee -a "$QLOG"
