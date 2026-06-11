#!/usr/bin/env bash
# Stage-B queue runner (host-agnostic). Jobs are "<mode>:<cat>" where mode in {main,b0}.
#   main -> starvla_pref_stageb_main_<cat>.yaml (warm-start token ckpt + pref suffix + cache)
#   b0   -> starvla_pref_stageb_b0_<cat>.yaml   (warm-start baseline ckpt, no suffix)
# pretrained_checkpoint in the YAML is a repo-relative results/ path → resolves on the
# machine that holds that Stage-A ckpt. Run main where the token ckpt is, b0 where the
# baseline ckpt is. ~1500 steps, L_action only.
set +u
CU=""
for u in kaiwenh kevin; do
  [ -f "/mnt/localssd/$u/miniconda3/etc/profile.d/conda.sh" ] && { source "/mnt/localssd/$u/miniconda3/etc/profile.d/conda.sh"; CU=$u; break; }
done
[ -z "$CU" ] && { echo "FATAL no conda" >&2; exit 1; }
conda activate starVLA; export CUDA_HOME="$CONDA_PREFIX"
disc(){ for p in "$@"; do [ -e "$p" ] && { echo "$p"; return; }; done; }
VLM=$(disc /mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct /mnt/localssd/kevin/pref/Qwen3-VL-4B-Instruct)
LOGDIR=$(disc /mnt/localssd/kaiwenh/logs /mnt/localssd/kevin/logs) || LOGDIR="/mnt/localssd/$CU/logs"; mkdir -p "$LOGDIR"
[ -f "/mnt/localssd/$CU/env.sh" ] && source "/mnt/localssd/$CU/env.sh"
cd "$HOME/starVLA" || exit 1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 NCCL_BLOCKING_WAIT=1 \
       TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
NUM_GPUS="${NUM_GPUS:-8}"; QLOG="$LOGDIR/stageb_queue_$(hostname).log"
echo "==== stageb queue $(date) $(hostname) ($CU) VLM=$VLM jobs: $* ====" | tee -a "$QLOG"
for job in "$@"; do
  mode="${job%%:*}"; cat="${job##*:}"
  yaml="starvla_pref_stageb_${mode}_${cat}.yaml"; rid="pref_stageb_${mode}_${cat}"
  log="$LOGDIR/stageb_${rid}.log"
  echo "[$(date)] START $rid ($yaml)" | tee -a "$QLOG"
  if accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
        --num_processes "$NUM_GPUS" starVLA/training/train_starvla.py \
        --config_yaml "examples/preference/train_files/$yaml" \
        --run_id "$rid" --framework.qwenvl.base_vlm "$VLM" > "$log" 2>&1; then
    echo "[$(date)] DONE $rid" | tee -a "$QLOG"
  else
    echo "[$(date)] FAILED $rid (rc=$?) — see $log" | tee -a "$QLOG"
  fi
done
echo "==== STAGEB QUEUE COMPLETE $(date) $(hostname) ====" | tee -a "$QLOG"
