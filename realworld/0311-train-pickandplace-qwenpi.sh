#!/bin/bash
# ============================================================
# StarVLA FastUMI pickandplace training -- QwenPI (π₀-style)
#
# Usage:
#   bash realworld/0311-train-pickandplace-qwenpi.sh          # use 4 GPUs (default)
#   bash realworld/0311-train-pickandplace-qwenpi.sh 2         # use 2 GPUs
#
# Prerequisites:
#   - conda env "starVLA" with all dependencies installed
#   - Pretrained model auto-downloaded to playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action/
#   - Dataset at playground/Datasets/FastUMI/${DATASET_NAME}/
# ============================================================

set -euo pipefail

# ── Paths ──
REPO_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen

# ── Dataset config (edit these to switch datasets) ──
# DATASET_NAME : subdirectory under playground/Datasets/FastUMI/
# DATA_MIX     : matching entry in starVLA/dataloader/gr00t_lerobot/mixtures.py
DATASET_NAME=pickandplace-real-0307
DATA_MIX=fastumi_pickandplace_real_0307

# ── HuggingFace / cache (keep everything off home dir) ──
export HF_HOME=${LAB_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export PIP_CACHE_DIR=${LAB_ROOT}/.cache/pip
export WANDB_DIR=${LAB_ROOT}/.cache/wandb
export WANDB_MODE=disabled
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

# ── Conda ──
# Try lab miniforge first, fall back to home miniforge
if [ -f "${LAB_ROOT}/miniforge3/etc/profile.d/conda.sh" ]; then
    source "${LAB_ROOT}/miniforge3/etc/profile.d/conda.sh"
elif [ -f "${HOME}/miniforge3/etc/profile.d/conda.sh" ]; then
    source "${HOME}/miniforge3/etc/profile.d/conda.sh"
else
    echo "[ERROR] Cannot find conda. Install miniforge or set up conda manually."
    exit 1
fi
conda activate starVLA

# ── CUDA module (cluster only; skip if nvcc already available) ──
if ! command -v nvcc &>/dev/null; then
    if command -v module &>/dev/null; then
        module load cuda/12.2.0-fasrc01 2>/dev/null || true
    fi
fi

cd "${REPO_DIR}"

# ── Auto-download pretrained model ──
MODEL_DIR=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Pretrained model not found at ${MODEL_DIR}. Downloading from HuggingFace..."
    huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
    echo "[INFO] Download complete."
fi

# ── GPU count ──
if [ $# -ge 1 ]; then
    NUM_GPUS=$1
else
    NUM_GPUS=4
fi

# Gradient accumulation to keep effective batch ~64
# effective = NUM_GPUS * per_device_batch * grad_accum
PER_DEVICE_BATCH=8
case ${NUM_GPUS} in
    1) GRAD_ACCUM=8 ;;
    2) GRAD_ACCUM=4 ;;
    4) GRAD_ACCUM=2 ;;
    8) GRAD_ACCUM=1 ;;
    *) GRAD_ACCUM=$(( 64 / (NUM_GPUS * PER_DEVICE_BATCH) ))
       [ "${GRAD_ACCUM}" -lt 1 ] && GRAD_ACCUM=1 ;;
esac

# ── Attention implementation ──
# flash_attention_2: works on most local servers (needs pip install flash-attn)
# sdpa: always works (PyTorch built-in), use if flash-attn has GLIBC issues
ATTN_IMPL=flash_attention_2
python3 -c "import flash_attn" 2>/dev/null || ATTN_IMPL=sdpa

# ── Diagnostics ──
echo "============================================"
echo "StarVLA FastUMI pickandplace training (QwenPI)"
echo "============================================"
echo "Repo:      ${REPO_DIR}"
echo "Model:     ${MODEL_DIR}"
echo "Dataset:   playground/Datasets/FastUMI/${DATASET_NAME}"
echo "Data mix:  ${DATA_MIX}"
echo "GPUs:      ${NUM_GPUS}"
echo "Batch:     ${PER_DEVICE_BATCH} x ${NUM_GPUS} x ${GRAD_ACCUM} = $(( PER_DEVICE_BATCH * NUM_GPUS * GRAD_ACCUM )) effective"
echo "Attention: ${ATTN_IMPL}"
echo "Python:    $(which python)"
echo "Torch:     $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:      $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "============================================"

# ── Preflight checks ──
ERRORS=0

if [ ! -d "${MODEL_DIR}" ]; then
    echo "[ERROR] Pretrained model not found: ${MODEL_DIR}/"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "playground/Datasets/FastUMI/${DATASET_NAME}/meta/info.json" ]; then
    echo "[ERROR] Dataset not found: playground/Datasets/FastUMI/${DATASET_NAME}/"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "examples/calvin/train_files/starvla_train_calvin.yaml" ]; then
    echo "[ERROR] Config YAML not found: examples/calvin/train_files/starvla_train_calvin.yaml"
    ERRORS=$((ERRORS + 1))
fi

if [ "${ERRORS}" -gt 0 ]; then
    echo "[ABORT] ${ERRORS} error(s) found. Fix them before training."
    exit 1
fi

# ── NCCL ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Log file ──
LOG_DIR=./results/Checkpoints/fastumi_pickandplace_qwenPI/logs
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to: ${LOG_FILE}"

# ── Training ──
# QwenPI: Qwen2.5-VL-3B + Flow-Matching (DiT-B) action expert (π₀-style).
# Key FastUMI overrides vs CALVIN defaults:
#   action_dim/state_dim: 10 (not 7) -- 3 pos + 6 rot6d + 1 gripper
#   future_action_window_size: 15 (chunk=16, same as QwenOFT)
#   data_root_dir: FastUMI (not CALVIN)
#   data_mix: from DATA_MIX variable
#   video_backend: torchvision_av (h264 video)
#
# Note: repeated_diffusion_steps is NOT set here because QwenPI.py
# hardcodes it to 2 (line 128), ignoring config. The YAML's
# diffusion_model_cfg.num_layers (16) is also overridden at runtime
# to match num_vl_layers (36) -- this is intentional for Layerwise FM.
#
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${NUM_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm ${MODEL_DIR} \
  --framework.qwenvl.attn_implementation ${ATTN_IMPL} \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH} \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 2000 \
  --trainer.gradient_accumulation_steps ${GRAD_ACCUM} \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI \
  --wandb_project starVLA_FastUMI \
  --wandb_entity kaiwenh-17-uiuc \
  2>&1 | tee "${LOG_FILE}"


accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm ${MODEL_DIR} \
  --framework.qwenvl.attn_implementation ${ATTN_IMPL} \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix ${DATA_MIX} \
  --datasets.vla_data.per_device_batch_size ${PER_DEVICE_BATCH} \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI \
  --wandb_project starVLA_FastUMI \
  --wandb_entity 2200011093-peking-university \



accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
 --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix fastumi_pickandplace_real_0307 \
  --datasets.vla_data.include_state true \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI \
  --wandb_project starVLA_FastUMI \
  --wandb_entity 2200011093-peking-university

#   --datasets.vla_data.data_mix fastumi_pickandplace_debug_1ep \