#!/bin/bash
# ============================================================
# StarVLA FastUMI pickandplace training -- standalone server
#
# Usage:
#   bash realworld/0306-train-pickandplace.sh          # use 4 GPUs (default)
#   bash realworld/0306-train-pickandplace.sh 2         # use 2 GPUs
#
# Prerequisites:
#   - conda env "starVLA" with all dependencies installed
#   - Pretrained model downloaded to playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/
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
echo "StarVLA FastUMI pickandplace training"
echo "============================================"
echo "Repo:      ${REPO_DIR}"
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

if [ ! -d "playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action" ]; then
    echo "[ERROR] Pretrained model not found: playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "playground/Datasets/FastUMI/${DATASET_NAME}/meta/info.json" ]; then
    echo "[ERROR] Dataset not found: playground/Datasets/FastUMI/${DATASET_NAME}/"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -f "examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml" ]; then
    echo "[ERROR] Config YAML not found: examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml"
    ERRORS=$((ERRORS + 1))
fi

if [ "${ERRORS}" -gt 0 ]; then
    echo "[ABORT] ${ERRORS} error(s) found. Fix them before training."
    exit 1
fi

# ── NCCL ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Training ──
# Key FastUMI overrides vs RoboTwin defaults:
#   action_dim/state_dim: 10 (not 14) -- 3 pos + 6 rot6d + 1 gripper
#   data_root_dir: FastUMI (not RoboTwin)
#   data_mix: from DATA_MIX variable
#   video_backend: torchvision_av (h264 video)
#
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.action_dim 10 \
  --framework.action_model.state_dim 10 \
  --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
  --datasets.vla_data.data_mix fastumi_pickandplace_real_0307 \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.video_backend torchvision_av \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 200 \
  --trainer.gradient_accumulation_steps 1 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id fastumi_pickandplace_qwenPI \
  --wandb_project starVLA_FastUMI \
  --wandb_entity 2200011093-peking-university



accelerate launch \
 --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
   -tion_model.representation -num_processes 8 \
   starVLA/training/train_starvla.py \
    --config_yaml starVLA/config/training/starvla_train_discrete_diffusion.yaml \
    --framework.name QwenDiscretenference_steps 8  \
    --framework.action_model.action_dim 10 \
    --framework.action_model.state_dim 10 \
    --datasDiffusion  \
    --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action  \
    --framework.qwenvl.vl_hidden_dim 4096 \
    --framework.qwenvl.atdata.per_device_batch_sizetn_implementation flash_attention_2 \
    --framework.action_model.representation bin  \
    --framework.action_model.num_bins 256  \
    --framework.action_model.action_low -1logging_frequency 50 \
    --t.0  \
    --framework.action_model.action_high 1.0  \
    --framework.action_model.num_inference_steps 8 \
    --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
    starVLA_Robotwin \
    --wand--datasets.vla_data.data_mix robotwin_task1 \
    --datasets.vla_data.action_type abs_qpos \
    --datasets.vla_data.per_device_batch_size 8 \
    --datasets.vla_data.video_backend torchvision_av  \
    --trainer.freeze_modules '' \
    --trainer.max_train_steps 30000  \
    --trainer.save_interval 10000  \
    --trainer.logging_frequency 50 \
    --trainer.eval_interval 100  \
    --trainer.gradient_accumulation_steps 1 \
    --run_root_dir ./results/Checkpoints \
    --run_id robotwin_discrete_diffusion  \
    --wandb_project starVLA_Robotwin \
    --wandb_entity 2200011093-peking-university







accelerate launch   --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml   --num_processes 8   starVLA/training/train_starvla.py   --config_yaml starVLA/config/training/starvla_train_discrete_diffusion.yaml   --framework.name QwenDiscreteDiffusion   --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action   --framework.qwenvl.attn_implementation flash_attention_2   --framework.action_model.action_dim 10   --framework.action_model.state_dim 10   --datasets.vla_data.data_root_dir playground/Datasets/FastUMI   --datasets.vla_data.data_mix fastumi_pickandplace_real_0307   --datasets.vla_data.per_device_batch_size 8   --datasets.vla_data.video_backend torchvision_av   --trainer.freeze_modules ''   --trainer.max_train_steps 15000   --trainer.save_interval 3000   --tra
iner.logging_frequency 100   --trainer.eval_interval 100   --trainer.gradient_accumulation_steps 1   --run_root_dir ./results/Checkpoints   --r
un_id fastumi_pickandplace_discrete_diffusion   --wandb_project starVLA_FastUMI   --wandb_entity 2200011093-peking-university