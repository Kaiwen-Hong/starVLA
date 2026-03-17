#!/bin/bash
# ============================================================
# Open-Loop Evaluation: QwenPI pick-and-place (FastUMI)
#
# Usage:
#   bash realworld/0312-evaluate-fastumi-pickandplace-qwenpi.sh          # 200 samples (default)
#   bash realworld/0312-evaluate-fastumi-pickandplace-qwenpi.sh 500      # 500 samples
#   bash realworld/0312-evaluate-fastumi-pickandplace-qwenpi.sh 0        # all samples
#
# Prerequisites:
#   - conda env "starVLA" with all dependencies
#   - Checkpoint downloaded via: python realworld/download_checkpoint_pickandplace_ur5_0314.py
#   - Dataset downloaded via:    python realworld/download_pickandplace_ur5_0314.py
# ============================================================

set -euo pipefail

# ── Paths (auto-detect from script location) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"

# ── Checkpoint ──
CHECKPOINT=checkpoints/DiscreteRTC/fastumi_pickandplace_qwenPI/checkpoints/steps_15000_pytorch_model.pt
NUM_SAMPLES=${1:-200}

# ── HuggingFace / cache ──
if [ -n "${LAB_ROOT:-}" ]; then
    export HF_HOME=${LAB_ROOT}/.cache/huggingface
    export HF_HUB_CACHE=${HF_HOME}/hub
    export TRANSFORMERS_CACHE=${HF_HOME}/transformers
    export HF_DATASETS_CACHE=${HF_HOME}/datasets
fi
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

# ── Conda ──
for conda_sh in \
    "${LAB_ROOT:-/nonexistent}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/miniforge3/etc/profile.d/conda.sh" \
    "${HOME}/anaconda3/etc/profile.d/conda.sh" \
    "${HOME}/miniconda3/etc/profile.d/conda.sh"; do
    if [ -f "${conda_sh}" ]; then
        source "${conda_sh}"
        break
    fi
done
conda activate starVLA

# ── CUDA (cluster only; skip if nvcc already available) ──
if ! command -v nvcc &>/dev/null; then
    if command -v module &>/dev/null; then
        module load cuda/12.2.0-fasrc01 2>/dev/null || true
    fi
fi

cd "${REPO_DIR}"

# ── Auto-download pretrained VLM (required by from_pretrained) ──
MODEL_DIR=playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action
if [ ! -d "${MODEL_DIR}" ] || [ -z "$(ls -A "${MODEL_DIR}" 2>/dev/null)" ]; then
    echo "[INFO] Pretrained VLM not found at ${MODEL_DIR}. Downloading..."
    huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
        --local-dir "${MODEL_DIR}" \
        --local-dir-use-symlinks False
    echo "[INFO] Download complete."
fi

# ── Preflight checks ──
ERRORS=0

if [ ! -f "${CHECKPOINT}" ]; then
    echo "[ERROR] Checkpoint not found: ${CHECKPOINT}"
    echo "        Run: python realworld/download_checkpoint_pickandplace_ur5_0314.py"
    ERRORS=$((ERRORS + 1))
fi

CKPT_DIR=$(dirname "$(dirname "${CHECKPOINT}")")
if [ ! -f "${CKPT_DIR}/config.yaml" ]; then
    echo "[ERROR] config.yaml not found in ${CKPT_DIR}/"
    ERRORS=$((ERRORS + 1))
fi
if [ ! -f "${CKPT_DIR}/dataset_statistics.json" ]; then
    echo "[ERROR] dataset_statistics.json not found in ${CKPT_DIR}/"
    ERRORS=$((ERRORS + 1))
fi

if [ ! -d "playground/Datasets/FastUMI/pickandplace-ur5-0314" ]; then
    echo "[ERROR] Dataset not found: playground/Datasets/FastUMI/pickandplace-ur5-0314/"
    echo "        Run: python realworld/download_pickandplace_ur5_0314.py"
    ERRORS=$((ERRORS + 1))
fi

if [ "${ERRORS}" -gt 0 ]; then
    echo "[ABORT] ${ERRORS} error(s). Fix them before evaluation."
    exit 1
fi

# ── Diagnostics ──
echo "============================================"
echo "Open-Loop Eval: QwenPI pick-and-place"
echo "============================================"
echo "Repo:       ${REPO_DIR}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Samples:    ${NUM_SAMPLES}"
echo "Python:     $(which python)"
echo "Torch:      $(python3 -c 'import torch; print(torch.__version__)')"
echo "CUDA:       $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "GPU:        $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "============================================"

# ── Run evaluation ──
python realworld/eval_openloop_qwenpi.py \
    --checkpoint "${CHECKPOINT}" \
    --num_samples "${NUM_SAMPLES}" \
    --include_state
