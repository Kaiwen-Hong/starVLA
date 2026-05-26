#!/usr/bin/env bash
# ============================================================
# Step 1: Download v61, v62 checkpoints from HF
#
# Usage:
#   bash r-preference/eval-local-v6162-vast/1_download_checkpoints.sh
# ============================================================
set -euo pipefail

STARVLA_ROOT="${STARVLA_ROOT:-/home/user/starVLA}"
HF_REPO="kaiwen2/prefvla-models"
STEP="${STEP:-60000}"

declare -A MODELS=(
    ["v0320_v61_qwenOFT_finetune_v2"]="steps_${STEP}_pytorch_model.pt"
    ["v0320_v62_qwenOFT_finetune_v2"]="steps_${STEP}_pytorch_model.pt"
)

RESULTS="${STARVLA_ROOT}/results/Checkpoints"
EXTRA_FILES=(config.yaml dataset_statistics.json summary.jsonl)

echo "========================================"
echo "Downloading checkpoints from ${HF_REPO}"
echo "========================================"

for model_dir in "${!MODELS[@]}"; do
    ckpt_name="${MODELS[$model_dir]}"
    local_dir="${RESULTS}/${model_dir}"
    hf_prefix="starvla/${model_dir}"

    echo ""
    echo "--- ${model_dir} ---"

    # Check if already downloaded
    if [ -f "${local_dir}/checkpoints/${ckpt_name}" ]; then
        size=$(du -h "${local_dir}/checkpoints/${ckpt_name}" | cut -f1)
        echo "  Already exists (${size}), skipping."
        continue
    fi

    mkdir -p "${local_dir}/checkpoints"

    # Download checkpoint
    echo "  Downloading checkpoint..."
    python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${hf_prefix}/checkpoints/${ckpt_name}',
    local_dir='${RESULTS}',
)
print('  checkpoint downloaded.')
"

    # Move file if needed (hf_hub_download may nest under starvla/)
    if [ -f "${RESULTS}/${hf_prefix}/checkpoints/${ckpt_name}" ] && \
       [ ! -f "${local_dir}/checkpoints/${ckpt_name}" ]; then
        mv "${RESULTS}/${hf_prefix}/checkpoints/${ckpt_name}" "${local_dir}/checkpoints/${ckpt_name}"
    fi

    # Download extra files
    for f in "${EXTRA_FILES[@]}"; do
        echo "  Downloading ${f}..."
        python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='${HF_REPO}',
    filename='${hf_prefix}/${f}',
    local_dir='${RESULTS}',
)
" 2>/dev/null || echo "  WARNING: ${f} not found on HF."
        # Move if needed
        if [ -f "${RESULTS}/${hf_prefix}/${f}" ] && [ ! -f "${local_dir}/${f}" ]; then
            mv "${RESULTS}/${hf_prefix}/${f}" "${local_dir}/${f}"
        fi
    done

    echo "  DONE: ${model_dir}"
done

# Cleanup starvla/ prefix directory if empty
rm -rf "${RESULTS}/starvla" 2>/dev/null || true

echo ""
echo "========================================"
echo "Download complete."
echo "========================================"

# Verify
echo ""
echo "Verification:"
for model_dir in "${!MODELS[@]}"; do
    ckpt_name="${MODELS[$model_dir]}"
    ckpt="${RESULTS}/${model_dir}/checkpoints/${ckpt_name}"
    if [ -f "$ckpt" ]; then
        size=$(du -h "$ckpt" | cut -f1)
        echo "  OK: ${model_dir} (${size})"
    else
        echo "  MISSING: ${model_dir}"
    fi
done
