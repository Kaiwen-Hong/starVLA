#!/usr/bin/env bash
# ============================================================
# Step 1: Download v61/v62 checkpoints from HuggingFace
#
# Run this on your 4090 machine.
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (enter your HF token)
#
# Usage:
#   bash r-preference/eval-local-v6162/1_download_checkpoints.sh
# ============================================================
set -euo pipefail

STARVLA_ROOT="${STARVLA_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HF_REPO="kaiwen2/prefvla-models"
CKPT_BASE="${STARVLA_ROOT}/results/Checkpoints"

MODELS=(
    "v0320_v61_qwenOFT_finetune_v2"
    "v0320_v62_qwenOFT_finetune_v2"
)

# Files to download per model
FILES=(
    "checkpoints/steps_60000_pytorch_model.pt"
    "config.yaml"
    "dataset_statistics.json"
    "summary.jsonl"
)

echo "============================================"
echo "Downloading checkpoints from ${HF_REPO}"
echo "Destination: ${CKPT_BASE}"
echo "============================================"

# Check HF login
if ! huggingface-cli whoami &>/dev/null && ! hf auth whoami &>/dev/null; then
    echo "ERROR: Not logged in to HuggingFace."
    echo "Run: huggingface-cli login"
    exit 1
fi

for model in "${MODELS[@]}"; do
    echo ""
    echo "--- ${model} ---"
    ckpt_file="${CKPT_BASE}/${model}/checkpoints/steps_60000_pytorch_model.pt"
    if [ -f "$ckpt_file" ]; then
        echo "  Already exists, skipping."
        continue
    fi

    mkdir -p "${CKPT_BASE}/${model}/checkpoints"

    for file in "${FILES[@]}"; do
        hf_path="starvla/${model}/${file}"
        local_path="${CKPT_BASE}/${model}/${file}"

        if [ -f "$local_path" ]; then
            echo "  Already have: ${file}"
            continue
        fi

        echo "  Downloading: ${file}..."
        huggingface-cli download "$HF_REPO" "$hf_path" \
            --local-dir "/tmp/hf_download_tmp"
        mkdir -p "$(dirname "$local_path")"
        mv "/tmp/hf_download_tmp/${hf_path}" "$local_path"
        echo "  Done: ${file}"
    done
done

# Cleanup temp dir
rm -rf /tmp/hf_download_tmp

echo ""
echo "============================================"
echo "Download complete. Verifying..."
echo "============================================"

ALL_OK=true
for model in "${MODELS[@]}"; do
    ckpt="${CKPT_BASE}/${model}/checkpoints/steps_60000_pytorch_model.pt"
    if [ -f "$ckpt" ]; then
        size=$(du -h "$ckpt" | cut -f1)
        echo "  OK: ${model} (${size})"
    else
        echo "  MISSING: ${model}"
        ALL_OK=false
    fi
done

if $ALL_OK; then
    echo ""
    echo "All checkpoints ready!"
    echo "Next step: bash r-preference/eval-local-v6162/2_run_policy_server.sh v62"
else
    echo ""
    echo "Some checkpoints are missing. Check the errors above."
    exit 1
fi
