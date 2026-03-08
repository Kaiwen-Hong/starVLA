#!/usr/bin/env bash
# Local eval: download checkpoint + start policy server (1 GPU, no DeepSpeed)
# Run from repo root: bash examples/Robotwin/eval_files/run_eval_robotwin_local.sh
#
# Override checkpoint via CKPT_DIR. Examples:
#   CKPT_DIR=/scratch/wangpc/starVLA/checkpoints/Qwen3-VL-OFT-Robotwin2
#   CKPT_DIR=/scratch/wangpc/starVLA/checkpoints/prefvla-models/robotwin_qwenOFT_4xH100

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO_ROOT"

# Use conda python (override with STAR_VLA_PYTHON if needed)
star_vla_python="${STAR_VLA_PYTHON:-$(conda info --base 2>/dev/null)/envs/starVLA/bin/python}"
if [ ! -x "$star_vla_python" ]; then
  star_vla_python="$(which python)"
fi

# Use GPU 2 or 4–7 if 0,1,3 have issues; override with GPU_ID=4
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-5694}"

# Checkpoint: override via CKPT_DIR; auto-detect .pt file
CKPT_DIR="${CKPT_DIR:-$REPO_ROOT/checkpoints/Qwen3-VL-OFT-Robotwin2}"
CKPT_PT="$(ls "$CKPT_DIR"/checkpoints/*.pt 2>/dev/null | head -1)"
if [ -z "$CKPT_PT" ] || [ ! -f "$CKPT_PT" ]; then
  CKPT_DIR="$REPO_ROOT/checkpoints/Qwen3-VL-OFT-Robotwin2"
  CKPT_PT="$CKPT_DIR/checkpoints/steps_40000_pytorch_model.pt"
  if [ ! -f "$CKPT_PT" ]; then
    echo "Downloading checkpoint to $CKPT_DIR ..."
    huggingface-cli download StarVLA/Qwen3-VL-OFT-Robotwin2 \
      --local-dir "$CKPT_DIR" \
      --local-dir-use-symlinks False
  fi
fi

# Staging: patch config base_vlm (relative paths like ./playground/... are rejected by HF)
# Prefer local absolute path when model exists; else use HF repo (would trigger download)
BASE_VLM="$(grep 'base_vlm:' "$CKPT_DIR/config.yaml" | sed 's/.*base_vlm:[[:space:]]*//')"
BASE_VLM="${BASE_VLM#./}"
LOCAL_BASE="$REPO_ROOT/${BASE_VLM}"
if [ -d "$LOCAL_BASE" ] && [ -f "$LOCAL_BASE/config.json" ]; then
  PATCHED_BASE="$(realpath "$LOCAL_BASE")"
  echo "Using local base VLM: $PATCHED_BASE"
else
  PATCHED_BASE="Qwen/Qwen3-VL-4B-Instruct"
  echo "Local base VLM not found at $LOCAL_BASE, using HF: $PATCHED_BASE"
fi

STAGING="${STAGING:-$REPO_ROOT/checkpoints/.staging_robotwin}"
rm -rf "$STAGING"
mkdir -p "$STAGING/checkpoints"
ln -sf "$(realpath "$CKPT_PT")" "$STAGING/checkpoints/$(basename "$CKPT_PT")"
ln -sf "$(realpath "$CKPT_DIR/dataset_statistics.json")" "$STAGING/"
cp "$CKPT_DIR/config.yaml" "$STAGING/"
sed -i "s|base_vlm:.*|base_vlm: $PATCHED_BASE|" "$STAGING/config.yaml"
CKPT_PATH="$STAGING/checkpoints/$(basename "$CKPT_PT")"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "Starting policy server on GPU $GPU_ID, port $PORT"
echo "Checkpoint: $CKPT_PATH (from $CKPT_DIR, base_vlm patched to HF)"

${star_vla_python} deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_PATH" \
  --port "$PORT" \
  --use_bf16
