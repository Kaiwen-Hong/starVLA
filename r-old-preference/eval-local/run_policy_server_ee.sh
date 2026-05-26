#!/bin/bash
# Start the StarVLA EE policy server on Desktop.
#
# Runs the WebSocket policy server for StarVLA EE-space inference.
# The eval script connects to this server via localhost:5695.
#
# Usage:
#   bash run_policy_server_ee.sh <ckpt_path>
#
# Example:
#   bash run_policy_server_ee.sh \
#     /home/kaiwen/Desktop/research/starVLA/results/Checkpoints/v0309_v31_ee_qwenPI_requeue/checkpoints/steps_50000_pytorch_model.pt
#
# Environment variables (override as needed):
#   STARVLA_ROOT  -- path to starVLA repo (default: ~/Desktop/research/starVLA)
#   STARVLA_PYTHON -- Python binary (default: ~/miniconda3/envs/starVLA/bin/python)
#   PORT          -- WebSocket port (default: 5695)
#   GPU_ID        -- GPU device to use (default: 0)

set -euo pipefail

STARVLA_ROOT="${STARVLA_ROOT:-${HOME}/Desktop/research/starVLA}"
STARVLA_PYTHON="${STARVLA_PYTHON:-${HOME}/miniconda3/envs/starVLA/bin/python}"
PORT="${PORT:-5695}"
GPU_ID="${GPU_ID:-0}"

if [ $# -lt 1 ]; then
    echo "Usage: bash run_policy_server_ee.sh <ckpt_path>"
    echo ""
    echo "Example:"
    echo "  bash run_policy_server_ee.sh \\"
    echo "    ~/Desktop/research/starVLA/results/Checkpoints/v0309_v31_ee_qwenPI_requeue/checkpoints/steps_50000_pytorch_model.pt"
    exit 1
fi

CKPT_PATH="$1"

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

if [ ! -f "$STARVLA_PYTHON" ]; then
    echo "ERROR: Python not found: $STARVLA_PYTHON"
    echo "Set STARVLA_PYTHON env var to the correct path."
    exit 1
fi

echo "============================================"
echo "StarVLA EE Policy Server"
echo "  STARVLA_ROOT: $STARVLA_ROOT"
echo "  Python:       $STARVLA_PYTHON"
echo "  Checkpoint:   $CKPT_PATH"
echo "  Port:         $PORT"
echo "  GPU:          $GPU_ID"
echo "============================================"

export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"
export STARVLA_ROOT

cd "$STARVLA_ROOT"

CUDA_VISIBLE_DEVICES=$GPU_ID \
    "$STARVLA_PYTHON" deployment/model_server/server_policy.py \
    --ckpt_path "$CKPT_PATH" \
    --port "$PORT" \
    --use_bf16
