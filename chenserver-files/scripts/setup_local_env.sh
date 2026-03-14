#!/usr/bin/env bash
# Source this to set up the running env for starVLA (training / policy server).
# Usage: source scripts/setup_local_env.sh   OR   . scripts/setup_local_env.sh

export REPO_ROOT=/scratch/wangpc/starVLA
export HF_HOME="${REPO_ROOT}/.cache/huggingface"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export PIP_CACHE_DIR="${REPO_ROOT}/.cache/pip"
export WANDB_DIR="${REPO_ROOT}/.cache/wandb"
export WANDB_CACHE_DIR="${REPO_ROOT}/.cache/wandb"
export TRITON_CACHE_DIR="/tmp/triton_cache_${USER:-nobody}"
export TOKENIZERS_PARALLELISM=false

mkdir -p "$HF_HOME/hub" "$PIP_CACHE_DIR" "$WANDB_DIR" "$TRITON_CACHE_DIR" 2>/dev/null || true

# Use repo in Python path when running from REPO_ROOT
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

echo "REPO_ROOT=$REPO_ROOT"
echo "Activate conda: conda activate starVLA"
