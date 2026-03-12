#!/bin/bash
# RoboTwin 2.0 evaluation: run this in the robotwin conda env.
#
# Prerequisites:
#   1. Start policy server first: bash examples/Robotwin/eval_files/run_eval_robotwin_local.sh
#   2. Set ROBOTWIN_PATH: export ROBOTWIN_PATH=/path/to/RoboTwin
#
# Usage:
#   bash eval.sh <task_name> <task_config> <ckpt_setting> [seed] [gpu_id]
#   bash eval.sh adjust_bottle demo_clean my_test_v1 0 0
#
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/scratch/wangpc/RoboTwin}"

policy_name="model2robotwin_interface"
task_name=${1}
task_config=${2}
ckpt_setting=${3:-starvla_demo}
seed=${4:-0}
gpu_id=${5:-0}

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mGPU: ${gpu_id} | Task: ${task_name} | Config: ${task_config}\033[0m"

EVAL_FILES_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STARVLA_PATH="${STARVLA_PATH:-$(cd "$EVAL_FILES_PATH/../../.." && pwd)}"
DEPLOY_POLICY_PATH=$EVAL_FILES_PATH/deploy_policy.yml

# Default: discrete diffusion steps_30000 (override via POLICY_CKPT_PATH)
POLICY_CKPT_PATH="${POLICY_CKPT_PATH:-$STARVLA_PATH/results/Checkpoints/robotwin_discrete_diffusion_local/checkpoints/steps_30000_pytorch_model.pt}"

export PYTHONPATH=$ROBOTWIN_PATH:$PYTHONPATH
export PYTHONPATH=$STARVLA_PATH:$PYTHONPATH
export PYTHONPATH=$EVAL_FILES_PATH:$PYTHONPATH

if [ ! -d "$ROBOTWIN_PATH" ]; then
  echo "ROBOTWIN_PATH=$ROBOTWIN_PATH does not exist."
  echo "Clone RoboTwin and set: export ROBOTWIN_PATH=/path/to/RoboTwin"
  exit 1
fi
cd "$ROBOTWIN_PATH"

echo "Policy checkpoint: $POLICY_CKPT_PATH"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config "$DEPLOY_POLICY_PATH" \
    --overrides \
    --task_name "${task_name}" \
    --task_config "${task_config}" \
    --ckpt_setting "${ckpt_setting}" \
    --seed "${seed}" \
    --policy_name "${policy_name}" \
    --policy_ckpt_path "${POLICY_CKPT_PATH}"


# bash eval.sh adjust_bottle demo_clean my_test_v1 0 0