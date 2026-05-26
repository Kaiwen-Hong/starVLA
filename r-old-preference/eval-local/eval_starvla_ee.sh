#!/bin/bash
# Evaluate one StarVLA EE checkpoint on one RoboTwin task config.
# Analogous to ar-research-kempner/policy/pi05/eval_ee.sh.
#
# Usage:
#   bash eval_starvla_ee.sh <task_name> <task_config> <deploy_yml> <seed> <gpu_id> <exp_idx>
#
# Args:
#   task_name    - RoboTwin task (e.g. place_object_stand)
#   task_config  - Environment variant (e.g. place_object_stand_wp4)
#   deploy_yml   - Path to deploy_policy_ee_v{30,31}.yml
#   seed         - Random seed (e.g. 0)
#   gpu_id       - GPU for the sim (not the server; server manages its own GPU)
#   exp_idx      - Experiment index for result directory
#
# Environment variables:
#   AR_ROOT       -- path to ar-research-kempner (default: ~/Desktop/research/ar-research-kempner)
#   STARVLA_ROOT  -- path to starVLA repo (default: ~/Desktop/research/starVLA)

set -euo pipefail

AR_ROOT="${AR_ROOT:-${HOME}/Desktop/research/ar-research-kempner}"
STARVLA_ROOT="${STARVLA_ROOT:-${HOME}/Desktop/research/starVLA}"

task_name="${1:?Usage: eval_starvla_ee.sh <task_name> <task_config> <deploy_yml> <seed> <gpu_id> <exp_idx>}"
task_config="${2:?}"
deploy_yml="${3:?}"
seed="${4:?}"
gpu_id="${5:?}"
exp_idx="${6:?}"

if [ ! -f "$deploy_yml" ]; then
    echo "ERROR: deploy yml not found: $deploy_yml"
    exit 1
fi

echo -e "\033[33mgpu id (sim): ${gpu_id}\033[0m"

export CUDA_VISIBLE_DEVICES="${gpu_id}"
# STARVLA_ROOT is needed by starVLA_ee policy module to locate websocket client
export STARVLA_ROOT
export PYTHONPATH="${STARVLA_ROOT}:${PYTHONPATH:-}"

cd "$AR_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py \
    --config "$deploy_yml" \
    --overrides \
    --task_name  "$task_name" \
    --task_config "$task_config" \
    --seed       "$seed" \
    --exp_idx    "$exp_idx"
