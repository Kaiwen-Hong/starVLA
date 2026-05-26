#!/usr/bin/env bash
# ============================================================
# v40 OFT finetune 训练脚本（交互式 GPU 节点直接运行）
#
# 从 Qwen3-VL-OFT-Robotwin2 checkpoint 微调
# 框架: QwenOFT (L1 MLP regression), VLM: Qwen3-VL-4B-Instruct
# 动作空间: 14D 关节空间 (非 EE)
#
# 前提：你已经在一个有 4xH100/H200 的计算节点上，比如通过：
#   salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
#     -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-07:00:00
#
# 数据：place_cup_tray (clean1, wp5) + place_stapler_stand (clean1, wp5)
#       4 variants, 250 episodes each, 1000 total
#
# 用法：
#   bash r-preference/0320-v40-training-finetune-version1.sh
# ============================================================
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

echo "============================================"
echo "Node:      $(hostname)"
echo "GPUs:      ${CUDA_VISIBLE_DEVICES:-all}"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:   $HF_HOME"
echo "============================================"

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_v0320_v40 \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.pretrained_checkpoint ./checkpoints/Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 50000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id v0320_v40_qwenOFT_finetune_v1 \
  --wandb_project starVLA_v40_finetune \
  --wandb_entity hca
