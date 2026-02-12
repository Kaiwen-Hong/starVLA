#!/bin/bash
set -e

###########################################################################################
# LIBERO Training Script for 2x H100 (80GB)
# 使用方法: bash 0130-train-libero.sh
###########################################################################################

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# === 加载 CUDA Module（HPC 集群必需）===
module load cuda/12.2.0-fasrc01 || module load cuda || echo "Warning: Could not load CUDA module"

# === 自动设置 CUDA_HOME（根据 nvcc 位置）===
NVCC_PATH=$(which nvcc 2>/dev/null)
if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname $(dirname $NVCC_PATH))
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
    echo "CUDA_HOME set to: $CUDA_HOME"
else
    echo "Error: nvcc not found. Please load CUDA module."
    exit 1
fi

# === Triton Cache 设置（避免 NFS 慢）===
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
mkdir -p $TRITON_CACHE_DIR

# === HuggingFace Cache 设置（避免使用 home 目录）===
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# === NCCL 设置（多卡通信）===
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000

###########################################################################################
# === 训练配置（根据需要修改）===
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all
run_root_dir=./results/Checkpoints
run_id=0130_libero_h100

# GPU 配置
num_gpus=2
batch_size=16

# 训练步数
max_train_steps=80000
save_interval=10000
###########################################################################################

# 创建输出目录
output_dir=${run_root_dir}/${run_id}
mkdir -p ${output_dir}

# 复制脚本到输出目录（方便复现）
cp $0 ${output_dir}/

echo "=============================================="
echo "Starting LIBERO Training"
echo "Run ID: ${run_id}"
echo "GPUs: ${num_gpus}"
echo "Batch size per GPU: ${batch_size}"
echo "Output: ${output_dir}"
echo "=============================================="

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes ${num_gpus} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  --framework.qwenvl.base_vlm ${base_vlm} \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir ${libero_data_root} \
  --datasets.vla_data.data_mix ${data_mix} \
  --datasets.vla_data.per_device_batch_size ${batch_size} \
  --trainer.vla_data.video_backend torchvision_av \
  --trainer.max_train_steps ${max_train_steps} \
  --trainer.save_interval ${save_interval} \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 100 \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --wandb_project starVLA_Libero

echo "=============================================="
echo "Training completed!"
echo "Checkpoints saved to: ${output_dir}"
echo "=============================================="
