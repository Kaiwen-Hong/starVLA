# Robocasa and LIBERO Training Setup

This guide covers training with **OFT (Orthogonal Fine-Tuning)** and **Qwen3-VL** on both benchmarks.

## Prerequisites

### Download Qwen3-VL with Action Tokens

OFT requires the model with special action tokens:

```bash
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

确认 Pretrained Model 已下载：
```bash
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/
```

如果没有，先下载（在 login node 执行）：
```bash
hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

### 环境变量（确保 HuggingFace cache 不在 home 目录）

```bash
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

### Flash Attention Configuration

| GPU | Config Setting |
|-----|----------------|
| RTX 4090 | `attn_implementation: flash_attention_2` |
| H100/H200 | `attn_implementation: flash_attention_3` |

See the **Flash Attention** section in [DOC-setup/install.md](install.md) for details.

### GPU Memory Considerations

| Setup | Model | VRAM Required |
|-------|-------|---------------|
| H100/H200 | Qwen3-VL-4B + OFT | ~40 GB |
| RTX 4090 | Qwen2.5-VL-3B + OFT | ~24 GB (tight) |
| RTX 4090 | Florence-2 + GR00T | ~16 GB |

**For 4090 debugging:** Use Florence-2 + QwenGR00T (smaller model, fits in 24GB).

```bash
Framework_name=QwenGR00T
base_vlm=microsoft/Florence-2-large
```

**For cluster training (H200):** Use Qwen3-VL + OFT for best results.

```bash
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
--framework.qwenvl.attn_implementation flash_attention_3
```

---

## LIBERO

### 1. Download Data

```bash
# Create directory and clone datasets
mkdir -p playground/Datasets/libero
cd playground/Datasets/libero

git lfs install
git clone https://huggingface.co/datasets/IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot
git clone https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot
git clone https://huggingface.co/datasets/IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot
git clone https://huggingface.co/datasets/IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot

# Go back to project root
cd ~/Desktop/research/starVLA

# Create symlink (training code expects LEROBOT_LIBERO_DATA)
ln -sf $(pwd)/playground/Datasets/libero $(pwd)/playground/Datasets/LEROBOT_LIBERO_DATA

# Copy modality.json to each dataset's meta folder
# This file defines action/state dimensions for the dataloader
for d in libero_spatial_no_noops_1.0.0_lerobot libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot; do
  mkdir -p playground/Datasets/libero/$d/meta
  cp examples/LIBERO/train_files/modality.json playground/Datasets/libero/$d/meta/
done
```

确认 LIBERO Dataset 已下载：
```bash
ls playground/Datasets/LEROBOT_LIBERO_DATA/
```

应该看到 4 个文件夹：
- `libero_spatial_no_noops_1.0.0_lerobot`
- `libero_object_no_noops_1.0.0_lerobot`
- `libero_goal_no_noops_1.0.0_lerobot`
- `libero_10_no_noops_1.0.0_lerobot`

### 2. Training

Edit `examples/LIBERO/train_files/run_libero_train.sh`:

**For H200 cluster (OFT + Qwen3-VL):**
```bash
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action

# Add to accelerate launch command:
--framework.qwenvl.attn_implementation flash_attention_3
```

**For 4090 debugging (GR00T + Florence):**
```bash
Framework_name=QwenGR00T
base_vlm=microsoft/Florence-2-large

# Use smaller batch size
--datasets.vla_data.per_device_batch_size 1
```

Run:
```bash
bash examples/LIBERO/train_files/run_libero_train.sh
```

#### 2x H100 训练（具体配置）

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash 0130-train-libero.sh
```

#### 训练配置说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_processes` | 2 | GPU 数量 |
| `Framework` | QwenOFT | 使用 OFT (Orthogonal Fine-Tuning) |
| `base_vlm` | Qwen3-VL-4B-Instruct-Action | 带 Action tokens 的模型 |
| `attn_implementation` | flash_attention_3 | H100 专用，更快 |
| `per_device_batch_size` | 16 | 每张卡的 batch size |
| `max_train_steps` | 80000 | 总训练步数 |
| `save_interval` | 10000 | 每 10000 步保存一次 |
| `data_mix` | libero_all | 训练全部 4 个 task suite |

### 3. 监控训练

#### 查看日志
```bash
tail -f results/Checkpoints/0130_libero_h100/train.log
```

#### 查看 GPU 使用
```bash
watch -n 1 nvidia-smi
```

#### WandB（可选）
如果配置了 WandB，可以在网页上查看训练曲线。

### 4. Checkpoints 位置

训练过程中会保存到：
```
results/Checkpoints/0130_libero_h100/
├── steps_10000/
│   └── pytorch_model.pt
├── steps_20000/
│   └── pytorch_model.pt
...
```

### 5. Evaluation

**Terminal 1 (starVLA env):**
```bash
python deployment/model_server/server_policy.py \
  --ckpt_path ./results/Checkpoints/your_run_id/steps_XXXXX/pytorch_model.pt \
  --port 10093 --use_bf16
```

**Terminal 2 (LIBERO env):**
```bash
python examples/LIBERO/eval_files/eval_libero.py \
  --task_suite_name libero_goal \
  --num_trials_per_task 50 \
  --pretrained_path ./results/Checkpoints/your_run_id/steps_XXXXX/pytorch_model.pt
```

### LIBERO Results (Qwen3-VL-OFT @ 30K steps)

| Spatial | Object | Goal | Long | Avg |
|---------|--------|------|------|-----|
| 97.8 | 98.6 | 96.2 | 93.8 | **96.6** |

---

## Robocasa

### 1. Download Data

```bash
python examples/Robocasa_tabletop/download_gr00t_ft_data.py
```

Downloads to `playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`.

### 2. Training

Edit `examples/Robocasa_tabletop/train_files/run_robocasa.sh`:

**For H200 cluster (OFT + Qwen3-VL):**
```bash
Framework_name=QwenOFT
base_vlm=./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
action_input_dim=2560
data_root_dir=./playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_mix=fourier_gr1_unified_1000

# Add to accelerate launch:
--framework.action_model.action_hidden_dim ${action_input_dim}
--framework.qwenvl.attn_implementation flash_attention_3
```

**For 4090 debugging (GR00T + Florence):**
```bash
Framework_name=QwenGR00T
base_vlm=microsoft/Florence-2-large
data_root_dir=./playground/Datasets/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
data_mix=fourier_gr1_unified_1000

# Use smaller batch size
--datasets.vla_data.per_device_batch_size 1
```

Run:
```bash
bash examples/Robocasa_tabletop/train_files/run_robocasa.sh
```

### 3. Evaluation

**Terminal 1 (starVLA env):**
```bash
python deployment/model_server/server_policy.py \
  --ckpt_path ${your_ckpt} \
  --port 5678 --use_bf16
```

**Terminal 2 (robocasa env):**
```bash
python examples/Robocasa_tabletop/eval_files/simulation_env.py \
  --args.env_name ${env_name} \
  --args.port 5678 \
  --args.n_episodes 50 \
  --args.pretrained_path ${your_ckpt}
```

### Robocasa Results (Qwen3-VL-OFT)

Average success rate: **48.8%** across 24 tasks.

---

## Comparison: LIBERO vs Robocasa

| Parameter | LIBERO | Robocasa |
|-----------|--------|----------|
| Action dim | 7 | 29 |
| State dim | 7 | 58 |
| Action horizon | 8 | 16 |
| Action type | `delta_qpos` | `delta_ee` |
| Dataset | `libero_all` | `fourier_gr1_unified_1000` |
| Training steps | 30K-80K | 100K |

---

## Framework Options

| Framework | Action Head | Speed | Model Required |
|-----------|-------------|-------|----------------|
| QwenOFT | MLP (L1) | Fast | `*-Action` |
| QwenFast | Autoregressive | Fast | `*-Action` |
| QwenGR00T | Diffusion (DiT) | Slower | Regular or `-Action` |

---

## Pretrained Checkpoints

- LIBERO: Not yet released for OFT
- Robocasa: [StarVLA/Qwen3-VL-OFT-Robocasa](https://huggingface.co/StarVLA/Qwen3-VL-OFT-Robocasa)

---

## 常见问题 / Common Issues

### CUDA Out of Memory
减小 `per_device_batch_size`：
```bash
--datasets.vla_data.per_device_batch_size 8
```

For 4090, use an even smaller batch size:
```bash
--datasets.vla_data.per_device_batch_size 1
```

### Flash Attention 报错
如果 flash_attention_3 不可用，改回 flash_attention_2：
```bash
--framework.qwenvl.attn_implementation flash_attention_2
```

### NCCL 超时
如果多卡通信超时，检查 NCCL 环境变量设置。
