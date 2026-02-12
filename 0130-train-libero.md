# LIBERO 训练指南

本指南适用于 2x H100 (80GB) 配置。

## 前置条件

### 1. 环境变量（确保 HuggingFace cache 不在 home 目录）

```bash
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
```

### 2. 确认 Pretrained Model 已下载

```bash
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/
```

如果没有，先下载（在 login node 执行）：
```bash
hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

### 3. 确认 LIBERO Dataset 已下载

```bash
ls playground/Datasets/LEROBOT_LIBERO_DATA/
```

应该看到 4 个文件夹：
- `libero_spatial_no_noops_1.0.0_lerobot`
- `libero_object_no_noops_1.0.0_lerobot`
- `libero_goal_no_noops_1.0.0_lerobot`
- `libero_10_no_noops_1.0.0_lerobot`

---

## 启动训练

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash 0130-train-libero.sh
```

---

## 训练配置说明

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

---

## 监控训练

### 查看日志
```bash
tail -f results/Checkpoints/0130_libero_h100/train.log
```

### 查看 GPU 使用
```bash
watch -n 1 nvidia-smi
```

### WandB（可选）
如果配置了 WandB，可以在网页上查看训练曲线。

---

## Checkpoints 位置

训练过程中会保存到：
```
results/Checkpoints/0130_libero_h100/
├── steps_10000/
│   └── pytorch_model.pt
├── steps_20000/
│   └── pytorch_model.pt
...
```

---

## 常见问题

### CUDA Out of Memory
减小 `per_device_batch_size`：
```bash
--datasets.vla_data.per_device_batch_size 8
```

### Flash Attention 报错
如果 flash_attention_3 不可用，改回 flash_attention_2：
```bash
--framework.qwenvl.attn_implementation flash_attention_2
```

### NCCL 超时
如果多卡通信超时，检查 NCCL 环境变量设置。
