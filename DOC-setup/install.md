# StarVLA Installation Guide

## 前置条件：HF Cache 重定向

**重要：** 在执行任何下载操作之前必须运行这些命令，或者添加到 `~/.bashrc` 中永久生效。

```bash
# 设置 HuggingFace cache 目录（根据需要修改路径）
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/transformers

# 创建 cache 目录
mkdir -p $HF_HOME/hub
```

### 环境变量说明

| 变量 | 作用 |
|------|------|
| `HF_HOME` | HuggingFace 主目录（存放 token、配置等） |
| `HF_HUB_CACHE` | 模型/数据集下载的 cache 目录 |
| `TRANSFORMERS_CACHE` | `from_pretrained()` 调用时使用的 cache |

### 永久配置（推荐）

将以下内容添加到 `~/.bashrc` 或 `~/.bash_profile`：

```bash
# HuggingFace cache 重定向
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/transformers
```

然后重新加载：
```bash
source ~/.bashrc
```

---

## 1. Environment Setup

```bash
# Clone the repo
git clone https://github.com/starVLA/starVLA
cd starVLA

# Create conda environment
conda create -n starVLA python=3.10 -y
conda activate starVLA

# Install requirements
pip install -r requirements.txt

# Install starVLA
pip install -e .
```

---

## 2. Flash Attention

The `attn_implementation` parameter is configurable via YAML config instead of being hardcoded.

### Environment-Specific Setup

| GPU | Attention Implementation | CUDA Requirement |
|-----|--------------------------|------------------|
| RTX 4090 (Ada Lovelace) | `flash_attention_2` | CUDA 11.x+ |
| H100/H200 (Hopper) | `flash_attention_3` | CUDA 12.3+ (12.8 recommended) |

### YAML Configuration

**Local Machine (4090):**

```yaml
framework:
  qwenvl:
    attn_implementation: flash_attention_2
```

**Cluster (H200):**

```yaml
framework:
  qwenvl:
    attn_implementation: flash_attention_3
```

### Installation

#### Flash Attention 2

```bash
pip install flash-attn --no-build-isolation
```

#### Flash Attention 3 (Hopper GPUs only)

```bash
pip install "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper"
```

Note: FA3 requires Hopper GPU (H100/H200) with compute capability 9.0+.

### Files Modified

- `starVLA/model/modules/vlm/QWen2_5.py` - reads `attn_implementation` from config
- `starVLA/model/modules/vlm/QWen3.py` - reads `attn_implementation` from config

Default is `flash_attention_2` if not specified.

---

## 3. Model Downloads

### Base VLM Models

| Model | Size | Command |
|-------|------|---------|
| Qwen3-VL-4B | ~8GB | `huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct --cache-dir $HF_HUB_CACHE` |
| Qwen2.5-VL-3B | ~6GB | `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct --cache-dir $HF_HUB_CACHE` |
| Florence-2-large | ~1.5GB | `huggingface-cli download microsoft/Florence-2-large --local-dir ./playground/Pretrained_models/Florence-2-large --cache-dir $HF_HUB_CACHE` |

### Action-Extended Models (with Fast Tokens)

```bash
# Qwen2.5-VL with action tokens
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE

# Qwen3-VL with action tokens
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

### Finetuned Checkpoints

| Model | Benchmark | Command |
|-------|-----------|---------|
| Qwen-GR00T-Bridge | SimplerEnv | `huggingface-cli download StarVLA/Qwen-GR00T-Bridge --local-dir ./checkpoints/Qwen-GR00T-Bridge --cache-dir $HF_HUB_CACHE` |
| Qwen3VL-GR00T-Bridge-RT-1 | SimplerEnv | `huggingface-cli download StarVLA/Qwen3VL-GR00T-Bridge-RT-1 --local-dir ./checkpoints/Qwen3VL-GR00T-Bridge-RT-1 --cache-dir $HF_HUB_CACHE` |
| Qwen2.5-VL-GR00T-LIBERO-4in1 | LIBERO | `huggingface-cli download StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1 --local-dir ./checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1 --cache-dir $HF_HUB_CACHE` |
| Qwen3-VL-OFT-Robocasa | RoboCasa | `huggingface-cli download StarVLA/Qwen3-VL-OFT-Robocasa --local-dir ./checkpoints/Qwen3-VL-OFT-Robocasa --cache-dir $HF_HUB_CACHE` |
| Qwen3-VL-OFT-Robotwin2 | RoboTwin | `huggingface-cli download StarVLA/Qwen3-VL-OFT-Robotwin2 --local-dir ./checkpoints/Qwen3-VL-OFT-Robotwin2 --cache-dir $HF_HUB_CACHE` |

---

## 4. Dataset Downloads

All datasets use **LeRobot format**.

**注意：** 建议在 login node 上执行下载，compute node 可能有网络限制。

### LIBERO (4 subsets)

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

DEST=./playground/Datasets
mkdir -p $DEST/libero

# 下载全部 4 个 LIBERO task suite
for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf download "$repo" \
    --repo-type dataset \
    --local-dir "$DEST/libero/${repo##*/}" \
    --cache-dir $HF_HUB_CACHE
done

# 创建 symlink（训练代码需要 LEROBOT_LIBERO_DATA 这个路径）
ln -sf $(pwd)/$DEST/libero $(pwd)/$DEST/LEROBOT_LIBERO_DATA
```

### OXE (Bridge + Fractal)

```bash
huggingface-cli download IPEC-COMMUNITY/bridge_orig_lerobot \
  --repo-type dataset \
  --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot \
  --cache-dir $HF_HUB_CACHE

huggingface-cli download IPEC-COMMUNITY/fractal20220817_data_lerobot \
  --repo-type dataset \
  --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/fractal20220817_data_0.1.0_lerobot \
  --cache-dir $HF_HUB_CACHE
```

### RoboCasa GR1

下载脚本使用 `hf_hub_download()`，会自动使用 `HF_HUB_CACHE` 环境变量。

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 确保环境变量已设置（见前置条件）
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py

# Or manually
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --repo-type dataset \
  --local-dir ./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim \
  --cache-dir $HF_HUB_CACHE
```

下载位置：`./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`

### BEHAVIOR

```bash
huggingface-cli download behavior-1k/2025-challenge-demos \
  --repo-type dataset \
  --local-dir ./playground/Datasets/BEHAVIOR_challenge \
  --cache-dir $HF_HUB_CACHE
```

### LLaVA-OneVision-COCO（Co-training 用）

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

DEST=./playground/Datasets

hf download StarVLA/LLaVA-OneVision-COCO \
  --repo-type dataset \
  --local-dir $DEST/LLaVA-OneVision-COCO \
  --cache-dir $HF_HUB_CACHE

# 解压 COCO 图片
unzip $DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip -d $DEST/LLaVA-OneVision-COCO/
```

---

## 5. Post-Download Setup

### Add modality.json to each dataset

Each dataset needs a `modality.json` in its `meta/` folder:

```bash
# LIBERO
cp examples/LIBERO/modality.json ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta/modality.json
cp examples/LIBERO/modality.json ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta/modality.json
cp examples/LIBERO/modality.json ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta/modality.json
cp examples/LIBERO/modality.json ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta/modality.json

# OXE Bridge
cp examples/SimplerEnv/train_files/modality.json ./playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot/meta/modality.json

# OXE Fractal (rename fractal_modality.json)
cp examples/SimplerEnv/train_files/fractal_modality.json ./playground/Datasets/OXE_LEROBOT_DATASET/fractal20220817_data_0.1.0_lerobot/meta/modality.json
```

---

## 6. Verify Installation

```bash
# Check framework
python starVLA/model/framework/QwenGR00T.py

# Check dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

---

## 7. 下载完成后的目录结构

```
playground/
├── Pretrained_models/
│   ├── Qwen3-VL-4B-Instruct/
│   ├── Qwen2.5-VL-3B-Instruct/
│   ├── Qwen3-VL-4B-Instruct-Action/
│   ├── Qwen2.5-VL-3B-Instruct-Action/
│   ├── Florence-2-large/
│   └── Qwen3-VL-OFT-Robocasa/
├── Datasets/
│   ├── libero/
│   │   ├── libero_spatial_no_noops_1.0.0_lerobot/
│   │   ├── libero_object_no_noops_1.0.0_lerobot/
│   │   ├── libero_goal_no_noops_1.0.0_lerobot/
│   │   └── libero_10_no_noops_1.0.0_lerobot/
│   ├── LEROBOT_LIBERO_DATA -> libero/
│   ├── OXE_LEROBOT_DATASET/
│   │   ├── bridge_orig_1.0.0_lerobot/
│   │   └── fractal20220817_data_0.1.0_lerobot/
│   ├── nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
│   ├── BEHAVIOR_challenge/
│   └── LLaVA-OneVision-COCO/
```

---

## Summary Table

| Component | Source | Local Path |
|-----------|--------|------------|
| **Qwen3-VL-4B** | Qwen/Qwen3-VL-4B-Instruct | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| **Qwen2.5-VL-3B** | Qwen/Qwen2.5-VL-3B-Instruct | `playground/Pretrained_models/Qwen2.5-VL-3B-Instruct` |
| **Florence-2** | microsoft/Florence-2-large | `playground/Pretrained_models/Florence-2-large` |
| **LIBERO data** | IPEC-COMMUNITY/libero_*_lerobot | `playground/Datasets/LEROBOT_LIBERO_DATA/` |
| **OXE data** | IPEC-COMMUNITY/bridge_orig_lerobot | `playground/Datasets/OXE_LEROBOT_DATASET/` |
| **RoboCasa data** | nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim | `playground/Datasets/nvidia/` |
| **BEHAVIOR data** | behavior-1k/2025-challenge-demos | `playground/Datasets/BEHAVIOR_challenge/` |
| **LLaVA-OneVision-COCO** | StarVLA/LLaVA-OneVision-COCO | `playground/Datasets/LLaVA-OneVision-COCO/` |

---

## 常见问题排查

### 检查当前 cache 位置
```bash
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
```

### 清理 home 目录中的旧 cache（如需要）
```bash
# 先检查大小
du -sh ~/.cache/huggingface/

# 确认后删除（小心操作）
# rm -rf ~/.cache/huggingface/
```

### 验证环境变量是否生效
```bash
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
```
