# 数据集和模型下载指南（Cache 重定向）

本指南说明如何下载 pretrained models 和 datasets，同时将 HuggingFace cache 重定向到自定义目录（避免使用 `~/.cache/huggingface/`）。

## 前置条件：设置环境变量

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

---

## 1. 下载 Pretrained Models

### Qwen3-VL with Action Tokens（OFT/Fast 框架必需）

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

hf download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

### Florence-2（4090 上使用 GR00T 框架时需要）

```bash
hf download microsoft/Florence-2-large \
  --local-dir ./playground/Pretrained_models/Florence-2-large \
  --cache-dir $HF_HUB_CACHE
```

### Robocasa Pretrained Checkpoint（可选，已训练好的模型）

```bash
hf download StarVLA/Qwen3-VL-OFT-Robocasa \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-OFT-Robocasa \
  --cache-dir $HF_HUB_CACHE
```

---

## 2. 下载 LIBERO Dataset

**注意：** 建议在 login node 上执行下载，compute node 可能有网络限制。

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

# 复制 modality.json 到每个数据集
for d in libero_spatial_no_noops_1.0.0_lerobot libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot; do
  mkdir -p $DEST/libero/$d/meta
  cp examples/LIBERO/train_files/modality.json $DEST/libero/$d/meta/
done
```

---

## 3. 下载 Robocasa Dataset

下载脚本使用 `hf_hub_download()`，会自动使用 `HF_HUB_CACHE` 环境变量。

**注意：** 建议在 login node 上执行下载。

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 确保环境变量已设置（见前置条件）
python examples/Robocasa_tabletop/train_files/download_gr00t_ft_data.py
```

下载位置：`./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim`

---

## 4. 下载 LLaVA-OneVision-COCO（Co-training 用）

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

## 5. 永久配置（推荐）

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

## 下载完成后的目录结构

```
playground/
├── Pretrained_models/
│   ├── Qwen3-VL-4B-Instruct-Action/
│   ├── Florence-2-large/
│   └── Qwen3-VL-OFT-Robocasa/
├── Datasets/
│   ├── libero/
│   │   ├── libero_spatial_no_noops_1.0.0_lerobot/
│   │   ├── libero_object_no_noops_1.0.0_lerobot/
│   │   ├── libero_goal_no_noops_1.0.0_lerobot/
│   │   └── libero_10_no_noops_1.0.0_lerobot/
│   ├── LEROBOT_LIBERO_DATA -> libero/
│   ├── nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/
│   └── LLaVA-OneVision-COCO/
```

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
