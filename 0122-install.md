# StarVLA Installation Guide

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

# Install FlashAttention2
pip install flash-attn --no-build-isolation

# Install starVLA
pip install -e .
```

---

## 2. Model Downloads

### Base VLM Models

| Model | Size | Command |
|-------|------|---------|
| Qwen3-VL-4B | ~8GB | `huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| Qwen2.5-VL-3B | ~6GB | `huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct` |
| Florence-2-large | ~1.5GB | `huggingface-cli download microsoft/Florence-2-large --local-dir ./playground/Pretrained_models/Florence-2-large` |

### Action-Extended Models (with Fast Tokens)

```bash
# Qwen2.5-VL with action tokens
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action --local-dir ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action

# Qwen3-VL with action tokens
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

### Finetuned Checkpoints

| Model | Benchmark | Command |
|-------|-----------|---------|
| Qwen-GR00T-Bridge | SimplerEnv | `huggingface-cli download StarVLA/Qwen-GR00T-Bridge --local-dir ./checkpoints/Qwen-GR00T-Bridge` |
| Qwen3VL-GR00T-Bridge-RT-1 | SimplerEnv | `huggingface-cli download StarVLA/Qwen3VL-GR00T-Bridge-RT-1 --local-dir ./checkpoints/Qwen3VL-GR00T-Bridge-RT-1` |
| Qwen2.5-VL-GR00T-LIBERO-4in1 | LIBERO | `huggingface-cli download StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1 --local-dir ./checkpoints/Qwen2.5-VL-GR00T-LIBERO-4in1` |
| Qwen3-VL-OFT-Robocasa | RoboCasa | `huggingface-cli download StarVLA/Qwen3-VL-OFT-Robocasa --local-dir ./checkpoints/Qwen3-VL-OFT-Robocasa` |
| Qwen3-VL-OFT-Robotwin2 | RoboTwin | `huggingface-cli download StarVLA/Qwen3-VL-OFT-Robotwin2 --local-dir ./checkpoints/Qwen3-VL-OFT-Robotwin2` |

---

## 3. Dataset Downloads

All datasets use **LeRobot format**.

### LIBERO (4 subsets)

```bash
# Quick script
export DEST=./playground/Datasets/LEROBOT_LIBERO_DATA
bash examples/LIBERO/data_preparation.sh

# Or manually
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot

huggingface-cli download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot --repo-type dataset --local-dir ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot

huggingface-cli download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot --repo-type dataset --local-dir ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot

huggingface-cli download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot --repo-type dataset --local-dir ./playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot
```

### OXE (Bridge + Fractal)

```bash
huggingface-cli download IPEC-COMMUNITY/bridge_orig_lerobot --repo-type dataset --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/bridge_orig_1.0.0_lerobot

huggingface-cli download IPEC-COMMUNITY/fractal20220817_data_lerobot --repo-type dataset --local-dir ./playground/Datasets/OXE_LEROBOT_DATASET/fractal20220817_data_0.1.0_lerobot
```

### RoboCasa GR1

```bash
# Use download script
python examples/Robocasa_tabletop/download_gr00t_ft_data.py

# Or manually
huggingface-cli download nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim --repo-type dataset --local-dir ./playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim
```

### BEHAVIOR

```bash
huggingface-cli download behavior-1k/2025-challenge-demos --repo-type dataset --local-dir ./playground/Datasets/BEHAVIOR_challenge
```

---

## 4. Post-Download Setup

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

## 5. Verify Installation

```bash
# Check framework
python starVLA/model/framework/QwenGR00T.py

# Check dataloader
python starVLA/dataloader/lerobot_datasets.py --config_yaml examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml
```

---

## Summary Table

| Component | Source | Local Path |
|-----------|--------|------------|
| **Qwen3-VL-4B** | Qwen/Qwen3-VL-4B-Instruct | `playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| **Florence-2** | microsoft/Florence-2-large | `playground/Pretrained_models/Florence-2-large` |
| **LIBERO data** | IPEC-COMMUNITY/libero_*_lerobot | `playground/Datasets/LEROBOT_LIBERO_DATA/` |
| **OXE data** | IPEC-COMMUNITY/bridge_orig_lerobot | `playground/Datasets/OXE_LEROBOT_DATASET/` |
| **RoboCasa data** | nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim | `playground/Datasets/nvidia/` |
| **BEHAVIOR data** | behavior-1k/2025-challenge-demos | `playground/Datasets/BEHAVIOR_challenge/` |
