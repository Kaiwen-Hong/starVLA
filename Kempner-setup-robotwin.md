# RoboTwin 2.0 Setup Guide for StarVLA (Kempner)

> **CRITICAL:** All downloads/caches go under `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/`, **never `~`**.

---

## Current Status

The **RoboTwin-Randomized** dataset has been downloaded and extracted. It is ready for training.

| Item | Status | Location |
|------|--------|----------|
| Randomized tar.gz | Downloaded | `data/RoboTwin-Randomized-targz/` (50 `.tar.gz` files) |
| Extracted dataset | Ready | `playground/Datasets/RoboTwin/` (50 task directories) |
| Dataset version | **Randomized** | 500 episodes/task, fps=15, mp4 video (AV1), modality.json included |
| Total data | 25,000 episodes | 50 tasks × 500 episodes each |
| Compatibility | Verified | No config changes needed — works directly with `bash examples/Robotwin/train_files/run_robotwin_train.sh` |

**Note on data sources:** There are two versions of the RoboTwin dataset (see [Section 4.1](#41-two-dataset-versions) for full comparison):

| | Demo-Clean (`YaoMarkMu/robotwin_dataset`) | Randomized (`StarVLA/RoboTwin-Randomized-targz`) |
|---|---|---|
| Episodes/task | 50 | **500** |
| FPS | 50 | 15 |
| Image storage | Image-in-parquet (163 GB) | MP4 video, AV1 codec (smaller) |
| Scene variation | Fixed | **Randomized** positions & appearances |
| Recommended | Debugging only | **Training** |

Both are LeRobot v2.1 format. The StarVLA dataloader auto-detects video vs image mode from `modality.json` — no config changes needed for either version.

A previously processed merged version of the demo-clean data also exists at `$HF_LEROBOT_HOME/demo_clean_all_repo/` (all 50 tasks combined into one dataset, 2,500 episodes, image-in-parquet, fps=50). This is **not** used by the per-task training pipeline in `mixtures.py`.

---

## TL;DR — Copy-Paste Quickstart

Run these blocks in order. Each step is self-contained. Details in later sections.

```bash
# ── 0. Env vars (add to ~/.bashrc-kaiwen, then source ~/.bashrc-kaiwen) ──
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
export HF_HOME=$LAB_ROOT/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export CONDA_PKGS_DIRS=$LAB_ROOT/.conda/pkgs
export PIP_CACHE_DIR=$LAB_ROOT/.cache/pip
export WANDB_DIR=$LAB_ROOT/.cache/wandb
export WANDB_CACHE_DIR=$LAB_ROOT/.cache/wandb
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME/hub $CONDA_PKGS_DIRS $PIP_CACHE_DIR $WANDB_DIR $TRITON_CACHE_DIR
```

```bash
# ── 1. Conda env + install ──
conda create -n starVLA python=3.10 -y && conda activate starVLA
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
module load cuda/12.2.0-fasrc01 || module load cuda
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

```bash
# ── 2. Download pretrained VLM (login node, ~8GB) ──
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

```bash
# ── 3. Download RoboTwin dataset (login node) ──
# Option A: Demo-clean (50 tasks × 50 episodes = 2,500 demos, image-in-parquet, fps=50)
mkdir -p playground/Datasets/RoboTwin
huggingface-cli download YaoMarkMu/robotwin_dataset \
  --repo-type dataset \
  --local-dir ./playground/Datasets/RoboTwin \
  --cache-dir $HF_HUB_CACHE

# Option B: Randomized (50 tasks × 500 episodes = 25,000 demos, mp4 video, fps=15)
bash scripts/download_robotwin_randomized.sh
# Then extract into playground/Datasets/RoboTwin:
mkdir -p playground/Datasets/RoboTwin
for f in data/RoboTwin-Randomized-targz/*.tar.gz; do
  tar -xzf "$f" -C playground/Datasets/RoboTwin/
done
```

```bash
# ── 4. Copy modality.json into every task folder (Option A only; Option B already includes it) ──
for d in playground/Datasets/RoboTwin/*/; do
  if [ ! -f "$d/meta/modality.json" ]; then
    mkdir -p "$d/meta"
    cp examples/Robotwin/train_files/modality.json "$d/meta/"
  fi
done
```

```bash
# ── 5. Smoke test ──
python starVLA/model/framework/QwenGR00T.py   # should print model and exit
```

```bash
# ── 6. Train (4×H100 ready-to-use command) ──
#
# Pre-conditions:
#   - conda activate starVLA
#   - module load cuda/12.2.0-fasrc01  (or equivalent CUDA module)
#   - Pretrained VLM downloaded to playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/
#   - Dataset extracted to playground/Datasets/RoboTwin/ (50 task directories)
#   - For QwenFast: also need playground/Pretrained_models/fast/ (see Section 7.2)
#
# Framework choices:
#   - QwenOFT:  recommended, no extra downloads needed
#   - QwenFast: needs FAST tokenizer (huggingface: physical-intelligence/fast)
#   - QwenGR00T / QwenPI: flow-matching variants
#

# before running, make sure to run:

source ~/.bashrc-kaiwen                                                             
conda activate starVLA
module load cuda/12.2.0-fasrc01                                                     
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA




accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.data_mix robotwin \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 100000 \
  --trainer.save_interval 10000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 1000 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id robotwin_qwenOFT_4xH100 \
  --wandb_project starVLA_Robotwin \
  --wandb_entity kaiwenh-17-uiuc
```

> **4×H100 notes:**
> - `--num_processes 4` — matches 4 GPUs
> - `--trainer.gradient_accumulation_steps 2` — compensates for fewer GPUs (effective global batch = 4 × 8 × 2 = 64, same as 8 GPU with batch 8)
> - `--framework.qwenvl.attn_implementation sdpa` — use this if flash_attn has GLIBC issues on your node (see [Troubleshooting](#kempner-glibc-and-flash-attention)); switch to `flash_attention_2` on nodes with GLIBC >= 2.32 for better performance
> - For quick debugging, replace `--datasets.vla_data.data_mix robotwin` with `robotwin_task1` (loads only adjust_bottle)

That's it. Scroll down for detailed explanations, config tuning, evaluation, and troubleshooting.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Environment Setup](#2-environment-setup)
3. [Download Pretrained Models](#3-download-pretrained-models)
4. [Download RoboTwin Dataset](#4-download-robotwin-dataset) (Demo-Clean vs Randomized)
5. [Post-Download Setup (modality.json)](#5-post-download-setup)
6. [Verify Installation](#6-verify-installation)
7. [Training](#7-training)
8. [Evaluation](#8-evaluation)
9. [Architecture Deep Dive](#9-architecture-deep-dive)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

**What is RoboTwin 2.0?**
A dual-arm robot simulation benchmark with 50 manipulation tasks (adjust_bottle, stack_bowls, etc.), each with 50 demonstrations. The robot has two 7-DOF arms, producing a 14-dim action/state space (6 joints + 1 gripper per arm).

**What does StarVLA do?**
StarVLA trains a Vision-Language-Action model that takes in camera images + language instruction and outputs 14-dim joint actions. It uses a VLM backbone (Qwen2.5/Qwen3-VL) combined with an action model (DiT diffusion, FAST tokenizer, or flow-matching).

**Architecture choices for RoboTwin:**

| Framework | Description | Action Head |
|-----------|-------------|-------------|
| `QwenOFT` | MLP/DiT action head on VLM hidden states | DiT-B diffusion |
| `QwenFast` | Autoregressive action token generation | FAST tokenizer |
| `QwenGR00T` | Dual-system: VLM reasoning + flow-matching | Flow-matching |
| `QwenPI` | Layerwise flow-matching | Flow-matching |

The default training script uses `QwenFast`, while the YAML config defaults to `QwenOFT`. Choose based on your needs.

---

## 2. Environment Setup

### 2.1 Set ALL Cache/Storage Environment Variables (DO THIS FIRST)

**This must be done BEFORE creating the conda env or installing anything.**

Add all of the following to `~/.bashrc-kaiwen`:

```bash
# ============================================================
# STORAGE REDIRECTION — keep everything off ~ (limited quota)
# ============================================================
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen

# --- HuggingFace ---
export HF_HOME=$LAB_ROOT/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets

# --- Conda packages (avoid filling ~/.conda/pkgs) ---
export CONDA_PKGS_DIRS=$LAB_ROOT/.conda/pkgs

# --- pip cache ---
export PIP_CACHE_DIR=$LAB_ROOT/.cache/pip

# --- WandB ---
export WANDB_DIR=$LAB_ROOT/.cache/wandb
export WANDB_CACHE_DIR=$LAB_ROOT/.cache/wandb

# --- Triton compilation (use local /tmp for speed) ---
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER

# --- Misc ---
export TOKENIZERS_PARALLELISM=false
```

Then reload and create the cache directories:

```bash
source ~/.bashrc-kaiwen

mkdir -p $HF_HOME/hub
mkdir -p $CONDA_PKGS_DIRS
mkdir -p $PIP_CACHE_DIR
mkdir -p $WANDB_DIR
mkdir -p $TRITON_CACHE_DIR
```

**Verify the variables are set correctly:**

```bash
echo "HF_HOME:       $HF_HOME"
echo "HF_HUB_CACHE:  $HF_HUB_CACHE"
echo "PIP_CACHE_DIR: $PIP_CACHE_DIR"
echo "CONDA_PKGS_DIRS: $CONDA_PKGS_DIRS"

# Should print the lab path, NOT anything under ~/
python -c "from huggingface_hub import constants; print('HF cache:', constants.HF_HUB_CACHE)"
```

### 2.2 Create Conda Environment

> **IMPORTANT:** Make sure `which conda` points to the **lab miniforge**
> (`/net/.../kaiwen/miniforge3/condabin/conda`), NOT the home miniforge
> (`~/miniforge3/condabin/conda`). Run `source ~/.bashrc-kaiwen` first.
> If you create the env under the wrong miniforge, `which python` will
> resolve to the wrong binary even after `conda activate`. See
> `0206-change-bashrc-conda-related.md` for full details.

```bash
# Option A: named env (conda will still use CONDA_PKGS_DIRS for package cache)
conda create -n starVLA python=3.10 -y
conda activate starVLA

# Option B: prefix env stored entirely under lab directory (recommended if ~ is tight)
conda create --prefix $LAB_ROOT/conda_envs/starVLA python=3.10 -y
conda activate $LAB_ROOT/conda_envs/starVLA
```

### 2.3 Load CUDA Module (HPC Only)

```bash
module load cuda/12.2.0-fasrc01 || module load cuda

# Verify CUDA
nvcc -V
# Expected: CUDA 12.0 or 12.4 (verified compatible versions)
```

### 2.4 Install Dependencies

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Install Python requirements
pip install -r requirements.txt

# Install FlashAttention2 (must match your CUDA version)
pip install flash-attn --no-build-isolation

# Install starVLA in editable mode
pip install -e .
```

**Key dependencies** (from `requirements.txt`):
- `transformers==4.57.0`
- `accelerate==1.5.2`
- `deepspeed==0.16.9`
- `torchvision==0.21.0`
- `numpy==1.26.4`
- `flash-attn` (verified: `2.7.4.post1` works with CUDA 12.0/12.4)

### 2.5 Verify Flash Attention

```bash
python -c "import flash_attn; print(flash_attn.__version__)"
pip list | grep -E 'torch|transformers|flash-attn'
```

If `flash-attn` fails to install, check that your `nvcc` version and PyTorch CUDA version match.

---

## 3. Download Pretrained Models

All models go into `playground/Pretrained_models/` (under the repo, which is under lab storage).

> **Reminder:** Every `huggingface-cli download` below uses `--cache-dir $HF_HUB_CACHE`
> to keep the HF blob cache under lab storage. **Never omit this flag** — without it,
> HuggingFace defaults to `~/.cache/huggingface/` which will fill your home quota.
> Verify `echo $HF_HUB_CACHE` prints the lab path before running any download.

### 3.1 Base VLM with Action Tokens (Required)

Run downloads on a **login node** (compute nodes may block outbound network).

For `QwenFast` or `QwenOFT` frameworks:

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Verify cache is NOT in home dir
echo $HF_HUB_CACHE
# Must print: /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/hub

# Qwen3-VL-4B with action tokens (~8GB)
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE

# OR Qwen2.5-VL-3B with action tokens (~6GB)
huggingface-cli download StarVLA/Qwen2.5-VL-3B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

### 3.2 Pre-trained RoboTwin Checkpoint (Optional, for evaluation only)

```bash
huggingface-cli download StarVLA/Qwen3-VL-OFT-Robotwin2 \
  --local-dir ./checkpoints/Qwen3-VL-OFT-Robotwin2 \
  --cache-dir $HF_HUB_CACHE
```

### 3.3 Verify Models Exist

```bash
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/
# Should see: config.json, model*.safetensors, tokenizer*, etc.
```

---

## 4. Download RoboTwin Dataset

The RoboTwin dataset must be in LeRobot format and placed at `playground/Datasets/RoboTwin/`.

> **Storage note:** The dataset goes into the repo's `playground/Datasets/` directory which
> is already under lab storage. The HF download cache (`--cache-dir`) also goes to lab storage.

### 4.1 Two Dataset Versions

There are two versions of the RoboTwin dataset, both in LeRobot v2.1 format:

| | Demo-Clean | Randomized |
|---|---|---|
| HuggingFace repo | `YaoMarkMu/robotwin_dataset` | `StarVLA/RoboTwin-Randomized-targz` |
| Episodes per task | 50 | **500** |
| Total episodes | 2,500 | **25,000** |
| FPS | 50 | 15 |
| Image storage | **Image-in-parquet** (dtype: `image`) | **MP4 video** (dtype: `video`, AV1 codec) |
| Dataset size | ~163 GB (images embedded in parquet) | Smaller (compressed video) |
| Task descriptions | ~48 unique per task | ~420 unique per task (more diverse) |
| Scene variation | Fixed object positions/appearances | **Randomized** object positions & appearances |
| Download format | Ready-to-use directories | `.tar.gz` archives (need extraction) |
| modality.json | **Not included** (must copy manually) | **Included** in each tar.gz |

**Which to use?**
- **Randomized** is recommended for training — 10x more data, better generalization, and matches the StarVLA per-task data loading pipeline directly.
- **Demo-clean** is useful for quick debugging or if you want to use the single merged dataset at `$HF_LEROBOT_HOME/demo_clean_all_repo/`.

### 4.2 Option A: Download Demo-Clean Dataset

Run on **login node**:
```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Verify cache var
echo $HF_HUB_CACHE  # must NOT be under ~/

mkdir -p playground/Datasets/RoboTwin

# Download all task datasets — each task is a separate dataset folder
huggingface-cli download YaoMarkMu/robotwin_dataset \
  --repo-type dataset \
  --local-dir ./playground/Datasets/RoboTwin \
  --cache-dir $HF_HUB_CACHE
```

If generating data from the RoboTwin simulator, follow their official documentation to collect demonstrations, then convert to LeRobot format.

### 4.3 Option B: Download Randomized Dataset (Recommended)

**Step 1: Download** the tar.gz archives (run on **login node**):
```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash scripts/download_robotwin_randomized.sh
```

This downloads 50 `.tar.gz` files to `data/RoboTwin-Randomized-targz/`.

**Step 2: Extract** all archives into `playground/Datasets/RoboTwin/`:
```bash
mkdir -p playground/Datasets/RoboTwin
for f in data/RoboTwin-Randomized-targz/*.tar.gz; do
  echo "Extracting $(basename $f) ..."
  tar -xzf "$f" -C playground/Datasets/RoboTwin/
done
```

**Step 3: Verify** extraction:
```bash
# Should list 50 task directories
ls playground/Datasets/RoboTwin/ | wc -l

# Check a sample task has all required files
ls playground/Datasets/RoboTwin/adjust_bottle/meta/
# Expected: episodes.jsonl  episodes_stats.jsonl  info.json  modality.json  tasks.jsonl

ls playground/Datasets/RoboTwin/adjust_bottle/videos/chunk-000/ | head -3
# Expected: observation.images.cam_high/  observation.images.cam_left_wrist/  observation.images.cam_right_wrist/
```

> **Note:** The randomized tar.gz archives already include `modality.json` in each task's `meta/` folder, so you can skip Section 5.1.

### 4.4 Expected Dataset Structure

After download/extraction, each task should be its own folder under `playground/Datasets/RoboTwin/`:

```
playground/Datasets/RoboTwin/
├── adjust_bottle/
│   ├── meta/
│   │   ├── modality.json          # Defines data modality mapping
│   │   ├── episodes.jsonl         # Episode metadata
│   │   ├── episodes_stats.jsonl   # Per-episode statistics
│   │   ├── tasks.jsonl            # Task descriptions (420 for randomized, fewer for clean)
│   │   └── info.json              # Dataset info (fps, num_episodes, etc.)
│   ├── data/
│   │   └── chunk-000/
│   │       └── episode_*.parquet  # Parquet files with state/action data
│   └── videos/                    # Only present in Randomized version
│       └── chunk-000/
│           ├── observation.images.cam_high/
│           │   └── episode_*.mp4
│           ├── observation.images.cam_left_wrist/
│           │   └── episode_*.mp4
│           └── observation.images.cam_right_wrist/
│               └── episode_*.mp4
├── beat_block_hammer/
│   └── (same structure)
├── blocks_ranking_rgb/
│   └── (same structure)
...
└── turn_switch/
    └── (same structure)
```

### 4.5 All 50 RoboTwin Tasks

The `robotwin` data mix in `starVLA/dataloader/gr00t_lerobot/mixtures.py` (line 90) expects all these task folders:

```
adjust_bottle          beat_block_hammer      blocks_ranking_rgb
blocks_ranking_size    click_alarmclock       click_bell
dump_bin_bigbin        grab_roller            handover_block
handover_mic           hanging_mug            lift_pot
move_can_pot           move_pillbottle_pad    move_playingcard_away
move_stapler_pad       open_laptop            open_microwave
pick_diverse_bottles   pick_dual_bottles      place_a2b_left
place_a2b_right        place_bread_basket     place_bread_skillet
place_burger_fries     place_can_basket       place_cans_plasticbox
place_container_plate  place_dual_shoes       place_empty_cup
place_fan              place_mouse_pad        place_object_basket
place_object_scale     place_object_stand     place_phone_stand
place_shoe             press_stapler          put_bottles_dustbin
put_object_cabinet     rotate_qrcode          scan_object
shake_bottle           shake_bottle_horizontally
stack_blocks_three     stack_blocks_two       stack_bowls_three
stack_bowls_two        stamp_seal             turn_switch
```

You can also use smaller subsets for debugging:
- `robotwin_task1` — only `adjust_bottle`
- `robotwin_task2` — only `place_a2b_left` + `place_a2b_right`

---

## 5. Post-Download Setup

### 5.1 Add modality.json to Each Task Dataset

Each task's `meta/` folder **must** have a `modality.json` file. The RoboTwin-specific modality mapping is at `examples/Robotwin/train_files/modality.json`.

> **Note:** If you used **Option B (Randomized)**, each tar.gz already includes `modality.json` — you can skip this step. The script below is safe to run regardless; it only copies when the file is missing.

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Copy modality.json to task folders that don't already have it
for task_dir in playground/Datasets/RoboTwin/*/; do
  if [ ! -f "$task_dir/meta/modality.json" ]; then
    mkdir -p "$task_dir/meta"
    cp examples/Robotwin/train_files/modality.json "$task_dir/meta/"
  fi
done
```

**What does modality.json define?** It maps the LeRobot parquet column names to the starVLA modality format:

| StarVLA Key | LeRobot Column | Dimensions |
|-------------|---------------|------------|
| `action.left_joints` | `action` [0:6] | 6 |
| `action.left_gripper` | `action` [6:7] | 1 |
| `action.right_joints` | `action` [7:13] | 6 |
| `action.right_gripper` | `action` [13:14] | 1 |
| `state.left_joints` | `observation.state` [0:6] | 6 |
| `state.left_gripper` | `observation.state` [6:7] | 1 |
| `state.right_joints` | `observation.state` [7:13] | 6 |
| `state.right_gripper` | `observation.state` [13:14] | 1 |
| `video.cam_high` | `observation.images.cam_high` | Head camera |
| `video.cam_left_wrist` | `observation.images.cam_left_wrist` | Left wrist camera |
| `video.cam_right_wrist` | `observation.images.cam_right_wrist` | Right wrist camera |
| `annotation.human.action.task_description` | `task_index` | Language instruction |

### 5.2 Verify Dataset Statistics

Each task folder should also have `meta/stats_gr00t.json` with normalization statistics. If missing, the dataloader will compute them on first load (slow but automatic).

---

## 6. Verify Installation

### 6.1 Smoke Test the Framework

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Test that the model framework builds correctly
python starVLA/model/framework/QwenGR00T.py
# Should print the model architecture and succeed

python starVLA/model/framework/QwenOFT.py
python starVLA/model/framework/QwenFast.py
```

### 6.2 Smoke Test the Dataloader (Optional)

```bash
# Test with the RoboTwin config
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml
```

Note: This requires a debugger to attach (the main block has `debugpy.wait_for_client()`). You can skip this and just go directly to training.

---

## 7. Training

### 7.1 Quick Start (4×H100, Verified)

The ready-to-use training command is in the [TL;DR Quickstart (Step 6)](#tldr--copy-paste-quickstart). Copy it directly.

Alternatively, you can use the provided shell script with edits:
```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
# Edit the script first (see 7.2), then:
bash examples/Robotwin/train_files/run_robotwin_train.sh
```

### 7.2 Configure the Training Script

Edit `examples/Robotwin/train_files/run_robotwin_train.sh`:

```bash
# === Key variables to edit ===
Framework_name=QwenOFT               # QwenOFT (recommended) | QwenFast | QwenGR00T | QwenPI
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
config_yaml=./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml
run_root_dir=./results/Checkpoints
data_mix=robotwin                    # "robotwin" for all 50 tasks, "robotwin_task1" for debug
run_id=robotwin_qwenOFT_4xH100      # Unique experiment name
```

Also edit `--num_processes` and add `gradient_accumulation_steps` to match your GPU count:
```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \                                  # <-- Set to your number of GPUs
  starVLA/training/train_starvla.py \
  ...
  --framework.qwenvl.attn_implementation sdpa \        # <-- Use sdpa if flash_attn has GLIBC issues
  --trainer.gradient_accumulation_steps 2 \            # <-- 2 for 4 GPUs (keeps global batch = 64)
  ...
```

**Framework-specific requirements:**

| Framework | Extra Downloads Needed | Notes |
|-----------|----------------------|-------|
| **QwenOFT** | None | Recommended. DiT-B action head, works out of the box |
| **QwenFast** | `physical-intelligence/fast` tokenizer | Download to `playground/Pretrained_models/fast/` |
| **QwenGR00T** | None | Flow-matching action head |
| **QwenPI** | None | Layerwise flow-matching |

To download the FAST tokenizer (only needed for QwenFast):
```bash
huggingface-cli download physical-intelligence/fast \
  --local-dir ./playground/Pretrained_models/fast \
  --cache-dir $HF_HUB_CACHE
```

**GPU scaling reference:**

| GPUs | `--num_processes` | `per_device_batch_size` | `gradient_accumulation_steps` | Effective global batch |
|------|-------------------|------------------------|-------------------------------|----------------------|
| 8 | 8 | 8 | 1 | 64 |
| **4** | **4** | **8** | **2** | **64** |
| 2 | 2 | 8 | 4 | 64 |

### 7.3 Training Config YAML Deep Dive

The config at `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` controls everything.

**Model architecture:**
```yaml
framework:
  name: QwenOFT                              # Overridden by shell script's Framework_name
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct  # Overridden by script
    attn_implementation: flash_attention_2    # Use "sdpa" if flash-attn unavailable
    vl_hidden_dim: 2048                       # VLM hidden dimension
  action_model:
    action_dim: 14                            # 7 per arm (6 joints + 1 gripper)
    state_dim: 14
    future_action_window_size: 15             # Predict 15 future steps
    action_horizon: 16                        # Total action chunk = 16 steps
```

**Dataset:**
```yaml
datasets:
  vla_data:
    dataset_py: lerobot_datasets              # Uses LeRobot format loader
    data_root_dir: playground/Datasets/RoboTwin
    data_mix: robotwin                        # All 50 tasks
    action_type: abs_qpos                     # Absolute joint positions
    per_device_batch_size: 16                 # Batch size per GPU
    obs: ["image_0"]                          # Number of observation image sets
    image_size: [224, 224]                    # Image resolution
    video_backend: torchvision_av             # Video decoder backend
```

**Trainer:**
```yaml
trainer:
  max_train_steps: 1000000
  save_interval: 5000
  eval_interval: 100
  learning_rate:
    base: 1e-05                               # Default LR
    qwen_vl_interface: 1.0e-05                # VLM LR
    action_model: 1.0e-04                     # Action head LR (10x higher)
  lr_scheduler_type: cosine_with_min_lr
  freeze_modules: ''                          # Empty = train everything
  loss_scale:
    vla: 1.0                                  # Action loss weight
    vlm: 0.1                                  # VLM co-training loss weight
```

### 7.4 DeepSpeed Configuration

Training uses DeepSpeed ZeRO Stage 2 (`starVLA/config/deepseeds/deepspeed_zero2.yaml` + `ds_config.yaml`):
- BF16 mixed precision enabled
- ZeRO-2: optimizer states + gradients partitioned across GPUs
- Gradient clipping: 1.0
- No CPU offloading

### 7.5 Multi-Node Training (Slurm)

For multi-node, uncomment the multi-server section in the training script:

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --main_process_ip $MASTER_ADDR \
  --main_process_port $MASTER_PORT \
  --machine_rank $SLURM_PROCID \
  --num_machines $SLURM_NNODES \
  --num_processes=${TOTAL_GPUS} \
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  ...
```

Add NCCL settings for multi-node communication:
```bash
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
```

### 7.6 Monitor Training

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Watch training logs
tail -f results/Checkpoints/<run_id>/train.log

# WandB: set these in the script to enable
# --wandb_project starVLA_Robotwin
# --wandb_entity your_name
# Remove "export WANDB_MODE=disabled" to enable WandB
```

### 7.7 Checkpoint Structure

Training saves checkpoints like:
```
results/Checkpoints/<run_id>/
├── config.yaml                    # Full training config (saved automatically)
├── dataset_statistics.json        # Action/state normalization stats
├── checkpoints/
│   ├── steps_10000/
│   │   └── pytorch_model.pt       # Model weights
│   ├── steps_20000/
│   │   └── pytorch_model.pt
│   └── ...
```

Both `config.yaml` and `dataset_statistics.json` are required at inference time (loaded by `starVLA/model/tools.py:read_mode_config()`).

---

## 8. Evaluation

Evaluation requires **two separate conda environments** and **two terminals** running simultaneously.

### 8.1 Setup RoboTwin Simulator Environment

In a **separate** conda environment (not starVLA):

```bash
# Follow the official RoboTwin install guide:
# https://robotwin-platform.github.io/doc/usage/robotwin-install.html

# Make sure CONDA_PKGS_DIRS and PIP_CACHE_DIR are set (see Section 2.1)

# Option A: named env
conda create -n robotwin python=3.10 -y
conda activate robotwin

# Option B: prefix env under lab storage
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
conda create --prefix $LAB_ROOT/conda_envs/robotwin python=3.10 -y
conda activate $LAB_ROOT/conda_envs/robotwin

# ... follow RoboTwin installation steps ...

# Then install additional deps for StarVLA communication:
pip install -r examples/Robotwin/eval_files/requirements.txt
```

Additional requirements (`examples/Robotwin/eval_files/requirements.txt`):
- `accelerate==1.5.2`
- `json_numpy==2.1.1`
- `websockets==15.0.1`
- `msgpack==1.1.2`
- `rich==14.2.0`
- `omegaconf==2.3.0`

### 8.2 Configure Evaluation

**Edit `examples/Robotwin/eval_files/deploy_policy.yml`:**
```yaml
policy_name: starVLA
host: "127.0.0.1"
port: 5694
policy_ckpt_path: "/path/to/your/checkpoint/steps_60000_pytorch_model.pt"
unnorm_key: "robotwin"       # Key to look up in dataset_statistics.json
```

**Edit `examples/Robotwin/eval_files/run_policy_server.sh`:**
```bash
# If using named env:
export star_vla_python=$(conda run -n starVLA which python)
# If using prefix env:
export star_vla_python=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/conda_envs/starVLA/bin/python

your_ckpt=results/Checkpoints/<run_id>/checkpoints/steps_XXXXX_pytorch_model.pt
gpu_id=0
port=5694
```

**Edit `examples/Robotwin/eval_files/eval.sh`:**
```bash
ROBOTWIN_PATH=/path/to/your/RoboTwin   # Path to RoboTwin simulator repo
```

### 8.3 Run Evaluation

**Terminal 1** (starVLA environment) — start the policy server:
```bash
conda activate starVLA
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash examples/Robotwin/eval_files/run_policy_server.sh
```

This loads the model checkpoint and starts a WebSocket server on the specified port.

**Terminal 2** (robotwin environment) — run the simulation:
```bash
conda activate robotwin
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA/examples/Robotwin/eval_files
bash eval.sh <task_name> <task_config> <ckpt_setting> <seed> <gpu_id>
```

**Example:**
```bash
bash eval.sh adjust_bottle demo_clean my_test_v1 0 0
```

**Arguments:**
| Arg | Description | Example |
|-----|-------------|---------|
| `task_name` | Task to evaluate | `adjust_bottle` |
| `task_config` | Difficulty mode | `demo_clean` or `demo_randomized` |
| `ckpt_setting` | Test identifier | `starvla_demo` |
| `seed` | Random seed | `0` |
| `gpu_id` | GPU for simulator | `0` |

### 8.4 Evaluation Data Flow

```
RoboTwin Simulator                              StarVLA Policy Server
─────────────────                              ────────────────────────
1. Captures observation
   (3 cameras: head, left, right)
   + joint state (14-dim)
                    ──── WebSocket ────>        2. Receives images + instruction
                                               3. Resizes images to 224x224
                                               4. VLM encodes images + text
                                               5. Action model predicts
                                                  normalized actions [-1,1]
                    <─── WebSocket ────         6. Returns action chunk (16 steps)
7. Client denormalizes actions
   using stats from checkpoint
8. Reorders action dims
   (training order -> robot order)
9. Executes action on robot
10. Repeats every 16 steps
```

---

## 9. Architecture Deep Dive

### 9.1 Data Config: AgilexDataConfig

Defined in `starVLA/dataloader/gr00t_lerobot/data_config.py` (line 829).

The RoboTwin robot type `"robotwin"` maps to `AgilexDataConfig`, which defines:

- **3 cameras**: `cam_high`, `cam_left_wrist`, `cam_right_wrist`
- **14-dim actions**: 6 left joints + 1 left gripper + 6 right joints + 1 right gripper
- **14-dim states**: same layout as actions
- **Normalization**: min-max for joints, binary threshold for grippers

### 9.2 Embodiment Tag

In `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py`, `"robotwin"` is NOT explicitly listed in `ROBOT_TYPE_TO_EMBODIMENT_TAG`. The code at `starVLA/dataloader/lerobot_datasets.py` (line 40-42) handles this:

```python
if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
    print(f"Warning: Robot type {robot_type} not found ..., using NEW_EMBODIMENT as default")
    embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
```

This is expected behavior — RoboTwin uses the `NEW_EMBODIMENT` tag (projector index 31).

### 9.3 Data Mixture Registry

In `starVLA/dataloader/gr00t_lerobot/mixtures.py` (line 90):

```python
"robotwin": [
    ("adjust_bottle", 1.0, "robotwin"),
    ("beat_block_hammer", 1.0, "robotwin"),
    ...  # 50 tasks total, each weight 1.0 (equal sampling)
]
```

Each tuple: `(dataset_folder_name, sampling_weight, robot_type)`.

### 9.4 Training Pipeline

```
train_starvla.py
├── load config (OmegaConf from YAML + CLI overrides)
├── build_model() → FRAMEWORK_REGISTRY[config.framework.name]
│   └── e.g. QwenOFT: Qwen VLM + DiT action head
├── prepare_data() → LeRobotMixtureDataset
│   ├── For each of 50 tasks:
│   │   └── LeRobotSingleDataset(path, AgilexDataConfig)
│   └── Wraps in mixture with equal weights
├── setup_optimizer_and_scheduler()
│   ├── AdamW with per-module LR groups
│   └── Cosine scheduler with warmup
└── VLATrainer.train()
    └── for each step:
        ├── sample batch (images, language, actions)
        ├── model.forward(batch) → action_loss
        ├── backward + gradient clipping
        ├── optimizer.step()
        └── save checkpoint periodically
```

### 9.5 Action Dimension Reordering at Eval Time

Important: The training data stores actions as:
```
[left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]  # indices 0-13
```

But the RoboTwin simulator expects a different ordering. The eval interface at `examples/Robotwin/eval_files/model2robotwin_interface.py` (line 122) applies a reorder:
```python
current_action = current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]
```

This moves left_gripper from index 6 to after right_joints.

---

## 10. Troubleshooting

### Kempner: GLIBC and Flash Attention

**Symptom:**
```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
  (required by .../flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so)
```

**Root cause:** The Kempner login/GPU nodes run Rocky Linux 8 with GLIBC 2.28. The `flash-attn` pip wheel was compiled against GLIBC 2.32+. This affects both `import flash_attn` and any code using `attn_implementation: flash_attention_2`.

**Solution:** Use PyTorch's built-in SDPA (Scaled Dot Product Attention) instead:
```bash
# In your training command, add:
--framework.qwenvl.attn_implementation sdpa
```

SDPA provides similar functionality (fused attention kernels) without the GLIBC dependency. Performance difference is minimal on H100.

**If you want to try flash_attn anyway:** Check your node's GLIBC version:
```bash
ldd --version | head -1
# If it says 2.32 or higher, flash_attention_2 will work
# If 2.28 (Rocky 8 default), use sdpa
```

### Kempner: QwenFast Missing FAST Tokenizer

**Symptom:**
```
HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name':
  'playground/Pretrained_models/fast'
```

**Root cause:** The `QwenFast` framework requires the FAST tokenizer from `physical-intelligence/fast` at `playground/Pretrained_models/fast/`. This is not included in the base VLM download and must be downloaded separately.

**Solution — Option A:** Download the FAST tokenizer:
```bash
huggingface-cli download physical-intelligence/fast \
  --local-dir ./playground/Pretrained_models/fast \
  --cache-dir $HF_HUB_CACHE
```

**Solution — Option B:** Use `QwenOFT` instead (recommended, no extra downloads):
```bash
--framework.name QwenOFT
```

### Kempner: DeepSpeed `nvcc` Not Found

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**Root cause:** DeepSpeed tries to find `nvcc` at `/usr/local/cuda/bin/nvcc` but Kempner uses environment modules. If `module load cuda` wasn't run, or if `CUDA_HOME` is unset/wrong, DeepSpeed can't find the CUDA compiler.

**Solution:**
```bash
module load cuda/12.2.0-fasrc01
# Verify:
which nvcc   # should print the module path, not /usr/local/cuda/bin/nvcc
```

If running from a script (not interactive), add `module load cuda/12.2.0-fasrc01` at the top.

### Kempner: `is_debug True` Hangs on Launch

**Symptom:** Training starts but rank 0 prints `Rank 0 waiting for debugger attach on port 10092...` and hangs forever, while other ranks crash from timeout.

**Root cause:** The `--is_debug True` flag enables `debugpy.wait_for_client()` which blocks until a VSCode/PyCharm debugger attaches.

**Solution:** Do **not** pass `--is_debug True` unless you have a debugger ready to attach:
```bash
# Remove from command, or explicitly set:
--is_debug False
```

### Kempner: Verified Working Configuration (2025-02-17)

The following configuration was tested end-to-end on Kempner 4×H100 80GB:

```
Framework:        QwenOFT (DiT-B action head)
VLM:              Qwen3-VL-4B-Instruct-Action
Attention:        sdpa (not flash_attention_2, due to GLIBC)
Dataset:          RoboTwin-Randomized (500 eps/task, video mode, AV1, fps=15)
GPUs:             4× H100 80GB
Batch:            4 per GPU × 4 GPUs × 2 grad_accum = 32 effective
DeepSpeed:        ZeRO Stage 2, BF16
GPU Memory:       ~17 GB peak per GPU (plenty of headroom)
Training speed:   ~0.33s per step after warmup
Result:           10 steps completed, loss 0.87 → 0.84
```

### `NotImplementedError: Framework QwenXXX is not implemented`
The framework file wasn't imported. Run the standalone test first:
```bash
python starVLA/model/framework/QwenGR00T.py
```

### CUDA Out of Memory
Reduce batch size:
```bash
--datasets.vla_data.per_device_batch_size 4  # Default is 16, script uses 8
```

### Missing modality.json
```
FileNotFoundError: .../meta/modality.json
```
Copy from the template:
```bash
cp examples/Robotwin/train_files/modality.json playground/Datasets/RoboTwin/<task>/meta/
```

### NCCL Timeout (Multi-GPU)
```bash
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=10000
export NCCL_SOCKET_TIMEOUT_MS=360000
```

### Dataset Not Found
Ensure the path structure matches exactly:
```bash
ls playground/Datasets/RoboTwin/adjust_bottle/meta/
# Should see: modality.json, episodes.jsonl, tasks.jsonl, info.json
```

### Video Backend Error
If `torchvision_av` fails, try the default:
```bash
--datasets.vla_data.video_backend decord
```

### WandB Not Logging
Remove the disable line in the training script:
```bash
# Remove or comment out:
# export WANDB_MODE=disabled
```

### Home Directory Filling Up
Something is writing to `~/.cache/`. Check and redirect:
```bash
# Find the offender
du -sh ~/.cache/huggingface/ ~/.cache/pip/ ~/.conda/pkgs/ 2>/dev/null

# Verify env vars are set (all should point to lab storage, not ~)
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "PIP_CACHE_DIR=$PIP_CACHE_DIR"
echo "CONDA_PKGS_DIRS=$CONDA_PKGS_DIRS"

# If any is empty or points to ~, re-source bashrc:
source ~/.bashrc-kaiwen

# Double-check with Python
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"
# Must print: /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/hub

# If old cache exists under ~, you can safely symlink or remove it:
# rm -rf ~/.cache/huggingface  # after confirming env vars are set
```

---

## Quick Reference: File Locations

| What | Path |
|------|------|
| Training script | `examples/Robotwin/train_files/run_robotwin_train.sh` |
| Training YAML config | `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` |
| Modality mapping | `examples/Robotwin/train_files/modality.json` |
| Data mixture definition | `starVLA/dataloader/gr00t_lerobot/mixtures.py` (line 90) |
| Robot data config | `starVLA/dataloader/gr00t_lerobot/data_config.py` (AgilexDataConfig, line 829) |
| Dataset loader | `starVLA/dataloader/lerobot_datasets.py` |
| Training entry point | `starVLA/training/train_starvla.py` |
| DeepSpeed config | `starVLA/config/deepseeds/deepspeed_zero2.yaml` + `ds_config.yaml` |
| Eval server | `examples/Robotwin/eval_files/run_policy_server.sh` |
| Eval client | `examples/Robotwin/eval_files/eval.sh` |
| Eval interface | `examples/Robotwin/eval_files/model2robotwin_interface.py` |
| Eval config | `examples/Robotwin/eval_files/deploy_policy.yml` |
| Policy server | `deployment/model_server/server_policy.py` |
| Config/stats loader | `starVLA/model/tools.py` (`read_mode_config()`) |
| Pretrained models | `playground/Pretrained_models/` |
| Dataset root | `playground/Datasets/RoboTwin/` |
| Checkpoints output | `results/Checkpoints/<run_id>/` |
