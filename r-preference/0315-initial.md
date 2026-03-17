# StarVLA End-to-End Guide: Data Processing → Training → Evaluation

> **Framework:** QwenPI (Qwen2.5-VL-3B-Instruct-Action + DiT-B flow-matching action expert)
> **Cluster:** Kempner HPC (Harvard)
> **Date:** 2026-03-15

This document consolidates the entire pipeline for training a StarVLA model on custom datasets using the QwenPI framework (π₀-style). It covers raw data processing (via the ar-research-kempner repo), dataset preparation for StarVLA, training on Kempner, checkpoint management, monitoring, and open-loop evaluation.

---

## 0. Quick Reference

### Key Paths

| Item | Path |
|------|------|
| StarVLA repo | `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA` |
| ar-research-kempner repo | `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner` |
| Lab root | `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen` |
| HF cache | `$LAB_ROOT/.cache/huggingface` |
| LeRobot cache | `$LAB_ROOT/.cache/huggingface/lerobot/` |
| Pretrained models | `starVLA/playground/Pretrained_models/` |
| Datasets | `starVLA/playground/Datasets/Custom/` |
| Training output | `starVLA/results/Checkpoints/` |
| SLURM logs | `starVLA/logs/` |

### Software Versions

| Component | Version |
|-----------|---------|
| Conda env | `starVLA` (lab miniforge) |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| DeepSpeed | 0.16.9 |
| Accelerate | 1.5.2 |
| Attention | SDPA (flash-attn blocked by GLIBC 2.28 on Kempner) |

### Document Map

| Section | What it covers |
|---------|---------------|
| [1. Environment Setup](#1-environment-setup) | Conda, env vars, CUDA, attention backend |
| [2. Data Processing](#2-data-processing-ar-research-kempner) | Raw data → HDF5 → merged LeRobot dataset |
| [3. Data Preparation for StarVLA](#3-data-preparation-for-starvla) | Merged LeRobot → per-task dirs, mixtures registration |
| [4. Training](#4-training) | Training command, parameters, SLURM, DeepSpeed |
| [5. Checkpoint Management & Resume](#5-checkpoint-management--resume) | Checkpoint structure, preemption recovery |
| [6. Monitoring](#6-monitoring) | SLURM status, logs, WandB, GPU monitoring |
| [7. Evaluation (Open-Loop)](#7-evaluation-open-loop) | MSE/L1 metrics, single checkpoint & sweep |
| [8. Quick Debugging](#8-quick-debugging) | Single-task SLURM test, interactive salloc test |
| [9. File Reference](#9-file-reference) | All relevant files across both repos |

---

## 1. Environment Setup

### 1.1 Conda Environment

```bash
# Clone and install (first-time only)
git clone https://github.com/starVLA/starVLA
cd starVLA
conda create -n starVLA python=3.10 -y
conda activate starVLA
pip install -r requirements.txt
pip install -e .
```

### 1.2 Environment Variables (`~/.bashrc-kaiwen`)

All cache and storage paths are defined in `~/.bashrc-kaiwen`. SLURM scripts load them via `source ~/.bashrc-kaiwen`.

```bash
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
HF_HOME=$LAB_ROOT/.cache/huggingface
HF_HUB_CACHE=$HF_HOME/hub
TRANSFORMERS_CACHE=$HF_HOME/transformers
HF_DATASETS_CACHE=$HF_HOME/datasets
CONDA_PKGS_DIRS=$LAB_ROOT/.conda/pkgs
PIP_CACHE_DIR=$LAB_ROOT/.cache/pip
WANDB_DIR=$LAB_ROOT/.cache/wandb
TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```

> `~/.bashrc-kaiwen` also switches PATH from the home-directory miniforge to the lab miniforge installation and initializes conda. It must be sourced **before** `conda activate`.

### 1.3 CUDA Module

DeepSpeed needs `nvcc` to detect the CUDA version:

```bash
module load cuda/12.2.0-fasrc01
```

Without this, you get:
```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

### 1.4 SDPA vs Flash-Attention

Kempner compute nodes run **Rocky Linux 8** with **GLIBC 2.28**:

```bash
ldd --version | head -1
# ldd (GNU libc) 2.28
```

The `flash-attn` pip wheel requires GLIBC >= 2.32, so `flash_attention_2` cannot be used. Always specify SDPA:

```bash
--framework.qwenvl.attn_implementation sdpa
```

Performance difference vs flash-attn is small on H100/H200.

### 1.5 Verification

```bash
# Check HF cache location
python -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)"

# Check env vars
echo "HF_HOME: $HF_HOME"
echo "HF_HUB_CACHE: $HF_HUB_CACHE"

# Check framework loads
python -c "from starVLA.model.framework.QwenPI import QwenPI; print('QwenPI OK')"
```

---

## 2. Data Processing (ar-research-kempner)

This section uses the `ar-research-kempner` repo to convert raw custom data into a merged LeRobot dataset. The pipeline runs on the Kempner cluster.

> **Repo:** `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner`

### 2.1 Pipeline Overview

```
raw tar.gz → extract → HDF5 → merge to training_data/ → generate merged LeRobot repo
```

Output: a merged LeRobot dataset at `$HF_LEROBOT_HOME/<repo_name>/` (e.g., `custom_v0225_v3_repo/`).

### 2.2 Terminal Setup

```bash
conda deactivate
export KEMPNER_BASE="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
cd ${KEMPNER_BASE}/ar-research-kempner
source policy/pi05/Kempner_env.sh
cd ${KEMPNER_REPO}/policy/pi05
```

### 2.3 Steps (using v0225-v3 as example)

#### Step 1: Extract raw data

```bash
bash ${KEMPNER_REPO}/script/Kempner_extract_custom_hv_lv.sh
```

Extracts tar.gz files, flattens double-nested directory structure. Verify:

```bash
ls ${KEMPNER_REPO}/data_custom/ | grep -E '_(hv|lv)$' | wc -l   # expect 10
ls ${KEMPNER_REPO}/data_custom/adjust_bottle_hv/data/ | wc -l    # expect 100 episodes
```

#### Step 2: Process raw data to HDF5

```bash
bash Kempner_process_custom_hv_lv_data.sh
```

Creates symlinks for `process_data.py` compatibility, processes raw data to HDF5, moves results to `training_data/custom_hv_lv/`. Verify:

```bash
ls training_data/custom_hv_lv/ | wc -l   # expect 10
```

#### Step 3: Prepare training_data directory

```bash
bash Kempner_prepare_custom_v0225_v3.sh
```

Hard-links clean variants from `training_data/custom_all/` and hv/lv from `training_data/custom_hv_lv/` into `training_data/custom_v0225_v3/` (20 directories total, zero extra disk space for clean data). Verify:

```bash
ls training_data/custom_v0225_v3/ | wc -l   # expect 20
```

#### Step 4: Generate merged LeRobot dataset

**Must run on a GPU node (salloc session)** — uses `/scratch` SSD for fast writes.

```bash
NUM_WORKERS=64 bash Kempner_generate_custom_v0225_v3.sh
```

Writes to local `/scratch` SSD, then rsyncs to isilon. The final dataset appears at:
`${KEMPNER_BASE}/.cache/huggingface/lerobot/custom_v0225_v3_repo/`

> Must complete generation + rsync within the same salloc session (local scratch is lost on node change).

Verify:

```bash
CACHE="${KEMPNER_BASE}/.cache"
cat ${CACHE}/huggingface/lerobot/custom_v0225_v3_repo/meta/info.json | \
  python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Episodes: {d[\"total_episodes\"]}, Frames: {d[\"total_frames\"]}')"
# Episodes: 2000
```

#### Step 5: Compute normalization stats (pi0/pi05 only)

```bash
bash Kempner_compute_norm_stats.sh pi05_aloha_robotwin_custom_v0225_v3_full --max-frames 25000
```

> For StarVLA/QwenPI training, normalization stats are computed automatically by the dataloader on first run (`stats_gr00t.json`). This step is only needed for pi0/pi05 training in ar-research-kempner.

### 2.4 Adapting for a New Dataset

<!-- SWAP-DATASET-MARKER -->

To use a different dataset, modify:

1. **Step 1:** Change the extraction script to point to your new tar.gz files
2. **Step 2:** Change the task list and episode count in the processing script
3. **Step 3:** Adjust the task variant assignments (which tasks get which variants)
4. **Step 4:** Update `DATA_DIR` and `REPO_ID` in the generation script
5. **In StarVLA:** Follow Section 3 to split the new merged dataset and register it

The pipeline structure stays the same — only paths and task lists change.

---

## 3. Data Preparation for StarVLA

This section converts the merged LeRobot dataset (output of Section 2) into the per-task directory structure that StarVLA requires.

### 3.1 Why Per-Task Splitting is Needed

StarVLA loads datasets on a **per-task** basis — each task variant must be its own LeRobot dataset directory. The merged dataset from ar-research-kempner combines all task variants into one directory. The split is a file-level operation (no re-downloading, no re-processing).

### 3.2 Split the Merged Dataset

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# For custom_all (33 variants):
bash scripts/split_custom_all.sh

# For v0225_v3 (20 variants):
bash scripts/split_custom_v0225_v3.sh

# With explicit worker count:
bash scripts/split_custom_v0225_v3.sh 8

# Sequential (debugging):
bash scripts/split_custom_all.sh 1
```

The shell wrapper calls `scripts/split_custom_lerobot.py`, which:
1. Reads `meta/episodes.jsonl` and `tasks.jsonl` from the merged repo
2. Splits episodes into groups (one per task variant)
3. Processes all tasks in parallel (one worker per task)
4. Creates per-task LeRobot directories with renumbered metadata (`episode_index`, `index`, `task_index`)
5. Adds `modality.json` to each task directory

### 3.3 Output Structure

```
playground/Datasets/Custom/
├── adjust_bottle/              (100 episodes)
│   ├── meta/
│   │   ├── info.json
│   │   ├── episodes.jsonl
│   │   ├── tasks.jsonl
│   │   └── modality.json
│   └── data/
│       └── chunk-000/
│           ├── episode_000000.parquet
│           └── ... (100 files)
├── adjust_bottle_hv/           (100 episodes)
├── adjust_bottle_lv/           (100 episodes)
├── beat_block_hammer/
├── ...
└── place_empty_cup/
```

### 3.4 LeRobot Format Requirements for StarVLA

Each per-task directory must contain:

#### meta/info.json (required fields)

```jsonc
{
  "codebase_version": "v2.1",
  "robot_type": "robotwin",
  "total_episodes": 100,
  "total_frames": 26543,
  "total_tasks": 1,
  "total_videos": 0,                // 0 = image-in-parquet; >0 = video mode
  "total_chunks": 1,
  "chunks_size": 1000,
  "fps": 50,
  "splits": { "train": "0:100" },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "features": { /* ... */ }
}
```

#### meta/modality.json (required)

Maps modality names to parquet column names and index ranges:

```json
{
  "action": {
    "left_joints":  { "start": 0,  "end": 6,  "original_key": "action" },
    "left_gripper": { "start": 6,  "end": 7,  "original_key": "action" },
    "right_joints": { "start": 7,  "end": 13, "original_key": "action" },
    "right_gripper":{ "start": 13, "end": 14, "original_key": "action" }
  },
  "state": {
    "left_joints":  { "start": 0,  "end": 6,  "original_key": "observation.state" },
    "left_gripper": { "start": 6,  "end": 7,  "original_key": "observation.state" },
    "right_joints": { "start": 7,  "end": 13, "original_key": "observation.state" },
    "right_gripper":{ "start": 13, "end": 14, "original_key": "observation.state" }
  },
  "video": {
    "cam_high":       { "original_key": "observation.images.cam_high" },
    "cam_left_wrist": { "original_key": "observation.images.cam_left_wrist" },
    "cam_right_wrist":{ "original_key": "observation.images.cam_right_wrist" }
  },
  "annotation": {
    "human.action.task_description": { "original_key": "task_index" }
  }
}
```

#### meta/episodes.jsonl

```json
{"episode_index": 0, "length": 265, "tasks": ["pick up the red block"]}
{"episode_index": 1, "length": 312, "tasks": ["pick up the red block"]}
```

Requirements: `episode_index` must be sequential 0..N-1, `length` must match parquet row count.

#### meta/tasks.jsonl

```json
{"task_index": 0, "task": "pick up the red block and place it on the blue pad"}
```

Every `task_index` in parquet must have an entry here.

#### Parquet files

Path: `data/chunk-{chunk:03d}/episode_{idx:06d}.parquet`

Required columns: `timestamp`, `frame_index`, `episode_index`, `task_index`, `index`, `observation.state`, `action`.

For image-in-parquet mode: image columns (e.g., `observation.images.cam_high`) contain `{"bytes": <PNG bytes>}` dicts.

#### Image-in-Parquet vs Video Detection

| `total_videos` | Mode | Images stored in |
|---|---|---|
| `0` | Image-in-parquet | Parquet columns as `{"bytes": <png_bytes>}` dicts |
| `> 0` | Video | Separate MP4 files under `videos/` |

The dataloader auto-detects based on `info.json["total_videos"]`.

### 3.5 Register in mixtures.py

Add your dataset mixture to `starVLA/dataloader/gr00t_lerobot/mixtures.py`:

```python
DATASET_NAMED_MIXTURES = {
    # ... existing entries ...

    "my_dataset": [
        ("task_folder_name_1", 1.0, "robotwin"),
        ("task_folder_name_2", 1.0, "robotwin"),
        # ...
    ],

    # Single-task debug subset
    "my_dataset_task1": [
        ("task_folder_name_1", 1.0, "robotwin"),
    ],
}
```

Each tuple: `(directory_name, sampling_weight, robot_type)`

- `directory_name`: folder name under `data_root_dir` (not a full path)
- `sampling_weight`: relative probability (all weights are normalized)
- `robot_type`: key in `ROBOT_TYPE_CONFIG_MAP` in `data_config.py`

**Why `robot_type = "robotwin"` for custom datasets:** The custom data uses the same robot configuration as RoboTwin (dual-arm, 14-dim state/action, 3 cameras). `"robotwin"` maps to `AgilexDataConfig` in `data_config.py`, which defines the correct modality keys, normalization modes, and camera names.

### 3.6 Register Robot Config (if new robot type)

If your robot has different dimensions or cameras, create a new config class in `starVLA/dataloader/gr00t_lerobot/data_config.py` and register it in `ROBOT_TYPE_CONFIG_MAP`. If the `robot_type` is not in `ROBOT_TYPE_TO_EMBODIMENT_TAG`, it defaults to `EmbodimentTag.NEW_EMBODIMENT` (projector index 31) — this is fine for custom robots.

### 3.7 Verification

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Check all task directories have modality.json
for d in playground/Datasets/Custom/*/; do
  [ ! -f "$d/meta/modality.json" ] && echo "MISSING: $d"
done

# Check info.json fields
TASK_DIR=playground/Datasets/Custom/adjust_bottle
python3 -c "
import json, sys
info = json.load(open('$TASK_DIR/meta/info.json'))
required = ['codebase_version','total_episodes','total_frames','total_videos',
            'total_chunks','chunks_size','fps','splits','data_path','features']
missing = [k for k in required if k not in info]
if missing:
    print(f'FAIL: missing fields: {missing}', file=sys.stderr); sys.exit(1)
print(f'OK: {info[\"total_episodes\"]} episodes, {info[\"total_frames\"]} frames, fps={info[\"fps\"]}')
"

# Test image dataset loading (all 5 tests should PASS)
python test_image_dataset.py
```

---

## 4. Training

### 4.1 Full Training Command

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 500000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_qwenPI_requeue \
  --wandb_project starVLA_Custom \
  --wandb_entity kaiwenh-17-uiuc
```

### 4.2 Parameter Tables

#### Framework & VLM

| Parameter | Value | Description |
|-----------|-------|-------------|
| `framework.name` | `QwenPI` | π₀-style flow-matching framework |
| `framework.qwenvl.base_vlm` | `playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action` | Pretrained VLM with action tokens (3B) |
| `framework.qwenvl.attn_implementation` | `sdpa` | Required on Kempner (GLIBC 2.28); use `flash_attention_2` elsewhere |

#### Action Model (DiT-B Flow-Matching Expert)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `action_dim` | 14 | Action dimensions (14 for dual-arm: 6+1+6+1) |
| `state_dim` | 14 | State dimensions (matches action_dim) |
| `future_action_window_size` | 15 | Future steps predicted (chunk = 15 + 1 = 16) |
| `past_action_window_size` | 0 | Past action context (none) |
| `action_hidden_dim` | 1024 | DiT hidden dimension |
| `hidden_size` | 1024 | Internal hidden size |
| `action_model_type` | `DiT-B` | DiT-Base architecture for action expert |
| `add_pos_embed` | True | Add positional embeddings to action sequences |
| `max_seq_len` | 1024 | Maximum sequence length |
| `noise_beta_alpha` | 1.5 | Beta distribution alpha for noise schedule |
| `noise_beta_beta` | 1.0 | Beta distribution beta for noise schedule |
| `noise_s` | 0.999 | Noise schedule parameter |
| `num_timestep_buckets` | 1000 | Number of diffusion timestep buckets |
| `num_inference_timesteps` | 4 | Denoising steps at inference (fewer = faster) |
| `num_target_vision_tokens` | 32 | Vision token count passed to action expert |

#### Dataset & Training

| Parameter | Value | Description |
|-----------|-------|-------------|
| `data_root_dir` | `playground/Datasets/Custom` | Parent dir of per-task LeRobot dirs |
| `data_mix` | `custom_all` | Mixture name in `mixtures.py` |
| `include_state` | true | Feed state observations to the action model |
| `per_device_batch_size` | 8 | Per GPU batch size |
| `gradient_accumulation_steps` | 2 | 4 GPU x 8 batch x 2 accum = **64 effective batch** |
| `max_train_steps` | 500,000 | Total training steps |
| `save_interval` | 50,000 | Checkpoint every 50K steps |
| `eval_interval` | 5,000 | Evaluation every 5K steps |
| `logging_frequency` | 100 | Log to WandB every 100 steps |
| `is_resume` | true | Auto-resume from latest checkpoint on preemption |
| `freeze_modules` | `''` (empty) | Train all parameters (full fine-tune) |
| `run_id` | `custom_qwenPI_requeue` | Unique identifier (same run_id = resume; different = fresh) |

#### Learning Rates (from base YAML `starvla_train_calvin.yaml`)

QwenPI uses separate learning rates for different model components:

| Component | Learning Rate | Description |
|-----------|--------------|-------------|
| `base` | 2.5e-05 | Default LR for unlisted parameters |
| `qwen_vl_interface` | 1.0e-05 | VLM backbone (lower to preserve pretrained features) |
| `action_model` | 1.0e-04 | DiT action expert (higher, trained from scratch) |

Scheduler: `cosine_with_min_lr` (min_lr: 1.0e-06), warmup ratio: 0.1.

### 4.3 SLURM Script

Full script: `scripts/slurm_custom_requeue.sh` (update the training command inside to use QwenPI as shown in §4.1).

#### SBATCH Headers

```bash
#SBATCH --partition=kempner_requeue     # 7-day limit, preemptable
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h200              # H200 80GB (change to h100 if needed)
#SBATCH --requeue                      # Auto re-queue after preemption
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64             # 64 for H200, 96 for H100
#SBATCH --mem=1440G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/starVLA_custom_%j.out
#SBATCH --error=logs/starVLA_custom_%j.err
```

#### Environment Setup Pattern

Every SLURM script must follow this sequence:

```bash
# ── Environment setup ──
set +u                          # disable "unbound variable" check (bashrc uses PROMPT_COMMAND)
source ~/.bashrc-kaiwen         # load env vars, switch to lab miniforge, init conda
set -euo pipefail               # re-enable strict mode

module load cuda/12.2.0-fasrc01 # DeepSpeed needs nvcc
conda activate starVLA          # activate the training environment

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
```

The `set +u` / `set -euo pipefail` sandwich is required because `~/.bashrc-kaiwen` references `PROMPT_COMMAND`, which is undefined in a SLURM batch context.

#### GPU Type Switching

When switching GPU type, change these three SLURM parameters together:

| Parameter | H100 | H200 | A100 |
|-----------|------|------|------|
| `--constraint` | h100 | h200 | a100 |
| `--cpus-per-task` | 96 | 64 | 64 |
| `--mem` | 1440G | 1440G | 960G |

All other SLURM parameters stay the same. A100 has 40GB VRAM (may not be enough for larger models).

#### Partition Comparison

| | kempner_h100 | kempner_requeue |
|---|---|---|
| Max time | 3 days | 7 days |
| Preemption | No | Yes |
| GPU types | H100 only | Mixed (H100 / H200 / A100) |
| Queue wait | Longer | Shorter |
| Use case | Stable critical runs | Long training, flexible scheduling |

#### Submit

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 1 node x 4 GPU
sbatch scripts/slurm_custom_requeue.sh

# 2 nodes x 8 GPU (see multi-node notes in DOC-training/custom-all.md §9)
sbatch scripts/slurm_custom_requeue-2node.sh
```

### 4.4 DeepSpeed ZeRO-2 Config

Accelerate config: `starVLA/config/deepseeds/deepspeed_zero2.yaml`

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_config_file: "./starVLA/config/deepseeds/ds_config.yaml"
  deepspeed_multinode_launcher: standard
  zero3_init_flag: false
distributed_type: DEEPSPEED
num_machines: 1
num_processes: 8
```

DeepSpeed config: `starVLA/config/deepseeds/ds_config.yaml`

```json
{
    "bf16": { "enabled": true },
    "fp16": { "enabled": false },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": 1,
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": true,
        "allgather_bucket_size": 5e8,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "overlap_comm": true,
        "contiguous_gradients": true,
        "cpu_offload": false
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 10
}
```

### 4.5 Image-in-Parquet Patches

Custom datasets use image-in-parquet format (`dtype: image`, `total_videos: 0`). Three patches to `starVLA/dataloader/gr00t_lerobot/datasets.py` are required (already applied):

| Patch | Location | Symptom Without It |
|-------|----------|-------------------|
| 1. `_get_metadata()` | line ~624 | `ValueError: 'channel' is not in list` / `KeyError: 'info'` |
| 2. `get_video()` | line ~1187 | `FileNotFoundError` on non-existent `.mp4` files |
| 3. `__getitem__()` | line ~2094 | Training hangs at 0% (infinite `while True` loop) |

**Patch 1:** Image datasets use `"channels"` (plural) in names array and `fps` at top level. Added third fallback branch in `_get_metadata()`.

**Patch 2:** Added `is_image_dataset` property and `_get_images_from_parquet()` method. Image datasets decode PNG from parquet instead of reading MP4 files.

**Patch 3:** The `while True` loop in `LeRobotMixtureDataset.__getitem__()` skips `os.path.exists(video_path)` for image datasets.

Verify patches work:
```bash
python test_image_dataset.py   # Tests 0-4 should all PASS
```

See `DOC-data/custom-dataset-split.md` for full patch code.

---

## 5. Checkpoint Management & Resume

### 5.1 Checkpoint Directory Structure

```
results/Checkpoints/<run_id>/
├── config.yaml                        # Complete training config
├── dataset_statistics.json            # Action/state normalization stats
├── checkpoints/
│   ├── steps_50000/                   # Full training state (model, optimizer, scheduler, RNG)
│   ├── steps_50000_pytorch_model.pt   # Standalone model weights (for deployment/eval)
│   ├── steps_100000/
│   ├── steps_100000_pytorch_model.pt
│   └── ...
```

### 5.2 Resume Logic

When `is_resume=true`, the trainer:
1. Scans `checkpoints/` for `steps_<N>/` directories
2. Picks the highest `N` (via `_get_latest_checkpoint()`)
3. Calls `accelerator.load_state()` to restore model weights, optimizer state, scheduler, and RNG state

Use the **same `--run_id`** to continue training; use a **different `run_id`** to start fresh.

### 5.3 Preemption Recovery Flow

On `kempner_requeue`:

1. Job preempted → SLURM auto re-queues (`#SBATCH --requeue`, status shows `PR` → `PD`)
2. Job re-scheduled on a new node
3. `--trainer.is_resume true` → auto-loads latest checkpoint
4. Training continues from last checkpoint

Training progress between the last checkpoint and the preemption event is **lost**.

On `kempner_h100` (no preemption but 3-day time limit): same `is_resume=true` logic applies — resubmit the same script and training continues.

### 5.4 `save_interval` Tradeoff

| `save_interval` | Max steps lost on preemption | Storage overhead |
|---|---|---|
| 50,000 | Up to 50K steps | Lower |
| 25,000 | Up to 25K steps | Medium |
| 10,000 | Up to 10K steps | Higher |

If preemptions are frequent, reduce to 10K-25K.

### 5.5 Standalone Weights

For deployment or evaluation, use `steps_<N>_pytorch_model.pt` — model weights only, no optimizer state. Much smaller than the full `steps_<N>/` directory.

---

## 6. Monitoring

### 6.1 SLURM Job Status

```bash
squeue -u $USER                        # list all your jobs
squeue --start -j <jobid>              # estimated start time for pending job
squeue -p kempner_requeue | wc -l      # count jobs in requeue partition
scancel <jobid>                        # cancel a job
```

### 6.2 Queue Status Meanings

| Status | Meaning | Action |
|--------|---------|--------|
| `PD (Resources)` | Waiting for free GPUs | Usually starts soon |
| `PD (Priority)` | Higher-priority jobs ahead | May wait longer |
| `PD (ReqNodeNotAvail)` | Selected nodes are down/draining | Cancel and resubmit |
| `R` | Running | -- |
| `PR` | Preempted | Auto re-queued by `--requeue` |

### 6.3 Training Logs

```bash
ls logs/starVLA_*.out                      # list all log files
tail -f logs/starVLA_custom_<jobid>.out    # live training output
tail -f logs/starVLA_custom_<jobid>.err    # live errors/warnings
```

If a job fails, check `.err` first for Python tracebacks.

### 6.4 Resource Availability

```bash
# GPU status per type (idle/total/queued)
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h200   # H200 only
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h100   # H100 only

# All partitions overview
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/most_empty_partition.sh

# Raw node status
sinfo -p kempner_requeue -o "%N %G %t %f" | grep -E "idle|mix"
```

### 6.5 WandB Setup

WandB is configured via training command flags:

```bash
--wandb_project starVLA_Custom \
--wandb_entity kaiwenh-17-uiuc
```

The `WANDB_DIR` environment variable (from `~/.bashrc-kaiwen`) directs WandB cache to lab storage.

### 6.6 GPU Monitoring (on compute node)

```bash
# Open interactive shell on running job's node
srun --jobid=<jobid> --pty bash

# Check GPU utilization
nvidia-smi
watch -n 5 nvidia-smi
```

---

## 7. Evaluation (Open-Loop)

Open-loop evaluation measures how well the trained model fits the training data by comparing predicted actions against ground-truth actions.

### 7.1 Prerequisites

- conda env `starVLA` activated
- At least 1 GPU with ~20 GB VRAM (single-GPU inference)
- Training completed with checkpoints

### 7.2 Environment Setup

```bash
export REPO_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
export LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen

export HF_HOME=${LAB_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

source "${LAB_ROOT}/miniforge3/etc/profile.d/conda.sh"
conda activate starVLA
module load cuda/12.2.0-fasrc01 2>/dev/null || true

cd "${REPO_DIR}"
```

### 7.3 Running Evaluation

```bash
# Single checkpoint (default: 200 samples)
python realworld/eval_openloop.py \
  --checkpoint results/Checkpoints/<run_id>/checkpoints/steps_100000_pytorch_model.pt \
  --num_samples 200

# Evaluate ALL checkpoints (learning curve sweep)
python realworld/eval_openloop.py --sweep --num_samples 100

# Custom output path
python realworld/eval_openloop.py \
  --checkpoint <path_to_checkpoint.pt> \
  --output results/my_eval_results.json
```

### 7.4 Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Overall MSE** | Mean squared error between predicted and GT **normalized** actions (range ~[-1,1]) |
| **Overall L1** | Mean absolute error (same space) |
| **Per-dimension** | Breakdown by each action dimension (e.g., pos_x, pos_y, rot6d_0, gripper) |
| **Per-group** | Aggregated by position, rotation, gripper |
| **Per-step MSE** | Error at each step in the action horizon; expect increasing error for later steps |

### 7.5 Interpreting Results

Since actions are normalized to [-1, 1]:

| MSE Range | L1 Range | Interpretation |
|-----------|----------|---------------|
| < 0.01 | < 0.05 | Model fits training data well |
| 0.01 - 0.1 | 0.05 - 0.2 | Partial learning, check per-dimension breakdown |
| > 0.1 | > 0.2 | Model hasn't learned the task |

- Per-step error should increase gradually; if step 0 is already high, the model is fundamentally not working
- **This evaluates on training data** — low error is necessary but not sufficient for deployment
- Metrics are in **normalized space** (both GT and predictions are normalized by the data pipeline)

### 7.6 Sweep Mode Output

Sweep mode evaluates all `steps_<N>_pytorch_model.pt` files, prints a summary table, and saves `openloop_eval_sweep.json`:

```
       Checkpoint         MSE          L1     MSE_pos    MSE_rot
----------------------------------------------------------------------
      steps_10000    0.045123    0.152340    0.032100    0.051200
      steps_50000    0.012456    0.078900    0.008300    0.014500
     steps_100000    0.005678    0.045230    0.003200    0.006800
```

---

## 8. Quick Debugging

### 8.1 Single-Task Test via SLURM

Modify the SLURM script's training command:

```bash
  --datasets.vla_data.data_mix custom_task1 \    # only loads adjust_bottle
  --trainer.max_train_steps 100 \                 # only 100 steps
  --trainer.save_interval 50 \
  --run_id custom_debug_test \
```

Then submit:
```bash
sbatch scripts/slurm_custom_requeue.sh
```

### 8.2 Interactive Test via salloc

```bash
# Get an interactive node
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h100 \
  -N 1 --gpus-per-node=4 --cpus-per-task=96 --mem=1440G -t 0-01:00:00

# Setup environment
source ~/.bashrc-kaiwen && conda activate starVLA && module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Quick 20-step test with single task
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/calvin/train_files/starvla_train_calvin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 14 \
  --framework.action_model.state_dim 14 \
  --framework.action_model.future_action_window_size 15 \
  --framework.action_model.past_action_window_size 0 \
  --framework.action_model.action_hidden_dim 1024 \
  --framework.action_model.hidden_size 1024 \
  --framework.action_model.action_model_type DiT-B \
  --framework.action_model.add_pos_embed True \
  --framework.action_model.max_seq_len 1024 \
  --framework.action_model.noise_beta_alpha 1.5 \
  --framework.action_model.noise_beta_beta 1.0 \
  --framework.action_model.noise_s 0.999 \
  --framework.action_model.num_timestep_buckets 1000 \
  --framework.action_model.num_inference_timesteps 4 \
  --framework.action_model.num_target_vision_tokens 32 \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_task1 \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_qwenPI_interactive_test
```

### 8.3 Common Issues

| Problem | Symptom | Fix |
|---------|---------|-----|
| Missing `modality.json` | `FileNotFoundError: .../meta/modality.json` | Copy from `examples/Robotwin/train_files/modality.json` |
| GLIBC flash-attn | `ImportError` or segfault | Use `--framework.qwenvl.attn_implementation sdpa` |
| DeepSpeed nvcc not found | `FileNotFoundError: .../nvcc` | `module load cuda/12.2.0-fasrc01` |
| PROMPT_COMMAND unbound | `.bashrc-kaiwen: line 14: PROMPT_COMMAND: unbound variable` | Wrap source in `set +u` / `set -euo pipefail` |
| Training hangs at 0% | Progress bar stuck, no errors | Image-in-parquet Patch 3 not applied (see §4.5) |
| `ReqNodeNotAvail` | Job stuck in PD forever | `scancel <jobid>` and resubmit |
| Wrong modality names | `KeyError` during data loading | Ensure names in `modality.json` match `data_config.py` keys |
| Non-sequential episode indices | Index out of range errors | Re-index episodes to 0..N-1 with no gaps |

---

## 9. File Reference

### StarVLA Repository

| File | Purpose |
|------|---------|
| `scripts/slurm_custom_requeue.sh` | SLURM script (1 node x 4 GPU, kempner_requeue) |
| `scripts/slurm_custom_requeue-2node.sh` | SLURM script (2 nodes x 8 GPU) |
| `scripts/slurm_custom_v0225_v3_requeue.sh` | SLURM script for v0225_v3 dataset |
| `scripts/split_custom_lerobot.py` | Python: splits merged LeRobot → per-task dirs |
| `scripts/split_custom_all.sh` | Shell wrapper: split custom_all (33 tasks) |
| `scripts/split_custom_v0225_v3.sh` | Shell wrapper: split v0225_v3 (20 tasks) |
| `starVLA/training/train_starvla.py` | Training loop entry point |
| `starVLA/training/trainer_utils/trainer_tools.py` | Checkpoint save/resume logic |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset classes (patched for image-in-parquet) |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | Dataset mixture definitions |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | Robot type configs, transforms, normalization |
| `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py` | Robot-to-embodiment-projector mapping |
| `starVLA/dataloader/lerobot_datasets.py` | Top-level dataset entry point |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 (1 node) |
| `starVLA/config/deepseeds/deepspeed_zero2_2node.yaml` | Accelerate + DeepSpeed ZeRO-2 (2 nodes) |
| `starVLA/config/deepseeds/ds_config.yaml` | DeepSpeed engine config (ZeRO stage 2, bf16) |
| `examples/calvin/train_files/starvla_train_calvin.yaml` | Base training YAML config (QwenPI defaults) |
| `examples/Robotwin/train_files/modality.json` | Reference modality.json (14-dim dual-arm, 3 cameras) |
| `realworld/0311-train-pickandplace-qwenpi.sh` | Reference QwenPI training script (FastUMI) |
| `test_image_dataset.py` | Image-in-parquet dataset loading tests |
| `realworld/eval_openloop.py` | Open-loop evaluation script |
| `playground/Datasets/Custom/` | Per-task dataset directories |
| `playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action/` | Pretrained VLM (QwenPI) |
| `results/Checkpoints/` | Training output root |

### ar-research-kempner Repository

All paths relative to `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner/`.

| File | Purpose |
|------|---------|
| `script/Kempner_process_custom-0225-v3.md` | Data processing documentation (hv/lv pipeline) |
| `script/Kempner_extract_custom_hv_lv.sh` | Step 1: Extract hv/lv tar.gz files |
| `policy/pi05/Kempner_env.sh` | Environment variables for ar-research-kempner |
| `policy/pi05/Kempner_process_custom_hv_lv_data.sh` | Step 2: Process raw data → HDF5 |
| `policy/pi05/Kempner_prepare_custom_v0225_v3.sh` | Step 3: Merge clean + hv/lv to training_data |
| `policy/pi05/Kempner_generate_custom_v0225_v3.sh` | Step 4: Generate merged LeRobot dataset |
| `policy/pi05/Kempner_compute_norm_stats.sh` | Step 5: Compute normalization statistics |
| `policy/pi05/Kempner_finetune.sh` | Training launcher (pi0/pi05) |

### Documentation

| Document | Location | Content |
|----------|----------|---------|
| `DOC-setup/install.md` | starVLA | Installation, model downloads, dataset downloads |
| `DOC-training/kempner-cluster.md` | starVLA | Cluster setup, GPU types, partitions, environment |
| `DOC-training/custom-all.md` | starVLA | Custom dataset training (33 variants) |
| `DOC-training/custom-0225-v2.md` | starVLA | Custom v0225_v3 training (20 variants, hv/lv) |
| `DOC-training/custom-v0218.md` | starVLA | Custom v0218 training (27 variants) |
| `DOC-training/robotwin.md` | starVLA | RoboTwin training guide |
| `DOC-data/custom-dataset-split.md` | starVLA | Dataset split procedure, image-in-parquet patches |
| `DOC-data/lerobot-requiremnt-for-starvla.md` | starVLA | LeRobot format requirements checklist |
| `realworld/0311-how-to-evaluate-openloop.md` | starVLA | Open-loop evaluation guide |
