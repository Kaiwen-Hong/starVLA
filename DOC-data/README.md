# FastUMI -> StarVLA Integration Guide

Complete guide for converting FastUMI datasets (LeRobot v3.0) to StarVLA-compatible
format (LeRobot v2.1) and registering them for training.

---

## Overview

```
FastUMI Collection Pipeline          StarVLA Training Pipeline
========================          ========================

Raw SLAM (500Hz)                  data_root_dir/
    |                                 |
    v                                 v
Downsample (20Hz)                 pickandplace_vla/
    |                                 ├── meta/
    v                                 │   ├── info.json        (v2.1)
Augmentation (5x)                     │   ├── modality.json    (required)
    |                                 │   ├── episodes.jsonl
    v                                 │   └── tasks.jsonl
LeRobot v3.0 Dataset              │   ├── data/chunk-000/
(single parquet + single mp4)         │   │   ├── episode_000000.parquet
    |                                 │   │   ├── episode_000001.parquet
    | convert_v30_to_starvla.py       │   │   └── ...
    |                                 │   └── videos/chunk-000/
    v                                 │       └── observation.images.wrist/
StarVLA v2.1 Dataset                  │           ├── episode_000000.mp4
(per-episode parquet + per-episode mp4)           └── ...
```

---

## Step 1: Generate LeRobot v3.0 Dataset (FastUMI side)

If not already done, run the data collection and conversion pipeline:

```bash
cd /home/kaiwen/Desktop/research/fastumipro-collection

# 1. Preprocess SLAM data (500Hz -> 20Hz)
cd data_collector_opt
python slam_analyze_and_vel.py --category pickandplace_vla

# 2. Generate augmentation versions (5x)
python V2-slam_analyze_and_vel-augment.py --step 5

# 3. Convert to LeRobot v3.0 format
cd ..
python scripts/V2-convert_to_lerobot_cropped-augment-mp.py \
    --data-dir data_collector_opt/DATA/main_250801DR48FP25002960 \
    --category pickandplace_vla \
    --augment-versions 0 5 10 15 20 \
    --num-workers 16
```

Output: `~/.cache/huggingface/lerobot/umi-pickandplacevla-rot6d-rel-256-cropped-5x/`

---

## Step 2: Convert v3.0 -> v2.1 (This Script)

```bash
python convert_v30_to_starvla.py \
    --src ~/.cache/huggingface/lerobot/umi-pickandplacevla-rot6d-rel-256-cropped-5x \
    --dst /path/to/starvla/playground/Datasets/FastUMI/pickandplace_vla \
    --task "pick up the object and place it on the target" \
    --train-only --val-ratio 0.2 \
    --video-codec h264 \
    --num-workers 8
```

### Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--src` | (required) | Source LeRobot v3.0 dataset directory |
| `--dst` | (required) | Output directory (one per task) |
| `--task` | auto from source | Language instruction for training |
| `--train-only` | false | Exclude validation sessions (session-level split) |
| `--val-ratio` | 0.2 | Fraction of sessions held out for validation |
| `--video-codec` | h264 | Output video codec: `h264` or `av1` |
| `--num-workers` | 8 | Parallel ffmpeg workers for video splitting |

### What the Converter Does

| Step | v3.0 (source) | v2.1 (output) |
|------|--------------|---------------|
| Parquet | Single `file-000.parquet` (all episodes) | Per-episode `episode_NNNNNN.parquet` |
| Video | Single `file-000.mp4` (all episodes concatenated) | Per-episode `episode_NNNNNN.mp4` |
| info.json | `codebase_version: v3.0` | `codebase_version: v2.1` + `total_videos`, `total_chunks` |
| Tasks | `tasks.parquet` | `tasks.jsonl` |
| Episodes | `episodes/chunk-000/file-000.parquet` | `episodes.jsonl` |
| Modality | (not present) | `modality.json` (generated) |
| Image shape | `[C, H, W]` = `[3, 256, 256]` | `[H, W, C]` = `[256, 256, 3]` |
| Video path | `videos/{video_key}/chunk-{N}/file-{N}.mp4` | `videos/chunk-{N}/{video_key}/episode_{N}.mp4` |
| Data path | `data/chunk-{N}/file-{N}.parquet` | `data/chunk-{N}/episode_{N}.parquet` |

---

## Step 3: StarVLA Code Changes

Three files need modification on the StarVLA side.

### 3a. data_config.py

Add `FastUMIDataConfig` to `starVLA/dataloader/gr00t_lerobot/data_config.py`.

See `fastumi_data_config.py` in this directory for the complete class.

```python
class FastUMIDataConfig:
    video_keys = ["video.wrist"]
    state_keys = ["state.eef_pos", "state.eef_rot6d", "state.gripper"]
    action_keys = ["action.eef_pos", "action.eef_rot6d", "action.gripper"]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    # ... modality_config() and transform() methods ...
```

Then register it in `ROBOT_TYPE_CONFIG_MAP`:

```python
ROBOT_TYPE_CONFIG_MAP = {
    ...
    "fastumi": FastUMIDataConfig(),
}
```

### 3b. mixtures.py

Add to `starVLA/dataloader/gr00t_lerobot/mixtures.py`:

```python
DATASET_NAMED_MIXTURES = {
    ...
    "fastumi_pickandplace": [
        ("pickandplace_vla", 1.0, "fastumi"),
    ],
}
```

### 3c. embodiment_tags.py (Optional)

If `"fastumi"` is not in `ROBOT_TYPE_TO_EMBODIMENT_TAG`, StarVLA defaults to
`EmbodimentTag.NEW_EMBODIMENT` (projector index 31). This is fine for custom robots.
No change needed unless you want a specific projector index.

---

## Step 4: Training

```bash
cd /path/to/starVLA

python -m starVLA.train \
    --datasets.vla_data.data_root_dir playground/Datasets/FastUMI \
    --datasets.vla_data.data_mix fastumi_pickandplace \
    --datasets.vla_data.video_backend torchvision_av \
    ...
```

---

## Data Format Details

### State Representation (10D)

```
observation.state = [x, y, z, rot6d_0, rot6d_1, rot6d_2, rot6d_3, rot6d_4, rot6d_5, gripper]
                     |--pos--|  |------------- rotation_6d -------------|  |grip|
                     3D EEF     6D continuous rotation (Zhou et al. 2019)   0/1
```

- Position: absolute EEF position in robot base frame (meters)
- Rotation: first two rows of 3x3 rotation matrix, flattened (Gram-Schmidt recovery)
- Gripper: 0 = open, 1 = closed

### Action Representation (10D)

```
action = [rel_x, rel_y, rel_z, rel_rot6d(6), gripper]
          |---relative position---|  |--relative rotation--|  |grip|
```

Actions are **pre-computed as relative poses** (UMI-style):

```
relative_transform = inv(current_pose) @ next_pose
action[0:3]  = relative_transform[:3, 3]     # relative translation
action[3:9]  = mat_to_rot6d(relative_transform[:3, :3])  # relative rotation
action[9]    = target_gripper_state           # absolute gripper
```

**Important:** Since actions are already relative, StarVLA should use `action_mode="abs"`
(pass-through). The `"absolute": false` flag in `modality.json` only affects padding
behavior (pad with zeros instead of edge values).

### Modality Mapping

```
modality.json key              data_config.py key         parquet column      slice
─────────────────              ──────────────────         ──────────────      ─────
action.eef_pos                 action.eef_pos             action              [0:3]
action.eef_rot6d               action.eef_rot6d           action              [3:9]
action.gripper                 action.gripper             action              [9:10]
state.eef_pos                  state.eef_pos              observation.state   [0:3]
state.eef_rot6d                state.eef_rot6d            observation.state   [3:9]
state.gripper                  state.gripper              observation.state   [9:10]
video.wrist                    video.wrist                observation.images.wrist (mp4)
annotation.human...            annotation.human...        task_index
```

### Normalization Strategy

| Key | Mode | Rationale |
|-----|------|-----------|
| `state.eef_pos` | `min_max` | Bounded workspace, scale to [-1, 1] |
| `state.eef_rot6d` | `min_max` | rot6d values are bounded [-1, 1] naturally |
| `state.gripper` | `binary` | Binary open/close, pass through as-is |
| `action.eef_pos` | `min_max` | Relative displacements, small bounded range |
| `action.eef_rot6d` | `min_max` | Relative rotation, close to identity |
| `action.gripper` | `binary` | Binary target state |

---

## Session-Level Train/Val Split

When using `--train-only`, the converter reads `episode_session_mapping.json`
from the source dataset to perform session-level splitting:

- Sessions are sorted chronologically (by session name which includes timestamp)
- Last `val_ratio` fraction of sessions become validation
- All augmented versions of a session go to the same split (no data leakage)

Example with 53 sessions, val_ratio=0.2:
- 43 sessions (215 episodes) -> training dataset
- 10 sessions (50 episodes) -> held out

The validation set is NOT exported. Use it separately with your own eval pipeline
if needed.

---

## Verification

After conversion, run these checks:

```bash
TASK_DIR=/path/to/starvla/playground/Datasets/FastUMI/pickandplace_vla

# 1. Check meta files exist
ls "$TASK_DIR/meta/info.json" \
   "$TASK_DIR/meta/modality.json" \
   "$TASK_DIR/meta/episodes.jsonl" \
   "$TASK_DIR/meta/tasks.jsonl"

# 2. Check episode count consistency
python3 -c "
import json, glob
info = json.load(open('$TASK_DIR/meta/info.json'))
n_parquet = len(glob.glob('$TASK_DIR/data/chunk-*/episode_*.parquet'))
n_videos = len(glob.glob('$TASK_DIR/videos/chunk-*/observation.images.wrist/episode_*.mp4'))
n_episodes = sum(1 for _ in open('$TASK_DIR/meta/episodes.jsonl'))
print(f'info.json:      {info[\"total_episodes\"]} episodes, {info[\"total_frames\"]} frames')
print(f'parquet files:  {n_parquet}')
print(f'video files:    {n_videos}')
print(f'episodes.jsonl: {n_episodes}')
assert n_parquet == n_videos == n_episodes == info['total_episodes'], 'MISMATCH!'
print('All counts match.')
"

# 3. Check a sample parquet file
python3 -c "
import pandas as pd
df = pd.read_parquet('$TASK_DIR/data/chunk-000/episode_000000.parquet')
print(f'Columns: {list(df.columns)}')
print(f'Rows: {len(df)}')
print(f'State dim: {len(df[\"observation.state\"].iloc[0])}')
print(f'Action dim: {len(df[\"action\"].iloc[0])}')
"

# 4. Check a sample video
ffprobe -v quiet -print_format json -show_streams \
    "$TASK_DIR/videos/chunk-000/observation.images.wrist/episode_000000.mp4" \
    | python3 -c "
import json, sys
s = json.load(sys.stdin)['streams'][0]
print(f'Codec: {s[\"codec_name\"]}, {s[\"width\"]}x{s[\"height\"]}, {s[\"r_frame_rate\"]} fps, {s[\"nb_frames\"]} frames')
"
```

---

## File Inventory

```
0srarvla-lerobot-convert/
├── README.md                    # This document
├── convert_v30_to_starvla.py    # Conversion script (run on FastUMI side)
├── fastumi_data_config.py       # DataConfig class (paste into StarVLA data_config.py)
├── fastumi_mixtures.py          # Mixture registration (paste into StarVLA mixtures.py)
└── modality.json                # Reference modality.json (auto-generated by converter)
```

---

## Troubleshooting

| Issue | Cause | Fix |
|-------|-------|-----|
| `KeyError: 'fastumi'` | `robot_type` not registered | Add `FastUMIDataConfig` to `ROBOT_TYPE_CONFIG_MAP` |
| `FileNotFoundError: modality.json` | Missing meta file | Re-run converter or copy `modality.json` manually |
| Video frame count mismatch | ffmpeg seek inaccuracy | Try `--video-codec h264` (more reliable keyframes) |
| `ModuleNotFoundError: pandas` | Wrong conda env | Use env with pandas installed: `conda activate lerobot` |
| Empty action/state in training | Wrong normalization | Check `stats_gr00t.json` is regenerated after data change |
| `action_mode` confusion | Actions already relative | Use `action_mode="abs"` in StarVLA; actions are pre-computed |
