# LeRobot Dataset Requirements for StarVLA

Checklist and format specification for using a LeRobot dataset with StarVLA's
training pipeline. Use this to verify your dataset before training.

---

## Quick Checklist

Before training, verify all items below. Each links to a detailed section.

### Files & Directories

- [ ] Each task is a **separate directory** under `data_root_dir` ([Section 1](#1-directory-structure))
- [ ] `meta/info.json` exists with required fields ([Section 2](#2-metainfojson))
- [ ] `meta/modality.json` exists and maps all modalities ([Section 3](#3-metamodalityjson))
- [ ] `meta/episodes.jsonl` exists, one JSON line per episode ([Section 4](#4-metaepisodesjsonl))
- [ ] `meta/tasks.jsonl` exists, one JSON line per unique task ([Section 5](#5-metatasksjsonl))
- [ ] Parquet files exist at `data/chunk-{NNN}/episode_{NNNNNN}.parquet` ([Section 6](#6-parquet-files))
- [ ] If video mode: MP4 files exist at the paths defined in `info.json` ([Section 7](#7-image--video-storage))

### Data Content

- [ ] Parquet columns match `original_key` references in `modality.json` ([Section 6](#6-parquet-files))
- [ ] Action and state dimensions match what `data_config.py` expects ([Section 8](#8-robot-data-config))
- [ ] Camera/video keys in `modality.json` match `data_config.py` video_keys ([Section 8](#8-robot-data-config))
- [ ] `task_index` values in parquet all have entries in `tasks.jsonl` ([Section 5](#5-metatasksjsonl))
- [ ] Episode indices are sequential: 0, 1, 2, ... N-1 ([Section 4](#4-metaepisodesjsonl))

### Registration

- [ ] Mixture registered in `mixtures.py` with correct `(folder_name, weight, robot_type)` ([Section 9](#9-registering-your-dataset))
- [ ] `robot_type` has an entry in `ROBOT_TYPE_CONFIG_MAP` in `data_config.py` ([Section 8](#8-robot-data-config))

### Auto-Generated (no action needed)

- [ ] `meta/stats_gr00t.json` -- computed automatically on first load if missing
- [ ] `meta/steps_data_index.pkl` -- index cache, rebuilt automatically

---

## 1. Directory Structure

StarVLA loads datasets on a **per-task** basis. Each task variant is its own
LeRobot dataset directory. All task directories live under a single root
(e.g. `playground/Datasets/Custom/`).

```
data_root_dir/
├── task_name_1/
│   ├── meta/
│   │   ├── info.json
│   │   ├── modality.json
│   │   ├── episodes.jsonl
│   │   └── tasks.jsonl
│   └── data/
│       └── chunk-000/
│           ├── episode_000000.parquet
│           ├── episode_000001.parquet
│           └── ...
├── task_name_2/
│   └── (same structure)
└── ...
```

If using **video mode** (MP4 files instead of image-in-parquet), add:

```
task_name/
└── videos/
    └── chunk-000/
        ├── observation.images.cam_high/
        │   ├── episode_000000.mp4
        │   └── ...
        ├── observation.images.cam_left_wrist/
        │   └── ...
        └── observation.images.cam_right_wrist/
            └── ...
```

---

## 2. meta/info.json

Required fields:

```jsonc
{
  "codebase_version": "v2.1",          // LeRobot version string (v2.0 or v2.1)
  "robot_type": "robotwin",            // Must match a key in ROBOT_TYPE_CONFIG_MAP
  "total_episodes": 100,               // Integer, total trajectories
  "total_frames": 26543,               // Integer, total frames across all episodes
  "total_tasks": 1,                    // Integer, number of unique tasks
  "total_videos": 0,                   // 0 = image-in-parquet; >0 = video mode
  "total_chunks": 1,                   // Number of chunk directories
  "chunks_size": 1000,                 // Max episodes per chunk directory
  "fps": 50,                           // Frames per second
  "splits": {
    "train": "0:100"                   // Episode range string "start:end"
  },
  "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
  "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
  "features": {
    // Every column in parquet must be described here.
    // See section 2.1 for examples.
  }
}
```

### 2.1 Features Section

Each parquet column needs an entry in `features`:

**Numeric columns** (state, action, timestamps, indices):
```json
"action": {
  "dtype": "float32",
  "shape": [14],
  "names": null
}
```

**Video/image columns** (only if using that storage mode):
```json
"observation.images.cam_high": {
  "dtype": "video",
  "shape": [480, 640, 3],
  "names": ["height", "width", "channels"],
  "info": {
    "video.height": 480,
    "video.width": 640,
    "video.channels": 3,
    "video.fps": 50.0,
    "video.codec": "av1",
    "video.pix_fmt": "yuv420p",
    "video.is_depth_map": false
  }
}
```

For image-in-parquet, use `"dtype": "image"` instead of `"video"`.

### 2.2 Image-in-Parquet vs Video Detection

The dataloader checks `info.json["total_videos"]`:

| `total_videos` | Mode | Images stored in |
|---|---|---|
| `0` | Image-in-parquet | Parquet columns as `{"bytes": <png_bytes>}` dicts |
| `> 0` | Video | Separate MP4 files under `videos/` |

No code change is needed -- the dataloader auto-detects.

---

## 3. meta/modality.json

Maps high-level modality names (used by `data_config.py`) to actual parquet
column names and index ranges. **This file is required.**

### 3.1 Full Example (RoboTwin, 14-dim dual-arm)

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

### 3.2 Rules

| Field | Required | Description |
|-------|----------|-------------|
| `start` / `end` | Yes (action, state) | Slice indices into the parquet column array. `end` is exclusive. |
| `original_key` | Yes | Parquet column name this modality reads from. |
| `dtype` | No | Defaults to `float64`. Options: `float32`, `float64`, `int64`. |
| `absolute` | No | `true` (default) = absolute values; `false` = delta. Controls padding. |
| `rotation_type` | No | If the slice contains rotation data: `quaternion`, `axis_angle`, `rotation_6d`, `euler_angles_rpy`, etc. |

### 3.3 How Names Map to data_config.py

The naming convention is: `data_config.py` uses `"<modality>.<name>"` keys,
and `modality.json` defines entries under `"<modality>"` with key `"<name>"`.

```
data_config.py key:  "action.left_joints"
                      ^^^^^^  ^^^^^^^^^^^
                      │       └── modality.json: action -> "left_joints" entry
                      └── modality section
```

The `original_key` + `start:end` tell the dataloader which parquet column to
read and which indices to slice.

---

## 4. meta/episodes.jsonl

One JSON object per line, one per episode:

```json
{"episode_index": 0, "length": 265, "tasks": ["pick up the red block"]}
{"episode_index": 1, "length": 312, "tasks": ["pick up the red block"]}
...
```

**Requirements:**

- `episode_index`: sequential integers starting at 0, no gaps
- `length`: exact frame count matching the parquet file's row count
- `tasks`: list of task description strings (used for language conditioning)

---

## 5. meta/tasks.jsonl

One JSON object per line, one per unique task description:

```json
{"task_index": 0, "task": "pick up the red block and place it on the blue pad"}
{"task_index": 1, "task": "grasp the bottle and move it to the shelf"}
```

**Requirements:**

- Every `task_index` value appearing in parquet data must have a corresponding
  entry here
- `task` is the human-readable instruction string -- this is what gets fed to
  the VLM as the language prompt during training

---

## 6. Parquet Files

Path pattern: `data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet`

Where `chunk_index = episode_index // chunks_size`.

### 6.1 Required Columns

Every parquet file must contain these columns:

| Column | Dtype | Description |
|--------|-------|-------------|
| `timestamp` | float32/64 | Time in seconds (used for video frame sync) |
| `frame_index` | int64 | Frame number within this episode (0-based) |
| `episode_index` | int64 | Episode ID (constant within a file) |
| `task_index` | int64 | Task ID (references `tasks.jsonl`) |
| `index` | int64 | Global frame index across all episodes |

### 6.2 State & Action Columns

Column names must match the `original_key` values in `modality.json`:

| Column | Dtype | Shape | Example |
|--------|-------|-------|---------|
| `observation.state` | float32/64 | `[state_dim]` | 14-dim for dual-arm |
| `action` | float32/64 | `[action_dim]` | 14-dim for dual-arm |

### 6.3 Image Columns (image-in-parquet mode only)

When `total_videos == 0`, image data is embedded directly in parquet.
Each image cell is a dict: `{"bytes": <PNG/JPEG bytes>}`.

Column names must match the `original_key` values in `modality.json`'s
`video` section:

| Column | Dtype | Example |
|--------|-------|---------|
| `observation.images.cam_high` | binary (dict) | `{"bytes": b"\x89PNG..."}` |
| `observation.images.cam_left_wrist` | binary (dict) | Same format |

---

## 7. Image / Video Storage

StarVLA supports two modes. Pick one per dataset.

### 7.1 Image-in-Parquet

- Set `total_videos: 0` in `info.json`
- Store images as `{"bytes": <compressed_image_bytes>}` dicts in parquet columns
- Larger on disk but simpler (no separate video files)
- Used by: Custom datasets, Demo-Clean RoboTwin

### 7.2 Video (MP4)

- Set `total_videos` to the actual count of `.mp4` files in `info.json`
- Store videos at path: `videos/chunk-{NNN}/{video_key}/episode_{NNNNNN}.mp4`
- Supported codecs: H.264, AV1
- Video backend config: `--datasets.vla_data.video_backend torchvision_av` (or `decord`)
- Used by: RoboTwin-Randomized

---

## 8. Robot Data Config

Each `robot_type` string (from the mixture tuple) maps to a Python class in
`data_config.py` via `ROBOT_TYPE_CONFIG_MAP`. This class defines:

| Attribute | What it controls |
|-----------|-----------------|
| `video_keys` | Which camera streams to load (e.g. `["video.cam_high", ...]`) |
| `state_keys` | Which state modalities to load (e.g. `["state.left_joints", ...]`) |
| `action_keys` | Which action modalities to load |
| `language_keys` | Language instruction key (usually `["annotation.human.action.task_description"]`) |
| `observation_indices` | Time indices for observations (usually `[0]`) |
| `action_indices` | Time indices for action chunks (e.g. `list(range(16))` = 16-step horizon) |
| `modality_config()` | Returns `ModalityConfig` dict grouping the above |
| `transform()` | Returns normalization + augmentation transforms |

### 8.1 Key Constraint: Names Must Align

The **keys** in `modality.json` and the **keys** in the data config class must
correspond:

```
modality.json                          data_config.py
─────────────                          ──────────────
action.left_joints   (start:0 end:6)   action_keys = ["action.left_joints", ...]
state.left_joints    (start:0 end:6)   state_keys  = ["state.left_joints", ...]
video.cam_high       (original_key)    video_keys  = ["video.cam_high", ...]
```

If you use an existing `robot_type` (e.g. `"robotwin"` = `AgilexDataConfig`),
your `modality.json` must define exactly the modality names that config expects.

### 8.2 Normalization Modes

Set per-key in the data config's `transform()`:

| Mode | Effect | Typical use |
|------|--------|-------------|
| `min_max` | Scale to [-1, 1] using dataset min/max | Joint positions |
| `q99` | Scale using 1st/99th percentiles (outlier-robust) | Noisy data |
| `binary` | Pass through as-is (expected 0 or 1) | Gripper open/close |

### 8.3 Using an Existing robot_type

If your new dataset has the **same robot morphology** as an existing one (same
number of joints, same cameras), you can reuse the existing `robot_type` string.
Just make sure your `modality.json` uses the same modality names.

**Example:** All our Custom datasets reuse `robot_type = "robotwin"` (which maps
to `AgilexDataConfig`) because they share the same 14-dim dual-arm format and
3 cameras.

### 8.4 Adding a New robot_type

If your robot has different dimensions or cameras, you need to:

1. Create a new config class in `data_config.py`
2. Register it in `ROBOT_TYPE_CONFIG_MAP`
3. (Optional) Add an embodiment tag in `embodiment_tags.py`

If the robot_type is not in `ROBOT_TYPE_TO_EMBODIMENT_TAG`, it defaults to
`EmbodimentTag.NEW_EMBODIMENT` (projector index 31). This is fine for custom
robots.

---

## 9. Registering Your Dataset

### 9.1 Add Mixture to mixtures.py

In `starVLA/dataloader/gr00t_lerobot/mixtures.py`, add your dataset:

```python
DATASET_NAMED_MIXTURES = {
    # ...existing entries...

    "my_new_dataset": [
        ("task_folder_name_1", 1.0, "robotwin"),
        ("task_folder_name_2", 1.0, "robotwin"),
        # ...
    ],

    # Optional: single-task debug subset
    "my_new_dataset_task1": [
        ("task_folder_name_1", 1.0, "robotwin"),
    ],
}
```

Each tuple: `(directory_name, sampling_weight, robot_type)`

- `directory_name`: folder name under `data_root_dir` (NOT a full path)
- `sampling_weight`: relative probability; all weights are normalized
- `robot_type`: key in `ROBOT_TYPE_CONFIG_MAP`

### 9.2 Set Training Flags

```bash
--datasets.vla_data.data_root_dir playground/Datasets/MyData \
--datasets.vla_data.data_mix my_new_dataset \
```

---

## 10. Validation Script

Quick sanity checks you can run before training:

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

TASK_DIR=playground/Datasets/MyData/my_task

# 1. Required meta files exist
ls "$TASK_DIR/meta/info.json" \
   "$TASK_DIR/meta/modality.json" \
   "$TASK_DIR/meta/episodes.jsonl" \
   "$TASK_DIR/meta/tasks.jsonl"

# 2. info.json has required fields
python3 -c "
import json, sys
info = json.load(open('$TASK_DIR/meta/info.json'))
required = ['codebase_version','total_episodes','total_frames','total_videos',
            'total_chunks','chunks_size','fps','splits','data_path','features']
missing = [k for k in required if k not in info]
if missing:
    print(f'FAIL: missing fields in info.json: {missing}', file=sys.stderr)
    sys.exit(1)
print(f'OK: {info[\"total_episodes\"]} episodes, {info[\"total_frames\"]} frames, '
      f'fps={info[\"fps\"]}, videos={info[\"total_videos\"]}')
"

# 3. modality.json has required sections
python3 -c "
import json, sys
mod = json.load(open('$TASK_DIR/meta/modality.json'))
for section in ['action', 'state', 'video', 'annotation']:
    if section not in mod:
        print(f'FAIL: missing section \"{section}\" in modality.json', file=sys.stderr)
        sys.exit(1)
print(f'OK: action keys={list(mod[\"action\"].keys())}, '
      f'state keys={list(mod[\"state\"].keys())}, '
      f'video keys={list(mod[\"video\"].keys())}')
"

# 4. Parquet files exist and have expected columns
python3 -c "
import pandas as pd, glob, sys
files = sorted(glob.glob('$TASK_DIR/data/chunk-*/episode_*.parquet'))
if not files:
    print('FAIL: no parquet files found', file=sys.stderr); sys.exit(1)
df = pd.read_parquet(files[0])
required_cols = ['timestamp','frame_index','episode_index','task_index','index']
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(f'FAIL: missing parquet columns: {missing}', file=sys.stderr); sys.exit(1)
print(f'OK: {len(files)} parquet files, columns={list(df.columns)}, '
      f'first file has {len(df)} rows')
"

# 5. Episode count matches
python3 -c "
import json, glob
info = json.load(open('$TASK_DIR/meta/info.json'))
n_parquet = len(glob.glob('$TASK_DIR/data/chunk-*/episode_*.parquet'))
n_episodes = sum(1 for _ in open('$TASK_DIR/meta/episodes.jsonl'))
assert n_parquet == info['total_episodes'] == n_episodes, \
    f'Mismatch: parquet={n_parquet}, info.json={info[\"total_episodes\"]}, episodes.jsonl={n_episodes}'
print(f'OK: {n_parquet} episodes consistent across all sources')
"
```

---

## 11. Common Pitfalls

| Problem | Symptom | Fix |
|---------|---------|-----|
| Missing `modality.json` | `FileNotFoundError: .../meta/modality.json` | Copy/create the file (see [Section 3](#3-metamodalityjson)) |
| Wrong modality names | `KeyError` during data loading | Ensure names in `modality.json` match `data_config.py` keys exactly |
| Non-sequential episode indices | Index out of range errors | Re-index episodes to 0..N-1 with no gaps |
| `task_index` not in `tasks.jsonl` | Missing task description, empty language input | Add all task indices to `tasks.jsonl` |
| Parquet column name mismatch | `KeyError` on column access | Check that `original_key` in `modality.json` matches actual parquet column names |
| Wrong action/state dimensions | Shape mismatch errors | `end - start` for each modality entry must match what the data config expects |
| No `robot_type` in config map | `KeyError` in `ROBOT_TYPE_CONFIG_MAP` | Register your robot type or reuse an existing one |
| Video files missing | `FileNotFoundError` on MP4 path | Either provide the MP4 files, or convert to image-in-parquet (`total_videos: 0`) |

---

## File Reference

| File | Purpose |
|------|---------|
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | Dataset mixture registry |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | Robot type configs, transforms, normalization |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset loader (handles both video and image-in-parquet) |
| `starVLA/dataloader/gr00t_lerobot/embodiment_tags.py` | Robot-to-embodiment-projector mapping |
| `starVLA/dataloader/lerobot_datasets.py` | Top-level dataset entry point |
| `examples/Robotwin/train_files/modality.json` | Reference modality.json for RoboTwin (14-dim dual-arm, 3 cameras) |
