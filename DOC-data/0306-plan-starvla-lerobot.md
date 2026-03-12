# FastUMI -> StarVLA Integration: Review & Bug Report

Code review of `DOC-data/` implementation files against the StarVLA codebase.
All issues identified below have been fixed in the actual implementation.

---

## 1. Bugs Found in convert_v30_to_starvla.py

### Bug 1 (HIGH): ffmpeg `-ss` before `-i` causes inaccurate seeks

**Location:** `split_video_episode()`, line 63-72

**Problem:** `-ss` is placed BEFORE `-i`, which triggers keyframe-based
seeking. For H.264, keyframes can be spaced 2-5 seconds apart. ffmpeg seeks to
the nearest keyframe BEFORE the target timestamp, then starts decoding from
there. Combined with `-frames:v N`, this can:
- Start the clip at the wrong frame (up to several seconds early)
- Include frames from the previous episode in the output

**Fix:** Move `-ss` AFTER `-i` for frame-accurate seeking. This is slower
(must decode from the start) but correct. For a one-time conversion script,
correctness matters more than speed.

```python
# BEFORE (wrong):
cmd = ["ffmpeg", "-y", "-ss", f"{from_ts:.6f}", "-i", str(video_src), ...]

# AFTER (correct):
cmd = ["ffmpeg", "-y", "-i", str(video_src), "-ss", f"{from_ts:.6f}", ...]
```

### Bug 2 (HIGH): episode_meta indexing by `.iloc[old_ep_idx]` is fragile

**Location:** `convert()`, line 412

**Problem:** `episode_meta.iloc[old_ep_idx]` assumes the parquet rows are
ordered by episode_index and that row position equals episode_index. If the
parquet has a non-default index, or episodes are not in order, this silently
gets the wrong metadata (wrong `from_timestamp`, wrong `length`).

**Fix:** Filter by `episode_index` column instead:

```python
# BEFORE:
ep_meta = episode_meta.iloc[old_ep_idx]

# AFTER:
ep_meta = episode_meta[episode_meta["episode_index"] == old_ep_idx].iloc[0]
```

### Bug 3 (MEDIUM): task description extraction from tasks.parquet is fragile

**Location:** `convert()`, line 304-306

**Problem:** `str(tasks_df.index[0])` assumes the task text is stored as the
DataFrame index. This works for the typical v3.0 format, but is not robust.
If the parquet has a different structure, it silently returns a wrong value
(like "0" or a column name).

**Fix:** Try multiple known formats:

```python
if task_description is None:
    if "task" in tasks_df.columns:
        task_description = str(tasks_df["task"].iloc[0])
    else:
        task_description = str(tasks_df.index[0])
```

### Bug 4 (MEDIUM): timestamp column not verified/reset

**Location:** `convert()`, parquet splitting loop (~line 367)

**Problem:** The `timestamp` column from the v3.0 source is passed through
without verification. If timestamps are not per-episode (not starting from 0),
the v2.0 video frame seeking in `datasets.py` will seek to wrong positions in
the per-episode MP4 files.

**Fix:** Add explicit timestamp reset as a safety measure:

```python
ep_data["timestamp"] = [i / fps for i in range(n_frames)]
```

This guarantees timestamps are `[0, 0.05, 0.10, ...]` (at 20Hz) regardless of
the source format.

### Bug 5 (LOW): no video frame count validation

**Location:** After video splitting

**Problem:** No validation that the output MP4 has the expected number of
frames. ffmpeg can silently produce shorter or longer clips due to seeking
issues, especially with AV1 codec.

**Fix:** Add optional `ffprobe` validation after encoding (added as a
`--verify-videos` flag to avoid slowing down the default path).

---

## 2. Issues in fastumi_data_config.py

### Issue 1: Missing import context note

The file says "paste into data_config.py" but doesn't explicitly list which
imports are needed. In practice, `data_config.py` already imports
`ModalityConfig`, `StateActionToTensor`, `StateActionTransform`, and
`ComposedModalityTransform` for the existing config classes, so this works. But
worth noting.

**Verdict:** No code change needed, just awareness.

### Issue 2: No video transforms

The `transform()` method only has state/action transforms, no video transforms
(VideoToTensor, VideoResize, etc.). This matches `AgilexDataConfig` (RoboTwin)
which also omits video transforms -- the image resize happens in
`_pack_sample()` at line 1496: `Image.fromarray(image).resize((224, 224))`.

**Verdict:** Correct. No change needed.

---

## 3. Missing: action_dim/state_dim Override

**Critical for training.** The training YAML config
(`starvla_cotrain_robotwin.yaml`) hardcodes:

```yaml
action_model:
  action_dim: 14    # RoboTwin: 6+1+6+1 = 14
  state_dim: 14
```

FastUMI uses 10D (3+6+1). The SLURM script MUST override these:

```bash
--framework.action_model.action_dim 10 \
--framework.action_model.state_dim 10 \
```

Without this, the model will try to output 14-dim actions but the dataloader
provides 10-dim, causing a shape mismatch crash.

---

## 4. _pack_sample Single Camera Behavior

`datasets.py:1494-1501`:

```python
if "wrist" not in video_key:
    prim_images.append(image)
else:
    wrist_views.append(image)
all_images = prim_images + wrist_views
```

FastUMI has only `video.wrist`, so `prim_images = []` and
`all_images = [wrist_image]`. The model receives a list with 1 image. This
won't crash, but the model was trained/tested with 3 images (RoboTwin). The
VLM should handle variable image count gracefully since it processes images
independently through the vision encoder.

**Verdict:** Should work. Verify with a smoke test (20 steps).

---

## 5. Implementation Checklist

All changes made in this session:

| Change | File | Status |
|--------|------|--------|
| Fix ffmpeg seek accuracy | `DOC-data/convert_v30_to_starvla.py` | Done |
| Fix episode_meta indexing | `DOC-data/convert_v30_to_starvla.py` | Done |
| Fix task description reading | `DOC-data/convert_v30_to_starvla.py` | Done |
| Add timestamp reset | `DOC-data/convert_v30_to_starvla.py` | Done |
| Add FastUMIDataConfig | `starVLA/dataloader/gr00t_lerobot/data_config.py` | Done |
| Register in ROBOT_TYPE_CONFIG_MAP | `starVLA/dataloader/gr00t_lerobot/data_config.py` | Done |
| Add mixtures | `starVLA/dataloader/gr00t_lerobot/mixtures.py` | Done |
| Create SLURM script | `scripts/slurm_fastumi_requeue.sh` | Done |

---

## 6. Training Parameters for FastUMI

| Parameter | FastUMI Value | RoboTwin Value | Notes |
|-----------|--------------|----------------|-------|
| `action_dim` | **10** | 14 | 3 pos + 6 rot6d + 1 gripper |
| `state_dim` | **10** | 14 | Same layout as action |
| `data_root_dir` | `playground/Datasets/FastUMI` | `playground/Datasets/RoboTwin` | |
| `data_mix` | `fastumi_pickandplace` | `robotwin` | |
| `per_device_batch_size` | 8 | 8 | Same (single camera may allow higher) |
| `action_horizon` | 16 | 16 | 16 steps @ 20Hz = 0.8s |
| `future_action_window_size` | 15 | 15 | |
| `video_backend` | `torchvision_av` | `torchvision_av` | |
| `max_train_steps` | 100,000 | 500,000 | Fewer episodes, start lower |
| `save_interval` | 10,000 | 50,000 | |
| `eval_interval` | 2,000 | 5,000 | |
