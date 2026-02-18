# Reusing Custom LeRobot Datasets for StarVLA Training

## Problem

The custom dataset at `$KEMPNER_BASE/.cache/huggingface/lerobot/custom_all_repo/` was generated for pi0/pi05 training (in `ar-research-kempner`). It is a **single merged** LeRobot v2.1 dataset with all task variants combined:

| Property | Value |
|----------|-------|
| Location | `$KEMPNER_BASE/.cache/huggingface/lerobot/custom_all_repo/` |
| Format | LeRobot v2.1, image-in-parquet (no video files) |
| Episodes | 3300 (33 task variants x 100 each) |
| FPS | 50 |
| Features | 14-dim state/action, 3 cameras (cam_high, cam_left_wrist, cam_right_wrist) |
| Tasks | 3163 unique language instructions |
| Size | ~230 GB (images embedded in parquet) |

**StarVLA requires per-task directories** (each task as an independent LeRobot dataset under `data_root_dir/`), loaded via `mixtures.py`. The merged format cannot be used directly.

## Solution

Split the merged dataset into 33 per-task directories. This is a **file-level operation only** -- no regeneration, no re-downloading, no re-processing. The script reads parquet files from the merged dataset, renumbers the metadata columns (`episode_index`, `index`, `task_index`), and writes them into per-task directories with proper `info.json`, `episodes.jsonl`, `tasks.jsonl`, and `modality.json`.

## Quick Start

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Step 1: Split the merged dataset into per-task directories (33 workers by default, one per task)
bash scripts/split_custom_all.sh

# Step 2: Already done -- custom_all and custom_task1 are registered in mixtures.py

# Step 3: Train
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 500000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_qwenOFT_4xH100
```

Key differences from the RoboTwin training command:
- `--datasets.vla_data.data_root_dir playground/Datasets/Custom` (not `RoboTwin`)
- `--datasets.vla_data.data_mix custom_all` (not `robotwin`)
- `--trainer.max_train_steps 525000` — ~5 epochs (custom dataset is ~5x smaller than RoboTwin, 1 epoch ≈ 105K steps)
- `--trainer.save_interval 50000` — ~2 checkpoints per epoch, ~10 total
- `--trainer.eval_interval 5000` — eval more frequently since dataset is smaller

---

## Step 1: Split the Merged Dataset

```bash
# Default: one worker per task (33 parallel processes)
bash scripts/split_custom_all.sh

# Or specify worker count explicitly
bash scripts/split_custom_all.sh 16

# Sequential mode (for debugging)
bash scripts/split_custom_all.sh 1
```

This runs `scripts/split_custom_lerobot.py` which:
1. Reads `custom_all_repo/meta/episodes.jsonl` and `tasks.jsonl`
2. Splits 3300 episodes into 33 groups of 100 (matching the task variant order)
3. **Processes all 33 tasks in parallel** (each task writes to its own directory, no shared state)
4. For each group, creates a per-task LeRobot directory with renumbered metadata
5. Adds `modality.json` to each task

> **Performance note:** The dataset is ~230 GB image-in-parquet. Each of the 3,300 parquet files (~70 MB each) must be read and rewritten to remap 3 metadata columns (`episode_index`, `index`, `task_index`). Parallelism across the 33 tasks saturates NFS I/O bandwidth and is the main lever for reducing wall-clock time.

**Output structure:**
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
├── adjust_bottle_wp1/          (100 episodes)
├── adjust_bottle_wp2/          (100 episodes)
├── beat_block_hammer/
├── ...
└── place_empty_cup_wp2/        (33 directories total)
```

### 33 Task Variants

11 base tasks x 3 variants (clean, wp1, wp2):

| Base Task | clean | wp1 | wp2 |
|-----------|-------|-----|-----|
| adjust_bottle | adjust_bottle | adjust_bottle_wp1 | adjust_bottle_wp2 |
| beat_block_hammer | beat_block_hammer | beat_block_hammer_wp1 | beat_block_hammer_wp2 |
| blocks_ranking_rgb | blocks_ranking_rgb | blocks_ranking_rgb_wp1 | blocks_ranking_rgb_wp2 |
| blocks_ranking_size | blocks_ranking_size | blocks_ranking_size_wp1 | blocks_ranking_size_wp2 |
| click_alarmclock | click_alarmclock | click_alarmclock_wp1 | click_alarmclock_wp2 |
| handover_block | handover_block | handover_block_wp1 | handover_block_wp2 |
| handover_mic | handover_mic | handover_mic_wp1 | handover_mic_wp2 |
| move_can_pot | move_can_pot | move_can_pot_wp1 | move_can_pot_wp2 |
| move_pillbottle_pad | move_pillbottle_pad | move_pillbottle_pad_wp1 | move_pillbottle_pad_wp2 |
| move_stapler_pad | move_stapler_pad | move_stapler_pad_wp1 | move_stapler_pad_wp2 |
| place_empty_cup | place_empty_cup | place_empty_cup_wp1 | place_empty_cup_wp2 |

---

## Step 2: Register in mixtures.py (Done)

Two mixtures have been added to `starVLA/dataloader/gr00t_lerobot/mixtures.py`:

| Mixture Name | What it loads | Use case |
|---|---|---|
| `custom_all` | All 33 task variants, equal weight 1.0 | Full training |
| `custom_task1` | Only `adjust_bottle` | Quick debugging |

**Why `robot_type` is `"robotwin"`:** The custom data has the same robot configuration as RoboTwin (dual-arm, 14-dim state/action, 3 cameras). In the code, `"robotwin"` maps to `AgilexDataConfig` in `data_config.py` (line 916), which defines the correct modality keys, normalization modes, and camera names. Since `"robotwin"` is not in `ROBOT_TYPE_TO_EMBODIMENT_TAG`, it falls back to `NEW_EMBODIMENT` (projector index 31) — this is expected and correct for finetuning.

**How it works at training time:**
1. You pass `--datasets.vla_data.data_mix custom_all`
2. `lerobot_datasets.py:get_vla_dataset()` looks up `DATASET_NAMED_MIXTURES["custom_all"]` → gets the list of 33 `(task_name, weight, "robotwin")` tuples
3. For each tuple, it creates a `LeRobotSingleDataset` by:
   - Loading parquet files from `data_root_dir/task_name/`
   - Applying `AgilexDataConfig` (3 cameras, 14-dim action/state, min_max + binary normalization)
   - Assigning `EmbodimentTag.NEW_EMBODIMENT`
4. All 33 datasets are wrapped into a `LeRobotMixtureDataset` with equal sampling weights

---

## Differences from RoboTwin-Randomized

| | RoboTwin-Randomized | Custom |
|---|---|---|
| Image storage | MP4 video (AV1) | Image-in-parquet |
| FPS | 15 | 50 |
| Episodes/task | 500 | 100 |
| Total tasks | 50 | 33 (11 base x 3 variants) |
| Total episodes | 25,000 | 3,300 |
| Variants | Single (randomized) | clean / wp1 / wp2 |
| `data_root_dir` | `playground/Datasets/RoboTwin` | `playground/Datasets/Custom` |
| `data_mix` | `robotwin` | `custom_all` |

The StarVLA dataloader handles both image-in-parquet and video mode transparently. The `video_backend: torchvision_av` config still works -- for image-in-parquet data, the video backend is simply not used (images are read directly from parquet).

---

## Files

| File | Purpose |
|------|---------|
| `scripts/split_custom_lerobot.py` | Python script: splits merged LeRobot dataset into per-task dirs (multiprocessing, `--workers N`) |
| `scripts/split_custom_all.sh` | Shell wrapper: runs the split with correct paths and task list (`bash split_custom_all.sh [workers]`) |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_all` and `custom_task1` mixtures (already registered) |
