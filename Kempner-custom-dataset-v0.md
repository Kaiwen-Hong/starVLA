# StarVLA Custom Dataset Training — v0

> Self-contained guide for training StarVLA (QwenOFT) on the Custom dataset
> on the Kempner cluster.  Updated 2026-02-18.

---

## Overview

Train StarVLA with the 33-variant Custom dataset (11 base RoboTwin tasks x 3
variants: clean / wp1 / wp2, 100 episodes each) on Kempner `kempner_requeue`.

| Item | Value |
|------|-------|
| Model | Qwen3-VL-4B-Instruct-Action (QwenOFT) |
| Dataset | `playground/Datasets/Custom/` (33 per-task LeRobot dirs) |
| Image format | Image-in-parquet (no video files) |
| Total episodes | 3,300 |
| FPS | 50 |
| Features | 14-dim state/action, 3 cameras (cam_high, cam_left_wrist, cam_right_wrist) |
| Cluster partition | `kempner_requeue` (7-day limit, preemptable) |
| GPUs | 4x H100/H200 80 GB (single node) |

---

## 1. Prerequisites

### 1.1 Dataset preparation (already done)

The Custom dataset was generated for pi0/pi05, then split into per-task
directories for StarVLA.  See `0218-reuse-custom-lerobot-dataset.md` for the
split procedure.

Verify all 33 task directories exist and contain `modality.json`:
```bash
for d in playground/Datasets/Custom/*/; do
  [ ! -f "$d/meta/modality.json" ] && echo "MISSING: $d"
done
```

### 1.2 Code patches (required)

The Custom dataset uses **image-in-parquet** format (`dtype: image`,
`total_videos: 0`), unlike RoboTwin which uses external `.mp4` video files.
Three patches to `starVLA/dataloader/gr00t_lerobot/datasets.py` are required:

#### Patch 1: `_get_metadata()` — image metadata parsing (line ~624)

Without this, initialization crashes with `ValueError: 'channel' is not in list`
or `KeyError: 'info'`.

Image datasets store `"channels"` (plural) in the `names` array and `fps` at
the top level of `info.json`, not inside a nested `info` or `video_info` dict.
A third fallback branch was added:

```python
except KeyError:
    # image-based datasets (dtype: image)
    names = le_video_meta.get("names", [])
    if isinstance(names, list) and "channels" in names:
        channels = le_video_meta["shape"][names.index("channels")]
    else:
        channels = 3
    fps = le_info.get("fps", 30)
```

#### Patch 2: `get_video()` — read images from parquet (line ~1187)

Without this, `get_step_data()` crashes with `FileNotFoundError` because it
tries to open `.mp4` files that don't exist.

Added `is_image_dataset` property and `_get_images_from_parquet()` method.
When `total_videos == 0`, images are decoded from the parquet DataFrame column
(each cell is `{'bytes': <PNG bytes>, 'path': ...}`) via PIL, returning the
same `(T, H, W, C)` uint8 ndarray that the video path produces.

#### Patch 3: `LeRobotMixtureDataset.__getitem__()` — skip video existence check (line ~2094)

Without this, training **hangs silently at step 0** — progress bar shows `0%`
and never moves.  The `while True` loop calls `os.path.exists(video_path)` on
a path that never exists for image datasets, looping forever.

Added `dataset.is_image_dataset` check to skip the video file existence
validation:

```python
while True:
    dataset, trajectory_id, step = self.sample_step(index)
    if dataset.is_image_dataset:
        break
    key = dataset.modality_keys["video"][0].replace("video.", "")
    video_path = dataset.get_video_path(trajectory_id, key)
    if os.path.exists(video_path):
        break
    index = random.randint(0, len(self) - 1)
```

#### Verification

Run the unit test to confirm all patches work:
```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
python test_image_dataset.py
```

Expected output: Tests 0-4 all PASS.  Also verifies RoboTwin video datasets
are not affected.

### 1.3 Mixtures registered (already done)

In `starVLA/dataloader/gr00t_lerobot/mixtures.py`:

| Mixture | Datasets | Use case |
|---------|----------|----------|
| `custom_all` | All 33 task variants, weight 1.0 each | Full training |
| `custom_task1` | `adjust_bottle` only | Quick debugging |

### 1.4 Environment

| Component | Details |
|-----------|---------|
| Conda env | `starVLA` (lab miniforge) |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| DeepSpeed | 0.16.9 |
| Accelerate | 1.5.2 |
| Attention | SDPA (flash-attn needs GLIBC >= 2.32, Kempner has 2.28) |

---

## 2. Training

### 2.1 Check resources and submit

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Check available GPUs before submitting
sinfo -p kempner_requeue -o "%N %G %t %f" | grep -E "idle|mix"
# Or use the summary script for a cleaner view
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh

# Submit (1 node × 4 GPU)
sbatch scripts/slurm_custom_requeue.sh

# Or submit (2 nodes × 8 GPU, see section 3.5)
sbatch scripts/slurm_custom_requeue-2node.sh

# Check estimated start time after submitting
squeue --start -j <jobid>
```

### 2.2 Full training command

```bash
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
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_qwenOFT_requeue \
  --wandb_project starVLA_Custom \
  --wandb_entity kaiwenh-17-uiuc
```

### 2.3 Parameter summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_root_dir` | `playground/Datasets/Custom` | Per-task LeRobot dirs |
| `data_mix` | `custom_all` | 33 tasks in `mixtures.py` |
| `per_device_batch_size` | 8 | Per GPU |
| `gradient_accumulation_steps` | 2 | Effective batch = 4 GPU x 8 x 2 = 64 |
| `max_train_steps` | 500,000 | ~5 epochs |
| `save_interval` | 50,000 | Checkpoint every 50K steps |
| `eval_interval` | 5,000 | Eval every 5K steps |
| `is_resume` | true | Auto-resume after preemption |
| `attn_implementation` | sdpa | Required (GLIBC 2.28) |

### 2.4 Differences from RoboTwin training

| | RoboTwin | Custom |
|---|---|---|
| `data_root_dir` | `playground/Datasets/RoboTwin` | `playground/Datasets/Custom` |
| `data_mix` | `robotwin` (50 tasks) | `custom_all` (33 tasks) |
| Image storage | MP4 video (AV1) | Image-in-parquet |
| FPS | 15 | 50 |
| Episodes/task | 500 | 100 |
| Total episodes | 25,000 | 3,300 |
| `max_train_steps` | 100,000 | 500,000 |
| `save_interval` | 10,000 | 50,000 |
| `eval_interval` | 1,000 | 5,000 |

---

## 3. SLURM / kempner_requeue

### 3.1 SBATCH settings

The SLURM script is at `scripts/slurm_custom_requeue.sh`.

```
#SBATCH --partition=kempner_requeue     7-day limit, preemptable
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h200              H200 80GB (change to h100 if needed)
#SBATCH --requeue                      Auto re-queue after preemption
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64             64 for H200, 96 for H100
#SBATCH --mem=1440G
#SBATCH --time=7-00:00:00
```

### 3.2 GPU type switching

| Parameter | H100 | H200 | A100 |
|-----------|------|------|------|
| `--constraint` | h100 | h200 | a100 |
| `--cpus-per-task` | 96 | 64 | 64 |
| `--mem` | 1440G | 1440G | 960G |

### 3.3 Preemption and resume

- `--requeue` makes SLURM auto re-queue preempted jobs
- `--trainer.is_resume true` makes the trainer auto-resume from the latest checkpoint
- Between the two, training continues transparently after preemption
- `save_interval=50000` means up to 50K steps lost per preemption; reduce to 10K-25K if preempted frequently

### 3.4 Environment setup in SLURM

The script must:
1. `set +u` before sourcing `~/.bashrc-kaiwen` (avoids `PROMPT_COMMAND: unbound variable`)
2. `module load cuda/12.2.0-fasrc01` (DeepSpeed needs `nvcc`)
3. `conda activate starVLA`

### 3.5 Multi-node training (2 nodes x 8 GPU)

```bash
sbatch scripts/slurm_custom_requeue-2node.sh
```

Key differences from single-node:

| | 1 node | 2 nodes |
|---|---|---|
| SLURM `--nodes` | 1 | 2 |
| Total GPUs | 4 | 8 |
| `gradient_accumulation_steps` | 2 | 1 |
| Effective batch | 64 | 64 |
| `max_train_steps` | 500,000 | 250,000 |
| `save_interval` | 50,000 | 25,000 |
| `run_id` | `custom_qwenOFT_requeue` | `custom_qwenOFT_2node` |
| Accelerate config | `deepspeed_zero2.yaml` | `deepspeed_zero2_2node.yaml` |
| Launch method | `accelerate launch` | `srun bash -c 'accelerate launch ...'` |

The 2-node script uses `srun` to launch one `accelerate launch` per node.
`SLURM_PROCID` provides the `--machine_rank` (0 on first node, 1 on second).
`MASTER_ADDR` is derived from the first node in `SLURM_JOB_NODELIST`.

See `Kempner-requeue-doc.md` section 9 for full details.

---

## 4. Checkpoints

### 4.1 Directory structure

```
results/Checkpoints/custom_qwenOFT_requeue/
  config.yaml                              Saved training config
  dataset_statistics.json                  Action/state normalization stats
  summary.jsonl                            One line per checkpoint
  checkpoints/
    steps_50000/                           Full accelerator state (for resume)
      (contents created by accelerator.save_state() —
       model shards, optimizer shards, scheduler.bin,
       random_states_*.pkl, etc.)
    steps_50000_pytorch_model.pt           Standalone weights (for deployment)
    steps_100000/
    steps_100000_pytorch_model.pt
    ...
```

### 4.2 Resume logic

When `is_resume=true`, the trainer:
1. Scans `checkpoints/` for directories matching `steps_<N>/`
2. Picks the highest `N`
3. Calls `accelerator.load_state()` to restore model, optimizer, scheduler, RNG
4. Continues from step `N`

Use the same `--run_id` to continue a previous run.  Use a different `--run_id`
to start fresh while keeping old checkpoints.

### 4.3 Standalone weights

For deployment or evaluation, use `steps_<N>_pytorch_model.pt` (or
`.safetensors` if `save_format` is set).  These are model-only files without
optimizer state.

---

## 5. Monitoring

```bash
squeue -u $USER                            # Job status
tail -f logs/starVLA_custom_<jobid>.out    # Training output
tail -f logs/starVLA_custom_<jobid>.err    # Errors/warnings
```

WandB:
- Project: `starVLA_Custom`
- Entity: `kaiwenh-17-uiuc`
- Run name: `custom_qwenOFT_requeue`

---

## 6. Quick debugging

### Single-task test (SLURM)

Change `data_mix` and step count:
```bash
  --datasets.vla_data.data_mix custom_task1 \
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --run_id custom_debug_test \
```

### Interactive test (no sbatch)

```bash
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h100 \
  -N 1 --gpus-per-node=4 --cpus-per-task=96 --mem=1440G -t 0-01:00:00

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
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_task1 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_interactive_test
```

---

## 7. Known issues and fixes

### Training hangs at 0% (image-in-parquet infinite loop)

**Symptom:** Progress bar shows `0%|          | 0/500000` and never moves.
No error messages.

**Root cause:** `LeRobotMixtureDataset.__getitem__()` in `datasets.py:2094` has
a `while True` loop that checks `os.path.exists(video_path)`.  For image-in-parquet
datasets (`total_videos: 0`), there are no `.mp4` files, so the check is always
False and the loop runs forever.

**Fix:** Check `dataset.is_image_dataset` before the video path loop (Patch 3
above).  Also need Patch 2 so `get_video()` reads images from parquet instead of
opening non-existent video files.

### `ValueError: 'channel' is not in list` / `KeyError: 'info'`

**Root cause:** `_get_metadata()` only handled video-format metadata keys.
Image datasets use `"channels"` (plural) in names and `fps` at top level.

**Fix:** Patch 1 above (third fallback branch in `_get_metadata()`).

### GLIBC / flash_attention_2

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
```

**Fix:** Use `--framework.qwenvl.attn_implementation sdpa` (already in training
command and SLURM script).

### DeepSpeed nvcc not found

```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**Fix:** `module load cuda/12.2.0-fasrc01` (already in SLURM script).

### `PROMPT_COMMAND: unbound variable`

**Fix:** `set +u` before `source ~/.bashrc-kaiwen`, then `set -euo pipefail`
after (already in SLURM script).

---

## 8. 33 Task variants

11 base tasks x 3 variants (clean / wp1 / wp2):

```
adjust_bottle          adjust_bottle_wp1          adjust_bottle_wp2
beat_block_hammer      beat_block_hammer_wp1      beat_block_hammer_wp2
blocks_ranking_rgb     blocks_ranking_rgb_wp1     blocks_ranking_rgb_wp2
blocks_ranking_size    blocks_ranking_size_wp1    blocks_ranking_size_wp2
click_alarmclock       click_alarmclock_wp1       click_alarmclock_wp2
handover_block         handover_block_wp1         handover_block_wp2
handover_mic           handover_mic_wp1           handover_mic_wp2
move_can_pot           move_can_pot_wp1           move_can_pot_wp2
move_pillbottle_pad    move_pillbottle_pad_wp1    move_pillbottle_pad_wp2
move_stapler_pad       move_stapler_pad_wp1       move_stapler_pad_wp2
place_empty_cup        place_empty_cup_wp1        place_empty_cup_wp2
```

---

## 9. File reference

| File | Purpose |
|------|---------|
| `scripts/slurm_custom_requeue.sh` | SLURM submission script (1 node x 4 GPU) |
| `scripts/slurm_custom_requeue-2node.sh` | SLURM submission script (2 nodes x 8 GPU) |
| `scripts/split_custom_lerobot.py` | Split merged dataset into per-task dirs |
| `scripts/split_custom_all.sh` | Shell wrapper for the split script |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset classes (patched for image support) |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_all` / `custom_task1` mixture definitions |
| `starVLA/dataloader/lerobot_datasets.py` | Dataset factory (`make_LeRobotSingleDataset`, `get_vla_dataset`) |
| `starVLA/training/train_starvla.py` | Training loop |
| `starVLA/training/trainer_utils/trainer_tools.py` | Checkpoint save/resume logic |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` | Base training YAML config |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 config (1 node) |
| `starVLA/config/deepseeds/deepspeed_zero2_2node.yaml` | Accelerate + DeepSpeed ZeRO-2 config (2 nodes) |
| `starVLA/config/deepseeds/ds_config.yaml` | DeepSpeed params (BF16, ZeRO-2) |
| `test_image_dataset.py` | Unit test for image-in-parquet dataset loading |
| `playground/Datasets/Custom/` | Custom dataset (33 per-task directories) |
| `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` | Pretrained VLM |
| `results/Checkpoints/custom_qwenOFT_requeue/` | 1-node training output |
| `results/Checkpoints/custom_qwenOFT_2node/` | 2-node training output |
| `logs/` | SLURM stdout/stderr logs |

---

## 10. Related documents

| Document | Content |
|----------|---------|
| `0218-reuse-custom-lerobot-dataset.md` | How the per-task dataset was created (split from merged repo) |
| `Kempner-requeue-doc.md` | Detailed kempner_requeue cluster guide (Chinese) |
