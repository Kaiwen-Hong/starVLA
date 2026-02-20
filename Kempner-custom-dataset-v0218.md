# StarVLA Custom Dataset v0218 Training (27 Variants)

> **Goal:** Train StarVLA (QwenOFT) on the v0218 Custom dataset subset (27 variants,
> 2700 episodes) on Kempner — both `kempner_requeue` and `kempner_h100`.
> Updated 2026-02-19.

---

## What changed: v0218 vs custom_all

The previous `custom_all` used all 33 task variants. v0218 removes the wp1/wp2
variants for 3 tasks that had problems with obstacle data, keeping only their
clean versions:

| Task | clean | wp1 | wp2 | Change |
|------|-------|-----|-----|--------|
| adjust_bottle | yes | yes | yes | unchanged |
| beat_block_hammer | yes | yes | yes | unchanged |
| blocks_ranking_rgb | yes | yes | yes | unchanged |
| **blocks_ranking_size** | yes | **no** | **no** | **removed wp1/wp2** |
| click_alarmclock | yes | yes | yes | unchanged |
| handover_block | yes | yes | yes | unchanged |
| **handover_mic** | yes | **no** | **no** | **removed wp1/wp2** |
| move_can_pot | yes | yes | yes | unchanged |
| **move_pillbottle_pad** | yes | **no** | **no** | **removed wp1/wp2** |
| move_stapler_pad | yes | yes | yes | unchanged |
| place_empty_cup | yes | yes | yes | unchanged |

**Summary:** 8 tasks x 3 + 3 tasks x 1 = **27 variants, 2700 episodes** (was 33 / 3300).

### Why no data regeneration is needed

In the `ar-research-kempner` repo (pi0/pi05), switching to v0218 required either
episode filtering or regenerating a new LeRobot dataset, because the data was stored
as a single merged repo (`custom_all_repo`).

StarVLA is different: it uses **per-task directories** loaded via `mixtures.py`.
All 33 directories already exist in `playground/Datasets/Custom/`. To use 27,
we simply register a new mixture that lists only the 27 we want. **Zero data work.**

---

## Current status

| Item | Status | Notes |
|------|--------|-------|
| Per-task datasets | Ready | All 33 dirs in `playground/Datasets/Custom/` (27 used, 6 ignored) |
| `mixtures.py` | Ready | `custom_v0218` (27 tasks) and `custom_v0218_task1` (debug) registered |
| SLURM scripts | Ready | `slurm_custom_v0218_requeue.sh` and `slurm_custom_v0218_h100.sh` created |
| Pretrained model | Ready | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` |
| conda env | Ready | lab miniforge `starVLA` |
| Image-in-parquet patches | Ready | `datasets.py` already patched (same as custom_all) |

---

## Required code changes

### 1. `starVLA/dataloader/gr00t_lerobot/mixtures.py`

Add after the existing `custom_task1` entry:

```python
    # ── Custom v0218: 27 variants (removed wp1/wp2 for 3 tasks with obstacle issues) ──
    "custom_v0218": [
        ("adjust_bottle", 1.0, "robotwin"),
        ("adjust_bottle_wp1", 1.0, "robotwin"),
        ("adjust_bottle_wp2", 1.0, "robotwin"),
        ("beat_block_hammer", 1.0, "robotwin"),
        ("beat_block_hammer_wp1", 1.0, "robotwin"),
        ("beat_block_hammer_wp2", 1.0, "robotwin"),
        ("blocks_ranking_rgb", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp1", 1.0, "robotwin"),
        ("blocks_ranking_rgb_wp2", 1.0, "robotwin"),
        ("blocks_ranking_size", 1.0, "robotwin"),           # clean only
        ("click_alarmclock", 1.0, "robotwin"),
        ("click_alarmclock_wp1", 1.0, "robotwin"),
        ("click_alarmclock_wp2", 1.0, "robotwin"),
        ("handover_block", 1.0, "robotwin"),
        ("handover_block_wp1", 1.0, "robotwin"),
        ("handover_block_wp2", 1.0, "robotwin"),
        ("handover_mic", 1.0, "robotwin"),                  # clean only
        ("move_can_pot", 1.0, "robotwin"),
        ("move_can_pot_wp1", 1.0, "robotwin"),
        ("move_can_pot_wp2", 1.0, "robotwin"),
        ("move_pillbottle_pad", 1.0, "robotwin"),           # clean only
        ("move_stapler_pad", 1.0, "robotwin"),
        ("move_stapler_pad_wp1", 1.0, "robotwin"),
        ("move_stapler_pad_wp2", 1.0, "robotwin"),
        ("place_empty_cup", 1.0, "robotwin"),
        ("place_empty_cup_wp1", 1.0, "robotwin"),
        ("place_empty_cup_wp2", 1.0, "robotwin"),
    ],

    # Debug subset: just 1 v0218 task
    "custom_v0218_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],
```

### 2. SLURM scripts

Create the scripts listed in sections 4 and 5 below.

No other code changes required. The image-in-parquet patches from `custom_all`
are already in place.

---

## TL;DR -- Submit commands

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── kempner_requeue (7-day, preemptable, H200) ──
sbatch scripts/slurm_custom_v0218_requeue.sh

# ── kempner_h100 (3-day, stable, H100) ──
sbatch scripts/slurm_custom_v0218_h100.sh
```

### Check resources & estimated queue time

```bash
# ── Available GPUs on requeue ──
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h200
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h100

# ── All partitions overview ──
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/most_empty_partition.sh

# ── After submitting: check estimated start ──
squeue --start -j <jobid>
```

### Monitor

```bash
squeue -u $USER                                # job status
tail -f logs/starVLA_v0218_<jobid>.out         # training output (requeue)
tail -f logs/starVLA_v0218_<jobid>.err         # errors/warnings
tail -f logs/starVLA_v0218_h100_<jobid>.out    # training output (h100)
tail -f logs/starVLA_v0218_h100_<jobid>.err    # errors/warnings
scancel <jobid>                                # cancel job
```

### Queue status meanings

| Status | Meaning | Expected |
|--------|---------|----------|
| `PD (Resources)` | SLURM ready to schedule, waiting for free GPUs | Usually fast |
| `PD (Priority)` | Higher-priority jobs ahead | May take longer |
| `PD (ReqNodeNotAvail)` | Selected nodes down/draining | Cancel and resubmit |
| `R` | Running | -- |
| `PR` | Preempted (requeue only) | Auto re-queued |

---

## 4. SLURM script: kempner_requeue (7-day, preemptable)

File: `scripts/slurm_custom_v0218_requeue.sh`

```bash
#!/bin/bash
#SBATCH --job-name=starVLA_v0218
#SBATCH --partition=kempner_requeue
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h200
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=1440G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# StarVLA Custom v0218 (27 variants) on kempner_requeue
# 1 node x 4 H200 80GB, DeepSpeed ZeRO-2, QwenOFT
# (change --constraint to h100 and --cpus-per-task to 96 for H100)
# ============================================================

# ── Environment setup ──
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── NCCL ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Diagnostics ──
echo "============================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:   $HF_HOME"
echo "============================================"

# ── Training ──
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_v0218 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 500000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_v0218_qwenOFT_requeue \
  --wandb_project starVLA_Custom_v0218 \
  --wandb_entity kaiwenh-17-uiuc
```

### kempner_requeue notes

- **7-day time limit.** Training 500K steps on 4x H200 should complete within one submission.
- **Preemptable.** `--requeue` auto re-queues; `--trainer.is_resume true` auto-resumes
  from latest checkpoint. Between the two, training continues transparently.
- **`save_interval=50000`** means up to 50K steps lost per preemption. If preempted
  frequently, reduce to 25000 or 10000 (more storage overhead).
- Default GPU is **H200** (`--constraint=h200`, `--cpus-per-task=64`).
  See section 8 for switching to H100.

---

## 5. SLURM script: kempner_h100 (3-day, stable)

File: `scripts/slurm_custom_v0218_h100.sh`

```bash
#!/bin/bash
#SBATCH --job-name=starVLA_v0218_h100
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_ydu_lab
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gpus-per-node=4
#SBATCH --mem=1440G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ============================================================
# StarVLA Custom v0218 (27 variants) on kempner_h100
# 1 node x 4 H100 80GB, DeepSpeed ZeRO-2, QwenOFT
# Stable partition (not preemptable), 3-day time limit
# ============================================================

# ── Environment setup ──
set +u
source ~/.bashrc-kaiwen
set -euo pipefail

module load cuda/12.2.0-fasrc01
conda activate starVLA

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── NCCL ──
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# ── Diagnostics ──
echo "============================================"
echo "Job:       $SLURM_JOB_ID"
echo "Node:      $(hostname)"
echo "GPUs:      $CUDA_VISIBLE_DEVICES"
echo "Python:    $(which python)"
echo "Torch:     $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA avail:$(python -c 'import torch; print(torch.cuda.is_available())')"
echo "HF_HOME:   $HF_HOME"
echo "============================================"

# ── Training ──
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_v0218 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 500000 \
  --trainer.save_interval 50000 \
  --trainer.logging_frequency 100 \
  --trainer.eval_interval 5000 \
  --trainer.gradient_accumulation_steps 2 \
  --trainer.is_resume true \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_v0218_qwenOFT_h100-0 \
  --wandb_project starVLA_Custom_v0218 \
  --wandb_entity kaiwenh-17-uiuc
```

### kempner_h100 notes

- **3-day time limit, not preemptable.** Job runs uninterrupted but may timeout
  before 500K steps finish.
- **`is_resume=true`** is still important: if the job times out, just resubmit
  the same script and it picks up from the latest checkpoint.
- **Longer queue time** than kempner_requeue (dedicated partition, fewer slots).
- If the queue is very long, consider switching to `kempner_requeue`.

### Partition comparison

| | kempner_h100 | kempner_requeue |
|---|---|---|
| Max time | **3 days** | **7 days** |
| Preemption | No | Yes (high-priority jobs can preempt) |
| GPU types | H100 only | Mixed (H100 / H200 / A100) |
| Queue time | Longer | Shorter |
| Best for | Stable critical runs | Long training, flexible scheduling |

---

## 6. Training parameters

### Parameter summary

| Parameter | Value | Notes |
|-----------|-------|-------|
| `data_root_dir` | `playground/Datasets/Custom` | Same directory as custom_all |
| `data_mix` | `custom_v0218` | 27 tasks in `mixtures.py` |
| `per_device_batch_size` | 8 | Per GPU |
| `gradient_accumulation_steps` | 2 | Effective batch = 4 GPU x 8 x 2 = **64** |
| `max_train_steps` | 500,000 | ~6 epochs (dataset is 18% smaller than custom_all) |
| `save_interval` | 50,000 | Checkpoint every 50K steps |
| `eval_interval` | 5,000 | Eval every 5K steps |
| `is_resume` | true | Auto-resume after preemption or timeout |
| `attn_implementation` | sdpa | Required on Kempner (GLIBC 2.28) |

### Comparison with custom_all

| | custom_all (33 variants) | custom_v0218 (27 variants) |
|---|---|---|
| `data_mix` | `custom_all` | `custom_v0218` |
| Total episodes | 3,300 | 2,700 |
| Estimated frames | ~873K | ~714K |
| `max_train_steps` | 500,000 (~5 epochs) | 500,000 (~6 epochs) |
| Normalization stats | Computed from 33 tasks | Recomputed from 27 tasks |
| `run_id` | `custom_qwenOFT_requeue` | `custom_v0218_qwenOFT_requeue` / `_h100` |
| WandB project | `starVLA_Custom` | `starVLA_Custom_v0218` |

> **Note on fresh start vs resume:** v0218 uses a new `run_id`, so training starts
> fresh. This avoids a subtle issue where normalization statistics (mean/std/q01/q99)
> would shift slightly if you resumed from a 33-task checkpoint with 27-task data.
> The normalization is recomputed from scratch each time the dataloader is created,
> so a new `run_id` is the cleanest approach.

---

## 7. Dataset info

| Property | Value |
|----------|-------|
| Location | `playground/Datasets/Custom/` (27 of 33 directories used) |
| Format | LeRobot v2.1, image-in-parquet (no .mp4 video files) |
| Variants | 27 (8 tasks x 3 + 3 tasks x clean only) |
| Episodes/task | 100 |
| Total episodes | 2,700 |
| FPS | 50 |
| Features | 14-dim state/action, 3 cameras (cam_high, cam_left_wrist, cam_right_wrist) |
| Resolution | 640 x 480 |
| Robot type | `robotwin` (dual-arm aloha, same config as RoboTwin) |
| Size on disk | ~200 GB total (27 dirs use ~164 GB; 6 unused dirs remain, no extra cost) |

### 27 task variants

```
8 tasks with clean + wp1 + wp2 (24 variants):
  adjust_bottle       adjust_bottle_wp1       adjust_bottle_wp2
  beat_block_hammer   beat_block_hammer_wp1   beat_block_hammer_wp2
  blocks_ranking_rgb  blocks_ranking_rgb_wp1  blocks_ranking_rgb_wp2
  click_alarmclock    click_alarmclock_wp1    click_alarmclock_wp2
  handover_block      handover_block_wp1      handover_block_wp2
  move_can_pot        move_can_pot_wp1        move_can_pot_wp2
  move_stapler_pad    move_stapler_pad_wp1    move_stapler_pad_wp2
  place_empty_cup     place_empty_cup_wp1     place_empty_cup_wp2

3 tasks with clean only (3 variants):
  blocks_ranking_size
  handover_mic
  move_pillbottle_pad
```

---

## 8. GPU type switching

When switching GPU type, change these SLURM parameters:

| Parameter | H100 | H200 | A100 |
|-----------|------|------|------|
| `--constraint` | h100 | h200 | a100 |
| `--cpus-per-task` | 96 | 64 | 64 |
| `--mem` | 1440G | 1440G | 960G |

> The requeue script defaults to H200; the h100 script uses H100.
> A100 40GB may not have enough VRAM for this model.

---

## 9. Checkpoints & resume

### Checkpoint structure

```
results/Checkpoints/custom_v0218_qwenOFT_requeue/   (or _h100)
  config.yaml                            Saved training config
  dataset_statistics.json                Action/state normalization stats
  summary.jsonl                          One line per checkpoint
  checkpoints/
    steps_50000/                         Full accelerator state (for resume)
    steps_50000_pytorch_model.pt         Standalone weights (for deployment)
    steps_100000/
    steps_100000_pytorch_model.pt
    ...
```

### Resume logic

When `is_resume=true`, the trainer:
1. Scans `checkpoints/` for directories matching `steps_<N>/`
2. Picks the highest `N`
3. Calls `accelerator.load_state()` to restore model, optimizer, scheduler, RNG
4. Continues from step `N`

This works for both:
- **Preemption** on kempner_requeue: `--requeue` auto re-queues, script re-runs, resume kicks in
- **Timeout** on kempner_h100: manually resubmit `sbatch scripts/slurm_custom_v0218_h100.sh`

In both cases, no parameter changes needed.

### Manual restart after timeout (kempner_h100)

```bash
# Just resubmit -- same run_id, is_resume=true finds latest checkpoint
sbatch scripts/slurm_custom_v0218_h100.sh
```

---

## 10. Monitoring

### Job status

```bash
squeue -u $USER                        # all your jobs
squeue --start -j <jobid>              # estimated start time
```

### Training logs

```bash
# Log file paths:
#   requeue: logs/starVLA_v0218_<jobid>.out / .err
#   h100:    logs/starVLA_v0218_h100_<jobid>.out / .err

ls logs/starVLA_v0218*.out             # list all log files
tail -f logs/starVLA_v0218_<jobid>.out # live training output
tail -f logs/starVLA_v0218_<jobid>.err # live errors/warnings
```

### WandB

- Project: `starVLA_Custom_v0218`
- Entity: `kaiwenh-17-uiuc`
- Run names: `custom_v0218_qwenOFT_requeue` or `custom_v0218_qwenOFT_h100`

### GPU usage (on compute node)

```bash
srun --jobid=<jobid> --pty bash
nvidia-smi
watch -n 5 nvidia-smi
```

---

## 11. Quick debugging

### Single-task test (SLURM)

Modify any of the SLURM scripts to use:
```bash
  --datasets.vla_data.data_mix custom_v0218_task1 \   # only adjust_bottle
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --run_id custom_v0218_debug_test \
```

### Interactive test (no sbatch)

```bash
# ── Request an interactive node ──
# kempner_requeue (H200):
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
  -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-01:00:00

# Or kempner_h100 (H100):
salloc -p kempner_h100 --account=kempner_ydu_lab --constraint=h100 \
  -N 1 --gpus-per-node=4 --cpus-per-task=96 --mem=1440G -t 0-01:00:00

# ── Setup environment ──
source ~/.bashrc-kaiwen
conda activate starVLA
module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── Quick test (20 steps) ──
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenOFT \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --datasets.vla_data.data_root_dir playground/Datasets/Custom \
  --datasets.vla_data.data_mix custom_v0218_task1 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id custom_v0218_interactive_test
```

---

## 12. Multi-node training (2 nodes x 8 GPU)

The same multi-node setup from `custom_all` applies. Create a 2-node variant by
adapting `scripts/slurm_custom_requeue-2node.sh`:

| | 1 node (4 GPU) | 2 nodes (8 GPU) |
|---|---|---|
| SLURM `--nodes` | 1 | 2 |
| Total GPUs | 4 | 8 |
| `gradient_accumulation_steps` | 2 | 1 |
| Effective batch | 64 | 64 |
| `max_train_steps` | 500,000 | 250,000 |
| `save_interval` | 50,000 | 25,000 |
| `run_id` | `custom_v0218_qwenOFT_requeue` | `custom_v0218_qwenOFT_2node` |
| accelerate config | `deepspeed_zero2.yaml` | `deepspeed_zero2_2node.yaml` |
| Launch method | `accelerate launch` | `srun bash -c 'accelerate launch ...'` |

See `Kempner-requeue-doc.md` section 9 for the full 2-node script template and
multi-node communication setup (`MASTER_ADDR`, `MASTER_PORT`, NCCL settings).

---

## 13. Common issues

### GLIBC / flash_attention_2

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
```

**Fix:** `--framework.qwenvl.attn_implementation sdpa` (already in all scripts).

### DeepSpeed nvcc not found

```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**Fix:** `module load cuda/12.2.0-fasrc01` (already in all scripts).

### `PROMPT_COMMAND: unbound variable`

**Fix:** `set +u` before `source ~/.bashrc-kaiwen` (already in all scripts).

### Image dataset `KeyError: 'info'` / `ValueError: 'channel' is not in list`

**Fix:** Already patched in `datasets.py` (Patch 1: image metadata fallback).

### Training hangs at 0% (image-in-parquet infinite loop)

**Fix:** Already patched in `datasets.py` (Patches 2 & 3: `is_image_dataset` +
`_get_images_from_parquet()`). Run `python test_image_dataset.py` to verify.

### `ReqNodeNotAvail, UnavailableNodes`

Selected nodes are down/draining. Cancel and resubmit:
```bash
scancel <jobid>
sbatch scripts/slurm_custom_v0218_requeue.sh   # or _h100.sh
```

### kempner_h100 timeout before training finishes

The 3-day limit may not be enough for 500K steps. Just resubmit:
```bash
sbatch scripts/slurm_custom_v0218_h100.sh
# is_resume=true auto-resumes from latest checkpoint
```

### Cluster full, long queue

```bash
# Check real-time GPU availability
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh

# If H200 full, switch to H100 in the requeue script:
#   --constraint=h100  --cpus-per-task=96
# Or submit to the other partition
```

---

## 14. Environment notes

### Environment variables

All loaded via `source ~/.bashrc-kaiwen` in SLURM scripts:
```bash
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
HF_HOME=$LAB_ROOT/.cache/huggingface
CONDA_PKGS_DIRS=$LAB_ROOT/.conda/pkgs
PIP_CACHE_DIR=$LAB_ROOT/.cache/pip
WANDB_DIR=$LAB_ROOT/.cache/wandb
```

### Software

| Component | Version |
|-----------|---------|
| Conda env | `starVLA` (lab miniforge) |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| DeepSpeed | 0.16.9 |
| Accelerate | 1.5.2 |
| Attention | SDPA (flash-attn blocked by GLIBC 2.28) |

### Account

```bash
sacctmgr show assoc user=$USER format=account%40 | grep kempner
# kempner_ydu_lab
```

---

## 15. File reference

| File | Purpose |
|------|---------|
| `scripts/slurm_custom_v0218_requeue.sh` | SLURM script: kempner_requeue, 1 node x 4 GPU |
| `scripts/slurm_custom_v0218_h100.sh` | SLURM script: kempner_h100, 1 node x 4 GPU |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_v0218` / `custom_v0218_task1` mixtures |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset classes (image-in-parquet patches) |
| `starVLA/dataloader/lerobot_datasets.py` | Dataset factory |
| `starVLA/training/train_starvla.py` | Training loop |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` | Base training YAML config |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 config |
| `starVLA/config/deepseeds/ds_config.yaml` | DeepSpeed params (BF16, ZeRO-2) |
| `test_image_dataset.py` | Image-in-parquet unit test |
| `playground/Datasets/Custom/` | All 33 per-task dirs (27 used by v0218) |
| `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` | Pretrained VLM |
| `results/Checkpoints/custom_v0218_qwenOFT_requeue/` | Requeue training output |
| `results/Checkpoints/custom_v0218_qwenOFT_h100/` | H100 training output |
| `logs/` | SLURM stdout/stderr logs |

---

## Related documents

| Document | Content |
|----------|---------|
| `Kempner-requeue-doc.md` | Full kempner_requeue guide (custom_all, multi-node details) |
| `Kempner-custom-dataset-v0.md` | Self-contained custom_all guide (English) |
| `0218-reuse-custom-lerobot-dataset.md` | Dataset split process and image-in-parquet patches |
