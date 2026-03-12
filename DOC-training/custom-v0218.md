# Custom Dataset v0218 Training (27 Variants)

> v0218 removes wp1/wp2 for 3 tasks with obstacle issues from `custom_all`.
> For cluster setup, see `DOC-training/kempner-cluster.md`.
> For common training patterns (image-in-parquet, multi-node), see `DOC-training/custom-all.md`.

---

## What Changed: v0218 vs custom_all

The previous `custom_all` used all 33 task variants. v0218 removes the wp1/wp2
variants for 3 tasks that had problems with obstacle data, keeping only their
clean versions:

| Task | clean | wp1 | wp2 | Change |
|------|-------|-----|-----|--------|
| **blocks_ranking_size** | yes | **no** | **no** | **removed wp1/wp2** |
| **handover_mic** | yes | **no** | **no** | **removed wp1/wp2** |
| **move_pillbottle_pad** | yes | **no** | **no** | **removed wp1/wp2** |
| *(8 other tasks)* | yes | yes | yes | unchanged |

**Summary:** 8 tasks x 3 + 3 tasks x 1 = **27 variants, 2700 episodes** (was 33 / 3300).

### Why no data regeneration is needed

StarVLA uses **per-task directories** loaded via `mixtures.py`. All 33
directories already exist in `playground/Datasets/Custom/`. To use 27, we
simply register a new mixture (`custom_v0218`) that lists only the 27 we want.
**Zero data work.**

---

## Current Status

| Item | Status |
|------|--------|
| Per-task datasets (27 of 33 dirs) | Ready |
| `mixtures.py` (`custom_v0218` + `custom_v0218_task1`) | Ready |
| SLURM scripts (requeue + h100) | Ready |
| Pretrained model (Qwen3-VL-4B-Instruct-Action) | Ready |
| conda env (`starVLA`) | Ready |
| Image-in-parquet patches (`datasets.py`) | Ready |

---

## TL;DR -- Submit Commands

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# kempner_requeue (7-day, preemptable, H200)
sbatch scripts/slurm_custom_v0218_requeue.sh

# kempner_h100 (3-day, stable, H100)
sbatch scripts/slurm_custom_v0218_h100.sh
```

### Check resources

```bash
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/most_empty_partition.sh
squeue --start -j <jobid>   # estimated start time
```

For full SLURM scripts, see `scripts/slurm_custom_v0218_requeue.sh` and
`scripts/slurm_custom_v0218_h100.sh`. For GPU switching, see `DOC-training/kempner-cluster.md`.

---

## Training Parameters

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

> v0218 uses a new `run_id`, so training starts fresh. Normalization stats are
> recomputed from scratch each time the dataloader is created.

---

## Dataset Info

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
| Size on disk | ~200 GB total (27 dirs use ~164 GB; 6 unused dirs remain) |

### 27 task variants

**24 variants** (8 tasks x clean/wp1/wp2): adjust_bottle, beat_block_hammer,
blocks_ranking_rgb, click_alarmclock, handover_block, move_can_pot,
move_stapler_pad, place_empty_cup

**3 variants** (clean only): blocks_ranking_size, handover_mic, move_pillbottle_pad

---

## Checkpoints & Resume

Output: `results/Checkpoints/custom_v0218_qwenOFT_requeue/` (or `_h100`).
Each contains `config.yaml`, `dataset_statistics.json`, and `checkpoints/steps_<N>/`
(full accelerator state) + `steps_<N>_pytorch_model.pt` (standalone weights).

When `is_resume=true`, the trainer scans `checkpoints/` for `steps_<N>/`
directories, picks the highest N, and calls `accelerator.load_state()` to
restore model, optimizer, scheduler, and RNG. This handles both preemption
(auto re-queue on `kempner_requeue`) and timeout (manual resubmit on `kempner_h100`)
with no parameter changes needed.

---

## Quick Debugging

### Single-task test (SLURM)

Modify any SLURM script to use:
```bash
  --datasets.vla_data.data_mix custom_v0218_task1 \   # only adjust_bottle
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --run_id custom_v0218_debug_test \
```

### Interactive test (no sbatch)

```bash
# Get an interactive node, then setup env
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
  -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-01:00:00
source ~/.bashrc-kaiwen && conda activate starVLA && module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Quick 20-step test with single task
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
  --trainer.max_train_steps 20 --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints --run_id custom_v0218_interactive_test
```

---

## File Reference

| File | Purpose |
|------|---------|
| `scripts/slurm_custom_v0218_requeue.sh` | SLURM script: kempner_requeue, 1 node x 4 GPU |
| `scripts/slurm_custom_v0218_h100.sh` | SLURM script: kempner_h100, 1 node x 4 GPU |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_v0218` / `custom_v0218_task1` mixtures |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset classes (image-in-parquet patches) |
| `starVLA/training/train_starvla.py` | Training loop |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 config |
| `playground/Datasets/Custom/` | All 33 per-task dirs (27 used by v0218) |
| `results/Checkpoints/custom_v0218_qwenOFT_*/` | Training output (requeue / h100) |

See `DOC-training/kempner-cluster.md` for monitoring, cluster issues, partition
comparison, and environment notes. See `DOC-training/custom-all.md` for
image-in-parquet patches and multi-node setup.
