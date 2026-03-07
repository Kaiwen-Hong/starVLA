# Custom Dataset Training (custom_all — 33 Variants)

> 目标：在 Kempner 上用 Custom 数据集训练 StarVLA (QwenOFT)。
> For cluster setup and environment, see `DOC-training/kempner-cluster.md`.

---

## Overview

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

## 当前状态

| 项目 | 状态 | 说明 |
|------|------|------|
| SLURM 脚本 | 已创建 | `scripts/slurm_custom_requeue.sh` |
| Custom 数据集 | 就绪 | `playground/Datasets/Custom/`（33 task 目录，各含 modality.json） |
| 预训练模型 | 就绪 | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` |
| conda 环境 | 就绪 | lab miniforge `starVLA`（torch 2.6.0+cu124, transformers 4.57.0, accelerate 1.5.2） |
| `mixtures.py` | 已注册 | `custom_all`（33 tasks）和 `custom_task1`（调试用） |
| `logs/` 目录 | 已创建 | SLURM 输出写入 `logs/` |
| Image-in-parquet 支持 | 已修复 | `datasets.py` 三处 patch（见第 8 节） |

---

## TL;DR — 提交命令

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 1 node × 4 GPU
sbatch scripts/slurm_custom_requeue.sh

# 2 nodes × 4 GPU = 8 GPU（见第 9 节多节点训练）
sbatch scripts/slurm_custom_requeue-2node.sh
```

查看资源 & 监控：
```bash
bash .../scripts/requeue_summary.sh          # 空闲 GPU 概览 (full path: /net/.../kaiwen/scripts/)
bash .../scripts/requeue_summary.sh h200     # 只看 H200
squeue --start -j <jobid>                    # 预估开始时间
squeue -u $USER                              # 作业状态
tail -f logs/starVLA_custom_<jobid>.out      # 实时日志
```

| 排队状态 | 含义 |
|----------|------|
| `PD (Resources)` | 等待空闲 GPU，通常很快 |
| `PD (Priority)` | 有更高优先级作业，可能较久 |
| `PD (ReqNodeNotAvail)` | 预选节点不可用，取消重提交 |
| `R` / `PR` | 运行中 / 被抢占（自动重新排队） |

---

## 1. Prerequisites

### 1.1 Dataset preparation (already done)

The Custom dataset was split into per-task directories for StarVLA.
See `DOC-data/custom-dataset-split.md` for the split procedure.

Verify all 33 task directories:
```bash
for d in playground/Datasets/Custom/*/; do
  [ ! -f "$d/meta/modality.json" ] && echo "MISSING: $d"
done
```

### 1.2 Code patches (required)

The Custom dataset uses **image-in-parquet** format (`dtype: image`, `total_videos: 0`).
Three patches to `starVLA/dataloader/gr00t_lerobot/datasets.py` are required:

| Patch | Location | Symptom without it |
|-------|----------|-------------------|
| 1. `_get_metadata()` — image metadata parsing | line ~624 | `ValueError: 'channel' is not in list` / `KeyError: 'info'` |
| 2. `get_video()` — read images from parquet | line ~1187 | `FileNotFoundError` on non-existent `.mp4` files |
| 3. `__getitem__()` — skip video existence check | line ~2094 | Training hangs silently at 0% (infinite `while True` loop) |

All three patches are already applied. See `DOC-data/custom-dataset-split.md` for full details.

Verification:
```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
python test_image_dataset.py   # Tests 0-4 should all PASS
```

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

For full environment setup details, see `DOC-training/kempner-cluster.md`.

---

## 2. Training

### 2.1 Full training command

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

### 2.2 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `data_root_dir` | `playground/Datasets/Custom` | Per-task LeRobot dirs |
| `data_mix` | `custom_all` | 33 tasks in `mixtures.py` |
| `per_device_batch_size` | 8 | 每 GPU 8 样本 |
| `gradient_accumulation_steps` | 2 | 4 GPU x 8 batch x 2 accum = **64 effective batch** |
| `max_train_steps` | 500,000 | 约 5 个 epoch |
| `save_interval` | 50,000 | 每 50K 步保存 checkpoint |
| `eval_interval` | 5,000 | 每 5K 步评估一次 |
| `is_resume` | true | **关键：** 被抢占后自动从 checkpoint 恢复 |
| `attn_implementation` | sdpa | Required (GLIBC 2.28, flash-attn unavailable) |

### 2.3 与 RoboTwin 训练的区别

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
| `partition` | kempner_h100 (3 天) | kempner_requeue (7 天) |

---

## 3. SLURM Configuration

Full script: `scripts/slurm_custom_requeue.sh`

### 3.1 SBATCH settings

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

### 3.2 Preemption and resume

- `--requeue` 让被抢占的作业自动重新排队
- `--trainer.is_resume true` 让训练代码自动从最近的 checkpoint 恢复
- 两者配合，抢占后训练透明地继续
- `save_interval=50000` 意味着每次抢占最多丢失 50K 步；如果抢占频繁，可减小到 10K-25K

### 3.3 Environment setup in SLURM

脚本中必须：
1. `set +u` before sourcing `~/.bashrc-kaiwen` (avoids `PROMPT_COMMAND: unbound variable`)
2. `module load cuda/12.2.0-fasrc01` (DeepSpeed needs `nvcc`)
3. `conda activate starVLA`

For GPU type switching and other cluster-generic SLURM settings, see `DOC-training/kempner-cluster.md`.

---

## 4. Dataset Info

格式：LeRobot v2.1, image-in-parquet（无视频文件）。33 tasks = 11 base tasks x 3 variants (clean/wp1/wp2), 100 episodes each, 3,300 total. 来源：`DOC-data/custom-dataset-split.md`.

### 4.1 33 个 Task Variants

11 个 base task x 3 variants（clean / wp1 / wp2）：

```
adjust_bottle       beat_block_hammer    blocks_ranking_rgb   blocks_ranking_size
click_alarmclock    handover_block       handover_mic         move_can_pot
move_pillbottle_pad move_stapler_pad     place_empty_cup
```
Each has `_wp1` and `_wp2` variants (33 total).

### 4.2 与 RoboTwin-Randomized 数据集对比

| | RoboTwin-Randomized | Custom |
|---|---|---|
| 图像存储 | MP4 video (AV1) | Image-in-parquet |
| FPS | 15 | 50 |
| 每 task episodes | 500 | 100 |
| 总 tasks | 50 | 33 |
| 总 episodes | 25,000 | 3,300 |
| `data_root_dir` | `playground/Datasets/RoboTwin` | `playground/Datasets/Custom` |
| `data_mix` | `robotwin` | `custom_all` |

---

## 5. Checkpoints and Resume

### 5.1 Checkpoint 结构

```
results/Checkpoints/custom_qwenOFT_requeue/
├── config.yaml                    # 完整训练配置
├── dataset_statistics.json        # action/state 归一化统计量
├── checkpoints/
│   ├── steps_50000/               # 完整训练状态 (model, optimizer, scheduler, RNG)
│   ├── steps_50000_pytorch_model.pt   # 独立模型权重（部署/评估用）
│   ├── steps_100000/ ...
```

### 5.2 Resume logic

When `is_resume=true`, the trainer scans `checkpoints/` for `steps_<N>/` directories, picks the highest `N`, and calls `accelerator.load_state()` to restore all state. Use the same `--run_id` to continue; use a different one to start fresh.

### 5.3 抢占恢复流程

1. 作业被抢占 → SLURM 自动重新排队（`--requeue`）
2. 重新调度到新节点 → `is_resume true` → 自动从最新 checkpoint 恢复
3. 训练从中断处继续

> 两次 checkpoint 之间的训练进度会丢失。抢占频繁时可减小 `save_interval`（如 10K-25K）。

手动恢复（timeout / 取消后）：直接 `sbatch scripts/slurm_custom_requeue.sh`，相同 `run_id` 自动恢复。

### 5.4 Standalone weights

For deployment or evaluation, use `steps_<N>_pytorch_model.pt` (model-only, no optimizer state).

---

## 6. Monitoring

```bash
squeue -u $USER                            # 作业状态
tail -f logs/starVLA_custom_<jobid>.out    # 训练输出
tail -f logs/starVLA_custom_<jobid>.err    # 错误/警告
```

WandB: project `starVLA_Custom`, entity `kaiwenh-17-uiuc`, run `custom_qwenOFT_requeue`.

For GPU monitoring and generic SLURM commands, see `DOC-training/kempner-cluster.md`.

---

## 7. Quick Debugging

### 7.1 用单 task 快速测试（通过 SLURM）

修改 SLURM 脚本中的 `data_mix` 和 `max_train_steps`：
```bash
  --datasets.vla_data.data_mix custom_task1 \    # 只加载 adjust_bottle
  --trainer.max_train_steps 100 \                 # 只跑 100 步
  --trainer.save_interval 50 \
  --run_id custom_debug_test \
```

### 7.2 交互式测试（不用 sbatch）

```bash
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h100 \
  -N 1 --gpus-per-node=4 --cpus-per-task=96 --mem=1440G -t 0-01:00:00

source ~/.bashrc-kaiwen && conda activate starVLA && module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
```

Then run the same training command from section 2.1 with debug overrides:
```bash
  --datasets.vla_data.data_mix custom_task1 \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --run_id custom_interactive_test
```

---

## 8. Image-in-Parquet Patches

Custom 数据集使用 image-in-parquet 格式（`dtype: image`, `total_videos: 0`），与 RoboTwin 的 MP4 视频格式不同。需要对 `starVLA/dataloader/gr00t_lerobot/datasets.py` 做三处修改。

| Patch | Symptom | Cause | Solution |
|-------|---------|-------|----------|
| 1. `_get_metadata()` (line ~624) | `ValueError: 'channel' is not in list` / `KeyError: 'info'` | Only handled video-format keys; image uses `"channels"` (plural) and top-level `fps` | Third fallback branch for image metadata |
| 2. `get_video()` (line ~1187) | `FileNotFoundError` on `.mp4` | No video files in image-in-parquet | `is_image_dataset` property + `_get_images_from_parquet()` decodes PNG from parquet |
| 3. `__getitem__()` (line ~2094) | Hangs at 0%, no errors | `while True` loop checks `os.path.exists(video_path)` — always False for images | Skip video existence check when `is_image_dataset` |

> For full patch code, see `DOC-data/custom-dataset-split.md`.

---

## 9. Multi-Node Training

脚本：`scripts/slurm_custom_requeue-2node.sh`

### 9.1 单节点 vs 双节点对比

| | 1 node (4 GPU) | 2 nodes (8 GPU) |
|---|---|---|
| GPU 总数 | 4 | 8 |
| `per_device_batch_size` | 8 | 8 |
| `gradient_accumulation_steps` | 2 | 1 |
| **Effective batch** | **64** | **64** |
| `max_train_steps` | 500,000 | 250,000 |
| `save_interval` | 50,000 | 25,000 |
| `run_id` | `custom_qwenOFT_requeue` | `custom_qwenOFT_2node` |
| Accelerate config | `deepspeed_zero2.yaml` | `deepspeed_zero2_2node.yaml` |
| Launch method | `accelerate launch` | `srun bash -c 'accelerate launch ...'` |

### 9.2 关键差异

- SLURM: `--nodes=2 --ntasks-per-node=1`
- 通信: `MASTER_ADDR` from first node in `SLURM_JOB_NODELIST`, InfiniBand enabled (`NCCL_IB_DISABLE=0`)
- 启动: `srun bash -c 'accelerate launch ...'`, `SLURM_PROCID` provides `--machine_rank`

### 9.3 注意事项

- 2 节点作业需要同时分配 2 个满足 constraint 的节点，排队时间通常更长
- 任一节点被抢占都会导致整个作业失败（但 `--requeue` + `is_resume` 仍有效）
- 如果 H200 排不上，改 `--constraint=h100`（同时改 `--cpus-per-task=96`）

---

## 10. File Reference

| File | Purpose |
|------|---------|
| `scripts/slurm_custom_requeue.sh` | SLURM 提交脚本 (1 node x 4 GPU) |
| `scripts/slurm_custom_requeue-2node.sh` | SLURM 提交脚本 (2 nodes x 8 GPU) |
| `scripts/split_custom_lerobot.py` / `split_custom_all.sh` | 数据集拆分脚本 |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | 核心数据集类 (patched for image support) |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_all` / `custom_task1` mixture definitions |
| `starVLA/training/train_starvla.py` | Training loop |
| `starVLA/training/trainer_utils/trainer_tools.py` | Checkpoint save/resume logic |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` | Base training YAML config |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 (1 node) |
| `starVLA/config/deepseeds/deepspeed_zero2_2node.yaml` | Accelerate + DeepSpeed ZeRO-2 (2 nodes) |
| `test_image_dataset.py` | Image-in-parquet 数据集加载单元测试 |
| `playground/Datasets/Custom/` | Custom dataset (33 per-task directories) |
| `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` | Pretrained VLM |
| `results/Checkpoints/custom_qwenOFT_requeue/` | 1-node training output |
| `results/Checkpoints/custom_qwenOFT_2node/` | 2-node training output |

---

## Related Documents

| Document | Content |
|----------|---------|
| `DOC-training/kempner-cluster.md` | Kempner cluster setup, GPU types, environment, generic SLURM issues |
| `DOC-data/custom-dataset-split.md` | Dataset split procedure and image-in-parquet patch details |
