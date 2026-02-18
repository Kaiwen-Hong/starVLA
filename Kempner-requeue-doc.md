# StarVLA Custom Dataset Training on kempner_requeue

> **目标：** 在 Kempner kempner_requeue 分区上用 Custom 数据集训练 StarVLA (QwenOFT)。
> 本文档基于实际环境排查结果编写（2026-02-18），所有路径和参数已验证。

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

---

## TL;DR — 提交命令

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
sbatch scripts/slurm_custom_requeue.sh
```

监控：
```bash
squeue -u $USER                                # 查看作业状态
scancel <jobid>                                 # 取消作业
```

日志位置（`<jobid>` 替换为实际 job ID，可从 `squeue` 输出获取）：
```bash
# 日志文件路径格式：logs/starVLA_custom_<jobid>.out / .err
ls logs/starVLA_custom_*.out                   # 查看所有日志文件
tail -f logs/starVLA_custom_<jobid>.out        # 实时看训练输出
tail -f logs/starVLA_custom_<jobid>.err        # 实时看错误/警告
# 如果作业失败，先看 .err 文件找 Traceback
```

---

## 1. 集群信息：kempner_requeue

### 1.1 分区对比

| | kempner_h100 | kempner_requeue |
|---|---|---|
| 最大时间 | 3 天 | **7 天** |
| 抢占 | 否 | **是**（高优先级作业可随时抢占） |
| GPU 类型 | H100 only | **混合**（H100 / H200 / A100） |
| 排队等待 | 长 | 短（闲置资源优先分配） |
| 适用场景 | 需要稳定运行的关键实验 | 长时间训练、可被中断的任务 |

### 1.2 kempner_requeue 中的 GPU 类型

通过 `sinfo -p kempner_requeue -o "%N %f %G"` 查到的实际配置：

| GPU 类型 | 节点范围 | 每节点 GPU | CPU 核心 | 内存 | constraint 标签 |
|----------|---------|-----------|---------|------|----------------|
| **H100 80GB** | holygpu8a[11xxx-17xxx] | 4 | 96 | 1.5 TB | `h100` |
| **H200** | holygpu8a[10xxx] | 4 | 96 | 1.5 TB | `h200` |
| **A100 40GB** | holygpu8a[19xxx] | 4 | 64 | 1 TB | `a100` |

> **必须加 `--constraint=h100` 或 `--constraint=h200`**，否则可能被分配到 A100 40GB 节点（显存不同、性能不同）。
> H200 与 H100 显存相同（80GB），性能更好，且通常排队更短。当 H100 满载时可切换到 H200。

#### 切换 GPU 类型时需改的参数

| 参数 | H100 | H200 | A100 |
|------|------|------|------|
| `--constraint` | h100 | h200 | a100 |
| `--cpus-per-task` | 96 | 64 | 64 |
| `--mem` | 1440G | 1440G | 960G |

> 当前脚本默认使用 **H200**（`--constraint=h200`, `--cpus-per-task=64`）。

### 1.3 抢占机制

- kempner_requeue 的作业随时可能被 kempner_h100 的高优先级作业抢占
- 抢占时作业收到 SIGTERM，然后被 kill
- `#SBATCH --requeue` 让被抢占的作业自动重新排队
- **必须配合 `--trainer.is_resume true`**，训练代码会自动从最近的 checkpoint 恢复
- `save_interval` 不宜设太大，否则抢占后损失过多训练进度

### 1.4 Account

```bash
# 查看你的 account
sacctmgr show assoc user=$USER format=account%40 | grep kempner
```

当前用户的 Kempner account 是 **`kempner_ydu_lab`**。

---

## 2. 环境注意事项

### 2.1 GLIBC 限制

Kempner 节点运行 Rocky Linux 8，GLIBC 版本为 **2.28**：

```bash
ldd --version | head -1
# ldd (GNU libc) 2.28
```

**影响：** `flash-attn` pip wheel 需要 GLIBC >= 2.32，因此 `flash_attention_2` 无法使用。

**解决：** 训练命令中必须指定：
```bash
--framework.qwenvl.attn_implementation sdpa
```

SDPA (Scaled Dot Product Attention) 是 PyTorch 内置的，不依赖 GLIBC 版本，在 H100 上性能差距很小。

### 2.2 环境变量

所有缓存/存储路径在 `~/.bashrc-kaiwen` 中定义，SLURM 脚本通过 `source ~/.bashrc-kaiwen` 加载：

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

> **注意：** `~/.bashrc-kaiwen` 还负责将 PATH 从 home miniforge 切换到 lab miniforge，并初始化 conda。SLURM 脚本中必须在 `conda activate` 之前 source 它。

### 2.3 CUDA Module

DeepSpeed 需要 `nvcc` 来检测 CUDA 版本。Kempner 通过 module system 管理 CUDA：

```bash
module load cuda/12.2.0-fasrc01
```

不加载会导致：
```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

---

## 3. SLURM 脚本详解

脚本位于 `scripts/slurm_custom_requeue.sh`。

### 3.1 SBATCH 参数

```bash
#SBATCH --job-name=starVLA_custom
#SBATCH --partition=kempner_requeue        # 7 天时限，可被抢占
#SBATCH --account=kempner_ydu_lab          # Kempner 专用 account
#SBATCH --constraint=h200                  # H200 80GB（可改为 h100）
#SBATCH --requeue                          # 被抢占后自动重新排队
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64                 # H200 节点有 64 核（H100 为 96）
#SBATCH --gpus-per-node=4                  # 每节点 4× H200
#SBATCH --mem=1440G                        # H200 节点有 1.5 TB
#SBATCH --time=7-00:00:00                  # 最大 7 天
#SBATCH --output=logs/%x_%j.out            # 训练输出日志
#SBATCH --error=logs/%x_%j.err             # 错误/警告日志
```

> 日志文件会写入 `logs/starVLA_custom_<jobid>.out` 和 `.err`。作业失败时先查 `.err` 文件。

### 3.2 训练参数

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

### 3.3 参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `--num_processes` | 4 | 匹配 4 GPU |
| `attn_implementation` | sdpa | GLIBC 2.28 不支持 flash_attention_2 |
| `data_root_dir` | `playground/Datasets/Custom` | Custom 数据集路径（不是 RoboTwin） |
| `data_mix` | `custom_all` | 33 个 task variant，在 `mixtures.py` 注册 |
| `per_device_batch_size` | 8 | 每 GPU 8 样本 |
| `gradient_accumulation_steps` | 2 | 4 GPU × 8 batch × 2 accum = **64 effective batch** |
| `max_train_steps` | 500,000 | 约 5 个 epoch（custom 数据集较小） |
| `save_interval` | 50,000 | 每 50K 步保存 checkpoint（约 2 次/epoch） |
| `eval_interval` | 5,000 | 每 5K 步评估一次 |
| `is_resume` | true | **关键：** 被抢占后自动从 checkpoint 恢复 |

### 3.4 与 RoboTwin 训练的区别

| 参数 | RoboTwin | Custom |
|------|----------|--------|
| `data_root_dir` | `playground/Datasets/RoboTwin` | `playground/Datasets/Custom` |
| `data_mix` | `robotwin` (50 tasks) | `custom_all` (33 tasks) |
| `max_train_steps` | 100,000 | 500,000（数据集小 ~5x，需更多 epoch） |
| `save_interval` | 10,000 | 50,000 |
| `eval_interval` | 1,000 | 5,000 |
| `partition` | kempner_h100 (3 天) | kempner_requeue (7 天) |

---

## 4. Custom 数据集信息

| 属性 | 值 |
|------|-----|
| 位置 | `playground/Datasets/Custom/` |
| 格式 | LeRobot v2.1, image-in-parquet（无视频文件） |
| Task 数 | 33（11 base tasks × 3 variants: clean, wp1, wp2） |
| 每 task episodes | 100 |
| 总 episodes | 3,300 |
| FPS | 50 |
| Features | 14-dim state/action, 3 cameras |
| 来源 | 从 `custom_all_repo` 合并数据集拆分（见 `0218-reuse-custom-lerobot-dataset.md`） |

### 4.1 33 个 Task Variants

11 个 base task，每个有 3 个变体（clean / wp1 / wp2）：

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

## 5. Checkpoint 与抢占恢复

### 5.1 Checkpoint 结构

```
results/Checkpoints/custom_qwenOFT_requeue/
├── config.yaml                     # 自动保存的完整训练配置
├── dataset_statistics.json         # action/state 归一化统计量
├── checkpoints/
│   ├── steps_50000/
│   │   └── pytorch_model.pt
│   ├── steps_100000/
│   │   └── pytorch_model.pt
│   └── ...
```

### 5.2 抢占恢复流程

1. 作业被抢占 → SLURM 自动重新排队（`--requeue`）
2. 作业重新调度到新节点 → 脚本从头执行
3. `--trainer.is_resume true` → 训练代码自动查找 `run_root_dir/run_id/` 下最新 checkpoint
4. 从最近的 checkpoint 恢复模型权重、优化器状态、RNG 状态等
5. 训练从中断处继续

> **注意：** 两次 checkpoint 之间的训练进度会丢失。`save_interval=50000` 意味着最多丢失 50K 步。如果抢占频繁，可以减小 `save_interval`（如 25000 或 10000），但会增加存储开销。

### 5.3 手动恢复

如果作业 timeout 或手动取消后想继续训练：

```bash
# 直接重新提交即可，is_resume=true 会自动恢复
sbatch scripts/slurm_custom_requeue.sh
```

不需要修改任何参数。`run_id` 相同时，训练代码会自动找到之前的 checkpoint。

---

## 6. 监控

### 6.1 作业状态

```bash
# 查看当前作业
squeue -u $USER

# 状态含义：
#   PD = Pending (排队中)
#   R  = Running
#   CG = Completing
#   PR = Preempted (被抢占)
```

### 6.2 训练日志

```bash
# 实时查看输出
tail -f logs/starVLA_custom_<jobid>.out

# 查看错误
tail -f logs/starVLA_custom_<jobid>.err
```

### 6.3 WandB

训练日志会上传到 WandB：
- Project: `starVLA_Custom`
- Entity: `kaiwenh-17-uiuc`
- Run: `custom_qwenOFT_requeue`

### 6.4 GPU 使用（需要在计算节点上）

```bash
# 交互式进入运行中的节点
srun --jobid=<jobid> --pty bash

# 查看 GPU 使用
nvidia-smi
watch -n 5 nvidia-smi
```

---

## 7. 调试与快速验证

如果想先快速验证脚本能跑通，可以：

### 7.1 用单 task 快速测试

修改 SLURM 脚本中的 `data_mix` 和 `max_train_steps`：

```bash
  --datasets.vla_data.data_mix custom_task1 \    # 只加载 adjust_bottle
  --trainer.max_train_steps 100 \                 # 只跑 100 步
  --trainer.save_interval 50 \
  --run_id custom_debug_test \
```

### 7.2 交互式测试（不用 sbatch）

```bash
# 申请一个交互式 GPU 节点
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h100 \
  -N 1 --gpus-per-node=4 --cpus-per-task=96 --mem=1440G -t 0-01:00:00

# 然后在节点上手动运行
source ~/.bashrc-kaiwen
conda activate starVLA
module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 跑几步看看
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

## 8. 常见问题

### GLIBC / flash_attention_2

```
ImportError: /lib64/libc.so.6: version `GLIBC_2.32' not found
```

**原因：** Kempner 节点 GLIBC 2.28，flash-attn 需要 2.32+。
**解决：** `--framework.qwenvl.attn_implementation sdpa`（脚本中已配置）。

### DeepSpeed nvcc not found

```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

**原因：** 没有 `module load cuda`。
**解决：** SLURM 脚本中已包含 `module load cuda/12.2.0-fasrc01`。

### 被抢占后丢失进度

**原因：** 抢占发生在两次 checkpoint 之间。
**解决：** 减小 `save_interval`。当前设为 50,000 步；如果抢占频繁，改为 10,000-25,000。

### 排队太久

```bash
# 查看 kempner_requeue 空闲节点
sinfo -p kempner_requeue -o "%N %G %t" | grep -E "idle|mix"
```

如果持续排不上，检查是否有大量高优先级作业在运行：
```bash
squeue -p kempner_requeue | wc -l
```

### WandB 未记录

确保没有设置 `WANDB_MODE=disabled`。检查：
```bash
echo $WANDB_MODE   # 应该为空
```

### 数据集找不到

```
FileNotFoundError: .../playground/Datasets/Custom/<task>/meta/modality.json
```

验证数据集完整性：
```bash
for d in playground/Datasets/Custom/*/; do
  if [ ! -f "$d/meta/modality.json" ]; then
    echo "MISSING: $d"
  fi
done
```

### `set -u` + bashrc PROMPT_COMMAND 报错

```
.bashrc-kaiwen: line 14: PROMPT_COMMAND: unbound variable
```

**原因：** SLURM 脚本中 `set -euo pipefail` 在 `source ~/.bashrc-kaiwen` 之前生效，而 bashrc 中引用了未定义的 `PROMPT_COMMAND` 变量，`-u` 选项导致立即退出。
**解决：** 在 source bashrc 之前先 `set +u`，source 完之后再 `set -euo pipefail`。脚本中已修复：

```bash
# ── Environment setup ──
set +u
source ~/.bashrc-kaiwen
set -euo pipefail
```

### Image 数据集 `KeyError: 'info'` / `ValueError: 'channel' is not in list`

```
ValueError: 'channel' is not in list
KeyError: 'info'
```

**原因：** Custom 数据集使用 image-in-parquet 格式（`dtype: image`），其 `info.json` 中 names 为 `["channels", "height", "width"]`（复数 `channels`）。而 `datasets.py` 的 `_get_metadata()` 只处理了两种 video 格式变体：`"channel"`（单数）和 `le_video_meta["info"]["video.channels"]`，都不匹配 image 格式。
**解决：** 已在 `starVLA/dataloader/gr00t_lerobot/datasets.py:624` 增加第三层 fallback，支持 image 格式的 `"channels"`（复数），并从顶层 `info.json` 读取 `fps`。

### ReqNodeNotAvail, UnavailableNodes

```
(ReqNodeNotAvail, UnavailableNodes:holygpu8a[...])
```

**原因：** SLURM 调度器为作业预选了节点，但这些节点处于 DOWN/DRAIN 状态（通常在维护中）。作业会一直等待这些不可用节点。
**排查：**
```bash
# 检查节点状态
scontrol show node holygpu8a<xxxxx> | grep -E "NodeName|State|Reason"
```
**解决：** 取消作业重新提交，SLURM 会重新调度到其他可用节点：
```bash
scancel <jobid>
sbatch scripts/slurm_custom_requeue.sh
```

### 排队状态 `(Resources)` vs `(Priority)`

- **`(Resources)`**：SLURM 已准备调度你的作业，只是当前没有足够空闲资源。通常很快就能跑上。
- **`(Priority)`**：有更高优先级的作业排在前面。可能需要等待更久。

查看预估开始时间：
```bash
squeue --start -j <jobid>
```

### 集群满载时怎么办

用监控脚本查看各 GPU 类型的实时空闲情况：
```bash
# 查看 requeue 分区所有 GPU 类型的详细状态
bash ~/kaiwen/scripts/requeue_summary.sh

# 只看某种 GPU
bash ~/kaiwen/scripts/requeue_summary.sh h200
bash ~/kaiwen/scripts/requeue_summary.sh h100

# 查看所有分区概览
bash ~/kaiwen/scripts/most_empty_partition.sh
```

如果 H100 满载（>95%），考虑切换到 H200 或等待非高峰时段。修改 `--constraint` 和 `--cpus-per-task` 即可（见 1.2 节的切换表）。

---

## 9. 文件清单

| 文件 | 用途 |
|------|------|
| `scripts/slurm_custom_requeue.sh` | SLURM 提交脚本（本文档的核心） |
| `scripts/split_custom_lerobot.py` | 拆分合并数据集为 per-task 目录 |
| `scripts/split_custom_all.sh` | 拆分脚本的 shell wrapper |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_all` / `custom_task1` 混合定义 |
| `examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml` | 基础训练 YAML 配置 |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 配置 |
| `starVLA/config/deepseeds/ds_config.yaml` | DeepSpeed 具体参数（BF16, ZeRO-2） |
| `~/.bashrc-kaiwen` | 环境变量、conda 初始化 |
| `playground/Datasets/Custom/` | Custom 数据集（33 task 目录） |
| `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` | 预训练 VLM |
| `results/Checkpoints/custom_qwenOFT_requeue/` | 训练输出（checkpoint、config、stats） |
| `logs/` | SLURM stdout/stderr 日志 |
