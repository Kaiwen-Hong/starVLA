# 过渡记录：v40/v42/v50/v52/v60/v62-ee 接入 StarVLA

> **日期：** 2026-03-23
> **实验：** v40/v42/v50/v52/v60/v62-ee（6 个 16D EE 数据集，来自 data_custom_0320）
> **框架：** QwenPI (Qwen2.5-VL-3B-Instruct-Action + DiT-B)
> **前置文档：** `r-preference/0315-transition-from-ar_research_kempner.md`（v30-ee 首次接入的完整记录）

---

## 0. 背景

v30-ee 首次将 16D 末端执行器（EE）空间数据接入了 StarVLA。那次建立了全部基础设施：
- `RobotwinEEDataConfig`（data_config.py）
- `modality_ee_16d.json`
- `split_custom_lerobot.py` 的 `--modality-file` 参数
- `"robotwin_ee"` robot type

**这次不需要修改任何基础设施代码。** 只需要：
1. 在 `mixtures.py` 中注册 6 个新 mixture
2. 创建 6 个 split 脚本
3. 创建 6 个训练脚本

---

## 1. 6 个新数据集概览

所有数据来自 `data_custom_0320/`，由 ar-research-kempner 流水线生成。

| 版本 | mixture 名 | 任务 1 | 任务 2 | 变体数 | episodes |
|------|-----------|--------|--------|--------|----------|
| **v40** | `custom_v0320_v40_ee` | place_cup_tray (clean1, wp5) | place_stapler_stand (clean1, wp5) | 4 | 4×250 = 1000 |
| **v42** | `custom_v0320_v42_ee` | place_cup_tray (clean1, wp5) | place_stapler_stand (clean1) | 3 | 3×250 = 750 |
| **v50** | `custom_v0320_v50_ee` | place_cup5_tray1 (clean1, wp5) | place_stapler_stand (clean1, wp5) | 4 | 4×250 = 1000 |
| **v52** | `custom_v0320_v52_ee` | place_cup5_tray1 (clean1, wp5) | place_stapler_stand (clean1) | 3 | 3×250 = 750 |
| **v60** | `custom_v0320_v60_ee` | place_cup5_tray5 (clean1, wp5) | place_stapler_stand (clean1, wp5) | 4 | 4×250 = 1000 |
| **v62** | `custom_v0320_v62_ee` | place_cup5_tray5 (clean1, wp5) | place_stapler_stand (clean1) | 3 | 3×250 = 750 |

### 版本间的关系

- **v40 vs v42**：v42 = v40 去掉 `place_stapler_stand_wp5`
- **v50 vs v52**：v52 = v50 去掉 `place_stapler_stand_wp5`
- **v60 vs v62**：v62 = v60 去掉 `place_stapler_stand_wp5`
- **v40 vs v50 vs v60**：`place_stapler_stand` 数据相同，区别在于搭配的 cup/tray 变体不同（cup_tray vs cup5_tray1 vs cup5_tray5）

### 共享数据说明

`place_stapler_stand_clean1` 和 `place_stapler_stand_wp5` 在 v40/v50/v60（以及 v42/v52/v62）之间共享**完全相同**的原始数据（都来自 `data_custom_0320/`）。因此在 `playground/Datasets/CustomEE/` 中，`place_stapler_stand_clean1_ee/` 和 `place_stapler_stand_wp5_ee/` 目录无论被哪个版本的 split 脚本写入，内容都一样，互相覆盖没有问题。

---

## 2. 两阶段流水线

```
阶段 A: ar-research-kempner                 阶段 B: starVLA
────────────────────────                    ──────────────────

Kempner-all-procedure-0320-vXX-ee.sh        split_custom_v0320_vXX_ee.sh
  Step 1: 验证原始数据                          │ (读取合并的 LeRobot repo)
  Step 2: 处理 → HDF5 (EE space)               │ (使用 --modality-file modality_ee_16d.json)
  Step 3: 生成合并的 LeRobot 数据集   ──────→    ▼
  Step 4: 计算 norm stats               playground/Datasets/CustomEE/
  Step 5: pi05 训练                        ├── place_cup_tray_clean1_ee/     (v40/v42)
  Step 6: 上传 HuggingFace                 ├── place_cup_tray_wp5_ee/        (v40/v42)
                                           ├── place_cup5_tray1_clean1_ee/   (v50/v52)
产出:                                       ├── place_cup5_tray1_wp5_ee/      (v50/v52)
  $CACHE/lerobot/custom_v0320_vXX_ee_repo/  ├── place_cup5_tray5_clean1_ee/   (v60/v62)
                                           ├── place_cup5_tray5_wp5_ee/      (v60/v62)
                                           ├── place_stapler_stand_clean1_ee/ (共享)
                                           └── place_stapler_stand_wp5_ee/   (共享)
                                              │
                                              ▼
                                           0320-vXX-training.sh
                                             读 mixtures.py → "custom_v0320_vXX_ee"
                                             读 data_config.py → RobotwinEEDataConfig
                                             读 meta/modality.json → 16D EE
```

---

## 3. 阶段 A：ar-research-kempner（生成数据 + pi05 训练）

### 3.1 脚本位置

```
ar-research-kempner/policy/pi05/
  ├── Kempner-all-procedure-0320-v40-ee.sh
  ├── Kempner-all-procedure-0320-v42-ee.sh
  ├── Kempner-all-procedure-0320-v50-ee.sh
  ├── Kempner-all-procedure-0320-v52-ee.sh
  ├── Kempner-all-procedure-0320-v60-ee.sh
  └── Kempner-all-procedure-0320-v62-ee.sh
```

### 3.2 运行

在 GPU 节点上（4×H100），依次或分批运行：

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner/policy/pi05
conda deactivate

bash Kempner-all-procedure-0320-v40-ee.sh 2>&1 | tee v0320_v40_ee_run.log
bash Kempner-all-procedure-0320-v42-ee.sh 2>&1 | tee v0320_v42_ee_run.log
bash Kempner-all-procedure-0320-v50-ee.sh 2>&1 | tee v0320_v50_ee_run.log
bash Kempner-all-procedure-0320-v52-ee.sh 2>&1 | tee v0320_v52_ee_run.log
bash Kempner-all-procedure-0320-v60-ee.sh 2>&1 | tee v0320_v60_ee_run.log
bash Kempner-all-procedure-0320-v62-ee.sh 2>&1 | tee v0320_v62_ee_run.log
```

每个脚本自动执行 6 步：验证数据 → 处理 HDF5(EE) → 生成 LeRobot → norm stats → pi05 训练 → 上传 HF。

### 3.3 产出

完成后生成 6 个合并的 LeRobot 数据集：

```
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v40_ee_repo/
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v42_ee_repo/
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v50_ee_repo/
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v52_ee_repo/
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v60_ee_repo/
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v62_ee_repo/
```

其中 `$LAB_ROOT = /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen`。

### 3.4 注意事项

- 脚本会在 Step 3 后自动 `rm -rf training_data/custom_v0320_vXX_ee/`（清理中间 HDF5）
- 脚本会在 Step 5 后自动 `rm -rf ${KEMPNER_CACHE}/huggingface/datasets/parquet/`（清理 Arrow cache）
- 这些清理行为是正常的，LeRobot repo 才是数据的最终来源

---

## 4. 阶段 B：starVLA（split + QwenPI 训练）

### 4.1 前提条件

- 阶段 A 的 LeRobot 数据集已生成（至少你要训练的版本已完成）
- starVLA conda 环境可用

验证 LeRobot repos 存在：
```bash
LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
for v in 40 42 50 52 60 62; do
  repo="${LAB_ROOT}/.cache/huggingface/lerobot/custom_v0320_v${v}_ee_repo"
  if [ -f "${repo}/meta/info.json" ]; then
    echo "OK: v${v}"
  else
    echo "MISSING: v${v} — ${repo}"
  fi
done
```

### 4.2 修改/新建的文件

| 操作 | 文件 | 说明 |
|------|------|------|
| **修改** | `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 新增 6 个 mixture 条目 |
| **新建** | `scripts/split_custom_v0320_v{40,42,50,52,60,62}_ee.sh` | 6 个 split 脚本 |
| **新建** | `scripts/slurm_v0320_v{40,42,50,52,60,62}_ee_requeue.sh` | 6 个 SLURM 训练脚本 |
| **新建** | `r-preference/0320-v{40,42,50,52,60,62}-training.sh` | 6 个交互式训练脚本 |

**不需要修改的文件：**
| 文件 | 原因 |
|------|------|
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | `RobotwinEEDataConfig` + `"robotwin_ee"` 已注册 |
| `examples/RobotwinEE/train_files/modality_ee_16d.json` | 所有 16D 双臂 EE 数据集共用 |
| `scripts/split_custom_lerobot.py` | `--modality-file` 参数已有 |

### 4.3 mixtures.py 中新增的条目

```python
# ── Custom v0320 v40 EE: 4 variants (cup_tray + stapler_stand, clean1/wp5) ──
"custom_v0320_v40_ee": [
    ("place_cup_tray_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup_tray_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_wp5_ee", 1.0, "robotwin_ee"),
],

# ── Custom v0320 v42 EE: 3 variants ──
"custom_v0320_v42_ee": [
    ("place_cup_tray_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup_tray_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
],

# ── Custom v0320 v50 EE: 4 variants ──
"custom_v0320_v50_ee": [
    ("place_cup5_tray1_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup5_tray1_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_wp5_ee", 1.0, "robotwin_ee"),
],

# ── Custom v0320 v52 EE: 3 variants ──
"custom_v0320_v52_ee": [
    ("place_cup5_tray1_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup5_tray1_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
],

# ── Custom v0320 v60 EE: 4 variants ──
"custom_v0320_v60_ee": [
    ("place_cup5_tray5_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup5_tray5_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_wp5_ee", 1.0, "robotwin_ee"),
],

# ── Custom v0320 v62 EE: 3 variants ──
"custom_v0320_v62_ee": [
    ("place_cup5_tray5_clean1_ee", 1.0, "robotwin_ee"),
    ("place_cup5_tray5_wp5_ee", 1.0, "robotwin_ee"),
    ("place_stapler_stand_clean1_ee", 1.0, "robotwin_ee"),
],
```

### 4.4 每个版本的 episode 顺序（split 脚本的 TASKS 顺序）

Episode 顺序按 HDF5 目录名字母排序。**TASKS 数组的顺序必须与此一致**，否则 episode 切分会错位。

**v40**（4 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup_tray_clean1_ee` |
| 250-499 | `place_cup_tray_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |
| 750-999 | `place_stapler_stand_wp5_ee` |

**v42**（3 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup_tray_clean1_ee` |
| 250-499 | `place_cup_tray_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |

**v50**（4 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup5_tray1_clean1_ee` |
| 250-499 | `place_cup5_tray1_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |
| 750-999 | `place_stapler_stand_wp5_ee` |

**v52**（3 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup5_tray1_clean1_ee` |
| 250-499 | `place_cup5_tray1_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |

**v60**（4 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup5_tray5_clean1_ee` |
| 250-499 | `place_cup5_tray5_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |
| 750-999 | `place_stapler_stand_wp5_ee` |

**v62**（3 variants, 250 each）：
| Episodes | 任务名 |
|----------|--------|
| 0-249 | `place_cup5_tray5_clean1_ee` |
| 250-499 | `place_cup5_tray5_wp5_ee` |
| 500-749 | `place_stapler_stand_clean1_ee` |

### 4.5 训练参数差异表

6 个版本之间只有 3 个参数不同，其余全部相同：

| 参数 | v40 | v42 | v50 | v52 | v60 | v62 |
|------|-----|-----|-----|-----|-----|-----|
| `data_mix` | `custom_v0320_v40_ee` | `custom_v0320_v42_ee` | `custom_v0320_v50_ee` | `custom_v0320_v52_ee` | `custom_v0320_v60_ee` | `custom_v0320_v62_ee` |
| `run_id` | `v0320_v40_ee_qwenPI_requeue` | `v0320_v42_ee_...` | `v0320_v50_ee_...` | `v0320_v52_ee_...` | `v0320_v60_ee_...` | `v0320_v62_ee_...` |
| `wandb_project` | `starVLA_v40` | `starVLA_v42` | `starVLA_v50` | `starVLA_v52` | `starVLA_v60` | `starVLA_v62` |

所有版本共享的固定参数：
```
framework.name              = QwenPI
action_dim / state_dim      = 16
action_model_type           = DiT-B
hidden_size / action_hidden = 1024
future_action_window_size   = 15
data_root_dir               = playground/Datasets/CustomEE
per_device_batch_size       = 8
max_train_steps             = 50000
save_interval               = 10000
gradient_accumulation_steps = 2
is_resume                   = true
wandb_entity                = hca
```

---

## 5. 操作步骤（复制粘贴即可）

### 5.1 Split 数据集（可并行，在任意节点上运行）

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 并行拆分所有 6 个数据集
bash scripts/split_custom_v0320_v40_ee.sh &
bash scripts/split_custom_v0320_v42_ee.sh &
bash scripts/split_custom_v0320_v50_ee.sh &
bash scripts/split_custom_v0320_v52_ee.sh &
bash scripts/split_custom_v0320_v60_ee.sh &
bash scripts/split_custom_v0320_v62_ee.sh &
wait
echo "All 6 splits done"

# 验证
ls playground/Datasets/CustomEE/
# 期望看到: place_cup_tray_clean1_ee, place_cup_tray_wp5_ee,
#           place_cup5_tray1_clean1_ee, place_cup5_tray1_wp5_ee,
#           place_cup5_tray5_clean1_ee, place_cup5_tray5_wp5_ee,
#           place_stapler_stand_clean1_ee, place_stapler_stand_wp5_ee
#           (加上之前 v30 系列的目录)
```

Split 脚本会自动验证 16D modality，如果看到 `OK: 16D confirmed` 就没问题。

### 5.2 训练（需要 GPU 节点，每次只能跑 1 个）

**方式 A：通过 SLURM sbatch 提交（推荐，支持 preemption 自动 resume）**

```bash
sbatch scripts/slurm_v0320_v40_ee_requeue.sh
sbatch scripts/slurm_v0320_v42_ee_requeue.sh
sbatch scripts/slurm_v0320_v50_ee_requeue.sh
sbatch scripts/slurm_v0320_v52_ee_requeue.sh
sbatch scripts/slurm_v0320_v60_ee_requeue.sh
sbatch scripts/slurm_v0320_v62_ee_requeue.sh
```

6 个 job 可以同时提交，SLURM 自动排队。每个 job 需要 1 node × 4 GPU。

**方式 B：已在 GPU 节点上（salloc），用交互式脚本**

```bash
# 先申请节点（如果还没有的话）
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
  -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-07:00:00

# 然后依次运行（每个占满 4 GPU，只能串行）
bash r-preference/0320-v40-training.sh
bash r-preference/0320-v42-training.sh
bash r-preference/0320-v50-training.sh
bash r-preference/0320-v52-training.sh
bash r-preference/0320-v60-training.sh
bash r-preference/0320-v62-training.sh
```

### 5.3 调试测试（正式训练前 sanity check）

任选一个版本（比如 v40），把 `max_train_steps` 改小：

```bash
source ~/.bashrc-kaiwen && conda activate starVLA && module load cuda/12.2.0-fasrc01
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  starVLA/training/train_starvla.py \
  --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation sdpa \
  --framework.action_model.action_dim 16 \
  --framework.action_model.state_dim 16 \
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
  --datasets.vla_data.data_root_dir playground/Datasets/CustomEE \
  --datasets.vla_data.data_mix custom_v0320_v40_ee \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id v0320_v40_ee_debug_test
```

跑完 20 步没崩 = 流水线正确。**记得删掉 debug checkpoint**：
```bash
rm -rf results/Checkpoints/v0320_v40_ee_debug_test/
```

---

## 6. 训练产出

### 6.1 Checkpoint 路径

```
results/Checkpoints/
  ├── v0320_v40_ee_qwenPI_requeue/
  ├── v0320_v42_ee_qwenPI_requeue/
  ├── v0320_v50_ee_qwenPI_requeue/
  ├── v0320_v52_ee_qwenPI_requeue/
  ├── v0320_v60_ee_qwenPI_requeue/
  └── v0320_v62_ee_qwenPI_requeue/
```

每个目录下：
- `checkpoints/steps_10000_pytorch_model.pt` ... `steps_50000_pytorch_model.pt`
- `dataset_statistics.json`（归一化统计量，resume 时复用）

### 6.2 WandB

| 版本 | WandB 项目 | entity |
|------|-----------|--------|
| v40 | `starVLA_v40` | `hca` |
| v42 | `starVLA_v42` | `hca` |
| v50 | `starVLA_v50` | `hca` |
| v52 | `starVLA_v52` | `hca` |
| v60 | `starVLA_v60` | `hca` |
| v62 | `starVLA_v62` | `hca` |

---

## 7. 常见问题与排错

### 7.1 `FileNotFoundError: .../custom_v0320_vXX_ee_repo/meta/info.json`

阶段 A 还没跑完（或没跑）。先去 ar-research-kempner 运行对应的 all-in-one 脚本。

### 7.2 `KeyError: 'place_cup_tray_clean1_ee'` 或类似

Split 还没跑，`playground/Datasets/CustomEE/` 下没有对应目录。先运行 split 脚本。

### 7.3 训练 resume 后数据集发生了变化

删除对应 run_id 下的 `dataset_statistics.json`，强制重新计算归一化统计量：
```bash
rm results/Checkpoints/v0320_v40_ee_qwenPI_requeue/dataset_statistics.json
```

### 7.4 想只跑其中几个版本

完全可以。Split 和训练都是独立的，跑 v40 不需要先跑 v42。只要对应的 LeRobot repo 存在就行。

### 7.5 命名一致性检查

如果遇到任何 `KeyError`，检查以下三者的名字是否完全匹配：
```
split 脚本的 TASKS 数组  ←→  mixtures.py 的 tuple 第一项  ←→  CustomEE/ 下的目录名
```

参考 v30 文档 §3.2 的命名映射链说明。

---

## 8. 文件速查表

所有路径相对于 `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA/`。

### 新增/修改的文件

| 文件 | 状态 | 用途 |
|------|------|------|
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 修改 | 新增 6 个 mixture |
| `scripts/split_custom_v0320_v40_ee.sh` | 新建 | v40 split (4 tasks, 250 ep each) |
| `scripts/split_custom_v0320_v42_ee.sh` | 新建 | v42 split (3 tasks, 250 ep each) |
| `scripts/split_custom_v0320_v50_ee.sh` | 新建 | v50 split (4 tasks, 250 ep each) |
| `scripts/split_custom_v0320_v52_ee.sh` | 新建 | v52 split (3 tasks, 250 ep each) |
| `scripts/split_custom_v0320_v60_ee.sh` | 新建 | v60 split (4 tasks, 250 ep each) |
| `scripts/split_custom_v0320_v62_ee.sh` | 新建 | v62 split (3 tasks, 250 ep each) |
| `scripts/slurm_v0320_v{40,42,50,52,60,62}_ee_requeue.sh` | 新建 | 6 个 SLURM 训练脚本 |
| `r-preference/0320-v{40,42,50,52,60,62}-training.sh` | 新建 | 6 个交互式训练脚本 |

### 复用的基础设施文件（v30 时已建立）

| 文件 | 用途 |
|------|------|
| `scripts/split_custom_lerobot.py` | split 主程序（`--modality-file`） |
| `examples/RobotwinEE/train_files/modality_ee_16d.json` | 16D EE 模态映射 |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | `RobotwinEEDataConfig` + `"robotwin_ee"` |

### 数据集路径

| 项目 | 路径 |
|------|------|
| 合并的 LeRobot repos | `$LAB_ROOT/.cache/huggingface/lerobot/custom_v0320_v{40,42,50,52,60,62}_ee_repo/` |
| 拆分输出 | `playground/Datasets/CustomEE/` |
| 训练输出 | `results/Checkpoints/v0320_v{40,42,50,52,60,62}_ee_qwenPI_requeue/` |

---

## 9. 未来：添加更多 0320 系列 EE 数据集

如果需要添加更多基于 `data_custom_0320/` 的 EE 数据集版本（比如 v70、v72），步骤：

1. 在 ar-research-kempner 中创建对应的 `Kempner-all-procedure-0320-vXX-ee.sh` 并运行
2. 在 starVLA 中复制一个 split 脚本，修改 `SRC`、`TASKS`、episodes-per-task
3. 在 `mixtures.py` 中添加新 mixture，robot_type 用 `"robotwin_ee"`
4. 复制一个训练脚本，修改 `data_mix`、`run_id`、`wandb_project`
5. **不需要改** `data_config.py` 和 `modality_ee_16d.json`
