# 过渡记录：v30-ee（16D 末端执行器空间）接入 StarVLA

> **日期：** 2026-03-16
> **实验：** v30-ee (custom_v0309_v30_ee)
> **框架：** QwenPI (Qwen2.5-VL-3B-Instruct-Action + DiT-B)
> **前置文档：** `r-preference/0315-initial.md`（完整端到端指南）

---

## 1. 做了什么

我们把一个 **16 维末端执行器（EE）空间** 的数据集接入了 StarVLA 训练流程。在此之前 StarVLA 只支持：
- **14D 关节空间**（RoboTwin/Custom）：`left_joints(6) + left_gripper(1) + right_joints(6) + right_gripper(1)`
- **10D EE**（FastUMI）：`eef_pos(3) + eef_rot6d(6) + gripper(1)`

新的 v30-ee 数据集使用 **16D EE**：
```
left_pos(3) + left_quat(4) + left_gripper(1) + right_pos(3) + right_quat(4) + right_gripper(1)
```

核心区别：使用**四元数旋转**（4D，范围 [-1,1]），不是 rotation_6d（6D）。因此归一化用 `min_max`，而不是 FastUMI 对 rot6d 用的 `none`。

### 1.1 修改/新建的文件

| 操作 | 文件 | 做了什么 / 为什么 |
|------|------|-------------------|
| **修改** | `scripts/split_custom_lerobot.py` | 加了 `--modality-file` 命令行参数，让 split 脚本可以写入自定义的 modality JSON，而不是写死的 14D |
| **新建** | `examples/RobotwinEE/train_files/modality_ee_16d.json` | 16D 模态定义——把 parquet 列切片映射到具名的 action/state/video key |
| **新建** | `scripts/split_custom_v0309_v30_ee.sh` | Shell 包装脚本，将合并的 v30-ee LeRobot 数据集拆分成 4 个 per-task 目录到 `playground/Datasets/CustomEE/` |
| **修改** | `starVLA/dataloader/gr00t_lerobot/data_config.py` | 新增 `RobotwinEEDataConfig` 类（16D key、pos/quat 用 min_max、gripper 用 binary），注册为 `"robotwin_ee"` |
| **修改** | `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 新增 `"custom_v0309_v30_ee"` mixture（4 个任务变体，全部使用 `"robotwin_ee"`） |
| **新建** | `scripts/slurm_v0309_v30_ee_requeue.sh` | SLURM 训练脚本：QwenPI，action_dim=16，state_dim=16，1 node x 4 H200 |

### 1.2 每个改动的原因

**`--modality-file`（split_custom_lerobot.py）：**
Split 脚本会在每个 per-task 数据集目录写入 `meta/modality.json`。这个文件告诉 StarVLA dataloader 怎么把原始 parquet 列切分成具名的 modality key（比如 `action.left_pos` = action 列 0:3）。旧脚本里 14D modality 是写死的。与其复制整个 split 脚本，不如加一个可选参数。已有的 14D 脚本不受影响——它们不传 `--modality-file`，用默认值。

**`modality_ee_16d.json`：**
这是 split 脚本消费、后续被 StarVLA dataloader 读取的映射文件。定义了 action/state 的 6 个子 key（left_pos, left_quat, left_gripper, right_pos, right_quat, right_gripper）及其正确的切片索引。

**`RobotwinEEDataConfig`：**
StarVLA 的 dataloader 需要知道：(a) 读哪些 modality key，(b) 每个怎么归一化，(c) 应用什么 transform。这个类模仿了 `AgilexDataConfig`（14D 关节空间）的结构，但使用 EE 空间的 key 和适合四元数的归一化。没有它，dataloader 不知道怎么处理 16D 数据。

**独立的 `CustomEE/` 目录：**
14D 关节空间数据在 `playground/Datasets/Custom/`，EE 空间数据放到 `playground/Datasets/CustomEE/`，避免混淆不兼容的动作表示。

---

## 2. v30-ee 数据集

### 2.1 来源

由 ar-research-kempner 流水线以 EE 空间输出生成。合并后的 LeRobot 数据集在：
```
$LAB_ROOT/.cache/huggingface/lerobot/custom_v0309_v30_ee_repo/
```

统计：**400 集（episodes），约 71K 帧，4 个任务变体**。

### 2.2 任务变体（episode 顺序）

| Episodes | 任务名 | 说明 |
|----------|--------|------|
| 0-99 | `place_container_plate_clean1_ee` | 干净环境，无障碍物 |
| 100-199 | `place_container_plate_wp4_ee` | 有 waypoint 4 障碍物 |
| 200-299 | `place_object_stand_clean1_ee` | 干净环境，无障碍物 |
| 300-399 | `place_object_stand_wp4_ee` | 有 waypoint 4 障碍物 |

顺序按 HDF5 目录名字母排序（和所有其他 custom 数据集惯例一致）。

### 2.3 Action/State 布局（16D）

| 索引 | 组件 | 维度 | 说明 |
|------|------|------|------|
| 0-2 | `left_pos` | 3 | 左臂 XYZ 位置 |
| 3-6 | `left_quat` | 4 | 左臂四元数 (w, x, y, z) |
| 7 | `left_gripper` | 1 | 左夹爪（0=张开，1=关闭） |
| 8-10 | `right_pos` | 3 | 右臂 XYZ 位置 |
| 11-14 | `right_quat` | 4 | 右臂四元数 (w, x, y, z) |
| 15 | `right_gripper` | 1 | 右夹爪（0=张开，1=关闭） |

### 2.4 归一化策略

| 组件 | 模式 | 原因 |
|------|------|------|
| `left_pos` / `right_pos` | `min_max` | 位置值由工作空间限定 |
| `left_quat` / `right_quat` | `min_max` | 四元数分量天然有界 [-1, 1] |
| `left_gripper` / `right_gripper` | `binary` | 离散开/关 |

四元数用 `min_max`（而不是 FastUMI 对 rot6d 用的 `none`），因为四元数分量本身有界，归一化有助于训练。rotation_6d 的各分量尺度差异大，`none` 可以避免放大噪声，但四元数不存在这个问题。

---

## 3. 各部分如何关联

### 3.1 数据流

```
ar-research-kempner                            starVLA
─────────────────                              ───────

原始数据 → HDF5 → 合并的 LeRobot repo    ──→   split_custom_v0309_v30_ee.sh
                                                  │  (使用 --modality-file modality_ee_16d.json)
                                                  ▼
                                              playground/Datasets/CustomEE/
                                                ├── place_container_plate_clean1_ee/
                                                ├── place_container_plate_wp4_ee/
                                                ├── place_object_stand_clean1_ee/
                                                └── place_object_stand_wp4_ee/
                                                  │
                                                  ▼
                                              train_starvla.py
                                                读 mixtures.py → "custom_v0309_v30_ee"
                                                读 data_config.py → RobotwinEEDataConfig
                                                读每个 task 的 meta/modality.json
```

### 3.2 命名映射链

三个文件之间的模态名必须一致：

```
modality_ee_16d.json          data_config.py                     mixtures.py
────────────────────          ──────────────                     ───────────
action.left_pos        →   action_keys = [                    "robotwin_ee"
action.left_quat             "action.left_pos",                  ↑
action.left_gripper          "action.left_quat",           每个 mixture
action.right_pos             "action.left_gripper",        tuple 的
action.right_quat            "action.right_pos",           robot_type
action.right_gripper         "action.right_quat",
                             "action.right_gripper",
                           ]
```

这三个文件里的名字只要有一个对不上，加载数据时就会报 `KeyError`。命名规则是 `<modality>.<sub_key>`，其中 sub_key 来自 modality JSON。

---

## 4. 待办 / 接下来需要做什么

### 4.1 立即执行：拆分数据集 + 提交训练

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# 1. 拆分数据集（400 episodes 大约 1-2 分钟）
bash scripts/split_custom_v0309_v30_ee.sh

# 2. 验证输出
ls playground/Datasets/CustomEE/
# 期望：4 个目录

# 3. 验证 modality 是 16D
python3 -c "
import json
m = json.load(open('playground/Datasets/CustomEE/place_container_plate_clean1_ee/meta/modality.json'))
total = max(v['end'] for v in m['action'].values())
print(f'Action dim: {total}')  # 应该是 16
"

# 4. 提交训练（两种方式二选一）

# 方式 A：通过 SLURM sbatch 提交（自动排队，支持 preemption 后自动 resume）
sbatch scripts/slurm_v0309_v30_ee_requeue.sh

# 方式 B：已经在 GPU 计算节点上（比如通过 salloc 拿到了 4xH100/H200），直接跑
bash r-preference/0315-v30-training.sh
```

### 4.2 调试测试（正式训练前先跑一下）

如果想在提交长任务前先 sanity check：

```bash
salloc -p kempner_requeue --account=kempner_ydu_lab --constraint=h200 \
  -N 1 --gpus-per-node=4 --cpus-per-task=64 --mem=1440G -t 0-01:00:00

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
  --datasets.vla_data.data_mix custom_v0309_v30_ee \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.include_state true \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 \
  --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints \
  --run_id v0309_v30_ee_debug_test
```

如果跑完 20 步没崩，说明整个流水线是对的。

### 4.3 训练后评估

Open-loop eval 需要适配 16D。现有的 `realworld/eval_openloop.py` 如果从 checkpoint config 读 action_dim 的话应该能直接用。验证方法：

```bash
python realworld/eval_openloop.py \
  --checkpoint results/Checkpoints/v0309_v30_ee_qwenPI_requeue/checkpoints/steps_50000_pytorch_model.pt \
  --num_samples 100
```

如果 eval 脚本里硬编码了 `action_dim=14`，需要改一下。

### 4.3.1 eval step limit（重要）

`ar-research-kempner/task_config/_eval_step_limit.yml` 定义了每个任务在评估时的最大步数。wp4/wp5 变体之前缺失，会默认回退到 1000 步（`_base_task.py:153`），而正确值应该和 clean 版本一样是 400。已修复，新增以下条目：

```yaml
# wp4 variants (same limits as clean)
place_container_plate_wp4: 400
place_object_stand_wp4: 400

# wp5 variants (same limits as clean)
place_container_plate_wp5: 400
place_object_stand_wp5: 400
```

添加新的 wp 变体时，务必同步更新此文件，否则评估时步数过多会导致成功率虚高。

### 4.4 未来：增加更多 EE 任务

要增加新的 EE 空间任务（更多变体或全新任务）：

1. 在 ar-research-kempner 中以 EE 输出**生成合并的 LeRobot 数据集**
2. **新建一个 split shell 脚本**（复制 `split_custom_v0309_v30_ee.sh`，改 SRC/DST/TASKS）
   - 复用同一个 `--modality-file modality_ee_16d.json`（所有双臂 EE 任务布局相同）
3. 在 `mixtures.py` 中**添加新的 mixture**，robot_type 用 `"robotwin_ee"`
4. **不需要改** `data_config.py`——`RobotwinEEDataConfig` 适用于任何 16D 双臂 EE 数据集

### 4.5 未来：关节空间 + EE 空间混合训练

如果将来想在同一次训练中同时用 14D 关节空间和 16D EE 数据，**目前不支持**，因为 `action_dim` 是一个全局参数。要实现需要：
- 多头 action model（不同输出头对应不同动作空间）
- 或者分开训练、分开 checkpoint

目前先作为独立实验运行。

---

## 5. 架构上下文：StarVLA 如何加载数据

了解这些有助于调试新数据集的问题。

### 5.1 加载链路

```
train_starvla.py
  → 读取 --datasets.vla_data.data_mix
  → 在 mixtures.py 中查找 DATASET_NAMED_MIXTURES["custom_v0309_v30_ee"]
  → 对每个 (task_name, weight, robot_type):
      → 从 data_root_dir/task_name/ 加载数据
      → 读取 meta/modality.json → 将 parquet 列切分成具名 key
      → 在 data_config.py 中查找 ROBOT_TYPE_CONFIG_MAP[robot_type]
      → 应用 config 中定义的 transform（归一化、二值化）
      → 送入训练循环
```

### 5.2 可能出错的地方

| 故障点 | 表现 | 根因 |
|--------|------|------|
| 找不到任务目录 | `FileNotFoundError` | `mixtures.py` 里的目录名和 `data_root_dir` 下的实际文件夹名不匹配 |
| 缺少 modality.json | `FileNotFoundError` on modality.json | Split 脚本没写入（或 --modality-file 路径错了） |
| modality key 报错 | `KeyError: 'action.left_pos'` | modality.json 里的名字和 data_config.py 的 `action_keys` 对不上 |
| action_dim 不对 | Shape mismatch error | SLURM 脚本的 `--framework.action_model.action_dim` 和数据实际维度不匹配 |
| robot type 找不到 | `KeyError: 'robotwin_ee'` | 忘了在 `ROBOT_TYPE_CONFIG_MAP` 里注册 |
| 数据维度不匹配 | 训练中 tensor size 错误 | modality.json 的切片索引加起来不等于预期维度 |

### 5.3 归一化统计文件

首次训练时，StarVLA 会扫描所有 episode 计算每个 key 的归一化统计量，保存到 `results/Checkpoints/<run_id>/dataset_statistics.json`。Resume 时从该文件重新加载。如果你改了数据集组成，需要删掉这个文件以强制重新计算。

---

## 6. 关节空间 vs EE 空间配置对比

方便快速查看两者差异：

| 方面 | 关节空间 (14D) | EE 空间 (16D) |
|------|---------------|---------------|
| Config 类 | `AgilexDataConfig` | `RobotwinEEDataConfig` |
| Robot type key | `"robotwin"` | `"robotwin_ee"` |
| Action 维度 | 14 | 16 |
| 组件 | joints(6) + gripper(1) x2 | pos(3) + quat(4) + gripper(1) x2 |
| 旋转表示 | 关节角度（隐式） | 四元数 (4D, 有界 [-1,1]) |
| 归一化 | joints 用 min_max，gripper 用 binary | pos/quat 用 min_max，gripper 用 binary |
| 数据根目录 | `playground/Datasets/Custom/` | `playground/Datasets/CustomEE/` |
| Modality JSON | split 脚本内置默认值 | `examples/RobotwinEE/train_files/modality_ee_16d.json` |
| Split 脚本 | `scripts/split_custom_v0225_v3.sh` | `scripts/split_custom_v0309_v30_ee.sh` |
| SLURM 脚本 | `scripts/slurm_custom_v0225_v3_requeue.sh` | `scripts/slurm_v0309_v30_ee_requeue.sh` |
| WandB 项目 | `starVLA_Custom_v0225_v3` | `starVLA_Custom_EE` |

---

## 7. 文件速查表

所有路径相对于 `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA/`。

### v30-ee 新增/修改的文件

| 文件 | 状态 | 用途 |
|------|------|------|
| `scripts/split_custom_lerobot.py` | 修改 | 加了 `--modality-file` 参数（向后兼容） |
| `examples/RobotwinEE/train_files/modality_ee_16d.json` | 新建 | 16D EE 模态映射 |
| `scripts/split_custom_v0309_v30_ee.sh` | 新建 | v30-ee 拆分包装脚本 |
| `starVLA/dataloader/gr00t_lerobot/data_config.py` | 修改 | 新增 `RobotwinEEDataConfig` + 注册 `"robotwin_ee"` |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | 修改 | 新增 `"custom_v0309_v30_ee"` mixture |
| `scripts/slurm_v0309_v30_ee_requeue.sh` | 新建 | SLURM 训练脚本（QwenPI, 16D） |

### 数据集路径

| 项目 | 路径 |
|------|------|
| 合并的 LeRobot repo | `$LAB_ROOT/.cache/huggingface/lerobot/custom_v0309_v30_ee_repo/` |
| 拆分输出 | `playground/Datasets/CustomEE/` |
| 训练输出 | `results/Checkpoints/v0309_v30_ee_qwenPI_requeue/` |
| WandB | 项目 `starVLA_Custom_EE`，entity `kaiwenh-17-uiuc` |
