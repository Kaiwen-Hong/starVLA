# Custom Dataset v0225_v3 Training (20 Variants, clean + hv/lv)

> v0225_v3 uses high-value / low-value variants (hv/lv) instead of wp1/wp2, with 10 tasks (down from v0218's 11).
> Data processing reference: `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner/script/Kempner_process_custom-0225-v3.md`.
> For cluster setup, see `DOC-training/kempner-cluster.md`.
> For common training patterns (image-in-parquet, multi-node), see `DOC-training/custom-all.md`.

---

## What Changed: v0225_v3 vs v0218 vs custom_all

| 版本 | Tasks | 变体组合 | 变体数 | Episodes |
|------|-------|----------|--------|----------|
| custom_all (原版) | 11 tasks | clean/wp1/wp2 | 33 | 3300 |
| v0218 (v1) | 11 tasks | 8 clean/wp1/wp2 + 3 clean only | 27 | 2700 |
| **v0225_v3 (本版, v2)** | **10 tasks** | **5 clean/hv/lv + 5 clean only** | **20** | **2000** |

### 与 v0218 的主要区别

1. **去掉 `click_alarmclock`** — v0218 有 11 个 task，v0225_v3 只有 10 个
2. **hv/lv 替代 wp1/wp2** — 5 个 full task 使用 high-value / low-value 变体
3. **更多 clean-only task** — 5 个 clean-only（v0218 只有 3 个）
4. **`move_stapler_pad` 和 `place_empty_cup` 降为 clean-only** — v0218 中它们有 wp1/wp2

### 10 个 task 的变体分配

| Task | clean | hv | lv | 小计 |
|------|-------|----|----|------|
| adjust_bottle | yes | yes | yes | 3 |
| beat_block_hammer | yes | yes | yes | 3 |
| blocks_ranking_rgb | yes | yes | yes | 3 |
| handover_block | yes | yes | yes | 3 |
| move_can_pot | yes | yes | yes | 3 |
| blocks_ranking_size | yes | **no** | **no** | 1 |
| handover_mic | yes | **no** | **no** | 1 |
| move_pillbottle_pad | yes | **no** | **no** | 1 |
| move_stapler_pad | yes | **no** | **no** | 1 |
| place_empty_cup | yes | **no** | **no** | 1 |
| **合计** | | | | **20** |

### 数据来源

| 变体 | 来源 | StarVLA 状态 |
|------|------|-------------|
| 10 个 clean | 已有 `playground/Datasets/Custom/` | Ready |
| 10 个 hv/lv | ar-research-kempner pipeline → 需拆分为 per-task 目录 | **需准备** |

---

## 当前状态

| 项目 | 状态 |
|------|------|
| Per-task datasets — 10 clean dirs | Ready (reuse from existing 33 dirs) |
| Per-task datasets — 10 hv/lv dirs | **Pending** (需从 ar-research-kempner 数据拆分，见第 1 节) |
| `mixtures.py` (`custom_v0225_v3` + `custom_v0225_v3_task1`) | Ready |
| SLURM script (requeue) | Ready |
| Pretrained model (Qwen3-VL-4B-Instruct-Action) | Ready |
| conda env (`starVLA`) | Ready |
| Image-in-parquet patches (`datasets.py`) | Ready |

---

## TL;DR — 提交命令

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# kempner_requeue (7-day, preemptable, H200)
sbatch scripts/slurm_custom_v0225_v3_requeue.sh
```

### 查看资源

```bash
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/most_empty_partition.sh
squeue --start -j <jobid>   # 预估开始时间
```

For GPU switching, see `DOC-training/kempner-cluster.md`.

---

## 1. 数据准备 (hv/lv per-task 目录)

10 个 clean 目录已在 `playground/Datasets/Custom/` 中。需要新建 10 个 hv/lv 目录：

```
playground/Datasets/Custom/
├── adjust_bottle_hv/       ← 新建
├── adjust_bottle_lv/       ← 新建
├── beat_block_hammer_hv/   ← 新建
├── beat_block_hammer_lv/   ← 新建
├── blocks_ranking_rgb_hv/  ← 新建
├── blocks_ranking_rgb_lv/  ← 新建
├── handover_block_hv/      ← 新建
├── handover_block_lv/      ← 新建
├── move_can_pot_hv/        ← 新建
├── move_can_pot_lv/        ← 新建
├── adjust_bottle/          ← 已有 (clean)
├── beat_block_hammer/      ← 已有 (clean)
├── ...
```

### 拆分命令

ar-research-kempner pipeline 已生成 merged LeRobot 数据集:
`/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface/lerobot/custom_v0225_v3_repo/` (2000 episodes)

拆分脚本会把 merged repo 拆成 20 个 per-task 目录（10 clean + 10 hv/lv）。
clean 目录已存在，会被同源数据覆盖（无损）。

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash scripts/split_custom_v0225_v3.sh
```

> 底层调用 `scripts/split_custom_lerobot.py`（与 `custom_all` 拆分相同），
> 每个 task 产出 100 episodes 的 image-in-parquet 数据 + `meta/modality.json`。
> 默认 20 个并行 worker，可指定: `bash scripts/split_custom_v0225_v3.sh 8`

### 验证

```bash
# 确认 10 个 hv/lv 目录已创建
for v in adjust_bottle beat_block_hammer blocks_ranking_rgb handover_block move_can_pot; do
  for s in hv lv; do
    dir="playground/Datasets/Custom/${v}_${s}"
    if [ -d "$dir" ]; then
      echo "OK: ${v}_${s}"
    else
      echo "MISSING: ${v}_${s}"
    fi
  done
done

# 确认 modality.json 存在
for v in adjust_bottle beat_block_hammer blocks_ranking_rgb handover_block move_can_pot; do
  for s in hv lv; do
    [ ! -f "playground/Datasets/Custom/${v}_${s}/meta/modality.json" ] && echo "MISSING modality.json: ${v}_${s}"
  done
done

# 确认全部 20 个目录 (10 clean + 10 hv/lv)
echo "Total dirs used by v0225_v3:"
for entry in $(python3 -c "
tasks = [
  'adjust_bottle', 'adjust_bottle_hv', 'adjust_bottle_lv',
  'beat_block_hammer', 'beat_block_hammer_hv', 'beat_block_hammer_lv',
  'blocks_ranking_rgb', 'blocks_ranking_rgb_hv', 'blocks_ranking_rgb_lv',
  'blocks_ranking_size',
  'handover_block', 'handover_block_hv', 'handover_block_lv',
  'handover_mic',
  'move_can_pot', 'move_can_pot_hv', 'move_can_pot_lv',
  'move_pillbottle_pad', 'move_stapler_pad', 'place_empty_cup',
]
for t in tasks: print(t)
"); do
  [ -d "playground/Datasets/Custom/$entry" ] && echo "  OK: $entry" || echo "  MISSING: $entry"
done
```

---

## 2. 训练参数

### 参数总结

| 参数 | 值 | 说明 |
|------|-----|------|
| `data_root_dir` | `playground/Datasets/Custom` | 与 custom_all / v0218 相同 |
| `data_mix` | `custom_v0225_v3` | 20 tasks in `mixtures.py` |
| `per_device_batch_size` | 8 | Per GPU |
| `gradient_accumulation_steps` | 2 | Effective batch = 4 GPU x 8 x 2 = **64** |
| `max_train_steps` | 500,000 | ~8 epochs (dataset 比 v0218 小 26%) |
| `save_interval` | 50,000 | Checkpoint every 50K steps |
| `eval_interval` | 5,000 | Eval every 5K steps |
| `is_resume` | true | 抢占后自动从 checkpoint 恢复 |
| `attn_implementation` | sdpa | Required on Kempner (GLIBC 2.28) |

### 与 v0218 的参数对比

| | v0218 (27 variants) | v0225_v3 (20 variants) |
|---|---|---|
| `data_mix` | `custom_v0218` | `custom_v0225_v3` |
| Total episodes | 2,700 | 2,000 |
| Estimated frames | ~714K | ~529K |
| `max_train_steps` | 500,000 (~6 epochs) | 500,000 (~8 epochs) |
| Normalization stats | Computed from 27 tasks | Recomputed from 20 tasks |
| `run_id` | `custom_v0218_qwenOFT_requeue` | `custom_v0225_v3_qwenOFT_requeue` |
| WandB project | `starVLA_Custom_v0218` | `starVLA_Custom_v0225_v3` |

> v0225_v3 uses a new `run_id`, so training starts fresh. Normalization stats are
> recomputed from scratch each time the dataloader is created.

---

## 3. Dataset Info

| Property | Value |
|----------|-------|
| Location | `playground/Datasets/Custom/` (20 of 33+ directories used) |
| Format | LeRobot v2.1, image-in-parquet (no .mp4 video files) |
| Variants | 20 (5 tasks x 3 + 5 tasks x clean only) |
| Episodes/task | 100 |
| Total episodes | 2,000 |
| FPS | 50 |
| Features | 14-dim state/action, 3 cameras (cam_high, cam_left_wrist, cam_right_wrist) |
| Resolution | 640 x 480 |
| Robot type | `robotwin` (dual-arm aloha, same config as RoboTwin) |

### 20 task variants

**15 variants** (5 tasks x clean/hv/lv): adjust_bottle, beat_block_hammer,
blocks_ranking_rgb, handover_block, move_can_pot

**5 variants** (clean only): blocks_ranking_size, handover_mic, move_pillbottle_pad,
move_stapler_pad, place_empty_cup

---

## 4. Checkpoints & Resume

Output: `results/Checkpoints/custom_v0225_v3_qwenOFT_requeue/`.
Each contains `config.yaml`, `dataset_statistics.json`, and `checkpoints/steps_<N>/`
(full accelerator state) + `steps_<N>_pytorch_model.pt` (standalone weights).

When `is_resume=true`, the trainer scans `checkpoints/` for `steps_<N>/`
directories, picks the highest N, and calls `accelerator.load_state()` to
restore model, optimizer, scheduler, and RNG. This handles both preemption
(auto re-queue on `kempner_requeue`) and timeout (manual resubmit on `kempner_h100`)
with no parameter changes needed.

---

## 5. Quick Debugging

### 单 task 测试 (SLURM)

修改 SLURM 脚本:
```bash
  --datasets.vla_data.data_mix custom_v0225_v3_task1 \   # only adjust_bottle
  --trainer.max_train_steps 100 \
  --trainer.save_interval 50 \
  --run_id custom_v0225_v3_debug_test \
```

### 交互式测试 (不用 sbatch)

```bash
# 获取 interactive node
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
  --datasets.vla_data.data_mix custom_v0225_v3_task1 \
  --datasets.vla_data.per_device_batch_size 8 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 20 --trainer.save_interval 10 \
  --trainer.gradient_accumulation_steps 2 \
  --run_root_dir ./results/Checkpoints --run_id custom_v0225_v3_interactive_test
```

---

## 6. File Reference

| File | Purpose |
|------|---------|
| `scripts/split_custom_v0225_v3.sh` | 拆分 merged repo → 20 per-task 目录 |
| `scripts/slurm_custom_v0225_v3_requeue.sh` | SLURM script: kempner_requeue, 1 node x 4 GPU |
| `starVLA/dataloader/gr00t_lerobot/mixtures.py` | `custom_v0225_v3` / `custom_v0225_v3_task1` mixtures |
| `starVLA/dataloader/gr00t_lerobot/datasets.py` | Core dataset classes (image-in-parquet patches) |
| `starVLA/training/train_starvla.py` | Training loop |
| `starVLA/config/deepseeds/deepspeed_zero2.yaml` | Accelerate + DeepSpeed ZeRO-2 config |
| `playground/Datasets/Custom/` | Per-task dirs (10 clean reused + 10 hv/lv new) |
| `results/Checkpoints/custom_v0225_v3_qwenOFT_requeue/` | Training output |

### ar-research-kempner 数据处理参考

以下文件路径相对于 `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/ar-research-kempner/`:

| File | Purpose |
|------|---------|
| `script/Kempner_process_custom-0225-v3.md` | 完整数据处理文档 (hv/lv 解压 → HDF5 → LeRobot) |
| `script/Kempner_extract_custom_hv_lv.sh` | 解压 hv/lv tar.gz |
| `policy/pi05/Kempner_process_custom_hv_lv_data.sh` | 处理 hv/lv → HDF5 |
| `policy/pi05/Kempner_prepare_custom_v0225_v3.sh` | 合并 clean + hv/lv 到 training_data |
| `policy/pi05/Kempner_generate_custom_v0225_v3.sh` | 生成 merged LeRobot repo |

See `DOC-training/kempner-cluster.md` for monitoring, cluster issues, partition
comparison, and environment notes. See `DOC-training/custom-all.md` for
image-in-parquet patches and multi-node setup.
