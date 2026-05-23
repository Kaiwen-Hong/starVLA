# Preference-conditioned VLA — Stage A baseline (no VQA), giveobj

**Scope: spec + design only.** Data, model, prompt, norm, loss, hyperparams,
ablation lock between baseline and main-method. **For ops** (launch scripts,
storage, training quirks, postmortems, troubleshooting) see
[`training-runbook.md`](training-runbook.md).

Written 2026-05-22. Branch `opd` ≡ `starVLA` (same commit).
Author: Kaiwen Huang. Audience: future Kaiwen, collaborator hand-off, paper drafting.

---

## 0. TL;DR

- **What**: Stage A baseline = preference-conditioned VLA trained on labeled source group A (`giveobj`, 16 task dirs, 1600 demos), action-only loss, no VQA cotrain. Backbone = `Qwen3-VL-4B-Instruct` from HF base (no StarVLA action ckpt). Action head = `LayerwiseFlowmatchingActionHead`, bimanual EE + 6D rotation (20D), chunk 16. Preference enters as a 2-word categorical suffix on the prompt: `... Preference: low contact` / `... Preference: high contact`.
- **Produces**: one ckpt that doubles as (1) **stage-1 of B0 baseline** (followed by Phase 1b: base-only SFT on unlabeled target group B), and (2) the **ablation control for the main-method's Stage A** (which adds `L = L_action + λ · L_vqa`).
- **Hard constraint**: §3 / §4 / §5 / §6 below are **byte-identical** between baseline and main-method. Only loss differs.
- **Run**: `NUM_GPUS=8 bash examples/preference/launch_pref_stage_a_baseline.sh`. Eff batch 64.
- **Status** (2026-05-22): smoke tests 1/2 pass; smoke test 3 (full forward) failed only because all 8 H100s were occupied by another job — no code defect; will pass when a GPU frees up. Group B data not yet on disk.

---

## 1. 项目定位 / Ablation 几何

### 1.1 角色
- 这是 **Stage A baseline**(no VQA),不是 main-method,但 **代码 / 数据 pipeline / 超参 / norm / split 必须和 main-method 完全一致**,只有 loss 不同。
- ckpt 双用途:
  1. **B0 baseline 的 stage-1 起手** —— 后接 Phase 1b: base-only SFT on unlabeled target group B(data 还没下载)。
  2. **main-method Stage A 的 ablation control** —— 同一份数据 / prompt / norm,加 `L_vqa` 之后跑出来,跟这条 baseline 对比 → 干净 ablation。

### 1.2 Ablation 矩阵

| 路线 | Stage A loss | Stage B loss | 这份 doc 对应 |
|---|---|---|---|
| **Baseline B0** | `L_action`(本 doc) | base-only SFT on B | stage-1 |
| **Main-method** | `L_action + λ·L_vqa` | base-only SFT on B(同 Phase 1b) | 同结构,只换 loss |

读者下次设计 main-method 时,**只改 loss**,其他全部抄。`§12 hand-off checklist` 写明哪些必须抄、哪些可改。

### 1.3 硬约束(从源头说一次,后面不再重复)
1. 数据 root、task group list、split seed、val fraction、prompt 函数、norm stats — **必须一致**。
2. 模型架构 / 超参 / optimizer / 相机选 / image_size / chunk_size — **必须一致**。
3. main-method 加 VQA 时,通过 **额外的数据字段 + 额外的 loss 项** 加,不动这条 baseline 的任何决策。

---

## 2. 数据(`/mnt/localssd/kaiwenh/pref/data/giveobj`)

### 2.1 目录结构

16 个 task subdirs(8 task-groups × 2 pref keys),命名 `<task_group>_{25|75}`:

```
giveobj/
├── give_boxdrink_{25,75}/
├── give_callbell_{25,75}/
├── give_fork_{25,75}/
├── give_screwdriver_{25,75}/
├── put_boxdrink_dustbin_{25,75}/
├── put_callbell_dustbin_{25,75}/
├── put_fork_dustbin_{25,75}/
└── put_screwdriver_dustbin_{25,75}/
```

每个 task subdir(用 `give_boxdrink_25/` 当 reference):
```
give_boxdrink_25/
├── data/episode{0..99}.hdf5         # 100 episodes
├── instructions/episode{0..99}.json # seen[100] + unseen[100] paraphrases
├── video/episode{0..99}.mp4         # 30 fps debug video
├── scene_info.json                  # per-episode metadata
└── seed.txt
```

**总计 100 demos × 16 task dirs = 1600 episodes**(注意:spec 原文有处写过 800,实测 1600,以 disk 为准)。每 episode T ≈ 121-211 帧 @ 30 Hz;具体 episode0 of `give_boxdrink_25` 是 T=166。

### 2.2 HDF5 内部 key 表(verified on `give_boxdrink_25/data/episode0.hdf5`)

| Key | Shape | Dtype | 说明 |
|---|---|---|---|
| `endpose/left_endpose` | `(T, 7)` | float64 | xyz(3) + quat-xyzw(4),**SAPIEN/ManiSkill convention** |
| `endpose/right_endpose` | `(T, 7)` | float64 | 同上 |
| `endpose/left_gripper` | `(T,)` | float64 | continuous [0, 1] |
| `endpose/right_gripper` | `(T,)` | float64 | 同上 |
| `joint_action/left_arm` | `(T, 6)` | float64 | 6 DoF,**当前 baseline 不用** |
| `joint_action/right_arm` | `(T, 6)` | float64 | 同上 |
| `joint_action/{left,right}_gripper` | `(T,)` | float64 | 同上 |
| `joint_action/vector` | `(T, 14)` | float64 | concat 14D,**当前 baseline 不用** |
| `observation/{head,front,left,right}_camera/rgb` | `(T,)` opaque bytes (`|S<N>`) | bytes | **JPEG-encoded**,需 `PIL.Image.open(BytesIO(...))` |
| `observation/<cam>/{intrinsic_cv,extrinsic_cv,cam2world_gl}` | `(T,3,3)` / `(T,3,4)` / `(T,4,4)` | float32 | 相机几何 |
| `pointcloud` | `(T, 0)` | float64 | **空,placeholder** |

相机分辨率(per-frame 检查):

| 相机 | (W, H) |
|---|---|
| `head_camera` | (320, 180) |
| `front_camera` | (320, 240)  **(大一档)** |
| `left_camera` | (320, 180) |
| `right_camera` | (320, 180) |

### 2.3 8 个 task group + pref label 含义

| Task group | Active arm | Object 姿态 | Pref `_25` = "low contact" | Pref `_75` = "high contact" |
|---|---|---|---|---|
| `give_boxdrink` | 随机左/右 per ep | 站立(竖立)| 抓 bottom 25% | 抓 top 75% |
| `give_callbell` | 随机左/右 | 站立 | 抓 base | 抓 dome(顶) |
| `give_fork` | 随机左/右 | 平躺 | 抓 handle 末端 | 抓 tine 末端 |
| `give_screwdriver` | 随机左/右 | 平躺 | 抓 grip 末端 | 抓 tip 末端 |
| `put_boxdrink_dustbin` | **始终左臂** | 站立 | 抓 bottom | 抓 top |
| `put_callbell_dustbin` | **始终左臂** | 站立 | 抓 base | 抓 dome |
| `put_fork_dustbin` | **始终左臂** | 平躺 | 抓 handle | 抓 tine |
| `put_screwdriver_dustbin` | **始终左臂** | 平躺 | 抓 grip | 抓 tip |

`put_*_dustbin` 全部左臂,右臂全 0 → 强 do-nothing signal,可能 over-learn。v1 不加 mask,§7 metric 看到再改。

### 2.4 pref label 在哪几处编码

1. **目录路径**:`<task_group>_{25,75}`。
2. **scene_info.json** 每 episode 的 `info.grasp_region`:`bottom_25` / `top_75` 等(见 episode0:`"grasp_region": "bottom_25"`、`"base_height_fraction": 0.25`、`"height_fraction": 0.2426...`)。dataset adapter **不读**这个,只从路径取 pref key,scene_info.json 仅供 debugging。
3. **instructions/.json `seen[]`** 里 paraphrase 文本会自然提到 "on the bottom" / "near the top" 等 —— **必须 strip**,否则 baseline 跟 main-method 都能从 base prompt 偷 pref signal,ablation 不干净。

### 2.5 已知 quirks(踩坑记录)

- **Quaternion convention 是 xyzw**(scalar-last,SAPIEN/ManiSkill 默认)。pytorch3d 默认 wxyz(scalar-first)。**走 pytorch3d 会静默交换 w 和 x**,所以 `rotation.py` 走 `scipy.spatial.transform.Rotation`,不走 pytorch3d。
- **没有 object mesh、没有 per-frame object pose**。`/mnt/localssd/kaiwenh/pref/data/giveobj/` 下没有 mesh,scene_info.json 只有 asset id 和 init pose。→ §7 / §8 eval 用 **task-relative** EE 信号,不能用 object-relative。
- **`_50` 中间档**被解压过然后从 disk 移除;只剩 `_25` / `_75`。如果未来要 25/50/75 三档,需要重新下载。
- **Group B(unlabeled target)数据不在 disk**(`/mnt/localssd/kaiwenh/pref/data/` 下只有 `giveobj/` 一个 sibling)。Phase 1b 需要另行下载。
- **Pref signal 在 paraphrase 文本里被直接说出来**("on the bottom"/"on the top")。**必须 strip 才能做干净 ablation**(§4)。

### 2.6 Pref signal 量级(实测,first 50 episodes × 2 pref)

中间帧 EE 位置 `mean(L75) - mean(L25)`,m(米),取自 `endpose/{left,right}_endpose`。`std` 从 `stats_giveobj_v1.json` 的 normalized 20D action(归一前):

| Task group | 主要 axis | Δ(m) `75 − 25` | norm σ | `|Δ|/σ` | 评级 |
|---|---|---|---|---|---|
| `give_boxdrink` | L_z | +0.0384 | 0.064 | 0.60 | STRONG |
| `give_callbell` | L_z | +0.0371 | 0.064 | 0.58 | STRONG |
| `give_fork` | L_x | -0.0263 / R_x +0.0213 | 0.106 / 0.082 | 0.25 / 0.26 | weak |
| `give_screwdriver` | L_x | +0.0262 / R_x +0.0318 | 0.106 / 0.082 | 0.25 / 0.39 | weak |
| `put_boxdrink_dustbin` | L_z | **+0.0739** | 0.064 | **1.16** | STRONG |
| `put_callbell_dustbin` | L_z | **+0.0814** | 0.064 | **1.27** | STRONG |
| `put_fork_dustbin` | L_x | **+0.0798** | 0.106 | **0.75** | decent |
| `put_screwdriver_dustbin` | L_x | -0.0240 / L_y +0.0249 | 0.106 / 0.141 | 0.23 / 0.18 | weak |

> 把它放在脑子里:站立物体(`boxdrink`, `callbell`)pref 信号在 z 轴,7-8 cm,σ 0.064 → 信号 1+ σ,**很强**。平躺物体(`fork`, `screwdriver`)pref 信号在 x 轴,4-8 cm,σ 较大 → 信号 0.2-0.7 σ,**弱**。`give_*`(随机臂)比 `put_*_dustbin`(固定左臂)进一步弱 ~2× —— 不同 episode 用不同臂,世界系下信号被抵消。
>
> **§7 metric 3(Δz/Δx sign accuracy)只在 STRONG/decent task 上报**:`put_boxdrink_dustbin`, `put_callbell_dustbin`, `put_fork_dustbin` 三个固定左臂 + 量级 ≥ 0.7σ 的 task。

> 注:第二 / 第三档 evaluation(`§7` 全部 8 task counterfactual MSE / `§8` sim rollout)不受 pref signal 弱影响 —— 即便 fork/screwdriver 信号弱,模型如果能用 pref token 输出不一样的 action,counterfactual MSE 就 >> 0。Sign accuracy 才是要求 magnitude 大到能可靠判正负。

---

## 3. 架构 / 模型

### 3.1 代码栈

- **Codebase**: `/home/kaiwenh/starVLA/`,branch `opd`(≡ `starVLA` 同 commit)。conda env: `starVLA` at `/mnt/localssd/kaiwenh/miniconda3/envs/starVLA`。
- **Framework**: `QwenPI`(`starVLA/model/framework/QwenPI.py:38-49`);通过 `--framework.name QwenPI` 选择。

### 3.2 Backbone

- **Model**: `Qwen3-VL-4B-Instruct`(HF: `Qwen/Qwen3-VL-4B-Instruct`)。**从 base 起手**,**不**加载 StarVLA action ckpt。
- **关键 backbone 数**(由 `QwenPI.py:69` 在 runtime 取):
  - `text_config.hidden_size = 2560`(对应 YAML `vl_hidden_dim`)
  - `num_hidden_layers = 36`(对应 `num_vl_layers`,hardcoded in `QwenPI.py:69`)
  - `attn_implementation: sdpa`

### 3.3 Action head + 重要陷阱

- **Class**: `LayerwiseFlowmatchingActionHead`(`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:246`)。
- **DiT 是 layer-wise cross-attn**:每个 backbone hidden layer 作为一个 DiT block 的 cross-attn key/value(`QwenPI.py:113-115`:`vl_embs_list = list(all_hidden[-expected_layers:])`)。
- **陷阱(写 doc 唯一不希望读者再踩的)**:`LayerwiseFlowmatchingActionHead.__init__` 第 256-262 行运行时 **override 三个 DiTConfig 字段**:

  ```python
  # LayerwiseFM_ActionHeader.py:256-262
  DiTConfig["num_layers"]          = global_config.framework.qwenvl.num_vl_layers          # = 36
  DiTConfig["input_embedding_dim"] = global_config.framework.qwenvl.vl_hidden_dim           # = 2560
  DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // attention_head_dim # = 2560/64 = 40
  diffusion_model_cfg.update(DiTConfig)
  diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"]                 # = 2560
  ```

- 这意味着:
  - YAML 写的 `diffusion_model_cfg.num_layers: 16` 是 **dead code**,真实 DiT 是 **36 层**。
  - YAML 写的 `cross_attention_dim: 2560` 跟 runtime override 一致(我们手动对齐了,不依赖默认),实际还是 runtime 那条覆盖。
  - DiT 真实 shape:**36 layers × hidden 2560 × 40 heads × head_dim 64**,**~3.3B params**。
- **Trainable footprint**:Qwen3-VL-4B(~4B,fully trainable in baseline)+ DiT(~3.3B)= **~7.79B params**(benchmark 2026-05-22 实测)。Zero2 sharded AdamW 下,**每 GPU peak ≈ 56 GB**(model bf16 15.6 + activations @ batch 8 ckpt OFF ~26 + sharded grads 2 + sharded optim 12)。80 GB H100 余 ~24 GB,**推荐 8× H100 + DeepSpeed Zero2**;4× H100 即便 smoke 过,step 时也可能 OOM。
- DiTConfig defaults(`LayerwiseFM_ActionHeader.py:214`):`{"num_layers": 36, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32}` — 默认是 Qwen2.5-VL 的,被我们 override 成 Qwen3-VL-4B 的 2560/40。

### 3.4 Flow matching 配置

| 参数 | 值 | 来源 |
|---|---|---|
| time sampling | `Beta(α=1.5, β=1.0)`,然后 `t = (s − sample)/s`,`s=0.999` | YAML `noise_beta_alpha/beta/s` |
| timestep buckets | 1000 | `num_timestep_buckets` |
| inference steps | 4 | `num_inference_timesteps` |
| objective | velocity field MSE: `loss = MSE(noise_pred, v=actions − noise)` | `LayerwiseFM_ActionHeader.py:316-318` |
| repeated diffusion steps | runtime 2(hardcoded in `QwenPI.py:128`,override YAML/cfg) | `QwenPI.py:128` 注 "NO repeat for big action FM" |

---

## 4. 数据处理 / Prompt

### 4.1 Prompt 模板(冻死,baseline / main-method / Phase 1b / eval 共用)

```
action_prompt = <base_prompt> + " Preference: " + <pref_label>
```

- `<pref_label>` ∈ `{"low contact", "high contact"}`(`examples/preference/dataset/prompt.py:25` — `PREF_LABELS`)
- `<base_prompt>` 来自 paraphrase strip 流程(§4.2),fallback 到 8 个固定 template(§4.3)。

### 4.2 v5 hybrid strip 流程(`prompt.py:76-89`,`strip_v5`)

输入:`(phrase, task_group)`。输出:`(stripped, mode)`,`mode ∈ {"stripped", "fallback"}`。

1. 用 `_STRIP_RE`(`prompt.py:52-63`)做一次 substitution。正则覆盖的动词 / 介词 / 名词组合:
   - 动词:`grasping/gripping/holding/grabbing/picking[up]/taking/lifting/handing over/carrying`(以及过去式 / 单原型变体)
   - 介词:`on/from/at/by/near/along/around/with/in/via/using`(全可选)
   - 限定:`(its|the)`(可选)
   - 名词:`top/bottom/upper/lower/higher/high/low/base`
   - 可选尾巴:`(part|end|section|portion|side|half|region|area|of|tip|cap)`
2. 用 `_LEAK_RE`(`prompt.py:66-71`)检查残留:残留 leak word 之一 → fallback。**_LEAK_WORDS** 共 16 个:`top, bottom, low, high, upper, lower, height, tall, short, above, below, elevated, raised, halfway, midway, base`。
3. Stripped 必须 `len > 15` 且**正则确实命中过**(`n_sub > 0`)且**无 leak 残留**,否则 fallback。

### 4.3 fallback clean templates(`prompt.py:40-49`,`CLEAN_TEMPLATE`)

| Task group | Clean template |
|---|---|
| `give_boxdrink` | `Hand the box drink over to the other side of the table.` |
| `give_callbell` | `Hand the call bell over to the other side of the table.` |
| `give_fork` | `Hand the fork over to the other side of the table.` |
| `give_screwdriver` | `Hand the screwdriver over to the other side of the table.` |
| `put_boxdrink_dustbin` | `Put the box drink into the dustbin.` |
| `put_callbell_dustbin` | `Put the call bell into the dustbin.` |
| `put_fork_dustbin` | `Put the fork into the dustbin.` |
| `put_screwdriver_dustbin` | `Put the screwdriver into the dustbin.` |

### 4.4 验证结果(verified 2026-05-22)

跑 v5 strip on **160,000 paraphrases**(全部:16 task dirs × 100 episodes × 100 `seen[]` per episode):

| Metric | Value |
|---|---|
| Stripped | 160,000 (**100.00%**) |
| Fallback | 0 (0.00%) |
| Residual leak in output | 0 (0.00%) |

> Docstring 里写 "85% strip / 15% fallback on 8000 sampled" 是早期 v4 数字;**v5 把那部分覆盖率拉到 100%**。Fallback path 留作 safety net,**实际从未触发**,但保留以保证未来加新 paraphrase 时也安全。

### 4.5 实例

输入:`"Handover the boxdrink to the other side of the table, grasping on the bottom of the boxdrink."`(seen[0] of `give_boxdrink_25/episode0`)

- `task_group = "give_boxdrink"`, `pref_key = "25"`
- `strip_v5(...)` → `"Handover the boxdrink to the other side of the table."` (mode=`stripped`)
- `build_action_prompt(...)` → `"Handover the boxdrink to the other side of the table. Preference: low contact"`

### 4.6 Paraphrase 选择

- 每次 `__getitem__` 从 `seen[100]` 中**随机选 1**(seed 在 `paraphrase_seed=42`,`_paraphrase_rng = random.Random(42)`,`pref_hdf5_dataset.py:146`,**dataset 实例级别**,**不是 sample 级别**)。
- 后续如果想让 main-method 跟 baseline **每条 sample paraphrase 完全对齐**,需要把 RNG 改为 `(idx)` 函数。v1 不做。

### 4.7 Action normalization

- 策略:**global q99 quantile-clip 到 [-1, 1]**(`pref_hdf5_dataset.py:190-198`,`_normalize`):
  ```
  out = 2 * (x - q01) / (q99 - q01) - 1; clip(out, -1, 1)
  ```
- 只用 **train split** 的 1280 episodes(192,927 frames)算 stats。
- 输出:`examples/preference/dataset/stats_giveobj_v1.json`(同时存 q01/q99/min/max/mean/std + meta)。
- State 和 action 共享同一份 stats(20D 同 layout)。
- 默认 `stats_json_path: examples/preference/dataset/stats_giveobj_v1.json`(YAML 里 `datasets.vla_data.stats_json_path`)。

### 4.8 Norm 范围(20D action / state,从 `stats_giveobj_v1.json` 抄)

| Idx | Dim | q01 | q99 | span | std |
|---|---|---|---|---|---|
| 0 | L_x | -0.350 | +0.041 | 0.391 | 0.106 |
| 1 | L_y | -0.314 | +0.102 | 0.416 | 0.141 |
| 2 | L_z | +0.775 | +1.102 | 0.327 | 0.064 |
| 3 | L_6d_0 | -0.079 | +1.000 | 1.079 | 0.368 |
| 4 | L_6d_1 | -0.573 | +0.539 | 1.112 | 0.162 |
| 5 | L_6d_2 | -0.061 | +1.000 | 1.061 | 0.396 |
| 6 | L_6d_3 | -0.914 | +0.808 | 1.722 | 0.345 |
| 7 | L_6d_4 | -0.999 | +0.881 | 1.881 | 0.464 |
| 8 | L_6d_5 | -0.025 | +1.000 | 1.025 | 0.349 |
| 9 | L_grip | 0.000 | +1.000 | 1.000 | 0.467 |
| 10 | R_x | -0.025 | +0.306 | 0.331 | 0.082 |
| 11 | R_y | -0.313 | +0.039 | 0.352 | 0.094 |
| 12 | R_z | +0.788 | +1.050 | 0.262 | 0.036 |
| 13 | R_6d_0 | -0.059 | +1.000 | 1.059 | 0.252 |
| 14 | R_6d_1 | -0.432 | +0.428 | 0.859 | 0.099 |
| 15 | R_6d_2 | -0.023 | +1.000 | 1.022 | 0.249 |
| 16 | R_6d_3 | -0.804 | +0.728 | 1.532 | 0.180 |
| 17 | R_6d_4 | -0.907 | +0.963 | 1.871 | 0.284 |
| 18 | R_6d_5 | -0.014 | +1.000 | 1.014 | 0.243 |
| 19 | R_grip | 0.000 | +1.000 | 1.000 | 0.319 |

观察:xyz 工作区 30-50 cm;z 跨度更窄(站立工作平面);6D 维度 ∈ [-1, 1] 与理论一致;grip ∈ [0, 1] 单调。

### 4.9 Train/val split

- 按 `(task_dir)` 分层,**每 task 80/20**,seed 42。
- `_split_episodes(...)`(`pref_hdf5_dataset.py:46-66`):
  - 每 task 内部 RNG 为 `random.Random(f"{42}-{task_dir}")` → deterministic per-task。
  - 1600 episodes 总 → 1280 train(每 task 80)/ 320 val(每 task 20)。
- 索引粒度 = (task_dir, ep_id, frame_idx);每 episode 可索引 `T - chunk_size + 1` 个起始 frame。
- 总 sample 数(verified 2026-05-22):
  - train: **173,727**
  - val: **42,983**

### 4.10 Image 处理

- 选 3 个相机:`head_camera`, `left_camera`, `right_camera`。
- 去掉 `front_camera`:跟 head 视角重叠且分辨率不一致(320×240 vs 320×180),不带额外信息。
- 每帧:`PIL.Image.open(BytesIO(<JPEG bytes>)).convert("RGB").resize((224, 224))`。
- `image_size: [224, 224]`(H, W),`PIL.resize` 传 `(W, H)` 等价。

---

## 5. Action 表示(20D bimanual EE + 6D rotation)

### 5.1 Layout

```
action / state ∈ R^20

[ L_x, L_y, L_z,                       (3)  left EE xyz
  L_6d_0, L_6d_1, L_6d_2,
  L_6d_3, L_6d_4, L_6d_5,              (6)  left 6D rotation
  L_grip,                              (1)  left gripper [0, 1]
  R_x, R_y, R_z,                       (3)  right EE xyz
  R_6d_0, R_6d_1, R_6d_2,
  R_6d_3, R_6d_4, R_6d_5,              (6)  right 6D rotation
  R_grip ]                             (1)  right gripper [0, 1]
```

实现见 `pref_hdf5_dataset.py:200-214`(`_build_ee_20d`)和 `precompute_stats.py:32-46`。

### 5.2 为什么是 20D EE+6D,不是 joint(14D)或 quaternion EE(16D)

| 方案 | dim | 缺点 | 选择 |
|---|---|---|---|
| joint(`joint_action/vector`) | 14D | task-space 不可解释;eval 不能直接看 EE z/x;实机迁移要重做 IK | ✗ |
| EE + quaternion(xyz + xyzw) | 16D | quaternion **double-cover**(q 和 -q 同旋转,loss 不连续);**MSE on quat 不 SO(3) geodesic** | ✗ |
| **EE + 6D**(xyz + 6D rotation) | 20D | 平滑连续表示,无 double-cover,与 Diffusion Policy / RT-2 / OpenVLA 主线一致 | ✓ |

### 5.3 Quat-xyzw ↔ 6D 公式(`rotation.py`)

- `quat_xyzw_to_6d(quat)`(`rotation.py:16-26`):`R = Rotation.from_quat(quat).as_matrix()`(scipy 默认 xyzw),取 `R[:, :, 0]` 和 `R[:, :, 1]` 拼成 `(..., 6)`。Identity quat `(0,0,0,1)` → `(1,0,0, 0,1,0)`(已 sanity 验证)。
- `six_d_to_quat_xyzw(d6)`(`rotation.py:29-44`):Gram-Schmidt 正交化 `(a1, a2)` 得 `(b1, b2)`,叉积得 `b3`,`R = [b1, b2, b3]`,`Rotation.from_matrix(R).as_quat()` 输出 xyzw。
- **走 scipy 不走 pytorch3d** 的原因:pytorch3d 默认 wxyz(scalar-first),HDF5 是 xyzw(scalar-last,SAPIEN/ManiSkill 默认),pytorch3d 会**静默交换 w 和 x**。
- Round-trip(50 random quats):max |1 − |q·q'|| ≈ `1e-5`(已 sanity 验证)。

### 5.4 Chunk / window

| 参数 | 值 | 来源 |
|---|---|---|
| `past_action_window_size` | 0 | YAML;**v1 不支持 > 0**(`pref_hdf5_dataset.py:105-108`) |
| `future_action_window_size` | 15 | YAML |
| `chunk_size` | 16 = future + 1 | derived |
| 物理时长 | 16 帧 / 30 Hz ≈ **0.53 s** | — |
| State | 当前帧 1 步(`(1, 20)`) | `pref_hdf5_dataset.py:253-261` |

---

## 6. Loss 与训练超参(逐字 main-method 复用)

### 6.1 Loss

```
L = L_action      (baseline)
```

`L_action` = flow-matching velocity MSE,计算见 `LayerwiseFM_ActionHeader.py:312-318`:
```python
noise = randn_like(actions)
t = sample_time(...)                          # Beta(1.5, 1.0) → (1 - β)/s
noisy_trajectory = (1 - t) * noise + t * actions
velocity = actions - noise                    # target
# DiT predicts velocity given noisy_trajectory + cross-attn(vl_embs_list)
# loss = MSE(pred_velocity, velocity)
```

**main-method**:`L = L_action + λ · L_vqa`,其他不变。`λ` propose 起始 0.5,后续 sweep `{0.1, 0.3, 0.5, 1.0}`。

### 6.2 完整超参表(全部 verified against YAML)

| Group | 参数 | 值 |
|---|---|---|
| **trainer** | epochs | 100 |
| | max_train_steps | 50000(必要时扩 80000) |
| | num_warmup_steps | 5000 |
| | warmup_ratio | 0.1 |
| | save_interval | **5000**(10 个 ckpt over 50k) |
| | eval_interval | 1000 |
| | logging_frequency | 100 |
| | gradient_accumulation_steps | 2(4 GPU) / 1(8 GPU)— launch 自动 |
| | max_grad_norm / gradient_clipping | 1.0 |
| | enable_gradient_checkpointing | **false**(benchmark 实测 OFF 比 ON 快 ~8%;peak 仅升 ~10 GB,Zero2 下还有 24 GB 余裕) |
| | enable_mixed_precision_training | true(bf16) |
| | freeze_modules | `''`(full-ft;**LoRA 不用**) |
| | is_resume | false |
| **learning_rate** | base | 2.5e-5 |
| | qwen_vl_interface | 1.0e-5 |
| | action_model | 1.0e-4 |
| | lr_scheduler_type | `cosine_with_min_lr`,`min_lr = 1e-6` |
| **optimizer** | name | AdamW |
| | betas | (0.9, 0.95) |
| | eps | 1e-8 |
| | weight_decay(trainer-level) | 0.0 |
| | weight_decay(optimizer-level) | 1e-8 |
| **datasets.vla_data** | per_device_batch_size | 8 |
| | num_workers | 4 |
| | cameras | `[head_camera, left_camera, right_camera]` |
| | image_size | `[224, 224]` |
| | future_action_window_size | 15 |
| | past_action_window_size | 0 |
| | include_state | true |

**有效 batch 永远 = 64**:
- 4 GPU × per_device 8 × grad_accum 2 = 64
- 8 GPU × per_device 8 × grad_accum 1 = 64(launch 自动切换)

### 6.3 DeepSpeed Zero2

- Config file: `starVLA/config/deepseeds/deepspeed_zero2.yaml`(在 launch 里 `--config_file` 传)
- Optimizer state + grad partition 分 N rank;参数不分 → 跟 Zero3 不同,**单卡仍要装得下 ~8B params 的参数本体**,所以 8× H100(80GB)是底线。

---

## 7. 文件 layout

### 7.1 `examples/preference/` 全树

| 文件 | 行数 | 干啥 | 关键 API / 常量 |
|---|---|---|---|
| `__init__.py` | 0 | namespace placeholder | — |
| `dataset/__init__.py` | 0 | namespace placeholder | — |
| `dataset/rotation.py` | 69 | quat-xyzw ↔ 6D | `quat_xyzw_to_6d(quat)`, `six_d_to_quat_xyzw(d6)` |
| `dataset/prompt.py` | 159 | strip 流程 + clean template + final prompt | `PREF_LABELS`, `TASK_GROUPS`, `CLEAN_TEMPLATE`, `_STRIP_RE`, `_LEAK_RE`, `strip_v5(phrase, tg)`, `build_action_prompt(tg, pk, paraphrase)` |
| `dataset/pref_hdf5_dataset.py` | 330 | 主 dataset 类 | `PrefHDF5Dataset`, `get_pref_dataset(data_cfg, mode)`, `collate_fn`(identity), `_split_episodes(...)`(seed-42 stratified) |
| `dataset/precompute_stats.py` | 143 | (re)生成 stats JSON | CLI:`--data_root --out --split_seed --val_fraction` |
| `dataset/stats_giveobj_v1.json` | 356 | 20D action/state q01/q99/min/max/mean/std + meta | meta.n_train_frames=192927, n_train_eps=1280, n_val_eps=320 |
| `train_files/starvla_pref_stage_a_baseline.yaml` | ~110 | 完整 config | run_id `pref_baseline_stage_a_v1_noVQA`,wandb entity `kaiwenh-17-uiuc` / project `pref-sim` |
| `launch_pref_stage_a_baseline.sh` | 65 | accelerate + Zero2 launch | NUM_GPUS env;auto grad_accum;auto stats precompute |
| `tests/test_smoke.py` | 112 | 三层 smoke | `test_dataset_sample`, `test_build_dataloader`, `test_full_forward(--full)` |
| `tests/__init__.py` | 0 | placeholder | — |

### 7.2 主仓库改动(就一处)

`starVLA/dataloader/__init__.py`,新增 `elif dataset_py == "pref_hdf5"` branch(`__init__.py:61-74`):
```python
elif dataset_py == "pref_hdf5":
    from examples.preference.dataset.pref_hdf5_dataset import get_pref_dataset, collate_fn
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset = get_pref_dataset(data_cfg=vla_dataset_cfg, mode="train")
    vla_train_dataloader = DataLoader(
        vla_dataset,
        batch_size=vla_dataset_cfg.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=vla_dataset_cfg.get("num_workers", 4),
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_dir = Path(cfg.output_dir)
        vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
    return vla_train_dataloader
```

**没动其他主仓库代码**(没 patch `QWen3.py`,没改 `LayerwiseFM_ActionHeader.py`,没碰 `QwenPI.py`)。所有 pref 逻辑都封在 `examples/preference/` 下;prompt 是在 dataset `__getitem__` 阶段就组装好放在 `sample["lang"]` 里,framework 看到的是 plain text。

### 7.3 Sample dict 契约

`PrefHDF5Dataset.__getitem__` 返回:
```python
{
  "image":     List[PIL.Image]  # 3 个,每个 224×224
  "lang":      str               # "<stripped base prompt> Preference: low/high contact"
  "language":  str               # alias of "lang"
  "action":    np.float16 (16, 20)
  "state":     np.float16 (1, 20)        # 如果 include_state
  "robot_tag": "pref_hdf5"
}
```

字段对齐 `QwenPI.forward`(`QwenPI.py:95-99`)需要的 `image / lang / action / state`。

---

## 8. 怎么跑

### 8.1 主路径

```bash
# Recommended: 8× H100
NUM_GPUS=8 bash examples/preference/launch_pref_stage_a_baseline.sh

# Fallback: 4× H100 (auto grad_accum=2)
NUM_GPUS=4 bash examples/preference/launch_pref_stage_a_baseline.sh
# 或省略 NUM_GPUS,默认 4
bash examples/preference/launch_pref_stage_a_baseline.sh
```

Eff batch 永远 64(`launch_pref_stage_a_baseline.sh:28-34`)。

### 8.2 前置依赖

1. **conda env**:`conda activate starVLA`(`/mnt/localssd/kaiwenh/miniconda3/envs/starVLA`)。`h5py` 已通过 pip 装入。
2. **数据**:`/mnt/localssd/kaiwenh/pref/data/giveobj/`(已 verified)。
3. **HF cache**:`Qwen3-VL-4B-Instruct` 第一次跑会从 HF 拉(~8 GB)到 `/mnt/localssd/kaiwenh/cache/hf/hub/models--Qwen--Qwen3-VL-4B-Instruct/`。config 已下,full weights 首次会拉。
4. **stats JSON**:`examples/preference/dataset/stats_giveobj_v1.json` 已生成(2026-05-22)。launch 脚本会幂等检测,缺失时自动重跑:
   ```bash
   python -m examples.preference.dataset.precompute_stats \
       --data_root /mnt/localssd/kaiwenh/pref/data/giveobj \
       --out examples/preference/dataset/stats_giveobj_v1.json
   ```

### 8.3 输出

- ckpt → `./results/Checkpoints/pref_baseline_stage_a_v1_noVQA/`
- dataset_statistics.json → 同目录(每次 rank 0 重存一份,内容 = stats JSON)
- wandb run → entity `kaiwenh-17-uiuc`, project `pref-sim`, run_id `pref_baseline_stage_a_v1_noVQA`

### 8.4 OOM / 内存预算 reminder

- 实测(2026-05-22, single H100, SGD): batch 8 grad_ckpt OFF peak **55 GB**;grad_ckpt ON peak 45 GB。
- 8× H100 Zero2 实际预估(model 15.6 + activations 26 + sharded grads 2 + sharded AdamW 12)≈ **56 GB / GPU**,~24 GB 余裕。
- **推荐组合:8× H100 80GB + Zero2 + grad_ckpt OFF + per_device_batch 8**。
- 4× H100 配同样 batch 仍可能 OOM(model + activations 大,sharded fraction 小)。

### 8.5 Wall time(8× H100 实测)

| 配置 | step time | 50k step 8 GPU ETA |
|---|---|---|
| 无 `PYTORCH_CUDA_ALLOC_CONF`(launch_pref_stage_a_baseline.sh 旧版)| **4.4 s/step** | ~60 h ❌ |
| 加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | **1.17 s/step** | **~17 h** ✓ |

**4× 速度差** 来自一个 env var(2026-05-22 在 step 187 实测)。原理:Zero2 sharded state 在 high-rank GPU 上碎片严重,**没有 expandable_segments 时 PyTorch 的 caching allocator 频繁 fallback 到 `cudaMalloc` 慢路径**(全 device 同步,每步多 ~3 秒)。这个 env var 在 launch_pref_stage_a_baseline.sh 里已默认带,**main-method 也必须带**,否则 ablation 速度比较不公平。

### 8.6 Optimization smoke

```bash
# Run on single GPU; sweep configs to compare ms/step + peak memory
CUDA_VISIBLE_DEVICES=0 python -m examples.preference.tests.benchmark
CUDA_VISIBLE_DEVICES=0 python -m examples.preference.tests.benchmark --no_grad_ckpt
CUDA_VISIBLE_DEVICES=0 python -m examples.preference.tests.benchmark --attn flash_attention_2
```

### 8.7 Smoke

```bash
# Fast: tests 1+2 (dataset + dataloader)
python -m examples.preference.tests.test_smoke

# Heavy: + full Qwen3-VL-4B forward (需要 ~8GB+ VRAM)
python -m examples.preference.tests.test_smoke --full
```

---

## 9. Smoke test 状态(2026-05-22)

| Test | 干啥 | 状态 |
|---|---|---|
| 1 | `PrefHDF5Dataset.val[0]` 结构 check;action / state ∈ [-1, 1];prompt 含 `" Preference: "` 且 base 无 leak | **PASS** — val=42,983;shape (16,20)/(1,20);lang OK |
| 2 | `build_dataloader(cfg, dataset_py="pref_hdf5")` 返回 DataLoader;一个 batch_size=2 batch 结构正确 | **PASS** |
| 3 | 完整 `Qwen_PI(cfg)` 实例化,`.to(cuda)`,`model(batch)` 得 scalar loss | **PARTIAL** — 模型构造 ✓、HF 权重下载 ✓、DiT 实例化 ✓;**`.to(cuda:0)` OOM** 当时 8× H100 被另一个 job 占 ~99%。**不是代码问题**,等空闲 GPU 即过 |

---

## 10. Eval 方法学(offline-only v1)

### 10.1 §7 baseline 验证(held-out val,3 个指标)

1. **Per-(task × pref) action MSE on val**
   - 跑 1 个 epoch over val(42,983 samples,16 task dirs × 2 pref = 32 buckets)。
   - 报每 bucket 的 mean MSE(normalized 20D space)。
   - 目的:**BC 基本健康度**;比较 baseline vs main-method 哪个更难 fit。

2. **Counterfactual pref-sensitivity**(**主指标**)
   - 对每个 val sample `(img, state)`,**分别**用 pref=25 / pref=75 各 forward 一次(只换 prompt 后缀)。
   - 报 `‖a_25 − a_75‖_2`(normalized 20D action space),每 task group 平均。
   - 直觉:>> 0 = 模型用了 pref token;~0 = 忽略。
   - **干净 ablation 的关键**:跟 main-method 同一套测,任何 base prompt 里的 pref leak 都会让 baseline 也"成功",这就是为什么 §4 strip 100% 是必要的。

3. **Δz / Δx sign accuracy**
   - **只在 3 个 task 报**(§2.6 表里 ≥ 0.7σ 的):`put_boxdrink_dustbin`(z), `put_callbell_dustbin`(z), `put_fork_dustbin`(x)。
   - 流程:counterfactual 首帧 EE。若 `(EE@pref=75) − (EE@pref=25)` 在 pref-canonical 方向(站立 → +z,平躺 → +x)符号正确,记 1。每 task 报平均符号正确率。
   - 注意 `put_screwdriver_dustbin` x 信号弱(<0.3σ)+ 双轴混合,**不报**。`give_*` 随机臂,世界系信号被抵消,**不报**。

### 10.2 §8 最终汇报(rollout 接入后)

- 复用 §7 三个指标(同函数,**main-method / baseline 测同一份 grid**)。
- Sim rollout 接入后追加 **task success rate**(目前用户没现成 sim rollout pipeline,后续做)。

### 10.3 Eval data 来源

- §7 全程用 train 内部 split 出来的 val(seed-42 stratified)。
- 后续如果要严格 unseen-prompt:`instructions/.json[unseen][0..99]` 提供 100 paraphrase / episode,可在 eval 时切换,**不混入 train**。当前 v1 train 用 `seen`,eval 也用 val 集 episode 里的 seen — 已经做了 episode-level held-out。

---

## 11. 已知 / 待办

### 11.1 数据相关

- **Group B(unlabeled target)data not on disk** — `/mnt/localssd/kaiwenh/pref/data/` 下只有 `giveobj/`。Phase 1b 跑之前要另行下载。
- **`_50` 中间档** 已从 disk 移除 — 若未来要 25/50/75 三档需要重新下载。
- **没有 object mesh / per-frame object pose** — eval 用 task-relative,不能 object-relative。

### 11.2 模型 / 训练

- **DiT 真实 size = 3.3B**,被 §3.3 的 runtime override 决定 — 强烈建议 8× H100 + Zero2。4× H100 可能 step OOM。
- **`put_*_dustbin` 右臂全 0 → 强 do-nothing signal**,模型可能 over-learn 这个 mask 模式。v1 不加 mask,§7 metric 1 看到再改。

### 11.3 pref signal 弱的 task

- `give_fork`, `give_screwdriver`(随机臂 × 平躺,世界系信号在臂间抵消):signal/σ 约 0.2-0.4。
- §7 metric 3(sign accuracy)**不在这些 task 上报**。
- counterfactual MSE(metric 2)还是能反映 — 即便信号弱,如果模型用 pref token,输出 action 也会有差。

### 11.4 prompt 来源

- **当前用 paraphrase strip**(v5 hybrid → 100% strip / 0% leak on 160k phrases),fallback path 留 safety net。
- 用户提过**未来可能换成手写 base instruction**(不走 paraphrase)。如果换,只需重写 `build_action_prompt`,数据 / norm / 模型不动 — paraphrase 随机性会消失,从这一点上 baseline 跟 main-method 更对齐(每条 sample 同 prompt)。但当前不换,因为 v5 已经 100% clean。

### 11.5 Smoke

- Test 3 还需要在空 GPU 上跑过一次 sanity check 才能正式动手训。

---

## 12. 给 main-method 的 hand-off checklist

要从这条 baseline 派生 main-method(Stage A + VQA cotrain)时,**逐字遵守**以下规则。

### 12.1 抄(byte-identical)

- `examples/preference/dataset/rotation.py`
- `examples/preference/dataset/prompt.py`(包括 `PREF_LABELS`, `CLEAN_TEMPLATE`, `_STRIP_RE`, `_LEAK_RE`, `strip_v5`, `build_action_prompt`)
- `examples/preference/dataset/precompute_stats.py`
- `examples/preference/dataset/stats_giveobj_v1.json`(同一份,**别重新算**,否则 train/val frames 数会变)
- `starVLA/dataloader/__init__.py` 的 `elif dataset_py == "pref_hdf5"` branch(可换成 `"pref_hdf5_vqa"` 之类新名,但 wiring 一致)
- YAML 99% 抄(改名 run_id / wandb)
- launch script 99% 抄(改名 ckpt 路径)

### 12.2 改(逐项明确)

| 改哪 | 怎么改 |
|---|---|
| `pref_hdf5_dataset.py` `__getitem__` | 加一个 `vqa` 字段,每条 sample 带 `{question, answer}`(简单 schema:Q = "What is the preference?",A = `low contact` / `high contact`,或者更复杂的 VQA pair) |
| `QwenPI.py`(或 fork 一个 `QwenPI_VQA.py`) | `forward` 算 LM head 的 `L_vqa`,**在 framework 内部**返回单个聚合的 `action_loss = L_action + λ * L_vqa`。理由:`train_starvla.py:_train_step` 只读 `output_dict["action_loss"]` 作为 `total_loss`(line ~427-428),trainer 本身没有 `loss_scale` 聚合逻辑;`loss_scale.vla / vlm` 字段在当前 script 里是 dead config。 |
| YAML | 加 `framework.vqa.lambda_vqa: 0.5`(或 sweep);不要在 trainer 里加 loss_scale —— trainer 不读 |
| 不需要改 trainer | 因为聚合发生在 framework 内部,trainer 拿到的 `action_loss` 已经是 total |

### 12.3 **不要**改

- 数据 root、task group list、split seed、val fraction → 训出来的 train/val 必须是同一 1280/320 episodes
- `paraphrase_seed`(虽然不是 sample-level 对齐,但 baseline / main-method 同 dataset 实例 RNG 至少把 macro 分布对齐了)
- norm stats(继续读 `stats_giveobj_v1.json`)
- 超参 / optimizer / scheduler / 相机选 / image_size / chunk_size
- prompt 函数(stripped + " Preference: ..."。VQA 在**额外字段**上,不动 action prompt)

### 12.4 验证 main-method ablation 干净度的方法

跑完两个 ckpt 后:
1. **同一份 val grid**(seed 42 split),分别测 §7 三指标。
2. **逐 sample 同 prompt**(用同一份 paraphrase RNG state — 如果 v1 paraphrase 不对齐造成噪声,把 `_paraphrase_rng = Random(idx)` 函数化,两边重训)。
3. 报 baseline vs main-method delta per metric;如果 main-method 加了 VQA 之后 metric 2(counterfactual MSE)↑ 而 metric 1(per-task MSE)持平或略降,则 main-method 的 VQA 信号在帮助 pref grounding。

---

## 13. References

### 13.1 Session memory(背景 + 决策记录)
- `/home/kaiwenh/.claude/projects/-home-kaiwenh-starVLA/memory/project_pref_vla_stage_a.md`
- `/home/kaiwenh/.claude/projects/-home-kaiwenh-starVLA/memory/reference_pref_data_and_starvla.md`
- `/home/kaiwenh/.claude/projects/-home-kaiwenh-starVLA/memory/feedback_pref_vla_proposal_first.md`

### 13.2 关键 starVLA 文件(只读 source-of-truth)
- 训练 entry:`starVLA/training/train_starvla.py`(line ~482)
- Framework:`starVLA/model/framework/QwenPI.py:38-49` 类、`80-144` forward、`146-196` predict_action
- Action head:`starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py:246`(`LayerwiseFlowmatchingActionHead`);`256-262` 是 runtime override;`303-340` forward
- DiT 默认:`LayerwiseFM_ActionHeader.py:214`(`DiTConfig = {"num_layers": 36, ...}`)
- VLM prompt 拼接(我们**不 patch**):`starVLA/model/modules/vlm/QWen3.py`(以及 `QWen2_5.py`)
- 主 dataloader 分派:`starVLA/dataloader/__init__.py:36-74`(`build_dataloader`)
- LeRobot 路线对照:`starVLA/dataloader/lerobot_datasets.py`

### 13.3 历史脚本(启动结构参考)
- `r-old-preference/0320-v40-training.sh` — 2025-03 老 RoboTwin / Qwen2.5-VL-3B / 16D quaternion EE / DiT-B / 50k steps / 4 GPU × batch 8 × grad_accum 2。**只看 accelerate launch 结构**;**别抄 action_dim / base_vlm / 数据 mix**。

### 13.4 YAML 结构参考
- `examples/calvin/train_files/starvla_train_calvin.yaml`(另一份 starVLA train config,看 YAML 顶层结构)

### 13.5 数据
- Source:`/mnt/localssd/kaiwenh/pref/data/giveobj/`(16 task dirs,1600 demos)
- Stats:`examples/preference/dataset/stats_giveobj_v1.json`(192,927 train frames pool)

---

End of doc.
