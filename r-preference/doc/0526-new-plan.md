# 2026-05-26 新 plan — place VQA multi-cam + late-frame retrain,跨 cat sampling strategy 设计

> **Self-contained planning + diagnostic doc.** 写于 2026-05-26 下午,在用 H200 retrain `place` VQA 之前 freeze 所有发现 + 决策。
> 读完这一份你应该完全理解:为什么 place VQA 在 taskB 不 work(三层 root cause)、为什么 fix 是 "active_wrist + late_8"(物理 + diagnostic 推理)、怎么把同一 design principle 推广到其它 4 个 cat。
>
> **配套 doc**(只引,不复述):
> - [`0525-problems-and-how-to-solve.md`](0525-problems-and-how-to-solve.md) — 2026-05-25 4 cat gate eval RED 的 postmortem(§3.2-§3.4 已被 0526 推翻)
> - [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) — 推翻 0525 的诊断 + 7-step pipeline audit
> - [`0524-stageB-contact-plan.md`](0524-stageB-contact-plan.md) — Stage B 框架(filter / firewall / 训练 recipe)

---

## 0. TL;DR

**问题**:`place` VQA 训练 healthy(L_action 0.014, L_vqa 0),taskA val acc 0.79,但 **taskB pseudo-label acc 全 strategy stuck 0.33-0.49**(uniform_8 / mid_8 / late_8 / dense_16 都 fail)。Stage B place 因此 blocked。

**3 个独立 diagnostic 角度的 finding**:
1. **数据 signal 存在**:taskB `_center` vs `_corner` placement xy 差 Δy=27mm,S/N_y=1.33(跟 taskA tray 同量级)
2. **视觉本质**:taskB `place_soap2_stand` 的 stand target 物理小(~5cm),head_camera 在 1.35m 高俯瞰,center/corner 在像素图上只差几个 pixel(L2 pixel distance = 0.018,比 taskA 平均 0.030 都低)
3. **模型实际学了啥**:**confident-wrong on taskB**(70% 错预测 \|logit_gap\|>5),taskA 自己只在 tray 上学会了(0.85-1.0),pad 只 0.50-0.71 — 用 **tray-specific quadrant features 当 shortcut**,迁到 stand 失败

**Root cause 三层**:
- **L1 texture shortcut**(tray 蓝色边界 anchor 中心位置)
- **L2 pixel resolution**(pad/stand 在 head_cam 看太小)
- **L3 single-view limit**(VQA forward 只用 head_camera,wrist 视角根本没喂)

**Fix**:**A+B 组合改 VQA training cache**:
- **A**:VQA forward 加入 **active_wrist** camera(active arm 的腕部 close-up;inactive arm 不喂避免浪费)
- **B**:cache 用 **late_8** frame strategy(fractions 0.65-0.98,围绕 placement moment frac=0.91),不是 uniform 整 episode

**Run plan**:H200 上重训 place VQA 25k 步,save_interval=2500(10 ckpts),`n_vqa_per_batch=1`(同原 VQA load),预计 **~9 h**;之后 eval 每个 ckpt 看 taskB acc trajectory,选最好的 + 决定是否未来 cat 用更少步。

**对 height/hvlv/orient 的扩展**:本 doc §6 给 5 cat 的 sampling strategy 表(每个 cat 的 pref signal 物理位置 → 推荐 cameras/strategy/n_frames)。

---

## 1. 前因 — place VQA 出了什么问题

### 1.1 2026-05-24/25 训完的 place VQA 表现

| 指标 | place VQA 25k | (参考)contact VQA 35k |
|---|---|---|
| L_action 收敛 | ✓ 0.04 | ✓ 0.014 |
| L_vqa 收敛 | ✓ ~0 | ✓ ~0 |
| taskA val VQA acc | **0.79**(看似还行)| 0.86 |
| **taskB acc (uniform_8)** | **0.44** ❌ | 0.61 → mid_8 救到 0.99 |
| Stage B pseudo-label launch gate(post-filter acc ≥0.95)| **0.194 — FAILED** | passed |

### 1.2 试过 6 个 frame strategy 都 stuck

用 `frame_window_test.py` + 新加 `late_8` / `late_dense_16` strategy(覆盖 release moment),全 inference 在原 25k ckpt 上:

| strategy | taskB acc | pred dist(center / corner)| signal gap |
|---|---|---|---|
| uniform_8 | 0.44 | 60 / 40 | +1.18 |
| mid_8 | 0.35 | 33 / 67 | −9.59 |
| dense_16 | 0.44 | 94 / 6 | +0.16 |
| mid_dense_16 | 0.33 | 31 / 69 | −10.66 |
| **late_8** | **0.49** | 97 / 3 | +2.16 |
| late_dense_16 | 0.48 | 88 / 12 | −1.00 |

**关键现象**:不同 strategy 让模型 **collapse 到不同的一类**(uniform/late 偏 center >90%,mid 偏 corner >65%)。signal gap 跟 contact 的 +49 比小 20-40 倍。**模型不是 "看不出来",是 "看了图后 confidently 错读"**。

→ **frame strategy alone 救不了 place**(跟 contact 0524 那次 uniform→mid 救回 0.99 的情况性质不同)。

---

## 2. 三个独立 diagnostic 角度的发现

### 2.1 Angle 1 — 数据 signal(直接量物理量)

测 taskB `place_soap2_stand` 100 ep 的 active arm xy at release moment:

| pref | n | mean x | mean y | std x | std y |
|---|---|---|---|---|---|
| center | 50 | +0.008 | -0.124 | 0.029 | 0.014 |
| corner | 50 | +0.018 | -0.096 | 0.081 | 0.021 |

**Δy = 27 mm, S/N_y = 1.33**(跟 taskA tray 任务同量级,见下)。**信号不弱**。

scene_info 还告诉我们:
- center: target_xy = (-0.017, -0.124) ≈ stand 中心
- corner: target_xy = (-0.069, -0.095),从 stand 中心 offset 52mm in x,30mm in y
- stand 几何上小,"corner" 实际意义是 "靠近 stand 边/偏离中心 ~5cm"

→ 物理上有 5cm 偏移信号,可学。

### 2.2 Angle 2 — 视觉 distance(pixel L2)

`/r-preference/eval/viz_place_diagnosis/` 下 3 张图 + JSON stats:

```
taskA → taskB pixel L2: 0.028  (taskA cross-task 内部 0.022-0.058)
center/corner pixel L2:
  - taskB stand     : 0.018  ← 所有 task 最低!
  - taskA tray avg  : 0.030
  - taskA pad avg   : ~0.025
```

→ taskB 全场景在 taskA 分布**内**(0.028 中位偏低,not OOD),但 **center vs corner 的视觉差异是所有 task 最小**。
→ stand 物理小 + head_camera 距离远 → center vs corner 在像素图上只差**几个 pixel**(soap 物体本身的尺寸都比 placement offset 大)。

### 2.3 Angle 3 — 模型实际行为(loaded place VQA 25k 在 H200,跑 diagnostic.py)

**taskB 100 ep dump**:
| GT | n | acc | logit_gap mean | wrongs 的 gap mean | confident-wrong(\|gap\|>5)|
|---|---|---|---|---|---|
| center | 50 | 0.54 | +1.88 | -5.98(自信预测 corner!)| **12 / 23 wrongs** |
| corner | 50 | **0.34** | +0.69 | +6.34(自信预测 center!)| **23 / 33 wrongs** |

**histogram 是 bimodal**:模型不是 "猜不到",是 **强自信地猜错**。

**taskA val per-task acc** 拆开:
```
place_*_tray  acc: 0.80-1.00  ← 模型只学好这 4 个 task
move_*_pad    acc: 0.50-0.71  ← 一半都没学好
move_soap_pad acc: 0.50       ← 完全 random
```

**unconditional bias check**(model 看 blank/noise 输入):
- blank black: gap = -1.50(略偏 corner)
- noise × 3: gap ≈ -0.1 ~ -0.25(几乎平局)

→ **没有 strong default bias** — 0.44 collapse 不是 LM head 偏一边,是真在 "看错图"。

### 2.4 三角合一的诊断

| 假设 | 证据 | 结论 |
|---|---|---|
| H1: taskB 没信号 | Δy=27mm S/N=1.33 | ❌ refuted |
| H2: taskB 全场 OOD | pixel L2(taskB, taskA) < cross-task 内部 | ❌ refuted |
| **H3: stand center/corner 在 head_cam 像素上太接近** | L2(center, corner)=0.018 = taskA 平均的一半 | ✅ 部分对 |
| **H4: 模型只学了 tray-specific shortcut,没学通用 center/corner** | tray 1.0 acc / pad 0.5-0.7,且 wrong 是 confident wrong | ✅ confirmed |
| **H5: VQA forward 完全没用 wrist cam(架构限制)**| `vqa_sample.py:344` cache builder hardcoded `camera=VQA_CAMERA=head_camera` | ✅ confirmed |

---

## 3. 架构澄清 — 实际 camera 配置(之前 doc 没说清)

跑了一个 inspector 确认 4 个 camera 的物理类型:

| camera | t=0 world pos | extrinsic_std 跨 T | 类型 |
|---|---|---|---|
| `head_camera` | (-0.03, -0.45, **+1.35**) | 0.000 | **静态 overhead** |
| `front_camera` | (0.00, -0.45, +0.85) | 0.000 | 静态 front(没用)|
| `left_camera` | (-0.33, -0.24, +1.01) | **0.455** | **MOVING = 左腕 wrist cam** |
| `right_camera` | (+0.27, -0.24, +1.01) | 0.000 / 0.455(取决于 active arm) | **右腕 wrist cam** |

→ **`left_camera` / `right_camera` 是腕部 wrist 摄像头,不是"左/右边的固定相机"**(0522 doc 一直没明说,我之前也误以为是固定 side cam)。

### 当前 2 stream 给 VLM 的 input 差异

| stream | cameras 用了几个 | 每 cam 几 frames | 总 images per VQA sample |
|---|---|---|---|
| **L_action** | 3(head + L wrist + R wrist) | **1 frame @ frame_idx** | 3 |
| **L_vqa** | **1(head_camera only!)** | **8 frames uniform_linspace 整 episode** | 8 |

→ **VQA 完全没用 wrist cam**!这是 0522 doc §2.2 的 design choice("Qwen3-VL 是 video VLM,head cam 8-frame 应该够"),但对**小 target task 不够**:
- head_camera 1.35m 高俯瞰
- pad/stand 5cm 大小
- center vs corner 物理偏移 3cm → 像素图上 ~10 个 pixel(stand 上)
- wrist cam 在物体旁 ~10cm,close-up,同样偏移在像素图上 ~50 个 pixel(5x resolution)

---

## 4. Fix design — A+B 组合

### 4.1 Solution A:VQA forward 加 wrist camera

把 VQA 输入从 1 cam(head only)变成 2 cam(head + **active_wrist**)。

**关键 design 选择 — 为什么 active_wrist 而不是直接 head+L+R 3 cam**:
- 这是 dual-arm 任务,每 episode 只有 1 只臂 active(从 scene_info `{a}` 或检查 endpose xyz range 知道)
- 另一只 wrist cam 在 home 位置静止,看的全是空场景
- 喂进 VQA = 浪费 8 个 frame 没信息,还 2× 增加 token / activation memory(H100 OOM 实测,见 §5.3)
- → **2 cam 模式 (head + 当前 active arm 的 wrist) = 16 images per VQA sample,跟原 head-only 2 sample 的 16 images 同 VLM load**

### 4.2 Solution B:VQA cache 用 late_8 frame strategy

把 cache 的 frame indices 从 `uniform_clip_indices(T, 8)` = `np.linspace(0, T-1, 8)` 改成 `late_clip_indices`:

```python
DEFAULT_LATE_FRACS = (0.65, 0.72, 0.79, 0.85, 0.88, 0.92, 0.95, 0.98)
```

物理上:place taskB placement 在 frac=0.91(实测 release moment 落点)。late_8 8 个 frame 集中在 0.85-0.98 zone,覆盖 placement before / during / after 短时窗。

**关键 — 为什么 train 时用 late_8 而不是 inference 时换 late_8**:
- 我们今天试过 inference 用 late_8 在原 uniform_8-cached ckpt 上 → acc 只 0.49(原 0.44 → 仅 +0.05)
- 原因:**模型的 attention 是按 train 时的 frame distribution learn 的**,inference 换分布 attention 没 align,新 frame 喂进去 model 不会自动 reweight
- **train 时 cache 就用 late_8 → 模型从 step 1 就学怎么从 placement-zone 8 帧 read center/corner → inference 同 distribution → 真 work**

### 4.3 A+B 合一的好处(orthogonal)

| axis | 解决哪层 root cause | 没解决的 |
|---|---|---|
| A (wrist) | L2 (pixel resolution) + L3 (single view) | L1 (texture shortcut) |
| B (late_8) | partial L2(timing 集中)| L1, L3 |
| **A+B** | **L2 + L3 完整,L1 partial**(texture 还在但 model 看 2 cam multi-view 后会减弱依赖)| 残留 L1 — 如果 A+B 不够好,补 image augmentation |

---

## 5. Implementation details

### 5.1 Code changes(4 files)

**1. `examples/preference/dataset/vqa_sample.py`**:
- 加 `DEFAULT_LATE_FRACS` + `late_clip_indices()` function
- `VQACategoryConfig` dataclass 加 3 个字段:`cameras: Tuple[str, ...]`, `clip_strategy: str`, `n_frames: int`(其它 4 个 cat 用 default = head_camera/uniform_8/8 → backward compat)
- 加 `ACTIVE_WRIST_SENTINEL = "active_wrist"` + `_resolve_active_wrist(h5)` helper(从 endpose ptp 判 active arm) + `_resolve_cameras_for_ep` 替换 sentinel
- `build_vqa_clip_cache` 接 `cameras: Sequence[str]` + `strategy: str`,cache shape:
  - single-cam: `(N, F, H, W, 3)`(backward compat)
  - multi-cam: `(N, F, n_cams, H, W, 3)`
- `cache_row_to_pil` 兼容 4D / 5D shape
- `load_clip_by_strategy` 加 `cameras` 参数(inference 路径)
- `_compute_clip_indices` shared helper(cache build 和 inference 同一逻辑)
- VQA_CATEGORIES["place"] 更新:`cameras=("head_camera", "active_wrist"), clip_strategy="late_8", n_frames=8`

**2. `examples/preference/dataset/pref_hdf5_vqa_dataset.py`**:
- Dataset 读 `VQA_CATEGORIES[category]` 拿 cameras / strategy / n_frames
- factory `get_pref_vqa_dataset` 支持 YAML 可选 override(`vqa_cameras`, `vqa_strategy`, `vqa_num_frames`)

**3. `starVLA/model/framework/QwenPI_VQA.py`**:
- `_vqa_forward` 用 `cache_row_to_pil(row_view)` 替代硬编码 4D index(automatically handle 5D shape)

**4. `examples/preference/train_files/starvla_pref_stage_a_vqa_place.yaml`**:
- 删除老 `vqa_camera: head_camera` 覆盖(否则 silently 退回 single-cam — code review 抓到的 launch blocker!)
- 加显式 `vqa_cameras: ["head_camera", "active_wrist"]`, `vqa_strategy: late_8`(自我 documentation)

### 5.2 Subagent code review(careful pass)

Spawn 一个 general-purpose subagent 跑了 12 项 checklist:
- ✓ Backward compat:contact/giveobj/height/hvlv/orient 默认值 = 旧行为,5D shape 只在 place 触发
- ✓ Cache shape correctness(single 4D vs multi 5D)
- ✓ PIL ordering(time-grouped:cam0_t0, cam1_t0, cam0_t1, ...,跟 inference path 一致)
- ✓ `_compute_clip_indices` 共享 → cache + inference 同 frame 选择
- ✓ Per-cat config propagation
- ✓ Qwen3-VL 24-image input token count = 1604(32k context 充裕)
- **❌ Launch blocker**:place YAML 里 `vqa_camera: head_camera` 是 legacy field,**会触发 factory 的 `elif vqa_camera is not None` 分支**,silent 覆盖 cat config → A+B 变 no-op。**已修**(删掉那行,显式加 `vqa_cameras: [head, active_wrist]`)
- ⚠ Stage B labeler `pseudo_label_offline.py` + `diagnostic.py` 还没改 multi-cam — train 完才 eval 时需要 patch(已记录)

### 5.3 Smoke test 过程

1. **H100 smoke v1**(default n_vqa=2,3 cam ablation):**CUDA OOM**(24 images × 2 samples,H100 80GB 不够)
2. **H100 smoke v2**(n_vqa=1,3 cam):**还 OOM**(24 images × 1 sample 也撑不住)
3. **改 active_wrist mode**(2 cam,16 images per sample),scp 到 H200
4. **H200 smoke v3**(n_vqa=2,active_wrist):**5 步成功,exit 0**
   - L_action 6750(init high,正常)
   - L_vqa 1.51-1.71(典型 binary CE 早期值)
   - model_time 1.99 s/step at step 5(还在 warmup,不稳)
   - No OOM(H200 141GB/GPU 够用)

---

## 6. Per-cat sampling strategy 设计(全 5 cat 推广)

`active_wrist` + `late_8` 不是 one-off fix。**design principle**:

> Pref signal 在 trajectory 哪个 phase / 哪个空间尺度,VQA cache 就 sample 哪儿、用哪些 cam。

按 pref 信号物理位置 + target 几何大小,给 5 cat 的推荐:

| cat | pref signal phase | target 大小 | 推荐 cameras | 推荐 strategy | 推荐 n_frames | 优先级 |
|---|---|---|---|---|---|---|
| **contact** | grasp moment + 全程 wrist 6D rot 持续(grasp top vs bottom → wrist orientation 一直不同)| 中等(bottle / call bell)| head_camera(已 work)| **mid_8**(frac 0.25-0.70,grasp 在 mid)| 8 | P3 — 50k VQA ckpt 已 ✓,不动 |
| **place** | **placement moment**(frac 0.91)| **小**(pad/stand 5cm)| **head + active_wrist** ★ | **late_8** ★ | 8 | **P0 — 正在 retrain** |
| **height** | **release moment**(frac 0.91)| 中等(target 是 pad/stand 但信号是 z 轴 drop 高度差 14 cm,head 能看)| head only(z 信号 head 够看)| **late_8** | 8 | **P2 — 先 fix 优化 basin trap(warm-start),然后 VQA late_8 retrain** |
| **hvlv** | **整段 transport**(detour shape)| 中等 | head only(transport 在 xy 平面,head overhead 看得清整段)| **dense_16** 或 **mid_dense_16**(frac 0.20-0.80 覆盖 transport)| **16**(trajectory-wide,需要更多帧)| **P2 — 同 height** |
| **orient** | grasp moment + 全程 wrist 6D rot 持续 | 中等 | head only(signal 是 wrist rotation,head 能分辨 horizontal vs vertical)| **mid_8** 或 **gripper_anchored** | 8 | **P1 — VQA cotrain 本身崩(resume + 2 次 fresh 都 stuck @ 1.5)需要先 lambda_vqa ablation** |

### 6.1 三条 design rules 总结

1. **Pref signal 在哪 → frame cluster 在哪**
   - grasp-moment → `mid_8` / gripper_anchored
   - release/placement-moment → `late_8`
   - 全程 trajectory → `dense_16` / `mid_dense_16`
   - **不要 uniform_8 if signal localized** — 6/8 frames 浪费成 distractor

2. **Camera 看 target 几何大小 vs head_cam 分辨率**
   - 大 target(tray / 站立物体): head_camera 够看
   - **小 target(pad / stand / 小物件): 加 active_wrist** — head_cam pixel resolution 不够
   - 用 `active_wrist` sentinel 自动 per-ep 解析,比硬 list head+L+R 省 50% token

3. **n_frames 看信号时间跨度**
   - 信号 localized at 1 moment(grasp / release): 8 frames 足够
   - 信号 trajectory-wide(hvlv detour 100+ frames): **16 frames** 才能 cover 形状

### 6.2 跨 cat 的 effort matrix

| cat | code change | 重训 | 总 effort |
|---|---|---|---|
| contact | 0 | 0 | 0(已 work) |
| place | ✓ A+B done | ~9h H200 | 当前 in flight |
| height baseline | 0 | warm-start from place baseline 25k,~10h H100 | low(只优化 fix) |
| hvlv baseline | 0 | warm-start,~10h H100 | low |
| height VQA | 改 VQA_CATEGORIES["height"] cat config 1 行 | warm-start + late_8 retrain,~9h | low |
| hvlv VQA | 同上 + n_frames=16 | warm-start + dense_16 retrain,~10h(16 frames 慢一点) | low |
| orient VQA | lambda_vqa ablation 先(0.05/0.1/0.3),找到 stable 值后 retrain | ~3h ablation + ~9h main | mid |

**关键观察**:height/hvlv 是 **优化问题**(warm-start 修),不是 sampling 问题。我们已验证 warm-start work — 1.45 stuck baseline 用 place ckpt 起手 100 步就降到 0.05。所以 height/hvlv 不一定需要 sampling strategy 改动也能 work,只是不 optimal。

---

## 7. Run plan(短期 = 今晚 + 明天)

### 7.1 当前 in flight(H200,2026-05-26 ~下午)

`pref_main_stage_a_v1_VQA_place_v2_multicam`:
- yaml: `starvla_pref_stage_a_vqa_place.yaml`(改完的 multi-cam late_8 版本)
- data: `/mnt/localssd/kevin/pref/data/place`
- pretrained: 无(from scratch — place 不是 basin trap,不需 warm-start)
- max_train_steps: 25000
- save_interval: 2500(10 ckpts: 2.5k, 5k, 7.5k, 10k, 12.5k, 15k, 17.5k, 20k, 22.5k, 25k)
- n_vqa_per_batch: **1**(原 default 2,但 multi-cam load 翻倍,改 1 维持 token 数等于原 head-only setup)
- ETA: **~9-10 h**

### 7.2 训完之后

1. **Update eval scripts**(我训练时同步做)— `stage_a_gate.py`, `frame_window_test.py`, `diagnostic.py`, `pseudo_label_offline.py` 都要支持 multi-cam(读 `VQA_CATEGORIES[cat].cameras` 而不是硬编码 head)
2. **Eval 全 10 ckpts on taskB**(~10 min/ckpt × 10 = ~100 min)
3. **看 taskB acc trajectory**:
   - 2.5k → 5k:看 L_vqa convergence
   - 10k → 15k:看 transfer 是不是 plateau
   - 20k → 25k:确认 saturate
4. **挑 best ckpt** + 决定**未来 cat 用多少 step 就够**(可能 15k 足够,省后续 4 个 cat 的训练时间)

### 7.3 后续(48 h 内)

**P1 — orient VQA lambda ablation**:
- 1h 跑 lambda_vqa=0.05 / 0.1 / 0.3 短跑(各 1500 步)
- 找到 stable lambda 后,从 contact VQA 50k warm-start + late_8(grasp moment)retrain 25k,~9h

**P2 — height / hvlv warm-start 重训**:
- 已验证 warm-start work
- 4 个 run:height baseline / hvlv baseline / height VQA / hvlv VQA
- 都从 place baseline 25k 起手
- 25k 步 each,~10h × 4 = sequential 2 天 OR parallel H100+H200 一天

**P3 — 如果 place A+B 不够好**(taskB acc < 0.70):
- 试 image augmentation(RandomCrop + ColorJitter)+ 第三轮 retrain(~10h)
- 或上 D = A+B+C 全部

---

## 8. 风险 / open questions

### 8.1 已知风险

| 风险 | 概率 | 影响 | mitigation |
|---|---|---|---|
| H200 OOM(虽然 smoke 过)| low | 重启 with n_vqa=1 + 进一步降 frames | 已 set n_vqa=1 buffer |
| step_time 比估计慢(smoke step 5 = 1.99s,可能 saturate 1.6-1.8s)| mid | ETA 拉到 12h instead of 9h | 不影响最终结果 |
| `active_wrist` 在 inactive ep 解析错(两只臂都不动)| very low | 默认 fallback right_camera | 已有 sentinel resolution 逻辑 |
| Eval script multi-cam 没改完导致 eval crash | mid | 推迟 eval 1-2h | 训练时同步改 |

### 8.2 Open questions(训练完才能知道)

1. **taskB acc trajectory 形状** — 早收敛(10k plateau)还是慢爬(25k 还在升)?
2. **active_wrist 是不是 enough,还是真需要 head+L+R 3 cam**? — 如果 active_wrist 给 0.65 而不是 0.85,可能要试 3 cam(但 OOM risk)
3. **late_8 fractions 是不是最 optimal** — 现在是 (0.65, 0.72, 0.79, 0.85, 0.88, 0.92, 0.95, 0.98),也许 (0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.95, 0.97)(更集中 placement moment)更好
4. **L1 texture shortcut 残留** — A+B 后 taskA pad 是不是从 0.5-0.7 升到 0.85+?如果没升,texture aug(C)也要加

### 8.3 这条线对 paper 的意义

如果 place A+B work(taskB acc ≥ 0.80):
- ✓ **multi-camera + task-anchored frame strategy** 成为 self-labeling 的**通用 design pattern**(可推广 height/hvlv/orient)
- ✓ paper 故事:"self-labeling 对 small-target tasks 通过 multi-view + late-frame fix 可救",是 positive contribution
- ✓ Stage B place main 可以跑

如果 not work(taskB acc < 0.65):
- 加 image augmentation(C),如果还不够考虑 manual labeling
- Paper 写 limitation:"self-labeling 对 visual-novel target geometry 有 transfer 上限"

---

## 9. References

### 9.1 这次的 diagnostic artifacts

- `/r-preference/eval/viz_place_diagnosis/`:
  - `placement_grid.png`(2行×9列 比对 taskA + taskB placement 时刻图)
  - `placement_taskB_sequence.png`(stand placement 6 帧序列)
  - `target_geometry_comparison.png`
  - `rgb_stats.json`
  - `run_diagnosis.log`
- `/r-preference/eval/diagnostic_place_25k.json`(per-sample logit + confusion + unconditional bias)
- `/r-preference/eval/frame_window_place_25k.json`(6 strategy × 100 ep taskB acc)

### 9.2 Code changes(this session)

- `examples/preference/dataset/vqa_sample.py` — multi-cam cache + late_8 strategy + active_wrist sentinel
- `examples/preference/dataset/pref_hdf5_vqa_dataset.py` — per-cat config dispatch
- `starVLA/model/framework/QwenPI_VQA.py` — cache_row_to_pil call replaces hardcoded 4D
- `examples/preference/train_files/starvla_pref_stage_a_vqa_place.yaml` — vqa_cameras override + late_8 strategy

### 9.3 跨 doc 引用

- 0522: original Stage A spec (contact data + pipeline)
- 0523-design: 3 new cat design + Stage A run (height/hvlv/orient)
- 0523-stageA-analysis: gate eval methodology + clip strategy comparison
- 0524-place-category: place cat registration + data
- 0524-stageB-contact-plan: Stage B 框架
- 0525-problems-and-how-to-solve: 4 cat RED postmortem(§3.2-3.4 已被 0526 推翻)
- 0526-corrections: 推翻 0525 + pipeline audit
- training-runbook: ops manual(ckpt 状态 + 故障树)

---

End of doc.
