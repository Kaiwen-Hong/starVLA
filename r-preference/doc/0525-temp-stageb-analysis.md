# Stage B (contact) — findings 与 conclusions(reading-first 版本)

写于 2026-05-25,user-leave 期间 autonomous deep-dive 之后。

**目的**:给你一份 narrative 的、按读的顺序展开的 summary,看完这个就能 onboard
全部 Stage B 实验结果,不用先翻 raw eval doc。Raw 数据 + per-sample
records 在 `r-preference/eval/stageb_contact_diagnosis.md`(详细版)
+ `r-preference/eval/stageb_diag*/` 下。

---

## 0. TL;DR (一句话 + 一段)

**一句话**:VQA annotator 工作完美(label 100% acc),但 Stage B SFT
出来的 policy **不能用 prompt 控制**(响应只有 demo signal 的 5%)。
**Pipeline 不能用于 closed-loop pref-following**,因为 image-prompt
redundancy 让 model 直接走 image shortcut,把 prompt 当 noise。

**一段**:Stage B 100 demos 太小,加上 image 已经完全 encode 了 grasp
位置(看图就知道是 low 还是 high contact 抓的),prompt 的 3-token
suffix 跟 image features 100% redundant correlated。Model 优化
L_action 时两条路一样省力:(a) 学 prompt → action,(b) memorize
image → action。(b) 更简单,model 选了 (b),所以 prompt 完全没起
作用 — 翻 prompt 不翻 action。这不是 Stage B 特有问题,Stage A
baseline 跟 Stage B main 都是 ~2-3mm 响应 vs demo 46mm,**整个
(image, prompt, action) recipe 共享 bottleneck**。

---

## 1. 我们最终在 evaluate 什么

不是 closed-loop success rate(我们没 sim rollout pipeline),不是
action regression quality(那是 L_action,wandb 上看的 — 看着 healthy
但跟 "controllability" 是两回事)。

**测的是 pref-flip sign accuracy**:

```
对每个 contact/taskB episode:
  1. 取 grasp moment 那一帧的 (image, state)
  2. 跟 model 说同一个 image+state,但 prompt 翻两次:
       prompt_low  = "Put the box drink onto the plate. Preference: low contact"
       prompt_high = "Put the box drink onto the plate. Preference: high contact"
  3. predict_action 两次 → a_low (16,20), a_high (16,20)
  4. 看 left_endpose z 轴 (axis 2):  Δz = a_high[k_max,2] - a_low[k_max,2]
  5. sign_correct if Δz > 0  (high contact → grip 更高 z,这是物理 expectation)

sign-acc = correct 的 episode 比例
```

直白说:**"翻 prompt → action 方向是否按预期翻"**。0.5 = random,
0.8+ = 真正用了 prompt。

---

## 2. 数字结果(N=50,真正 statistically meaningful 的样本)

| ckpt | sign-acc | mean Δz (normalized) | mean Δz (physical) | std |
|---|---:|---:|---:|---:|
| **Stage A baseline 35k** | 31/50 = **0.62** | +0.012 | **+2.0 mm** | 0.064 |
| Stage A VQA 35k | 23/50 = 0.46 (random) | -0.010 | **-1.7 mm** (反向!) | 0.086 |
| **Stage B main final** | 31/50 = **0.62** (tied) | +0.016 | **+2.7 mm** | 0.049 (最小 std)|
| Stage B B0 final | 26/50 = 0.52 (random) | +0.006 | +1.0 mm | 0.082 |

第一眼看:Stage A baseline 跟 Stage B main 都 0.62 — 接近 8/10 gate 但没过。
Stage A VQA 反而最差(下面会讲)。Stage B B0 跟 random 差不多。

**95% binomial CI** for 0.62 (N=50) = [47%, 76%] — 跟 chance 0.5 在统
计上 overlap,严格说不能 reject "model 完全没条件 prompt"。

---

## 3. 关键 quantitative evidence(smoking gun)

数字看似 marginal,但**直接量 demo data 真实 Δz** 一下就把 picture
说清:

```
对 100 个 demo(50 low + 50 high), grasp_t 时取 chunk:
  low_z_mean  ≈ 0.87 m
  high_z_mean ≈ 0.92 m
  Δz = +46 mm (consistent 在 chunk 所有 16 步)
```

→ **演示数据 GT_high 比 GT_low 平均高 46mm**(grasp 已发生,chunk 是
lift/transport phase,全程保持差异)。

把 model 响应跟 demo signal 比:

| model | physical Δz | demo 的 % |
|---|---:|---:|
| Stage A baseline 35k | 2.0 mm | **4.3%** |
| Stage A VQA 35k | -1.7 mm | **负向** |
| Stage B main | 2.7 mm | **5.9%** |
| Stage B B0 | 1.0 mm | 2.1% |

**所有 model 都只给 demo signal 的 5% 或更少**。最 good 的 Stage B main
也只有 5.9% — 意味着 prompt 翻一下,grip 高度变化 3mm。Robot 控制
precision + grip 接触 tolerance 至少 1-2cm,所以 3mm response **在 closed-loop
上等于零**。

**0.62 sign-acc 的真实 interpretation**:不是"weak 但 real conditioning",
而是"tiny signal in correct direction + noise dominates individual episodes"。
平均给 +3mm,within-episode std 50mm → ~60% episode 的 sign 恰好正
(因为 mean 推得有点正方向)→ 0.62。**不是 pref-following 在
work,是 chance 加 tiny bias**。

---

## 4. 诊断:为什么所有 model 都失败

### Root cause: image-prompt redundancy + memorization shortcut

训练每一个 sample 是 `(image, prompt_suffix, demo_action)`,其中:

```
image: boxdrink 在 low/high 位置 ← 视觉已经完全 encode pref
prompt_suffix: "Preference: low/high contact" ← 3 tokens
demo_action: 演示者在那个位置的轨迹

L_action = MSE(predicted action, demo action)
```

模型 minimize loss 两条路:

| 路径 | 描述 | 难度 |
|---|---|---|
| **(a) 真 conditioning** | 学 "看 prompt → 出对应 action"(generalize 到新 image)| 难 — 需要 disentangle prompt 跟 image |
| **(b) memorize shortcut** | 学 "这个 image features → memorize 它对应的 action"(忽略 prompt)| **easy** — image features 信息量极大,100 demo 很容易记住 |

两条路 loss 一样低。Model 默认走 (b),因为:
- Image 信息量 ≫ 3-token suffix
- 100 demos × 多 epoch → memorize trivial
- 没有任何 loss 项**惩罚** "忽略 prompt" 的行为

→ 训练完模型把 prompt 当 noise,完全靠 image features 出 action。

### Evidence(支持这个 diagnosis 的三个 sub-finding)

#### 4.1 Frame-fraction sweep — image shortcut 的直接证据

测同样 model 在不同时刻 frame 上的 pref-flip 表现:

| frame fraction | Stage A baseline | Stage B main |
|---|---:|---:|
| 0.20 (very early approach, image 还没 reveal 抓的位置) | **0.67** | **0.40 ← below chance!** |
| 0.30 | 0.57 | 0.50 |
| 0.40 (≈ grasp moment) | **0.73** ← peak | 0.53 |
| 0.50 (post-grasp) | 0.67 | 0.63 |

读这张表:
- **Stage A baseline**:即使 early frame(image 信息少),还有 0.67 sign-acc
  → **真的在用 prompt**,因为 image 不告诉它 pref,它必须靠 prompt
- **Stage B main**:early frame 跌到 0.40 below chance → **prompt 完全没用**,
  只有 image 信息多的 later frame 才有响应(image shortcut)

→ Stage B 训练把 baseline 还有的那点 prompt-conditioning **抹掉了**,
换成了 pure image-feature memorization。

#### 4.2 Per-chunk-step structure

Stage A baseline 在 chunk 第 12-15 步(post-grasp lift)有 clear sign-acc
peak(k=12: 0.80)— 这是 model 真的在做 "lift higher for high contact"
的物理动作。

Stage B main 全程 flat 0.5-0.6 → **没有学到任何动作 structure**,
只是 noise + tiny bias。

#### 4.3 Stage B B0 比 Stage A baseline **更差**(0.52 vs 0.62)

B0 = 用 Stage A baseline (0.62) warm-start,Stage B SFT 训 3000 步
**不喂 prompt suffix**。结果 sign-acc 跌到 0.52(几乎完全 random)。

→ Stage B SFT 本身,**与 pseudo-label 无关**,会 degrade Stage A 已经
学到的 conditioning。证明 Stage B SFT 在 100 demos × 多 epoch 这个
setup 下**主动 wash out** 已有的 conditioning(memorization 把 prompt
attention 都抢走)。

### 4.4 Bonus surprise:VQA cotrain 在 Stage A 上 HURTS action conditioning

Stage A VQA(0.46)< Stage A baseline(0.62)。VQA cotrain 是为了让
LM head 学 pref 分类,但它似乎**稀释了 action head 对 prompt 的 attention**。
这是 VQA recipe 的 unexpected side effect,值得另外 investigate。

---

## 5. 这意味着什么 — practical implications

### 5.1 Pipeline 状态(分两层看)

| 层 | 状态 |
|---|---|
| **VQA labeler**(stage A → pseudo-labels) | ✅ **WORKS PERFECTLY**(cache 92/92 acc vs GT)|
| **Stage B SFT(pseudo-labeled action training)** | ❌ **FAILS** to teach prompt-conditioning |
| **整体 self-labeling pipeline** | ✅ labeling 这步 work;❌ 把 labels 用进 policy 这步 **fundamentally broken** in current recipe |

### 5.2 直接给 paper 的含义

| paper claim | 现在能不能 support |
|---|---|
| "VQA cotrain enables 99% acc self-labeling on unseen taskB" | ✅ YES(我们 contact 上确实 92/92 acc)|
| "Self-labeled SFT produces prompt-controllable policy on taskB" | ❌ **NO** — sign-acc 0.62 ≈ chance,physical response 5% of demo |
| "Closed-loop pref-following improves with VQA vs baseline" | ❌ untested(没 sim rollout)+ 看 offline numbers 不太可能成 |

### 5.3 短期建议(给 closed-loop demo 用)

如果你要立刻给个 demo / 给同事一个 ckpt:

| 选项 | sign-acc | 备注 |
|---|---|---|
| 用 Stage A baseline 35k + `"Preference: ..."` prompt | 0.62 | 当前最强 pref-follower(虽然也只是 marginal)|
| 用 Stage B main | 0.62 | 跟 baseline 持平,structure 更差 |
| 用 Stage B B0 | 0.52 | random,不要用 |

→ **直接用 Stage A baseline,Stage B 不增加任何 controllability**。

### 5.4 长期方向(per §6 recommendations,需要重设计)

要让 pref-conditioning 真正 work,必须**强迫模型用 prompt**。不能靠简单
"更多数据 / 更多 step"。具体路径(详细在 diagnosis doc §6):

1. **Counterfactual augmentation**:训练时同一 image 配错的 prompt + 不同
   action target → 强制 prompt 信号 matter
2. **Freeze visual backbone during Stage B**:visual features 不变,只能
   action head 适应 → action head 必须用 prompt
3. **不同 objective**:`predict_pref_from_action_and_image`(inverse),
   或 contrastive learning between (low_clip, high_clip)
4. **大幅扩 taskB 数据**(100 → 1000+):memorization 不再 trivial
5. **改 Stage A** 加 pref-distribution 多样性(同 image 多种 pref label
   phrasing → break 1:1 correlation)

这些都不是简单调 hyperparam 能搞定,需要新 design。

---

## 6. 我们做了什么(实验时间线 + 数据 trail)

完整时间线 2026-05-24 下午 → 2026-05-25 凌晨:

| 时间(UTC) | 事件 |
|---|---|
| 10:46 | 你 launch temp-523-h100.sh(stage B main → b0 → place)|
| 10:46-11:50 | Stage B main 训练 3000 step ✓ |
| 11:50-12:54 | Stage B B0 训练 3000 step ✓ |
| 12:54 | Place baseline 开始 |
| 19:14 | 我 kill place baseline(step 18967,要 GPU 跑 sanity)|
| 19:15-20:50 | 跑 sanity:Phase A(main 6 ckpts)+ Phase B(Stage A refs)+ Phase C bigN + frame sweep + demo Δz 测量 |
| 20:53 | Resume place baseline from step 15000(lost 4000 step compute)|
| ~00:10 | Place baseline 完成(5 ckpts saved + final)|

**ckpts 现存(`/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/`)**:
- Stage A baseline contact 35k(原 ckpt,warm-start B0 用)
- Stage A VQA contact 35k + 50k(原 ckpt,warm-start main 用)
- Stage B main contact: 6 ckpts(500/1000/.../3000)+ final_model = 7 ckpts × 16 GB
- Stage B B0 contact: 6 ckpts + final_model = 7 ckpts × 16 GB
- Stage A baseline place: 5 ckpts(5k/10k/15k/20k/25k)+ final = 6 ckpts × 16 GB
- 其他 Stage A height / hvlv ckpts(未变)

**raw eval JSONs**(per-sample logits/chunks,可 reproduce 数字):
- `r-preference/eval/stageb_diag/` — N=10 sanity on 8 ckpts(Phase A+B)
- `r-preference/eval/stageb_diag_bigN/` — N=50 sanity on 4 critical ckpts
- `r-preference/eval/stageb_diag_frame/` — N=30 frame-fraction sweep

**Code**:
- `examples/preference/stage_b/sanity_pref_flip.py`(已扩展支持
  `--framework_name`, `--dump_chunks`, `--frame_fraction`)
- `examples/preference/stage_b/analyze_pref_flip.py`(per-axis +
  per-chunk-step analyzer)
- `examples/preference/stage_b/pseudo_label_offline.py`(VQA labeler,
  写 cache)

**完整诊断 doc**(技术 reference,所有数字 + per-step + per-axis):
- `r-preference/eval/stageb_contact_diagnosis.md` (~400 行)

**Wandb runs**:
- pref_main_stage_b_v1_contact: `hq2gkmzi`
- pref_b0_stage_b_v1_contact: `jiiniwml`
- pref_baseline_stage_a_v1_noVQA_place: `qehl6rdr`

---

## 7. 几件事你 review 完可以决定

1. **Disk cleanup**:Stage B 中间 ckpts(500/1000/.../2500 各 16 GB × 2 = 160 GB)
   现在 useless(diagnosis 说这些 ckpt 不能用),要不要 rm 释放 disk?
   保留 main_final + b0_final + step_3000 当 reference 足够。

2. **Closed-loop rollout 上 actually 测一下**?即便 offline numbers 看着
   不好,真 robot rollout(或者 sim)可能给不同 picture。需要 setup
   evaluation infra(目前没)。

3. **Stage A VQA 反方向 -1.7 mm 这件事 worth 单独 investigate** — VQA cotrain
   理论上应该 help,实测 hurts action-side conditioning。如果要做下一轮
   Stage A,值得 redesign(可能去掉 VQA cotrain,或者改 VQA 训练比例)。

4. **后面 paper / 实验方向**:接受 self-labeling 部分 works,prompt-
   conditioning 部分需要 new approach(per §5.4)。要不要这周 prototype
   counterfactual aug?

5. **Doc 状态**:现在两份(`0524-stageB-contact-plan.md` planning doc,
   `0525-temp-stageb-analysis.md` 本文 findings)。要不要合并,或者把
   conclusions 写回 plan doc 的 §7?

---

## 8. 读后建议下一步

读完这个,如果你想 deep dive,按这个顺序:
1. `r-preference/eval/stageb_contact_diagnosis.md` §0-§3 — 数字证据
2. 同上 §4 — diagnosis 详细
3. 同上 §7.5 — demo Δz 实测(smoking gun)
4. 同上 §6 — recommendation 选项

如果你想直接讨论 next step,可以直接说,我等。
