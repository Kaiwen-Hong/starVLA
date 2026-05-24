# Preference-conditioned VLA — Stage B (main method) — contact

> Self-contained 实现 doc(Stage B for contact category)。延续 baseline doc 与 VQA delta doc 的 house style。
>
> **Background**:Stage A 已验证(`r-preference/doc/0523-stageA-analysis.md`):
> 用 mid_8 clip strategy 在 35k VQA ckpt 上,contact taskB pseudo-label acc = **0.99**(uniform_8 是 fluke 0.61);baseline ckpt 是 chance(0.49-0.51);signal gap GT_low − GT_high = **+41.97**;VQA cotrain load-bearing。
>
> **本 doc 写 Stage B**:拿那个 frozen VQA-Stage-A 当 pseudo-labeler、warm-start 一份**另一个 copy** 在 unlabeled group B 上续训 action policy。
>
> **Iteration history (with Claude Code, 2026-05-24)**:
> §1-§3 收纳了对原稿的 11 处校正(frame 一致性 framing 改写、100 vs 50 demos 数确认、TaskB dir-name GT 防护、`put_boxdrink3_plate` 加进 contact 而非新 category、confidence 改 logit_gap、step 都用 35k validated ckpt)。
> §6 阈值跟 filter 数字来自实跑(`r-preference/eval/diagnostic_contact_35k_{mid8,gripper}.json`,2026-05-24)。

---

## 0. TL;DR

- **What**:(1) 用**冻结的 VQA-Stage-A 35k ckpt** 对 `contact/taskB/put_boxdrink3_plate_{25,75}` 共 100 条 episodes **离线 pseudo-label**(mid_8 + |logit_gap|≥1.0 filter,一次性、缓存);(2) 从 **同一个 VQA-Stage-A 35k ckpt warm-start 另一份 copy**,在这 100 条上**续训 action policy**(framework = plain `Qwen_PI`,**无 VQA loss**),conditioning 在 pseudo-pref 上;(3) 并行跑 **B0 baseline**(no-VQA-Stage-A 35k 续训 + base-only SFT,no pref suffix)。
- **LOCKED 决策**(per Stage A 实跑数):
  - labeler 与 trainable policy **两份独立 instance**,labeler 完全冻结
  - frame = **mid_8**(`fracs=(0.25,0.32,0.40,0.45,0.50,0.55,0.62,0.70)`,validated for taskB 0.99 on 35k);**绝不用 uniform_8**(0.61 是 fluke,且会 silently 漂掉 GT_high)
  - filter = **mid_8 单 strategy + |logit_gap| ≥ 10.0**(Q-A 实测 gap 分布是 bimodal:wrong 在 0.75,correct cluster 在 |gap| 20-49,中间 1-20 是空带;threshold 放在空带中心 ~10 → keep 98/100 with acc=1.000、balance 48/50,且对噪声 margin ~10 而非 0.25);unanimous(mid+gripper) 留作 future-cat fallback
  - 起手 ckpt = **35k**(step-matched,validated by Stage A gate eval)
  - Stage B **不喂 VQA loss**(label 已缓存),framework = `Qwen_PI`(non-VQA)
- **Deferred (§7)**:counterfactual MSE / sign-acc 的 offline proxy eval + ablation 第三条线(VQA-A 起手 + base-only)。之后用 **closed-loop rollout** 做。
- **Run**:数据小(100 demo),轻量。max_train_steps ≈ 800(~7 epoch,eff batch 64)。

---

## 1. 定位 / 与 Stage A 的关系

### 1.1 两份独立 instance(防漏 label-train coupling)

- **Frozen labeler**:VQA-Stage-A 35k ckpt,**完全冻结**,只在 §3 离线跑一次 `predict_preference` 出 pseudo-label。**不参与训练。**
- **Trainable policy**:VQA-Stage-A 35k ckpt 的**另一份 copy**,warm-start 后在 B 上续训(§4)。backbone 会更新、LM head 在无 LM loss 下不直接收 gradient 但 input 漂(backbone 变了)→ 行为隐式漂。**无所谓**,因为我们这次性 cache 完 label 之后就再也不用 LM head 了。
- ⚠ **绝不在同一份上边训边 label**,否则 label 随训练变。
- v2 note(目前不实现):**真要做 self-improvement 迭代 pseudo-labeling**,正确做法是**始终用 frozen labeler(原始 35k VQA)跨轮标注**,而不是用漂过的 policy 去标 — 这样连 backbone 都不用 freeze。

### 1.2 三条对比线

| 条件 | Stage A 起手 ckpt | Stage B conditioning | Stage B framework | 本 doc |
|---|---|---|---|---|
| **main** | VQA-Stage-A 35k | pseudo-pref(`Preference: low contact`/`high contact`) | `Qwen_PI`(no VQA loss) | **做** |
| **B0 baseline** | no-VQA-Stage-A 35k | base-only(**无 `Preference:` 后缀**) | `Qwen_PI`(no VQA loss) | **做**(§5) |
| ablation | VQA-Stage-A 35k | base-only(无 pref) | `Qwen_PI`(no VQA loss) | deferred(§7) |

### 1.3 契约(复用 Stage A,不重写)

- **照抄 Stage A**:action 数据 pipeline、prompt 函数(`build_action_prompt` / clean template path)、norm(`stats_contact_v1.json` = byte-identical `stats_giveobj_v1.json`)、架构(`Qwen_PI` / `LayerwiseFM` DiT)、`L_action`(flow-matching velocity MSE)。
- **新增**:
  - §3 离线 pseudo-labeling(脚本 + JSON cache)
  - §4 warm-start 续训到 B(YAML + launch sh)
  - 一处 `prompt.py` 改动:在 `PREF_CATEGORIES["contact"]` 加 `put_boxdrink3_plate` task_group + clean_template

### 1.4 Frame strategy 一致性(校正原稿)

**正确说法**:**Stage A inference 和 Stage B labeling 一致(都 mid_8)即可;训练 cache 是 uniform_8 是 history、不回去改**。train-eval mismatch 在这里是 feature 而且是 validated 的(mid_8 0.99 / gripper 0.97 两个 grasp-centered 都高,uniform 0.61 反而是 matching-train 的那个)— 所以不是 fluke,可靠。

**Future-work (paper v2, 非 blocker)**:用 grasp-centered frame **重训 VQA**,消除 "mismatch is load-bearing" 这个略 fragile 的依赖。**现在不为这个回炉 Stage A**。

### 1.5 Self-labeling 的 calibrated uncertainty(why this works at all)

Q-A 揭示了 frame selection 的**更深一层价值**,值得作为 paper 主线 framing:

| | uniform_8 wrong (39/100 on taskB) | mid_8 wrong (1/100 on taskB) |
|---|---|---|
| 失败模式 | **confident-wrong**(`|gap|` ~20-25,跟 correct 同样自信) | **honest-borderline**(`|gap|=0.75`,接近 coin flip) |
| Filter 是否有效 | ✗ 无效(gap 高、 conf~1.0,filter 抓不到)| ✓ 有效(gap 低,filter 直接 drop) |
| Model 的 uncertainty 状态 | 自信地走错(shortcut + cache memorize)| "知道自己不知道" → calibrated |

**核心 framing**:`frame selection` 不只是提了 transfer accuracy,**它让 pseudo-labeler 的 uncertainty 变得 calibrated**:grasp-centered frame 让模型在错的时候 gap 自动 collapse 到 0 附近(因为关键视觉信息 + 二分类任务下,如果看不清就两边相当),错的 → gap 低 → filterable → 不会教反 policy。

→ **这才是 self-labeling 不会教反 policy 的根本原因**,不是单纯 "accuracy 更高"。**正是 Tweak 1(threshold = 10 = bimodal 空带中心)能成立的根本原因** — calibration 把 wrong 推到一边、correct 推到另一边、留出空带。

---

## 2. 数据(group B = `contact/taskB`)

### 2.1 实际 disk layout
```
/mnt/localssd/kaiwenh/pref/data/contact/taskB/
├── put_boxdrink3_plate_25/data/episode{0..49}.hdf5   # 50 ep, GT_low
└── put_boxdrink3_plate_75/data/episode{0..49}.hdf5   # 50 ep, GT_high
```
- **总计 100 episodes**(50/50 perfectly balanced)。原稿 "50 demos" 数错,以 disk 为准。
- **HDF5 schema**:与 taskA 一致(已 verified by Stage A eval pipeline 读 taskB 100/100 无报错)。`endpose/{left,right}_endpose`, `endpose/{left,right}_gripper`, `observation/head_camera/rgb` 都在。
- **instructions/**:**空**(0/50,training-runbook §11 提过 taskB 没 mount instructions)→ paraphrase 路径 N/A,只走 clean_template path。
- **Norm**:**复用 `stats_contact_v1.json`**(= `stats_giveobj_v1.json` byte-identical)。绝不在 100 条上重算 stats(样本太少 + 必须和 Stage A 同套)。

### 2.2 base_prompt for B

`put_boxdrink3_plate` **不在** `PREF_CATEGORIES["contact"].task_groups`,直接调 `build_action_prompt` 会 raise KeyError。**改动**:

```python
# in examples/preference/dataset/prompt.py
PREF_CATEGORIES["contact"]:
    task_groups += ("put_boxdrink3_plate",)
    clean_templates["put_boxdrink3_plate"] = "Put the box drink onto the plate."
```

clean_template 措辞跟现有 `put_X_dustbin → "Put the X into the dustbin."` 风格一致;**无高度/位置词,无 pref leak**。

### 2.3 GT-leakage firewall(load-bearing)

`put_boxdrink3_plate_25` / `_75` 的 dir name **直接含 GT pref_key**。Stage B 流程必须**完全屏蔽**这个,否则 "unlabeled B" 的 claim 直接作废。

实现:
- 新建 `PrefHDF5StageBDataset`(独立 class,继承 `PrefHDF5Dataset` 或 thin wrapper):
  - `__getitem__` **永不**从 dir name 解析 pref_key
  - prompt 拼装 **只从 `pref_pseudo_labels_B.json` cache 查**(main)或**不拼 pref 后缀**(B0,`with_pref_suffix=False`)
- 加 **defensive assert**:dataset 初始化时检查 cache 文件存在(main)或 with_pref_suffix=False(B0);否则 raise。
- cache loader **白名单**:只读 `action_prompt_label` / `decision` 字段;碰到 `gt_*` 字段直接 ignore(GT 字段写在 cache 里只是为了 §3 launch gate 验证,**dataset 路径里禁止读**)。
- **B0 也走同一个 firewall**:`PrefHDF5StageBDataset(with_pref_suffix=False)` — 既不读 cache、也不从 dir name 解析。B0 作为 baseline 尤其不能漏 pref。

---

## 3. Pseudo-labeling(离线,一次性)

### 3.1 流程

1. **载入冻结的 VQA-Stage-A 35k ckpt** 到 `Qwen_PI_VQA` framework(只为了能调 `predict_preference`)。
2. 对每条 B episode 跑 **两次** `predict_preference(clip)`:
   - **clip = mid_8**(`vqa_sample.load_clip_by_strategy(strategy="mid_8")`,validated 0.99) ← **primary**
   - **clip = gripper_anchored**(同一个文件,`strategy="gripper_anchored"`,validated 0.98) ← **second opinion(可选 fallback)**
   - 用 §1.4 锁的 sampler(`DEFAULT_MID_FRACS` 已 hardcode 在 `vqa_sample.py`),**绝不用 uniform_8**。
3. 对每个 episode 算 `logit_gap = A_logit - B_logit`(A=`low`, B=`high`);predicted pref_key = argmax(A_logit, B_logit)。
4. **Filter decision**(per Q-A 实测,2026-05-24):
   - **Default (primary)**: keep iff **mid_8 |logit_gap| ≥ 10.0**
     - Q-A: keep 98/100, acc=1.000, balance 48/50 (96% / 100%, Δ=4%)
     - Q-A bimodal gap distribution on mid_8:
       - GT_low correct: |gap| 范围 +0.75 ~ +35.81(主簇在 +15 ~ +30,p10=+7,p25=+16)
       - GT_high correct: |gap| 范围 +6.25 ~ +31.06(主簇在 +15 ~ +25,p10=+15)
       - 唯一 wrong (`put_boxdrink3_plate_25/ep1`): |gap| = 0.75
     - 中间 (1, 20) 几乎是空带;**阈值 10.0 = 空带中心**,distance-to-wrong = 10 - 0.75 ≈ **9.25 margin**(对噪声、对未来 borderline-wrong @ |gap|=1.x 都鲁棒)。阈值 1.0 也能 drop ep1 但 margin 只有 0.25 → 贴着坏簇边,不稳。
     - drop 2 条全是 GT_low borderline-correct (gap 在 (1, 10) 之间);ep1 (gap=0.75) 是其中之一。
   - **Fallback (defensive)**: keep iff **mid_8 == gripper_anchored AND mid_8 |gap| ≥ 10.0**
     - 不推荐 contact taskB 用:Q-A 显示 gripper 在 contact 上反而引入 3 个错(vs mid_8 1 个),unanimous 净效应是 drop 多但不更准。
     - **future-cat 决策规则**:换 height/hvlv/orient 时,先跑 Q-A 看那个 cat 的 mid_8 **失败模式**:
       - 如果 wrong 在 |gap| < threshold(像 contact 这样 borderline-honest)→ **单 strategy gap-filter 够**
       - 如果 wrong 在 |gap| 高(confident-wrong,像 uniform_8 那 39 条)→ **才上 mid+gripper ensemble** 作 second opinion
       - 不要默认所有 cat 都用 default;**先看 wrong 的 |gap| 分布**
   - **不使用 confidence(`p_pair`)阈值**:饱和到 ~1.0 即使错,filter 无效。

5. **Output cache**:`r-preference/eval/pref_pseudo_labels_contact_B.json`,schema:
   ```json
   {
     "put_boxdrink3_plate_25/episode0": {
       "action_prompt_label": "low contact",   // ← dataset 只读这个 + decision
       "pref_key": "25",                        // mapped from VQA's "low"
       "decision": "keep",                      // or "reject:low_gap" / "reject:disagree"
       "mid_8":            {"pred": "25", "logit_gap": +24.13, "p_pair": 0.9999, "indices": [48,61,...]},
       "gripper_anchored": {"pred": "25", "logit_gap": +28.51, "p_pair": 0.9999, "indices": [30,39,...], "grasp_t": 62},
       "gt_pref_key": "25",                     // ← dataset 必须 ignore (firewall §2.3)
       "gt_match": true                          // for §3.2 launch-gate only
     },
     ...
   }
   ```
   `gt_pref_key` / `gt_match` 字段写出来仅供 **§3.2 launch-gate 验证 + 你之后做 §7 eval**,**dataset firewall 必须白名单掉**(§2.3)。

6. **Label 映射**(VQA 输出 → action prompt label):
   ```python
   VQA_PRED → ACTION_PROMPT_LABEL:
     "25" (VQA "low")  → "low contact"
     "75" (VQA "high") → "high contact"
   ```
   = `PREF_CATEGORIES["contact"].pref_labels`,已存在的 mapping,直接复用。

### 3.2 Launch gate(go/no-go before training)

Cache 生成后,**这是唯一合法用 GT 的地方**(验 labeler 性能,不是训练):
- 比较 `pseudo_label.pref_key` vs `dir_name.pref_key`
- 预期 ≥ 0.99(Q-A 实测 mid_8 = 0.99,filter 后 100%)
- **如果 < 0.95 → STOP**,不要进训练。
- 同时报 post-filter per-class balance(预期 48 low / 50 high in default filter)。

### 3.3 reject 处理

`decision != "keep"` 的 episode → **dataset 直接 skip(不 include 在 `_index` 里)**,不是只打 flag。100 条里预期 keep 98(default)或 keep 96(fallback)。

---

## 4. Policy 训练(continue from VQA-Stage-A 35k)

### 4.1 起手 ckpt + framework

- **warm-start ckpt**:VQA-Stage-A **35k**(step-matched with B0,validated by Stage A gate eval)
- **framework**:**plain `Qwen_PI`**(不是 `Qwen_PI_VQA`)— Stage B 无 VQA loss,不需要 VQA bookkeeping
- **load 方式**:`torch.load + load_state_dict(strict=False)` — backbone + action_head 权重对位载入(VQA-VQA-trained ckpt 跟 Qwen_PI 架构兼容,VQA-specific 字段如 `lambda_vqa` 不在 ckpt 里,也无需);**LM head 也会载入(它就是 backbone 一部分)**,但 Stage B 不用到。
- **optimizer state 不续**(新 stage、新 LR,重开 AdamW)。

### 4.2 Loss / prompt

- **`L = L_action` only**(flow-matching velocity MSE,同 Stage A baseline);no VQA loss。
- prompt = `clean_template["put_boxdrink3_plate"] + " Preference: " + <pseudo_label>`:
  ```
  "Put the box drink onto the plate. Preference: low contact"
  "Put the box drink onto the plate. Preference: high contact"
  ```
- pseudo_label per-episode(从 §3 cache 按 `<task_dir>/episode<N>` 查),episode 内所有 frame-sample 用同一个 label。
- dataloader:复用 Stage A 的 per-frame 索引;`PrefHDF5StageBDataset.__getitem__` 多查一次 cache 拼 prompt(§2.3 firewall)。
- 主仓库改动 ≈ 0:`dataloader/__init__.py` 加一条 `elif dataset_py == "pref_hdf5_stageb": ...` branch(复制 `pref_hdf5` 的 wiring,把 dataset class 换成 `PrefHDF5StageBDataset`)。

### 4.3 过拟合控制(100 条,认真做)

100 ep × ~150 frame/ep ≈ **15k samples**;eff batch 64 → **~234 step/epoch**;cap 1500 step ≈ **~6.4 epoch**。

| 项 | 值 | 备注 |
|---|---|---|
| **LR base** | **2.5e-6** | Stage A 的 1/10 |
| LR qwen_vl_interface | 1.0e-6 | Stage A 的 1/10 |
| LR action_model | 1.0e-5 | Stage A 的 1/10 |
| lr_scheduler_type | `cosine_with_min_lr` | 同 Stage A |
| min_lr | 1.0e-7 | 比 base LR 再低一档 |
| warmup_ratio | 0.1 | 同 Stage A;~150 step warmup |
| **max_train_steps** | **1500** | **~6.4 epoch**;给 warm-start 适配新 object 足够 budget,deferred eval 时挑 sweet spot |
| **save_interval** | **250** | **6 ckpts (250/500/750/1000/1250/1500)**;main 和 B0 同 grid |
| eval_interval | 2000(> max_train_steps,= 不做 step-eval) | val split 不分(§4.4) |
| logging_frequency | 50 | 短 run,更密 |
| gradient_accumulation_steps | 1(8 GPU)or 2(4 GPU) | eff batch 64 |
| eff batch | 64 | 同 Stage A |
| max_grad_norm / weight_decay / optimizer / betas | 同 Stage A | 不动 |
| enable_gradient_checkpointing | true(同 Stage A baseline)| 80GB H100 + Qwen3-VL-4B 一定要 |

**Main 和 B0 用完全相同 hyperparam**(只差起手 ckpt 跟 pref suffix flag),保证对比干净。

**Caveat — 不要把 under-training 误读成方法失败**:1500 step 留 6 ckpts grid 是为了 deferred eval 挑 sweet spot。如果 §7.5 sanity(pref-flip Δz)在 final 1500 step ckpt 上**弱或 sign-acc 不过**,第一反应应该是:
1. 先看更早的 ckpt(250/500/750)— 万一在 750 就够、1500 已过 — 或者
2. 把 cap 加到 2500(11 epoch),不是判 Stage B 方法失败
3. 同时检查 §3 cache 的 acc 是否真 ~1.000(launch gate)
4. 检查 §4.2 prompt 是不是真 pref-conditioned(打 stdout 看)

只有上述都排除后,才能怀疑方法本身。

### 4.4 不分 val(per agreement)

100 ep × 0.1 = 10 val ep,per-pref 5 ep,binomial CI ±31% — **太吵,不分**。
- 防过拟合靠 **max_train_steps=800 cap**
- sanity 用 **训练 episode 上的 pref-flip 检验**(§7 末)— 不需要泛化 sample,只检"翻 pref → action 方向变了吗"
- closed-loop eval(deferred)用 **fresh rollout**(不消耗 demo)

### 4.5 Per-class 不平衡 fallback(预期不触发)

Q-A default filter 后 balance = 48/50(Δ=4%)。**< 15% trigger**,不需要 fallback。

如果未来 cat / threshold 调整后 Δ ≥ 15%(主动停下来 confirm):
- 轻度(15-20%)→ γ:接受 + sanity 双类都看
- 中重度(≥20%)→ α:per-class balanced sampling(oversample 少数类)
- **避免 β**(loss weight)— 会跟 flow-matching loss scale 纠缠,且会放大少数类残余 noise 权重

---

## 5. Baseline B0 / Phase 1b(并行做)

- **起手 ckpt**:**no-VQA-Stage-A 35k**(`pref_baseline_stage_a_v1_noVQA_contact/checkpoints/steps_35000_pytorch_model.pt`,disk 上只剩这一个 step)。
- **Stage B = base-only SFT on B**:prompt **不带 `" Preference: "` 后缀**(只 base);`L_action` only。
- **过拟合控制与 §4.3 完全一致**(LR/steps/warmup/eff batch/save_interval/...)— 只差:
  - 起手 ckpt(no-VQA vs VQA)
  - dataset `with_pref_suffix=False`(不读 cache 也不拼 pref suffix)
- **Firewall 也适用 B0**(§2.3):走同一个 `PrefHDF5StageBDataset(with_pref_suffix=False)`,既不读 cache 也不解析 dir name pref_key — B0 作为 baseline 尤其不能漏 pref(否则 "VQA conditioning 的贡献" headline 不成立)。
- 用途:§7 headline 对比的对照组。

---

## 6. 超参 delta 速查

相对 Stage A(`starvla_pref_stage_a_baseline_contact.yaml`):

| 项 | Stage A | Stage B (main + B0) |
|---|---|---|
| LR base | 2.5e-5 | **2.5e-6** (↓10×) |
| LR qwen_vl_interface | 1.0e-5 | **1.0e-6** (↓10×) |
| LR action_model | 1.0e-4 | **1.0e-5** (↓10×) |
| max_train_steps | 50000 | **1500** |
| num_warmup_steps | 5000 | **150** (warmup_ratio 0.1 × 1500) |
| save_interval | 5000 | **250** (= 6 ckpts grid) |
| eval_interval | 1000 | **2000**(> max → skipped) |
| min_lr | 1e-6 | **1e-7** (↓10× to match base ratio) |

其他(optimizer / scheduler 类型 / 相机 / image_size / chunk / norm / DeepSpeed Zero2 / `expandable_segments`)**全同 Stage A**。

main 和 B0 共用同一份 hyperparam,只差起手 ckpt 与 `with_pref_suffix` flag。

---

## 7. Eval + Ablation  ——  [OPTIONAL / 先不做]

> Deferred:等 closed-loop rollout pipeline 就绪后用 **B 的 policy success rate + grasp position** 来做。

### 7.1 Deferred 主 eval(closed-loop)

- 在 B 任务上 rollout(新初始条件):
  - **task success rate**:完成任务比例
  - **preference-following**:给 `Preference: low` / `high` 各 rollout,量**实际 grasp 接触高度**(object/task-relative),看是否落在正确一侧
- headline:**main 在 unlabeled B 上闭环跟随 preference,B0 不跟随**

### 7.2 Deferred offline proxy(可选,更便宜)

- counterfactual MSE(`‖a_low − a_high‖`,同一 obs 翻 pref)+ sign-acc,在 held-out B 上。main vs B0。

### 7.3 Deferred ablation

- **VQA-Stage-A 起手 + base-only SFT on B**(不喂 pseudo-pref)。
- 与 main、B0 三点把贡献夹出来:
  - **main − ablation** = pseudo-label conditioning 的功劳
  - **ablation − B0** = Stage-A-VQA 塑形的残余功劳

### 7.4 Future work — paper v2 (非 blocker)

- 用 **grasp-centered frame 重训 VQA**(stochastic / mid_8-baked cache),消除当前 train-eval mismatch 的隐性 fragility。
- 当前依赖是 **validated**(0.99 / 0.98 两个 grasp-centered strategy 都高)所以不 fragile in practice,但 paper 写起来更干净。

### 7.5 最低 sanity (在 §8 之前必做)

> ⚠ Full eval deferred,但 launch 前**最低限度 sanity**:训完抽 **10 条 B(5 GT_low + 5 GT_high)**,对每条:
> - 取 **靠近 grasp 的 frame**(`grasp_t` 或 mid-episode,**不是 first frame** — 开头机械臂还远,pref 没在 action 里表达,Δz≈0 会假性 fail)
> - 给 `Preference: low contact` 和 `Preference: high contact` 各跑一次 `predict_action`,拿 chunk (16, 20)
> - 报 **sign-acc**:Δz = `action_high[k, 2] - action_low[k, 2]` 在 chunk-max(or grasp-relevant step)的符号是否 > 0(站立 boxdrink3 → 高 contact = 高 z)
> - 报 **mean Δz**(normalized space)
> - **pass 标准**:**sign-acc ≥ 8/10 且 mean Δz 明显 > 0**

独立 script `examples/preference/stage_b/sanity_pref_flip.py`,可在任意 ckpt(包括中间 200/400/600/800)复用。

---

## 8. Run checklist

### 准备(一次性)
1. ☐ 改 `prompt.py`:`PREF_CATEGORIES["contact"]` 加 `put_boxdrink3_plate` + clean_template(§2.2)
2. ☐ 新建 `PrefHDF5StageBDataset` + `dataloader/__init__.py` 加 `pref_hdf5_stageb` branch(§2.3, §4.2)
3. ☐ 新建 `pseudo_label_offline.py`(§3 离线 labeler)
4. ☐ 新建 `train_files/starvla_pref_stage_b_{main,b0}_contact.yaml`(§6 超参)
5. ☐ 新建 `launch_pref_stage_b_{main,b0}_contact.sh`(§4.1 ckpt path)
6. ☐ 新建 `sanity_pref_flip.py`(§7.5)

### Pseudo-labeling + launch gate
7. ☐ 跑 `pseudo_label_offline.py` 用 mid_8 + |gap|≥1.0 filter:
   - 产出 `r-preference/eval/pref_pseudo_labels_contact_B.json`
   - 预期 keep 98/100,balance 48/50,vs dir-GT acc 1.000
   - 同时跑 gripper_anchored,写进 cache 作为 second opinion(future fallback 用)
8. ☐ Launch gate:print cache stats,vs dir-GT acc < 0.95 → STOP(§3.2)

### 训练
9. ☐ launch main:`bash launch_pref_stage_b_main_contact.sh`(单机 4-8 GPU,~15-30 min for 800 step)
10. ☐ launch B0 :同一台机器,**等 main 跑完再启动**(GPU full)(同上时间)
11. ☐ 最低 sanity(§7.5):main 训完跑 sanity_pref_flip on 10 ep,pass → go;fail → diagnose

### Ckpt + tracking
12. ☐ 存 ckpt:
    - main: `results/Checkpoints/pref_main_stage_b_v1_contact/`(800 step,4 ckpt)
    - b0:   `results/Checkpoints/pref_b0_stage_b_v1_contact/`(同 grid)
13. ☐ wandb:project `pref-sim`,run_id `pref_main_stage_b_v1_contact` / `pref_b0_stage_b_v1_contact`

### Deferred
14. ☐ [deferred] closed-loop eval + offline proxy + ablation 3 线(§7)

---

## 9. 已知风险 / 注意

### 数据 / labeling 风险
- **pseudo-label 错的会教反**:错的 (base + 错 pref → 真 pref 的 action) 会反向教 conditioning。§3 filter (|gap|≥1.0) 把 mid_8 的 1 wrong (ep1) 直接 drop;Q-A 实测 post-filter acc = 1.000,**预期 0 错误 label**。
- **GT-leakage firewall**:dir name 含 `_25`/`_75`。`PrefHDF5StageBDataset` 必须**白名单**只读 cache 的 `action_prompt_label` / `decision` 字段,**defensive assert** 不允许 dir-pref 解析(§2.3)。B0 也走同一个 firewall。
- **low/high split 当前 perfect 50/50**;post-filter 48/50(Δ=4%,< 15% trigger);不需要 per-class fallback。

### Frame strategy 风险
- **frame 一致性**(校正原稿):**Stage A inference 和 Stage B labeling 用同一份 mid_8 sampler**(`vqa_sample.load_clip_by_strategy("mid_8")`,validated DEFAULT_MID_FRACS hardcoded)。训练 cache 是 uniform_8 是 history,不动。**绝不让 Stage B 用 uniform_8**(acc 直接退化 0.61,且 GT_high 被 silently 砍掉,而且是 confident-wrong → filter 无效 → 直接教反 policy)。
- **mid_8 fractions 锁死**:DEFAULT_MID_FRACS = (0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70),hand-tuned 跑出 0.99。不要在 Stage B 调它。如果未来要换 strategy,**重跑 Q-A 验**。
- **Future cat 不要默认用 contact 的 default filter**:换 height/hvlv/orient 时,先跑 Q-A diagnostic 看那个 cat 的 mid_8 **失败模式 + |gap| 分布**:
  - 如果 wrong 在 |gap| 低(borderline-honest,calibrated like contact)→ 单 strategy gap-filter 就够,threshold 放空带中心
  - 如果 wrong 在 |gap| 高(confident-wrong,uncalibrated like uniform_8 of contact)→ 必须上 mid+gripper ensemble 作 second opinion;或者更激进,**换 phase-targeted frame**(e.g., height 信号在 release moment → `late_8`;hvlv 信号在 trajectory → `dense_16`)
  - **决策 anchor**:看 wrong 的 |gap| 分布相对 correct cluster 是 separable 还是 overlapping。separable → filter 够;overlapping → 需要 ensemble 或换 frame。

### Pseudo-labeler 可靠性是 task-family-dependent(paper limitation)
- **contact taskB (`put_boxdrink3_plate`) 属于 taskA 强信号族**(站立 boxdrink/callbell 类,taskA val 1.000)→ 0.99 是该族的预期 transfer 表现,不是 outlier。
- **弱信号族**(taskA 实测 fork/screwdriver, 平躺物体,acc 0.50-0.75)→ pseudo-labeler 在这类 taskB 上**预期会失败**;当前 contact taskB 不属于这族,所以 OK。
- **paper 写 limitation 必须诚实说**:Stage B pseudo-labeling 的可靠性跟 task-family 视觉信号强度耦合;非强族需要 fallback strategy(ensemble、phase-targeted frame、或 human-in-loop 兜底)。
- 当前不阻塞 contact Stage B;但**未来 cat 的 taskB 若落在弱族,必须先跑 Q-A 看 |gap| 分布**(per §9 frame 决策规则),不能盲套 contact 的 default。

### 模型 / 训练风险
- **backbone 漂导致 LM head 隐式漂**:Stage B 用 plain Qwen_PI 无 LM loss,但 backbone 更新 → LM head input 变 → output 隐式变。**对 frozen labeler 无影响**(独立 ckpt);只是**不能用训完的 policy 重 label**(LM head 已不可信)。**当前不打算迭代,所以无所谓**。
- **norm 复用**:B 用 A 的 stats_contact_v1.json;同 robot/同 workspace,预期没问题。保险起见:`pseudo_label_offline.py` 末尾画一下 B 的 normalized action histogram,确认没贴 clip 边界(若贴,标出来,但不动 stats — 跟 Stage A 同套是 ablation 干净的硬约束)。
- **VQA 已 0.99 还要 Stage B 的理由**:VQA head 是 zero-shot perception(读 grasp 几何);action head 的 visuomotor **没在 B 新 object 上训过**,Stage B 负责适配 action + 装 conditioning。

### B0 风险
- **B0 的 base prompt 必须无 pref leak**:B 的 clean_template "Put the box drink onto the plate." 已无高度/位置词(§2.2)。如果换 prompt 措辞,**再 verify 一遍 `_LEAK_RE_GIVEOBJ` 不命中**。

---

## 10. 文件 layout(规划,未实现)

```
examples/preference/
├── dataset/
│   ├── pref_hdf5_stageb_dataset.py            # NEW: GT-firewall dataset (§2.3)
│   ├── prompt.py                              # MODIFIED: add put_boxdrink3_plate (§2.2)
│   └── vqa_sample.py                          # already has mid_clip_indices / gripper_anchored (§1.4 lock)
├── stage_b/
│   ├── __init__.py
│   ├── pseudo_label_offline.py                # NEW: §3 labeler
│   └── sanity_pref_flip.py                    # NEW: §7.5
├── train_files/
│   ├── starvla_pref_stage_b_main_contact.yaml # NEW: §6
│   └── starvla_pref_stage_b_b0_contact.yaml   # NEW: §6
└── launch_pref_stage_b_main_contact.sh        # NEW
└── launch_pref_stage_b_b0_contact.sh          # NEW

starVLA/dataloader/__init__.py                 # MODIFIED: add pref_hdf5_stageb branch

r-preference/eval/pref_pseudo_labels_contact_B.json   # NEW: §3 cache output
```

---

## 11. References

- Stage A spec: [`0522-a-giveobj.md`](0522-a-giveobj.md)
- Stage A gate eval methodology + extension guide: [`0523-stageA-analysis.md`](0523-stageA-analysis.md)
- Contact gate result (this is the run we're warm-starting from): [`../eval/stage_a_gate_contact_35k_analysis.md`](../eval/stage_a_gate_contact_35k_analysis.md)
- Q-A diagnostic results (filter threshold sourcing): `r-preference/eval/diagnostic_contact_35k_mid8.json` + `_gripper.json`
- Ops manual: [`training-runbook.md`](training-runbook.md)
- 3-new-cat design (for height/hvlv/orient Stage B later): [`0523-height-hv-oreint-design-doc.md`](0523-height-hv-oreint-design-doc.md)

---

End of Stage B doc — contact category。

Next step (after your review):
1. Patch `prompt.py` (§2.2 add put_boxdrink3_plate)
2. Write `PrefHDF5StageBDataset` (§2.3 firewall)
3. Write `pseudo_label_offline.py` (§3)
4. Run §3 + verify launch gate (§3.2)
5. ☐ Await your confirm before writing YAML / launch / training
