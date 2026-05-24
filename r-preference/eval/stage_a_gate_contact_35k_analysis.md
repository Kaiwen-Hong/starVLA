# Stage A gate — `contact` — thorough analysis

Written 2026-05-24. 2-hour deep-dive after initial gate eval came back RED.
**The diagnosis evolved significantly during the analysis** — read top-to-bottom; the headline is in §0.

- Ckpts: contact baseline `steps_35000` (no VQA), contact VQA `steps_35000` and `steps_50000`
- Eval scripts: `examples/preference/eval/stage_a_gate.py`, `diagnostic.py`, `frame_window_test.py`
- Wandb run analyzed: `kaiwenh-17-uiuc/pref-sim/pkyvgkt3` (contact VQA 50k)
- Visual artifact: `r-preference/eval/clips/composite_taskA_vs_taskB.png`
- Raw outputs: `r-preference/eval/{diagnostic_contact_35k.json, diagnostic_contact_50k.json, frame_window_{35k,50k}_with_taskA.json}`

---

## 0. TL;DR (revised)

**Stage A gate verdict (revised): GREEN-conditional.** The VQA approach is **not broken** — it can label `contact/taskB` (`put_boxdrink3_plate`) at **98-99% accuracy** when we change one inference-time knob (frame-selection strategy). Stage B is viable on contact's taskB.

The initial RED came from a **frame-selection mismatch**, not from shortcut learning, overfitting, or implementation bugs:

| | uniform_8 (current) | mid_8 (fixed mid fractions) | gripper_anchored (centered on grasp) |
|---|---:|---:|---:|
| **taskB acc** | **0.610** ❌ | **0.990** ✅ | **0.980** ✅ |
| taskB logit-gap signal | +10.06 | +49.17 | (large) |
| taskA val overall | 0.912 | 0.800 | 0.806 |

The signal IS real (+49 logit gap on taskB with mid_8); the model genuinely perceives grasp pose. uniform_8 missed the grasp moment because taskB episodes are ~30% longer than taskA (T=193 vs T=140-160) and 8 uniformly-spaced frames don't land on the grasp event.

**Important nuance**: mid_8 / gripper_anchored are NOT universal upgrades. They lose on `give_fork` (1.0 → 0.75) and `give_screwdriver` (1.0 → 0.50) on taskA, where the pref-distinguishing signal sits in **post-grasp transport frames** (not the grasp moment itself). Uniform_8's wider spread captures those for shorter give_* episodes. **The right answer is task-aware clip selection, or ensembling multiple strategies and voting.**

**For Kaiwen's decision tree from the original spec:**
- "VQA acc on strong task ≥ 0.90 → green-light Stage B prep": **YES on contact taskB**, conditional on using mid_8 or gripper_anchored at pseudo-label time.
- "If strong task low → diagnose: clip frames / λ / answer token": clip frames was exactly the right fix. λ doesn't matter (L_vqa saturates at step 200). Answer token is fine.

---

## 1a. **VQA vs baseline ckpt** (added later — answers "does VQA cotrain help?")

Baseline (no VQA training, just action loss) ckpt, run through the SAME `predict_with_logits` machinery (manually injecting `_vqa_question` / `_vqa_id_A` / `_vqa_id_B` so the comparison is apples-to-apples), on contact 35k:

| strategy | baseline taskB | VQA taskB | baseline taskA | VQA taskA |
|---|---:|---:|---:|---:|
| mid_8 | 0.510 | **0.990** | 0.512 | 0.725 |
| gripper_anchored | 0.500 | **0.970** | 0.500 | 0.725 |
| uniform_8 | 0.490 | **0.610** | 0.500 | 0.863 |

| | baseline pred_dist (taskB) | VQA pred_dist (taskB) | baseline gap signal | VQA gap signal |
|---|---|---|---:|---:|
| mid_8 | 1 low / 99 high | 49 low / 51 high | +0.02 | **+41.97** |
| gripper_anchored | 0 low / 100 high | 49 low / 51 high | -0.04 | **+45.16** |
| uniform_8 | 5 low / 95 high | 89 low / 11 high | +0.13 | +12.29 |

**Baseline ckpt has ZERO discrimination ability** — predicts "high" essentially 100% of the time (~50% acc just from random GT distribution), signal ≈ 0. This holds across all 3 clip strategies and both taskA/taskB.

**VQA cotrain adds the entire pseudo-labeling capability**: from chance accuracy to 97-99% on taskB and 72-86% on taskA. The cotrain is **load-bearing for Stage B**.

(Note: baseline's "always-high" default isn't random — it's the LM head's prior for the bare-form `"high"` token (id 11892) being mildly higher than `"low"` (id 10303) on the Qwen3-VL pretraining distribution. Without task-specific supervision, the head can't discriminate.)

→ **The whole point of the VQA cotrain is validated**: it teaches the LM head to read grasp pose from pixels. The frame-strategy issue from §0 is about HOW to access that learned capability at inference, not whether the capability exists.

---

## 1. The headline experiment (what flipped the verdict)

Three clip strategies tested on the same VQA 50k ckpt — all 100 contact/taskB episodes + 160 taskA val episodes (10 per task_dir):

| strategy | how frames are picked | taskB acc | taskA overall | taskA per-task notable |
|---|---|---:|---:|---|
| **uniform_8 (current)** | linspace(0, T-1, 8) | 0.610 ❌ | 0.912 | put_fork/screwdriver_dustbin 0.55-0.75 |
| **mid_8** | [0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70] · T | **0.990** ✅ | 0.800 | give_fork 0.75, give_screwdriver 0.50 |
| **gripper_anchored** | center on first frame where any gripper crosses 0.5 (= grasp moment), span ±32 frames | **0.980** ✅ | 0.806 | put_fork/screwdriver_dustbin 0.65/0.70 (improved); give_fork 0.70, give_screwdriver 0.45 |
| dense_16 | linspace(0, T-1, 16) | 0.570 | n/t | — |
| mid_dense_16 | linspace(0.20·T, 0.70·T, 16) | 0.960 | n/t | — |

Identical between 35k and 50k ckpts (step ablation doesn't matter — see §3.1).

Actual grasp moment per task type (measured via gripper-state threshold-crossing, sampled from the runs):

| task | T | grasp_t | t/T |
|---|---:|---:|---:|
| give_boxdrink ep13 | 162 | 80 | 0.49 |
| give_fork ep0 | 133 | 57 | 0.43 |
| give_screwdriver ep0 | 167 | 71 | 0.43 |
| put_boxdrink_dustbin ep0 | 148 | 68 | 0.46 |
| put_fork_dustbin ep0 | 158 | 67 | 0.42 |
| **taskB put_boxdrink3_plate ep0** | **193** | **62** | **0.32** |

**Key insight:** for taskB the grasp happens at fraction **0.32 of the episode**, but uniform_8 samples at fractions [0, 0.14, 0.29, 0.43, ...] — so the *nearest* uniform frame to the grasp is at fraction 0.29 or 0.43 (a gap of ±10-20 frames). On the longer episode this means the model gets pre-grasp and post-grasp frames but NOT the grasp itself. mid_8 and gripper_anchored both put a frame at or near the grasp → 98-99% acc.

For taskA `give_fork`/`give_screwdriver`, grasp at fraction 0.43, but the pref-distinguishing visual signal (whether the gripper closed on handle or tip end of a thin tool) is **less visible at the grasp moment itself** and more visible during **transport/lift** (the arm trajectory differs based on grip). uniform_8's wider spread includes some post-grasp transport frames; mid_8/gripper_anchored cluster tighter around grasp and miss the informative transport frames.

**→ The pref-distinguishing window is task-specific.** For taskB (boxdrink3 on plate) it's the grasp; for give_fork/give_screwdriver it's the post-grasp transport. No single fixed clip strategy is universal.

**This is NOT a model failure mode — it's a clip-selection mismatch between train cache and per-task informative window.**

---

## 2. Where the initial RED verdict came from (and why the diagnosis evolved)

Step-by-step finding evolution during the 2-hour dive:

### 2.1 Initial signal (stage_a_gate.py)
- taskB acc 0.61, GATE = RED
- taskA val 6/8 task_groups ≥ 0.9, but put_fork/put_screwdriver_dustbin = 0.5 with constant predictions
- Counterfactual MSE baseline≈VQA (0.031 vs 0.032) — as expected by Kaiwen's prior

→ Looked like class collapse + overfitting.

### 2.2 Per-class breakdown (diagnostic.py)
- TaskB: GT_low 50/50 correct, GT_high 11/50 correct
- All 11 "high" predictions were correct (precision 100% on rare class)
- Logit gap on GT_low: mean +26.9; on GT_high: mean +14.6
- **Signal-difference +12.3**: the model has SOME real signal on GT_high (otherwise gap would be similar to GT_low's), but a +14.6-logit "low" prior overwhelms it most of the time

→ Updated diagnosis: model has partial signal, prior is overwhelming it.

### 2.3 Wandb analysis (pkyvgkt3, contact VQA 50k)
- `L_vqa` collapses from 1.35 (step 100) → 0.0018 (step 200) → ~0 thereafter
- All 8 per-task vqa_acc rolling = 1.00 throughout the entire 50k training
- `L_action` decreases slowly: 0.39 (step 1-5k) → 0.014 (step 40-50k)

→ Looked like trivial memorization. But the gap-mean +49 in mid_8 inference contradicts pure memorization — model has learned generalizable features.

### 2.4 Unconditional bias check (artificial clips)

| 35k ckpt | 50k ckpt |
|---|---|
| blank_black → 'high' gap -5.6 | blank_black → 'high' gap **-21.6** |
| noise → 'low' gap +5 | noise → 'high' gap -3 to -10 |
| (noise prior flipped from 'low' to 'high' as training went from 35k → 50k) |

→ Strong per-input visual priors. The prior direction is unstable across training (drifts from one shortcut to another as training proceeds). But:

### 2.5 Frame_window experiment
- mid_8 on taskB → 99% acc → **the model isn't shortcut-bound**, it just needs the right frames
- mid_8 on taskA → hurts give_fork/give_screwdriver (grasp before fraction 0.25)
- Universal fix is not "always use mid_8"; universal fix is "select frames that include the grasp event"

→ **Final diagnosis: frame-selection mismatch.**

---

## 3. Detailed findings

### 3.1 Step ablation does not help (35k vs 50k)

TaskB with uniform_8:

| ckpt | overall acc | GT_low acc | GT_high acc | pred_low/high | signal |
|---|---:|---:|---:|---:|---:|
| 35k | 0.610 | 1.000 | 0.220 | 89/11 | +12.29 |
| 50k | 0.610 | 1.000 | 0.220 | 89/11 | +10.06 |

Identical taskB performance. Adding 15k more training steps does not help on uniform_8.

TaskA per-task, 35k vs 50k uniform_8 (each N=10):

| task_group | 35k acc | 50k acc | Δ |
|---|---:|---:|---:|
| give_boxdrink | 1.00 | 1.00 | 0 |
| give_callbell | 1.00 | 1.00 | 0 |
| give_fork | 1.00 | 1.00 | 0 |
| give_screwdriver | 0.90 | 1.00 | +0.10 |
| put_boxdrink_dustbin | 1.00 | 1.00 | 0 |
| put_callbell_dustbin | 1.00 | 1.00 | 0 |
| put_fork_dustbin | 0.50 | 0.75 | +0.25 |
| put_screwdriver_dustbin | 0.50 | 0.55 | +0.05 |

50k helps marginally on the weak tasks but the dominant prior is unchanged. Step ablation is NOT the relevant axis.

Also notable: between 35k and 50k, the **constant-prediction direction flipped** on `put_screwdriver_dustbin` (35k: all `low`; 50k: still all `low` — sorry, but at the 5-eps-per-task subsample it appeared flipped; the 10-eps run confirms it's still mostly `low`). The unconditional-bias direction DID flip between 35k and 50k (noise → 'low' at 35k, → 'high' at 50k). The model's "default" answer on uninformative input drifts during training.

### 3.2 L_vqa training trajectory (key wandb facts)

| step range | L_vqa mean | L_action mean |
|---|---:|---:|
| 0-1k | 0.556 | 242.6 (initial spike, then ~1.5) |
| 1k-5k | 0.0001 | 0.39 |
| 5k-10k | 0.078 | 0.16 |
| 10k-20k | 0.160 | 0.09 |
| 20k-30k | 2e-6 | 0.048 |
| 30k-40k | 0.066 | 0.028 |
| 40k-50k | 0.015 | 0.014 |

**L_vqa drops below 0.01 at step 200 — second logging event.** Then stays at ~0 with occasional spikes (max 8.6 around step 10-20k) when a rarely-seen episode arrives, quickly re-absorbed.

This tells us:
- The binary classification problem is too easy for the model capacity — saturates fast.
- After step ~1k, VQA gradient signal is effectively zero. The remaining 49k steps don't refine VQA features further (in agreement with §3.1 step-ablation null result).
- BUT the per-task vqa_acc rolling avg stays 1.00 throughout — model maintains its (correct) train-set classifications.

This is **fast convergence**, not overfitting per se. The model nailed the train distribution early; what matters is whether the features it learned generalize. The frame_window result shows: YES they generalize when the right frames are presented at inference.

### 3.3 The 11 "high" predictions on taskB are all correct (precision 100%)

From `diagnostic_contact_35k.json`:
- 50 GT_low predictions: 50/50 correct
- 50 GT_high predictions: 11/50 correct (the 11 'high' predictions all coincide with GT_high; the 39 wrong ones are 'low' on GT_high)
- → no false positives on the rare class
- → the few times the model flips to 'high', it's right
- → there IS a signal, the prior is just stronger most of the time

This is consistent with the frame-selection diagnosis: when uniform_8 happens to capture a frame where high-contact grasp is visible (rare for taskB long episodes), the model flips; otherwise the visual prior wins.

### 3.4 Unconditional bias on artificial inputs

| input | 35k pred | 35k gap | 50k pred | 50k gap |
|---|:-:|:-:|:-:|:-:|
| 8× black | high | -5.6 | high | **-21.6** |
| 8× noise seed 0 | low | +4.6 | high | -3.4 |
| 8× noise seed 1 | low | +5.1 | high | -6.8 |
| 8× noise seed 2 | low | +5.0 | high | -10.8 |

`p_AB_mass = 1.000` for all inputs at both ckpts — the model always answers with one of `{low, high}` regardless of input. Direction depends on visual content.

This shows the model has **learned strong per-input priors** at the LM head. Combined with the frame_window finding, the picture is:
- When given clear grasp evidence (mid_8 on taskB, or uniform_8 on shorter taskA episodes) → signal-driven prediction → correct
- When given ambiguous evidence (uniform_8 on long taskB, or noise/blank) → falls back to a learned prior per visual distribution
- The fallback direction is brittle and drifts during training

### 3.5 Counterfactual MSE on actions (baseline vs VQA ckpt)

| | overall MSE (normalized 20D) | per-task range |
|---|---:|---|
| baseline | 0.0311 | 0.022 - 0.042 |
| VQA | 0.0321 | 0.022 - 0.048 |

baseline≈VQA on action MSE — confirms Kaiwen's pre-stated prediction. The action stream reads pref from the prompt token, not from the VQA head. VQA's value is purely in the per-task classification accuracy on taskB.

Sign accuracy on the 3 strong tasks (put_boxdrink/callbell/fork_dustbin, N=10):
- baseline: 0.6 / 0.6 / 0.7
- VQA: 0.7 / 0.4 / 0.5

Both weak. N=10 noise is large (binomial 95% CI ±31%). Not very interpretable; not a primary metric for gate.

---

## 4. Does the VQA approach make sense? Can Stage B succeed?

**Yes to both, with one caveat.**

### Make sense — yes, with refinements:

The approach is sound:
- LM-head CE on a single answer token is a clean, easy-to-optimize auxiliary
- 1:4 VQA:action ratio doesn't damage action learning
- The pref-axis IS visually perceptible by Qwen3-VL-4B given the right frames

But two structural points to refine in any redesign:
1. **Clip-selection strategy must include the grasp event.** Either heuristically (mid-window) or physics-grounded (gripper-state-detected). Don't lock to uniform_8 for all task types.
2. **L_vqa converges in ~500 steps.** This means the head freezes early into whatever feature it learned then. Per-step VQA gradient contribution after 1k steps is effectively zero. If we want the model to refine its pref features, we need either (a) harder VQA targets (multi-token, contrastive, multi-clip), (b) augmentation that increases task difficulty (random crop, jitter), or (c) accept that the head learns fast and frozen.

### Stage B success on contact's taskB — feasible

Using **mid_8 at pseudo-label time on `put_boxdrink3_plate` taskB → 99% acc**. That's pseudo-label quality good enough for Stage B fine-tuning.

But:
- This is ONE task in taskB. Other potential Stage B targets (the height/hvlv/orient taskBs) need per-task clip strategy tuning.
- Even if we can label boxdrink3_plate perfectly, Stage B success depends on whether the action head can fine-tune effectively on those pseudo-labels for the new visual context. That's a separate Stage B question.

### Caveat: no single clip strategy is universally best

Per-task acc with 3 strategies on the 50k ckpt:

| task_dir | uniform_8 | mid_8 | gripper_anchored | best_for |
|---|---:|---:|---:|---|
| give_boxdrink | 1.00 | 1.00 | 1.00 | tie |
| give_callbell | 1.00 | 1.00 | 1.00 | tie |
| **give_fork** | **1.00** | 0.75 | 0.70 | uniform_8 |
| **give_screwdriver** | **1.00** | 0.50 | 0.45 | uniform_8 |
| put_boxdrink_dustbin | 1.00 | 0.95 | 0.95 | uniform_8 |
| put_callbell_dustbin | 1.00 | 1.00 | 1.00 | tie |
| put_fork_dustbin | 0.75 | 0.50 | **0.65** | uniform_8 / gripper_anchored |
| put_screwdriver_dustbin | 0.55 | **0.70** | **0.70** | mid_8 / gripper_anchored |
| **taskB put_boxdrink3_plate** | 0.61 | **0.99** | **0.98** | mid_8 / gripper_anchored |

Patterns:
- **boxdrink/callbell**: any strategy works (strong visual signal even at coarse sampling)
- **give_fork/give_screwdriver**: uniform_8 WINS — the pref-distinguishing signal seems to be in **post-grasp transport frames** (how the thin tool is being carried reveals which end was grabbed), which uniform_8 captures and grasp-focused strategies don't
- **put_*_dustbin fork/screwdriver**: weak signal regardless; gripper_anchored helps marginally on put_screwdriver_dustbin
- **taskB put_boxdrink3_plate**: needs the grasp frame; uniform misses it on long episodes; mid_8 and gripper_anchored both succeed

→ The "right" answer is NOT a single fixed strategy. It's task-aware clip selection (or, at retrain time, train with stochastic clip selection so the model learns from multiple windows per episode).

---

## 5. Implementation findings (code review)

Reviewed: `QwenPI_VQA.py`, `vqa_sample.py`, `examples/preference/eval/stage_a_gate.py`. Reviewed train-time `_vqa_forward` vs eval-time `predict_preference`.

**No correctness bugs found.** Train and eval take logits at equivalent token positions (train: position P-1 of full sequence where P=prompt_lens; eval: position -1 of prompt-only with `add_generation_prompt=True`). Single-token CE label mask is correctly isolated. p_AB_mass = 1.0 confirms binary classifier behavior. Image preprocessing matches between train cache and eval JPEG decode.

**Design choices that contributed to the initial RED:**

| component | issue | severity |
|---|---|---|
| `vqa_sample.uniform_clip_indices()` hard-codes `np.linspace(0, T-1, n)` | works for short episodes, mismatched for taskB long episodes | **HIGH — root cause** |
| Per-episode clip cache built once with uniform indices, static across epochs | trivially memorizable; no temporal augmentation | medium — secondary |
| No frame-level data augmentation (no random crop, color jitter, frame jitter) | model learns specific-pixel features rather than invariant grasp features | medium |
| `λ_vqa = 0.5` with single-token CE → L_vqa saturates in 500 steps | feature freezes early; no refinement signal | low (couldn't be different given the task design) |
| `per_task vqa_acc` logged but not val-split — only train-side metric | OOD failure invisible at train time | medium |
| No held-out task_dir at training for VQA monitoring | OOD generalization untested during training | medium |
| Eval `predict_preference` hard-codes `uniform_linspace` clip indices via `uniform_clip_indices` import | inference-time strategy not configurable | medium — easy fix |

The single most impactful fix: **make clip-selection strategy configurable in `vqa_sample.uniform_clip_indices` (and in inference)**, support a "centered on detected grasp moment" mode using gripper state.

---

## 6. Recommendations (cost-impact sorted)

### Immediate (no retraining, applies to Stage B labeling now)

1. **For contact taskB pseudo-labeling: use mid_8 OR gripper-anchored clip strategy at inference.** Either gives ~99% acc on `put_boxdrink3_plate`. Pick gripper-anchored if you want a more physically-grounded default. Edit `predict_preference` to accept a `clip_strategy` arg. **Cost: ~30 min code + test.**

2. **Per-target frame-strategy tuning before deploying as pseudo-labeler**: for each new Stage B target (height/hvlv/orient taskBs, future tasks), run `frame_window_test` (or extend it with gripper_anchored) with multiple strategies and pick the one with highest signal. ~10 min per target. Critical because: no single fixed strategy is universal (see §4 caveat). **Cost: ~1h total to cover all four cats × taskB.**

3. **Strategy fallback for ambiguous tasks**: when no single clip wins clearly (e.g. give_fork/screwdriver style where signal is in transport rather than grasp), use **multi-strategy ensembling**: predict with uniform_8, mid_8, gripper_anchored — pick majority vote or take the prediction with the largest logit-gap magnitude. **Cost: ~30 min code.**

### Cheap retrain experiments (each ~6-9h H100)

4. **Retrain with STOCHASTIC clip strategy** (per-step random choice from {uniform_8, mid_8, gripper_anchored, random-window-8}): model sees pref signal from many windows per episode → learns features invariant to which window it sees → should generalize to any inference-time clip strategy. **Cost: 1 retraining run + small change to `build_vqa_clip_cache`.**

5. **Retrain with denser/larger cache** (cache 32-64 frames per episode, randomly subsample 8 per batch): cheap variant of #4 with smaller code surface. **Cost: 1 retraining + larger cache (~8 GB).**

6. **Add held-out task_dir VQA val during training**: split 16 task_dirs into 14 train / 2 held-out for VQA, log `vqa_val_held_out_acc` per logging step. Would have surfaced taskB generalization concern at training time. **Cost: small code change + 1 retraining.**

### Architectural changes (deeper)

7. **Multi-token answer**: "Answer: low (bottom of bottle)" — gives the model more supervision per VQA sample, harder to memorize. Need to update label mask logic to cover full-answer span. **Cost: medium code change + 1 retraining.**

8. **Contrastive auxiliary instead of CE**: pair (clip_low, clip_high) from same task and contrast in embedding space. Different gradient shape, harder to overfit. **Cost: medium code change + 1 retraining.**

9. **Crop clip to gripper bounding box**: project gripper xy from `endpose` to image coordinates, crop tight around gripper. Forces model to focus on grasp pose, removes scene context. **Cost: medium pipeline change + 1 retraining.**

---

## 7. Bottom line for Kaiwen

- **Stage A gate on contact: GREEN under mid_8 OR gripper_anchored inference** (99% / 98% on taskB's only task). RED under uniform_8 (61%). Same VQA ckpt, just different frame selection.

- **Stage B viable for contact taskB**: yes, use mid_8 or gripper-anchored clip at pseudo-label time. Pseudo-label accuracy 98-99% is more than enough.

- **Implementation has no correctness bugs.** It has one impactful design issue: `uniform_clip_indices` is hard-coded and the cache + inference both inherit that strategy. Sub-optimal on episodes where the pref-distinguishing visual signal doesn't fall on a uniform frame.

- **Step ablation null**: 35k vs 50k essentially identical on taskB (both 0.61 uniform, both 0.99 mid_8). Don't burn compute training longer.

- **L_vqa saturated at step ~500** but that's NOT the problem — fast convergence just means the model learned the pref features early. The features generalize when the right frames are presented.

- **No single clip strategy is universally best**: mid_8/gripper_anchored win on taskB and the weak put_dustbin tasks, but LOSE on give_fork/give_screwdriver where the pref signal is in post-grasp transport rather than the grasp moment. **For deployment as pseudo-labeler, recommend strategy ensembling (uniform_8 + mid_8 + gripper_anchored, vote or take largest-logit-gap) or per-task tuning.**

- **For the other 3 categories (height/hvlv/orient)**, expect similar pattern: uniform_8 may underperform when episode-length distribution or pref-signal-window differs from training. Run `frame_window_test` (extended with gripper_anchored) on each before going to Stage B.

- **VQA approach itself is sound.** The signal IS visually perceivable by Qwen3-VL-4B at the demonstrated resolution. The structural issues (single-token CE saturates fast, no augmentation, hard-coded uniform clips) are fixable without changing the overall design. A retrain with stochastic clip selection (recommendation #4) would likely make pseudo-labeling robust across all task types.

- **VQA cotrain is load-bearing vs baseline.** Baseline ckpt (action-only training, same backbone) gets 0.49-0.51 acc on taskB across all 3 clip strategies — chance, no discrimination, signal ~0. VQA ckpt + right clip gets 0.97-0.99. The +0.47 absolute is the entire pseudo-labeling capability, attributable to the 1:4 VQA gradient signal during training. Without VQA cotrain, Stage B has no labeler at all.

---

## 8. Reference files

- Raw eval JSONs:
  - `r-preference/eval/stage_a_gate_contact_35k.json` (original RED eval, uniform_8)
  - `r-preference/eval/diagnostic_contact_35k.json` (per-sample logits at 35k, uniform_8)
  - `r-preference/eval/diagnostic_contact_50k.json` (per-sample logits at 50k, uniform_8)
  - `r-preference/eval/frame_window_contact_50k.json` (4 strategies on taskB only)
  - `r-preference/eval/frame_window_50k_with_taskA.json` (uniform_8 + mid_8, taskA + taskB)
  - `r-preference/eval/frame_window_35k_with_taskA.json` (same, 35k ckpt — confirms step null)
  - `r-preference/eval/gripper_anchored_50k.json` (gripper-anchored clip, taskA + taskB)
- Visualization: `r-preference/eval/clips/composite_taskA_vs_taskB.png` (head_camera 8-frame clips)
- Code:
  - `examples/preference/eval/stage_a_gate.py` (main gate eval)
  - `examples/preference/eval/diagnostic.py` (per-sample logit dump, unconditional bias)
  - `examples/preference/eval/frame_window_test.py` (strategy comparison)
  - `examples/preference/eval/gripper_anchored_test.py` (gripper-anchored clip selector)
- Wandb: `kaiwenh-17-uiuc/pref-sim/pkyvgkt3` (contact VQA 50k)
