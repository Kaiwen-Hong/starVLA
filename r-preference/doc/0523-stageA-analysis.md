# Stage A gate eval — methodology, implementation, contact results, how to extend

> **Self-contained reference doc** for the Stage A pseudo-labeler readiness
> evaluation. Written 2026-05-24 after the contact-category analysis. Lets
> you (a) understand what we measured and why, (b) reproduce on contact,
> (c) extend the same pipeline to `height` / `hvlv` / `orient` / future
> categories with minimal new code.
>
> **Companion docs:**
> - [`0522-a-giveobj.md`](0522-a-giveobj.md) — original contact (legacy giveobj) Stage A spec.
> - [`0523-height-hv-oreint-design-doc.md`](0523-height-hv-oreint-design-doc.md) — 3-new-categories design + run state.
> - [`training-runbook.md`](training-runbook.md) — H100/H200 ops, storage, launch scripts.
>
> **Raw output of contact run** (referenced throughout): [`../eval/stage_a_gate_contact_35k_analysis.md`](../eval/stage_a_gate_contact_35k_analysis.md).

---

## 0. TL;DR

- **Stage A "gate" question**: can the VQA-cotrained ckpt produce
  high-quality pseudo-labels on the category's **unseen taskB**, so that
  Stage B fine-tuning has reliable labels?
- **Decision metric** is per-task VQA accuracy on **taskB** episodes.
  Secondary metrics: VQA acc on taskA val (sanity that training worked),
  counterfactual action MSE (sanity that VQA didn't break action), sign
  accuracy on strong tasks (action-head sanity), baseline-vs-VQA drift.
- **Implementation**: 5 scripts under `examples/preference/eval/` plus a
  `clips/` subdirectory of head_camera composites. ~40 minutes of GPU on
  a single H100 per category.
- **Key finding on contact**: the original `uniform_8` clip strategy
  caused taskB to look RED (0.61 acc, severe class collapse). Swapping to
  `mid_8` or `gripper_anchored` at inference time → 0.97-0.99 acc on
  taskB (97-99 of 100 episodes correct). **NO retraining needed** —
  model weights already learned the pref features; the clip-selection
  function was the only weak link, and it lives in `predict_preference`
  inference only.
- **The "design issue" is fixable at inference**:
  `examples/preference/dataset/vqa_sample.py:uniform_clip_indices()` is
  hard-coded to `np.linspace(0, T-1, n)`. For Stage B labeling, bypass it
  with `mid_8` / `gripper_anchored` clip selection — see §5.
- **Universally**: no single clip strategy is best across all task types.
  give_fork/give_screwdriver style tasks need uniform_8 (their pref
  signal lives in post-grasp transport, not at-grasp). For each new cat,
  re-run §4.5 the strategy ablation on its taskB before pseudo-labeling.
- **VQA cotrain IS load-bearing**: baseline (action-only) ckpt gets
  chance accuracy (0.49-0.51) on taskB with all 3 clip strategies. The
  +0.47 absolute gap = the entire pseudo-labeling capability comes from
  the 1:4 VQA gradient.

---

## 1. What this evaluation measures (and why)

Stage A produces 2 ckpts per category:
- **baseline** (`pref_baseline_stage_a_v1_noVQA_<cat>`): action-only training,
  no VQA head supervision. Lives in `Qwen_PI` framework.
- **VQA** (`pref_main_stage_a_v1_VQA_<cat>`): action + 1:4 VQA cotrain
  (`L_action + λ_vqa·L_vqa`). Lives in `Qwen_PI_VQA` framework.

The whole point of the VQA cotrain is that the LM head can be used at
Stage B as a **per-episode pseudo-labeler** on `<cat>/taskB/*` episodes
(visually different from taskA). If VQA acc on taskB is low, Stage B
gets garbage labels.

So the gate test asks 3 questions in priority order:

| # | Question | Metric | Decision |
|---|---|---|---|
| 1 | Can the VQA ckpt pseudo-label `<cat>/taskB` episodes well? | per-task VQA acc on taskB | **gate**: ≥0.90 on at least the dominant taskB task type → GREEN |
| 2 | Did training even work? (sanity) | VQA acc on taskA val (held-out 20% from train split) | sanity: should be ≥0.85 on strong tasks |
| 3 | Did VQA cotrain damage the action stream? (sanity) | counterfactual MSE on actions, baseline vs VQA | should be similar (Kaiwen's prior: same) |

Each of these has a script (§3).

> **⚠ Sanity check before interpreting #1 / #3 — added 2026-05-26**:
> Always read `baseline_counterfactual_taskA.overall_mse_normalized` from the
> gate JSON FIRST. In normalized [-1, 1] action space, contact reference
> baseline_CF ≈ 0.031 (model has learned a usable action regressor). If
> `baseline_CF ≈ 1.4`, the model is at the **noise floor** — it has not
> learned to fit the action distribution at all; output is essentially random
> per inference call. In that regime, taskB VQA acc / clip-strategy ablations /
> baseline-vs-VQA comparisons are **meaningless**: there is no working action
> model to interrogate. Debug the action training first (see
> [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) §4 for
> the height + hvlv example where this trap was set, and §6 for what to test
> when baseline_CF ≈ 1.4).

### 1.1 The two-axis ablation embedded in the eval

For each (ckpt, clip strategy) pair we measure VQA acc — this exposes
**brittleness from frame selection vs brittleness from training**:

- If baseline → chance, VQA(any strategy) → high → VQA training is load-bearing ✓
- If VQA(uniform_8) → low but VQA(mid_8 / gripper_anchored) → high →
  features exist, inference strategy is the issue (no retrain needed)
- If VQA(any strategy) → low → features missing, retrain needed

This 2-axis ablation was added late in the contact analysis and turned
out to be **the key insight**. The first-pass analysis on uniform_8 only
gave a misleading RED.

---

## 2. Scripts at a glance

All in `examples/preference/eval/`. Each is a CLI runner; total ~1500
lines but designed to be readable + reusable.

| script | purpose | typical wall (H100) | what to look at in output |
|---|---|---|---|
| `stage_a_gate.py` | First-pass gate eval. VQA per-task acc on taskA+taskB, counterfactual MSE for baseline AND VQA, sign acc on strong tasks. Writes JSON + md summary. | ~20 min for 1 cat (loads 2 ckpts sequentially) | `vqa_taskB.per_task_acc`, `gate.verdict` |
| `diagnostic.py` | Per-sample logit dump (raw `low_logit`, `high_logit`, gap, full-vocab marginal mass). Unconditional bias probe with blank/noise clips. Per-task confusion. | ~3 min (VQA ckpt only) | `taskB_records[].logit_gap`, `unconditional.*`, `pred_dist` |
| `frame_window_test.py` | Compares clip-selection strategies (uniform_8, mid_8, dense_16, mid_dense_16; extensible) on taskA + taskB. | ~5 min for 4 strategies on 260 ep | per-strategy `taskB.acc`, `taskA per_task` |
| `gripper_anchored_test.py` | Clip strategy centered on first gripper closure detected from `endpose/{left,right}_gripper`. Universal-by-design. | ~3 min on 260 ep | `taskB acc`, `grasp_t/T` distribution |
| `baseline_vqa_compare.py` | Apples-to-apples: same `predict_with_logits` on BOTH baseline and VQA ckpts, multiple strategies. Quantifies VQA cotrain's value. | ~5 min for 2 ckpts × 3 strategies | per-(ckpt, strategy) `taskB.acc` |

All scripts share helpers from `stage_a_gate.py` (dataset loading,
episode subsampling, framework instantiation, ckpt loading) and from
`diagnostic.py` (`predict_with_logits`).

---

## 3. Detailed implementation walk-through

### 3.1 `stage_a_gate.py` — main gate eval

**File**: [`examples/preference/eval/stage_a_gate.py`](../../examples/preference/eval/stage_a_gate.py) (~600 lines)

**Core flow** (`main()`):

1. Build taskA val dataset via `PrefHDF5Dataset(split="val", category=<cat>)`
   — same loader the trainer used, so split is consistent.
2. Subsample `taskA_n_eps_per_task` episodes per task_dir (defaults 5).
3. Enumerate `taskB` episodes (all of them) via `list_taskB_episodes`
   from `<root>/<cat>/taskB/<task_group>_<pref_key>/data/episode*.hdf5`.
4. PASS 1 — baseline ckpt:
   - `build_framework(baseline_yaml, "QwenPI")` → instantiates `Qwen_PI`
   - `load_ckpt_into_model(...)` → `torch.load + load_state_dict(strict=False)`
   - For each taskA frame sample, run 2× `predict_action` (pref=A vs B)
     to get counterfactual action chunks; compute MSE + sign acc.
   - Cache per-sample (a_A, a_B) for later baseline-vs-VQA drift.
   - Unload before pass 2 (model is ~16 GB, can't hold both).
5. PASS 2 — VQA ckpt:
   - Same `predict_action` on taskA → counterfactual MSE + sign acc.
   - `predict_preference` on taskA episodes → VQA per-task acc + confidence.
   - `predict_preference` on taskB episodes → **GATE METRIC**.
6. Compute baseline-vs-VQA drift from cached (a_A, a_B) tensors.
7. Apply gate verdict rule: `taskB acc ≥ 0.90` for ≥1 task → GREEN.
8. Save `r-preference/eval/stage_a_gate_<cat>_<step>.json` + `.md` summary.

**Key helpers** (also imported by other scripts):

```python
# (paths relative to repo root)
build_framework(yaml_path, framework_name) -> (model, cfg)    # line ~205
load_ckpt_into_model(model, ckpt_path)                        # line ~220
load_clip_from_h5(h5_path, n=8, camera="head_camera",         # line ~58
                  image_size=(224, 224)) -> List[PIL.Image]
list_taskB_episodes(taskB_root, pref_keys) -> List[FrameSample] # line ~80
subsample_taskA_episodes(ds, n_per_task, rng) -> List[FrameSample] # line ~105
pick_random_frame_for_sample(ds, fs, rng) -> int              # line ~130
```

**Sample dict expected by `predict_action`**:
```python
{"image": [PIL.Image x3 (head/left/right cam)],
 "lang":  "<base prompt> Preference: <label>",
 "state": np.float16 shape (1, 20)}
```

`predict_preference` (in `QwenPI_VQA.py:296`) takes a single `clip: List[PIL.Image]`
(8 frames default, head_camera only) and returns
`{"pref_key": ..., "label": ..., "confidence": ..., "p_A": ..., "p_B": ...}`.

**CLI**:

```bash
python -m examples.preference.eval.stage_a_gate \
    --category contact \
    --baseline_yaml examples/preference/train_files/starvla_pref_stage_a_baseline_contact.yaml \
    --vqa_yaml      examples/preference/train_files/starvla_pref_stage_a_vqa_contact.yaml \
    --baseline_ckpt /mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_contact/checkpoints/steps_35000_pytorch_model.pt \
    --vqa_ckpt      /mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_main_stage_a_v1_VQA_contact/checkpoints/steps_35000_pytorch_model.pt \
    --taskA_n_eps_per_task 5 \
    --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
    --out r-preference/eval/stage_a_gate_contact_35k.json
```

If you only want VQA pass, add `--skip_baseline`.

**Output JSON shape**:
```json
{
  "category": "contact",
  "baseline_counterfactual_taskA": {"overall_mse_normalized":..., "per_task_mse":{...}, "sign_accuracy_strong":{...}},
  "vqa_taskB": {"overall_acc":..., "per_task_acc":{...}, "pred_dist":{...}, "confidence_mean":...},
  "vqa_taskA": {... same shape ...},
  "vqa_counterfactual_taskA": {... same shape as baseline ...},
  "baseline_vs_vqa_drift_taskA": {"overall_drift_normalized":..., "per_task_drift":{...}},
  "gate": {"verdict": "RED|GREEN ...", "passed_ge_0.90": [...]}
}
```

The md summary (`_render_markdown`) gives a paste-ready table version.

### 3.2 `diagnostic.py` — per-sample logits + unconditional bias

**File**: [`examples/preference/eval/diagnostic.py`](../../examples/preference/eval/diagnostic.py) (~290 lines)

**What it adds over `stage_a_gate.py`**:

| field | meaning | why useful |
|---|---|---|
| `A_logit`, `B_logit` | raw logit at the answer position for the two candidate tokens | tells you HOW confidently the model decided. mean +25 is "very confident", mean +2 is "borderline" |
| `logit_gap = A_logit - B_logit` | the same in differential form | sign tells you direction; magnitude tells you strength |
| `p_A_pair`, `p_B_pair` | softmax over just those 2 logits | the "confidence" reported by `predict_preference` (current API). NB: always saturates near 1.0 once gap > ~5 |
| `p_A_full`, `p_B_full`, `p_AB_mass_full` | softmax over full vocab then sum of just the 2 tokens | tells you whether the model is "really" thinking about answering with one of these tokens (mass = 1.000 in practice for both ckpts → trained binary classifier behavior) |
| `top5` | top-5 raw logit tokens (id, decoded text, logit) | sanity: what would the model say if unconstrained? |
| `unconditional.{blank_black, noise_seed0/1/2}` | predict_preference on 8 black frames / 8 noise frames | reveals input-distribution-triggered priors |
| confusion matrix per task (TP_A, FP_A, ...) | which class gets confused with which | granular failure mode |

**The unconditional bias check** turned out to be very informative on
contact — see §4.4 of the analysis md (linked at top). For Stage B
purposes, the more important field is the per-sample logit_gap on
taskB, because we can use confidence-based filtering (e.g. only accept
pseudo-labels where |gap| > threshold).

**CLI**:

```bash
python -m examples.preference.eval.diagnostic \
    --category contact \
    --vqa_yaml  examples/preference/train_files/starvla_pref_stage_a_vqa_contact.yaml \
    --vqa_ckpt  .../steps_35000_pytorch_model.pt \
    --taskA_n_eps_per_task 5 \
    --taskB_data_root /mnt/localssd/kaiwenh/pref/data/contact/taskB \
    --out r-preference/eval/diagnostic_contact_35k.json
```

The output is large (~150 KB per ckpt) because it dumps per-sample
records. That's the point — use it to investigate specific failure
modes.

### 3.3 `frame_window_test.py` — clip strategy comparison

**File**: [`examples/preference/eval/frame_window_test.py`](../../examples/preference/eval/frame_window_test.py) (~180 lines)

**Strategies**:

| name | indices computed via | rationale |
|---|---|---|
| `uniform_8` | `np.linspace(0, T-1, 8)` | what the training cache uses |
| `mid_8` | fractions `[0.25, 0.32, 0.40, 0.45, 0.50, 0.55, 0.62, 0.70]` × `(T-1)` | clusters around mid-episode where grasp typically lives |
| `dense_16` | `np.linspace(0, T-1, 16)` | uniform but 2× the frames |
| `mid_dense_16` | `np.linspace(0.20*(T-1), 0.70*(T-1), 16)` | dense AND mid-clustered |

Add new strategies by editing `load_clip_strategy(h5_path, strategy)`
(line ~36). Each new strategy is one elif-branch.

**Why this script matters**: it caught the entire taskB regression of
the original gate eval. On contact:

```
strategy           taskB acc
uniform_8          0.610 ← RED
mid_8              0.990
gripper_anchored   0.980 (separate script)
dense_16           0.570
mid_dense_16       0.960
```

**CLI**:

```bash
python -m examples.preference.eval.frame_window_test \
    --category contact \
    --vqa_yaml ... --vqa_ckpt ... --taskB_data_root ... \
    --taskA_n_eps_per_task 10 \           # set to 0 to skip taskA
    --strategies "uniform_8,mid_8" \      # subset OK
    --out r-preference/eval/frame_window_contact_35k.json
```

### 3.4 `gripper_anchored_test.py` — physics-anchored clip

**File**: [`examples/preference/eval/gripper_anchored_test.py`](../../examples/preference/eval/gripper_anchored_test.py) (~210 lines)

**Heuristic**: find the **first frame where either gripper crosses below
0.5** (= transition open → closed = grasp moment). Sample 8 frames
densely around that moment (span ±32 frames, downsampled to 8).

Fallback: if no transition found (rare), use `uniform_8`.

**Why this script matters**: gripper-anchored is **universal by
construction** — works on any episode length, any task type, as long as
the gripper actually closes during the task. On contact taskB: 0.98 acc,
matching `mid_8` (0.99). For Stage B labeling on new task types where
we don't know the grasp-fraction distribution, gripper-anchored is the
safest default.

**Caveat**: same issue as `mid_8` — for tasks where the pref signal is
in post-grasp **transport** (e.g., contact's `give_fork`, `give_screwdriver`),
gripper-centering misses the relevant frames. See §6.3.

**CLI**:

```bash
python -m examples.preference.eval.gripper_anchored_test \
    --category contact \
    --vqa_yaml ... --vqa_ckpt ... --taskB_data_root ... \
    --taskA_n_eps_per_task 10 \
    --out r-preference/eval/gripper_anchored_contact_50k.json
```

Each output record includes `grasp_t` (the detected grasp frame) and
`T` (total frames), letting you compute the grasp-fraction distribution
per task type (useful for §5.2 when extending to new categories).

### 3.5 `baseline_vqa_compare.py` — does VQA cotrain help?

**File**: [`examples/preference/eval/baseline_vqa_compare.py`](../../examples/preference/eval/baseline_vqa_compare.py) (~190 lines)

**What it does**: loads BOTH baseline and VQA ckpts sequentially.
Manually injects `_vqa_question`, `_vqa_id_A`, `_vqa_id_B` etc. onto
the baseline `Qwen_PI` so the same `predict_with_logits` works
identically — apples-to-apples binary classification eval.

Tests N strategies × 2 ckpts on taskA + taskB, prints comparison table.

**Critical finding on contact** (validates that VQA training mattered):

```
strategy             baseline_taskB  vqa_taskB    baseline_taskA  vqa_taskA
mid_8                0.510           0.990        0.512           0.725
gripper_anchored     0.500           0.970        0.500           0.725
uniform_8            0.490           0.610        0.500           0.863
```

→ Baseline is at chance (~0.50) across all strategies, predicts "high"
essentially 100% of time, signal ≈ 0. VQA cotrain adds the entire
pseudo-labeling capability (+0.47 absolute on taskB with mid_8).

**CLI**:

```bash
python -m examples.preference.eval.baseline_vqa_compare \
    --category contact \
    --baseline_yaml ... --baseline_ckpt ... \
    --vqa_yaml ... --vqa_ckpt ... \
    --taskB_data_root ... --taskA_n_eps_per_task 5 \
    --strategies "mid_8,gripper_anchored,uniform_8" \
    --out r-preference/eval/baseline_vqa_compare_contact.json
```

---

## 4. Contact results in detail

All numbers from `r-preference/eval/*.json`. Full analysis in
[`../eval/stage_a_gate_contact_35k_analysis.md`](../eval/stage_a_gate_contact_35k_analysis.md).

### 4.1 Headline (gate metric on taskB, 100 episodes = 50 GT_low + 50 GT_high)

| strategy | overall acc | GT_low acc | GT_high acc | pred 25/75 | gap mean | signal (gap GT_low - GT_high) |
|---|---:|---:|---:|---:|---:|---:|
| uniform_8 (current) | 0.610 | 1.000 | 0.220 | 89/11 | +20.7 | +10.06 |
| mid_8 | **0.990** | 0.980 | 1.000 | 49/51 | +0.9 | **+49.17** |
| gripper_anchored | **0.980** | (similar) | (similar) | 49/51 | ~+0 | **+45.16** |
| dense_16 | 0.570 | 1.000 | 0.140 | 93/7 | +24.9 | +5.37 |
| mid_dense_16 | 0.960 | 0.920 | 1.000 | 46/54 | ~+1 | +46.34 |

**Read**: the same ckpt goes from 0.61 to 0.99 by changing only the clip
strategy at inference. The model has learned the right features; the
default strategy fails to surface them on taskB because taskB episodes
are longer (T≈193) and uniform 8 frames don't land on the grasp moment.

### 4.2 TaskA val acc per task_group (10 ep per task, 80 total)

| task_group | uniform_8 | mid_8 | gripper_anchored | best strategy |
|---|---:|---:|---:|---|
| give_boxdrink | 1.00 | 1.00 | 1.00 | any |
| give_callbell | 1.00 | 1.00 | 1.00 | any |
| **give_fork** | **1.00** | 0.75 | 0.70 | uniform_8 |
| **give_screwdriver** | **1.00** | 0.50 | 0.45 | uniform_8 |
| put_boxdrink_dustbin | 1.00 | 0.95 | 0.95 | uniform_8 |
| put_callbell_dustbin | 1.00 | 1.00 | 1.00 | any |
| put_fork_dustbin | 0.75 | 0.50 | **0.65** | uniform_8 / gripper |
| put_screwdriver_dustbin | 0.55 | **0.70** | **0.70** | mid_8 / gripper |
| **overall** | 0.91 | 0.80 | 0.81 | uniform_8 |

**Important pattern**: `give_fork` / `give_screwdriver` LOSE accuracy
when we cluster frames around the grasp moment. Why? In those tasks the
pref-distinguishing visual signal (whether the gripper closed on the
handle end or the tip end of a thin tool) is more visible during the
**post-grasp transport** (the arm trajectory differs based on grip
location) than at the grasp moment itself. uniform_8's wider spread
includes those transport frames; mid_8/gripper miss them.

→ The right strategy is **task-specific**. For Stage B labeling on a
new task type, run `frame_window_test` first; for unknown tasks, default
to gripper_anchored (most physically grounded) but consider ensembling
across multiple strategies (vote, or take prediction with largest |gap|).

### 4.3 Baseline vs VQA (validates VQA cotrain is load-bearing)

Already shown in §3.5. Bottom line: baseline = chance (0.50) on all
strategies; VQA + right strategy = 0.97-0.99 on taskB.

### 4.4 Counterfactual MSE on actions (Kaiwen's prior was right)

baseline overall MSE 0.0311 vs VQA 0.0321 across 80 frames — identical.
Both action heads read pref from the prompt token; VQA training did not
change action quality (positive nor negative). This was the expected
result per spec.

### 4.5 Step ablation (35k vs 50k) is null

Tested both 35k and 50k VQA ckpts under uniform_8 and mid_8. Results
within ~1% on taskB. Conclusion: **L_vqa saturates fast (drops below
0.01 at step 200, see wandb pkyvgkt3), and the features it learned
then don't refine further**. Don't burn compute training past 35k for
similar setups.

### 4.6 Wandb (run pkyvgkt3)

L_vqa trajectory (mean per logging bin):

| step range | L_vqa | L_action |
|---|---:|---:|
| 0-1k | 0.556 | 242.6 |
| 1k-5k | 0.0001 | 0.39 |
| 5k-10k | 0.078 | 0.16 |
| ... | ~0 with spikes | slowly decreasing |
| 40k-50k | 0.015 | 0.014 |

Per-task vqa_acc rolling avg = 1.00 throughout for ALL 8 task_groups.
This made the failure on `put_fork_dustbin` / `put_screwdriver_dustbin`
at val time invisible during training. **A held-out task_dir during
training would have surfaced it earlier** — see recommendation #4 in
the analysis md.

---

## 5. How to extend this to other categories

The pipeline is category-agnostic by construction — every script takes
`--category {contact, height, hvlv, orient}` and reads the per-category
constants from `PREF_CATEGORIES` and `VQA_CATEGORIES`. But there are a
few things to set up per category before the eval runs cleanly. Below
is the checklist for a new category.

### 5.1 Prerequisites for category `<cat>`

1. **Disk data**:
   - `<root>/<cat>/<task_group>_<pref_key>/data/episode*.hdf5` (taskA)
   - `<root>/<cat>/taskB/<task_group>_<pref_key>/data/episode*.hdf5` (taskB, optional — only needed for the gate test)
   - Where `<root>` = `/mnt/localssd/$USER/pref/data/`.
2. **Stats**: `examples/preference/dataset/stats_<cat>_v1.json` (generated via `precompute_stats.py`).
3. **YAML pair**: `examples/preference/train_files/starvla_pref_stage_a_{baseline,vqa}_<cat>.yaml`.
4. **Trained ckpts**:
   - baseline: `<results>/Checkpoints/pref_baseline_stage_a_v1_noVQA_<cat>/checkpoints/steps_<N>_pytorch_model.pt`
   - VQA: same naming with `_VQA_<cat>` (often on H200; rsync to H100 first).
5. **Registry entries already there**:
   - `PREF_CATEGORIES["<cat>"]` in `examples/preference/dataset/prompt.py`
   - `VQA_CATEGORIES["<cat>"]` in `examples/preference/dataset/vqa_sample.py`
6. **Sanity-check VQA answer tokens are single bare-form**: run
   ```bash
   python -c "from examples.preference.dataset.vqa_sample import VQA_CATEGORIES; \
              from transformers import AutoTokenizer; \
              t=AutoTokenizer.from_pretrained('Qwen/Qwen3-VL-4B-Instruct'); \
              c=VQA_CATEGORIES['<cat>']; \
              print({pk: (text, t.encode(text)) for pk,text in c.answer_text.items()})"
   ```

### 5.2 Recommended evaluation order

```bash
CAT=height   # or hvlv, orient, ...
USER_PATH=kaiwenh   # or kevin on H200
CKPT_BASELINE=/mnt/localssd/$USER_PATH/starVLA_runs/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_${CAT}/checkpoints/steps_25000_pytorch_model.pt
CKPT_VQA=/mnt/localssd/$USER_PATH/starVLA_runs/results/Checkpoints/pref_main_stage_a_v1_VQA_${CAT}/checkpoints/steps_25000_pytorch_model.pt
TASKB=/mnt/localssd/$USER_PATH/pref/data/${CAT}/taskB
YAML_BASE=examples/preference/train_files/starvla_pref_stage_a_baseline_${CAT}.yaml
YAML_VQA=examples/preference/train_files/starvla_pref_stage_a_vqa_${CAT}.yaml

# 1. First-pass gate (uniform_8 baseline) — this is what was run first on contact
python -m examples.preference.eval.stage_a_gate \
    --category $CAT --baseline_yaml $YAML_BASE --vqa_yaml $YAML_VQA \
    --baseline_ckpt $CKPT_BASELINE --vqa_ckpt $CKPT_VQA \
    --taskA_n_eps_per_task 5 --taskB_data_root $TASKB \
    --out r-preference/eval/stage_a_gate_${CAT}_25k.json

# 2. If gate is RED on taskB, run frame_window_test to check if it's clip strategy
python -m examples.preference.eval.frame_window_test \
    --category $CAT --vqa_yaml $YAML_VQA --vqa_ckpt $CKPT_VQA \
    --taskB_data_root $TASKB --taskA_n_eps_per_task 10 \
    --strategies "uniform_8,mid_8,gripper_anchored,dense_16,mid_dense_16" \
    --out r-preference/eval/frame_window_${CAT}_25k.json

# 3. Diagnostic for per-sample logit + unconditional bias (root-cause investigation)
python -m examples.preference.eval.diagnostic \
    --category $CAT --vqa_yaml $YAML_VQA --vqa_ckpt $CKPT_VQA \
    --taskA_n_eps_per_task 5 --taskB_data_root $TASKB \
    --out r-preference/eval/diagnostic_${CAT}_25k.json

# 4. Sanity: VQA cotrain vs baseline
python -m examples.preference.eval.baseline_vqa_compare \
    --category $CAT --baseline_yaml $YAML_BASE --vqa_yaml $YAML_VQA \
    --baseline_ckpt $CKPT_BASELINE --vqa_ckpt $CKPT_VQA \
    --taskB_data_root $TASKB --taskA_n_eps_per_task 5 \
    --strategies "uniform_8,mid_8,gripper_anchored" \
    --out r-preference/eval/baseline_vqa_compare_${CAT}_25k.json

# 5. Once you know the best strategy for this cat's taskB, deploy at Stage B label time
```

Wall time per category on a free H100: ~40 min total for the full
suite (mostly ckpt loading; per-strategy inference is <10s for 100 ep).

### 5.3 Things that DIFFER per category and may need code edits

| component | contact | height | hvlv | orient | what to check / edit |
|---|---|---|---|---|---|
| `PREF_CATEGORIES[cat].pref_keys` | `("25", "75")` | `("high", "low")` | `("hv", "lv")` | `("0", "90")` | already in registry |
| `VQA_CATEGORIES[cat].answer_text` | `low`/`high` | `low`/`high` | `far`/`near` | `horizontal`/`vertical` | already in registry |
| stats JSON | `stats_contact_v1.json` | `stats_height_v1.json` | `stats_hvlv_v1.json` | `stats_orient_v1.json` | already generated |
| strong tasks (sign acc) | 3 hardcoded in `stage_a_gate.py:STRONG_TASKS_CONTACT` | not yet defined | not yet defined | not yet defined | **edit** `stage_a_gate.py` to add `STRONG_TASKS_HEIGHT` etc. (only if you care about sign acc; for taskB gate it's irrelevant) |
| gripper-anchor threshold | 0.5 (contact/give-and-put) | likely 0.5 | likely 0.5 | likely 0.5 | check empirically; some tasks may have inverted gripper state |
| taskA task_groups | 8 (in registry) | 8 | 8 | 8 | already in registry |
| taskB task_groups | `put_boxdrink3_plate` (1 group × 2 pref keys = 2 dirs) | `place_playingcards1_box` (1 × 2) | `stamp_seal6` (1 × 2) | `move_can5_away` (1 × 2) | enumerated by `list_taskB_episodes` automatically |
| typical episode length T | 135-193 | TBD | TBD (longer; ~220 avg per training-runbook §9) | TBD | **important to know** for picking clip strategy |
| pref signal location (grasp vs transport) | mixed (boxdrink=at grasp, fork=post-grasp transport) | TBD (likely at-drop, not at-grasp — height = drop height!) | TBD (likely throughout transport — distance from obstacle is integrated over trajectory) | TBD (at-grasp probably — orientation is set at pickup) | **major unknown**; will be revealed by frame_window_test results |

### 5.4 Per-category caveats to expect

Based on the contact analysis, here's what to ANTICIPATE on each new
category — verify after running the eval but use as priors:

**height** (`high drop` / `low drop` = release-from-high vs close-to-surface):
- Pref signal location = at the **release/drop** moment, not at-grasp.
- Gripper-anchored = grasp moment ≠ pref signal. **Expect gripper-anchored to UNDERPERFORM**.
- mid_8 or dense_8-centered-on-late-fraction (0.55-0.85 maybe) might work better.
- Test a strategy like `late_8` with fractions [0.55, 0.62, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] before deployment.
- height's CLEAN_TEMPLATE is template-only (no paraphrase), so prompt-side is deterministic.

**hvlv** (`wide detour` / `narrow detour` = obstacle clearance):
- Pref signal is the entire **trajectory** distance from the can obstacle. NOT a single moment.
- 8 frames at any position may not capture trajectory shape well.
- **Expect dense_16 or longer clips to help.**
- The `mid_8` strategy may or may not help; the integrated nature of the signal suggests no single window is "right". Try uniform_16 if available; if not, ensembling multiple strategies likely best.

**orient** (`horizontal grasp` / `vertical grasp`):
- Pref signal is at the **grasp moment** (orientation of gripper at pickup).
- **Expect gripper-anchored to be the BEST strategy** here, by far.
- mid_8 should also work.
- uniform_8 likely fine because grasp tends to be mid-episode.

### 5.5 Strategy ensembling (recommended deployment default)

If you're unsure which strategy to use for a new category's taskB, run
the 3 main strategies and ensemble:

```python
# Pseudocode for Stage B labeling
for ep in stage_b_episodes:
    preds = []
    for strategy in ["uniform_8", "mid_8", "gripper_anchored"]:
        clip = load_clip_strategy(h5_path, strategy)
        result = model.predict_preference(clip)
        preds.append(result)
    # Option A: majority vote
    pseudo_label = Counter([p["pref_key"] for p in preds]).most_common(1)[0][0]
    # Option B: take the strategy with the largest |logit_gap| (most confident)
    best = max(preds, key=lambda p: abs(p["logit_gap"]))  # need to extend predict_preference to return logit_gap
    pseudo_label = best["pref_key"]
    # Option C: only label if all 3 agree (high precision, lower recall)
    if len(set(p["pref_key"] for p in preds)) == 1:
        pseudo_label = preds[0]["pref_key"]
    else:
        pseudo_label = None  # skip this episode
```

Option C is the most conservative and recommended if Stage B is
sensitive to label noise. Option A is the practical default.

---

## 6. Findings & lessons

### 6.1 Implementation lessons baked into the eval suite

1. **Always test multiple clip strategies on the gate-relevant set**.
   The single biggest mistake in the first-pass eval was running only
   uniform_8 — we got a RED that flipped to GREEN once we tested mid_8.
2. **Always compare against baseline (no VQA training)**. Validates
   that the VQA cotrain is load-bearing for the metric in question.
   If baseline → high acc too, the test isn't measuring what you think.
3. **Diagnostic per-sample dumps are cheap and high-value**. ~150 KB per
   ckpt eval; trivially saves you a re-run when you want to investigate
   "which 11 predictions were the rare ones, and were they correct?"
4. **Unconditional bias probes (blank/noise clip) take 4 forward passes
   and reveal LM-head priors**. Useful for any binary VQA classifier.

### 6.2 The "design issue" that caused the initial RED

`examples/preference/dataset/vqa_sample.py:149`:

```python
def uniform_clip_indices(T: int, n: int = VQA_NUM_FRAMES) -> np.ndarray:
    """Uniform-linspace frame indices into a T-frame episode."""
    return np.linspace(0, T - 1, n).round().astype(np.int64)
```

This is used in 2 places:
- **Train-time cache** (`vqa_sample.build_vqa_clip_cache:189-204`) — bakes
  in uniform frames per episode for the entire training run.
- **Inference `predict_preference`** — uses uniform clip selection,
  hard-coded through the function call chain.

For Stage A training the uniform cache is mostly fine — the model
sees enough variation across episodes to learn pref features. For
Stage B labeling on a different task distribution, uniform clip
selection is brittle to episode-length and pref-signal-location shifts.

**Fix is inference-only**: edit `Qwen_PI_VQA.predict_preference` to
accept a `clip` argument (it already does!) — the caller is the one
who decides what clip to pass. So the "fix" is changing the caller from
`uniform_8` to a more appropriate strategy. **No retrain needed**.

For a more permanent fix, modify `predict_preference` to accept a
`clip_strategy` enum and load the clip internally — but this is just
ergonomics; the model is already capable.

### 6.3 No single clip strategy is universally best

Discovered on contact (§4.2). For each category × task_group, the
"informative window" depends on where the pref-distinguishing physical
signal lives. Examples:

| pref axis | informative window | best strategy |
|---|---|---|
| grasp position on vertical object (contact boxdrink/callbell) | at grasp | uniform/mid/gripper-anchored all work |
| grasp position on flat tool (contact fork/screwdriver) | post-grasp transport | uniform_8 (wide spread) |
| drop height (height) | at release | late-clustered (frames 0.55-0.95) |
| trajectory clearance (hvlv) | whole transport | dense, possibly multiple clips ensembled |
| grasp orientation (orient) | at grasp | gripper-anchored |

This is dataset physics, not VQA-training brittleness. The signal is in
the pixels somewhere; the question is whether your 8 chosen frames
contain it.

### 6.4 L_vqa saturates fast — not the bottleneck

Single-token binary CE with a 4B+3.3B-param backbone converges in <500
steps. After that, L_vqa gradient is ~0 and the head is frozen-in-place.
Training longer (50k vs 35k) does not improve val acc on taskB. Don't
burn compute past convergence; instead invest in better clip selection
or stronger VQA targets (multi-token answers, contrastive aux).

### 6.5 Wandb logging gap

`vqa_acc/<task_group>` is logged per training step (one rolling sample
per logged step), and it stays at 1.0 throughout — the model memorizes
the train distribution trivially. **It does NOT log val-time vqa_acc**,
so OOD failures are invisible. Recommendation when retraining: add an
eval hook every save_interval that runs `predict_preference` on a few
val episodes per task_dir, log as `val/vqa_acc/<task_group>`.

---

## 7. File index

### Code (committed in `d016c47` and `089aabd`)

| path | purpose |
|---|---|
| `examples/preference/eval/__init__.py` | namespace placeholder |
| `examples/preference/eval/stage_a_gate.py` | main gate eval (CLI; JSON + md output) |
| `examples/preference/eval/diagnostic.py` | per-sample logit / unconditional bias dump |
| `examples/preference/eval/frame_window_test.py` | clip-strategy ablation |
| `examples/preference/eval/gripper_anchored_test.py` | gripper-anchored clip selector |
| `examples/preference/eval/baseline_vqa_compare.py` | baseline vs VQA pseudo-labeler comparison |

### Eval outputs (contact)

| path | what's in it |
|---|---|
| `r-preference/eval/stage_a_gate_contact_35k.json` | original first-pass gate eval |
| `r-preference/eval/stage_a_gate_contact_35k.md` | md summary (paste-ready) |
| `r-preference/eval/stage_a_gate_contact_35k_analysis.md` | **the 370-line thorough analysis** (read this when extending to other cats) |
| `r-preference/eval/diagnostic_contact_35k.json` | per-sample logits at 35k |
| `r-preference/eval/diagnostic_contact_50k.json` | per-sample logits at 50k (step ablation) |
| `r-preference/eval/frame_window_contact_50k.json` | first frame_window test (taskB only) |
| `r-preference/eval/frame_window_{35k,50k}_with_taskA.json` | frame_window incl. taskA val |
| `r-preference/eval/gripper_anchored_50k.json` | gripper-anchored clip eval |
| `r-preference/eval/baseline_vqa_compare_35k.json` | baseline vs VQA comparison |
| `r-preference/eval/clips/composite_taskA_vs_taskB.png` | visual head_camera composite |

### Dataset / framework code referenced

| path | what to look at |
|---|---|
| `examples/preference/dataset/prompt.py` | `PREF_CATEGORIES` registry (line 198), `build_action_prompt` |
| `examples/preference/dataset/vqa_sample.py` | `VQA_CATEGORIES` (line 97), `uniform_clip_indices` (line 149, **the hard-coded function**), `build_vqa_clip_cache` (line 172) |
| `examples/preference/dataset/pref_hdf5_dataset.py` | `PrefHDF5Dataset` (used as val ds), `_split_episodes` (line 53) for train/val split |
| `examples/preference/dataset/pref_hdf5_vqa_dataset.py` | adds clip-cache build at __init__ |
| `starVLA/model/framework/QwenPI.py` | `Qwen_PI.predict_action` (line 147) |
| `starVLA/model/framework/QwenPI_VQA.py` | `Qwen_PI_VQA.forward` (line 275), `_vqa_forward` (line 208), `_build_vqa_inputs` (line 123), `predict_preference` (line 296) |

### Wandb references

| run | category | use |
|---|---|---|
| `kaiwenh-17-uiuc/pref-sim/pkyvgkt3` | contact VQA 50k | the only run that completed full 50k with per-task vqa_acc logs |
| (others in pref-sim) | various | see `wandb api` or §3 of `0523-height-hv-oreint-design-doc.md` |

### Commits

| commit | what was added |
|---|---|
| `d016c47` | Stage A gate eval suite + contact thorough analysis (5 scripts + 8 JSONs + analysis.md + composite png) |
| `089aabd` | Baseline-vs-VQA comparison script + JSON + analysis.md §1a / §7 update |
| (this doc) | meta-doc for the eval methodology |

---

## 8. TODOs / open questions for next session

1. **Per-cat eval pass on height/hvlv/orient**: the suite is ready, just
   need ckpts available + ~40 min GPU per cat. Run §5.2 sequence.
2. **Add `predict_preference(clip_strategy=...)` arg** to
   `Qwen_PI_VQA.predict_preference` for cleaner Stage B integration.
   Currently strategy is decided at the caller level (5-line change).
3. **Strategy ensembling helper**: add `predict_preference_ensemble`
   that runs 3 strategies and returns majority-vote or max-confidence.
4. **STRONG_TASKS_* dicts** for height/hvlv/orient (currently
   `stage_a_gate.py:STRONG_TASKS_CONTACT` is the only one). Only matters
   for sign-acc reporting; gate metric doesn't depend on it.
5. **Confidence-based filtering for Stage B**: extend
   `predict_preference` to also return raw `logit_gap`; Stage B
   pseudo-labeler should only accept labels with `|gap| > threshold`
   (e.g., > 5 for reliable).
6. **Test gripper-anchor robustness** on tasks where gripper might not
   close cleanly (e.g., if drop is between two pre-closed states, or if
   left/right gripper convention changes per task).
7. **Retrain with stochastic clip strategy** (optional, ~8h H100, would
   give universal robustness across task types — see recommendations in
   analysis md §6).
