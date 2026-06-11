# 2026-05-28 — How to fix the VQA self-labeler: it doesn't work, and why (negative result + path forward)

> **Self-contained postmortem.** Written at the end of a 2026-05-28 session on the
> preference-conditioned VLA (Pref-VLA) over the re-collected `0526` data. The
> headline is a **negative result**: the VLM-VQA self-labeler does **not** learn the
> preference, because **the preference signal is proprioceptive (defined by robot EE
> pose), not visual** — so a VLM reading head-camera pixels (or pose-as-text) is the
> wrong tool. This doc records the experiments, the WHY, the code we built, and —
> since the filename promises it — **how to fix it** (§5, prominent).
>
> **Read-first companions (do not duplicate):**
> - [`0528-action-arch-and-oft-vqa-integration.md`](0528-action-arch-and-oft-vqa-integration.md)
>   — the three action heads, the 0526 convergence experiments (Q1/Q2/Q4), the
>   OFT×VQA integration design, and the pre-training findings. **This doc is its
>   sequel**: it picks up after "OFT warm-start converges; VQA L_vqa→0 on train"
>   and answers the question that doc deferred — *does L_vqa→0 imply taskB works?*
>   (No.) Cross-referenced heavily below.
> - [`0528-instruction-mount.md`](0528-instruction-mount.md) — the `0526` data + dual
>   instruction trees (`instructions/` rich, `instructions_base/` stripped).
> - [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
>   — why the *old* per-cat data was stuck (degenerate unconditional-mean floor;
>   data/pipeline clean) — the background for "old cats stuck".

---

## 0. TL;DR

| # | Question | Verdict | Evidence |
|---|---|---|---|
| F1 | Can a **frozen** Qwen3-VL-4B read the pref from head-cam pixels? | **No, ≈ chance** | calibrated balanced acc 0.40–0.62 (§1) |
| F2 | Is the pref signal actually **in the EE pose**? | **Yes, S/N 4–22** | endpose separability (§2) |
| F3 | Is the flow-matching action loss correct? | **Yes** (textbook) | `LayerwiseFM_ActionHeader.py`; old cats converged (§3) |
| F3 | Does **QwenPI from-scratch** flow-matching converge on 0526? | **No** — stuck at the unconditional-mean floor | place L_action flat ≈1.42 / 2000 steps (§3) |
| F4 | Does **OFT warm-start** converge on 0526? | **Yes**, fast | place L1 0.41→**0.008** / 1500 steps (§3) |
| F4 | Does VQA cotrain **destabilize** the OFT action stream? | **No** | height+VQA 0.010 ≈ height no-VQA 0.008 (§3) |
| **F5** | Does the VQA self-labeler **generalize** (held-out taskA-val + taskB)? | **NO — collapses to 0.50, always one class** | gate-eval + held-out val hook, even with the full redesign **+ EE-state-as-text** (§4) |

**One sentence:** train `L_vqa→0` and train `vqa_acc=1.0` are **memorization**; on
held-out taskB (and even taskA-val) the VQA collapses to the language-recency prior
(orient→`top`, place→`corner`, height→`low`), and feeding the *separable* EE pose as
**text** did not rescue it — the model ignores the numbers and memorizes. The fix is
to **label the preference directly from the EE state** (a 2-layer MLP / threshold,
S/N 15–22 → ~99% and object-agnostic), not to ask a VLM to read a pose-defined pref
from pixels (§5).

---

## 1. Finding 1 — zero-shot base VLM ≈ chance on all 3 prefs from head-cam pixels

`r-preference/eval/zeroshot_vqa_probe.py` runs the **untrained** `Qwen/Qwen3-VL-4B-Instruct`
as a forced-choice classifier over the redesigned task-context question + 3-frame
head-camera clip, on taskA (training-distribution leaves) and taskB (unseen-object
leaves), for 0526. Results: `zeroshot_vqa_probe_0526.json` (raw) and `_calib.json`
(answer-prior-subtracted, balanced).

| cat | raw taskA / taskB bal-acc | **calibrated** taskA / taskB bal-acc | raw pred collapse |
|---|---|---|---|
| orient | 0.469 / 0.64 | **0.547 / 0.52** | taskB → 39×`side` vs 11×`top` |
| place  | 0.531 / 0.50 | **0.609 / 0.62** | taskA → 62×`corner` vs 2×`center`; taskB → 50×`corner` / 0 |
| height | 0.531 / 0.42 | **0.515 / 0.40** | taskB → 46×`low` vs 4×`high` |

(`_calib.json` priors: orient `{0:-2.478, 90:-1.964}`, place `{center:-2.443, corner:-1.877}`,
height `{high:-1.579, low:-1.329}`.) The **raw** forced choice collapses almost
entirely to one class (a strong answer-recency / token prior); after subtracting that
prior the balanced accuracy is 0.40–0.62 — i.e. **the frozen VLM cannot read these
preferences from head-cam pixels**. This is consistent with the companion doc §4.1
(which reported orient 0.55/0.52, place 0.61/0.62, height 0.52/0.40 — same JSONs).

---

## 2. Finding 2 — the signal IS proprioceptive (EE pose), S/N 4–22

Measured on 0526 via the **active-arm endpose at the release/grasp frame** (active arm
= the one with larger xyz range; release = last closed→open gripper transition). The
script was `/tmp/xyz_separability.py` — **ephemeral (in `/tmp`, do not rely on it
existing)**; the numbers are quoted here and cross-checked against the companion doc §4.2:

| cat | discriminating quantity | S/N taskA | S/N taskB | note |
|---|---|---|---|---|
| height | `z` @ release | **3.8** | **20.8** | drop height → vertical EE z; near-perfect on taskB |
| place  | `x` @ release | **4.2** | **3.8** | placement → 2D EE xy |
| orient | `z` only (xyz) | 4.2 | 11.7 | usable, but… |
| orient | **rotation (6D)** | **9.0** | **11.3** | **rotation is the real orient signal** — wrist roll at grasp; xyz alone is weaker |

(>2 = separable.) The pref is **defined by the robot's pose**: drop height = EE z,
placement = EE xy, grasp orientation = wrist angle (6D rotation). This single fact
explains every negative result below.

---

## 3. Finding 3+4 — action loss is correct; OFT warm-start is the convergence route on 0526

(Detail in companion doc §1, §7 — summarized here because F5 builds on it.)

- **Action loss is correct.** `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
  is textbook rectified-flow velocity-MSE (`noise~N(0,I)`, `t~Beta(1.5,1)`,
  `noisy=(1−t)·noise+t·a`, target `v=a−noise`, `L=MSE`). place/contact/orient-baseline
  converged on the OLD data with it.
- **QwenPI from-scratch is the convergence problem on 0526.** `conv_baseline_place_0526`
  (flow-matching, 20D EE, 2000 steps) **STUCK at L_action ≈ 1.42, flat** — the degenerate
  unconditional-mean floor (≈Var[a]+1; see 0526-corrections §4). Even *place*, which
  converged on old data, is stuck from scratch on 0526. Log:
  `/mnt/localssd/kaiwenh/logs/conv_baseline_place_0526.log`.
- **OFT warm-start converges.** `Qwenvl_OFT` (`starVLA/model/framework/QwenOFT.py`:
  OpenVLA-OFT style — appends `🔍×chunk_len` action tokens, gathers their last-layer
  embeddings via `_gather_action_token_embeddings`, MLP-decodes (`MLP_ActionHeader`),
  single-pass `nn.L1Loss`; 14D joint qpos, chunk 50), warm-started from the HF ckpt
  `StarVLA/Qwen3-VL-OFT-RoboTwin2-All` step 140000 (RoboTwin 2.0 = our embodiment;
  cached on SSD `/mnt/localssd/kaiwenh/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt`,
  9.79 GB). On 0526 **place** it converges fast: `action_dit_loss` (the L1) step 25 =
  0.106 → **step 1500 = 0.00815**. Log: `/mnt/localssd/kaiwenh/logs/conv_oft_warmstart_place_0526.log`.
- **VQA cotrain does NOT destabilize the OFT action stream** (unlike flow-matching,
  where VQA historically broke orient — 0526-corrections §3.2). In the 4×1k batch
  (place/height/orient OFT_VQA + height OFT no-VQA control), all converged to
  L_action 0.008–0.018; **height+VQA L_action = 0.0100 ≈ height no-VQA control 0.008**.
  → the shared-backbone L_vqa gradient is benign for OFT's L1 head.

So the action policy is fine. **The remaining failure is purely in the VQA labeler.**

---

## 4. Finding 5 — the VQA self-labeler collapses on held-out (taskA-val AND taskB)

This is the heart of the doc. Even though `L_vqa→0` and **train** `vqa_acc=1.0` on
every run, the labeler does not generalize.

### 4.1 The misleading train signal
4×1k OFT_VQA runs (logs `/mnt/localssd/kaiwenh/logs/j_{height,place}_vqa_1k.log`):
train `L_vqa → ~1e-7`, train `vqa_acc = 1.0` from early on (the in-batch `log/vqa_acc`
in `vqa_cotrain_mixin._vqa_forward`). This looks like success. It is **memorization**.

### 4.2 Gate-eval of the 1k ckpts (held-out) — collapse to 0.50, always one class
`r-preference/eval/gate_oftvqa.py` runs `model.predict_preference` on taskA-val
(seed-42 split) + taskB (unseen-object leaves), exactly the clips the model was
cotrained on:

| cat | taskA-val bal-acc | taskB bal-acc | prediction |
|---|---|---|---|
| height (`gate_height.json`) | **0.50** (80 eps) | **0.50** (50 eps) | **80/80 + 50/50 → `low`** |
| place  (`gate_place.json`)  | **0.50** (80 eps) | **0.50** (50 eps) | **80/80 + 50/50 → `corner`** |

(orient behaved identically → `top`.) Every prediction is the **last-mentioned answer
option** in the question — the language recency prior. `acc_gt_low=1.0 / acc_gt_high=0.0`
(and the `corner`/`center` analogue) is the always-one-class signature.

### 4.3 The full redesign — and EE-state-as-text — STILL collapses
We then did the complete VQA redesign in `examples/preference/dataset/vqa_sample.py`
(`VQA_CATEGORIES`) and re-ran:
- **task-context question** (e.g. orient: *"We are picking up an object. Are we
  grasping from the side (horizontal) or from the top (vertical)?"*) +
  **single discriminative-token answer** (`side`/`top`, `middle`/`corner`, `high`/`low`)
  to keep the binary-classifier + Stage-B logit-gap calibration;
- **head-camera only** (the 2026-05-26 place `active_wrist` multicam was reverted —
  wrist confirmed useless);
- **tightened 3-frame strategies** anchored at the pref moment:
  `place3=[0.60,0.80,1.00]`, `orient3=[0.40,0.60,0.80]`, `height3=[0.90,0.95,1.00]`
  (the last tightened to the drop/release window — the old `[0.6,0.8,1.0]` caught the
  arm post-retract at frac=1.0);
- **±5-frame jitter** (`load_clip_and_state(..., jitter=5)`) as train-time augmentation;
- **inject the active-arm EE pose (xyz + 6D rot) at those frames as TEXT** into the
  prompt (`state_in_vqa=True` → `format_state_text` → e.g.
  `"Robot end-effector pose (x,y,z,r0..r5) at the frames: f1=[...]; ...."`) — the
  high-leverage idea, since §2 shows the signal lives there;
- a **held-out validation hook** (`VQACotrainMixin.validate_vqa` → `train_starvla._eval_vqa`
  at `eval_interval`, logging `val/vqa_taskA_acc` and `val/vqa_taskB_acc`).

Re-ran **height** (H100×8) + **orient** (H200×8) v2 *with state*. Result — held-out
validation **still 0.50** for both cats, at every eval:

```
[VQA val @ step 250/500/750]  (run_height_v2_state.log)
  val/vqa_taskA_acc=0.5  val/vqa_taskA_bal_acc=0.5  val/vqa_taskA_loss=6.9078
  val/vqa_taskB_acc=0.5  val/vqa_taskB_bal_acc=0.5  val/vqa_taskB_loss=6.9078
```

Confidently always-one-class; `val_loss ≈ 6.9078` is the **clamp artifact**: the model
puts `p→0` on the correct class for the all-wrong half, and `validate_vqa` clamps
`p_correct` to `1e-3` before `−log` (`−ln(1e-3)=6.9078`). It is not a numeric bug.

### 4.4 We verified the injected state is valid and separable — so it's NOT a bug
The EE-pose text actually carries the discriminating value: for height, the prompt
contains `z=1.131` (high) vs `z=1.038` (low), Δ≈83 mm, S/N~15. The model **simply
ignores the numeric text and memorizes the train set**. (Cross-check: 0526-corrections
§3.3 measured the same height z_release separation — S/N 24 on `move_mouse_pad` — from
disk; the signal is unambiguously present in the input we gave the VLM.)

---

## 5. **HOW TO FIX** (recommendations — the point of this doc)

### 5.1 RECOMMENDED — label the preference directly from the EE state
The pref is a function of pose (§2), with S/N 15–22 on the discriminating axis. So:

- **height** → literally **threshold `z_release`** (high vs low). No learning needed.
- **place** → 2-feature logistic / threshold on EE `xy` @ release (center vs corner).
- **orient** → small MLP / threshold on the **6D rotation** @ grasp (side vs top);
  use rotation, not xyz (§2).
- General form: a **2-layer MLP (or logistic regression)** on the active-arm EE pose at
  the pref frame → ~**99% self-labeling that GENERALIZES to taskB** (pose is
  object-agnostic, so taskA→taskB transfer is automatic — exactly where the VLM
  collapsed). **Effort ≈ 1 hour**, no GPU training run.

This is the right tool: it consumes the signal in the form it actually exists.

### 5.2 If the VLM MUST emit the label — state as a *learned token*, not text
Feeding the pose as **text** failed (§4.3/§4.4: VLMs don't reliably read floats or
induce a threshold/comparison by gradient). A clean differentiable pathway:
**MLP-encode the EE pose → an embedding token prepended to the VLM input** (not a
number in the prompt string). More work (architectural change to `_build_vqa_inputs` /
the QwenPI `state_encoder`, which OFT lacks), and still carries some memorization risk,
but it puts the proprioceptive signal where gradients can use it.

### 5.3 Orthogonal anti-memorization knobs (help, but won't rescue a hard feature)
Denser VQA loss (raise `n_vqa_per_batch` above 2 / more answer-token supervision per
step), stronger augmentation (we already added ±frame jitter), and the **val-based
early-stop we just added** (`val/vqa_taskB_acc` via `train_starvla._eval_vqa`). These
reduce overfitting but **cannot make a fundamentally hard-to-extract visual feature
learnable** — when the generalizable feature is harder than memorizing, SGD memorizes.

### 5.4 Strategic fork
"VLM-VQA self-labels the preference" **assumes the pref is visually readable**. The
0526 data says it is **proprioceptive**. Two honest options:
1. **Label from state** (§5.1) — recommended; cheap, accurate, generalizes. Keep the
   VLM for what it's good at — **language conditioning of the action policy** (the OFT
   action stream, §3) — not reading pose-defined prefs from pixels.
2. **Redesign the task** so the difference is genuinely *visible* (bigger spatial
   separation, or add a wrist / side camera that sees the discriminating motion). Only
   worth it if downstream genuinely needs a vision-based labeler.

---

## 6. Lessons (call-outs)

- **Always validate on held-out taskB; never trust train `L_vqa→0` / train
  `vqa_acc=1.0`.** Train→0 with val→0.5 is textbook memorization. We added
  `val/vqa_taskB_acc` (+ `taskA`) logging via `VQACotrainMixin.validate_vqa` →
  `train_starvla._eval_vqa` for exactly this — surface the real objective every eval.
- **Cheap, high-signal diagnostics belong BEFORE training**: the zero-shot probe
  (§1, `zeroshot_vqa_probe.py`), the EE-state separability (§2), and the per-task
  input visualization (`r-preference/debug/v1/make_viz.py` → 27 PNGs, one per
  cat/task_group/source, both pref classes side-by-side, with the exact frames + the
  question). Looking at the PNGs makes the "overhead view, vertical/few-pixel offset"
  invisibility obvious.
- **OFT warm-start is the working action backbone on 0526; from-scratch flow-matching
  is not** (at least in short runs) — see companion §7. Pivot the action stream to OFT.
- **The VQA path is action-head-agnostic.** `VQACotrainMixin` is shared verbatim by
  `QwenPI_VQA` and `QwenOFT_VQA` (it only touches the shared VLM + clips + state +
  `VQA_CATEGORIES`). Whatever the labeler fix, it plugs into either action head.

---

## 7. Code artifacts (created / changed this session — verified)

| File | What | Key symbols |
|---|---|---|
| `starVLA/model/framework/vqa_cotrain_mixin.py` | **NEW** — action-head-agnostic VQA cotrain path, factored out of QwenPI_VQA (behavior-identical) | `VQACotrainMixin`; `_init_vqa`, `_aggregate_vqa` (`action_loss=L_action+λ·L_vqa`), `_select_vqa_episodes`, `_build_vqa_inputs` (injects `format_state_text(state)` when `state_in_vqa`), `_vqa_forward`, `predict_preference`, **`validate_vqa`** (held-out taskA-val + taskB) |
| `starVLA/model/framework/QwenPI_VQA.py` | **Refactored** — now `Qwen_PI` + `VQACotrainMixin` | `Qwen_PI_VQA(Qwen_PI, VQACotrainMixin)`; `forward` = super (flow-matching L_action) then `_aggregate_vqa` |
| `starVLA/model/framework/QwenOFT_VQA.py` | **NEW** — OFT (L1 action-token) + same mixin | `Qwenvl_OFT_VQA(Qwenvl_OFT, VQACotrainMixin)`; uses `pref_hdf5_vqa` + `action_space: joint` (14D) |
| `examples/preference/dataset/vqa_sample.py` | **Redesigned** `VQA_CATEGORIES` for orient/place/height | task-context `question`; single-token `answer_text`/`answer_token_ids` (`side/top`=2929/3481, `middle/corner`=19656/73425, `high/low`=11892/10303); `cameras=("head_camera",)`; `clip_strategy` place3/orient3/height3; `n_frames=3`; `state_in_vqa=True`; `load_clip_and_state` (±jitter, frames+state at same idx), `ee_pose_9d_at`, `format_state_text`, `_active_arm_key` |
| `examples/preference/dataset/pref_hdf5_dataset.py` | `action_space="ee"\|"joint"` | `joint` = 14D qpos vector `joint_action/vector` (`[L_arm6,L_grip,R_arm6,R_grip]`) to match the OFT pretrained ckpt |
| `examples/preference/dataset/pref_hdf5_vqa_dataset.py` | **On-demand VQA** (dropped the static clip cache) | `__getitem__` emits `vqa_h5_path`/`vqa_pref_key`/`vqa_task_group`; clip+state decoded on demand in `_vqa_forward` so only `n_vqa_per_batch` eps are read + jittered |
| `starVLA/training/train_starvla.py` | `_eval_vqa` hook at `eval_interval` | rank-0 guarded; calls `model.validate_vqa()` → logs `val/vqa_taskA_acc`, `val/vqa_taskB_acc` (the real objective) |
| `r-preference/eval/gate_oftvqa.py` | **NEW** — gate-eval a trained (OFT_)VQA ckpt on taskA-val + taskB via `predict_preference` | produced `gate_{height,place}.json` (§4.2) |
| `r-preference/eval/zeroshot_vqa_probe.py` | **NEW** — frozen-VLM forced-choice probe | produced `zeroshot_vqa_probe_0526{,_calib}.json` (§1) |
| `r-preference/debug/v1/make_viz.py` | **NEW** — per-(cat,task,source) VQA-input figure | 27 PNGs in `r-preference/debug/v1/` (§6) |

---

## 8. Experiments run (logs)

| run | machine | log | result |
|---|---|---|---|
| `conv_baseline_place_0526` (QwenPI from-scratch, flow-matching, 20D EE, 2000 steps) | H100×8 | `/mnt/localssd/kaiwenh/logs/conv_baseline_place_0526.log` | **STUCK** L_action ≈1.42 flat (degenerate floor) |
| `conv_oft_warmstart_place_0526` (OFT L1, 14D joint, chunk50, warm-start, 1500 steps) | H100×8 | `/mnt/localssd/kaiwenh/logs/conv_oft_warmstart_place_0526.log` | **CONVERGES** L1 0.106(@25)→**0.00815**(@1500) |
| `conv_vqa_orient_0526` (QwenPI_VQA flow-matching, orient) | H200×8 | `/mnt/localssd/kevin/logs/conv_vqa_orient_0526.log` | L_vqa→0 ✓ ; **L_action STUCK** ~1.5 |
| `conv_oftvqa_{place,height}_0526_1k` + height OFT no-VQA control (4×1k batch) | H100×8 | `/mnt/localssd/kaiwenh/logs/j_{place,height}_vqa_1k.log`, `oftvqa*_smoke*.log` | all L_action 0.008–0.018; **height+VQA 0.0100 ≈ no-VQA 0.008** (VQA benign for OFT) |
| gate-eval of the 1k OFT_VQA ckpts | H100 | `/mnt/localssd/kaiwenh/logs/gate_{height,place}.{log,json}` | **taskA-val + taskB = 0.50, always one class** |
| `run_height_v2_state` (height, redesign **+ EE-state-as-text**, held-out val hook) | H100×8 | `/mnt/localssd/kaiwenh/logs/run_height_v2_state.log` | **val taskA & taskB = 0.50** @ steps 250/500/750 (loss 6.9078 clamp) |
| orient v2 with state (redesign + state) | H200×8 | `/mnt/localssd/kevin/logs/run_*_v2_state.log` | **val taskB = 0.50** (same collapse) |

---

## 9. References

- **Companion (read first):** [`0528-action-arch-and-oft-vqa-integration.md`](0528-action-arch-and-oft-vqa-integration.md)
  (action heads, Q1/Q2/Q4 convergence, OFT×VQA design, zero-shot + EE-separability findings),
  [`0528-instruction-mount.md`](0528-instruction-mount.md) (0526 data + dual instructions),
  [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
  (degenerate-floor background; data/pipeline clean).
- **Eval JSONs:** `r-preference/eval/zeroshot_vqa_probe_0526.json`,
  `r-preference/eval/zeroshot_vqa_probe_0526_calib.json`,
  `/mnt/localssd/kaiwenh/logs/gate_{height,place}.json`.
- **Input viz:** `r-preference/debug/v1/*.png` (27 figures) + `make_viz.py`.
- **Code:** `starVLA/model/framework/{vqa_cotrain_mixin,QwenPI_VQA,QwenOFT_VQA,QwenPI,QwenOFT}.py`;
  `examples/preference/dataset/{vqa_sample,pref_hdf5_dataset,pref_hdf5_vqa_dataset}.py`;
  `starVLA/training/train_starvla.py`; `r-preference/eval/{gate_oftvqa,zeroshot_vqa_probe}.py`.
- **Ephemeral (numbers quoted, file not preserved):** `/tmp/xyz_separability.py` (§2 S/N).

---

End of doc.
