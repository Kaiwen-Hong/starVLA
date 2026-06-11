# 2026-05-28 — Action architectures, 0526 convergence experiments, OFT×VQA integration

> **Self-contained working doc** written during an autonomous session (user away ~1h).
> Goal: keep the moving pieces straight. Covers (1) the three action-head architectures
> in play, (2) the convergence experiments launched on 0526 + where their logs are,
> (3) how VQA cotrain attaches and a concrete design to integrate it into QwenOFT,
> (4) findings so far (zero-shot probe + EE-state separability), (5) open questions.
>
> Companion: [`0528-instruction-mount.md`](0528-instruction-mount.md) (0526 data + dual
> instructions), [`0522-a-giveobj.md`](0522-a-giveobj.md) (Stage A spec),
> [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
> (why old cats were stuck).

---

## 0. TL;DR of what's running (2026-05-28 ~08:43 UTC)

| # | Question | Run | Machine | Log | Answers |
|---|---|---|---|---|---|
| Q1 | Does **QwenPI from-scratch** converge on 0526? | `conv_baseline_place_0526` (flow-matching, 20D EE, 2000 steps) | H100 ×8 | `/mnt/localssd/kaiwenh/logs/conv_baseline_place_0526.log` | "is 0526 trainable + does flow-matching work" |
| Q2 | Does **OFT continue-train** (warm-start) work? | `conv_oft_warmstart_place_0526` (L1, 14D joint, chunk50, from `RoboTwin2-All` step140k) 1500 steps | H100 ×8 (after Q1) | `/mnt/localssd/kaiwenh/logs/conv_oft_warmstart_place_0526.log` | "does pretrained warm-start converge" |
| Q4 | Does **VQA cotrain** L_vqa/L_action behave on 0526? | `conv_vqa_orient_0526` (QwenPI_VQA, orient = the prev-stuck cat) 1500 steps | H200 ×8 | `/mnt/localssd/kevin/logs/conv_vqa_orient_0526.log` | "does L_vqa drop; does 0526 unstick orient L_action" |
| Q3 | Can **OFT integrate with VQA**? | design only (this doc §3) | — | — | integration recipe below |

Wrappers: H100 `run_oft_after_baseline.sh` (waits for baseline PID, then launches OFT).
H200 `/tmp/run_vqa_orient_h200.sh`. All caches forced to SSD (HF/torch/triton/tmp).

---

## 1. The three action-head architectures (verified from code)

All share the same `Qwen3-VL-4B-Instruct` backbone + the same sample-dict contract
(`image`, `lang`, `action`, optional `state`). They differ in the **action head** and **loss**.

### 1.1 QwenPI (baseline + VQA main-method) — flow-matching DiT
`starVLA/model/framework/QwenPI.py`. Action head =
`LayerwiseFlowmatchingActionHead` (`LayerwiseFM_ActionHeader.py`):
- Conditions the DiT on the **last 36 VLM hidden layers** (layerwise cross-attn) + current `state`.
- Loss = rectified-flow velocity MSE: `noise~N(0,I)`, `t~Beta(1.5,1)→(s−x)/s`,
  `noisy=(1−t)·noise+t·a`, target `v=a−noise`, `L=MSE(pred,v)`. `repeated_diffusion_steps=2` (hardcoded).
- Inference = 4-step Euler integration from noise.
- Action rep in pref: **20D EE** (`[L_xyz, L_6Drot, L_grip, R_xyz, R_6Drot, R_grip]`), chunk 16.

### 1.2 QwenOFT — action-token query + L1 regression (OpenVLA-OFT style)
`starVLA/model/framework/QwenOFT.py`. Action head = `MLP_ActionHeader.get_action_model`.
- Appends `🔍 × chunk_len` action tokens to the prompt:
  `" Please predict the next {chunk_len} robot actions: <action>🔍🔍…<action>."`
- Takes the **last hidden layer**, gathers the action-token embeddings as queries
  (`_gather_action_token_embeddings`), MLP-decodes them → actions in ONE pass (no diffusion).
- Loss = **L1** (`nn.L1Loss`) between pred and target chunk. **Does NOT consume `state`.**
- RoboTwin recipe (and the pretrained ckpt): **14D joint qpos**, chunk **50**, 3 cams.
- Pretrained ckpt `StarVLA/Qwen3-VL-OFT-RoboTwin2-All` (step140k, 88% RoboTwin avg) — same
  embodiment as our data (our 0526 is RoboTwin-built). On SSD:
  `/mnt/localssd/kaiwenh/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt`.

### 1.3 VQA cotrain (QwenPI_VQA) — auxiliary LM-head classification
`starVLA/model/framework/QwenPI_VQA.py` (subclasses QwenPI). On top of the action path:
- A **separate** VLM forward over an 8-frame clip + a binary question; CE on a single
  answer token. `n_vqa_per_batch=2` (dedup batch episodes, take first 2). `λ_vqa=0.5`.
- Aggregation INSIDE `forward`: `action_loss = L_action + λ_vqa·L_vqa` (trainer reads `action_loss`).
- The VQA machinery (`_select_vqa_episodes`, `_build_vqa_inputs`, `_vqa_forward`,
  `predict_preference`) only touches `qwen_vl_interface` (the LM head) + the clip cache +
  `VQA_CATEGORIES`. **It is independent of the action head.**

---

## 2. Why the action rep / head matters here

- **The preference signal lives in proprioception (EE pose), not head-cam pixels** (see §4).
  So whichever head we use, conditioning richness on the prompt/state is the crux.
- QwenPI uses `state` (current frame, 20D) + 36-layer cross-attn; QwenOFT uses neither
  `state` nor layerwise (just last-hidden action-token queries). For pref-following this
  matters: QwenPI has a `state_encoder` we could feed EE pose to; QwenOFT currently does not.

---

## 3. OFT × VQA integration design (the deliverable)

**Feasible and clean** — because the VQA path is action-head-agnostic. Recipe, mirroring
how `QwenPI_VQA` extends `QwenPI`:

### 3.1 New framework `QwenOFT_VQA(Qwenvl_OFT)`
```
class Qwenvl_OFT_VQA(Qwenvl_OFT):
    __init__: super().__init__(); then copy QwenPI_VQA.__init__'s VQA setup block
        (reads framework.vqa.{category,lambda_vqa,n_vqa_per_batch}; sets
         self._vqa_question / _vqa_id_A/_B from VQA_CATEGORIES[cat]).
    forward(examples):
        action_out = super().forward(examples)      # OFT L1 action path
        L_action   = action_out["action_loss"]
        L_vqa, log = self._vqa_forward(examples)     # REUSE QwenPI_VQA's method verbatim
        total = L_action + self.lambda_vqa * L_vqa
        return {"action_loss": total, "log/...": ...}
```
`_select_vqa_episodes`, `_build_vqa_inputs`, `_vqa_forward`, `predict_preference` can be
**lifted unchanged** from `QwenPI_VQA` (they only use `self.qwen_vl_interface` + the clip
cache). Cleanest: factor them into a `VQACotrainMixin` that both `QwenPI_VQA` and
`QwenOFT_VQA` inherit, so there's one copy.

### 3.2 Dataset
Use `pref_hdf5_vqa` with `action_space: joint` (the joint mode added 2026-05-28 to
`pref_hdf5_dataset.py`) so the action stream is 14D joint (OFT) while the VQA clip cache
is built exactly as today. The VQA sample fields (`vqa_episode_id`, `vqa_pref_key`,
`vqa_task_group`) are already emitted by `PrefHDF5VQADataset.__getitem__` regardless of
action rep.

### 3.3 Caveats / decisions
- **Shared-backbone coupling**: L_vqa gradient flows into the same VLM the action-token
  queries read from. This is the suspected orient-VQA destabilizer (0526 §3.2). With OFT's
  L1 head the dynamics differ from flow-matching — Q4's result on orient informs whether
  this coupling is still a problem.
- **Token budget**: OFT already appends `chunk_len(=50)` action tokens to the action prompt;
  the VQA forward is a *separate* prompt (clip + question, no action tokens) so no conflict.
- **State for VQA**: per the redesign (below), feed EE pose to the VQA. For OFT, the action
  head ignores state, but the VQA forward can still take pose-as-text or a state token.

### 3.4 Effort
~½ day: add the mixin + `QwenOFT_VQA` registration + a YAML. No change to the OFT action
path or the VQA cache. Gate: Q2 (OFT converges) AND Q4 (VQA L_vqa drops) must both pass first.

---

## 4. Findings already in hand (2026-05-28, pre-training)

### 4.1 Zero-shot base VLM ≈ chance on all 3 prefs from head-cam pixels
`r-preference/eval/zeroshot_vqa_probe_0526{,_calib}.json`. Calibrated balanced acc:
orient 0.55/0.52, place 0.61/0.62, height 0.52/0.40 (taskA/taskB). Strong per-class
priors; no real image-reading. → a frozen base VLM **cannot** self-label these; training
(VQA cotrain) is needed.

### 4.2 EE pose IS cleanly separable (the signal is proprioceptive)
`/tmp/xyz_separability.py` on 0526 (S/N, >2 = separable):
- height: z@release S/N **3.8 (taskA) / 20.8 (taskB)** — xyz(z) alone perfect.
- place : x@release S/N **4.2 / 3.8** — xyz alone works.
- orient: xyz(z) S/N 4.2/11.7 (usable) but **rotation S/N 9.0/11.3** (2× better) → feed
  full EE pose (xyz+6D) for orient, not just xyz.
→ **Feeding EE state into the VQA (and ideally the policy) is the high-leverage change.**

### 4.3 Action loss implementation is correct
`LayerwiseFM_ActionHeader.py:303-355` is textbook flow-matching; contact/place/orient-baseline
converged on old data with it. "Loss won't drop" on some old cats = data/dynamics, not a loss bug.

---

## 5. Open questions (what the runs will tell us)

1. **Q1** baseline place 0526: loss should fall ~1.6→<0.1 (like old place). If yes → 0526 trainable.
2. **Q2** OFT warm-start: does L1 fall fast from the pretrained init? Confirm ckpt loaded
   (grep "Loaded pretrained checkpoint" + key match). If yes → warm-start is a viable route
   to skip from-scratch instability.
3. **Q4** VQA orient 0526: (a) L_vqa should drop <0.05 within ~500 steps; (b) does L_action
   converge (old orient-VQA was STUCK ~1.5)? If L_action now drops on 0526 → re-collection
   fixed the orient-VQA failure. **Note**: L_vqa dropping (on taskA-train) is necessary but
   NOT sufficient for taskB — prior runs hit train vqa_acc=1.0 yet taskB acc 0.44 (shortcut,
   not transfer). To answer "does L_vqa↓ imply taskB↑" we must gate-eval the saved ckpt on
   taskB (`stage_a_gate.py` / `frame_window_test.py`).

---

## 6. Pending redesign (separate from the convergence checks)
Per 2026-05-28 discussion (not yet implemented): VQA question gets **task context** in the
QUESTION ("we are picking… side or top?") but answer stays a **single discriminative token**
(`side/top`, `middle/corner`, `high/low`) to keep the binary-classifier + Stage-B logit-gap
calibration; **head-cam only** (wrist confirmed useless → revert place's active_wrist);
**feed EE pose** into the VQA. These apply to both QwenPI_VQA and QwenOFT_VQA.

---

## 7. RESULTS (2026-05-28 ~09:25 UTC)

**Headline: from-scratch flow-matching (QwenPI) is STUCK at the degenerate floor on
0526; warm-started OFT converges cleanly on the SAME data.**

| Q | Run | Trajectory | Verdict |
|---|---|---|---|
| **Q1** QwenPI from-scratch (place, flow-matching 20D EE) | step25=1743→ flat **1.4–1.6 for all 2000 steps** (final 1.42) | NO convergence in 2000 steps — sits at unconditional-mean floor (≈Var[a]+1). | ✗ stuck |
| **Q2** OFT warm-start (place, L1 14D joint, ckpt loaded ✓) | L1 0.41→0.20→0.12→**0.05 @ step225** (running to 1500) | converges fast from pretrained init. | ✓ works |
| **Q4** VQA cotrain (orient, flow-matching) | **L_vqa 1.31→~0 by step50, vqa_acc=1.0**; **L_action STUCK ~1.4–1.6** all 1500 | VQA aux learns instantly; action stream stuck (same as Q1). | L_vqa ✓ / L_action ✗ |

### Interpretation
- **0526 data IS learnable** — OFT learns place to L1 0.05. So Q1/Q4's stuck L_action is
  NOT a data problem; it's the **from-scratch flow-matching path** failing to escape the
  degenerate basin (consistent with the old height/hvlv stuck cats — now even *place* and
  *orient*, which converged on old data, are stuck on 0526 from scratch).
- **OFT warm-start is the working route.** Caveat: OFT differs from QwenPI in 3 ways
  (warm-start, L1-vs-flow-matching, 14D-joint-vs-20D-EE) — clean attribution needs ablations
  (OFT-from-scratch; QwenPI-warm-start), but the *practical* result is unambiguous: OFT
  converges, QwenPI-from-scratch does not.
- **Does L_vqa↓ ⇒ taskB works? NO (necessary, not sufficient).** L_vqa→0 + vqa_acc=1.0 is
  *train-set memorization* of the binary VQA; prior runs hit train acc 1.0 yet taskB 0.44
  (shortcut, not transfer). Also here the action policy itself is degenerate (Q4 L_action
  stuck), so the end-to-end pref-following policy is not usable regardless. To answer taskB
  concretely: gate-eval the saved VQA ckpt (`conv_vqa_orient_0526/checkpoints/steps_1500_pytorch_model.pt`
  on H200) with `frame_window_test.py` on orient taskB (`move_can5_away_{0,90}`).

### Caveats
- Q1/Q4 are 2000/1500-step quick checks; the real runs used 25k steps + 2500 warmup. A late
  escape isn't fully ruled out — but the trajectories are FLAT (not the old "monotonic from
  1.4" drop), and OFT converging in <200 steps on the same data shows the signal is there.
- Q1/Q4 used a shortened warmup (200/150) for the quick check.

### Implication for next steps
1. **Pivot the action backbone to OFT (warm-start)** — it converges where from-scratch QwenPI doesn't.
2. **Integrate VQA into OFT** per §3 (`QwenOFT_VQA` mixin) — VQA path is head-agnostic.
3. Before trusting QwenPI-from-scratch anywhere, run the ablations (longer steps + full warmup;
   QwenPI warm-start) to confirm whether it's *fundamentally* stuck or just slow.

---

End of working doc.
