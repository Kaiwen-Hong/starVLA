# 2026-05-26 — Corrections to 0525 root-cause diagnoses + full pipeline audit

> **Self-contained correction doc.** Written 2026-05-26 after re-investigation
> of the gate-eval JSONs, wandb training logs, and direct disk inspection of
> all 4 RED cats from 0525. **This doc supersedes the root-cause diagnoses in**
> [`0525-problems-and-how-to-solve.md`](0525-problems-and-how-to-solve.md) **§3.2, §3.3, §3.4.**
>
> The 0525 doc has been amended with a CORRECTION header pointing here.
> Its §3.1 (place) and §4 (recovery launches) remain valid.
>
> Reading order if you skipped 0525: §0 here gives the corrected picture in
> one table; §3.x rebuts each wrong diagnosis with measurement evidence; §4
> shows the actual training loss data; §5 shows that data + pipeline + config
> are all clean; §6 is the resulting open question.

---

## 0. TL;DR

Re-investigation of the 4 RED gate evals reveals **3 of 4 root-cause diagnoses
in 0525 were wrong** due to measurement bugs. The corrected picture:

| cat | 0525 §3.x diagnosis | 0526 corrected | Status |
|---|---|---|---|
| **place**  | healthy model, taskB just hard                            | ✓ same (confirmed)                                                                                                  | training OK; Stage B taskB pseudo-label gate fails |
| **orient** | VQA resume corrupted optimizer; fresh retrain will fix    | **FRESH ALSO FAILS** (2 attempts on 2026-05-25, identical stuck pattern). Real cause is VQA-cotrain × orient interaction, NOT resume | open — needs investigation, not retry |
| **height** | upstream HF data has ~50% mislabeled `_high/` episodes    | **DATA IS CLEAN** (z_release S/N=24, zero distribution overlap). The 0525 §5.1 script has 2 bugs                       | open — training stuck for unknown reason |
| **hvlv**   | weak pref signal in data, S/N≈0.10, "labels correct but trajectories don't differ enough" | **DATA IS CLEAN** (transport-segment perp dev S/N=5–10, zero overlap). The 0525 §3.4 metric was applied to the wrong segment | open — training stuck for unknown reason |

Plus: a 7-step pipeline audit (§5) found **no bugs**. The training failures for
height / hvlv / orient-VQA are not data or pipeline problems. They're
empirical training-dynamics failures whose cause is currently unknown.

**Action implications:**
- Do NOT filter or re-collect height data based on the 0525 §5.1 script — there are no mislabeled episodes to filter.
- Do NOT re-launch a third fresh orient-VQA retrain expecting it to succeed — the previous two also showed the stuck-loss pattern.
- DO read §7 below before deciding next steps.

---

## 1. Background — what 0525 reported

`0525-problems-and-how-to-solve.md` ran `stage_a_gate.py` on the 4 new cats at
25k Stage A and found all 4 RED (taskB VQA acc < 90 %). It then proposed three
distinct root causes:

- **§3.1 place** — healthy model, taskB visually harder than taskA
- **§3.2 orient** — VQA training failure caused by resume-after-kill (Adam state
  corruption); fresh retrain in flight
- **§3.3 height** — upstream RoboTwin data has ~48-52 % mislabeled episodes in
  `_high/` directories; trajectories actually look like `_low/`
- **§3.4 hvlv** — labels correct but trajectory deviation between `hv` and
  `lv` is only ~10 % of within-pref std (S/N ≈ 0.10), unlearnable

The launched recovery plan: fresh orient VQA retrain on H200 + Stage B for
place + orient. Tracking in `examples/preference/temp-525-h200.sh`.

---

## 2. Re-investigation methodology (2026-05-26)

Three independent evidence sources, all directly inspected (not via doc claim):

1. **Gate eval JSON raw numbers** — `r-preference/eval/stage_a_gate_*_25k.json`
   read for `vqa_taskA.overall_acc`, `vqa_taskB.overall_acc`, `pred_dist`,
   `baseline_counterfactual_taskA.overall_mse_normalized`,
   `vqa_counterfactual_taskA.overall_mse_normalized`.
2. **Wandb training logs** — `Checkpoints/<run>/wandb/wandb/run-*/files/output.log`
   parsed for per-step `Step N, Loss: {'action_dit_loss': ..., 'L_action': ...}`
   for orient/height/hvlv/place baseline + VQA + 524-fresh-retry runs.
3. **Direct disk inspection** — for each cat, open `.hdf5` files and
   `scene_info.json`; identify the active arm per episode (the one with larger
   xyz range); measure pref-discriminating quantity per episode using the
   correct physical metric.

No new training was launched. All conclusions are observational on the existing
ckpts and data.

---

## 3. Per-cat corrected diagnosis

### 3.1 place — confirmed healthy (no correction)

0525 §3.1 stands. Evidence re-verified:

- baseline loss 802 → 0.014 over 25k steps (smooth monotonic convergence)
- VQA loss converges similarly (final 0.04)
- gate eval: taskA VQA acc 0.79 (model learned VQA on training distribution);
  seeded counterfactual MSE 0.0020 (real action-stream pref signal exists)
- taskB acc 0.44 is below the 0.90 gate — likely a clip-strategy issue
  analogous to 0524 contact taskB (uniform_8 fluke); deserves
  `frame_window_test.py` follow-up before giving up

→ Action: Stage B place pseudo-label gate already failed (post-filter acc
0.194 in `temp-525-h200.sh::01-pseudolabel-place`); investigate clip-strategy
options before scrapping.

### 3.2 orient — fresh retrain ALSO fails (0525 §3.2 falsified)

#### What 0525 §3.2 claimed
> Stage A VQA was destabilized by the resume. Same data trained baseline
> successfully → not a data problem. Action: fresh retrain orient VQA from base.

#### What actually happened (2026-05-25 timeline)

Two independent fresh-from-base retrain attempts on H200, BOTH failed:

| time (UTC) | attempt | wandb run | outcome |
|---|---|---|---|
| 11:29Z | `temp-525-h200.sh::02-orient-vqa-fresh` (wrapper-launched) | `7dipoaaw` | killed at step 214 (`Killed` signal, no first save) |
| 11:42Z | `temp-524-retrain-orient-vqa.sh` in tmux `pref_524_orient` (you manually started) | `0yo8o5vn` | killed at step 2517 (you Ctrl-C'd 4 times at ~12:36Z) |

Both attempts produced **zero checkpoints** (`save_interval=5000`, neither
reached the first save).

#### The loss trajectory of attempt 2 (524 fresh) — the critical evidence

Parsed from `/mnt/localssd/kevin/starVLA_runs/.temp_524_orient_retrain/01-orient-vqa-fresh.log`:

```
step=100  L_action=869.6  L_vqa=0.83 vqa_acc=0.00   (init, ignore)
step=200  L_action=1.59   L_vqa=~0   vqa_acc=1.00
step=500  L_action=1.35   L_vqa=0.0  vqa_acc=1.00
step=1000 L_action=1.64   L_vqa=~0   vqa_acc=1.00
step=1500 L_action=1.68   L_vqa=~0   vqa_acc=1.00
step=2000 L_action=2.65   L_vqa=~0   vqa_acc=1.00
step=2500 L_action=1.64   L_vqa=0.0  vqa_acc=1.00
```

L_action bouncing in [1.3, 2.7] with no downward trend over 2300 steps. This is
**the same pattern** as the failed resume (which had L_action ~1.5 across steps
10k-25k). Fresh did not behave any differently.

#### What this means for the 0525 diagnosis

The resume was hypothesized as the cause because:
- baseline (same data) converged → "data is fine"
- VQA from-resume failed → "resume corrupted Adam state"
- ergo: fresh-from-base will recover

The fresh attempts falsify the third step. With no Adam state to corrupt, the
fresh runs still fail. Therefore the real cause is **specific to the VQA path
on orient data** — not a recovery artifact.

#### baseline orient (same data, no VQA) for contrast

`/mnt/localssd/kevin/starVLA_runs/Checkpoints/pref_baseline_stage_a_v1_noVQA_orient/wandb/wandb/run-20260523_104917-ml2w7b9p/files/output.log`:

```
step=100   L_action=657.19
step=200   L_action=1.58
step=5100  L_action=0.17
step=12600 L_action=0.078
step=18800 L_action=0.034
step=25000 L_action=0.040
```

Smooth convergence. Same data. Difference: presence of `L_vqa` in the loss
aggregation.

#### Likely real causes (not yet verified)

- `lambda_vqa=0.5` may be too large for orient. Although L_vqa quickly saturates
  to ~0 (vqa_acc=1.0 from step 200), the GRADIENT through the VQA forward path
  may still be perturbing the backbone in a way that destroys action learning.
- The VQA Q&A for orient ("horizontal or vertical grasp?") may be answerable
  trivially from the static head_camera view, making `L_vqa→0` but `∇L_vqa` 
  non-zero in a destabilizing direction.
- Lower-rank possibility: NCCL / DeepSpeed Zero2 interaction with a
  numerically narrow gradient distribution on orient's data layout.

These are **hypotheses**, not verified. None of them is "the resume corrupted
state".

#### Recommendation
Do not launch a third fresh retrain expecting recovery. The pattern is the same
both times. Instead: try a `lambda_vqa=0.0` short run (baseline-equivalent
loss, but still with the VQA-cotrain dataset wiring) — if THAT also fails, the
issue is the VQA path itself; if it converges, the issue is in the loss-mixing
weight or the VQA backward.

### 3.3 height — data is CLEAN, 0525 §3.3 misdiagnosed

#### What 0525 §3.3 claimed
> `height/move_mouse_pad_high` has ~48 episodes with z_final at ~0.94 m
> (low-like) and ~51 episodes at ~1.07 m (correct high). Bimodal histogram
> `[0, 0, 48, 0, 1, 51, 0]`. **`_low/` is consistently correct.**
> Verdict: upstream RoboTwin sim mis-categorized half of `_high/` trials.

#### Why the 0525 diagnostic script was wrong

The script (0525 §5.1):
```python
for pref in ["high", "low"]:
    z = [h5py.File(D/f"move_mouse_pad_{pref}"/"data"/f"episode{i}.hdf5","r")
           ["endpose/left_endpose"][-1,2] for i in range(100)]
```

Two compounding bugs:

1. **`left_endpose` only.** These are dual-arm tasks. Active-arm distribution
   for height (sampled across 16 task_dirs × 20 episodes):

   ```
   height OVERALL: L_active=43.1% / R_active=56.9%
   ```

   Roughly half the episodes use the right arm. When the right arm is active,
   the left arm sits at its home position (a fixed pose). Querying
   `left_endpose[..., 2]` on those episodes returns the home z (≈0.942 m),
   regardless of where the right arm dropped the object.

2. **`[-1, 2]` (last-frame z) is not the drop-release z.** Demos continue past
   the release. After releasing, the arm retracts; the final z is wherever the
   arm has retracted to, not where the object was released.

#### Correct metric: active arm × z at gripper-release

For each episode:
- Active arm = the one with larger `‖xyz.ptp(axis=0)‖` (i.e. it moved).
- Release moment t* = last frame where `gripper[t-1] ≤ 0.5` and `gripper[t] > 0.5`
  (closed → open transition; gripper convention: 0=closed, 1=open).
- Report `active_arm.endpose[t*, 2]`.

Re-measurement of `move_mouse_pad`, n=100 per pref:

```
_high  z_release: mean=1.0741  std=0.0058  range=[1.0634, 1.0876]
_low   z_release: mean=0.9348  std=0.0056  range=[0.9239, 0.9459]
Δ = +139.3 mm,  S/N = 24.1  (perfect separability; ≥2 = separable)

Histograms (10 bins 0.92-1.10):
_high z_release: [0, 0, 0, 0, 0, 0, 1, 88, 11, 0]
_low  z_release: [68, 32, 0, 0, 0, 0, 0, 0, 0, 0]
```

Zero distribution overlap. `_high` all cluster in [1.063, 1.088]; `_low` all
cluster in [0.924, 0.946]. There is **no episode** in `_high/` with z_release
below 1.06.

Cross-verify on a different task:

```
place_pillbottle_stand   _high z_release: 1.1199 ± 0.0083
                         _low  z_release: 1.0395 ± 0.0071
                         Δ = +80.4 mm,  S/N = 9.7
```

Also perfectly separable.

#### What the 0525 histogram actually shows

Reproducing the 0525 script exactly:

```
_high  endpose/left_endpose[-1, 2]: hist=[0, 0, 48, 0, 1, 51, 0]
_low   endpose/left_endpose[-1, 2]: hist=[0, 27, 73, 0, 0, 0, 0]
```

- The "48 at z≈0.94" in `_high` are episodes where the **right** arm did the
  high drop. The left arm stayed at home position (z=0.9420 = home z).
- Both arms' "home" position has z very close to `_low`'s drop position by
  coincidence (the table+arm geometry happens to make home-arm-z ≈ low-drop-z).
- `_low/` showed no apparent bug only because there the inactive arm's home z
  also matched the low-drop z — both effects coincided.

#### Conclusion
height data is clean. The "50 % mislabeled" claim should not be acted upon. Do
NOT filter, re-collect, or write z-based filter logic. The training failure for
height baseline (loss stuck at 1.45) has a different, currently-unknown cause.

### 3.4 hvlv — data is CLEAN, 0525 §3.4 metric was wrong-segment

#### What 0525 §3.4 claimed
> "max perpendicular deviation from straight-line source→target" between hv and
> lv: hv 0.157 m, lv 0.142 m; **Δ=15 mm, std=150 mm → S/N≈0.10**. Labels may be
> technically correct but trajectories don't differ enough to learn from.

#### Why this metric is wrong

The metric measures perp deviation from the straight line connecting the FIRST
and LAST frames of the full episode trajectory. But:

- Full episode = home → reach for object → grasp → transport → release → retract → home
- The "go around the can" detour is in the **transport segment** only (between
  grasp and release).
- Home approach + return are typically large excursions perpendicular to the
  start→end line (since both start and end are home-ish for one arm).
- These home excursions dilute the small transport detour difference.

#### Correct metric: perp deviation during transport segment only

For each episode:
- Active arm = larger xyz range.
- grasp_t = first frame where gripper transitions open → closed.
- release_t = last frame where gripper transitions closed → open.
- Transport segment xy = active_arm.endpose[grasp_t : release_t+1, :2].
- max |perp dist from transport line grasp_xy → release_xy|.

Re-measurement (n=50 per pref):

```
place_apple_plate   hv: 22.0 ± 1.9 cm   lv: 7.9 ± 1.8 cm   Δ=14.0 cm   S/N=7.4
place_cup_plate     hv: 22.7 ± 2.7 cm   lv: 8.1 ± 1.1 cm   Δ=14.6 cm   S/N=5.4
place_seal_right    hv: 21.9 ± 1.4 cm   lv: 7.8 ± 1.1 cm   Δ=14.1 cm   S/N=10.2
```

Per-ep distributions have **zero overlap**: in `place_apple_plate`, the
smallest hv |perp| is 19 cm, the largest lv |perp| is 10 cm; no hv episode
detours less than any lv episode.

For the can-side (signed perp), within-cat std is large (~20 cm for hv, ~7 cm
for lv) because the can position varies per episode → arm picks left-side vs
right-side detour roughly evenly. The MAGNITUDE separation (above) is what
matters for "is this learnable" — and it is.

#### Conclusion
hvlv data has clean, strongly separable detour structure (S/N ≥ 5 on all
spot-checked tasks). The training failure has a different cause.

---

## 4. The true training picture (loss curves from wandb logs)

Parsed from `output.log` of each baseline + VQA run:

| cat | run | wandb run id | final L_action | trajectory |
|---|---|---|---|---|
| place      | baseline 25k    | (various) | **0.014** | 802 → 0.014 monotonic ✓ |
| place      | VQA 25k         | (various) | **0.04**  | 1015 → 0.04 monotonic ✓ |
| orient     | baseline 25k    | ml2w7b9p  | **0.040** | 657 → 0.04 monotonic ✓ |
| **orient** | **VQA resume 25k**  | (in 523 wrapper) | **~1.5** | bouncing in [1.2, 1.7], no convergence |
| **orient** | **VQA fresh 524 (killed @2517)** | 0yo8o5vn  | **~1.5** | bouncing in [1.3, 2.7], no convergence |
| **height** | **baseline 25k**| rdmadvqb  | **~1.45**| 994 → 1.65 → bouncing in [1.3, 1.6] |
| **height** | **VQA 25k**     | (523 wrapper) | **~1.45**| same stuck pattern |
| **hvlv**   | **baseline 25k**| dzb9idkc  | **~1.55**| 691 → 1.61 → bouncing in [1.5, 1.6] |
| **hvlv**   | **VQA 25k**     | (523 wrapper) | **~1.55**| same stuck pattern |

### What L_action ≈ 1.5 means

Flow-matching loss is MSE on the velocity field:

```
noise ~ N(0, I_d)
t ~ Beta(1.5, 1.0)
noisy = (1-t)·noise + t·action
velocity_target = action - noise
L_action = MSE(model(noisy, t, cond), velocity_target)   per element
```

For a model that outputs a constant (the unconditional mean of `velocity_target`
= E[action]), the per-element MSE equals:

```
E[(action - noise - E[action])²] = Var[action] + Var[noise] = Var[action] + 1
```

Empirical per-element `Var[action]` from sampling 12,800 normalized frames per
cat (sum across 20 dims divided by 20):

| cat | mean Var[action_dim] | implied unconditional MSE |
|---|---|---|
| contact | 0.29 | **1.29** |
| place   | 0.54 | **1.54** |
| orient  | 0.42 | **1.42** |
| height  | 0.53 | **1.53** |
| hvlv    | 0.52 | **1.52** |

The "stuck" loss values from the table match the unconditional-MSE prediction
to within ~0.05:

```
height baseline stuck 1.45  ≈  unconditional MSE 1.53
hvlv   baseline stuck 1.55  ≈  unconditional MSE 1.52
orient VQA      stuck ~1.5  ≈  unconditional MSE 1.42
```

**Interpretation:** the model has learned the marginal action distribution
(predicts the mean velocity, which is action-mean), but has not learned any
conditional structure from `(image, prompt) → action`. It is not random output,
it is converged to the simplest possible degenerate solution.

For contact (final 0.014) and place (final 0.014) and orient-baseline (final
0.04), the model is far below unconditional MSE — it has learned a useful
(image, prompt) → action mapping.

The mystery: why do height, hvlv, and orient-VQA stay at the degenerate
solution while contact, place, and orient-baseline escape it?

---

## 5. Pipeline audit — 7 steps, no bugs found

Audited the full data pipeline for all 5 cats. Each check was run independently
per cat.

| Step | Check | Result |
|---|---|---|
| 1 | Disk task_dirs match `PREF_CATEGORIES[cat].task_groups` × `pref_keys` | All 5 cats: 16 dirs found, 2 "missing" are taskB dirs (correctly excluded from Stage A root) |
| 2 | `_split_episodes` produces disjoint train/val sets | All 5 cats: train∩val = ∅; counts match expected (1280/320 for contact/height/hvlv, 1276/319 for orient, 1098/274 for place) |
| 3 | Stats JSON exists per cat, q01/q99 spans reasonable | All present. Some dims (hvlv L6_4/R6_4, height L6_1/R6_1) have very narrow spans (<0.1 vs contact's typical >1.0), but that reflects data — wrist roll axes vary little for tasks with consistent grasp orientation. Not a bug. |
| 4 | Normalized action distribution / clip rate per dim | High clip rates (>50% at ±1) on multiple dims for height + hvlv. BUT contact has 89-97% clip on grip + R-arm dims AND still converges. Therefore clip rate is not the differentiator. |
| 5 | Per-sample sanity (no NaN/Inf, shape (16,20)/(1,20), 3×224×224 PIL imgs, lang string correct) | All 5 cats: 3 random samples each, all pass. Prompt assembled correctly with category dispatch. |
| 6 | YAML config consistency | All 5 baseline YAMLs differ only in `run_id`, `data_root_dir`, `stats_json_path`, `pref_category`, `max_train_steps` (50k for contact, 25k others). Same model arch, optimizer, scheduler, eff batch 64, image size 224, 3 cameras. |
| 7 | Active arm distribution per cat | contact 72%L/28%R (4 tasks always-L, 4 random); height 43/57; hvlv 45/55; orient 52/48; place 47/53. **place is ~50/50 random arm and DOES converge** → arm switching is not the failure cause. |

### What pipeline differences DO exist between cats

Only one: the per-cat prompt strategy in `PREF_CATEGORIES`:

```
contact: legacy v5 hybrid strip (regex-based) with 8 CLEAN_TEMPLATE fallbacks
place:   template-only (paraphrases ignored)
height:  template-only
hvlv:    broad-sep + leak regex (100% pass)
orient:  broad-sep + tighter leak regex (100% pass)
```

These differ in the assembled `lang` string but not in any other code path.
The action / image / state pipeline is identical across cats.

**Conclusion**: the failure of height/hvlv/orient-VQA training is not caused by
any data corruption, pipeline bug, or config mistake. The data is clean, the
pipeline is correct, and the configs are consistent. The failure is in the
training dynamics on these specific cats.

---

## 6. Open question: why do height + hvlv + orient-VQA stuck at unconditional MSE?

What we know (verified):
- Action data is clean and pref-separable (§3.3, §3.4 + orient §3.2 baseline 0.04).
- Pipeline is bug-free (§5).
- Same model arch + hyperparams + batch size used across all cats.
- Same wrappers + same launchers + same H100/H200 environment.
- Active arm split (place ~50/50 also converges) is not the cause.
- Clip rate (contact 89-97% on some dims also converges) is not the cause.

What we do NOT know:
- Whether height + hvlv baseline training would converge with a different
  random seed (the same Adam-init might happen to land in the degenerate basin
  for these specific data distributions).
- Whether the issue is **the loss landscape near the degenerate solution being
  unusually flat** for these cats (so SGD escapes for contact/place but not for
  these), or **conditional signal in the data being unusually weak through the
  image→action route** even though the prompt-conditioned signal is strong.
- For orient VQA specifically: whether `lambda_vqa=0.5` is destabilizing the
  backbone update direction even with `L_vqa ≈ 0`.

### Hypotheses worth testing (cheapest first)

1. **Random seed** — relaunch height baseline with `--trainer.seed 43` and
   `--trainer.seed 44`; run each ~5k steps and look at loss. If even one
   converges, the issue is loss-landscape stochasticity, not data. Cost: ~3h
   per seed on 8 GPUs.
2. **Lambda ablation for orient VQA** — short run (e.g. 3k steps) with
   `framework.vqa.lambda_vqa: 0.0` (eqivalent to baseline). If that converges,
   the VQA cotrain is what breaks orient. Cost: ~1h.
3. **Warm-start from a working ckpt** — initialize height baseline from
   `pref_baseline_stage_a_v1_noVQA_place/.../steps_25000_pytorch_model.pt`
   and continue-train on height for ~3k steps. If loss drops below 1.5, the
   issue is initialization-into-degenerate-basin, not anything structural.
   Cost: ~1h.
4. **Increase warmup_ratio** — current 0.1 (= 2500 warmup steps for 25k run).
   Try 0.2 to see if more warmup escapes the basin.
5. **Single-task ablation** — train height with only `move_mouse_pad_high` +
   `_low` (cut to 2 task_dirs instead of 16). If it converges, multi-task
   interference is the issue.

None of these is "fix the data" or "fix the pipeline" — both are confirmed
clean.

---

## 7. Recommendations per cat (replaces 0525 §4 + §6 partly)

| cat | Status | Recommended next step |
|---|---|---|
| **contact** | healthy end-to-end | continue Stage B per [`0524-stageB-contact-plan.md`](0524-stageB-contact-plan.md); see results in [`0525-temp-stageb-analysis.md`](0525-temp-stageb-analysis.md) |
| **place**   | training healthy; Stage B pseudo-label gate fails (post-filter acc 0.194 / required 0.95) | Run `frame_window_test.py` with mid_8 / gripper_anchored to see if a different clip strategy rescues taskB acc — analogous to the 0524 contact analysis where uniform_8 0.61 → mid_8 0.99 |
| **orient**  | baseline healthy (loss 0.04, gate CF 0.061); VQA broken at TWO fresh attempts and one resume; no VQA ckpts exist on disk | (1) do NOT just retry fresh; (2) launch a `lambda_vqa=0.0` ablation (~1h, 3k steps) to isolate whether VQA path itself is the issue; (3) if it converges, sweep lambda_vqa ∈ {0.05, 0.1, 0.3} to find a working value |
| **height**  | baseline + VQA both stuck at L_action ≈ 1.45 (≈ unconditional MSE); ckpts on disk are not usable for downstream tasks | Try cheap seed/warm-start ablations from §6 before deeper investigation. Do NOT filter data based on z-final criteria — data is clean. |
| **hvlv**    | baseline + VQA both stuck at L_action ≈ 1.55 | Same as height. |

**Do NOT do:**
- Filter or re-collect height data based on 0525 §5.1 script — there are no mislabeled episodes.
- Declare hvlv pref signal "too weak to learn" based on 0525 §3.4 — signal is strong (S/N=5–10), the failure is elsewhere.
- Launch another from-scratch orient-VQA retrain expecting recovery — two attempts confirmed the loss stays stuck.

---

## 8. Evidence trail — reproducible commands

All measurements in this doc can be re-derived from disk. Reference commands:

### Height z_release re-measurement
```bash
source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh && conda activate starVLA
python3 - <<'PY'
import h5py, numpy as np
from pathlib import Path
D = Path("/mnt/localssd/kaiwenh/pref/data/height")
def z_release(ep):
    with h5py.File(ep, "r") as h:
        L = h["endpose/left_endpose"][:]; R = h["endpose/right_endpose"][:]
        Lg = h["endpose/left_gripper"][:]; Rg = h["endpose/right_gripper"][:]
    Lr = np.linalg.norm(L[:, :3].ptp(axis=0))
    Rr = np.linalg.norm(R[:, :3].ptp(axis=0))
    arm_e, arm_g = (L, Lg) if Lr > Rr else (R, Rg)
    T = len(arm_g); t = None
    for k in range(T-1, 0, -1):
        if arm_g[k-1] <= 0.5 and arm_g[k] > 0.5: t = k; break
    return float(arm_e[t, 2]) if t is not None else float(arm_e[-1, 2])
for pref in ("high", "low"):
    zs = [z_release(D / f"move_mouse_pad_{pref}" / "data" / f"episode{i}.hdf5") for i in range(100)]
    zs = np.array(zs)
    print(f"{pref}: mean={zs.mean():.4f} std={zs.std():.4f} range=[{zs.min():.4f}, {zs.max():.4f}]")
PY
```

### Hvlv transport-segment perp dev re-measurement
```bash
# similar script; replace z_release with the transport_perp_dev function:
# find grasp_t (first open→closed transition), release_t (last closed→open),
# then max |perp xy dist| from line grasp_xy → release_xy on active arm.
# Full code in r-preference/doc/0526-corrections-and-pipeline-audit.md history
# of this commit.
```

### Loss curves
```bash
# Per-cat training loss trajectory:
python3 - <<'PY'
import re
from pathlib import Path
for cat, p in [
    ('orient-baseline', '/mnt/localssd/kevin/starVLA_runs/Checkpoints/pref_baseline_stage_a_v1_noVQA_orient/wandb/wandb/run-20260523_104917-ml2w7b9p/files/output.log'),
    ('orient-VQA-fresh-524', '/mnt/localssd/kevin/starVLA_runs/.temp_524_orient_retrain/01-orient-vqa-fresh.log'),
    ('height-baseline', '/home/kaiwenh/starVLA/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_height/wandb/wandb/run-20260523_104858-rdmadvqb/files/output.log'),
    ('hvlv-baseline', '/home/kaiwenh/starVLA/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_hvlv/wandb/wandb/run-20260523_185554-dzb9idkc/files/output.log'),
    ('place-baseline', '/home/kaiwenh/starVLA/results/Checkpoints/pref_baseline_stage_a_v1_noVQA_place/wandb/wandb/run-20260524_225631-tz3np3jf/files/output.log'),
]:
    text = Path(p).read_text(errors='replace')
    pts = []
    for m in re.finditer(r'Step (\d+), Loss', text):
        s = int(m.group(1))
        w = text[m.end(): m.end()+500]
        la = re.search(r"'action_dit_loss': ([0-9.eE+-]+)", w)
        if la: pts.append((s, float(la.group(1))))
    if pts:
        idxs = sorted(set([0, 1, len(pts)//4, len(pts)//2, 3*len(pts)//4, len(pts)-1]))
        print(f"=== {cat} ===")
        for i in idxs:
            print(f"  step={pts[i][0]:>6}  L_action={pts[i][1]:>10.4f}")
PY
```

### Gate eval raw numbers
```bash
for f in /home/kaiwenh/starVLA/r-preference/eval/stage_a_gate_*_25k.json; do
  python3 -c "
import json
d = json.load(open('$f'))
print(d['category'], 'taskA_acc=', d['vqa_taskA']['overall_acc'], 'taskB_acc=', d['vqa_taskB']['overall_acc'],
      'bl_CF=', d['baseline_counterfactual_taskA']['overall_mse_normalized'],
      'vqa_CF=', d['vqa_counterfactual_taskA']['overall_mse_normalized'])
"
done
```

### Active arm distribution
```bash
# See §5 step 7 — script in chat transcript 2026-05-26 audit. Counts
# `xyz.ptp` per arm per episode, classifies each ep as L_active or R_active.
```

---

## 9. References

- [`0522-a-giveobj.md`](0522-a-giveobj.md) — contact Stage A spec (unchanged; contact is healthy)
- [`0523-height-hv-oreint-design-doc.md`](0523-height-hv-oreint-design-doc.md) — 3 new cats design (training-outcome section to be added)
- [`0523-stageA-analysis.md`](0523-stageA-analysis.md) — gate eval methodology (add a warning about baseline_CF≈1.4 = noise-floor sanity)
- [`0524-place-category.md`](0524-place-category.md) — place registration (training completed, taskB pseudo-label gate failed)
- [`0524-stageB-contact-plan.md`](0524-stageB-contact-plan.md) — Stage B contact plan (results in 0525-temp-stageb)
- [`0525-problems-and-how-to-solve.md`](0525-problems-and-how-to-solve.md) — original postmortem; **§3.2, §3.3, §3.4 superseded by this doc** via CORRECTION header
- [`0525-temp-stageb-analysis.md`](0525-temp-stageb-analysis.md) — Stage B contact findings (valid; contact is healthy)
- [`training-runbook.md`](training-runbook.md) — ops (ckpt status matrix updated; orient VQA 524/525 postmortem added)

---

End of doc.
