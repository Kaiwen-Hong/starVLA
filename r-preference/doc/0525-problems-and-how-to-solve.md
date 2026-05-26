# 2026-05-25 — Stage A gate eval RED across 4 new cats: root cause + recovery

> **CORRECTION 2026-05-26** — §3.2, §3.3, §3.4 contain root-cause diagnoses that
> have since been falsified by direct measurement. The summary tables in §0 and
> §3 should NOT be acted upon as written. Specifically:
>
> - **§3.2 orient** "VQA resume corrupted optimizer; fresh retrain will fix" —
>   **fresh ALSO fails**. Two from-scratch retrains on 2026-05-25
>   (525 wrapper + 524 manual tmux) showed identical stuck-loss patterns to
>   the failed resume. Real cause: VQA-cotrain × orient interaction, NOT resume.
> - **§3.3 height** "upstream HF data 50% mislabeled" — **data is CLEAN**.
>   The §5.1 diagnostic script uses `endpose/left_endpose[-1, 2]`, which has
>   two compounding bugs (left arm only; last-frame not release-frame). Correct
>   metric (active arm × z at release): _high z=1.074±0.006 vs _low z=0.935±0.006,
>   Δ=139 mm, S/N=24, zero distribution overlap.
> - **§3.4 hvlv** "weak signal S/N≈0.10" — **data is CLEAN**. The metric was
>   applied to full-trajectory perp dev (which includes home approach + return
>   excursions diluting the detour signal). Correct metric (transport-segment
>   perp dev between grasp_t and release_t): hv 22 cm vs lv 8 cm, S/N=5–10,
>   zero per-episode overlap.
>
> See [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
> for full evidence and corrected per-cat diagnosis.
>
> **What in this doc still holds:** §3.1 (place healthy), §4 (the launch
> sequence is a factual log of what was run), §5 (commands are correct as
> commands; only the height-mislabel script in §5.1 is misleading), §7
> (references). The text below is preserved AS-WRITTEN for historical
> traceability (so future readers can see how the wrong diagnosis came about).
> Do NOT act on §3.2 / §3.3 / §3.4 root-cause claims without reading
> [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) first.

---

> Self-contained postmortem. Reading this should let you know:
> - What happened (gate eval results),
> - Three distinct root causes (one per affected cat),
> - What we're doing about each, and
> - The exact commands / paths to reproduce or extend.
>
> Written 2026-05-25 after ~3h of focused debugging.

---

## 0. TL;DR

After running `stage_a_gate.py` on all 4 new cats (height/hvlv/orient/place at
25k Stage A) on H200, **all 4 cats showed RED gate** (taskB VQA acc < 90%).
But the underlying problems are **three different**:

| cat | gate result | seeded CF | root cause | recovery path |
|---|---|---:|---|---|
| **place** | taskB acc 0.44 (RED) but model fundamentally healthy | 0.0020 ✓ | normal model — taskB just visually harder than taskA | proceed to Stage B (with awareness) |
| **orient** | taskB acc 0.50, all-constant pred | 0.0018 baseline / **0.0000 VQA** | **VQA-only training failure** — resume after OOM kill corrupted optimizer / LR schedule. Same data trained the baseline ckpt fine. | **fresh retrain VQA** from base ckpt (in flight 2026-05-25) |
| **height** | taskB acc 0.50, all-constant pred | 0.0000 (both bl + vqa) | **upstream data corruption**: 48–52% of episodes in `_high/` dirs have actions that look like `_low/` (z_final histogram is bimodal at 50/50). | **user is checking data**; if confirmed, filter mislabeled eps + retrain |
| **hvlv** | taskB acc 0.50, all-constant pred | 0.0000 (both bl + vqa) | **weak pref signal in data**: mid-trajectory deviation differs by 0.015 m between hv vs lv, but per-episode std is 0.15 m → S/N ≈ 0.10. Even correct labels can't be learned at this S/N. | **user is checking data**; if confirmed, need either new data with stronger pref or a different probe axis |

Currently in flight on H200 (nohup, ~16h):
1. pseudo-label place taskB (will likely keep few samples per filter threshold)
2. **orient VQA fresh retrain** 25k steps
3. pseudo-label orient taskB
4. Stage B place: main + B0
5. Stage B orient: main + B0

For height/hvlv, user is handling data-side investigation separately.

---

## 1. What happened (timeline)

### 1.1 Stage A finished

By 2026-05-25 07:55 UTC, all 5 cats (contact / height / hvlv / orient / place)
had Stage A baseline + VQA ckpts saved (per `r-preference/doc/0524-place-category.md`
and `0523-height-hv-oreint-design-doc.md`).

### 1.2 stage_a_gate.py × 4 new cats

Ran `stage_a_gate.py` on H200 for height/hvlv/orient/place at 25k. Output:
`r-preference/eval/stage_a_gate_{cat}_25k.json`.

All 4 reported RED. The eval log printed counterfactual MSE values 30–40× larger
(1.41) than the contact reference (0.031), which made us suspect a metric bug,
not a gate fail.

### 1.3 Layered debugging

Eliminated environment / scp / stats issues:

| check | result |
|---|---|
| md5 scp'd baseline ckpts H100 vs H200 | MATCH ✓ |
| md5 stats files H100 vs H200 | MATCH ✓ |
| H200 has updated eval/dataset code | rsync confirmed ✓ |

Then ran a custom debug script that exposed the real picture (§3 below).

---

## 2. Methodology (the metrics we used)

`stage_a_gate.py` produces these 5 metrics per cat:

1. **VQA per-task acc on taskB** — the gate-main metric (binary classifier).
2. **VQA per-task acc on taskA val** — sanity (should be high on training distribution).
3. **Counterfactual action MSE on taskA val** — feed same (image, state) with two
   different pref-suffixed prompts; measure action diff. Historically interpreted
   as "model uses pref token".
4. **Sign accuracy on contact strong tasks** (contact-only; n/a for 4 new cats).
5. **Baseline ↔ VQA action drift** — sanity (VQA shouldn't break action quality).

We added two new diagnostics this session:

6. **Same-prompt RMSE (`predict_action` × 2 with same input)** — measures noise
   floor of flow-matching inference.
7. **Counterfactual RMSE with FIXED random seed** — set `torch.manual_seed(SEED)`
   before each call so both A and B start from the SAME initial noise. The
   resulting CF RMSE is the **pure pref signal** (no inference noise).

The interpretation:
- If `same_prompt_rmse ≈ counterfactual_rmse` → output is dominated by noise; the
  model isn't using the prompt to differentiate.
- If `seeded_CF` is essentially zero → model truly ignores the pref token.

### 2.1 The CF-MSE-as-pref-signal interpretation was wrong

The 0524 contact analysis interpreted CF MSE ≈ 0.031 as "baseline≈VQA, both
action heads read pref from the prompt token." Actually:

```
contact:
  baseline_same_prompt_rmse ≈ baseline_CF_noseed ≈ 0.03  → both at noise floor
  (we lacked the same_prompt_rmse measurement in 0524)
```

The action head **always** has stochastic noise (4-step flow matching from random
init). The 0.03 was just **the noise floor**, not the signal.

Now with seeded CF:
- place baseline: **0.0020** (real signal, small)
- place VQA: **0.0005**
- orient baseline: **0.0018**
- orient VQA: **0.0000** (zero)
- height baseline: **0.0000** (zero)
- height VQA: **0.0000**
- hvlv baseline: **0.0000**
- hvlv VQA: **0.0000**

So the prompt-→-action signal exists for place and orient baseline, **vanishes
for everything else**.

---

## 3. Per-cat root cause (with evidence)

### 3.1 place — **healthy, just hard taskB**

- Stage A training loss: 1.4 → **0.014** (smooth monotonic convergence)
- Final action std: 0.84, range [-1.04, 1.03] (clamped to normalized space)
- Seeded CF: 0.0020 ≥ 0 (real signal in action head)
- taskA val VQA acc: **0.79** (reasonable training-dist behavior)
- taskB VQA acc: **0.44** (worse than chance — but uniform_8 frames; mid_8 might help)

**Verdict**: model trained correctly. taskB (`place_soap2_stand`) is a different
target object than training-time tasks, and uniform_8 frames may miss the
discriminative moment (analogous to 0524 contact taskB analysis where uniform_8
got 0.61 but mid_8 hit 0.99).

**Action**: include place in Stage B queue. Pseudo-labels go through
`pseudo_label_offline.py` with `--filter_threshold 10.0` — only high-confidence
(\|logit_gap\| ≥ 10) samples become pseudo-labels. With 0.44 taskB acc and
strongly biased predictions, the filtered set may be small but should be high
quality if any kept samples have high gap.

### 3.2 orient — **training failure, NOT data**

- Stage A baseline: loss converged, seeded CF = 0.0018 (real signal). Action
  range [-1.01, 1.03]. **Model is fine.**
- Stage A VQA: loss bounced **1.2–1.7 throughout 25k steps**, never converged.
  Action range [-3.84, 3.49] (~3× out of bounds). Seeded CF = 0.0000.

What was different? The VQA run was a **resumed run**:
- 03-orient-vqa first started 2026-05-24 03:33 UTC, killed at step ~13480 (user
  killed it to free H200 for other work)
- Restarted via temp-523-h200.sh's `01-orient-vqa-resume` with `IS_RESUME=1`
  from steps_10000 ckpt. Adam momentum reset (cosine LR scheduler also state-
  restored mid-curve).
- Loss never recovered after resume.

The orient **data is good** — we verified: `place_bottle_box_0` vs `_90`
quaternions differ by quat[0]=0.805 vs 0.657, quat[2]=0.016 vs 0.304 — that's
strong rotational signal between pref labels.

**Verdict**: VQA training was destabilized by the resume. Same data trained
baseline successfully → not a data problem.

**Action**: **fresh retrain** orient VQA from base (no resume). In-flight on
H200 in `temp-525-h200.sh::02-orient-vqa-fresh`. ETA ~9h.

### 3.3 height — **upstream data 50% mislabeled** 🔥

Histogram of L_z final per episode, n=100 per pref:

```
height/move_mouse_pad_high (bins from 0.85 → 1.15):
  [0, 0, 48, 0, 1, 51, 0]
        ↑↑                ↑↑
     48 ep at z≈0.94    51 ep at z≈1.07
     (LOW-like — MISLABELED!)   (correctly HIGH)

height/move_mouse_pad_low (same bins):
  [0, 27, 73, 0, 0, 0, 0]
            ↑↑
     100% at z≈0.94 (consistent, correctly LOW)
```

Same pattern across multiple task_groups:

| task_dir | mislabel count (in `_high/`) | correct |
|---|---|---|
| `move_mouse_pad_high` | 48 | 52 |
| `place_mouse_stand_high` | 52 | 48 |
| `move_soap_pad_high` | ~50 | ~50 |

**`_low/` is consistently correct.** `_high/` directories contain a ~50/50
mix of actually-low and actually-high trajectories. The model sees "Preference:
high drop" prompt paired with low-trajectory actions half the time → optimal
solution is to ignore the prompt and predict mean (loss = velocity variance ≈ 1.5).

Evidence this is data, not training:
- **Baseline (no VQA) also stuck at loss 1.5**. Baseline uses pure action MSE.
  If training mechanism were broken, baseline shouldn't fail. → not training.
- **place baseline trained on same code + arch + hyperparams → converged**.
  Only differential is data → data is the cause.
- **Histogram evidence is direct**: actual EE z values in `_high` are
  bimodal at ~0.94 and ~1.08, not unimodal at ~1.08 as labels suggest.

**Verdict**: **upstream RoboTwin sim or HF data pipeline mis-categorized half of
the high-drop trials**. Likely cause: sim trial failures (high drop not
executed → robot stayed low → ep saved under `_high/` regardless of outcome).

**Action**: **user is checking data side**. Two options once confirmed:
1. Fix upstream — re-collect or re-classify episodes in `kaiwen2/robotwin-prefvla-pod3`
2. Local filter — write a dataloader filter that requires `z_final > 1.0` for
   `_high` eps, drop the ~48 mislabeled per task_group. Resulting clean dataset:
   ~52 high + 100 low per task × 8 task_groups = ~1216 eps train.

### 3.4 hvlv — **weak pref signal in data**

hvlv pref is "wide detour (hv) vs narrow detour (lv) around the `can` obstacle"
between source and target. The signal should be in **trajectory shape**, not
endpoint position.

We measured "max perpendicular deviation from straight-line source→target" per
episode (a proxy for detour magnitude):

```
hvlv/place_apple_plate:
  hv: n=100  mean=0.157 m  std=0.163 m  range=[0, 0.56]
  lv: n=100  mean=0.142 m  std=0.149 m  range=[0, 0.45]
  
  mean diff (hv vs lv) = 0.015 m
  within-pref std       = 0.150 m
  Signal/Noise           ≈ 0.10
```

The difference between hv and lv mean trajectories is **10× smaller** than the
episode-to-episode noise within either pref. With 100 episodes per pref, this
signal is barely detectable statistically, and certainly invisible to a model
trying to learn pref-conditional action chunks from individual examples.

Different from height: hvlv labels may be **technically correct**, but the
trajectories of the two prefs **just don't differ enough** to learn from.

Possible causes:
1. Sim trial randomness larger than configured pref difference
2. Both hv and lv trials are physically valid paths and the sim chose
   similar paths regardless of label
3. Or the discriminative signal is in an axis we didn't probe

**Action**: **user is checking data side**. We did not investigate deeper —
deferred to user. Possible follow-ups:
- Probe other axes: path length, curvature, mid-frame xyz
- If signal really is this weak, re-collect with stronger pref differentiation
  (sim config change, out of our scope)
- Or accept hvlv isn't learnable from this dataset

---

## 4. The recovery in flight on H200 (`temp-525-h200.sh`)

Launched 2026-05-25 ~11:30 UTC via nohup. tmux session optional (logs are
`/home/kevin/starVLA/results/.temp_525_h200_runs/_summary.log`).

| # | tag | duration | depends | notes |
|---|---|---|---|---|
| 01 | `pseudolabel-place` | ~5 min | place VQA 25k | filter @ `|logit_gap| ≥ 10`; may keep few samples |
| 02 | `orient-vqa-fresh` | ~9h | orient baseline (warm-start? no — base Qwen3-VL-4B) | fresh, no resume |
| 03 | `pseudolabel-orient` | ~5 min | depends on (02) | |
| 04 | `stageb-place-main` | ~1.5h | depends on (01) + place VQA | YAML uses relative `results/Checkpoints/...` so resolves on H200 |
| 05 | `stageb-place-b0` | ~1.5h | place baseline | no pseudo-label needed (B0) |
| 06 | `stageb-orient-main` | ~1.5h | depends on (02) + (03) | |
| 07 | `stageb-orient-b0` | ~1.5h | orient baseline (already on H200) | |

**Pre-launch action taken**: deleted
`/mnt/localssd/kevin/starVLA_runs/Checkpoints/pref_main_stage_a_v1_VQA_orient`
(92 GB freed) — the failed-resume orient VQA ckpts are gone. Fresh retrain
will populate the same dir from scratch.

Total ETA: ~16h. Completion ETA: **~2026-05-26 03:30 UTC**.

### 4.1 Failure handling

`temp-525-h200.sh` continues to next job on any failure (set -e is off). Per-job
logs at `results/.temp_525_h200_runs/<tag>.log`. Top-level summary at
`_summary.log`.

If orient VQA fresh retrain still fails: orient Stage B steps (06, 07) will
also fail (missing warm-start ckpt). Place Stage B steps (04, 05) still work.

### 4.2 Expected pseudo-label gate result

`pseudo_label_offline.py` includes a launch-gate verification (per Stage B
contact plan §3.2): compares pseudo-labels against dir-derived GT and aborts
if pseudo-vs-GT acc < 0.95.

- **Place**: gate eval taskB acc was 0.44. After |gap|≥10 filter, kept set may
  be small but high-confidence on a biased subset. **Likely to pass the
  launch-gate** if filtered subset is internally consistent. If not, Stage B
  place main fails — Stage B place B0 still runs (B0 doesn't use cache).
- **Orient**: after fresh retrain, gate acc unknown but should be ≥0.5 if data
  is good (which we believe). If still below 0.95, may need to threshold-tune.

---

## 5. Reproducibility — key commands

### 5.1 Verify the H100 data mislabeling

```bash
source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh && conda activate starVLA
python - <<'PY'
import h5py, numpy as np
from pathlib import Path
D = Path("/mnt/localssd/kaiwenh/pref/data/height")
for pref in ["high", "low"]:
    z = [h5py.File(D/f"move_mouse_pad_{pref}"/"data"/f"episode{i}.hdf5","r")["endpose/left_endpose"][-1,2] for i in range(100)]
    z = np.array(z)
    bins = np.linspace(0.85, 1.15, 8)
    h, _ = np.histogram(z, bins=bins)
    print(f"{pref}: mean={z.mean():.4f} median={np.median(z):.4f} hist={list(h)}")
PY
```

### 5.2 Re-run the seeded CF debug

Script `/tmp/debug_param_seed.py` (scp'd to H200 at `/tmp/debug_param_seed.py`)
checks: parameter norms + seeded-vs-noseed CF + state norm range.

### 5.3 Check H200 wrapper progress

```bash
ssh kevin@34.34.93.23 'cat /home/kevin/starVLA/results/.temp_525_h200_runs/_summary.log'
```

### 5.4 Per-cat ckpt locations

H100:
- `results/Checkpoints/pref_baseline_stage_a_v1_noVQA_{contact,height,hvlv,place}/checkpoints/steps_25000_pytorch_model.pt`
- `results/Checkpoints/pref_main_stage_a_v1_VQA_contact/checkpoints/steps_*.pt`
- (place VQA only on H200 currently)

H200:
- `Checkpoints/pref_baseline_stage_a_v1_noVQA_{contact?,height,hvlv,orient,place}/checkpoints/steps_25000_pytorch_model.pt`
  (note: H200 ckpts at `/mnt/localssd/kevin/starVLA_runs/Checkpoints/`, no `/results/` in physical path)
- `Checkpoints/pref_main_stage_a_v1_VQA_{contact,height,hvlv,place}/checkpoints/steps_25000_pytorch_model.pt`
- `Checkpoints/pref_main_stage_a_v1_VQA_orient` — DELETED 2026-05-25; will be repopulated by `temp-525-h200.sh::02-orient-vqa-fresh`.

### 5.5 Stage B YAML pretrained_checkpoint paths

All 4 new Stage B YAMLs (place + orient × main + b0) now use **relative**
paths (`results/Checkpoints/...`) for `pretrained_checkpoint`, resolving via
each machine's `results/` symlink to the right SSD ckpt. This was changed
2026-05-25 from absolute H100-side paths. Diff:

```
< pretrained_checkpoint: /mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/...
> pretrained_checkpoint: results/Checkpoints/...
```

This unblocks Stage B on either H100 or H200 from the same YAML.

---

## 6. Open questions / TODO

1. **height + hvlv data**: user is investigating. Once root cause is confirmed
   (50% mislabeling for height; weak signal for hvlv), decide:
   - height: fix upstream OR write a filter to drop mislabeled `_high` eps. The
     filter approach: in `pref_hdf5_dataset.py::_split_episodes`, after listing
     eps, for height filter out `_high` eps with z_final < 1.0.
   - hvlv: probe alternate axes (path length, curvature, mid-frame xyz across
     multiple frames); if signal still weak, accept hvlv not learnable.

2. **place taskB gate acc 0.44**: if Stage B main passes the launch-gate (i.e.
   filtered pseudo-labels are internally consistent), Stage B might still
   succeed even with biased coverage. If gate fails, retry frame_window_test
   with mid_8 / gripper_anchored — analogous to 0524 contact analysis.

3. **orient VQA fresh retrain success**: if loss STILL doesn't converge after
   fresh retrain, then it's not the resume — there's a deeper issue with
   orient VQA cotrain. In that case, may need to look at λ_vqa tuning or
   different clip strategy at train time.

4. **gate eval methodology fix**: the 0524 contact analysis interpreted CF MSE
   as "model uses pref token", but that turned out to be the noise floor. The
   eval should be augmented with the seeded-CF metric (see §2.1) to give a
   clean signal-only read. Need to update `stage_a_gate.py` to add this.

---

## 7. References

- Eval result JSONs: `r-preference/eval/stage_a_gate_{height,hvlv,orient,place}_25k.json`
- Debug reports: `/tmp/debug_report.json` (phase 1), `/tmp/debug_phase2.json` (param + seeded)
- Wrapper: `examples/preference/temp-525-h200.sh`
- Stage A spec: `r-preference/doc/0522-a-giveobj.md`
- Stage A new-cat additions: `r-preference/doc/0523-height-hv-oreint-design-doc.md`
- place category addition: `r-preference/doc/0524-place-category.md`
- Stage B contact plan: `r-preference/doc/0524-stageB-contact-plan.md`
- Training runbook: `r-preference/doc/training-runbook.md`

---

End of doc.
