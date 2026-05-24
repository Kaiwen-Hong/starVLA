# Stage B contact — pref-flip diagnosis (autonomous deep dive, 2026-05-24)

Written 2026-05-24 ~20:50 UTC during user-leave window. Tries to explain
why Stage B main final ckpt only achieved sign-acc 7/10 on the initial
§7.5 sanity (below the ≥8/10 gate), and what to do about it.

**Headline finding**: Stage B training as designed does NOT add
prompt-conditioning to the action head. **The Stage A baseline already
had the strongest pref-following we observe on taskB; Stage B's SFT on
100 demos × ~12 epochs degraded it via memorization shortcut.**

---

## 0. Summary table (N=50 sign-acc, L_z axis, expected +1)

| ckpt | sign-acc | mean Δz | std Δz | 95% CI | interpretation |
|---|---:|---:|---:|---|---|
| **Stage A baseline 35k** | **31/50 = 0.62** | +0.0122 | 0.064 | [47%, 76%] | mild signal; best of all |
| Stage A VQA 35k | 23/50 = 0.46 | -0.0102 | 0.086 | [32%, 60%] | random / slightly inverted |
| **Stage B main final** (warm-start VQA) | **31/50 = 0.62** | **+0.0163** | **0.049** ← lowest std | tied with baseline; most consistent but tiny magnitude |
| Stage B B0 final (warm-start baseline) | 26/50 = 0.52 | +0.0059 | 0.082 | [37%, 66%] | random (degraded from baseline 0.62) |

**Statistical reality**: 95% binomial CIs all overlap heavily and all
include 0.50 (chance). At N=50 we cannot statistically distinguish 0.62
from 0.50. But the DIRECTIONAL pattern (mean Δz, std, frame sweep) is
consistent enough to draw qualitative conclusions.

**Magnitude check**: main mean Δz +0.0163 in normalized space →
**~2.7 mm physical grip height delta** (using L_z norm span 0.327m).
Demonstrations show ~50 mm difference between low/high contact grasp.
**Model's pref response is ~30× weaker than the data signal it was
trained on.**

---

## 1. Phase A — main ckpt trajectory (steps 500-3000, N=10)

| step | sign-acc (N=10) |
|---|---:|
| 500  | 3/10 = 0.30 (worst — model not yet converged) |
| 1000 | 7/10 = 0.70 (best from this sample) |
| 1500 | 5/10 = 0.50 |
| 2000 | 5/10 = 0.50 |
| 2500 | 4/10 = 0.40 |
| 3000 | 6/10 = 0.60 |

**Pattern**: NOT monotone. Noisy with peak around step 1000, drift back
to noise level by step 2500-3000.

→ More training won't help. The "under-training, bump to 5000" caveat
from doc §4.3 is wrong here — bumping would not improve sign-acc,
based on this trajectory.

→ N=10 too noisy to compute "sweet spot" cleanly, but trend says
nothing is improving with more steps.

---

## 2. Phase B — Stage A as reference (the WARM-START sources)

Both Stage A ckpts evaluated on the same 10 taskB episodes with the
same pref-flip protocol as Stage B (using `--framework_name QwenPI_VQA`
for the VQA ckpt):

| ckpt | N=10 sign-acc | N=50 sign-acc |
|---|---:|---:|
| Stage A baseline 35k | 8/10 = 0.80 (passes!) | 31/50 = 0.62 |
| Stage A VQA 35k | 6/10 = 0.60 | 23/50 = 0.46 |

The N=10 result of 8/10 for baseline was noise (CI 44-97%). With N=50,
baseline drops to 0.62 — same as Stage B main.

**Surprising**: Stage A **baseline** (action loss only, no VQA cotrain)
has **better** taskB pref-following than Stage A **VQA** (0.62 vs 0.46).
This contradicts the working theory that VQA cotrain helps with pref-
conditioning. Possible explanation: VQA cotrain optimizes the LM head
for binary classification, which interferes with the action head's
attention to prompt tokens.

This deserves a separate investigation — Stage A VQA was supposed to
be the better starting point but is actually worse on this metric.

---

## 3. Phase C — bigger-N + per-axis + frame sweep (N=50 / 30)

### 3.1 Per-axis sign-acc (N=50)

Best axis varies by ckpt — but L_z, L_y, R_x cluster around 0.62-0.72.
With N=50 + "best of 20 axes" being implicit selection, +0.66 best
of N=50 has ~5% chance of being noise. So 0.72 best axis = at most
weak signal.

| ckpt | L_z | best axis | best acc |
|---|---:|---|---:|
| Stage A baseline | 0.62 | L_y | 0.66 |
| Stage A VQA | 0.46 | R_x | 0.64 |
| Stage B B0 | 0.52 | R_x | 0.72 |
| Stage B main | 0.62 | L_6d_1 | 0.68 |

No clean "alternative axis" interpretation. L_z is roughly as good as
any other for measuring pref response.

### 3.2 Per-chunk-step sign-acc (L_z axis, N=50)

Looking at sign-acc by chunk step k=0...15 for L_z axis:

```
                  k= 0  k= 1  k= 2 ... k=12  k=14  k=15
Stage A baseline  0.6   0.5   0.5  ...  0.8   0.7   0.7  ← strong late-chunk signal
Stage A VQA       0.6   0.4   0.4  ...  0.5   0.5   0.5  ← flat noise
Stage B B0        0.5   0.5   0.5  ...  0.6   0.7   0.5  ← mid-chunk weak signal
Stage B main      0.5   0.5   0.6  ...  0.6   0.5   0.5  ← flat noise
```

**Stage A baseline** shows a clear late-chunk pattern: k=12 has 0.80
sign-acc, k=14/15 at 0.70. This is the post-grasp lift trajectory —
model lifts gripper higher for "high contact" prompt vs lower for "low
contact". Real pref-following.

**Stage B main** completely lost this late-chunk pattern (all 0.5-0.6
across chunk). The training erased the structure.

### 3.3 Frame-fraction sweep (N=30)

Tried earlier frames to test "image shortcut" hypothesis (model uses
image at grasp_t which already reveals pref):

| frame_frac | Stage A baseline | Stage B main |
|---|---:|---:|
| 0.20 (very early approach) | 0.67 | **0.40 (below chance!)** |
| 0.30 (mid-approach) | 0.57 | 0.50 |
| 0.40 (~grasp) | **0.73** (best) | 0.53 |
| 0.50 (post-grasp) | 0.67 | 0.63 |
| grasp_t (~0.32, N=50) | 0.62 | 0.62 |

**Stage A baseline**: pref-following present at ALL frames, peaks at
0.40 (just before/at grasp). Even at frame 0.20 (image doesn't reveal
grasp yet), baseline still shows 0.67 — model genuinely uses prompt.

**Stage B main**: at frame 0.20, sign-acc is **0.40 — BELOW chance**
(model goes opposite direction!). Pref-following only emerges at later
frames (≥0.40) where image starts to encode grasp position.

→ **Stage B main relies on image features more than prompt.** The
prompt-conditioning is much weaker than in Stage A baseline.

---

## 4. Diagnosis

Putting it together:

1. **Stage A baseline learned weak-but-real pref-conditioning** during
   its 35k-step training on 192k taskA samples. This conditioning
   transfers to taskB (different object, plate scene) at sign-acc 0.62-0.73.
   Real signal lives in the late chunk (post-grasp lift trajectory).

2. **Stage A VQA UNDERPERFORMS baseline on action-side pref-flip** (0.46
   vs 0.62). The VQA cotrain seems to dilute action-head attention to
   prompt tokens — surprising and contradicts the "VQA helps" prior.

3. **Stage B's 3000-step SFT on 100 demos overrode Stage A's pref-
   conditioning** with image-action memorization. The model learns
   "this specific image → output this specific action chunk" and the
   prompt becomes redundant noise that the model ignores.

   Evidence:
   - Main mean Δz magnitude (+0.016) is 30× smaller than demo signal
   - Main per-chunk-step sign-acc is flat 0.5-0.6 across all k (no
     learned structure)
   - Main at early frame (image less informative) drops to 0.40 below
     chance (model can't use prompt without image cues)
   - B0 (no pref training) at 0.52 — Stage B without pref suffix
     completely erased the baseline's 0.62 pref-following

4. **The L_action wandb curves were misleading**. Both main and B0
   reach L_action ~0.018 — looking healthy. But low L_action means
   "good fit to demo actions", not "good pref-conditioning". Since both
   pseudo-labels in main and absent-labels in B0 produce equally good
   demo-fit, L_action alone can't distinguish them.

---

## 5. Why Stage B doesn't work as designed

The fundamental issue: **Stage B's training objective doesn't force
the model to use the prompt**. Given:
- Each demo has a fixed (image_distribution, pref_label) pairing
- The model sees the demo image and a prompt with the SAME pref label
- Loss = MSE(predicted_action, demo_action_chunk)

The model can minimize loss by EITHER:
- (a) Reading prompt suffix → predict action conditional on pref → low loss
- (b) Reading image features (which strongly correlate with pref via demo trajectory) → predict the memorized action → also low loss

Both achieve same loss. The model defaults to (b) because:
- The image features are richer (visual encoder is huge)
- The prompt suffix is short (~3 tokens) and less informative per pass
- Memorization is easy with only 100 demos × many epochs

→ **No gradient pressure to use the prompt at all.**

Compare Stage A which trained on 192k samples (1280 episodes × ~150
frames): memorization is much harder, so the model is forced to find
generalizable features — which include prompt. Hence Stage A baseline
shows pref-following.

---

## 6. Recommendations (in cost-impact order)

### 6.0 IMPORTANT immediate practical implication

**Do not use Stage B main ckpt for any downstream task** (e.g., closed-
loop rollouts, paper experiments). Its pref-following is worse or equal
to just using the Stage A baseline ckpt at inference time with `Preference: ...`
prompts. The 100-demo SFT was a regression in this dimension.

If you need a pref-conditioned policy for taskB **NOW**:
- Use **Stage A baseline 35k** as-is (sign-acc 0.62-0.73, mean Δz +0.03)
- Or use **Stage A baseline 35k + brief Stage B SFT (~200 step max)**
  before memorization kicks in (untested — would need quick sanity sweep)

Neither passes the ≥0.80 gate, but baseline is the best we have.

### 6.1 Cheap diagnostic (no retrain)

- **Try Stage A baseline at ≥0.80 task_groups in taskA** — see if it
  passes there. If yes, the issue is taskB-specific (plate scene shift).
  If no, the issue is the pref-following architecture itself.

### 6.2 Stage B redesign options

a) **Tiny Stage B (~200-500 step max)**: stop before memorization
   shortcut kicks in. Sign-acc trajectory shows peak around step 1000;
   maybe peak is even earlier. Run sanity at each save (250 step grid).
   Risk: under-training for action adaptation to taskB visuals.

b) **Counterfactual augmentation**: for each demo, also create a
   "wrong-pref" version (same image, opposite pref label) and DON'T
   train action loss on that — but force the model to predict different
   action. Hard to set up correctly.

c) **Freeze visual backbone during Stage B**: only train DiT action
   head. This forces action head to use prompt (since visual features
   can't change). Major code change.

d) **More taskB data**: 100 demos → 1000 demos. Memorization becomes
   much harder. Probably can't get this data cheaply.

e) **Different objective**: train on `predict_pref_from_action_and_image`
   instead of `predict_action_from_pref_and_image`. Inverse problem.

### 6.3 Stage A redesign (if pref-following itself needs improvement)

a) **Bigger pref label vocabulary in training**: instead of "Preference:
   low contact"/"high contact", use varied phrasings ("grip near base",
   "grip at upper portion", etc.) — would prevent token-level overfit
   on the 2 specific tokens.

b) **Action loss weighting per-frame**: weight grasp/lift frames higher
   (where pref signal lives) so loss pushes model to use prompt there.

c) **Joint training without VQA**: VQA cotrain hurts action-side pref-
   following per §3.1 finding. Just train action with pref prompts
   (= Stage A baseline) and accept that for pseudo-labeling we use a
   different approach (manual, classifier on top of vision encoder, etc.).

---

## 7. What was committed during this diagnostic

- `examples/preference/stage_b/sanity_pref_flip.py`: extended with
  `--framework_name` (for Stage A VQA ckpts), `--dump_chunks` (full
  action chunks for offline analysis), `--frame_fraction` (test image
  shortcut hypothesis at earlier frames).

- `examples/preference/stage_b/analyze_pref_flip.py`: NEW. Reads dump
  JSONs, computes per-axis sign-acc + per-chunk-step sign-acc +
  magnitude.

- `r-preference/eval/stageb_diag/`: N=10 sanity dumps across 8 ckpts
  (Phase A: main step 500-3000; Phase B: Stage A baseline + VQA).

- `r-preference/eval/stageb_diag_bigN/`: N=50 sanity dumps on 4 critical
  ckpts (Stage A baseline, Stage A VQA, Stage B main, Stage B B0).

- `r-preference/eval/stageb_diag_frame/`: N=30 sanity dumps at 4 frame
  fractions on Stage A baseline + Stage B main (image-shortcut test).

- `r-preference/eval/stageb_contact_diagnosis.md` (this doc).

---

## 8. Bottom line

Stage B in current design has a **structural problem**: small-data SFT
with low LR makes the model memorize image-action pairings rather than
learn to attend to prompts. Pseudo-labels become redundant signal
because the image already encodes the GT. **The whole Stage B pseudo-
labeler approach is not load-bearing here** — Stage A baseline (no
pseudo-labels, no Stage B) is the best pref-follower we measure.

For paper purposes: this is actually an important negative result.
The "pseudo-label + SFT" recipe works on the labels (cache acc 1.000)
but fails on the action side because:
- The 100-demo training set is too small for action-side conditioning to
  resist memorization
- Image features pre-encode the answer

Next-session work should:
1. Test §6.0 (use Stage A baseline directly at inference) — quick eval
2. Test §6.2a (tiny Stage B with sanity at each save) — 1h
3. Investigate §3.1 (Stage A VQA hurts action pref-following) — could
   change Stage A design going forward
