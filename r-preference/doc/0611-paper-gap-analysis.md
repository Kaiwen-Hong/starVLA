# 2026-06-11 — Paper-claims gap analysis + new evidence (orient geom labeler, Tier-1 height/orient)

> **Purpose.** The current paper draft (SPT, CORL-2026 style; saved as
> [`../paper/paper_draft-v7.tex`](../paper/paper_draft-v7.tex)) states concrete numbers in
> Tables 1–3. This doc maps every claimed number to the evidence we actually have on disk,
> records what was newly measured today, and lists the prioritized work to close each gap.
>
> Context docs: [`0529-stageB-pseudolabel-and-eval.md`](0529-stageB-pseudolabel-and-eval.md)
> (Stage-B infra + geom labeler + Tier-1 proxy), [`0529-5090-eval-and-stageb-controllability.md`](0529-5090-eval-and-stageb-controllability.md)
> (closed-loop methodology), [`../eval/0529_50ep/README.md`](../eval/0529_50ep/README.md)
> (the archived 50ep/5ep paired-controllability results).
>
> **Eval-box status (2026-06-11):** the OLD 4×5090 box (`99.148.65.10:18970`) is DEAD
> (connection refused). A NEW box was rented: `ssh -p 17278 root@169.40.1.214`
> (⚠ nvidia-smi shows **2×RTX5090**, not the advertised 4; disk 126G total / ~32G free;
> pre-existing robomme content must not be touched). Setup is in progress (background agent;
> notes will land in `0611-5090-new-box-setup.md`). The starvla_joint bridge survived on the
> collection box: `kaiwen@100.97.239.33:~/Desktop/research/ar-research_exp/policy/starvla_joint/`.
> The place 5ep per-episode metric JSONs did NOT survive (only the README aggregate);
> `eval/0529_50ep/per_episode/pref_stageb_main_place_geom/*/` are empty dirs.

---

## 1. Table 1 (recognition / pseudo-label accuracy) — claim vs evidence

Paper claims (per axis, chance 50): contact 100, height(release) 100, hvlv(clearance) 92,
orient 100, place 90 → avg 96.4.

| axis | paper | measured (cache on disk) | labeler | status |
|---|---|---|---|---|
| contact | 100 | **1.000** (100/100) `pref_pseudo_labels_contact_B.json` | geom (`height_fraction`, scene_info) | ✓ backed |
| height | 100 | **1.000** (100/100) `..._height_B.json` | token-VQA | ✓ backed |
| hvlv | 92 | **0.920** (92/100) `..._hvlv_B.json` | geom (EE detour, no privilege) | ✓ backed |
| orient | 100 | token 0.95 → **geom 1.000 (100/100), NEW 2026-06-11** `..._orient_B_geom.json` | geom (wrist R[2,2]@frac0.40, pure EE) | ✓ **backed as of today** |
| place | 90 | token 0.90 / **geom 1.000** (`..._place_B.json` is the geom cache) | both exist | ⚠ paper number = token; geom = 1.00 |

**New today — orient geom labeler** (probe `/tmp/orient_geom_probe.py`, production in
`examples/preference/stage_b/pseudo_label_geom.py --cat orient`):
- Feature = active-arm wrist rotation `R[2,2]` (tool z-axis vertical component) at grasp
  frac 0.40. Threshold fit on taskA → taskA acc **1.000** (n=640), taskB acc **1.000**
  (n=100, pred 50/50 balanced). Several redundant rotation features (R20/R22 at fracs
  0.4/0.6/0.8) all give 0.97–1.00 with S/N 5–11.
- The 5 episodes the token-VQA confidently mislabeled (`move_can5_away_0/ep{7,27,31,33,38}`,
  GT 0 → pred 90) are all **correctly labeled 0** by the rotation feature → those were
  labeler errors, NOT data noise. Cache: `r-preference/eval/pref_pseudo_labels_orient_B_geom.json`
  (kept separate; the original token cache that trained `pref_stageb_main_orient` is untouched).

**Decision needed (paper):** with orient-geom the honest per-axis "best validated labeler"
set is {100, 100, 92, 100, **100**} → avg **98.4**, vs the paper's current
{100, 100, 92, 100, 90} → 96.4 which mixes token-place (0.90) with geom elsewhere. Either
(a) report best-per-axis (avg 98.4; "weakest axis" narrative becomes hvlv-only), or
(b) keep place=90 (token) — then state the labeler-per-axis choice explicitly.

## 2. Table 2 (sim following / success) — claim vs evidence

Paper: Naive FT following 59/53/57/61/67 (avg 59.4), SPT 90/81/75/95/83 (avg 84.8);
success FT 92/89/85/94/80 (avg 88), SPT 95/91/82/92/86 (avg 89.2).

Measured closed-loop (RoboTwin paired same-seed two-prompt; `eval/0529_50ep/`):

| axis | SPT follow (measured) | SPT success (measured) | FT closed-loop | gap to paper |
|---|---|---|---|---|
| contact | **0.75** @5ep (sep +0.021, right direction, weak) | 4/5, 4/5 | not run | paper 90 — **largest conditioning gap** |
| height | **0.81** @50ep ✓ (=paper) | 50/50, 41/50 (~91 ✓) | not run | FT column unmeasured |
| hvlv | **0.50** @5ep, success **0/5 both prompts** | 0/5 | not run | paper 75/82 — **task transfer itself fails** (stamp_seal6) |
| orient | **0.81** @50ep (0-side 1.00, 90-side 0.61 IK-capped) | 48/50, 43/50 ✓ | not run | paper 95 — 90-side needs fix |
| place | **0.88** @5ep (per README; raw records lost with old box) | center 4/5 (corner 0/5 = recipe artifact) | not run | needs 50ep re-run |

**Key fact: ALL "Naive FT" closed-loop numbers in Table 2 are currently unmeasured.**
The b0 (=Naive FT) ckpts exist for all 5 cats on the H100
(`pref_stageb_b0_{height,orient,contact,hvlv,place}`, steps_1500, all converged), but only
offline Tier-1 proxy numbers exist:

| arm | Tier-1 follow-acc (offline proxy) | effect | file |
|---|---|---|---|
| b0 place | 0.50 | 0.054 | `ctrl_b0_place.json` |
| b0 contact (2500) | 0.54 | 0.0002 | `ctrl_b0contact_2500.json` |
| b0 hvlv (2500) | 0.625 | 0.013 | `ctrl_b0hvlv_2500.json` |
| **b0 height (NEW)** | **0.875** | **0.027** | `ctrl_b0height_1500.json` |
| **b0 orient (NEW)** | **0.417** | 0.0004 | `ctrl_b0orient_1500.json` |
| main height (NEW) | 0.958 | 0.037 | `ctrl_height_1500.json` |
| main orient (NEW) | 0.375 | 0.0004 | `ctrl_orient_1500.json` |

Decode of the place 2×2 ablation runs on disk (for the paper's commented-out Table 4;
verified from the YAMLs 2026-06-11):
- `pref_stageb_b0_place` = baseline-Stage-A start, no suffix → **B0**, proxy 0.50
- `pref_stageb_yn_place` = token-Stage-A start, NO suffix → **"VQA-only"** arm, proxy 0.50
- `pref_stageb_ny_place` = baseline-Stage-A start, suffix with **vanilla-VLM labels (0.61 acc)**
  (`pref_pseudo_labels_place_B_vanillaVLM.json`) → label-source ablation, proxy 0.75
- `pref_stageb_main_place_geom` = token start + geom labels (1.00) → **SPT**, proxy 1.00
So offline: B0 0.50 / VQA-only 0.50 / weak-labels 0.75 / SPT 1.00 — "either component alone
is below SPT" is already supported offline for place; closed-loop versions still needed.

Also: `pref_stageb_main_{contact,hvlv}_geom` were in fact trained to **2500 steps**
(10 ckpts each) — the "more steps" experiment is already done: contact 1500→2500 flat
(0.667→0.667), hvlv slightly up (0.667→0.708). More steps alone does not rescue them.

Two caveats from today's runs:
1. **The Tier-1 proxy is a false negative for orient** (main 0.375 offline vs closed-loop
   82°/0–2° = clearly controllable). L1-to-demo over the 50×14 chunk is insensitive to
   small/late wrist-rotation deltas (effect 0.0004). Same mechanism plausibly depresses the
   contact/hvlv proxy reads. → Proxy is only trustworthy for large/late spatial deltas
   (place, height). Closed-loop is the paper metric; treat proxy as a filter only.
2. **b0 height shows real offline conditioning (0.875 / effect 0.027)** — Stage-A source
   training (whose prompts DO carry the suffix) instills height conditioning that 1500
   target SFT steps do not wash out. The paper's FT-height=53 assumption is at risk;
   the closed-loop b0-height run on the new box will decide. If closed-loop b0 height
   follows >> chance, the Naive-FT baseline definition (or its training length) needs
   revisiting — e.g. FT trained longer on target, or the mixed-finetune variant the paper
   text literally describes.

## 3. Table 3 (real UR5e) — owned by the realworld workstream; not covered here.

## 4. Prioritized plan

| P | item | route | status |
|---|---|---|---|
| P0 | New 5090 box up (server + RoboTwin + bridge) | bg agent; bridge from collection box | in flight |
| P0 | Closed-loop **b0 ×5 cats** (paired, same seeds as SPT) → fills the entire FT column | new box, `run_pref_control.sh` pattern | blocked on box |
| P0 | Closed-loop SPT 50ep for **place / contact / hvlv** (currently 5ep / lost-raw) | new box | blocked on box |
| P1 | **orient relabel (geom 1.00) + Stage-B retrain** (`pref_stageb_main_orient_geom`) — the 5 wrong token labels were exactly GT-0→"90" confident-wrongs, i.e. 5 side-grasp demos trained under a "vertical grasp" prompt; removing them should sharpen 90-side following (0.61 → toward paper's 95) | H100, ~17 min | **launched 2026-06-11** (see §5) |
| P1 | hvlv diagnosis: stamp_seal6 0/5 under BOTH prompts = transfer failure, not conditioning; check rollout videos (`debug/0529_videos/hvlv/failure/`), teacher-forcing MAE on new box, consider longer Stage-B (2500-step proxy 0.708 > 1500's 0.667) or a different hvlv taskB env | new box + H100 | open |
| P2 | contact conditioning (proxy 0.58–0.67, closed-loop 0.75@5ep vs paper 90): options = more steps (2500 ≈ 1500, no), conditioning-LR up, pref-segment loss weighting, dedicated pref token | H100 experiments | open |
| P2 | Table-1 place number decision (90 token vs 1.00 geom) | paper editing | user call |

## 4b. ADDENDUM (2026-06-11 PM) — baseline-recognition row + the early-frame probe

### Baseline recognition measured for all 5 cats (raw JSONs now on disk)

`gate_baseline_<cat>_10k.json` = the no-VQA Stage-A ckpt (`pref_oft_baseline_*_10k`)
through the same token-framework readout (φ random ⇒ the honest "no recognition
objective" control; the old box's height/orient JSONs were lost — these replace them
and add the 3 never-measured cats):

| cat | taskA acc (pred_dist) | taskB acc (pred_dist) |
|---|---|---|
| height | 0.50 (all "low") | 0.50 (100/100 "low") |
| orient | 0.36 ("0" 67/80) | 0.43 ("0" 93/100) |
| contact | 0.475 ("25" 78/80) | 0.50 ("25" 74/100) |
| place | 0.50 (all "center") | 0.50 (100/100 "center") |
| hvlv | 0.35 (mixed) | 0.51 (mixed) |

All ≈ chance on taskB, mostly **confident single-class collapse** (recency/token prior),
exactly the expected baseline behavior. Metric design holds: taskB is exactly 50/50
balanced ⇒ a constant predictor scores exactly 0.50 and class prior cannot inflate it;
sub-0.5 taskA readings (orient 0.36, hvlv 0.35) are collapse+anti-correlated noise, to be
read as "no signal" (n=100 ⇒ chance band ≈ 0.40–0.60). SPT row vs baseline row on taskB:
{1.00, 1.00, 0.92, 1.00, 1.00geom/0.90token} vs {0.50, 0.43, 0.50, 0.50, 0.51}.

### The early-frame probe — fixes the Tier-1 proxy's false verdicts

Discovery: the Tier-1 proxy queried at the per-cat DISC_FRAC (orient 0.45 etc.), where the
demo image has already **visually committed** the preference — an image-dominant policy
reproduces the demo regardless of prompt there, so conditioning is invisible (orient main
read 0.375 "not controllable" while closed-loop shows 82°/0–2°). Querying at an EARLY,
pre-commitment frame exposes the true prompt→plan mapping:

| cat (frac) | arm | peak per-dim effect | follow-acc all14 | follow-acc argmax-dim |
|---|---|---|---|---|
| orient (0.10) | main (token95 labels) | 0.863 (dim3) | **1.000** | 1.000 |
| orient (0.10) | main_geom (1.00 labels) | **0.992** (dim3) | **1.000** | 1.000 |
| orient (0.10) | **b0 (Naive FT)** | **0.985** (dim3) | **1.000** | 1.000 |
| contact (0.10) | main_geom | 0.0029 (dim1) | 0.625 | 0.750 |
| contact (0.10) | b0 | 0.0009 | 0.417 | 0.708 |
| hvlv (0.25) | main_geom | 0.0013 | 0.375 | 0.458 |
| hvlv (0.25) | b0 | 0.0024 | 0.458 | 0.542 |

(For reference, at the late frame orient main/main_geom/b0 all read effect ≈ 0.0004 —
pure frame artifact. Probe scripts: `/tmp/orient_wrist_probe*.py`, `/tmp/early_frame_probe_ch.py`;
n=24 eps/cell, 12 per class.)

Three regimes, now cleanly separated offline and consistent with closed-loop where known:
1. **orient: strongly conditioned** (huge effect, 24/24 follow) — the earlier "0.375" was
   methodology, not model. geom100 labels give ~15% larger effect than token95 (0.992 vs
   0.863 on dim3); both saturate follow-acc, so closed-loop must decide if the relabel helps.
2. **contact: weakly conditioned** (~3× the b0 effect but ~300× smaller than orient's;
   follow 0.63–0.75) — a genuine conditioning-strength problem, NOT a frame artifact.
3. **hvlv: unconditioned** (effect ≤ b0's, follow ≤ chance) — genuine failure, on top of
   the stamp_seal6 transfer failure (0/5 success both prompts).

### ⚠ Naive-FT-baseline risk (raised by the b0 rows above)

b0-orient retains FULL offline conditioning (1.000) and b0-height most of it (0.875,
late-frame), inherited from the suffix-conditioned SOURCE phase and NOT washed out by the
1500-step suffix-free target SFT. The paper's FT following numbers (53–67) therefore
cannot be assumed from "nothing ties motion to preference" — for proprioceptive axes the
source conditioning may transfer through Naive FT offline. Closed-loop b0 runs decide;
if closed-loop b0 height/orient also follow, options: (a) train Naive FT longer / as the
literal mixed finetune so washout occurs, (b) report measured FT numbers and reframe the
delta (SPT's edge = relational axes + label-quality + closed-loop robustness; place is the
clean 0.50→1.00 case). Note the 0529 closed-loop precedent: the Stage-A token ckpt (no
target SFT) read the pref but did NOT follow in rollout (drop 0.074 = 0.074) — closed-loop
behaves differently from offline here, so this risk is real but undecided.

## 4c. ADDENDUM 2 (2026-06-11 evening) — first closed-loop Naive-FT cell measured (height); FT-risk RESOLVED via threshold convention

New box up (smoke 1/1; see `0611-5090-new-box-setup.md`). First FT-column measurement,
`pref_stageb_b0_height`, paired same-seed two-prompt, 10 metric eps/prompt (test_num 11,
seed 0), task success **11/11 AND 11/11**:

- prompt high: drop = **0.0885 ± 0.029** (only ~4/10 episodes genuinely lift high; vals 0.061–0.152)
- prompt low: drop = **0.0601 ± 0.004** (tight — gentle-place is the default attractor)
- separation **+0.0283** (= 40% of SPT's +0.0705; SPT 50ep: 0.1327 / 0.0622)

**Threshold convention decides the FT number** (computed from per-episode records;
archive `eval/0611_ctrl/pref_stageb_b0_height/` + `eval/0529_50ep/per_episode/...`):

| follow-threshold convention | SPT (50ep) | b0 / Naive FT (10ep) |
|---|---|---|
| per-run two-arm midpoint (current `build_summary.py`) | 0.81 | 0.80 (flattered — its own midpoint adapts down) |
| fixed @ SPT-run midpoint 0.0974 | 0.81 | 0.65 |
| fixed @ expert midpoint 0.1375 | 0.79 | **0.55** |

Reading: SPT is **threshold-robust** (0.79–0.81 under every convention) because its
modulation is large; b0's apparent 0.80 exists only under the self-referential midpoint.
Under the paper's OWN stated convention ("continuous axes thresholded from source
statistics"), closed-loop height reads **FT ≈ 0.55–0.65, SPT ≈ 0.79–0.81 — consistent
with Table 2's claimed 53 / 81.** The §4b b0-retains-conditioning risk is therefore
RESOLVED for height, with two required follow-ups:
1. **Lock the convention**: fixed source-statistics threshold per axis; update
   `build_summary.py` (currently per-run midpoint) and recompute all archived follow
   numbers under it (SPT numbers barely move; FT numbers are the sensitive ones).
   Compute the true source(taskA)-stats threshold rather than the taskB-demo midpoint
   used here as a stand-in.
2. b0 height task success = 100% (vs paper FT 92/89) — fine for the "success stays high"
   claim; per-axis success cells should come from these same paired runs.

b0_orient ckpt transferred to box (md5 6d8f3b90… verified); paired b0_orient eval
launched (driver `/root/drive_b0_orient_0611.sh`; orient verdicts must be read from
per-episode `ee_x_tilt`, not the analyze_pref default aggregate).

### 4c.1 — b0_orient closed-loop (after the track_step obs-fix; see setup doc fix #13)

First run was metric-invalid (deploy_policy didn't pass `obs` → no EE fields → obj-tilt
fallback). Fixed + rerun, 10 metric eps/prompt, success 10/11 both prompts:

- prompt 0 (side): ee_x_tilt **81.8°** mean, all 10 side grasps
- prompt 90 (top): ee_x_tilt **4.1°** mean, **ALL 10 top-down** (max 9.4°)
- fixed-45° follow: **1.00 / 1.00 → overall 1.00** (archive `eval/0611_ctrl/pref_stageb_b0_orient/`)

**Matched-seed comparison kills the IK narrative**: on the SAME first-10 seed-walk
episodes, SPT main_orient (token-0.95 labels) flips only **5/10** under prompt-90
(84.4, 1.4, 0.4, 2.3, 83.4, 79.0, 0.7, 85.6, 78.0, 24.6) vs b0's 10/10. The 0529
"IK-marginal spawns" explanation for SPT's 90-side losses is falsified — b0 executed
top-down grasps on those very spawns. The remaining suspect is exactly the **5 wrong
token labels** (GT-0 demos trained under the "vertical grasp" prompt), i.e. the
mechanism the orient_geom relabel removes.

**Status of the FT-risk per axis (closed-loop):** place RESOLVED-contrast (b0 0.50
offline), height RESOLVED via fixed-threshold convention (FT 0.55–0.65 vs SPT 0.79–0.81),
**orient REALIZED — FT follows at 1.00 on this taskB**, ≥ SPT-token's 0.75 (matched
10) / 0.81 (49ep).

### 4c.2 — orient_geom closed-loop: label-noise mechanism CONFIRMED (2026-06-11 ~08:35Z)

`pref_stageb_main_orient_geom` (this morning's retrain on the 1.00 geom labels), same
paired protocol, same first-10 seed walk (archive `eval/0611_ctrl/pref_stageb_main_orient_geom/`):

| arm (matched 10 seeds) | prompt-0 ee_x | prompt-90 ee_x | fixed-45° follow | success |
|---|---|---|---|---|
| SPT token-0.95 labels | ~81° all side | 5/10 flip | 1.00 / 0.50 → **0.75** | (49ep: 48/50, 43/50) |
| **SPT geom-1.00 labels** | 81.6° all side | **10/10 flip (3.5° mean, max 9.0°)** | 1.00 / 1.00 → **1.00** | 11/11, 9/11 |
| b0 (Naive FT) | 81.8° all side | 10/10 flip (4.1°) | 1.00 / 1.00 → **1.00** | 10/11, 10/11 |

Conclusions:
1. **Label quality → controllability is now a measured causal chain on orient**: labels
   0.95 → 1.00 lifts closed-loop follow 0.75 → 1.00 on identical spawns. The 5 wrong
   token labels (GT-0 demos trained under the "vertical grasp" prompt) were the cause of
   SPT's 90-side losses; "IK-marginal spawns" (0529) is dead as an explanation.
   Combined with place (vanilla-0.61 labels → 0.75; geom-1.00 → 1.00; b0 → 0.50 offline),
   this is a strong, paper-grade **label-quality ablation row**.
2. **The orient FT-vs-SPT contrast does not exist on move_can5_away at n=10**:
   FT = SPT(geom) = 1.00. Options for Table 2's orient row (user decision):
   (a) extend both arms to the full ~50-seed list (later spawns may separate them —
   token did *better* on later seeds, so not guaranteed); (b) choose a different orient
   target task where source-conditioning does not transfer through Naive FT; (c) report
   orient at ceiling for both and let the contrast live on the other axes + the
   label-quality ablation. Use `pref_stageb_main_orient_geom` as the SPT-orient ckpt
   going forward either way.

## 4d. OFFICIAL follow numbers locked (2026-06-11 night, user-approved convention D4)

Fixed source(taskA)-statistics thresholds computed (`eval/source_follow_thresholds.json`):
drop_height **0.1245** (high 0.1795±.006 / low 0.0694±.031), ee_x_tilt **41.2°**,
height_fraction **0.484**, place_offset **0.0536 m**, detour **0.1501 m**.
`build_summary.py` now emits BOTH conventions and scans `eval/0611_ctrl/` too. Official table:

| ckpt | n/prompt | follow (midpoint) | **follow (FIXED src-thr)** | success |
|---|---|---|---|---|
| SPT height | 49 | 0.81 | **0.84** | 50/50, 41/50 |
| Naive-FT height | 10 | 0.80 | **0.55** | 11/11, 11/11 |
| SPT orient (token labels) | 49 | 0.81 | **0.81** | 48/50, 43/50 |
| SPT orient (geom labels) | 10 | 1.00 | **1.00** | 11/11, 9/11 |
| Naive-FT orient | 10 | 1.00 | **1.00** | 10/11, 10/11 |
| SPT hvlv (geom) | 4 | 0.50 | **0.50** | 0/5, 0/5 |
| SPT contact (geom) | 4 | 0.75 | n/a† | 4/5, 4/5 |

† contact closed-loop records used the meters fallback (EE.z−obj.z; contact points
unavailable on the old box) while the source threshold is a height_fraction — re-run
contact closed-loop with working contact-point span (or add an EE-fallback source thr).

**Height row vs paper Table 2: measured FT 0.55 / SPT 0.84 ↔ claimed 53 / 81. Match.**

## 5. Launched today (H100)

- `ctrl_{height,orient}_1500.json`, `ctrl_b0{height,orient}_1500.json` — Tier-1 proxy (§2).
- `pref_pseudo_labels_orient_B_geom.json` — orient geom cache, gate PASS 1.000.
- `pseudo_label_geom.py` extended with `--cat orient` (wrist R22 feature).
- `pref_stageb_main_orient_geom` Stage-B retrain from the geom labels (yaml
  `starvla_pref_stageb_main_orient_geom.yaml`) — DONE, final `action_dit_loss` 0.0115
  (≈ token version's 0.0114), 6 ckpts. Offline: late-frame proxy 0.542 vs token's 0.375
  (`ctrl_orient_geom_1500.json`; both depressed by the frame artifact), early-frame
  effect 0.992 vs 0.863 (§4b). Decide via paired closed-loop orient_geom-vs-orient on
  the new box.
- `gate_baseline_{height,orient,contact,place,hvlv}_10k.json` — baseline recognition
  row, all 5 cats (§4b).
- `eval/0529_50ep/summary.{md,json}` regenerated (place row still absent — its per-episode
  records are gone; README aggregate is the only surviving place record).
