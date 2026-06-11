# 2026-06-12 — hvlv stamp_seal6 closed-loop 0/5: root cause found (eval ran the WRONG HEAD CAMERA) + fix verified

> **Self-contained overnight diagnosis.** Written on the H100 (`/home/kaiwenh/starVLA`),
> work executed on the new 2×5090 eval box (`ssh -p 17278 root@169.40.1.214`, see
> [`0611-5090-new-box-setup.md`](0611-5090-new-box-setup.md)). Companion docs:
> [`0529-5090-eval-and-stageb-controllability.md`](0529-5090-eval-and-stageb-controllability.md)
> (bridge architecture), [`0611-paper-gap-analysis.md`](0611-paper-gap-analysis.md) (§4 P1 item
> "hvlv diagnosis"), [`../eval/0529_50ep/README.md`](../eval/0529_50ep/README.md) (the original 0/5 record).
>
> **TL;DR.** The hvlv Stage-B policies were trained on data collected with a special
> **top-down head camera** (`aloha-agilex-topdown`, 73° pitch — created 2026-05-28 precisely
> because the default 53° camera could not see the HV/LV swing apex), but every closed-loop
> eval — the old box's 0/5+0/5 AND tonight's reproduction — rendered the **default**
> `aloha-agilex` 53° head camera. The policy's main observation was geometrically OOD on
> every frame. Switching the eval env's embodiment to `aloha-agilex-topdown`
> **immediately recovers task success on the same seeds** (first episode succeeds where the
> same-seed default-cam episode failed). A second, independent (smaller) train/deploy gap was
> also found and fixed along the way: the collection pipeline stores hdf5 JPEGs
> **channel-swapped** (cv2.imencode on RGB), so training saw R/B-swapped colors while the
> bridge deployed true colors (global, all 5 categories).
>
> **The 0529 README's hypothesis ("the policy cannot perform the never-trained stamp motion
> — hard taskB transfer") is FALSIFIED**: Stage-B hvlv was SFT'd directly on 100 stamp_seal6
> episodes (`task_groups: [stamp_seal6]` in the run yaml), and with the right camera it
> performs the stamp motion fine.
>
> **Bonus**: under the fixed camera the **steps_2500** ckpt also FOLLOWS the preference
> closed-loop (hv detour 0.145 vs lv 0.070, ≈ expert values; §2.13) — hvlv layer-2 was
> never measurable before tonight and looks solvable by just using the later ckpt.

---

## 1. The problem (as of 2026-06-11 morning)

`pref_stageb_main_hvlv_geom` (QwenOFT 14D joint, chunk-50, geom pseudo-labels, steps_1500)
scored **0/5 task success under BOTH prompts** on its taskB env `stamp_seal6` (old 4×5090 box,
2026-05-29), while height/orient/place/contact policies rolled out fine through the same
bridge. Detour metric separation was noise (hv −0.001 m vs lv). Offline, hvlv conditioning is
absent (early-frame probe effect ≤ b0's) — but that is layer 2; this doc is layer 1: why the
task itself fails.

## 2. Evidence chain (in the order it was gathered)

### 2.1 The failure videos show a localization failure, not a motion-skill failure
`r-preference/debug/0529_videos/hvlv/failure/*.mp4` (old box, head_camera):
- ep0 (seed 100000): left arm approaches, descends NEAR the seal, fails to grasp, retreats,
  hovers for the remaining ~450 steps. Seal never moves.
- ep1 (seed 100001): seal spawns far-right of the image; the arm never approaches it —
  twitches near home and closes the gripper on air at step ~91 (recorded grasp_step=91,
  release=None).
- All episodes time out at n_steps=500. Archived per-episode records
  (`eval/0529_50ep/per_episode/pref_stageb_main_hvlv_geom/`) show grasp_step=0 artifacts
  (first gripper command ≤0.8-open reads as "closed"; `is_*_gripper_open` threshold is >0.8).

### 2.2 Stage-B hvlv trained ON stamp_seal6 — "never-trained motion" is wrong
`results/Checkpoints/pref_stageb_main_hvlv_geom/config.yaml`: `task_groups: [stamp_seal6]`,
data_root `0526/hvlv/taskB`, warm-start from `pref_oftvqa_token_hvlv_10k`. 50 hv + 50 lv
episodes, pseudo-labels `pref_pseudo_labels_hvlv_B.json` (0.92 acc, 8 hv→lv flips).
A model SFT'd 1500 steps on the exact task failing 10/10 ⇒ systematic train/deploy mismatch,
not transfer difficulty.

### 2.3 The demo data itself: stamp = grasp(t≈61) → transport → press → release(t≈190-206)
`stamp_seal6_{hv,lv}/data/episode0.hdf5`: T=224/206, LEFT arm only (right arm constant),
gripper closes at t≈61 (z=0.913), transport at z≈0.99 (hv apex 1.06), press to z=0.933,
gripper REOPENS near the end. So the task does close and reopen the gripper — the env's
`check_success` (seal within ±2 cm of pad + both grippers open + obstacle unmoved) is
reachable by the demo recipe. Success predicate is NOT the blocker (motion fails first).

### 2.4 Train/deploy mismatch #1 (real but secondary): hdf5 images are R/B-swapped
- Demo hdf5 frames decode with WRONG colors: scene_info says pad= "Red" but the decoded
  frame shows a BLUE pad; the coke-can obstacle decodes as a BLUE (Pepsi-looking) can; the
  yellow seal decodes cyan-blue. Magenta pads (R/B-swap invariant) decode correctly —
  the fingerprint of a channel swap.
- Root cause in the collection pipeline: `envs/utils/pkl2hdf5.py images_encoding()` calls
  `cv2.imencode(".jpg", img)` on the sim's RGB array (cv2 assumes BGR) ⇒ stored JPEGs are
  channel-swapped. The data-dir demo mp4s and the eval episode mp4s use ffmpeg `rgb24`
  (true colors) — that asymmetry is how it was caught.
- Scope: GLOBAL (all 5 categories; height taskB demo deck/box decode blue vs the red/wood
  eval appearance). Height/orient/place/contact succeeded closed-loop DESPITE deploying
  true-color obs into swapped-color-trained policies — geometry/contrast cues carried them.
- Fix (box edit, env-gated): `client_joint.py _encode` now supports `STARVLA_SWAP_RB=1`
  → deployed obs channel-reversed to match the training color distribution.

### 2.5 Teacher forcing: the model reproduces demos (and color costs little offline)
Method per 0529 §2.4, script `/root/tf_hvlv.py` on the box (server = hvlv_geom@1500, slim-VLM
strict load, md5-verified). Demo `stamp_seal6_hv/episode0`, training prompt
`'Stamp the seal. Preference: wide detour'` (= its pseudo-label), chunk starts t0∈{0,50,100,150}:

| input coloring | mean MAE (rad) | grasp-close timing in t0=50 chunk |
|---|---|---|
| as-stored (training-like, swapped) | **0.0228** | exact (1.0→0.4→0.0 matches demo) |
| R/B-swapped → true color (deploy-like) | 0.0286 | ~1-2 steps late |

Both ≪ the 0.05 reference MAE ⇒ ckpt/stats/bridge pairing is HEALTHY; the color shift alone
degrades offline prediction only ~25%. (Caveat: episode0 is in the train set — all 50 hv
episodes are; no held-out stamp episodes exist.) This exonerated the ckpt and pointed back
at the rollout-side observation.

### 2.6 Closed-loop control arm: color fix alone does NOT rescue (0/6)
New box, kempner assets incl. the missing `100_seal` (shipped tonight), default camera,
`STARVLA_SWAP_RB=1`: **0/6 (hv) AND 0/6 (lv)**, identical failure signature to 0529
(timeout 500, grasp_step≈0, detour 0.019-0.085 m). First action of the first chunk
`phys[0] = [0.017, 0.84, 0.739, -0.595, …]` — NOT the home pose (demos always start at
home). The policy "sees" a mid-episode-looking scene at t=0 ⇒ the observation itself is OOD.
Archived: `eval/0611_ctrl/pref_stageb_main_hvlv_geom_defaultcam_swap/`.

### 2.7 ROOT CAUSE: the hvlv data was collected with a different head camera
Compared per-frame camera parameters stored in the hdf5s:

| dataset | head intrinsics | head extrinsic (world→cam) |
|---|---|---|
| hvlv taskA+taskB (collected 5/28) | fx=fy=217.28, 320×180 | pitch ≈ **73°** down, cam at z≈**1.55**, y≈−0.20 |
| height taskB (and all other cats) | identical | pitch ≈ **53°** down, cam at z≈1.35, y≈−0.45 |

The collection box's `doc/0528-hvlv-topdown-camera.md` documents it explicitly: a sibling
embodiment **`aloha-agilex-topdown`** (head_camera `position=[-0.032,-0.20,1.55]`,
`forward=[0,0.3,-1.0]`) was created for the entire HV/LV suite because from the default 53°
pitch "the swing apex projects onto the upper half of the head RGB frame and the obstacle
frequently occludes the cup"; §8.1 of that doc explicitly warns that models trained on
topdown data and evaluated under the default camera "will see distribution shift in the
head_camera input". The collection wrappers patch `task_config/<task>.yml` to the topdown
embodiment ONLY during collection and restore the files afterward — so
`task_config/stamp_seal6_hv.yml` carried `embodiment: [aloha-agilex]` at every eval
(old box 0529 and the new box). **Every hvlv closed-loop rollout to date ran a head camera
the policy never trained on.** (Wrist cameras are shared between embodiments; only the head
view differs — but the head view is the only camera that sees the full workspace.)

### 2.8 Fix verified: topdown embodiment → success on the same seeds
Box edit: `task_config/stamp_seal6_{hv,lv}.yml` `embodiment: [aloha-agilex]` →
`[aloha-agilex-topdown]` (originals kept as `*.defaultcam-bak`). Same server, same seeds,
`STARVLA_SWAP_RB=1`:
- First action returns to `phys[0] ≈ home` (as in training).
- **Episode 0 (seed 100003): SUCCESS** — the same seed that failed 20 minutes earlier under
  the default camera. Video shows a clean grasp → transport → press-on-pad → release.
  (Cross-box too: seed 100004 = old-box failure video `hvlv_hv_ep3_seed100004.mp4` → tonight
  a 194-step textbook success. Note tonight's expert walk skips 100000-100002 — expert-phase
  plan failures on this box — so episode indices differ from 0529 but the seed lists overlap.)

### 2.9 Results matrix (all `pref_stageb_main_hvlv_geom` steps_1500, env stamp_seal6_hv, test_num 6, seed 0 ⇒ same expert-filtered seed walk; box = new 2×5090)

| arm | head camera | SWAP_RB | hv success | lv success | behavior |
|---|---|---|---|---|---|
| 0529 old box (the original record) | default 53° | off | 0/5 | 0/5 | never grasps; wanders; all timeout 500 |
| control (tonight) | default 53° | **on** | **0/6** | **0/6** | identical signature ⇒ color fix alone does nothing |
| **fix (tonight)** | **topdown 73°** | on | **3/6** | **3/6** | grasp ≈ t60, transport, press, release ≈ t236-243 in 10/10 recorded eps — the demo script |

Per-episode (fix run; pref-JSON has the N−1 quirk so 5 records per 6-ep arm):

| prompt | per-ep (n_steps, grasp→release, detour m) |
|---|---|
| hv | (82, push-success*, 0.007) ✓ · (194, 58→193, 0.096) ✓ · (500, 64→243, 0.060) ✗ · (240, 63→239, 0.105) ✓ · (500, 61→236, 0.052) ✗ |
| lv | (500, 84→240, 0.086) ✗ · (195, 58→194, 0.077) ✓ · (236, 64→235, 0.066) ✓ · (239, 63→238, 0.057) ✓ · (500, 60→236, 0.055) ✗ |

\* ep0/hv succeeded in 82 steps without a detected gripper-close (carried/pushed the seal to the pad; same seed failed under the lv prompt with a late grasp).

- **Residual failures are placement near-misses**: the seal is released ~at t≈240 just outside the ±2 cm predicate (verified visually, e.g. hv ep2 presses beside the green pad), then the episode idles to the 500 cap. No grasp failures remain.
- **Layer 2 unchanged**: detour separation hv−lv = −0.004 m (hv 0.064, lv 0.068; expert reference on the same whole-trajectory metric: hv 0.109 / lv 0.036). The prompt still does not modulate the trajectory — consistent with the 0611 early-frame probe (hvlv effect ≤ b0). The 0529 "−0.001 m separation" number is void though: it was measured on a policy that never performed the task; tonight's −0.004 m is the first valid measurement.
- Teacher forcing on b0 (`pref_stageb_b0_hvlv`@1500, prompt "Stamp the seal."): as-stored MAE 0.0302, true-color 0.0656 (2.2×) — b0 fits the demos too and is *more* color-sensitive offline than main_geom.

### 2.10 b0 (Naive FT) arm under the fixed camera — motion recovers, success does not

`pref_stageb_b0_hvlv`@1500 (baseline Stage-A start, suffix-free target SFT), topdown +
SWAP_RB=1, same protocol/seeds: **0/6 (hv) and 0/6 (lv)** — but ALL 10 recorded episodes
execute the full stamp motion (grasp ≈ t60-85 → release ≈ t194-238) and then time out as
±2 cm placement near-misses. So the camera fix restores the *motion* for b0 too; b0 is
simply less precise at the press placement than the SPT arm.

**SPT vs Naive-FT task success on hvlv: 6/12 vs 0/12 (pooled hv+lv, same seeds; Fisher
one-sided p ≈ 0.005).** This is a NEW paper-relevant contrast — under the broken camera
both arms were 0, and Table 2's hvlv success cells were unmeasurable. (b0's detour
"follow 70%" in the analyze_pref printout is an artifact of the diluted whole-trajectory
metric on all-timeout episodes; treat as noise.)

### 2.11 Credit assignment: camera-only control (swap OFF) reproduces the fix

`pref_stageb_main_hvlv_geom`@1500, topdown camera, `STARVLA_SWAP_RB=0` (true-color deploy,
i.e. ONLY the camera fixed), hv prompt: **3/6 — identical to the swap+camera arm.**
Same motion signature (grasp 58-84 → release 192-240; successes at n=193/240/241).
⇒ For hvlv closed-loop the **camera accounts for the entire recovery**; the color swap is
measurable offline (TF +25% MAE main, +120% b0) but does not move task success at n=6.
The R/B finding remains a real, global data-pipeline bug worth fixing at the source, and
`STARVLA_SWAP_RB=1` remains the strictly-more-faithful deploy setting.

### 2.12 Full results matrix (env stamp_seal6_hv, test_num 6, seed 0, same expert-filtered seed walk)

| arm | ckpt | head camera | SWAP_RB | hv | lv |
|---|---|---|---|---|---|
| 0529 original (old box) | main_geom@1500 | default 53° | off | 0/5 | 0/5 |
| control (tonight) | main_geom@1500 | default 53° | on | 0/6 | 0/6 |
| **FIX** | main_geom@1500 | **topdown 73°** | on | **3/6** | **3/6** |
| camera-only | main_geom@1500 | topdown 73° | off | 3/6 | — |
| Naive FT | b0@1500 | topdown 73° | on | 0/6 | 0/6 |
| longer Stage-B | main_geom@2500 | topdown 73° | on | **4/6** | **2/6** |

### 2.13 BONUS — the steps_2500 ckpt FOLLOWS the preference in closed loop

`pref_stageb_main_hvlv_geom`@**2500** (same protocol, topdown + swap): task success
hv 4/6, lv 2/6, and — unlike @1500 — a clear **detour separation in the right direction**:

| ckpt | hv detour (mean ± per-ep) | lv detour | separation | follow (per-run midpoint) |
|---|---|---|---|---|
| @1500 | 0.064 [0.007, 0.096, 0.060, 0.105, 0.052] | 0.068 [0.086, 0.077, 0.066, 0.057, 0.055] | −0.004 | 50% (chance) |
| **@2500** | **0.145** [0.176, 0.110, 0.191, 0.195, 0.052] | **0.070** [0.092, 0.080, 0.049, 0.063, 0.068] | **+0.075** | **90%** (hv 4/5, lv 5/5) |

Expert reference on the same whole-trajectory metric: hv 0.109 / lv 0.036 — the @2500
policy's hv detours (0.11-0.195) sit squarely in expert-hv territory while its lv episodes
stay low. Even under a fixed expert-midpoint threshold (0.0726) follow reads 70%
(hv 4/5, lv 3/5). With n=5 records/arm this needs the 50-ep confirmation run, but the
direction is unambiguous: **hvlv conditioning EXISTS at 2500 Stage-B steps and was masked
by (a) the broken eval camera and (b) evaluating the 1500-step ckpt.** This refines the
0611 "hvlv unconditioned" verdict — the early-frame probe and the 5-ep closed-loop both
used steps_1500-era artifacts; the offline proxy already hinted 2500 > 1500
(0.708 vs 0.667). The recommended hvlv SPT ckpt going forward is **steps_2500**.

## 3. Conclusions

1. **Layer-1 root cause: eval-side head-camera embodiment mismatch** (default 53° vs
   training 73° topdown), introduced by the deliberately-temporary embodiment swap during
   the 5/28 hvlv re-collection and never propagated to the eval configs. The 0529
   "transfer failure" interpretation is dead.
1b. **Layer 2 is better than believed**: with the camera fixed, the steps_2500 ckpt shows
   closed-loop preference following (separation +0.075 m, follow 90% per-run-midpoint /
   70% expert-threshold at n=5/arm), while steps_1500 does not. Every prior "hvlv
   unconditioned" reading was confounded by the camera and/or the 1500 ckpt choice.
2. **Secondary global finding: the collection pipeline stores R/B-swapped JPEGs**
   (cv2.imencode on RGB). All five categories trained on swapped colors and deploy true
   colors. hvlv closed-loop shows the swap alone neither causes (camera did) nor fixes the
   failure; offline TF puts its cost at ~25% MAE. Recommended: keep `STARVLA_SWAP_RB=1`
   for hvlv (it matches training exactly) and A/B it for the other categories when
   convenient; longer-term fix the encoder (`cv2.imencode(".jpg", img[:, :, ::-1])`) and
   recollect/retrain consistently.
3. The pref-metric `grasp_step=0` readings in failed runs are an artifact of the >0.8-open
   threshold vs the policy's ~0.79 first gripper command — harmless once behavior is sane,
   but worth knowing when reading the JSONs.

## 4. Box edits made tonight (all on 169.40.1.214, documented per protocol)

1. `/root/ar-research/policy/starvla_joint/client_joint.py` — `_encode` gained the
   env-gated `STARVLA_SWAP_RB=1` R/B swap (default off; legacy behavior unchanged).
2. `/root/ar-research/task_config/stamp_seal6_{hv,lv}.yml` — `embodiment:` →
   `[aloha-agilex-topdown]`; pre-edit copies at `stamp_seal6_{hv,lv}.yml.defaultcam-bak`.
   NOTE: leave the topdown setting in place for all future hvlv evals (it IS the correct
   eval condition for these policies).
3. Checkpoints shipped via GCS relay (md5-verified against H100):
   `pref_stageb_main_hvlv_geom/steps_{1500,2500}` (77ec7110…, 5d72880c…),
   `pref_stageb_b0_hvlv/steps_1500` (87aef5f6…) + run sidecars. Disk freed by deleting the
   4 whitelisted height/orient steps_1500 ckpts (H100 originals verified first).
   `100_seal` assets (53 M) pulled from the collection box kempner tree → box
   `assets/objects/100_seal` (was missing from the 0611 assets subset; without it the env
   cannot load at all).
4. New scratch files on the box: `/root/tf_hvlv.py` (teacher forcing), `/root/probe_env.py`
   (spawn/camera prober), run drivers
   `/root/drive_{hvlv_swap,hvlv_td,b0hvlv_td,hvlv_td_noswap,hvlv_td_2500}_0612.sh`,
   `/root/pull_hvlv.sh` (transfer script), demo episodes `/root/tf_{hv,lv}_ep0.hdf5`
   (kept for future teacher forcing).
5. Operational: killed the leftover `pref_stageb_main_orient_geom` server (its eval was
   archived 0611); GCS staging objects removed after verified pulls; bearer token deleted.

## 5. Next actions

- LAYER 2 (conditioning): steps_1500 shows no separation (−0.004 m) but **steps_2500
  does (+0.075 m, §2.13)** — first action: run the 50-ep paired controllability on
  **main_hvlv_geom@2500** under the topdown camera (the box has the ckpt + server script;
  ~4-5 h at observed episode times). Only if that disconfirms is the training-side lever
  (loss weighting / pref token, 0611-paper-gap-analysis §4 P2) needed. Also re-run the
  early-frame probe on the 2500 ckpt offline (H100) to reconcile it with the closed-loop
  result. Two measurement upgrades for that work:
  1. `pref_metric` hvlv should window the detour on the **grasp→release transport
     segment** (matching the geom labeler's `feat_hvlv`), not the whole trajectory:
     on the demos this doubles the expert separation (hv 0.206 / lv 0.083 segment-windowed
     vs 0.109 / 0.036 whole-trajectory, both with 25-85% chord).
  2. Timeout episodes (n=500 with post-release idling) dilute the whole-traj window
     further — another reason for the segment window.
- Re-run the 0529-style 50-ep paired controllability for hvlv under the topdown camera
  to produce the paper number.
- Decide whether to A/B `STARVLA_SWAP_RB` for height/orient/place/contact (cheap; may
  improve numbers since it removes a real train/deploy gap).
- Fix `pkl2hdf5.py` color encoding for all future collections.
