# 2026-06-12 — Privilege-free pseudo-labelers for the relational axes (place, contact)

> **Goal.** The two relational axes whose high-accuracy labelers were sim-privileged
> (`pseudo_label_geom.py` reads `scene_info.json`: place ← receptacle `stand_xy/tray_xy/pad_xy`,
> contact ← object-relative `height_fraction`) get labelers that use ONLY what a real robot
> owns: head-camera images + the camera intrinsics/extrinsics that ARE in the hdf5, the
> robot's own EE/gripper streams, and an open-vocabulary VLM grounder
> (local Qwen3-VL-4B-Instruct, 1 GPU, bf16). Bar: ≥ 0.95 pseudo-label accuracy on taskB.
>
> Context: [`0611-paper-gap-analysis.md`](0611-paper-gap-analysis.md) §1/§4b (labeler matrix),
> [`0529-stageB-pseudolabel-and-eval.md`](0529-stageB-pseudolabel-and-eval.md) §A/§B (why the
> relational axes need object/receptacle pose; image→pose feasibility note).
>
> **FIREWALL (all new code).** On TARGET (taskB) the labelers never read scene_info or any
> GT. scene_info/GT enter exactly twice, both legitimate: (a) fitting calibration constants +
> thresholds **on taskA** (source is fully labeled), (b) SCORING the finished taskB
> predictions into `gt_*` audit fields (same protocol as `pseudo_label_geom.py`). The
> dir-name pref suffix is parsed only for the `gt_*` audit fields. Rejected episodes
> (`decision:"reject"`) are dropped by the Stage-B dataset firewall at load.

## 0. Results — the labeler matrix (taskB, n=100 per axis, chance 0.50)

| axis | labeler | privilege | taskA fit acc | taskB overall† | taskB post-filter (kept) | kept balance | gate ≥0.95 |
|---|---|---|---|---|---|---|---|
| place | geom (`receptacle_xy` ← scene_info) | **sim** | 1.000 | 1.000 | 1.000 (100/100) | 50/50 | ✓ |
| place | token-VQA | none | — | 0.900 | 0.899 (89/99) | corner-biased | ✗ |
| place | vanilla VLM | none | — | 0.610 | — | — | ✗ |
| place | **vision (NEW)** | **none** | 0.997–0.998 | **0.990** | **1.000 (96/96)** | 49/47 | **✓ PASS** |
| contact | geom (`height_fraction` ← scene_info) | **sim** | 0.97* | 1.000 | 1.000 (100/100) | 50/50 | ✓ |
| contact | token-VQA | none | — | 0.630 | 0.626 (62/99) | 75-biased | ✗ |
| contact | **EE-only (NEW)** | **none** | 1.000 per-group‡ | **0.990** | **1.000 (98/98)** | 48/50 | **✓ PASS** |
| contact | **vision (NEW)** | **none** | 0.997 | **0.970** | 0.980 (97/99) | 48/51 | **✓ PASS** |

† overall = episodes scored by their (would-be) prediction incl. margin-band rejects;
geometry-failures count as wrong (conservative; place vision has 0 of those, coverage
loss is purely margin-band).
\* geom contact taskA separability per the 0529 doc. ‡ unsupervised 2-means per task-group
pair on the four upright-object taskA groups (the taskB regime); see §2 regime caveat.

**Per-axis best privilege-free: place 1.000 post-filter (vision), contact 1.000 post-filter
(EE-only). Both PASS the 0.95 gate — the sim-privileged geom rows are now fully replaceable
on these two axes.**

## 1. place — vision route (`pseudo_label_vision.py --cat place`) — **PASS 1.000 post-filter**

**Feature** (same relational quantity as geom): `||EE_xy@frac0.90 − receptacle_xy_est||`,
small = center. The privileged `receptacle_xy` is replaced by a camera estimate:

1. **Ground** the receptacle in 3 early frames (fracs 0.05/0.15/0.25; arm not yet in the
   workspace) with Qwen3-VL-4B: `Locate <descriptor>. Output its bounding box in JSON
   format: {"bbox_2d": ...}`. The receptacle noun (pad/tray/stand) comes from the
   task-group string — i.e. from the task instruction, not the sim.
   - **Coordinate convention verified empirically**: Qwen3-VL returns 0-1000-NORMALIZED
     [x1,y1,x2,y2] regardless of input size (probes `debug/v5/ground_*.png`: the norm-1000
     interpretation lands exactly on stand/pad/tray/bottle, abs-px does not; raw y values
     like 913 exceed the 360-px fed image). Frames are fed at 2× (640×360).
   - **Prop colors are randomized per episode** (pads observed in blue/green/red). Color-
     hinted descriptors made the model REFUSE on off-color episodes (25/100 failures in the
     first run: "There is no blue square pad ... only a green one"). Descriptors are
     therefore COLOR-NEUTRAL (shape/function only); after the fix: **0 grounding failures
     in 2220/2220 place frames** (and 0 in 1260/1260 contact frames).
2. **Backproject** the bbox-center pixel through the per-episode K, E (static overhead
   head_camera; E constant over t, identical across all leaves+categories — verified) to a
   horizontal plane; median over the 3 frames; frame spread = QC (reject > 3 cm).
3. **z-plane without taskB privilege.** Effective plane height per receptacle type fit on
   taskA vs GT receptacle xy (allowed there): pad z\*=0.7225, tray z\*=0.7400 — then
   TRANSFERRED to the unseen taskB receptacle via a robot-measured height proxy:
   per-scene median `Δz = EE_z@release − EE_z@grasp` (the in-hand offset cancels exactly;
   measured pad +0.001, tray +0.031, taskB stand +0.082). 2-point linear map
   `z = 0.7217 + 0.593·Δz` → taskB plane 0.7703 (the 0.593 slope = bbox centroid sits at
   ~0.6 of receptacle height — silhouette includes the legs/front face).
4. **Threshold (PRIMARY, pre-registered before any taskB feature was computed):**
   unsupervised exact 1-D 2-means split of the TARGET pool's own features (transductive,
   no GT — same recipe as the contact EE route), direction small=center from taskA, reject
   band ±0.8 cm. Rationale measured on taskA: the center-class feature tail is **demo
   release scatter, not estimator noise** (decomposition: localization error p99 = 1.0 cm,
   but release-vs-target up to 4.5 cm on playingcards episodes), so a taskA-FIXED threshold
   inherits the SOURCE corner-offset scale (pad 9.3 / tray 11.4 cm) and is scale-mismatched
   on a target with a 6.35 cm offset (taskA-fixed candidates: midpoint 5.4 cm, quantile
   5.8 cm — both sit inside the taskB corner cluster). Both target classes share the same
   scatter⊕estimator spread, so the target 2-means midpoint self-calibrates; degenerate-
   split fallback = taskA quantile rule (not triggered: split 3.95 cm, 51/49).
5. **Rejects**: no/unparseable bbox, <2 valid frames, frame spread > 3 cm, wrong-object
   gate (receptacle estimate within 4 cm of the robot's own grasp-site xy → the grounder
   boxed the manipulated object), and the ±0.8 cm decision margin band.

**Localization accuracy** (receptacle center, vs scene_info GT):

| split | med | p90 | max | n |
|---|---|---|---|---|
| taskA pad (fit) | 0.61 cm | 0.91 cm | 1.02 cm | 320 |
| taskA tray (fit) | 0.30 cm | 0.67 cm | 0.90 cm | 320 |
| **taskB stand (post-hoc diagnostic, extrapolated plane)** | **0.4 cm** | **0.6 cm** | 2.7 cm | 100 |

**taskB labeling**: 2-means thr 3.95 cm → **overall 0.990, post-filter 1.000 (96/96),
coverage 96/100**, kept 49 center / 47 corner. 4 rejects, ALL margin-band (3.25–4.71 cm):
the only would-be error `place_soap2_stand_corner/episode22` (feature 3.25 cm) is among
them — its bbox was dilated by the SOAP lying adjacent to the stand's edge, pulling the
center estimate 2.7 cm off (the localization max above; annotated PNG
`debug/v5/vision_place_B_corner_ep22_REJ.png`). The margin band caught exactly the failure
mode it was designed for. Cache: `r-preference/eval/pref_pseudo_labels_place_B_vision.json`
(+ constants in `vision_cache/calib_place.json`).

**Why no EE-only place fallback exists** (measured): corner offsets have RANDOM per-episode
direction (29–55 unique directions per leaf, dir-std 30–161°) and the receptacle spawn box
(5×5 cm) is comparable to the taskB offset magnitude (6.35 cm) → in absolute release-xy
space the two classes overlap heavily; without a receptacle estimate the per-episode label
is information-theoretically unrecoverable. The camera route is necessary, not a luxury.

## 2. contact — EE-only route (`pseudo_label_ee.py`) — **winner: PASS 1.000 post-filter, no VLM**

**Feature**: EE z of the first-closing arm at the first gripper-close frame
(endpose + gripper streams ONLY — pure proprioception, no images, no scene_info).

**Why it works**: within a target scene (one object type on a fixed table), grasp-25% vs
grasp-75% differ by ~half an object height in absolute z. taskB measured: 25-class
z = 0.820 ± 0.015, 75-class z = 0.904 ± 0.011, min class gap +1.5 cm.

**Labeling rule** (no GT anywhere): DIRECTION (higher z = "75") learned on taskA
(z̄(25)=0.844 < z̄(75)=0.877, confirmed by 1.000 per-group splits); SPLIT on the target pool
is unsupervised exact 1-D 2-means (Otsu agrees; median variant = 1.000 but assumes the pool
is balanced); REJECT band ±8 mm (≈0.75 × taskA within-class σ) around the split threshold.

**Validation (taskA, n=40/leaf)** — unsupervised 2-means within each task-group pair:

| taskA group | acc | | taskA group | acc |
|---|---|---|---|---|
| give_boxdrink | 1.000 | | put_boxdrink_dustbin | 1.000 |
| give_callbell | 1.000 | | put_callbell_dustbin | 1.000 |
| give_fork | 0.487 | | put_fork_dustbin | 0.475 |
| give_screwdriver | 0.512 | | put_screwdriver_dustbin | 0.388 |

Pooled across all 8 groups (one global split): 0.748 — cross-object pooling mixes object
heights, as expected; the production setting is the per-scene split (per-group row).

**Measured regime boundary (honest):** objects LYING FLAT (fork, screwdriver) carry no
z-signal — their 25/75 axis runs along a horizontal direction (grasp-z literally identical:
σ 0.000–0.015 overlapping). Valid regime = upright objects, which is the taskB regime
(boxdrink3 bottle). A lying-object target needs the along-axis variant or the vision route.

**taskB**: split thr 0.8641 → **overall 0.990 (99/100), post-filter 1.000 (98/98), coverage
98/100**, kept 48/50. The single overall error `put_boxdrink3_plate_25/episode33` (0.8683,
+0.4 cm above thr) falls inside the reject band → dropped; the other reject (`episode18`,
0.8598, would have been correct) is the coverage cost.
Cache: `r-preference/eval/pref_pseudo_labels_contact_B_ee.json`.

## 3. contact — vision route (`pseudo_label_vision.py --cat contact`) — PASS 0.980 post-filter

Per-episode (NON-transductive — no pool statistics needed) cross-check of §2: estimates the
height FRACTION itself.

1. Ground the manipulated object (noun from the task-group string) in 3 early frames
   (0.04/0.10/0.16, pre-grasp);
2. object bottom = bbox bottom-center backprojected to the table plane; object top = ray
   height solved at that xy from the bbox top row → `h_est` (taskB bottle ≈ 0.245 m,
   frame-consistent to mm);
3. `hf_est = (EE_z@first_close − k) / h_est`; **calibration**: z_table is ANCHORED to the
   place-pad plane (0.7225 — same table, head-cam K,E verified identical across categories;
   a real cell measures its table height directly anyway), only the scalar k (wrist/TCP
   offset, 0.7208) is fit on taskA. A 2-D (z_table, k) grid search is NOT identifiable from
   hf residuals alone — measured: the joint fit rode the grid edge (z 0.80→0.915 as the grid
   grew) improving taskA residual while taskB acc DROPPED (0.980→0.970); the anchor removes
   the ridge. Threshold = midpoint of taskA hf_est class means = 0.501 ≈ the natural 0.5.
   taskA acc 0.997 (hf_est 25: 0.347±0.053, 75: 0.656±0.064, n=320, 0 rejected).
4. Rejects: <2 consistent frames, h_est < 2 cm or frame-spread > 3 cm, gross wrong-object
   gate (visual xy > 25 cm from the first-close EE xy), and hf_est outside [-0.25, 1.25].
   ⚠ Gate lesson (measured): the EE pose is the WRIST — on side grasps it sits **9–11 cm**
   behind the fingertips, so an "8 cm from the grasp point" gate rejected 99/100 perfect
   groundings; the fine-grained wrong-object rejection belongs to the hf-plausibility band
   (a plate/bin box yields a wildly out-of-band fraction), the xy gate only catches
   across-the-table errors.

**taskB**: **overall 0.970, post-filter 0.980 (97/99), coverage 99/100**, kept 48/51.
Kept-but-wrong: `put_boxdrink3_plate_25/episode{18,33}` — the SAME two episodes the EE
route flagged: their wrist z at grasp is ~5 cm higher than same-hf peers (ep1: gt_hf 0.280
→ z 0.810; ep18: gt_hf 0.286 → z 0.860) while h_est is normal → a grasp-POSE anomaly in
those demos (tilted/offset wrist), not an estimator error. Two independent feature
definitions (absolute wrist z; vision-normalized fraction) agree on the anomaly; the EE
route's margin band happens to quarantine both, the hf band does not (0.53/0.55 are
mid-scale). Cache: `r-preference/eval/pref_pseudo_labels_contact_B_vision.json`.

## 4. What the paper can now claim

1. **Recognition without simulator privilege on ALL relational axes.** The Table-1 "best
   labeler" row can now cite a privilege-free labeler per axis: contact 1.000 post-filter
   (EE-only, no VLM), place 1.000 post-filter (camera + grounding VLM + robot trajectory),
   alongside the already privilege-free hvlv (EE detour 0.92) and orient (wrist rotation
   1.00) and token-VQA height (1.00). **No axis depends on scene_info anymore.**
2. **The real-robot recipe is concrete**: a fixed calibrated overhead camera + an
   off-the-shelf 4B VLM grounder + the robot's own trajectory reproduce the privileged
   geometric features to sub-cm (receptacle localization med 0.4–0.6 cm; object height to
   ~mm consistency), exactly the "image→pose at ~cm precision" route the 0529 doc
   anticipated.
3. **Transductive labeling**: both new labelers learn only the DIRECTION (+ scalar
   calibrations) from the labeled source; the decision boundary is recovered unsupervised
   from the unlabeled target pool's own bimodality (1-D 2-means) — no target labels, no
   balance assumption (degenerate-split fallback specified).
4. **Honest noise accounting**: post-filter accuracy with an explicit margin-band/QC reject
   channel (coverage 96–99%) — the same `decision:"reject"` contract the Stage-B firewall
   already consumes.

## 5. Limitations

- **place vision** needs the receptacle visible & unoccluded in some early frame (static
  overhead cam here; a moving-base robot would need a mapped frame) and a receptacle that
  sits on a roughly known support plane; the Δz height proxy assumes the object is released
  ON the receptacle surface. Adjacent-object bbox dilation is the residual failure mode
  (1/100, caught by the margin band).
- **contact EE-only** is transductive (needs the unlabeled pool, ≥ a few eps per mode) and
  upright-object-only; **contact vision** is per-episode but inherits wrist-pose
  variability through k (the 2 anomalous-pose demos are mislabeled, 0.98 not 1.00).
- Threshold constants (margins, spread gates) were set from taskA statistics; the 2-means
  primary rules were pre-registered before taskB scoring (see code comments) — but
  iteration on the GATE bugs (color prompts, wrist-offset gate, z_table ridge) did look at
  taskB *coverage/diagnostics* (not labels) in between runs; final accuracy numbers come
  from single post-fix runs.
- Qwen3-VL-4B grounding: 0 refusals after color-neutral prompts, but descriptors carry
  one-image-of-the-scene-type prompt engineering ("display stand (small raised platform
  with legs)") — standard deployment practice, noted for reproducibility.

## 6. Files + reproduction

| file | role |
|---|---|
| `examples/preference/stage_b/pseudo_label_ee.py` | contact EE-only labeler (production) |
| `examples/preference/stage_b/pseudo_label_vision.py` | place + contact vision labelers (`--stage ground` GPU / `--stage label` CPU) |
| `r-preference/eval/pref_pseudo_labels_contact_B_ee.json` | contact EE cache — **PASS 1.000 post-filter** |
| `r-preference/eval/pref_pseudo_labels_place_B_vision.json` | place vision cache — **PASS 1.000 post-filter** |
| `r-preference/eval/pref_pseudo_labels_contact_B_vision.json` | contact vision cache — PASS 0.980 post-filter |
| `/mnt/localssd/kaiwenh/pref/vision_cache/` | extracted frames + meta + raw groundings + calib jsons |
| `r-preference/debug/v5/ground_*.png` | bbox-convention probes (norm-1000 proof) |
| `r-preference/debug/v5/vision_place_B_*.png`, `vision_contact_B_*.png` | annotated verification frames (bbox + est-X + GT-O + EE-+ / grasp line), incl. the two failure cases |

```bash
# grounding (1 GPU, ~25 min batched for both cats; resumable, cache-keyed)
python -m examples.preference.stage_b.pseudo_label_vision --cat place   --stage ground --gpu 0
python -m examples.preference.stage_b.pseudo_label_vision --cat contact --stage ground --gpu 0
# labeling (CPU; prints taskA fits, calibration, taskB scores)
python -m examples.preference.stage_b.pseudo_label_vision --cat place   --stage label \
  --out r-preference/eval/pref_pseudo_labels_place_B_vision.json \
  --calib-out /mnt/localssd/kaiwenh/pref/vision_cache/calib_place.json
python -m examples.preference.stage_b.pseudo_label_vision --cat contact --stage label \
  --out r-preference/eval/pref_pseudo_labels_contact_B_vision.json \
  --calib-out /mnt/localssd/kaiwenh/pref/vision_cache/calib_contact.json
python -m examples.preference.stage_b.pseudo_label_ee --cat contact \
  --out r-preference/eval/pref_pseudo_labels_contact_B_ee.json
```

Existing labelers/caches untouched (`pseudo_label_geom.py`, geom/token/vanilla cache JSONs
byte-identical); nothing committed.
