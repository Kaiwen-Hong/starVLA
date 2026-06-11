# 2026-05-29 — Stage B: pseudo-labeling + continue-train + eval plan (READ THIS FIRST)

> Self-contained handoff for the Pref-VLA **Stage B** work. If you are a fresh
> chat window with none of yesterday's context, this doc + the two cross-links
> below are enough to pick up.
>
> Cross-links:
> - `r-preference/doc/0528-token-vqa-fix-and-overnight-runbook.md` — how the
>   Stage-A **token-VQA** was built (EE-pose soft token, `state_mode=token`) and
>   the 10 overnight runs.
> - `r-preference/doc/0524-stageB-contact-plan.md` — the original Stage-B design
>   (frozen labeler, warm-start a copy, **no VQA loss**, GT-leakage firewall).
>   NB: that doc was written for `Qwen_PI` + `mid_8` + a 35k ckpt; today's work
>   **ported the same plan to `QwenOFT` + token-VQA + 10k ckpts**.
>
> Box: this was done on the **H100** (`/home/kaiwenh/starVLA`, user `kaiwenh`).
> `results/` is a symlink → `/mnt/localssd/kaiwenh/starVLA_runs/results`.

---

## 0. TL;DR + status

Stage B = use the **frozen Stage-A token-VQA as an OFFLINE pseudo-labeler** on
taskB (unseen-object) demos → conf-filter → cache JSON; then **warm-start a
separate policy copy** and continue-train on taskB conditioned on the
pseudo-pref text suffix `"... Preference: <label>"`, **action L1 loss ONLY** (no
VQA loss, no distillation).

| Thing | Status |
|---|---|
| Stage-B infra ported to OFT + token-VQA (labeler, dataset, YAML-gen, eval, queue) | **DONE** |
| DeepSpeed micro-batch auto-detect bug on `pref_hdf5_stageb` | **FIXED** |
| Pseudo-label reliability gate (5 cats) | **height 1.00 / orient 0.95 PASS; contact 0.63 FAIL** (place/hvlv FAIL — see §2) |
| Stage-B **main** continue-train | **DONE for height + orient** (the only two that pass the label gate) |
| Stage-B **b0** control | **NOT DONE** (H200 env bug + cross-machine copy broke — see §4.3) |
| Tier-1 offline controllability proxy | **built, NOT yet run** (GPUs free on H100 now) |
| Tier-2 closed-loop rollout (5090 RoboTwin box) | **NOT yet run** — the real test |

**Headline result:** the token-VQA labeler is reliable **only for the
proprioceptive prefs (height, orient)**. The **relational** prefs (contact,
place, hvlv) fail because the labeler sees **only the absolute active-arm EE
pose** and the 0526 hdf5 has **no object/target/obstacle pose** (pointcloud is
empty, shape `(T,0)`). So **Stage-B main is viable for height + orient only** —
a real, honest limitation, not a bug. Details in §2.

**Immediate next step (cheap):** run the Tier-1 offline proxy
(`pref_follow_eval.py`) on `main:height` + `main:orient` to sanity-check
conditioning before doing the closed-loop rollout. See §5–§7.

---

## 1. What Stage B is (and the no-distillation clarification)

Three lines (per `0524-stageB-contact-plan.md` §1):

- **main** — warm-start from the **token-VQA Stage-A** ckpt; condition on the
  pseudo-pref suffix (`with_pref_suffix=True` + `pseudo_label_cache`).
- **b0** — warm-start from the **OFT-baseline Stage-A** ckpt; **no** pref suffix
  (`with_pref_suffix=False`). The conditioning-ablation control.
- **ablation** — deferred (not built today).

**No distillation, no VQA loss in Stage-B training.** The VQA is used *purely
as an offline labeler* in step (1); the training in step (2) is plain `QwenOFT`
with only the flow-matching action loss (`action_dit_loss`). Verified:
`grep -rni "distill"` across `examples/ starVLA/ r-preference/` = **0 hits**.

**GT-leakage firewall (load-bearing).** taskB directory names embed the GT pref
key (e.g. `place_playingcards1_box_high`, `move_can5_away_0`). The Stage-B
dataset must NEVER read the dir-name pref for prompt construction. Implemented in
`PrefHDF5StageBDataset` (`examples/preference/dataset/pref_hdf5_stageb_dataset.py`):

- `__getitem__` parses the dir-name pref into a var literally named
  `pk_DIR_IGNORE_THIS` and never uses it; the label comes from the cache
  (`action_prompt_label`) only.
- `_load_cache` applies a strict **whitelist**
  `{action_prompt_label, decision, pref_key}` — even though the cache JSON also
  contains `gt_pref_key` / `gt_match` (written for the launch gate), those are
  dropped on load and a `RuntimeError("firewall broken: ...")` fires if anything
  outside the whitelist survives.
- Episodes with cache `decision != "keep"` are dropped from `_index` (don't
  appear in `__len__` or `__getitem__`).
- **b0** mode never reads the cache at all; it uses the pref-independent base
  template and strips any `" Preference: ..."` suffix.

---

## 2. Pseudo-label reliability — the result + root cause (the headline)

Run with `examples/preference/stage_b/pseudo_label_token.py`: builds
`QwenOFT_VQA` from a Stage-A **token** YAML + ckpt, then for each taskB episode
calls `load_clip_and_state(...)` (3-frame head-cam clip + active-arm EE pose) →
`model.predict_preference(clip, state)` → conf-filter (`min_conf=0.55`) → writes
the cache JSON. Launch gate = post-filter acc **≥ 0.95** (else `SystemExit(2)`).

### 2.1 Measured accuracy (pseudo vs dir-GT)

Verified by reading the cache JSONs on H100:

| cat | overall acc | post-filter acc | kept | pseudo-class bias | gate |
|---|---|---|---|---|---|
| **height** | **1.00** (100/100) | 100/100 | 100 | balanced (50 high / 50 low) | **PASS** |
| **orient** | **0.95** (95/100) | 95/100 | 100 | 45 `0` / 55 `90` | **PASS** |
| **contact** | **0.63** (63/100) | 62/99 = 0.626 | 99 | biased **high**: 87 `75` / 12 `25` | **FAIL** |
| **place** | ~0.90 (see note) | — | — | biased "corner" | **FAIL** |
| **hvlv** | ~0.50 (see note) | — | — | collapses to ALL `hv` | **FAIL** |

- **height** cache: all entries `decision=keep`, `gt_match=true`, `conf=1.0`,
  `logit_gap≈20.7`. Textbook.
- **orient** 5 wrong are all the **same failure mode**: GT `0` (horizontal) →
  predicted `90` (vertical), and all **confidently** wrong (conf 0.92–0.999).
  Episodes: `move_can5_away_0/episode{7,27,31,33,38}`. The filter can't catch
  them because they're high-confidence; still ≥0.95 so it passes.
- **contact** is biased to "high" (87 of 99 kept labeled `75`) → 0.626, fails.

> ⚠ **Correction to the brief / what actually exists on H100:** only
> `pref_pseudo_labels_{height,orient,contact}_B.json` exist in
> `r-preference/eval/`. **`place_B.json` and `hvlv_B.json` were NOT produced on
> this box** — the **place/hvlv token Stage-A ckpts are not on H100** (only
> `pref_oftvqa_token_{height,orient,contact}_10k` are here; place/hvlv token runs
> were on the H200 per the 0528 doc). The place≈0.90 / hvlv≈0.50 figures are from
> the design discussion / prior probes, not a cache file on H100. Either way the
> conclusion (place + hvlv FAIL, relational) stands; just don't expect those two
> JSONs to be present until the labeler is run where their ckpts live.

### 2.2 Root cause (confirmed against the data)

The token-VQA inputs **only the ABSOLUTE active-arm EE pose** — 9D = xyz(3) +
6D-rotation(6), z-scored on taskA stats — alongside a 3-frame head-cam clip. It
has **no object / target / obstacle state**. Verified in
`examples/preference/dataset/vqa_sample.py`:

- `ee_pose_9d_at(h5, arm, indices)` (line ~406) = `[xyz(3), 6D-rot(6)]` of the
  active arm; `_active_arm_key` picks the arm with the larger xyz range.
- `predict_preference` (in `starVLA/model/framework/vqa_cotrain_mixin.py`
  line ~335, the `VQACotrainMixin` that `QwenOFT_VQA` mixes in) prepends the
  soft-token marker and injects `phi(state)` when `state_mode == token`.

And the **0526 taskB hdf5 has no object pose**. Verified by inspecting
`/mnt/localssd/kaiwenh/pref/data/0526/height/taskB/.../episode0.hdf5`:

```
endpose/left_endpose   (T,7)   # xyz + quat_xyzw, per arm
endpose/right_endpose  (T,7)
endpose/{left,right}_gripper (T,)
joint_action/{left_arm(T,6), left_gripper(T,), right_arm(T,6), right_gripper(T,), vector(T,14)}
observation/{head,left,right,front}_camera/rgb   # + intrinsics/extrinsics
pointcloud             (T, 0)   # <-- EMPTY. no object/obstacle/target geometry
```

**Why the split is what it is:**

- **PASS (proprioceptive):** `height` = drop height (absolute EE Z at release),
  `orient` = wrist orientation at grasp. Both are readable from the **absolute
  EE pose alone** → transfer to unseen objects.
- **FAIL (relational):** `hvlv` = avoidance **margin vs the obstacle**,
  `place` = placement **location vs the target**, `contact` = grasp height vs a
  **taller object** (the absolute Z baseline shifts with object height). All need
  the **other body's** pose, which isn't in the EE state and isn't in the hdf5
  (pointcloud empty). So they can't transfer.

This matches the Phase-1 standalone probes (`r-preference/phase1/probe_*_lmhead.json`,
the MLP-on-clean-state upper bounds referenced in `0528-...-runbook.md` §1.4):
the proprioceptive cats are separable from EE state; the relational ones are
fundamentally limited without object/obstacle features.

### 2.3 Implication

- **Stage-B main is viable for `height` + `orient` only.**
- `contact` / `place` / `hvlv` main are **LABEL-BLOCKED**. This is a real
  limitation: fixing it needs **object/obstacle/target pose** — which requires
  **re-collection** (the 0526 hdf5 won't get it) or a **perception module**, or
  **object-relative features**. The empty pointcloud means there's no free lunch.

---

## 3. What was built / fixed today

All paths relative to `/home/kaiwenh/starVLA`. Each verified to exist + read.

| File | What |
|---|---|
| `examples/preference/stage_b/pseudo_label_token.py` | **NEW.** Token-VQA offline labeler (see §2). Replaces the old `pseudo_label_offline.py` (which was QwenPI + head-cam clip strategies — superseded). Writes cache JSON with `action_prompt_label / pref_key / decision` (+ `gt_*` for the launch gate, which the dataset firewall ignores). `--launch_gate_min_acc 0.95`. |
| `examples/preference/dataset/pref_hdf5_stageb_dataset.py` | **FIXED.** `get_pref_stageb_dataset` now passes `action_space` through (was defaulting to `"ee"`; OFT needs `joint`/14D — see factory line ~290). Plus the firewall + main/b0 `with_pref_suffix` logic (§1). |
| `examples/preference/train_files/gen_stageb_yamls.py` | **NEW.** Generates the 10 Stage-B YAMLs from the `oft_baseline` templates. main → token-VQA ckpt + suffix + cache; b0 → baseline ckpt + no suffix. Both: `max_train_steps=1500`, `warmup=150`, `save_interval=250`, LR base/qwen `1e-6` & action `1e-5` (i.e. /10 vs Stage A). Drops the `vqa_state_proj` LR key (no VQA in Stage B). |
| `examples/preference/train_files/starvla_pref_stageb_{main,b0}_{height,orient,contact,place,hvlv}.yaml` | **NEW** (10 files). See §3.1 for the real hyperparams. |
| `examples/preference/dataset/precompute_joint_stats.py` | **NEW.** Computes 14D joint (qpos) action/state stats over the TRAIN split (seed 42, val_fraction 0.2). Reproduces `stats_place_0526_joint.json` exactly. |
| `examples/preference/dataset/stats_{contact,hvlv}_0526_joint.json` | **NEW** (created 05-28). height/orient/place joint stats already existed. |
| `examples/preference/stage_b/pref_follow_eval.py` | **NEW.** FK-free offline controllability proxy (Tier 1, §6). 14D-joint-compatible (no forward kinematics). |
| `r-preference/overnight/stageb_queue.sh` | **NEW.** Host-agnostic sequential Stage-B queue runner. Jobs `"main:cat"` / `"b0:cat"`. conda + `CUDA_HOME`, VLM/log discovery, `accelerate launch` w/ `deepspeed_zero2`. |
| `starVLA/training/train_starvla.py` | **FIXED** a real DeepSpeed bug — see §3.2. |

### 3.1 Real Stage-B YAML hyperparams (read from `..._main_height.yaml` + `..._b0_height.yaml`)

Both main + b0 share:

```yaml
framework:
  name: QwenOFT                       # plain OFT — NO VQA in Stage B
  action_model: {action_dim: 14, state_dim: 14, action_horizon: 50,
                 future_action_window_size: 49, action_model_type: DiT-B}
datasets.vla_data:
  dataset_py: pref_hdf5_stageb
  action_space: joint                 # 14D qpos (the dataset-factory fix)
  data_root_dir: /mnt/localssd/kaiwenh/pref/data/0526/<cat>/taskB
  stats_json_path: examples/preference/dataset/stats_<cat>_0526_joint.json
  per_device_batch_size: 8
  task_groups: [<taskB group>]        # height: place_playingcards1_box
  pref_keys: [<keyA>, <keyB>]         # height: [high, low]
  split_val_fraction: 0.0             # Stage B uses ALL episodes
trainer:
  max_train_steps: 1500
  num_warmup_steps: 150
  save_interval: 250                  # -> ckpts at 250/500/750/1000/1250/1500
  eval_interval: 100000               # effectively off
  learning_rate: {base: 1e-6, qwen_vl_interface: 1e-6, action_model: 1e-5}
  lr_scheduler_type: cosine_with_min_lr   (min_lr: 1e-7)
```

main vs b0 differences:

| | main | b0 |
|---|---|---|
| `with_pref_suffix` | `true` | `false` |
| `pseudo_label_cache` | `r-preference/eval/pref_pseudo_labels_<cat>_B.json` | (absent) |
| `pretrained_checkpoint` | `results/Checkpoints/pref_oftvqa_token_<cat>_10k/checkpoints/steps_10000_pytorch_model.pt` | `results/Checkpoints/pref_oft_baseline_<cat>_10k/checkpoints/steps_10000_pytorch_model.pt` |

The `pretrained_checkpoint` is a repo-relative `results/` path so it resolves on
whichever machine holds that Stage-A ckpt.

### 3.2 The DeepSpeed fix (verified at `train_starvla.py` lines ~145–157)

DeepSpeed's `train_micro_batch_size_per_gpu: 'auto'` is normally resolved by
`accelerate` from the dataloader during `prepare()`. For the `pref_hdf5_stageb`
dataloader that auto-detection **fails** and `prepare()` raises (Stage-A
`pref_hdf5` / `pref_hdf5_vqa` worked fine — so it's path-specific). The patch in
`prepare_training()` sets it explicitly **before** `setup_distributed_training`:

```python
_dp = getattr(self.accelerator.state, "deepspeed_plugin", None)
if _dp is not None and _dp.deepspeed_config.get("train_micro_batch_size_per_gpu") == "auto":
    _dp.deepspeed_config["train_micro_batch_size_per_gpu"] = int(
        self.config.datasets.vla_data.per_device_batch_size)
```

Note: the dataloader itself was verified **valid** (10071 samples, batch_size 8)
— this was purely the `auto` detection, not bad data. This fix is necessary but
**not sufficient on H200** (see §4.3).

---

## 4. Stage-B main training results

### 4.1 What ran

Ran on **H100** via `r-preference/overnight/stageb_queue.sh main:height main:orient`.
Queue log `/mnt/localssd/kaiwenh/logs/stageb_queue_instance-20260316-physical-h100-2.log`:

```
START pref_stageb_main_height  00:37:37 UTC  -> DONE 00:54:33
START pref_stageb_main_orient  00:54:33 UTC  -> DONE 01:11:27
==== STAGEB QUEUE COMPLETE 2026-05-29 01:11:27 UTC ====
```

### 4.2 Final losses (from the per-run logs)

| run | final `action_dit_loss` | steps | ckpts |
|---|---|---|---|
| `pref_stageb_main_height` | **0.00811** | 1500 | 250/500/750/1000/1250/1500 ✓ |
| `pref_stageb_main_orient` | **0.01137** | 1500 | 250/500/750/1000/1250/1500 ✓ |

Both converged. Checkpoints verified present (6 each, ~9.8 GB):
`/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_stageb_main_{height,orient}/checkpoints/steps_*_pytorch_model.pt`.

### 4.3 b0 control — NOT DONE

- b0 was launched on **H200** but **FAILED**: `pref_hdf5_stageb` `prepare()`
  returns a **None** dataloader on H200 (`TypeError: 'NoneType' object is not
  iterable` at `_create_data_iterators`), **even with the §3.2 micro-batch fix**.
  H200-env-specific — Stage-A `pref_hdf5` ran fine on H200; H100 trains stageb
  fine.
- Cross-machine ckpt copy (H200→H100) repeatedly broke (rsync broken pipe), so
  the b0 runs aren't available on H100 either.
- **Net: b0 controls are NOT trained.** The "before-Stage-B" control to compare
  against is the **Stage-A token ckpts** (`pref_oftvqa_token_{height,orient}_10k`).
- TODO: either debug the H200 stageb env, or run b0 on H100 — which needs the
  **OFT-baseline-{height,orient}-10k** ckpts. `pref_oft_baseline_{height,orient}_10k`
  **are present on H100** (verified in `results/Checkpoints/`), so b0:height +
  b0:orient can in fact be run on H100 right now via
  `stageb_queue.sh b0:height b0:orient`.

---

## 5. CHECKPOINTS TO EVALUATE (this is what the next window needs)

Model is `QwenOFT`, `action_space=joint` (14D qpos), `action_horizon`/chunk = **50**.

| cat | ckpt (on H100) | YAML | taskB task | base instruction |
|---|---|---|---|---|
| **height** | `/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_stageb_main_height/checkpoints/steps_1500_pytorch_model.pt` | `examples/preference/train_files/starvla_pref_stageb_main_height.yaml` | `place_playingcards1_box` | `"Place the playing cards into the box."` |
| **orient** | `/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/pref_stageb_main_orient/checkpoints/steps_1500_pytorch_model.pt` | `starvla_pref_stageb_main_orient.yaml` | `move_can5_away` | `"Move the can away."` |

Earlier ckpts (`steps_250/500/750/1000/1250`) exist for sweet-spot picking —
**taskB is only 100 demos, watch for overfit** (pick by the eval, not by final loss).

### Pref-conditioned rollout prompts

```
height  HIGH : "Place the playing cards into the box. Preference: high drop"
        LOW  : "Place the playing cards into the box. Preference: low drop"
orient  HORIZ: "Move the can away. Preference: horizontal grasp"
        VERT : "Move the can away. Preference: vertical grasp"
```

(`Preference:` labels are the `action_prompt_label` from
`PREF_CATEGORIES[<cat>].pref_labels`; the labeler/dataset use the exact same
strings.)

---

## 6. Evaluation plan (2-tier)

### Tier 1 — offline controllability proxy (fast filter, NOT yet run)

`examples/preference/stage_b/pref_follow_eval.py` — FK-free, action-space-agnostic
(works on 14D joint, no forward kinematics). For each taskB episode at a
pref-discriminative frame, predict actions under the HIGH vs LOW pref prompt and
compare to the demo chunk:

- **effect** = `mean|a_hi − a_lo|` — does the policy respond to the pref at all?
- **follow-acc** = fraction where the GT-pref prediction is **closer (L1)** to the
  demo chunk than the flipped pref.
- Controllable ⇒ **effect ≫ 0 AND follow-acc ≫ 0.5**. A b0 (no-suffix) policy
  should show **effect ≈ 0**.

Per-cat discriminative frame fraction (`DISC_FRAC` in the file):
`height 0.85, orient 0.45, contact 0.45, place 0.85, hvlv 0.6`.

Example invocation (from the file header):

```bash
python -m examples.preference.stage_b.pref_follow_eval --cat height \
  --yaml examples/preference/train_files/starvla_pref_stageb_main_height.yaml \
  --ckpt results/Checkpoints/pref_stageb_main_height/checkpoints/steps_1500_pytorch_model.pt \
  --taskB_root /mnt/localssd/kaiwenh/pref/data/0526/height/taskB --tag main_height
```

**This is only a filter.** If it fails we know it's not conditioning — this is
exactly where the prior QwenPI Stage-B died (sign-acc 0.62, see
`r-preference/doc/0525-temp-stageb-analysis.md`). Passing it does **NOT** prove
the policy is usable.

### Tier 2 — closed-loop rollout (the REAL test, NOT yet run)

On the **4×5090 eval box** (`ssh -p 18970 root@99.148.65.10`) running RoboTwin:
deploy the ckpt, roll out under HIGH vs LOW pref, measure (a) **task success**
and (b) **actual preference-following** (e.g. real release height high > low for
the height cat). The joint-action bridge needs **identity-reorder + chunk_step=50**
(the existing 5090 setup). **The offline proxy ≠ knowing it works — closed-loop
is required** to know whether there's a problem.

---

## 7. Open items / next steps

1. **Run Tier-1 proxy** on `main:height` + `main:orient` (GPUs free on H100 now)
   — quick sanity before any rollout. (§6 Tier 1.)
2. **Closed-loop rollout** of `main:height` / `main:orient` on the 5090 box (the
   real eval) — compare HIGH vs LOW pref, and vs the Stage-A "before" token ckpt.
   (§6 Tier 2.)
3. **Relational cats (contact / place / hvlv):** labeling is **blocked** by the
   missing object/obstacle/target pose (pointcloud empty). Fix needs re-collection
   or a perception module or object-relative features. **Document as a limitation**
   — do not chase it without new data. (§2.)
4. **b0 controls:** either debug the H200 stageb env, OR run **b0 on H100**
   (`pref_oft_baseline_{height,orient}_10k` ckpts are already on H100) via
   `stageb_queue.sh b0:height b0:orient`. (§4.3.)
5. (If place/hvlv ever revisited) the place/hvlv pseudo-label caches don't exist
   on H100 — run the labeler where the `pref_oftvqa_token_{place,hvlv}_10k` ckpts
   live (H200 per the 0528 doc), or copy those ckpts over first. (§2.1 note.)

---

## 8. Key paths + facts

| Thing | Value |
|---|---|
| Repo / box | `/home/kaiwenh/starVLA` on **H100** (`kaiwenh`); `results/` → `/mnt/localssd/kaiwenh/starVLA_runs/results` |
| conda env | `starVLA` (base env has no `h5py`/`torch`) |
| VLM (local, offline) | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` |
| taskB data root | `/mnt/localssd/kaiwenh/pref/data/0526/<cat>/taskB` |
| 0526 hdf5 keys | `endpose/{left,right}_endpose (T,7)`, `endpose/{l,r}_gripper`, `joint_action/{*,vector(T,14)}`, RGB cams, **`pointcloud (T,0)` empty** — NO object pose |
| labeler | `examples/preference/stage_b/pseudo_label_token.py` (gate ≥0.95) |
| Stage-B dataset | `examples/preference/dataset/pref_hdf5_stageb_dataset.py` (firewall) |
| YAML generator | `examples/preference/train_files/gen_stageb_yamls.py` |
| Stage-B YAMLs | `examples/preference/train_files/starvla_pref_stageb_{main,b0}_{height,orient,contact,place,hvlv}.yaml` |
| joint stats | `examples/preference/dataset/stats_<cat>_0526_joint.json` (14D qpos) + `precompute_joint_stats.py` |
| Tier-1 eval | `examples/preference/stage_b/pref_follow_eval.py` |
| queue runner | `r-preference/overnight/stageb_queue.sh` |
| DeepSpeed fix | `starVLA/training/train_starvla.py` ~L145–157 (`train_micro_batch_size_per_gpu`) |
| pseudo-label caches (present on H100) | `r-preference/eval/pref_pseudo_labels_{height,orient,contact}_B.json` (place/hvlv absent — §2.1) |
| Stage-A token ckpts on H100 | `results/Checkpoints/pref_oftvqa_token_{height,orient,contact}_10k/` |
| Stage-A baseline ckpts on H100 | `results/Checkpoints/pref_oft_baseline_{height,orient,place,hvlv}_10k/` |
| Stage-B main ckpts | `results/Checkpoints/pref_stageb_main_{height,orient}/checkpoints/steps_{250..1500}_pytorch_model.pt` |
| logs | `/mnt/localssd/kaiwenh/logs/stageb_queue_*.log`, `stageb_pref_stageb_main_{height,orient}.log` |
| pseudo-label accuracy | height **1.00**, orient **0.95** (PASS); contact **0.626** (FAIL); place/hvlv FAIL |
| Stage-B main final loss | height `action_dit_loss=0.00811`, orient `0.01137` |
| 5090 eval box | `ssh -p 18970 root@99.148.65.10` (RoboTwin); joint bridge = identity-reorder + chunk_step=50 |

### Facts that differ from the original handoff brief (so the brief can be corrected)

1. **place + hvlv pseudo-label cache JSONs do NOT exist on H100.** Only
   `height/orient/contact` were produced here, because the place/hvlv **token**
   Stage-A ckpts are not on H100. The place≈0.90 / hvlv≈0.50 numbers aren't
   grounded in a cache file on this box (the FAIL conclusion still holds).
2. **contact post-filter acc = 0.626** (62/99 kept; 1 episode rejected for low
   conf), not exactly 0.63 over 100. Pseudo-class bias to "high": 87 `75` vs 12
   `25` among kept.
3. **All 5 VQA categories have `state_in_vqa=True`** in `vqa_sample.py` (the
   2026-05-28 token redesign). The inline comments for orient/height/place still
   say "No robot-state input (per user)", but the actual flag is `True` — so the
   token-VQA **does** receive the active-arm EE pose for all 5 cats. The §2
   root-cause is unchanged (the state is *only* the absolute EE pose; no object
   pose anywhere).
4. `predict_preference` lives in `VQACotrainMixin`
   (`starVLA/model/framework/vqa_cotrain_mixin.py` ~L335), which `QwenOFT_VQA`
   mixes in — not directly in `QwenOFT_VQA.py`.

---

## 2026-05-29 (later) — Geometric relational labeler + controllability results (the real ③ eval)

> This section **overturns** the §2 claim that relational cats are label-blocked,
> and adds the controllability eval that §6 only planned. Everything above stays
> as-is (it was true given what we then knew: the token-VQA *labeler* is blind to
> object pose, and the *hdf5* has no object pose). The new facts: (a) object/
> receptacle pose lives in a **`scene_info.json` sidecar** next to each task_group
> (not in the hdf5); (b) a **geometric/trajectory labeler** fixes relational
> labeling (place/contact 1.00, hvlv 0.92); (c) Stage-B main re-trained with those
> labels; (d) **controllability proxy says place WORKS but contact/hvlv do NOT** —
> and good labels turned out to be necessary-but-not-sufficient.

### A. Object/obstacle pose IS in the data — `scene_info.json` (not the hdf5)

Each task_group dir has a `scene_info.json` sidecar. Per episode, `info` holds
`{A}`=object, `{B}`=target, `{a}`=active arm, plus geometry. Verified by reading
the place/contact/hvlv taskB sidecars:

| cat | scene_info `info` keys (verified) | usable relational feature | obstacle/target pose? |
|---|---|---|---|
| **place** | `stand_xy`/`tray_xy`/`pad_xy` (receptacle center), `target_xy` (placement), `{A}{B}{a}`, **`preference`** (GT — NEVER use) | receptacle xy | YES (receptacle) |
| **contact** | `height_fraction` (grasp height as **fraction of object**, object-relative), `grasp_z` (abs), `base_height_fraction`, `drop_clearance`, `give_target`, **`grasp_region`** (GT-ish — don't use), `{A}{B}{a}` | `height_fraction` | YES (object-relative) |
| **hvlv** | only `{A}{B}{C}{a}` — `{C}`=obstacle **IDENTITY** (e.g. `071_can`), **NO obstacle pose** | (none from scene_info) → must be **pose-free** | NO |

So place/contact get a privileged relational feature from `scene_info`; **hvlv
gets nothing from scene_info** and needs a pose-free feature (we use the EE
trajectory, below). GT fields (`preference`, `grasp_region`) are read-and-ignored,
never fed to label/prompt.

### B. A geometric/trajectory labeler fixes relational labeling

`examples/preference/stage_b/pseudo_label_geom.py` (read). Per-cat relational
feature; threshold **fit on taskA** (midpoint of the two class means, `n=40`/leaf),
then applied to unlabeled taskB. Writes the same cache schema the Stage-B dataset
firewall consumes.

| cat | feature (code) | small / large side | privilege |
|---|---|---|---|
| place | `‖EE_release_xy − receptacle_xy‖` (release frac 0.90; receptacle from `scene_info`) | center / corner | sim (receptacle pose) |
| contact | `height_fraction` (object-relative, from `scene_info`) | `25`(low) / `75`(high) | sim (object-relative) |
| hvlv | max perpendicular detour of active-arm xy path from its transport chord (frac 0.25–0.85) | `lv` / `hv` | **none — pure EE trajectory** |

**Labeling accuracy on taskB (pseudo-vs-GT)** — verified by counting `gt_match`/`n`
in the cache JSONs `r-preference/eval/pref_pseudo_labels_{place,contact,hvlv}_B.json`:

| cat | geom labeler taskB acc | pred balance | vs old token-VQA |
|---|---|---|---|
| **place** | **1.00** (100/100) | 50 center / 50 corner | 0.90 |
| **contact** | **1.00** (100/100) | 50 `25` / 50 `75` | 0.63 |
| **hvlv** | **0.92** (92/100) | 42 `hv` / 58 `lv` | 0.50 |

Oracle separability check `r-preference/debug/v2/place_relational_separability.py`
(+ `.png`, both present): place **scene-offset thr-acc 1.00 on taskA AND taskB**
(center ≈0, corner ~5–10 cm), **traj-offset 1.00**. From the separability prints
(noted as measured): contact `height_fraction` taskA 0.97 / taskB 1.00 (S/N 7/16);
hvlv detour taskA 1.00 / taskB 0.94 (S/N 9.3/3.7).

> **For the paper:** the receptacle/object features here are **sim-privileged**
> (from `scene_info`); hvlv used **pure trajectory (no privilege)**. The real-robot
> relational labeler is the **image→token route** — estimate receptacle/object pose
> from the head-cam + the camera intrinsics/extrinsics that **ARE in the hdf5**.
> Feasible because the signal separates at the ~cm precision that route can hit.

### C. Stage-B main re-trained with geom labels

H100, run_ids `pref_stageb_main_{place,contact,hvlv}_geom`, 1500 steps, OFT, geom
caches. Final `action_dit_loss` — verified from
`/mnt/localssd/kaiwenh/logs/stageb_pref_stageb_main_{place,contact,hvlv}_geom.log`:

| run | final `action_dit_loss` |
|---|---|
| `pref_stageb_main_place_geom` (1.00 labels) | **0.00715** |
| `pref_stageb_main_contact_geom` | **0.01417** |
| `pref_stageb_main_hvlv_geom` | **0.01613** |
| `pref_stageb_main_place` (0.90 token labels, earlier) | 0.00622 |
| `pref_stageb_b0_place` (control) | 0.00877 |

**Gotcha (launch-timing collision):** `place_geom` FAILED on the first attempt — the
driver `stageb_driver_geom_h100.log` shows `FAILED pref_stageb_main_place_geom` only
~7 s after START (03:12:33 UTC), i.e. a launch-time crash. The cause was the
`accelerate` default `main_process_port 29500` still being held by the previous
`place` queue's `b0:place` that hadn't exited — a timing collision, not a code bug.
Re-ran fine once the queue freed the port (`stageb_driver_place_geom_retry.log`:
START 03:47:09 → DONE 04:04:40). *(NB: the literal "port 29500 … already utilizing"
stderr was not captured to a readable log on this box — only the fast-FAILED-then-
clean-retry pattern is preserved in the driver logs.)*

### D. HEADLINE — controllability proxy: place WORKS, contact/hvlv do NOT

`examples/preference/stage_b/pref_follow_eval.py` (read) — FK-free offline proxy:
at a pref-discriminative frame, predict the action chunk under `Preference:
<highlabel>` vs `<lowlabel>`; report **follow-acc** (fraction where the GT-pref
prediction's L1-to-demo < the flipped-pref's) and **effect** = `mean|a_hi−a_lo|`.
Verified from `r-preference/eval/ctrl_*.json` (`acc`/`effect`/`per_gt`):

| run | labels | follow-acc | effect | per_gt | verdict |
|---|---|---|---|---|---|
| **place_geom** | 1.00 | **1.00** | 0.057 | center 1.0 / corner 1.0 | **CONTROLLABLE** |
| **place_v090** | 0.90 | **1.00** | 0.053 | center 1.0 / corner 1.0 | works (10% label noise didn't hurt place) |
| **b0_place** (control, no pref suffix in training) | — | **0.50** | 0.054 | center 0.0 / corner 1.0 | doesn't follow (correct control) |
| **contact_geom** | 1.00 | **0.58** | 0.0006 | `25` 0.417 / `75` 0.75 | **NOT controllable** (near-identical actions) |
| **hvlv_geom** | 0.92 | **0.58** | 0.009 | `hv` 0.333 / `lv` 0.833 | **NOT controllable** |

**Interpretation (key): follow-acc — not effect — is the discriminator.** b0 also
has effect ≈0.05 (the VLM reacts to *any* text change) yet follow-acc 0.50; only
place-main has follow-acc 1.00. So the proxy is **VALID** (it cleanly separates
place-main 1.00 vs b0 0.50). **Labeling quality (now fixed for all 3) is NECESSARY
BUT NOT SUFFICIENT:** place's pref (placement *location* = large/late/distinct
action delta) is learnable → controllable; contact (grasp height) and hvlv (detour)
prefs are **subtle** → at 1500 steps the policy averages them out and does not
condition. **CAVEAT:** this is an OFFLINE proxy; the real test is closed-loop
rollout on the 4×5090 RoboTwin box (height rollout was reported strong by the user,
consistent with the proxy showing the placement/location-type prefs controllable).

### E. contact has NEVER been controllable (consistency with 0525)

Cross-checked against `r-preference/doc/0525-temp-stageb-analysis.md`: the OLD
**QwenPI** Stage-B contact was sign-acc **0.62** (31/50), and Stage-A baseline was
**ALSO 0.62** (tied → Stage B added nothing); physical Δz **+2.7 mm** vs demo
**+46 mm** (~5.9%); 95% CI **[47%, 76%]** overlaps chance 0.5. Contact *labeling*
was always good (old VQA 92/92; geom now 1.00) but *controllability* was always
≈chance (QwenPI 0.62, now OFT 0.58). Both action heads — flow-matching (QwenPI) and
L1 (OFT) — fail → it is the **inherent subtlety of the grasp-height pref**, not the
action head.

### F. Frame visualization (debug/v4)

`r-preference/debug/v4/make_viz.py` (adapted from debug/v1) shows the CURRENT VQA
frames (head_camera) for contact (`contact3`, grasp window) and hvlv (`hvlv3`,
transport), both pref classes. **18 PNGs** in `r-preference/debug/v4/`. Observation:
hvlv's pref (detour magnitude) is a **whole-PATH-shape** property that single
late-transport frames structurally cannot show → explains the head-cam single-frame
VQA = 0.50 for hvlv, and why the **trajectory-detour** feature (whole path) works.
contact frames sit at the grasp window. **NB:** frame/VQA-input tuning only affects
VQA self-labeling quality; it does NOT address the Stage-B controllability failure
(that is a policy-conditioning problem, separate).

### G. Infra fixes (this round)

- `starVLA/training/train_starvla.py` — set deepspeed `train_micro_batch_size_per_gpu`
  explicitly (auto-detect failed for `pref_hdf5_stageb`). *(Same fix described in §3.2.)*
- `examples/preference/dataset/pref_hdf5_stageb_dataset.py` `get_pref_stageb_dataset`
  — now forwards `action_space` (was defaulting to `ee`; OFT needs `joint`). *(Same as §3.)*
- **H200 has a `pref_hdf5_stageb` prepare bug** (dataloader `None` even with the
  micro-batch fix) → **all Stage-B run on H100.** Cross-machine ckpt copy: H200's
  `results` symlink → `/mnt/localssd/kevin/starVLA_runs` (NO `/results/` subdir,
  unlike H100); use `rsync --partial -e "ssh -o ServerAliveInterval=15"` (plain
  rsync got broken pipe on the 9.8 GB ckpts).

### H. Checkpoints (verified present, ~9.2 GB each)

All at `/mnt/localssd/kaiwenh/starVLA_runs/results/Checkpoints/<run_id>/checkpoints/steps_1500_pytorch_model.pt`:

`pref_stageb_main_height`, `pref_stageb_main_orient`, `pref_stageb_main_place` (0.90
labels), `pref_stageb_main_place_geom` (1.00), `pref_stageb_main_contact_geom`,
`pref_stageb_main_hvlv_geom`, `pref_stageb_b0_place` — all 7 confirmed on H100.

### I. Bottom line (status table)

| cat | type | best labeler (taskB acc) | Stage-B controllable? |
|---|---|---|---|
| **height** | proprioceptive | token-VQA 1.00 | **YES** (closed-loop strong, per user) |
| **orient** | proprioceptive | token-VQA 0.95 | likely (proxy/metric pending) |
| **place** | relational | geom **1.00** (token 0.90 also works) | **YES** — proxy follow-acc 1.00 vs b0 0.50 |
| **contact** | relational | geom **1.00** | **NO** — follow-acc 0.58 (always ≈chance, also 0525) |
| **hvlv** | relational | geom **0.92** (traj, no privilege) | **NO** — follow-acc 0.58 |

### J. Open items / next

1. **Closed-loop rollout on 5090 for all cats** (the real eval) — especially
   `place_geom` vs `place_v090` vs `b0_place` (does label quality / conditioning
   matter physically).
2. **contact/hvlv controllability is the hard open problem:** ideas = more steps;
   higher conditioning LR; pref as a **dedicated learned token** (not just a text
   suffix); amplify the loss on the pref-relevant trajectory segment (grasp for
   contact / detour for hvlv).
3. **Real-robot relational labeler = image→token** (estimate object/receptacle pose
   from head-cam + camera calib); signal confirmed separable.
