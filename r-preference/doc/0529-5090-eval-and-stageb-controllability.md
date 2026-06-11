# 2026-05-29 — 4×5090 eval box + Stage-B controllability (READ-FIRST)

> **⚠ BOX UPDATE 2026-06-11**: the rented box described here (`ssh -p 18970 root@99.148.65.10`)
> is **GONE** (connection refused). Everything that lived only on it (`/root/ar-research`,
> `/root/EVAL_HANDOFF.md`, gate/rollout JSONs under `/root/`, the place 5ep per-episode metric
> records) is lost unless noted otherwise. **Survivors**: the `starvla_joint` bridge is backed
> up on the collection box at `kaiwen@100.97.239.33:~/Desktop/research/ar-research_exp/policy/starvla_joint/`
> (incl. `*_desk.sh` variants); aggregated results were archived in `r-preference/eval/0529_50ep/`;
> videos in `r-preference/debug/{v3,0529_videos}/`. A **NEW box** is being set up:
> `ssh -p 17278 root@169.40.1.214` (nvidia-smi shows 2×RTX5090, 126G disk ~32G free,
> pre-existing robomme content — do not touch). Setup notes: `0611-5090-new-box-setup.md`;
> gap analysis + eval plan: [`0611-paper-gap-analysis.md`](0611-paper-gap-analysis.md).
> The architecture/fixes documented below (identity reorder, chunk_step=50, pref_metric)
> remain the authoritative spec for the new box.

> **Self-contained handoff.** Written 2026-05-29 on the **H100** box
> (`/home/kaiwenh/starVLA`, user `kaiwenh`), recording the evaluation work done on a
> rented **4×RTX 5090** box over 2026-05-28/29. A fresh chat with zero prior context
> can pick up from here: the eval box, the client–server joint+preference bridge and
> its two fatal-until-fixed bugs, the VQA gate-eval, the closed-loop rollout success,
> the preference-following metric, the Stage-A "reads-but-doesn't-follow" negative
> result, and the **Stage-B result that resolves it** (stated preference → controllable
> behavior) for height and orient.
>
> **Project:** Pref-VLA — a preference-conditioned VLA. The action stream predicts robot
> actions; an auxiliary **VQA cotrain** teaches the shared VLM to *read* the preference,
> so it can later act as a Stage-B pseudo-labeler. This is the empirical backing for the
> paper's **SPT (Self-labeled Preference Transfer)** claim.
>
> **Companion docs (background, do not duplicate):**
> - [`0528-token-vqa-fix-and-overnight-runbook.md`](0528-token-vqa-fix-and-overnight-runbook.md)
>   — the token-VQA fix + the 10-run overnight ablation that produced the Stage-A token ckpts evaluated here.
> - [`0529-stageB-pseudolabel-and-eval.md`](0529-stageB-pseudolabel-and-eval.md) — the Stage-B pseudo-label training side (H100).
> - [`0528-action-arch-and-oft-vqa-integration.md`](0528-action-arch-and-oft-vqa-integration.md) — the 3 action heads / OFT warm-start.

---

## 0. TL;DR + current status

**One paragraph.** We rented a 4×RTX 5090 box and stood up the full RoboTwin
client–server eval for the joint (14D qpos) Pref-VLA checkpoints. The new
`policy/starvla_joint/` bridge needed **two fixes that each caused 0% until corrected**
(identity reorder; chunk_step=50). With those, the **token-VQA reads the preference
near-perfectly** on held-out objects (height 1.00/1.00, orient 1.00/0.98) while the
no-VQA baseline is degenerate at chance. Closed-loop **task success** is strong for
height, weaker for orient. The key scientific result: a decoupled **preference-following
metric** shows **Stage-A reads but does NOT act** on the preference (prompt high vs low
both release at drop≈0.074), whereas **Stage-B pseudo-label-conditioned SFT makes the
behavior follow the prompt** — height drop 0.157 (high) vs 0.071 (low), and orient
gripper approach `ee_x_tilt` ≈82° (horizontal prompt) vs ≈0–2° (vertical prompt). The
prompt is a real control knob.

| Item | Status | Headline number(s) |
|---|---|---|
| 4×5090 eval box | **UP** | `ssh -p 18970 root@99.148.65.10`, driver 580.95.05, 4× 32 GB |
| Joint+pref bridge | **WORKING** (2 fixes) | identity reorder + `chunk_step=50`; teacher-forcing MAE ~0.05 rad |
| VQA gate-eval (token) | **DONE** | height 1.00/1.00, orient 1.00/0.98 (taskA-val / taskB) |
| VQA gate-eval (baseline) | **DONE** (control) | height 0.50/0.50, orient 0.36/0.42 — degenerate |
| Rollout success (Stage-A token) | **DONE** | height high 100/100, low 50/90; orient 0 60/0, 90 10/0 |
| Pref metric + Stage-A finding | **DONE** | reads (VQA 1.00) but does NOT follow (drop 0.074=0.074) |
| **Stage-B controllability** | **RESOLVED** (height, orient) | height 0.157 vs 0.071; orient ee_x 82° vs ~0–2° |
| Other 3 cats (contact/place/hvlv) Stage-B eval | **TODO** | ckpts on H100, not yet transferred to box |

---

## 1. The 4×RTX 5090 eval box

| | |
|---|---|
| Access | `ssh -p 18970 root@99.148.65.10` (passwordless from the H100) |
| HW | 4× **RTX 5090** (Blackwell **sm_120**), driver **580.95.05**, 32 607 MiB each |
| Host | rented VM, hostname `ubuntu`, Ubuntu 22.04 / kernel 6.8, `/dev/vda1`, internet OK |
| Provenance | pre-existing for `robomme`; we added the starVLA policy-server half |
| Handoff on box | `/root/EVAL_HANDOFF.md` |

**Repos on box**
- `/root/starVLA` — rsync of the H100 `opd` working tree (no `.git`).
- `/root/ar-research` — the **ar-research_exp** eval repo content (rsync snapshot, no `.git`).
  This is **NOT** the kempner copy (`/home/kaiwen/Desktop/research/ar-research-kempner`) — confirmed: use `/root/ar-research`.

**Two conda envs** (`source /root/miniconda3/etc/profile.d/conda.sh`)
- **`starVLA`** = the **policy server**. Built fresh for Blackwell: `torch 2.11.0+cu128`,
  `torchvision 0.26`, `transformers 4.57`, **`h5py`**. h5py was **MISSING** and blocked
  `QwenOFT_VQA` registration (the framework imports it at load) — adding it was required
  or the model won't register. Attention = **sdpa** (no flash-attn needed on Blackwell).
- **`RoboTwin`** = the **simulator** (pre-existing): sapien 3, mplib, curobo. We
  pip-added **`websockets` + `msgpack`** for the websocket client.

**Key data/model paths on box**
- Base VLM: `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` — **mirrors the H100 path**
  so each ckpt's baked `config.yaml` `base_vlm` resolves with **no edit**.
- 0526 data: `/mnt/localssd/kaiwenh/pref/data/0526/{height,orient}` (taskA leaves + taskB).
- Stage-A token ckpts: `/root/starVLA/results/Checkpoints/pref_oftvqa_token_{height,orient}_10k/` (`steps_10000`).
- Stage-B ckpts: `/root/starVLA/results/Checkpoints/pref_stageb_main_{height,orient}/` (`steps_1500` only on box; full series on H100, §8).

---

## 2. Eval pipeline architecture + the two critical fixes

### 2.1 Client–server
- **Server** = `deployment/model_server/server_policy.py` in the **`starVLA`** env —
  a WebSocket policy server (`msgpack_numpy`), idle-timeout disabled (`--idle_timeout -1`).
- **Client** = `ar-research/script/eval_policy.py` (RoboTwin) + our **new joint+preference
  bridge** at **`/root/ar-research/policy/starvla_joint/`**. We wrote this from scratch
  because the 0526 OFT_VQA ckpts are **14D joint (abs_qpos)** while the older
  `starvla_bridge` was **20D EE** — a hard mismatch.
- The bridge sends obs `{image:[3 cams], lang:"<clean template> Preference: <label>"}`
  (no state — the QwenOFT action stream ignores it). Prompt construction mirrors training
  exactly (`examples/preference/dataset/prompt.py`, suffix `" Preference: <label>"`;
  taskB templates are embedded in the bridge because the hand-synced `prompt_table.py`
  copy lacks them). Labels e.g. height `{low:"low drop", high:"high drop"}`, orient
  `{0:"horizontal grasp", 90:"vertical grasp"}`.
- Server returns `(1, 50, 14)` **normalized** joint qpos; the bridge **denormalizes**
  with `q01/q99` from the run's `dataset_statistics.json`
  (`phys = 0.5*(clip(n,-1,1)+1)*(q99-q01)+q01`), then `TASK_ENV.take_action(action)`
  (joint; no `'ee'` / no 6D→quat).

Bridge files (`/root/ar-research/policy/starvla_joint/`): `client_joint.py`,
`deploy_policy.py`, `pref_metric.py`, `analyze_pref.py`, `prompt_table.py`,
`msgpack_numpy.py`, `rotation.py`, and the runner scripts (§7).

### 2.2 Fix (a) — NO reorder (identity)
`pref_hdf5` joint training reads `h5["joint_action/vector"]` **directly**
(`examples/preference/dataset/pref_hdf5_dataset.py:278`), layout
`[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]` = the RoboTwin **native**
joint order that `take_action()` also expects. So the model output already matches
take_action order → **no reorder needed (identity)**:
```python
_REORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]   # identity
```
The `model2robotwin_interface` reorder `[0,1,2,3,4,5,12,6,7,8,9,10,11,13]` is for the
**RoboTwin2-OFT grouped-arms** layout, which this prefVLA OFT does **not** use; applying
it **scrambles** the joints → **0/3** on a trained task until fixed.
> Note: `client_joint.py`'s module docstring still contains a stale line describing the
> `[0..5,12,6..11,13]` reorder; the **active** `_REORDER` constant and the inline comment
> are correct (identity). Behavior is identity — the stale docstring is cosmetic.

### 2.3 Fix (b) — chunk_step = 50 (execute the full open-loop chunk)
OFT predicts a **coherent open-loop 50-step abs-qpos trajectory** (`chunk_size = 50` =
future_action_window 49 + 1). Re-querying **mid-chunk** breaks it: `chunk_step=25` →
**0/3** on a trained task (`place_mouse_stand_high`); **`chunk_step=50`** (execute the
full chunk before re-querying) → **2/2 = 100%**. Default is now **50** in all runner
scripts (override via `CHUNK_STEP=N`).
> Caveat: `deploy_policy.get_model` and `client_joint.__init__` still default
> `chunk_step=25` as a bare kwarg, but **every** runner script writes
> `chunk_step: 50` into the deploy yml, so launched evals always use 50.

### 2.4 Teacher-forcing validation
The whole action pipeline (denorm / identity-reorder / prompt / image encode) was
verified by **teacher-forcing**: feeding recorded demo observations, the model
reproduces the recorded demo action trajectory with per-dim **MAE ~0.05 rad**, and
deployment `action[0]` = home (correct). taskA (seen) success then confirms end-to-end
correctness.

### 2.5 Env scaffolding fixes (prefVLA tasks were missing these)
- `description/task_instruction/<task>.json` — created for taskB (4) + height taskA (16);
  the bridge ignores the env instruction, so minimal text is fine.
- `objects_description/<obj>/base{N}.json` — auto-filled every missing one from each
  object's `model_data{N}.json` (text-only; the expert demo uses mesh/points_info).

---

## 3. VQA gate-eval (does the preference *read* generalize?)

Script: `r-preference/eval/gate_oftvqa.py` (`predict_preference`, token-aware). It builds
the model **from the YAML** (so `state_mode: token` initializes `phi`), sets
`pretrained_checkpoint=None`, then `load_state_dict(..., strict=False)` from the ckpt.

**How to run** (in the `starVLA` env, on the box):
```bash
source /root/miniconda3/etc/profile.d/conda.sh; conda activate starVLA
cd /root/starVLA; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
python r-preference/eval/gate_oftvqa.py --cat height \
  --yaml examples/preference/train_files/starvla_pref_stage_a_oftvqa_token_height.yaml \
  --ckpt results/Checkpoints/pref_oftvqa_token_height_10k/checkpoints/steps_10000_pytorch_model.pt \
  --data_root /mnt/localssd/kaiwenh/pref/data/0526/height --out /root/gate_token_height_10k.json
# argparse: --cat --yaml --ckpt --data_root [--n_taskA 5] [--n_taskB 25] [--out]
```

**Results** (steps_10000; VQA preference accuracy, taskA-val / taskB; verified from the
JSONs on box — taskA n=80, taskB n=50):

| cat | token-VQA | OFT-baseline (no VQA loss) |
|---|---|---|
| height | **1.00 / 1.00** | 0.50 / 0.50 — degenerate, always "low" (pred_dist high=0) |
| orient | **1.00 / 0.98** (taskB gt_0 0.96, gt_90 1.00) | 0.362 / 0.42 — degenerate, biased "side"/0 (gt_90≈0) |

The **OFT-baseline** = the no-VQA-loss ckpt (`pref_oft_baseline_*`) loaded into the
**token** `QwenOFT_VQA` framework via `strict=False` → the `vqa_state_proj` (`phi`) is
**untrained/random** (load reports **"6 missing keys"**). It collapses to chance and
degenerate, proving the VQA reading ability comes **entirely from the VQA cotrain**, not
from a pixel prior or the base VLM. (Baseline control: `run_baseline_vqa.sh`.)
JSONs: `/root/gate_token_{height,orient}_10k.json`, `/root/gate_baseline_{height,orient}_10k.json`.

---

## 4. Closed-loop rollout success (Stage-A token ckpts)

Open-loop chunk_step=50, 10 episodes each. taskA = seen objects; taskB = held-out
objects/tasks. Verified from `/root/eval_taskA_t10_s0_202128/*.log` and
`/root/eval_taskB_t10_s0_200721/*.log` (`grep "Success rate"`).

| cat / pref | taskA (seen) | taskB (unseen) |
|---|---|---|
| height **high** | `place_mouse_stand_high` **10/10 = 100%** | `place_playingcards1_box_high` **10/10 = 100%** |
| height **low**  | `place_mouse_stand_low` **5/10 = 50%** | `place_playingcards1_box_low` **9/10 = 90%** |
| orient **0 (side)** | `place_bottle_box_0` **6/10 = 60%** | `move_can5_away_0` **0/10 = 0%** |
| orient **90 (top)** | `place_bottle_box_90` **1/10 = 10%** | `move_can5_away_90` **0/10 = 0%** |

**Read:** height generalizes strongly (taskB ≥ 90%). Orient is weaker, especially
vertical/90, and orient **taskB `move_can5_away` = 0%** is a **genuine hard transfer**
(a "move-away" motion the orient model never trained on), **not** a bridge bug — orient
**seen** `place_bottle_box` succeeds and orient **VQA reads** the preference at 0.98.
Videos under `r-preference/debug/v3/<task>/starvla_joint/` (and on box under
`eval_result/<task>/starvla_joint/...`).

> Note: `check_success` here uses the **expert recipe param** (`_used_lift_z_high` set in
> `play_once`), so it is **preference-agnostic** — it measures task completion, not
> whether the policy followed the preference. The following question needs the §5 metric.

---

## 5. The preference-following metric + the Stage-A finding

### 5.1 Why a new metric
`check_success` is **decoupled** from whether the policy obeyed the preference (it keys
off the expert recipe, §4 note). `pref_metric.py` instead measures the **policy's actual
preference-relevant quantity** from sim state during the rollout. The bridge tracks per
step (`deploy_policy.eval` → `pref_metric.track_step`, recording object pose, gripper
open/close, and the EE endpose from `obs`), and `pref_metric.measure` computes the axis
at episode end (`reset()`); `analyze_pref.py` aggregates per `(axis, preference)`.

**Per category** (axis = what is measured):

| cat | axis | definition | expected |
|---|---|---|---|
| height | `drop_height` | obj.z at release − obj.z at grasp. **Sampled at `release_step − 1`** (the step *before* the gripper opens), because `take_action` runs many physics substeps and the object has already fallen by the open step. | high > low |
| contact | `grasp_height` | obj.z at grasp − obj.z resting | 75 > 25 |
| orient | **`ee_x_tilt`** (the GRIPPER approach) | tilt of the EE local-x axis (gripper +x approach axis) from world vertical, from the obs endpose. **NOT the object tilt** — the can lies flat so `obj_tilt_deg`≈90° for both prompts and is useless. | side(0)≈90°, top(90)≈0° |
| place | `place_offset_m` | \|obj.xy(final) − target.xy\| | corner > center |
| hvlv | `obstacle_clear_m` | min over traj of \|obj.xy − obstacle.xy\| | hv > lv |

> **Aggregation caveat (orient).** `analyze_pref.py` groups orient under `grasp_tilt_deg`
> and uses the JSON `value` field, which `pref_metric` sets to **`ee_z_tilt`** (`ax[2]`),
> NOT `ee_x_tilt`. So the default aggregate prints a muddied orient separation
> (prompt 0 → 47.5°, prompt 90 → 72.2°, ~20% follow). The **correct, separating signal
> is `ee_x_tilt`** (per-episode field in each JSON, §6) — read it directly. height/place/
> hvlv aggregate cleanly via the default.

### 5.2 Stage-A: reads but does NOT follow (height, validated)
Aggregating Stage-A height (`/root/eval_pref/height`, prompt-paired):
**prompt high drop = 0.074, prompt low drop = 0.074** — **no separation**, peak_lift≈0.19
for both (it lifts correctly, then descends and gently places regardless of the prompt).

**Expert sanity-check** (recorded 0526 hdf5 endpose) cleanly **separates**:
high drop = **0.175** vs low drop = **0.100** (peak_lift ~0.18 for *both* — by design only
the release height differs). So the **metric is valid** (it recovers the expert
separation when behavior actually differs); the Stage-A null result is **real, not an
artifact**.

**Diagnosis.** The token-VQA **reads** the preference (1.00, §3) but the OFT **action
stream does not act on it**. Conditioning is only via the language prompt feeding the
`L1RegressionActionHead` MLP; the preference is a small / late / low-dimensional
modulation that gets **diluted by the uniform L1 over the 50×14 chunk**. → The labeler
works; the **conditioner** is the bottleneck.

---

## 6. Stage-B controllability — the fix (height + orient)

**Stage-B** = pseudo-label-conditioned SFT (`pref_stageb_main_{height,orient}`, QwenOFT
joint, L1, 1500 steps, **warm-started from the Stage-A token ckpt**; checkpoints
`steps_250..1500`). The eval is a **paired same-seed two-prompt controllability** run:
**one env, fixed seeds, the prompt is the only thing that changes**, and `pref_metric`
tags each rollout by the **PROMPTED `pref_key`** (not the env recipe) — so it directly
asks "does the prompt control the trajectory?". Runners: `run_pref_control.sh` /
`run_stageb_control_all.sh` (§7). Records: `/root/eval_pref_ctrl/pref_stageb_main_{height,orient}/<env>__prompt_<key>/*.json`.

### 6.1 Height — FOLLOWS
Env `place_playingcards1_box_high`, prompts {high, low} (4 paired episodes each;
verified per-episode):

| prompt | drop_height (mean) | per-episode | peak_lift | task success |
|---|---|---|---|---|
| **high** | **0.157** | 0.150, 0.132, 0.177, 0.169 | ~0.18–0.20 | 5/5 (separate 5-ep run) |
| **low**  | **0.071** | 0.052, 0.057, 0.097, 0.077 | ~0.18–0.19 | 4/5 |

Separation **+0.086**; **every** paired episode has high > low; near the expert
0.175/0.100; vs Stage-A 0.074 = 0.074 (no separation). → **pseudo-label-conditioned SFT
turns the stated preference into a controllable release height.**

### 6.2 Orient — FOLLOWS (after two fixes)
Env `move_can5_away_0`, prompts {0 = horizontal, 90 = vertical} (5 paired episodes each).
**Two issues found and fixed (both were the user's hypotheses, both confirmed):**

1. **Measure the GRIPPER, not the object.** The can lies flat → `obj_tilt_deg`≈90° for
   both prompts. The right signal is **`ee_x_tilt`** (EE local-x = gripper +x approach
   axis): a side grasp → ee_x≈82–87°, a top-down grasp (wrist inverted) → ee_x≈0–2°.
2. **Reachability.** The vertical/top-down grasp is **IK-marginal** (the wrist inverts
   and "runs out of IK room"; the env spawn sits near the centerline where IK
   degenerates). The can spawns lying flat; we **narrowed ylim** from `[-0.2, 0.05]` to
   **`[-0.15, 0.05]`** in `move_object_lift.py` `load_actors` (verified: file now reads
   `xlim=[-0.1,-0.04]`, `ylim=[-0.15,0.05]`) to make the vertical grasp feasible.

**Result (`ee_x_tilt` per episode, the separating signal):**

| prompt | ee_x_tilt per episode | summary | task success |
|---|---|---|---|
| **horizontal (0)** | 87.0, 81.5, 73.7, 82.4, 86.4 | **5/5 side** (≈82° mean) | 5/5 (and 5/6 in a 6-ep run) |
| **vertical (90)** | 84.4, **1.36, 0.37, 2.27**, 83.4 | **3/5 top-down** (≈0–2°); 2/5 stay horizontal (residual IK-marginal spawns) | 5/5 (and 6/6 in a 6-ep run) |

→ The prompt is a real control knob: vertical drives the gripper to a top-down,
wrist-inverted approach in 3/5 spawns; the 2/5 misses are residual marginal spawns, not a
conditioning failure. **Visually confirmed** in the rollout videos
(`r-preference/debug/v3/stageb_orient_ctrl_narrowed/`).

> The default `analyze_pref.py` orient aggregate (47.5° vs 72.2°, ~20%) understates this
> because it sums `ee_z_tilt`; the `ee_x_tilt` per-episode values above are the truth (§5.1 caveat).

### 6.3 The pseudo-label / SPT framing
Pseudo-labeling essentially **sets the preference LABEL in the prompt**. But Stage-A
showed the action **ignores** the prompt label → labeling is **not** the bottleneck;
**conditioning** is. Stage-B SFT (training the action on the pseudo-labeled,
preference-conditioned prompts) makes the action **follow**. Clean separation of roles:
**VQA = the labeler**; the **(Stage-B-trained) prompt-conditioning = the conditioner**.
This is the empirical backing for the paper's **SPT** claim: *stated preference →
controllable behavior*, demonstrated for height (continuous release height) and orient
(discrete approach orientation).

---

## 7. Reproduce-it commands (exact arg signatures, from the box scripts)

All scripts live in `/root/ar-research/policy/starvla_joint/`. Env note: `conda activate`
exports `AR`/`CC`/… which clobber plain vars — scripts use `AR_ROOT`/`STAR_ROOT`.

```bash
# --- 1. start a policy server (starVLA env):  <cat> <port> <gpu> [run] [step]
bash launch_server.sh height 10093 0
bash launch_server.sh height 10094 0 pref_stageb_main_height 1500   # Stage-B server

# --- 2. one Stage-A rollout eval (RoboTwin env):  <cat> <task_name> <pref_key> <port> <gpu> [test_num] [seed]
#        server must already serve on PORT; chunk_step defaults to 50 (override CHUNK_STEP=N)
bash run_joint_eval.sh height place_playingcards1_box_high high 10093 1 10 0

# --- 3. PAIRED controllability (RoboTwin env): one env, every prompt in turn
#        <cat> <env_task> <task_group> <ckpt_run> <port> <gpu> <test_num> <seed> <prefkey>...
bash run_pref_control.sh height place_playingcards1_box_high place_playingcards1_box \
     pref_stageb_main_height 10094 1 5 0 high low
bash run_pref_control.sh orient move_can5_away_0 move_can5_away \
     pref_stageb_main_orient 10095 1 5 0 0 90
#   writes /root/eval_pref_ctrl/<ckpt_run>/<env>__prompt_<key>/*.json + ends with analyze_pref summary

# --- 4. launch BOTH Stage-B servers + paired ctrl for height+orient in parallel:  [step] [test_num] [seed]
bash run_stageb_control_all.sh 1500 5 0

# --- 5. aggregate the preference metric (needs numpy -> run in a conda env)
source /root/miniconda3/etc/profile.d/conda.sh; conda activate RoboTwin
PYTHONPATH=/root/ar-research/policy/starvla_joint \
  python /root/ar-research/policy/starvla_joint/analyze_pref.py /root/eval_pref_ctrl/pref_stageb_main_height
#   NB: for orient, read ee_x_tilt per-episode from the JSONs (the aggregate uses ee_z_tilt)

# --- 6. VQA gate-eval (starVLA env): see §3 for the full command
```

---

## 8. Index of artifacts

**Code (the bridge, on box)** — `/root/ar-research/policy/starvla_joint/`:
`client_joint.py`, `deploy_policy.py`, `pref_metric.py`, `analyze_pref.py`,
`launch_server.sh`, `run_joint_eval.sh`, `run_joint_eval_parallel.sh`,
`run_pref_control.sh`, `run_stageb_control_all.sh`, `prompt_table.py`, `msgpack_numpy.py`.
Env edit: `/root/ar-research/envs/move_object_lift.py` (`load_actors` ylim narrowed to `[-0.15,0.05]`).

**Checkpoints**

| ckpt | on box | on H100 (`/home/kaiwenh/starVLA/results/Checkpoints/`) |
|---|---|---|
| Stage-A token | `pref_oftvqa_token_{height,orient}_10k` (steps_10000) | (overnight runs; token height/orient/contact here, place/hvlv on H200) |
| Stage-B main | `pref_stageb_main_{height,orient}` (**steps_1500 only**) | `pref_stageb_main_{height,orient}` (**steps_250..1500**), `pref_stageb_main_contact_geom`, `pref_stageb_main_place`, `pref_stageb_b0_place` |

**Data / VLM** — box: `/mnt/localssd/kaiwenh/pref/data/0526/{height,orient}` (full 5 cats on H100),
base VLM `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` (both).

**Results / logs (on box)**
- VQA: `/root/gate_token_{height,orient}_10k.json`, `/root/gate_baseline_{height,orient}_10k.json`.
- Stage-A pref metric: `/root/eval_pref/height/*.json`.
- Stage-B controllability: `/root/eval_pref_ctrl/pref_stageb_main_{height,orient}/<env>__prompt_<key>/*.json`.
- Rollout logs: `/root/eval_taskA_t10_s0_202128/`, `/root/eval_taskB_t10_s0_200721/`,
  `/root/ctrl_*` (per-prompt + `*_ALL.log`).

**Videos (H100)** — `/home/kaiwenh/starVLA/r-preference/debug/v3/`:
per-task Stage-A dirs (`place_mouse_stand_{high,low}/`, `place_playingcards1_box_{high,low}/`,
`place_bottle_box_{0,90}/`, `move_can5_away_{0,90}/`, each under `starvla_joint/`), and
Stage-B controllability `stageb_orient_ctrl/` + **`stageb_orient_ctrl_narrowed/`**
(the post-narrowing run; `ctrl_pref_stageb_main_orient_{0,90}/.../episodeN.mp4`).

---

## 9. Open items / next steps

1. **More episodes.** Current paired runs are 4–5 episodes/prompt — enough to show the
   effect, thin for a number. Bump to ~10–20 for the paper.
2. **Other 3 categories.** contact / place / hvlv Stage-B ckpts exist on **H100**
   (`pref_stageb_main_contact_geom`, `pref_stageb_main_place`, `pref_stageb_b0_place`)
   but are **not on the box**, and their 0526 data (contact/place/hvlv) is on H100 only.
   Transfer ckpts + data + dataset_statistics to the box and run the §7 paired
   controllability. Expect place/hvlv to remain **relational-capped** (the 0526 hdf5
   logs no target/obstacle pose — see the token-VQA-fix doc §1.4).
3. **Orient 2/5 miss.** Two vertical-prompt spawns stay horizontal (residual IK-marginal).
   Options: narrow the spawn further, increase episodes, or report 3/5 with the IK caveat.
4. **`analyze_pref.py` orient fix.** Make the orient aggregate key off `ee_x_tilt`
   (not `ee_z_tilt`) so the default summary surfaces the 82°/~0° separation directly.
5. **Before/after figure.** A single Stage-A-vs-Stage-B controllability plot (height drop
   and orient ee_x_tilt, two prompts each) — the headline figure for the SPT claim.
6. **Token-vs-baseline action ablation.** OFT-baseline (no-VQA) action ckpts were not on
   H200 for the rollout side; only token ckpts are on the box. Transfer to test whether
   the VQA cotrain helps the *action* (not just the read).

---

### Verified-against-files note
All numbers in this doc were re-checked on **2026-05-29** against the actual files:
VQA JSONs (`/root/gate_*_10k.json`), rollout logs (`/root/eval_task{A,B}_*`), and the
preference-metric JSONs re-aggregated with `analyze_pref.py` plus per-episode reads
(`/root/eval_pref`, `/root/eval_pref_ctrl/...`). The one place where the synthesized
record and the files diverge in *presentation* (not substance): the orient headline
82°/~0° is **`ee_x_tilt`** read per-episode, whereas the default `analyze_pref.py`
aggregate reports `ee_z_tilt` (47.5°/72.2°) — see §5.1 / §6.2.
