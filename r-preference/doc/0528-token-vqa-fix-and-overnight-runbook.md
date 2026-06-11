# 2026-05-28 — Token-VQA fix + overnight ablation runbook (READ THIS FIRST)

> **Self-contained handoff.** Written 2026-05-28 ~14:05 UTC on the **H100** box
> (`/home/kaiwenh/starVLA`, user `kaiwenh`). A fresh chat with zero prior context
> can fully pick up from here: what was discovered tonight, what is running now, how
> to monitor/kill/relaunch it, and exactly what to run in the morning.
>
> **Project:** Pref-VLA Stage A — a preference-conditioned VLA. The action stream
> predicts robot actions; an auxiliary **VQA cotrain** teaches the shared VLM to
> *read the preference* (e.g. "are we dropping high or low?") so it can later act as
> a Stage-B pseudo-labeler. Tonight's work fixes a VQA generalization collapse.
>
> **Companion docs (background, do not duplicate):**
> - [`0528-how-to-fix-last-minute.md`](0528-how-to-fix-last-minute.md) — the negative
>   result this doc **resolves**: VQA-as-text collapsed to 0.50 on held-out. (Morning
>   TODO §5.3 adds a "RESOLVED" pointer back here.)
> - [`0528-action-arch-and-oft-vqa-integration.md`](0528-action-arch-and-oft-vqa-integration.md)
>   — the 3 action heads, 0526 convergence (OFT warm-start works, QwenPI-from-scratch
>   does not), OFT×VQA integration design.
> - [`0528-instruction-mount.md`](0528-instruction-mount.md) — the `0526` data tree.

---

## 0. TL;DR + live status snapshot

**The fix (one sentence):** the VQA collapsed on held-out because feeding the robot
end-effector (EE) pose to the VLM as numeric **text** left a language/recency
backdoor (the LM-logit readout memorized train→1.0 and emitted the last-mentioned
answer word on val). Replacing the text pose with a learned **soft token** (a tiny
MLP `phi` whose output is injected at an existing special-token marker) closes the
backdoor — a frozen-VLM Phase-1 probe now generalizes to taskA-val **and** taskB on
the 3 proprioceptive prefs, and a zeroed-token control collapses back to 0.50
(proving the model genuinely reads the token).

**What's running:** 10 overnight runs = 5 categories × {token-VQA, OFT-baseline},
split across H100 (5 jobs) and H200 (5 jobs), 10k steps each, OFT warm-start.

### Live status snapshot — captured 2026-05-28 ~14:05 UTC

| Machine | Queue PID | Jobs (in order) | Current | Done / Run / Fail |
|---|---|---|---|---|
| **H100** (`kaiwenh`) | `946966` | token:height, token:orient, token:contact, base:place, base:hvlv | **job 1/5** `pref_oftvqa_token_height_10k` @ **~step 925/10000**, **~1.4 it/s**, `L_action≈0.024`, `L_vqa→0.0`, `vqa_acc=1.0` | **0 / 1 / 0** |
| **H200** (`kevin@34.34.93.23`) | queue running | base:height, base:orient, base:contact, token:place, token:hvlv | **job 1/5** `pref_oft_baseline_height_10k` @ **~step 1306/10000**, **~2.0 it/s** | **0 / 1 / 0** |

- Both machines launched ~13:48 UTC. **0 of 10 done, 2 running (1 per machine), 0 failed.**
- H100 token run confirmed healthy: marker `<|fim_pad|>`(151662), `k=3`, `phi=9→512→2560`,
  OFT ckpt loaded (2 shards, no crash), `L_vqa` already at numerical 0 with `vqa_acc=1.0`
  on train — exactly the expected "train memorizes" signal; **real generalization is the
  morning gate-eval, not this number** (§5.1).
- ETA ≈ **8–9 h per machine** (token ≈ 2 h/run, baseline ≈ 1.3 h/run). Expect all 10
  complete roughly **2026-05-28 ~22:00 UTC**.

---

## 1. The token-VQA finding

### 1.1 Problem
Stage-A preference VQA (binary "what is the preference?" over a short head-camera
clip, single-token answer, CE) **memorized train** (`L_vqa→0`, `vqa_acc=1.0`) but
**collapsed on held-out**: taskA-val + taskB ≈ **0.50**, a degenerate constant
predictor always emitting the last-mentioned answer word (orient→`top`,
place→`corner`, height→`low`). This held **even after** feeding the EE pose as text.

### 1.2 Root cause
The readout compares `logit(answerA)` vs `logit(answerB)` at the answer position. With
the pose as **text**, the model has two ways to be right on train: (a) actually read
the pose, or (b) exploit a language/recency prior over the two answer words. Path (b)
is the easy minimum, so it memorizes train and emits the word-prior on val → 0.50.
The backdoor is in the **readout/representation**, not in missing signal (the EE pose
*is* separable — see §1.4 MLP upper bound).

### 1.3 Fix — EE pose as a learned SOFT TOKEN
Implemented in **`starVLA/model/framework/vqa_cotrain_mixin.py`** (enabled by
`framework.vqa.state_mode: token` in the YAML):

- **`phi` = `self.vqa_state_proj`**: `nn.Sequential(Linear(9,512), GELU, Linear(512,hidden))`,
  `hidden = text_config.hidden_size = 2560`. Maps a 9D active-arm EE pose
  (xyz + 6D rotation) → one token embedding. Trained at **LR 1e-3** (vs 1e-5/1e-4 for
  the rest; see YAML `learning_rate.vqa_state_proj`).
- **Marker = an EXISTING special token `<|fim_pad|>` (id 151662).** Reusing a token
  already in the vocab means **NO `resize_token_embeddings`**, so the pretrained OFT
  checkpoint loads cleanly (no shape mismatch on the embedding/LM-head). `k = n_frames = 3`
  marker tokens are prepended to the question; `phi(state)` overwrites their embeddings.
- **Injection = a forward-hook on the VLM input-embedding layer**
  (`_vqa_inject_hook`, registered on `model.get_input_embeddings()`): for the in-flight
  batch it overwrites positions where `input_ids == marker_id` with the precomputed
  `phi(state)`. It is a **no-op when no marker is present** (e.g. the action-stream
  forward), so it is safe on every VLM call. `_set_vqa_state_embeds(st_np)` standardizes
  the pose with buffered mean/std then runs `phi` and stashes `self._vqa_cur_state`
  `(B*k, hidden)`; the hook reads that.
- **`_compute_vqa_state_stats`**: deterministic mean/std of the active-arm 9D EE pose at
  the category's clip frames, from a 64-episode sample (endpose-only read, no image
  decode), registered as buffers `_vqa_state_mean` / `_vqa_state_std`.
- `predict_preference` and the held-out `validate_vqa` are **token-aware**: when
  `state_mode == token` they prepend the marker and call `_set_vqa_state_embeds` before
  the forward, so eval injects the token exactly like training.

The same mixin still supports `state_mode: text` (legacy `format_state_text`) and the
plain no-state path; **token** is the new default for these runs. The mixin is action-
head-agnostic — `QwenOFT_VQA` and `QwenPI_VQA` both mix it in identically.

### 1.4 Phase-1 probe results (the evidence)
Standalone diagnostic — **`r-preference/phase1/state_token_probe.py`** — frozen
Qwen3-VL-4B, **only `phi` trainable**, `lm_head` readout, marker `<|pref_state|>`
(here it *does* resize, because it's a throwaway probe), k=3, 1000 steps, LR 1e-3,
jitter ±5. JSON in `r-preference/phase1/probe_<cat>_lmhead.json`. Final-step (1000):

| Cat | taskA-val acc | taskB acc | zeroed-state acc | MLP-on-state upper bound (taskA / taskB) |
|---|---|---|---|---|
| **height** (proprio) | **1.00** | **1.00** | 0.50 | 1.00 / 1.00 |
| **orient** (proprio) | **1.00** | **1.00** | 0.50 | 1.00 / 1.00 |
| **contact** (proprio)¹ | ~0.92 | **1.00**² | ~0.53 | 0.938 / 1.00 |
| **place** (relational) | 1.00 | **0.75** (overfits, taskB drops) | 0.50 | 1.00 / **0.55** |
| **hvlv** (relational) | ~0.94 | **0.55** | 0.50 | 0.95 / **0.70** |

¹ `probe_contact_lmhead.json` was **not** written (the run reached step 600 in
`/mnt/localssd/kaiwenh/logs/phase1_contact_lmhead.log` without a FINAL line); numbers
above are from that log's VAL lines (step 600: taskA 0.919, taskB 1.00, zeroedA 0.544).
² contact taskB bounces (step 200=0.85, step 400=0.65, step 600=1.00) on n=20 — small
sample, but clearly readable from state.

**Interpretation:**
- **3 PROPRIOCEPTIVE prefs** (height=drop height, orient=wrist orientation,
  contact=grasp height) are **fully readable** from the EE state via the token.
- **2 RELATIONAL prefs** (place=placement location, hvlv=obstacle-avoidance margin)
  are **capped** — they need the *target/obstacle* pose, which the 0526 hdf5 does
  **not** log (only `endpose` + gripper). The MLP-on-state upper bound confirms this
  ceiling (place taskB 0.55, hvlv taskB 0.70), so it is a **data limitation, not a
  model/readout bug**.
- The **zeroed-state control → 0.50** for every cat proves the model reads the token
  (not pixels/prior). Earlier "head-cam ≈ chance" results were a zero-shot-prompting
  artifact; a fresh 2-way `probe` head and even pixels-only also generalize once the
  backdoor is welded — i.e. the original collapse was the **readout**, not signal.

### 1.5 Evidence figures (`r-preference/debug/v2/`)
| File | What it shows |
|---|---|
| `phase1_summary.png` | per-cat taskA/taskB/zeroed across probe steps (the headline) |
| `state_separability.png` | EE-pose separability per cat (signal is in proprioception) |
| `relative_separability.png` | relative-pose separability (why relational cats cap) |
| `vqa_result_summary.png` | summary of the VQA results |

Generators: `make_phase1_summary.py`, `make_state_separability.py`,
`make_relative_separability.py`, `make_viz_v2.py` (same dir).

### 1.6 Smoke before the overnight — caveat, read carefully
The pre-overnight smoke logs (`/mnt/localssd/kaiwenh/logs/oftvqa_state_smoke.log`,
`oftvqa_smoke2.log`, `oftvqa_smoke.log`) ran to **"Training complete, final model
saved"** with no crash and the OFT ckpt loaded — BUT their per-rank logged
`L_action`/`L_vqa` show **`nan`**, and `oftvqa_state_smoke.log` was a **state-as-text**
smoke (it does *not* contain `state_mode=TOKEN`). **Do not cite those as the token
smoke.** The real, trustworthy validation of the **token** path is the **live H100
height run** itself: it loaded the OFT ckpt over 2 shards, ran `_compute_vqa_state_stats`
(`state_mean` printed), confirmed `state_mode=TOKEN marker='<|fim_pad|>'(151662) k=3
proj=9->512->2560`, and is training healthily (`L_action≈0.024`, `L_vqa→0.0`,
`vqa_acc=1.0`). If a run shows persistent `nan` in `L_action`, that is a real failure —
investigate (the queue continues to the next job on failure).

---

## 2. The overnight ablation

**Goal:** confirm the token-VQA fix generalizes inside the **full cotrain** (not just
the frozen probe), and measure whether the VQA cotrain helps / is neutral / hurts the
**action** model — across all 5 categories.

**10 runs = 5 cats {contact, height, hvlv, orient, place} × {token-VQA, OFT-baseline}.**

### 2.1 Job split (clean/proprioceptive cats first, relational last)

| Machine | Jobs (run sequentially, in this order) |
|---|---|
| **H100** | `token:height` → `token:orient` → `token:contact` → `base:place` → `base:hvlv` |
| **H200** | `base:height` → `base:orient` → `base:contact` → `token:place` → `token:hvlv` |

So each machine runs 3 token + 2 baseline (H100) / 2 token + 3 baseline (H200), and
every (cat × variant) cell is covered exactly once across the two machines.

### 2.2 Backbone & why
**`QwenOFT` warm-started from the RoboTwin2 OFT checkpoint `steps_140000`.** This is
the **only** backbone that converges on 0526 (QwenPI-from-scratch went degenerate at
the unconditional-mean floor `L≈1.45`). Action space = **joint** (14D qpos,
`action_type: abs_qpos`) to match the OFT pretrained ckpt.

### 2.3 Hyperparameters (verified from the YAMLs)

| Field | token (`..._oftvqa_token_<cat>.yaml`) | baseline (`..._oft_baseline_<cat>.yaml`) |
|---|---|---|
| `framework.name` | **`QwenOFT_VQA`** | **`QwenOFT`** |
| `framework.vqa` | `lambda_vqa: 0.5`, `n_vqa_per_batch: 2`, **`state_mode: token`**, `category: <cat>` | — (no VQA) |
| `datasets.vla_data.dataset_py` | `pref_hdf5_vqa` | `pref_hdf5` |
| `action_space` / `action_type` | `joint` / `abs_qpos` | `joint` / `abs_qpos` |
| `max_train_steps` | **10000** | **10000** |
| `num_warmup_steps` | 1000 | 1000 |
| `save_interval` | **5000** → ckpts at `steps_5000` + `steps_10000` | **5000** (same) |
| `learning_rate` | base `1e-5`, qwen_vl_interface `1e-5`, action_model `1e-4`, **`vqa_state_proj 1e-3`** | base `1e-5`, qwen_vl_interface `1e-5`, action_model `1e-4` |
| `lr_scheduler` | `cosine_with_min_lr`, `min_lr 5e-7` | same |
| `per_device_batch_size` | 8 (×8 GPU = 64) | 8 |
| `cameras` | head + left + right (action stream); VQA clip is head-only 3-frame per `VQA_CATEGORIES` | head + left + right |
| `pretrained_checkpoint` | OFT `steps_140000` (overridden per-host on CLI) | same |
| `seed` | 42 | 42 |
| `run_id` (queue override) | `pref_oftvqa_token_<cat>_10k` | `pref_oft_baseline_<cat>_10k` |

Per-category VQA recipe (from `examples/preference/dataset/vqa_sample.py`,
`VQA_CATEGORIES`, redesigned 2026-05-28) — all **head-camera only, 3-frame,
`state_in_vqa=True`**:

| Cat | question (suffix `Answer:`) | answers (tokens) | clip strategy (fracs) |
|---|---|---|---|
| contact | "…grasping it from high or from low?" | `25→low(10303)` / `75→high(11892)` | `contact3` (0.40/0.60/0.80) |
| height | "…dropping it from high or from low?" | `high→high(11892)` / `low→low(10303)` | `height3` (0.90/0.95/1.00) |
| hvlv | "…keep far from it or near it?" | `hv→far(23559)` / `lv→near(51659)` | `hvlv3` (0.60/0.70/0.80) |
| orient | "…from the side (horizontal) or from the top (vertical)?" | `0→side(2929)` / `90→top(3481)` | `orient3` (0.40/0.60/0.80) |
| place | "…in the middle or on the corner?" | `center→middle(19656)` / `corner→corner(73425)` | `place3` (0.60/0.80/1.00) |

### 2.4 How the 10 YAMLs were generated
**`examples/preference/train_files/gen_overnight_yamls.py`**: loads the place token
template (`starvla_pref_stage_a_oftvqa_token_place.yaml`) and the baseline place
template (`conv_oft_warmstart_place_0526.yaml`); for each cat it sets
`data_root_dir`, `data_mix=<cat>_v1`, `stats_json_path=stats_<cat>_0526_joint.json`,
`pref_category`, and (token) `framework.vqa.category`; writes the 4 derived files per
cat (place token already exists). Baseline also forces `max_train_steps=10000`,
`num_warmup_steps=1000`, `save_interval=5000`. **Total = 10 YAMLs.**

### 2.5 Key paths

| What | H100 (`kaiwenh`) | H200 (`kevin@34.34.93.23`) |
|---|---|---|
| repo | `/home/kaiwenh/starVLA` | `/home/kevin/starVLA` |
| OFT ckpt | `/mnt/localssd/kaiwenh/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt` | `/mnt/localssd/kevin/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt` |
| 0526 data | `/mnt/localssd/kaiwenh/pref/data/0526/<cat>` | `/mnt/localssd/kaiwenh/pref/data/0526/<cat>`* |
| base VLM | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct`* |
| logs | `/mnt/localssd/kaiwenh/logs/` | `/mnt/localssd/{kaiwenh,kevin}/logs/` (runner auto-discovers) |
| results | `results/` → `/mnt/localssd/kaiwenh/starVLA_runs/results` (symlink) | `results/` → `/mnt/localssd/kevin/starVLA_runs` |
| ckpts | `results/Checkpoints/<run_id>/checkpoints/steps_{5000,10000}_pytorch_model.pt` | same |

\* On the H200 snapshot the data/VLM were discovered under `/mnt/localssd/kaiwenh/...`
(those paths exist on that host too); the runner's `disc()` picks whichever exists.
**The code was rsynced from H100 → H200 tonight** because the H200 was at an older
commit missing the OFT-VQA infra.

---

## 3. The scripts

### 3.1 `r-preference/overnight/run_queue.sh` (host-agnostic queue runner)
Runs the given jobs **sequentially**, one full 8-GPU run at a time, **continuing to
the next job if one fails**. Each arg is `<variant>:<cat>` (`token` or `base`).

**Discovery / env logic (in order):**
1. `set +u` (CRITICAL — else `conda activate` trips `SYS_SYSROOT: unbound variable`),
   then source whichever `conda.sh` exists under `/mnt/localssd/{kaiwenh,kevin}/miniconda3`;
   `conda activate starVLA`.
2. `export CUDA_HOME=$CONDA_PREFIX` (CRITICAL — else DeepSpeed raises
   `MissingCUDAException`).
3. `disc()` helper = "echo the first path arg that exists". Used to auto-discover, per
   host: **CKPT** (OFT ckpt), **VLM** (base VLM), **DATA_BASE** (`.../pref/data/0526`),
   **LOGDIR** (`.../logs`). Hard-fails with `FATAL:` if any is missing or `results`
   symlink is absent.
4. `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (+ NCCL/tokenizer/alloc env).
5. `NUM_GPUS=${NUM_GPUS:-8}`. Queue driver log = `$LOGDIR/overnight_queue_$(hostname).log`.

For each job it writes `[date] START <rid>` to the queue log, runs the **accelerate**
command below (per-run log `$LOGDIR/overnight_<rid>.log`), then logs `DONE` or
`FAILED (rc=…) — continuing`. Finishes with `==== QUEUE COMPLETE ====`.

### 3.2 The exact per-job accelerate command (reproduced from the script)
`$CKPT`, `$VLM`, `$DATA_BASE` are the discovered host paths; `$yaml` and `$rid` derive
from the `<variant>:<cat>` arg; `$cat` is the category.
```bash
accelerate launch \
    --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
    --num_processes "$NUM_GPUS" \
    starVLA/training/train_starvla.py \
    --config_yaml "examples/preference/train_files/$yaml" \
    --run_id "$rid" \
    --datasets.vla_data.data_root_dir "$DATA_BASE/$cat" \
    --trainer.pretrained_checkpoint "$CKPT" \
    --framework.qwenvl.base_vlm "$VLM" \
    > "$log" 2>&1
```
where (token) `yaml=starvla_pref_stage_a_oftvqa_token_<cat>.yaml`,
`rid=pref_oftvqa_token_<cat>_10k`; (base) `yaml=starvla_pref_stage_a_oft_baseline_<cat>.yaml`,
`rid=pref_oft_baseline_<cat>_10k`.

### 3.3 Env quirks (must hold if you relaunch by hand)
- `set +u` **before** `conda activate starVLA` (unbound-var crash otherwise).
- `export CUDA_HOME=$CONDA_PREFIX` (DeepSpeed `MissingCUDAException` otherwise).
- `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (offline VLM load).
- `base_vlm`, `pretrained_checkpoint`, `data_root_dir` are **overridden per-host on the
  CLI** via `disc()` discovery — the values baked into the YAML are H100 defaults.

---

## 4. Monitor & control (copy-paste)

### 4.1 Status — H100 (you are here)
```bash
# queue driver: START / DONE / FAILED markers (how many of the 5 H100 jobs done)
grep -hE "START|DONE|FAILED|COMPLETE" /mnt/localssd/kaiwenh/logs/overnight_queue_*.log

# action + vqa loss for the CURRENT run (replace <run_id>; strip ANSI/Rich wrapping)
sed -e 's/\x1b\[[0-9;]*m//g' /mnt/localssd/kaiwenh/logs/overnight_pref_oftvqa_token_height_10k.log \
  | tr -s ' \n' ' ' | grep -oE "Step [0-9]+, Loss[^}]*'L_action': [0-9.e-]+, 'L_vqa': [0-9.e-]+[^}]*'vqa_acc': [0-9.]+" | tail -5

# progress bar / it-per-sec for the current run
grep -aoE "[0-9]+/10000 \[[^]]*it/s[^]]*\]" /mnt/localssd/kaiwenh/logs/overnight_pref_oftvqa_token_height_10k.log | tail -1

# GPUs
nvidia-smi
```

### 4.2 Status — H200 from H100
```bash
ssh -o ConnectTimeout=12 kevin@34.34.93.23 \
  'echo "== queue =="; cat /mnt/localssd/*/logs/overnight_queue_*.log;
   echo "== run tails =="; for f in /mnt/localssd/*/logs/overnight_pref_*_10k.log; do echo "## $f"; tail -3 "$f"; done;
   echo "== procs =="; ps aux | grep -E "run_queue[.]sh|train_starvla[.]py" | grep -v grep | wc -l'
```

### 4.3 Kill everything (note the **bracket** trick so the pattern doesn't match itself)
```bash
# H100
pkill -f 'run_queue[.]sh'      # stop the queue driver first (so it won't launch the next job)
pkill -f 'train_starvla[.]py'  # then kill the 8 worker processes

# H200
ssh kevin@34.34.93.23 "pkill -f 'run_queue[.]sh'; pkill -f 'train_starvla[.]py'"
```
The `[.]` makes the regex literal-dot AND prevents the running `pkill` command line
(which contains the pattern) from matching itself.

### 4.4 Relaunch the queue (or a single run)
```bash
cd /home/kaiwenh/starVLA
# H100 full queue (background; auto-discovers paths/env):
nohup bash r-preference/overnight/run_queue.sh \
  token:height token:orient token:contact base:place base:hvlv \
  > /mnt/localssd/kaiwenh/logs/queue_driver_h100.log 2>&1 &

# H200 full queue:
ssh kevin@34.34.93.23 "cd /home/kevin/starVLA && nohup bash r-preference/overnight/run_queue.sh \
  base:height base:orient base:contact token:place token:hvlv \
  > /mnt/localssd/kevin/logs/queue_driver_h200.log 2>&1 &"

# a SINGLE run (e.g. just token:place on H100): pass one arg
nohup bash r-preference/overnight/run_queue.sh token:place \
  > /mnt/localssd/kaiwenh/logs/queue_driver_h100_place.log 2>&1 &
```

---

## 5. Morning TODO (what to run next)

### 5.1 Gate-eval the 5 TOKEN checkpoints (the real generalization test)
When all 10 are done, run **`r-preference/eval/gate_oftvqa.py`** on the 5 token
`steps_10000` ckpts to confirm the token-VQA generalizes inside the full cotrain
(taskA-val + taskB preference accuracy), not just the frozen probe.

`gate_oftvqa.py` builds the model **from the YAML** (so `state_mode: token` initializes
the token path and `phi`), sets `pretrained_checkpoint=None`, then `load_state_dict(...,
strict=False)` from the given ckpt; `predict_preference` is **token-aware**. Argparse:
`--cat --yaml --ckpt --data_root [--n_taskA 5] [--n_taskB 25] [--out]`.

> token checkpoints live on whichever machine ran that cat: **H100** has token
> height/orient/contact; **H200** has token place/hvlv. Run each eval on its machine
> (or rsync the ckpt first). Paths below are H100-form; swap `kaiwenh`→`kevin` &
> `/home/kaiwenh`→`/home/kevin` for the H200 ones (place, hvlv).

```bash
cd /home/kaiwenh/starVLA
set +u; source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh; conda activate starVLA
export CUDA_HOME=$CONDA_PREFIX HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

for cat in height orient contact; do   # H100 token cats
  python r-preference/eval/gate_oftvqa.py \
    --cat $cat \
    --yaml examples/preference/train_files/starvla_pref_stage_a_oftvqa_token_${cat}.yaml \
    --ckpt results/Checkpoints/pref_oftvqa_token_${cat}_10k/checkpoints/steps_10000_pytorch_model.pt \
    --data_root /mnt/localssd/kaiwenh/pref/data/0526/${cat} \
    --out /mnt/localssd/kaiwenh/logs/gate_token_${cat}_10k.json
done
# on H200, same loop for: place, hvlv  (kevin paths)
```

**CAVEAT to verify (this exact path has NOT been run yet):** confirm the ckpt loads
and the token actually injects. (a) Watch the `N missing, N unexpected keys` line —
`vqa_state_proj.*` and the `_vqa_state_*` buffers must be present (not "missing"); the
base VLM is rebuilt from `base_vlm` so its keys may legitimately show as unexpected.
(b) **Sanity-check a zeroed-state control gives ~0.50** — temporarily set
`self._vqa_cur_state = torch.zeros_like(...)` (or add a `--zero_state` flag mirroring
the probe's `zero_state`) and confirm acc drops to chance; if zeroed-state still scores
high, the token isn't being read and the eval is invalid. Expect token height/orient ≈
1.0 on taskA+taskB; contact high; place/hvlv capped (relational, §1.4).

### 5.2 Action-loss comparison (does VQA cotrain help / hurt the action model?)
Parse the final `L_action` per run (token vs baseline, per cat) from the logs and make
a small table/figure (mirror the `r-preference/debug/v2/make_*.py` matplotlib style).
```bash
# pull the last logged L_action from every overnight run log (H100; repeat on H200):
for f in /mnt/localssd/kaiwenh/logs/overnight_pref_*_10k.log; do
  v=$(sed -e 's/\x1b\[[0-9;]*m//g' "$f" | tr -s ' \n' ' ' \
       | grep -oE "'L_action': [0-9.e-]+" | tail -1)
  echo "$(basename "$f"): $v"
done
```
Build a 5×2 table (cat × {token L_action, baseline L_action}); the question is
help / neutral / hurt. Prior single-cat evidence (§3 of the integration doc) suggests
**neutral** (height+VQA 0.010 ≈ height no-VQA 0.008), but this 5-cat sweep is the
clean test. Save the figure under `r-preference/debug/v2/`.

### 5.3 Document the resolution
Add a short **"RESOLVED (2026-05-28 night)"** banner to the top of
[`0528-how-to-fix-last-minute.md`](0528-how-to-fix-last-minute.md) pointing here, and
record the token-VQA fix as the resolution of its F5 negative result (text-state
collapse → soft-token closes the backdoor for the 3 proprioceptive cats; relational
cats remain data-capped). Then fold the morning gate-eval + action-loss numbers into
this doc's §1.4 / §5.

---

## 6. Files created / edited tonight (reproducibility)

All verified present on the H100 as of 2026-05-28 14:05 UTC.

**Code (framework / dataset):**
- `starVLA/model/framework/vqa_cotrain_mixin.py` — **edited**: added token-VQA
  (`state_mode`, `vqa_state_proj`/`phi`, `_vqa_inject_hook`, `_set_vqa_state_embeds`,
  `_compute_vqa_state_stats`; `predict_preference`/`validate_vqa` made token-aware).
- `starVLA/model/framework/QwenOFT_VQA.py` — OFT action stream + VQA cotrain mixin
  (`QwenOFT_VQA` framework).
- `examples/preference/dataset/vqa_sample.py` — **edited**: added `contact3`/`hvlv3`
  (and `height3`) clip strategies; rewrote the `contact`, `hvlv` (+ all) `VQA_CATEGORIES`
  to head-cam 3-frame + `state_in_vqa=True`; added `ee_pose_9d_at`, `_active_arm_key`,
  `format_state_text`, `load_clip_and_state`.

**Stats:**
- `examples/preference/dataset/precompute_joint_stats.py` — **new**.
- `examples/preference/dataset/stats_contact_0526_joint.json` — **new**.
- `examples/preference/dataset/stats_hvlv_0526_joint.json` — **new**.
  (height/orient/place 0526 joint stats already existed.)

**Train configs:**
- `examples/preference/train_files/starvla_pref_stage_a_oftvqa_token_{height,orient,contact,hvlv,place}.yaml` (5).
- `examples/preference/train_files/starvla_pref_stage_a_oft_baseline_{height,orient,contact,hvlv,place}.yaml` (5).
- `examples/preference/train_files/gen_overnight_yamls.py` — generator for the above.

**Overnight / probe / eval / figures:**
- `r-preference/overnight/run_queue.sh` — the queue runner.
- `r-preference/phase1/state_token_probe.py` (+ `probe_{height,orient,place,hvlv,smoke}_lmhead.json`;
  contact probe JSON missing, see §1.4 note).
- `r-preference/debug/v2/{make_phase1_summary,make_state_separability,make_relative_separability,make_viz_v2}.py`
  and `{phase1_summary,state_separability,relative_separability,vqa_result_summary}.png`.
- `r-preference/eval/gate_oftvqa.py` — morning gate-eval script.
- `r-preference/doc/0528-token-vqa-fix-and-overnight-runbook.md` — **this doc**.

---

## 7. Key facts table

| Fact | Value |
|---|---|
| Date / box | 2026-05-28, **H100** `instance-20260316-physical-h100-2`, user `kaiwenh` |
| Repo (H100 / H200) | `/home/kaiwenh/starVLA` / `/home/kevin/starVLA` |
| H200 host | `kevin@34.34.93.23` |
| Marker token | `<|fim_pad|>` id **151662** (existing special token → no embed resize) |
| `phi` (`vqa_state_proj`) | `Linear(9,512)→GELU→Linear(512,2560)`, LR **1e-3** |
| VLM hidden size | **2560** (`text_config.hidden_size`) |
| Marker count k | 3 (= `n_frames`) |
| `lambda_vqa` / `n_vqa_per_batch` | 0.5 / 2 |
| Backbone | `QwenOFT` warm-start from OFT `steps_140000` |
| OFT ckpt (H100 / H200) | `/mnt/localssd/kaiwenh/cache/oft_ckpt/checkpoints/steps_140000_pytorch_model.pt` / same under `kevin` |
| Base VLM | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` |
| Action space | `joint` (14D qpos, `abs_qpos`) |
| Steps / save_interval | 10000 / 5000 (ckpts at 5000 + 10000) |
| Throughput | token **~0.71 s/it** (~1.4 it/s, 10k ≈ 2 h); baseline (H200) **~0.48 s/it** (~2.0 it/s, ≈ 1.3 h) |
| Launch PIDs | H100 queue `946966`; H200 queue `4142370` (per launch); current H100 driver PID seen `946966` |
| Launched / ETA | ~13:48 UTC / ~8–9 h per machine (all 10 ≈ 22:00 UTC) |
| Per-run log | `/mnt/localssd/<user>/logs/overnight_<run_id>.log` |
| Queue driver log | `/mnt/localssd/<user>/logs/overnight_queue_<hostname>.log` |
| Proprioceptive cats (token works) | height, orient, contact |
| Relational cats (data-capped) | place, hvlv (no target/obstacle pose in 0526 hdf5) |
| Zeroed-state control | must → **0.50** (proves token is read) |
