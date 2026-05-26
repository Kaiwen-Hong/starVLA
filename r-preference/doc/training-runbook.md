# StarVLA Training Runbook — H100 baseline + H200 main-method

> **Self-contained operational doc.** Where things live, how to launch, what
> breaks, how to debug. Last revised 2026-05-23.
>
> **For the spec / byte-identical lock** (§3-§6 of Stage A definition: data,
> model, prompt, norm, loss, hyperparams) see
> [`0522-a-giveobj.md`](0522-a-giveobj.md). The two docs intentionally don't
> overlap — spec there, ops here.

---

## 0. TL;DR

| | Stage A baseline (no VQA) | Stage A main-method (VQA cotrain) |
|---|---|---|
| Machine | **H100** (this box) | **H200** `kevin@34.34.93.23` |
| User | `kaiwenh` | `kevin` |
| Launch | `examples/preference/launch_pref_stage_a_baseline.sh` | `examples/preference/launch_pref_stage_a_vqa.sh` |
| YAML | `examples/preference/train_files/starvla_pref_stage_a_baseline.yaml` | `examples/preference/train_files/starvla_pref_stage_a_vqa.yaml` |
| Framework class | `QwenPI` | `QwenPI_VQA` (fork) |
| Run dir (via symlink) | `results/Checkpoints/pref_baseline_stage_a_v1_noVQA/` | `results/Checkpoints/pref_main_stage_a_v1_VQA/` |
| `results/` symlink target | `/mnt/localssd/kaiwenh/starVLA_runs/results` | `/mnt/localssd/kevin/starVLA_runs/results` |
| Git branch / HEAD | `opd` @ `ac27ba4` | `opd` @ `ac27ba4` (synced) |

### 0.1 Stage A ckpt status across 5 cats (audit 2026-05-26)

10 expected ckpt sets (5 cats × 2 variants). 9 exist on disk; 1 does NOT.

| cat | baseline (noVQA) on disk | VQA (main) on disk | training loss (final) | usable? |
|---|---|---|---|---|
| **contact** | H100 35k only (run died mid-50k, see §8.1) | H100 35k+50k, H200 full 5k–50k (10 ckpts) | converged (≈0.014) | ✓ both |
| **place**   | H100 5k–25k, H200 25k | H200 full 5k–25k | converged (bl 0.014, vqa 0.04) | ✓ both |
| **orient**  | H200 full 5k–25k | **does NOT exist** — two fresh-from-base attempts on 2026-05-25 both killed before first save_interval (see §8.3) | bl 0.04 ✓ / VQA stuck ~1.5 ❌ | bl only |
| **height**  | H100 5k–25k, H200 25k | H200 full 5k–25k (H100 dir empty) | **both stuck at L_action ≈ 1.45** (= unconditional mean MSE) ❌ | ckpts present but model predicts marginal — see §8.4 + [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) |
| **hvlv**    | H100 5k–25k, H200 25k | H200 full 5k–25k | **both stuck at L_action ≈ 1.55** ❌ | ckpts present but degenerate, same as height |

**Bottom line**: 5/10 sets are usable end-to-end (contact + place baseline+VQA, orient baseline). orient VQA is missing on disk; height + hvlv baselines + VQAs exist but their model output is essentially marginal-mean prediction (verified via gate eval baseline_CF ≈ 1.4 = noise floor in normalized action space).

**MUST always be set** (4× step-time speedup, verified):
```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```
Already in every launch script. If you launch by hand, set it.

---

## 1. Two-machine path matrix

| What | H100 (`kaiwenh`) | H200 (`kevin@34.34.93.23`) |
|------|------------------|----------------------------|
| Repo | `/home/kaiwenh/starVLA` | `/home/kevin/starVLA` |
| Conda env | `starVLA` (in `/mnt/localssd/kaiwenh/miniconda3`) | `starVLA` (in `/mnt/localssd/kevin/...`) |
| **Run dir entry (symlink)** | `/home/kaiwenh/starVLA/results` | `/home/kevin/starVLA/results` |
| **Run dir physical** | `/mnt/localssd/kaiwenh/starVLA_runs/results/` | `/mnt/localssd/kevin/starVLA_runs/results/` |
| Pref data | `/mnt/localssd/kaiwenh/pref/data/giveobj/` | `/mnt/localssd/kevin/pref/data/giveobj/` |
| HF cache | `/mnt/localssd/kaiwenh/cache/hf` (via `$HF_HOME`) | `/mnt/localssd/kevin/cache/hf` |
| Root disk | `/dev/root` 194 G | (similar small root, 82 % used 2026-05-23) |
| SSD | `/dev/md0` 5.9 T | `/dev/md0` 11 T (38 % used 2026-05-23) |
| wandb run | `kaiwenh-17-uiuc/pref-sim/pref_baseline_stage_a_v1_noVQA` | `kaiwenh-17-uiuc/pref-sim/pref_main_stage_a_v1_VQA` |

Same `wandb_entity` (`kaiwenh-17-uiuc`) and `wandb_project` (`pref-sim`) on both
machines — see `~/.netrc` for the API key (copied to H200 during initial setup,
NEVER committed).

---

## 2. Code layout — two versions, one branch

The repo holds **both versions** in parallel files. Each machine just runs
its own launcher; spec-level lock (`§3-§6` byte-identical) is enforced by
sharing the **same source files**, not by manual sync.

```
/home/<user>/starVLA/                            ← same on both machines
├── examples/preference/
│   ├── dataset/
│   │   ├── rotation.py                        # SHARED (quat ↔ 6D, scipy)
│   │   ├── prompt.py                          # SHARED (v5 strip + templates)
│   │   ├── precompute_stats.py                # SHARED (q99 norm)
│   │   ├── stats_giveobj_v1.json              # SHARED (norm stats)
│   │   ├── pref_hdf5_dataset.py               # BASELINE  (action-only)
│   │   ├── pref_hdf5_vqa_dataset.py           # MAIN      (action + VQA samples)
│   │   └── vqa_sample.py                      # MAIN      (VQA construction)
│   ├── train_files/
│   │   ├── starvla_pref_stage_a_baseline.yaml # BASELINE config
│   │   └── starvla_pref_stage_a_vqa.yaml      # MAIN     config
│   ├── tests/
│   │   ├── benchmark.py                       # SHARED (single-GPU SGD)
│   │   ├── test_smoke.py                      # BASELINE smoke
│   │   └── test_smoke_vqa.py                  # MAIN     smoke
│   ├── launch_pref_stage_a_baseline.sh        # BASELINE launcher
│   ├── launch_pref_stage_a_vqa.sh             # MAIN     launcher
│   ├── manual_run.sh                          # BASELINE tmux launcher (hardcoded)
│   └── watchdog.sh                            # BASELINE monitor (hardcoded run_id)
└── starVLA/
    ├── model/framework/
    │   ├── QwenPI.py                          # BASELINE
    │   └── QwenPI_VQA.py                      # MAIN (forks QwenPI; aggregates L_action + λ·L_vqa)
    └── dataloader/__init__.py                 # SHARED — both branches:
                                               #   elif dataset_py == "pref_hdf5":      → BASELINE
                                               #   elif dataset_py == "pref_hdf5_vqa":  → MAIN
```

**Spec lock**: `§3-§6` of `0522-a-giveobj.md` (data / model / prompt / norm /
loss / hyperparams) is enforced by:
1. `rotation.py`, `prompt.py`, `precompute_stats.py`, `stats_giveobj_v1.json`
   — single file each, both versions import.
2. `pref_hdf5_dataset.py` defines the 20D action layout and Beta time-sampling;
   `pref_hdf5_vqa_dataset.py` `from pref_hdf5_dataset import *` and only adds
   the VQA channel.
3. The two YAMLs intentionally agree on every action-stream knob; diff is
   *only* `framework.name` (`QwenPI` vs `QwenPI_VQA`), the dataset adapter,
   and the VQA loss weight.

---

## 3. Launch scripts family

### 3.1 Roles

| Script | Role | Currently bound to |
|--------|------|---------------------|
| `launch_pref_stage_a_baseline.sh` | **Production baseline launcher** (foreground; for nohup wrap if needed) | H100 baseline run_id |
| `launch_pref_stage_a_vqa.sh` | **Production main-method launcher** (host-aware: detects `kaiwenh` vs `kevin` conda; supports profile mode) | H200 main-method run_id |
| `manual_run.sh` | **tmux-friendly interactive launcher** for baseline. `Ctrl-C` propagates cleanly. GPU pre-check + `CLEAN=1` cleanup. | Baseline YAML (hardcoded) |
| `watchdog.sh` | **30-min monitoring loop**. Touches `WATCHDOG_ALERT` on OOM / PROCESS_DEAD / STEP_STUCK / STEP_DEGRADED. | Baseline run_dir (hardcoded). See §7.5 — known false-positive bug. |

If you need `manual_run.sh` or `watchdog.sh` for the VQA run, **parameterize
first** — they were written for baseline only. Don't dual-launch them as-is.

### 3.2 Env var reference

| Var | `launch_baseline.sh` | `launch_vqa.sh` | `manual_run.sh` |
|-----|----------------------|------------------|-------------------|
| `NUM_GPUS` (4 → grad_accum=2; ≥8 → grad_accum=1) | ✓ | ✓ | – (8 fixed) |
| `MAX_STEPS` (override `trainer.max_train_steps`) | – | ✓ | ✓ (called `STEPS`) |
| `NO_SAVE` (bump eval_interval to ∞ for clean profile) | – | ✓ | – |
| `CLEAN` (kill existing train/watchdog/accelerate, 30s GPU wait, then go) | – | – | ✓ |
| `MAIN_PORT` (default 29501 to avoid clash with default 29500) | – | – | ✓ |

### 3.3 Startup hard-fail checks (in all 3 launchers)

1. `mountpoint -q /mnt/localssd` — refuse if SSD not mounted (would silently fill root)
2. `[ -L /home/<user>/starVLA/results ]` — refuse if `results/` is not a symlink

Plus `manual_run.sh` adds:

3. `nvidia-smi` max-used > 5000 MiB → refuse unless `CLEAN=1`

### 3.4 The actual command (what all 3 wrap)

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes <NUM_GPUS> \
  starVLA/training/train_starvla.py \
  --config_yaml examples/preference/train_files/starvla_pref_stage_a_<baseline|vqa>.yaml \
  --trainer.gradient_accumulation_steps <1 or 2>
```

Eff batch = `8 × NUM_GPUS × grad_accum` (locked to 64). DeepSpeed Zero2 + bf16.

---

## 4. Storage layout (the symlink design)

```
H100:
  /home/kaiwenh/starVLA/results  →  /mnt/localssd/kaiwenh/starVLA_runs/results

H200 (set up 2026-05-22 by Kaiwen, already in place):
  /home/kevin/starVLA/results    →  /mnt/localssd/kevin/starVLA_runs/results

Inside results/Checkpoints/<run_id>/:
  checkpoints/steps_<N>_pytorch_model.pt   (16 G each; trainer never auto-deletes — see §7.4)
  config.yaml
  train.log
  train.log.attempt*                        (rotation history; if launcher was Ctrl-C'd and restarted)
  summary.jsonl                             (one line per save — AUTHORITATIVE step counter)
  wandb/
  watchdog.log
  WATCHDOG_ALERT                            (touched on real alerts; see §7.5)
  STATUS.md
  dataset_statistics.json
```

### Why symlink over absolute paths in YAML

| Aspect | Symlink (chosen) | Absolute paths in YAML |
|--------|------------------|-------------------------|
| YAML changes per machine | 0 | One edit per machine |
| Git diff exposure | `results/` already gitignored | Absolute paths leak into VCS |
| Cross-machine portability | Same YAML works everywhere | Need host-aware logic |
| Downstream scripts using `results/...` | All keep working unchanged | Each script needs update |
| Rollback | `rm symlink && mv data back` | Revert YAML + restart |

### How the H100 symlink was created (2026-05-23)

```bash
# 1. SSD target
mkdir -p /mnt/localssd/kaiwenh/starVLA_runs/results

# 2. Move existing data (sha256-verified before swap:
#    e8ae1b4cd539e4d7f726935daaa719b31d901b6ce4326169b78420a7a01b7265
#    matched on both sides for steps_35000_pytorch_model.pt)
rsync -ah /home/kaiwenh/starVLA/results/ /mnt/localssd/kaiwenh/starVLA_runs/results/

# 3. Backup-rename, then create the symlink
mv /home/kaiwenh/starVLA/results /home/kaiwenh/starVLA/results.pre_symlink_<timestamp>
ln -s /mnt/localssd/kaiwenh/starVLA_runs/results /home/kaiwenh/starVLA/results

# 4. Sanity check
ls -la /home/kaiwenh/starVLA/results
readlink -f /home/kaiwenh/starVLA/results/Checkpoints/.../some_ckpt.pt

# 5. After verifying everything reads through the symlink, delete the backup
rm -rf /home/kaiwenh/starVLA/results.pre_symlink_<timestamp>
```

Root disk: 102 G used → 87 G used (15 GB reclaimed by removing the verified backup).

### How to set this up on a new machine

```bash
SSD_DIR=/mnt/localssd/$USER/starVLA_runs/results
mkdir -p "$SSD_DIR"
[ -e $HOME/starVLA/results ] && mv $HOME/starVLA/results $HOME/starVLA/results.bak
ln -s "$SSD_DIR" $HOME/starVLA/results
```

---

## 5. Startup checklist for any new training run

Before launching anything on a new branch / new config:

1. **SSD mounted?** `mountpoint /mnt/localssd` → should say `is a mountpoint`.
2. **Symlink in place?** `ls -la $HOME/starVLA/results` → should show `→ /mnt/localssd/.../results`.
3. **GPUs free?** `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1` → < 1000 MiB.
4. **No old training alive?** `pgrep -af "train_starvla|accelerate"` → empty.
5. **SSD has headroom?** `df -h /mnt/localssd | tail -1` → ≥ 200 GB free recommended (10 × 16 GB ckpts + cache + buffer).
6. **HF cache primed?** `ls -lh $HF_HOME/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/` → ~10 GB.
7. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** is set (4× speedup; §7.1).
8. **git branch synced** if running cross-machine: `git log --oneline -1` matches.

The launcher hard-fails on items 1-2, exports item 7 unconditionally, and
`manual_run.sh` also checks item 3 (gated by `CLEAN=1`). The rest are operator
judgment.

---

## 6. Known footguns

### 6.1 `rm -rf results/` (trailing slash) deletes the SSD contents

```bash
rm /home/kaiwenh/starVLA/results        # safe: removes ONLY the symlink
rm -rf /home/kaiwenh/starVLA/results/   # DANGER: follows symlink, wipes SSD
```

If you really want to delete the ckpts, do it explicitly on the SSD path:
`rm -rf /mnt/localssd/kaiwenh/starVLA_runs/results/...`.

### 6.2 SSD unmount = writes go to root instead

If `/mnt/localssd` fails to mount, the symlink target stops existing as a real
SSD path, but Linux may still create files under `/mnt/localssd/...` on the
**root** filesystem, silently filling root again. All launchers now
**hard-fail** at startup via `mountpoint -q /mnt/localssd`. If you ever
manually `accelerate launch` outside the wrappers, do the same check.

### 6.3 `find /home/.../starVLA -name "*.pt"` misses the SSD ckpts

`find` does not follow symlinks by default. Use:
```bash
find -L /home/kaiwenh/starVLA -name "*.pt"
# or, faster:
find /mnt/localssd/kaiwenh/starVLA_runs -name "*.pt"
```

### 6.4 `du -sh starVLA/` under-reports

`du` of the repo root will report only the symlink (a few bytes), not the SSD
contents. This is mostly a feature — keeps repo size reports honest — but be
aware when budgeting disk.

### 6.5 SSD shared with `dreamzero/` and others

`/mnt/localssd/` had 955 G free on H100 / 6.3 T free on H200 as of 2026-05-23.
Each Stage-A training run produces ≤ 160 G (10 × 16 GB ckpts kept) + small
logs. **Do not delete anything under `/mnt/localssd/kaiwenh/`** without
checking — `robomme/` and `pref/` projects share that tree. Cleanup target is
`dreamzero/` only.

---

## 7. Known ops quirks (the "why-file" reference)

This is the "I forget why we did X" lookup table.

### 7.1 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory

**Verified 2026-05-22 on H100 8-GPU Zero2**: 1.17 s/step vs 4.4 s/step **(4× speedup)**.
Without it, PyTorch's caching allocator hits the `cudaMalloc` slow path
frequently on high-rank GPUs because Zero2's sharded optimizer state fragments
unevenly across ranks.

This was the difference between "10 h" (Kaiwen's other-machine reference) and
"60 h" (Claude's initial mis-estimate). It's now hardcoded into all three
launchers. **If you ever launch outside the wrappers, set it manually.**

### 7.2 `gradient_checkpointing: true` even though single-GPU bench said OFF is faster

Single-GPU SGD benchmark on `examples/preference/tests/benchmark.py` showed
grad_ckpt **OFF** was ~8 % faster (875 vs 953 ms per step). But under
distributed Zero2 + 8 GPUs:

- GPU 7 (the highest-rank shard) dropped to **0.4 GB free** with grad_ckpt OFF.
- Zero2 shards optimizer state **unevenly** across ranks (later ranks carry
  heavier shards). The single-GPU bench has no shard imbalance to expose, so
  it doesn't predict the OOM risk.
- The 8 % single-GPU win does not materialise in distributed training (which
  is compute-bound, not activation-memory-bound).

**Decision**: keep `gradient_checkpointing: true` in
`starvla_pref_stage_a_baseline.yaml`. Don't second-guess based on the
single-GPU number alone.

### 7.3 `MAIN_PORT=29501` (not accelerate's default 29500)

If a previous training run didn't release `29500` cleanly, or two
trainings are launched concurrently on one host, the second launch fails
with "address already in use". `manual_run.sh` defaults to `29501`. To run
a third concurrent job, set `MAIN_PORT=29502` etc.

Practical: if you see "port in use", `pkill -9 -f "accelerate|pt_elastic"`
and wait 30 s, or just bump `MAIN_PORT`.

### 7.4 The trainer NEVER auto-deletes old ckpts — H100's "rolling" was a human

The save logic in `starVLA/training/train_starvla.py:220-239` writes a new
file `steps_<N>_pytorch_model.pt` and never deletes earlier ones. There is no
`save_total_limit` field anywhere.

On H100's 2026-05-22 baseline run we saw only `steps_35000_*.pt` and assumed
the trainer was rolling/overwriting. **It wasn't.** Kaiwen had been manually
`rm`-ing earlier ckpts during training because `/home` (194 GB root, holding
the run dir back then) was filling up: 16 GB × 10 = 160 GB does not fit.

H200's 2026-05-22 main-method run kept all 10 ckpts intact (`steps_5000`
through `steps_50000` on disk, 163.6 GB total) because the run dir was already
on `/mnt/localssd/` (11 TB) and there was no disk pressure.

**Now that H100 also points `results/` at the SSD**, there is no reason to
hand-delete. Let the trainer keep all 10.

### 7.5 Watchdog `STEP_STUCK` alerts are false positives if you Ctrl-C and tmux-restart

`watchdog.sh` tracks step progress by scraping `train.log`. But `train.log`
is whatever-was-opened by the original PID. If you `Ctrl-C` and relaunch
manually under tmux, the new run's stdout goes to **tmux** (or to a rotated
`train.log.attemptN.<pid>` file the wrapper creates), and `train.log` itself
freezes at the old PID's startup output.

`summary.jsonl`, written by the trainer itself
(`train_starvla.py:237`), is the authoritative step counter — **one line per
`save_interval`**, never lies, never depends on stdout redirection.

This bug produced **840 KB of false `STEP_STUCK` alerts** during the
2026-05-22 baseline run (every 30 min watchdog cycle for hours, even though
the run was healthy and ckpts were saving). Mitigation today: **ignore
`WATCHDOG_ALERT` contents unless `summary.jsonl` ALSO stops advancing**.

**TODO**: rewrite watchdog to read `summary.jsonl` not `train.log`. Until
then, treat watchdog alerts as a *hint*, not a verdict; cross-check
`summary.jsonl` and `ls -lt results/.../checkpoints/`.

### 7.6 `set -e` + `bc` returns 1 = silent watchdog death (history, fixed)

`watchdog v1` had `set -e` and `(( $(bc -l <<< "a < b") ))` which returns
exit-code 1 (not just value "0") when the inequality is false. With `set -e`
that killed the watchdog on the very first OK reading. v2 removes `set -e`
(keeps `-uo pipefail`). If you ever rewrite the watchdog, **don't** add
`set -e` back.

### 7.7 Why the launchers `set +u` then `set -eo pipefail` (no `-u`)

Conda's activation hooks reference variables before defining them
(`SYS_SYSROOT`, `_CE_M`, etc.). With `set -u`, sourcing
`conda.sh` aborts. We:

1. `set +u` before sourcing conda
2. `conda activate starVLA`
3. `set -eo pipefail` (keep `-u` OFF for the rest)

This is intentional. DeepSpeed's startup hooks also have unbound vars.

---

## 8. Postmortems

### 8.1 H100 baseline 2026-05-22 → died at step ~35 k (70 % done)

**Timeline (UTC):**
- 2026-05-22 11:09 — launched in tmux session 21 via `manual_run.sh`
- 11:31 — `train.log` frozen at step 187 (this is the stale tmux artefact, not
  a real stuck step; the actual stdout went to tmux scrollback)
- 23:02 — `steps_35000_pytorch_model.pt` written (last successful save)
- 23:02 → 2026-05-23 ~00:39 — process died sometime in this window (next save
  at step 40000 ≈ 00:39 never occurred)
- 01:06 — watchdog's last log line (it survived past the trainer's death and
  kept reporting `STEP_STUCK` because train.log was frozen — see §7.5)
- Sometime between 01:06 and 07:00 — watchdog itself died (tmux session 21
  gone by morning)

**Confirmed not the cause:**
- Not OOM (no "out of memory" in any log)
- Not the `expandable_segments` env var (was set; step time was healthy 1.17 s)
- Not disk full (16 GB rolling, /home had 92 GB free that night)
- Not watchdog killing the trainer (watchdog only writes files, never sends signals)

**Unconfirmed (no smoking gun in disk/logs):**
- Possible network blip causing NCCL hang then auto-kill
- Possible OOM killed by external (machine eviction / sibling process)
- Possible single-process crash with no stack trace because stdout was tmux'd

**Lessons baked into this runbook:**
- Watchdog must read `summary.jsonl`, not `train.log` (§7.5)
- Always redirect stdout to a file from the launcher itself, not just rely on
  tmux scrollback (`exec accelerate launch ... 2>&1 | tee train.log` if needed)
- 35 k ckpt **preserved on SSD** — can resume training from there or use as
  partial eval baseline (chunk policy already useful at 35 k since loss is
  visibly converging in wandb)

### 8.2 H200 main-method 2026-05-22 → 2026-05-23 — clean 50 k completion

**Timeline (UTC):**
- 2026-05-22 11:09 — launched as PID 2901345 via `launch_pref_stage_a_vqa.sh`
- 2026-05-22 13:10 — `steps_5000_pytorch_model.pt` (first save, on schedule)
- 2026-05-23 05:29 — `steps_50000_pytorch_model.pt` (final, on schedule)
- Total wall: ~18 h 20 min (vs profile-extrapolated 17.8 h; slightly slower from
  eval pauses) — process exited cleanly afterwards.

**Final state (verified via SSH, 2026-05-23 08:26 UTC):**
- 10 ckpts intact: `steps_5000` ... `steps_50000`, each 16.36 GB. Total 163.6 GB.
- `summary.jsonl` has 10 entries, all 5 k multiples, in order.
- Loss snapshots: step 100 `L_action=2044.6, L_vqa=1.36, vqa_acc=0.0`; step 200
  `L_action=1.57, L_vqa=0.0018, vqa_acc=1.0`. VQA converges within ~200 steps.
- log: `/mnt/localssd/kevin/logs/pref_vqa_50k_20260522_110949.log`
- wandb: `kaiwenh-17-uiuc/pref-sim/pref_main_stage_a_v1_VQA`

**Pre-launch patches Kaiwen applied to H200 (recorded for reproducibility):**
- `~/.netrc` copied from H100 → H200 for wandb auth (as user `kevin`)
- `/home/kevin/starVLA/results` symlinked to `/mnt/localssd/kevin/starVLA_runs/`
  (the H200 root was 36 GB free at the time — same disk-pressure motivation
  as H100, fixed pre-emptively here)
- `h5py` pip-installed into H200 `starVLA` conda env
- Two robomme paired-ablation FSDP-4 jobs (PIDs 2841421/2841422) killed to free
  GPUs. Resume notes at
  `/home/kevin/robomme/RESUME_NOTES/killed_for_pref_vla_20260522_105028.md`.
- Trainer (`train_starvla.py:_train_step`) patched to relay `log/*` keys
  through to wandb (was previously dropping `L_action` / `L_vqa` / `vqa_acc`).

### 8.3 Orient VQA 2026-05-25 — two fresh retrain attempts both killed pre-save

Background: The original orient-VQA run (2026-05-24, `temp-523-h200.sh`) was
killed mid-training and resumed via `IS_RESUME=1` from steps_10000. Loss never
recovered (L_action bouncing 1.2–1.7 across steps 10k–25k). 0525 doc §3.2
diagnosed this as Adam-state corruption and proposed fresh-from-base retrain.

**Attempt 1** — `temp-525-h200.sh::02-orient-vqa-fresh` (wrapper):
- 2026-05-25 11:29:47Z BEGIN; deleted old ckpts dir as pre-launch step
- 11:31:17Z accelerate launch; wandb run `7dipoaaw`
- Step 200 L_action=1.58 (already at unconditional-mean MSE)
- ~11:35:50Z step 214: `Killed` signal (likely OOM-killer or external)
- No ckpt saved (save_interval=5000)

**Attempt 2** — `examples/preference/temp-524-retrain-orient-vqa.sh` in tmux
`pref_524_orient` (manually launched after attempt 1 died):
- 2026-05-25 11:41:02Z tmux session created
- 11:41:10Z BEGIN; wandb run `0yo8o5vn`
- Step 200 L_action=1.59
- Step 1000 L_action=1.64; step 2000 L_action=2.65; step 2500 L_action=1.64
  → loss **bouncing in [1.3, 2.7] with no downward trend** over 2300 steps
- ~12:36Z user Ctrl-C'd 4 times in the tmux pane
- No ckpt saved (still pre-save_interval at step 2517)

**The diagnostic value**: Attempt 2's loss pattern is **identical to the failed
resume** — bouncing in 1.3–2.7 range, no convergence. This falsifies 0525's
"resume corrupted optimizer" hypothesis: a fresh-from-base run with brand-new
optimizer state still fails identically.

For comparison, orient **baseline** (same data, no VQA) converged smoothly:
657 → 1.58 (step 200) → 0.17 (step 5k) → 0.04 (step 25k).

→ Real root cause is something specific to the VQA path on orient data, not
optimizer recovery. See [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
§3.2 for the falsifying evidence and recommended next steps (lambda_vqa
ablation first, before another fresh retrain).

### 8.4 Height + hvlv baseline 2026-05-23 → 2026-05-24 — completed 25k but loss never converged

Both runs completed cleanly (5 ckpts saved each, summary.jsonl entries 5k–25k).
But the loss trajectory:

| run | step 200 | step 5k | step 12k | step 18k | step 25k |
|---|---|---|---|---|---|
| height baseline (wandb rdmadvqb) | 1.65 | 1.55 | 1.42 | 1.34 | 1.45 |
| hvlv baseline (wandb dzb9idkc)   | 1.61 | 1.51 | 1.54 | 1.54 | 1.55 |
| (reference: place baseline tz3np3jf) | 1.74 | 0.09 | 0.06 | 0.04 | 0.014 |

For comparison, place baseline (same arch, same hyperparams, same wrapper) on
the SAME GPUs converged from 1.74 (step 200) to 0.014 (step 25k).

→ height + hvlv ckpts are on disk but their model output is essentially
marginal-mean prediction (L_action ≈ `Var[action] + Var[noise]` per element).
The gate eval confirms this: baseline_CF_MSE ≈ 1.41 for both (= noise-floor
in normalized [-1,1] action space; for comparison contact = 0.031). 0525
doc §3.3 / §3.4 blamed the data (mislabel / weak signal); 0526 re-investigation
shows data is clean (height z_release S/N=24, hvlv transport perp S/N=5–10).
Real cause is currently unknown. See
[`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) §6 for hypotheses to
test (random seed / lambda_vqa ablation / warm-start from a working ckpt) and
§7 for per-cat recommendations.

---

## 9. Observed sizing (Stage-A, Qwen3-VL-4B + LayerwiseFM 3.3B DiT, 8 GPUs)

| Resource | Per file / event | Steady state |
|----------|-------------------|---------------|
| Single ckpt | 16.36 GB (fp32 weights + Zero2 optimizer states gathered to rank 0) | All 10 kept on SSD → 163.6 GB per full 50 k run |
| Step time | 1.17 s/step (H100, `expandable_segments`) / 1.28 s/step (H200) | — |
| `train.log` (live) | ~10-100 KB / 1 k steps | < 5 MB / run |
| `wandb/` (offline staging) | ~100 KB / step | < 500 MB / run |
| `summary.jsonl` | 1 line / save | negligible |
| Dataset on disk | — | 11 GB (1600 demos × ~7 MB) |
| HF Qwen3-VL-4B snapshot | — | ~10 GB (one-time) |
| Conda env `starVLA/` | — | ~12 GB |
| Peak GPU memory (8× H100, grad_ckpt ON) | — | ~70 GB / GPU; GPU 7 down to ~2.6 GB free, others 3-5 GB free |

Wall-time estimate: 50 k steps ÷ 3600 ≈ 14-17 h on 8× H100/H200. Plan for ~18 h
including eval / save overhead.

---

## 10. Troubleshooting decision tree

### "OOM during forward / backward"
1. Is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` exported? → §7.1
2. Is `gradient_checkpointing: true` in YAML? → §7.2
3. `nvidia-smi --query-gpu=memory.free` — is GPU 7 (highest-rank shard) the
   worst? If yes that's the Zero2 imbalance, grad_ckpt mandatory → §7.2
4. Did you bump `per_device_batch_size` above 8? Stay at 8 for Zero2 + Qwen3-VL-4B base.

### "Address already in use"
1. `pgrep -af "accelerate|pt_elastic|train_starvla"` — any survivors?
2. `pkill -9 -f "..."` to kill them, wait 30 s
3. Or `MAIN_PORT=29502 bash manual_run.sh` to side-step → §7.3
4. `CLEAN=1 bash manual_run.sh` does the kill+wait+launch for you

### "Step counter frozen but ckpts still advancing"
→ §7.5. The watchdog's `STEP_STUCK` is reading `train.log` which got
   redirected. `cat summary.jsonl` is the truth.

### "results/ disappeared / can't find ckpts"
1. `ls -la results` — shows a symlink target? Confirm it points to
   `/mnt/localssd/<user>/starVLA_runs/results` (§4)
2. `mountpoint /mnt/localssd` — yes? If not the launcher should have refused;
   if it didn't, manual launch suspected (§6.2)
3. `rm -rf results/` with trailing slash recently? → §6.1. If yes, ckpts on
   SSD are gone. There is no undo.

### "Training slowed down 4× / 3 s+ per step"
→ §7.1. Double-check the env var actually exported in the process:
```bash
cat /proc/<train_pid>/environ | tr '\0' '\n' | grep PYTORCH_CUDA
```

### "/dev/root disk full"
1. `ls -la results` — is it a symlink? If not, runs have been writing to root.
   See §4 to fix.
2. `du -sh /home/$USER/starVLA/*` — is HF cache misplaced? Should be
   `$HF_HOME=/mnt/localssd/$USER/cache/hf`, not the default `~/.cache/huggingface`.
3. `du -sh /tmp /var/tmp` — leaked temp files from a crashed accelerate? Safe
   to clear if no current run.

### "Watchdog spamming WATCHDOG_ALERT but training looks fine"
→ §7.5. Cross-check `tail summary.jsonl`. If the jsonl is advancing, watchdog
   is the one wrong, not the trainer. Plan: rewrite watchdog (TODO).

### "L_action stuck at ~1.5 after step 200, never goes down"
This is the model converging to **unconditional mean-velocity prediction**
(L_action ≈ `Var[action] + Var[noise]` per element). Verified pattern for
height baseline (1.45 stuck), hvlv baseline (1.55 stuck), and orient VQA
(1.5 stuck across both resume and fresh retry). Data + pipeline are clean
(verified per `0526-corrections-and-pipeline-audit.md` §5). Real cause is
currently unknown. Do NOT:
- "Fix" data via 0525 §5.1's z-final script — that script has 2 bugs (left
  arm only; not release-frame); height + hvlv data are clean (S/N 24 / 5–10).
- Launch another fresh-from-base retrain expecting recovery — orient VQA has
  failed in both resume and fresh modes with identical loss patterns.
Do try (cheapest first):
1. Different random seed (`--trainer.seed 43`) — ~3h, isolates loss-landscape stochasticity.
2. For orient VQA specifically: `framework.vqa.lambda_vqa: 0.0` ablation — ~1h, isolates whether VQA path is destabilizing the backbone.
3. Warm-start from a working same-cat ckpt (e.g. height baseline init from place baseline ckpt) — ~1h, isolates init-into-degenerate-basin.

### "Wandb run missing or duplicate"
- Wandb dir lives at `$run_dir/wandb/`, which moves with the symlink. If the
  symlink wasn't in place when the run started, wandb staging may have written
  to root. Check `find -L / -name "wandb-resume.json" 2>/dev/null`.
- Two runs with the same `run_id` → wandb auto-appends; pick one in the UI.

---

## 11. Related files

- [`examples/preference/launch_pref_stage_a_baseline.sh`](../../examples/preference/launch_pref_stage_a_baseline.sh) — baseline launcher.
- [`examples/preference/launch_pref_stage_a_vqa.sh`](../../examples/preference/launch_pref_stage_a_vqa.sh) — main-method launcher (host-aware).
- [`examples/preference/manual_run.sh`](../../examples/preference/manual_run.sh) — tmux launcher for baseline.
- [`examples/preference/watchdog.sh`](../../examples/preference/watchdog.sh) — monitor (false-positive bug — see §7.5).
- [`examples/preference/tests/benchmark.py`](../../examples/preference/tests/benchmark.py) — single-GPU SGD bench (caveat in §7.2).
- [`examples/preference/tests/test_smoke.py`](../../examples/preference/tests/test_smoke.py) — baseline dataset / dataloader / forward smoke.
- [`examples/preference/tests/test_smoke_vqa.py`](../../examples/preference/tests/test_smoke_vqa.py) — main-method smoke.
- [`starVLA/training/train_starvla.py`](../../starVLA/training/train_starvla.py) — trainer (ckpt logic L220-239, log relay L237).
- [`starVLA/model/framework/QwenPI.py`](../../starVLA/model/framework/QwenPI.py) — baseline framework.
- [`starVLA/model/framework/QwenPI_VQA.py`](../../starVLA/model/framework/QwenPI_VQA.py) — main-method framework (forks QwenPI).

---

## 12. Spec doc & memory pointers

- **Spec (byte-identical lock §3-§6)**: [`0522-a-giveobj.md`](0522-a-giveobj.md) —
  data, model, prompt, norm, loss, hyperparams. Don't put ops detail there.
- **Data setup snapshot**: [`../setup.md`](../setup.md) — HF download +
  unzip recipe (2026-05-21).
- **Instruction inject**: [`../9addinstructions.md`](../9addinstructions.md) —
  paraphrase tree from Kaiwen's personal box, copied into each task.
- **Strip paraphrase for task B**: `../strip_preference_taskb.py`.
- **Cross-session memory** (Claude private, not committed to repo):
  `~/.claude/projects/-home-kaiwenh-starVLA/memory/*.md`. Contents you can't
  read from the repo:
  - `project_pref_vla_stage_a.md` — career-importance flag + this run's state
  - `project_pref_vla_stage_a_vqa_overnight.md` — H200 launch state
  - `reference_starvla_storage_layout.md` — pointer back to this doc
  - `reference_pref_data_and_starvla.md` — data path index
  - `feedback_pref_vla_proposal_first.md` — Kaiwen's "verify before code" rule
  - `feedback_setup_env_is_expected.md` / `feedback_cleanup_target_dreamzero.md`
    — non-pref-VLA feedback
