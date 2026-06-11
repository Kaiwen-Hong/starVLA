# 0529 — H200 Stage-B `vla_train_dataloader is None` crash (diagnosis handoff)

Date: 2026-05-29. Author handoff doc for a fresh engineer with zero context.
Status of bug: **ROOT CAUSE IDENTIFIED with high confidence** (see §4 + §5 H1).
You are reading this on the **H100 box** (`/home/kaiwenh/starVLA`); the failing box is the **H200** (`kevin@34.34.93.23`).

> NOTE: This doc only diagnoses. Do NOT launch training or push a fix from this doc
> unless explicitly told to. The single-line fix is in §6 but is left for the operator.

---

## 1. Summary

On the **H200** box, Stage-B training (`dataset_py: pref_hdf5_stageb`) crashes ~30s after
launch, **before any training step**, on all 8 ranks with `TypeError: 'NoneType' object is
not iterable` at `iter(self.vla_train_dataloader)`. The same Stage-B config runs fine on H100.

**Crux (one line):** the H200's `starVLA/dataloader/__init__.py` is an **out-of-sync / older
copy** (89 lines) whose `build_dataloader()` dispatcher has branches only up to
`pref_hdf5_vqa` and is **missing the `pref_hdf5_stageb` branch** that exists on H100
(105 lines, branch at lines 90–105). `build_dataloader()` has **no `else`/fallthrough
`return`**, so when called with `dataset_py == "pref_hdf5_stageb"` it falls off the end of
the function and returns `None` implicitly. That `None` is carried all the way to
`iter(None)` and crashes. This exactly explains the diagnostic contrast: `pref_hdf5_vqa`
works on H200 (its branch exists), `pref_hdf5_stageb` does not (its branch is missing), and
H100 has both branches.

This matches the still-**pending Task #17 "Sync H100↔H200 code"** — the H200 sync was
partial: the Stage-B *dataset file* and the *trainer* got copied, but the *dataloader
dispatcher* did not.

---

## 2. Quick repro

On H200:
```bash
ssh kevin@34.34.93.23                 # the H200 box
cd /home/kevin/starVLA
# yaml ships with H100 (kaiwenh) data paths; repoint to kevin paths first:
sed -i "s#/mnt/localssd/kaiwenh/pref#/mnt/localssd/kevin/pref#g" \
  examples/preference/train_files/starvla_pref_stageb_main_hvlv_geom.yaml
setsid bash r-preference/overnight/stageb_queue.sh main:hvlv_geom \
  > /mnt/localssd/kevin/logs/h200_main_hvlv.log 2>&1
```
- Per-run log lands at: `/mnt/localssd/kevin/logs/stageb_pref_stageb_main_hvlv_geom.log`
- Wrapper log: `/mnt/localssd/kevin/logs/h200_main_hvlv.log`

Fast check (no training needed) — confirm the dispatcher is missing the branch:
```bash
ssh kevin@34.34.93.23 'grep -n "pref_hdf5_stageb" /home/kevin/starVLA/starVLA/dataloader/__init__.py; \
  wc -l /home/kevin/starVLA/starVLA/dataloader/__init__.py'
# Expected (buggy state): grep prints NOTHING, wc prints 89.
```

---

## 3. Observed facts (CONFIRMED this session)

### 3.1 The traceback (all 8 ranks, identical)
```
[rankN]:   File "/home/kevin/starVLA/starVLA/training/train_starvla.py", line 548, in main
[rankN]:     trainer.train()
[rankN]:   File "/home/kevin/starVLA/starVLA/training/train_starvla.py", line 363, in train
[rankN]:     self._create_data_iterators()
[rankN]:   File "/home/kevin/starVLA/starVLA/training/train_starvla.py", line 344, in _create_data_iterators
[rankN]:     self.vla_iter = iter(self.vla_train_dataloader)
[rankN]: TypeError: 'NoneType' object is not iterable
→ torch.distributed.elastic.multiprocessing.errors.ChildFailedError (exitcode 1)
```
i.e. `self.vla_train_dataloader` is `None`.

### 3.2 Smoking gun in the H200 log
In `/mnt/localssd/kevin/logs/stageb_pref_stageb_main_hvlv_geom.log`:
- Line 414 prints `Creating VLA Dataset with Mixture 'hvlv_taskB_main'`
  (this print is in `prepare_data`, emitted **before** `build_dataloader` is called).
- **No `[PrefHDF5StageBDataset/...]` lines appear anywhere** — no `cache filter:`,
  no `unique_eps=`. Those prints live inside the Stage-B dataset's
  `_filter_index_by_cache` / `_summarize_index`. Their absence proves the **Stage-B
  dataset class is never instantiated** — i.e. `build_dataloader` returned `None`
  without ever entering a matching branch.
- Crash lands at line ~660+ (`iter(self.vla_train_dataloader)`).

### 3.3 The 4 causes ALREADY ruled out (do not re-investigate)
| # | Ruled-out cause | How it was falsified |
|---|---|---|
| 1 | Data path wrong | yaml `data_root_dir` switched kaiwenh→kevin; data dir exists; still None. |
| 2 | Missing pseudo_label_cache | H200 HAS `r-preference/eval/pref_pseudo_labels_hvlv_B.json` (main-mode cache, n=100). |
| 3 | cache↔data mismatch | cache keys like `stamp_seal6_hv/episode0`; H200 dir `/mnt/localssd/kevin/pref/data/0526/hvlv/taskB/stamp_seal6_hv/data/` EXISTS. Verified in python: `task_group in os.listdir(root) == True`, `data_exists == True`. |
| 4 | auto-micro-batch issue | The micro-batch fix is present in H200 `train_starvla.py` (lines 145–157) and the log shows it fired: `train_micro_batch_size_per_gpu=8 (explicit)`. So the None is NOT this. |

> These ruled-out causes all assume the Stage-B dataset gets built. The real bug is
> upstream of all of them: the dataset is **never reached** because the dispatcher has no
> branch for it.

### 3.4 The key diagnostic contrast (CONFIRMED)
- Stage-A `pref_hdf5_vqa` **runs fine on H200** (the contact/height/orient baseline +
  OFT-VQA ckpts were all trained there).
- Stage-B `pref_hdf5_stageb` → **None dataloader on H200**.
- H100 runs `pref_hdf5_stageb` **fine** (every Stage-B run this project did was on H100).

### 3.5 Versions — NOT a factor (CONFIRMED equal)
Both boxes (env `starVLA`) report **identical** versions:
```
torch 2.6.0+cu124   accelerate 1.5.2   deepspeed 0.16.9   transformers 4.57.0
```
H100 cmd: `source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh && conda activate starVLA && python -c "import torch,accelerate,deepspeed,transformers as t;print(...)"`
H200 cmd: same with `/mnt/localssd/kevin/miniconda3/...`. → **Hypothesis "version mismatch" is FALSIFIED** (see §5 H2).

### 3.6 File-state diff between boxes (CONFIRMED — this IS the bug)
| File | H100 (`/home/kaiwenh/starVLA`) | H200 (`/home/kevin/starVLA`) |
|---|---|---|
| `starVLA/dataloader/__init__.py` | **105 lines**, HAS `elif dataset_py == "pref_hdf5_stageb"` (lines 90–105) | **89 lines**, last branch is `pref_hdf5_vqa` (return at line 89), **NO stageb branch** |
| `examples/preference/dataset/pref_hdf5_stageb_dataset.py` | present | **present** (13530 bytes, `action_space` passthrough at line 290) — synced OK |
| `starVLA/training/train_starvla.py` | micro-batch fix lines 145–157 | **present** (synced OK) |

So the H200 sync was **partial**: dataset file + trainer copied, but `dataloader/__init__.py`
(the dispatcher) was left on its older version. On H100 the stageb branch was introduced by
commit `173b0aa` ("Stage B contact: add prompt extension + StageB dataset + offline labeler
+ cache"); H200's copy of this one file predates that commit.

---

## 4. Code walkthrough — every path to `vla_train_dataloader = None`

All file refs below are the **H100** copies unless stated; the H200 copy of
`dataloader/__init__.py` is the one that differs.

### 4.1 The call chain (where None originates and where it crashes)
1. `starVLA/training/train_starvla.py:535` —
   `vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)`
2. `train_starvla.py:80-88` `prepare_data()` —
   ```python
   logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")  # line 82 (the print seen at H200 log L414)
   vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)  # line 83  ← returns None on H200
   ...
   return vla_train_dataloader
   ```
   `prepare_data` does **no None-check**; whatever `build_dataloader` returns is passed up verbatim.
3. `train_starvla.py:538-545` — the `None` is stored: `VLATrainer(..., vla_train_dataloader=vla_train_dataloader, ...)`
   → `__init__` sets `self.vla_train_dataloader = vla_train_dataloader` (line 121). Still None.
4. `train_starvla.py:159-164` `prepare_training()` →
   `self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(accelerator, model, optimizer, self.vla_train_dataloader)`.
   - `setup_distributed_training` (`trainer_utils/trainer_tools.py:280-290`) just does
     `accelerator.prepare(*components)` and returns them in order.
   - **`accelerate.prepare(None)` returns `None`** (it passes non-(model/opt/dataloader/scheduler)
     objects through untouched; a `None` stays `None`, no exception). So None survives prepare.
5. `train_starvla.py:363` `train()` → `self._create_data_iterators()`
   → `train_starvla.py:342-344`:
   ```python
   def _create_data_iterators(self):
       self.vla_iter = iter(self.vla_train_dataloader)   # iter(None) → TypeError
   ```
   **This is the crash site.** It is the FIRST place that actually dereferences the dataloader
   as an iterable, which is why the crash is ~30s in (after model load + deepspeed init), not
   at construction.

**Conclusion:** there is exactly ONE way `vla_train_dataloader` becomes `None` in this run:
`build_dataloader` returns `None`. Nothing between build and `iter()` guards or replaces it.

### 4.2 `build_dataloader` — the implicit-None dispatcher (THE BUG)
`starVLA/dataloader/__init__.py:36-105` (H100). Structure:
```python
def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"):
    if   dataset_py == "lerobot_datasets":  ... return vla_train_dataloader   # 38-55
    elif dataset_py == "vlm_datasets":       ... return vlm_train_dataloader   # 56-60
    elif dataset_py == "pref_hdf5":          ... return vla_train_dataloader   # 61-74
    elif dataset_py == "pref_hdf5_vqa":      ... return vla_train_dataloader   # 75-89   ← LAST branch on H200
    elif dataset_py == "pref_hdf5_stageb":   ... return vla_train_dataloader   # 90-105  ← MISSING on H200
    # (no else, no trailing return) → falls off end → returns None
```
Key properties:
- **No `else` clause and no fallthrough `return`.** Any `dataset_py` not matched by a branch
  yields an implicit `return None`. There is no log line, no warning, no exception — the
  function silently returns None. (This is the design flaw that turns a missing branch into a
  cryptic downstream `iter(None)` instead of a clear "unknown dataset_py" error.)
- On **H100** the stageb branch (lines 90-105) instantiates
  `get_pref_stageb_dataset(...)`, wraps it in a `DataLoader(..., collate_fn=collate_fn,
  num_workers=...)`, saves dataset stats on rank 0, and returns the loader.
- On **H200** that branch does not exist (file ends at the `pref_hdf5_vqa` return, line 89).
  The yaml sets `dataset_py: pref_hdf5_stageb` (see `starvla_pref_stageb_main_hvlv_geom.yaml:46`),
  so on H200 NO branch matches → `None`.

### 4.3 Stage-A (`pref_hdf5_vqa`) vs Stage-B (`pref_hdf5_stageb`) divergence — the crux
The two code paths are **structurally identical** within `build_dataloader` (same DataLoader
wrap, same `collate_fn`, same stats save). The ONLY difference that matters for this bug:
- `pref_hdf5_vqa` branch **exists on both boxes** (H100 lines 75-89; H200 lines 75-89).
- `pref_hdf5_stageb` branch **exists only on H100** (lines 90-105); **absent on H200**.

So the divergence is **not** in dataset logic, batch size, drop_last, or sampler — it is purely
**presence/absence of the dispatch branch** between the two repo checkouts. That is why VQA
works and Stage-B returns None on the very same H200 environment.

### 4.4 Stage-B dataset 0-length branches (would ALSO yield an empty loader — but NOT this bug)
For completeness, document where the Stage-B dataset *could* legitimately produce a 0-length
index (a different failure mode that you'd see if the branch existed but data/cache were
wrong). File: `examples/preference/dataset/pref_hdf5_stageb_dataset.py`.
- Factory `get_pref_stageb_dataset` (lines 269-299): reads `task_groups`, `pref_keys`,
  `pseudo_label_cache`, `with_pref_suffix`, and passes `action_space=str(_g("action_space",
  "ee"))` (**line 290**). **Confirmed present on both H100 and H200** — this is the prior
  "action_space passthrough" fix (default "ee", OFT needs "joint"; the yaml sets
  `action_space: joint`, line 47). So the action_space fix is NOT regressed.
- `_filter_index_by_cache` (lines 158-181): drops every frame-sample whose
  `episode_key = f"{task_dir}/episode{ep_id}"` is **not in the cache** (`n_no_cache`) or whose
  cache `decision != "keep"` (`n_rejected`). **If the cache keys don't match any enumerated
  episode, `self._index` becomes `[]`** → `__len__() == 0`. A 0-length dataset makes a
  `DataLoader` *empty* (its `iter()` yields nothing / first `next()` raises `StopIteration`),
  **but the DataLoader object itself is NOT None** — so `iter()` would NOT raise
  `TypeError: NoneType`. Hence a 0-length index produces a *different* symptom and is **not**
  the current bug. (Ruled-out cause #3 already confirmed the keys DO match.)
- The cache whitelist (`_CACHE_WHITELIST`, line 56; `_load_cache`, lines 124-156) only reads
  `action_prompt_label`, `decision`, `pref_key` and raises on malformed entries — none of
  these can yield `None` for the loader; at worst they raise a clear ValueError (which would
  appear in the log — it does not).

> Bottom line for §4: the loader is None **solely** because `build_dataloader` has no
> `pref_hdf5_stageb` branch on H200 and silently returns None. The dataset's own 0-length
> paths are a real-but-separate failure mode (different symptom) and are not implicated here.

---

## 5. Hypotheses (ranked) with concrete tests

### H1 — (PRIMARY, effectively CONFIRMED) H200 dispatcher missing the stageb branch → implicit None
`build_dataloader` on H200 has no `elif dataset_py == "pref_hdf5_stageb"` branch and no
fallthrough return; called with `dataset_py="pref_hdf5_stageb"` it returns `None`.
**Evidence already collected:** H200 grep prints nothing for `pref_hdf5_stageb` in the file;
file is 89 lines vs H100 105; H200 log shows the "Creating VLA Dataset" print but ZERO
`[PrefHDF5StageBDataset/...]` prints (dataset never constructed). 
**Confirm-test (1 line, no training):**
```bash
ssh kevin@34.34.93.23 'grep -c "pref_hdf5_stageb" /home/kevin/starVLA/starVLA/dataloader/__init__.py'
# 0 ⇒ H1 confirmed. (H100 returns ≥1.)
```

### H2 — (FALSIFIED) accelerate/deepspeed version mismatch makes prepare() return None
Versions are byte-identical across boxes (§3.5): torch 2.6.0+cu124 / accelerate 1.5.2 /
deepspeed 0.16.9 / transformers 4.57.0. Also, `accelerate.prepare(None)` returns `None`
without error regardless of version, so even a mismatch wouldn't manifest as this. **Rejected.**

### H3 — (FALSIFIED) a conditional branch inside train_starvla.py returns None on H200's path
Walked the whole chain (§4.1): `prepare_data` → `build_dataloader` → `VLATrainer.__init__`
→ `setup_distributed_training` (`accelerator.prepare`) → `_create_data_iterators`. None of
these guard, rank-gate, or replace the loader with None; they pass it through verbatim. The
None is introduced upstream in `build_dataloader`, not in train_starvla.py. **Rejected.**

### H4 — (FALSIFIED for THIS crash) Stage-B dataset `__len__ == 0`
A 0-length dataset yields an *empty but non-None* DataLoader, so `iter()` would not raise
`TypeError: NoneType` (§4.4). It would instead surface as `StopIteration` on first `next()`,
or a div-by-zero in `_log_metrics` (`len(self.vla_train_dataloader)`), or no batches. Symptom
mismatch. Also ruled-out cause #3 already showed cache keys match the data dirs. **Not this
crash** (but worth a `len()` check after the branch is restored — see §6 step 2).

---

## 6. Recommended diagnostic sequence (fastest path)

1. **Confirm H1 (10 sec, no GPU):**
   ```bash
   ssh kevin@34.34.93.23 'grep -n "pref_hdf5_stageb" /home/kevin/starVLA/starVLA/dataloader/__init__.py; wc -l /home/kevin/starVLA/starVLA/dataloader/__init__.py'
   ```
   If grep is empty / wc is 89 → bug fully localized; skip to step 3.

2. **(Defensive) split "dataset empty" vs "prepare returns None"** — only needed if H1 were
   somehow NOT the cause. Instantiate the dataset directly on H200 and print its length:
   ```bash
   ssh kevin@34.34.93.23 'source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh && conda activate starVLA && cd /home/kevin/starVLA && python - <<PY
   from omegaconf import OmegaConf
   from examples.preference.dataset.pref_hdf5_stageb_dataset import get_pref_stageb_dataset
   cfg = OmegaConf.load("examples/preference/train_files/starvla_pref_stageb_main_hvlv_geom.yaml")
   ds = get_pref_stageb_dataset(data_cfg=cfg.datasets.vla_data, mode="train")
   print("len(dataset) =", len(ds))   # >0 ⇒ dataset fine, problem is the dispatcher (H1); ==0 ⇒ index/cache issue (H4)
   PY'
   ```
   (Make sure the yaml's data paths point at kevin/ first — re-run the §2 `sed` if needed.)

3. **The fix (operator decision — do not apply from this doc unless told):**
   Sync `starVLA/dataloader/__init__.py` from H100 to H200 (or `git pull` the branch on H200
   so it includes commit `173b0aa`). The minimal change is appending the 16-line
   `elif dataset_py == "pref_hdf5_stageb":` branch (H100 lines 90-105) to the H200 file.
   Optional hardening (recommended, separate change): add a final
   `else: raise ValueError(f"build_dataloader: unknown dataset_py={dataset_py!r}")`
   so a future missing branch fails loudly at build time instead of as `iter(None)` 30s later.

4. **Re-run the §2 repro** and confirm the H200 log now shows
   `[PrefHDF5StageBDataset/train] cache filter: ...` and `unique_eps=...` lines, then a normal
   training step. (This is the success signal that was absent in §3.2.)

---

## 7. Open questions / TODO for the diagnoser

- **(Resolved this session)** H200 versions: obtained — identical to H100 (§3.5). No TODO.
- **Why was the sync partial?** Task #17 ("Sync H100↔H200 code + copy contact baseline ckpt")
  is still pending. Worth checking whether *other* files on H200 are also stale (the dataset
  file and trainer were up to date, but `dataloader/__init__.py` was not). A quick
  `git status` / `git log -1` per file on H200, or a `diff -r` of the two repos for the
  `starVLA/dataloader/` and `examples/preference/` trees, would catch any other partial-sync
  gaps before the next H200 run.
- **Hardening:** decide whether to add the `else: raise` to `build_dataloader` (step 3
  optional) so this failure class is loud next time. Currently the function silently returns
  None for any unknown/unsynced `dataset_py`.

---

### Appendix A — key file:line index
| What | File:line |
|---|---|
| Crash site `iter(None)` | `starVLA/training/train_starvla.py:344` (`_create_data_iterators`) |
| `train()` calls it | `starVLA/training/train_starvla.py:363` |
| Dataloader stored (None) | `train_starvla.py:121` (`VLATrainer.__init__`) |
| `prepare_data` (no None-check) | `train_starvla.py:80-88` (build call at :83, print at :82) |
| `prepare_training` → prepare | `train_starvla.py:159-164` |
| micro-batch fix (present both boxes) | `train_starvla.py:145-157` |
| `setup_distributed_training` (=`accelerator.prepare`) | `starVLA/training/trainer_utils/trainer_tools.py:280-290` |
| **`build_dataloader` dispatcher (THE BUG)** | `starVLA/dataloader/__init__.py:36-105` (H100) |
| stageb branch (H100 only) | `dataloader/__init__.py:90-105` |
| vqa branch (both boxes) | `dataloader/__init__.py:75-89` |
| Stage-B dataset factory + `action_space` passthrough | `examples/preference/dataset/pref_hdf5_stageb_dataset.py:269-299` (action_space line 290) |
| cache filter (0-len index source) | `pref_hdf5_stageb_dataset.py:158-181` |
| cache whitelist loader | `pref_hdf5_stageb_dataset.py:56, 124-156` |
| Stage-B yaml (`dataset_py`, `action_space`) | `examples/preference/train_files/starvla_pref_stageb_main_hvlv_geom.yaml:46-47` |

### Appendix B — environment / paths
- **H200:** ssh `kevin@34.34.93.23`; repo `/home/kevin/starVLA`; conda
  `/mnt/localssd/kevin/miniconda3` (env `starVLA`); `results` symlink →
  `/mnt/localssd/kevin/starVLA_runs` (**no `/results/` subdir**, unlike H100); logs
  `/mnt/localssd/kevin/logs`; data root `/mnt/localssd/kevin/pref/data/0526`; VLM
  `/mnt/localssd/kevin/pref/Qwen3-VL-4B-Instruct`.
  - Non-interactive ssh needs explicit conda sourcing:
    `source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh && conda activate starVLA`
- **H100:** repo `/home/kaiwenh/starVLA`; `results` → `/mnt/localssd/kaiwenh/starVLA_runs/results`;
  data `/mnt/localssd/kaiwenh/pref/data/0526`; conda `/mnt/localssd/kaiwenh/miniconda3` (env `starVLA`).
- **Versions (both boxes, env `starVLA`):** torch 2.6.0+cu124, accelerate 1.5.2,
  deepspeed 0.16.9, transformers 4.57.0.
