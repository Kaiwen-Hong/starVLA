# Pref-VLA dataset extension — `pod3` categories (contact / height / hvlv / orient)

> Working snapshot 2026-05-23. Captures the **state of newly-downloaded preference
> datasets** and the decisions taken so a future session can pick up cold.
>
> **Scope of this doc**: data acquisition + relationship to legacy `giveobj` +
> per-category pref semantics + agreed prompt/training design. **Implementation
> (prompt.py / pref_hdf5_dataset.py / YAML / launch) NOT YET DONE** — §9 lists
> what's pending.
>
> Companion docs (unchanged, still authoritative for what they cover):
> - [`0522-a-giveobj.md`](0522-a-giveobj.md) — original Stage A spec for giveobj;
>   §3-§6 (model / loss / hyperparams) byte-identical reuse for the 3 new baselines.
> - [`training-runbook.md`](training-runbook.md) — H100/H200 ops, symlink storage, launch wrappers.
> - [`../9addinstructions.md`](../9addinstructions.md) — paraphrase rsync + `copy_instructions.py`
>   recipe (originally for giveobj; mechanism reused, filter rewritten).

---

## 0. TL;DR

- **New HF dataset**: `kaiwen2/robotwin-prefvla-pod3` (the old one was
  `kaiwen2/robotwin-prefvla-ours/giveobj`; that's now superseded — see §3).
- **5 categories on disk** (both H100 + H200, byte-parity verified): `giveobj`
  (legacy = contact/taskA pre-rename), `contact`, `height`, `hvlv`, `orient`.
  `place` deliberately **skipped** per user instruction.
- **Each category** has `taskA` (flat under `<cat>/`, **100 ep / task**) and
  `taskB` (kept in `<cat>/taskB/`, **50 ep / task** — half-size by design).
- **48 new taskA task-dirs** (16 per category × 3 categories: height/hvlv/orient).
  Their **paraphrases verified present** on 5090 box (48/48, 100 json/task,
  nested layout matches giveobj). **Not yet rsync'd / mounted.**
- **3 independent Stage A baselines** planned (one per of height/hvlv/orient);
  byte-identical spec lock from giveobj reused (§3-§6 of `0522-a-giveobj.md`).
- **Pref labels** (2-word, decided): height `"high drop"/"low drop"`, hvlv
  `"wide detour"/"narrow detour"`, orient `"horizontal grasp"/"vertical grasp"`.
- **Prompt format**: same as giveobj — `"<comma-cut base>. Preference: <label>"`.
  Comma-cut strip (no hand-written `CLEAN_TEMPLATE`); sample-level skip on
  leak/too-short. Base prompt is kept (not stripped to pref-only).
- **Pending implementation**: see §9.

---

## 1. Categories on disk (verified 2026-05-23)

H100 root: `/mnt/localssd/kaiwenh/pref/data/`
H200 root: `/mnt/localssd/kevin/pref/data/`

```
<root>/
├── giveobj/                                    # legacy. = contact/taskA in old HF naming
│   ├── give_{boxdrink,callbell,fork,screwdriver}_{25,75}/   (8 dirs)
│   ├── put_{boxdrink,callbell,fork,screwdriver}_dustbin_{25,75}/  (8 dirs)
│   │   each: data/episode{0..99}.hdf5 + video/ + instructions/ + scene_info.json
│   │   (1 partial: put_screwdriver_dustbin_75 = 15 ep; rest = 100 ep)
│   └── *.zip   (16 backup)
│
├── contact/                                    # NEW. 1 task only (the same partial)
│   ├── put_screwdriver_dustbin_75/  data=15 video=15   ← duplicate of giveobj's partial
│   ├── taskB/
│   │   ├── put_boxdrink3_plate_25/  data=50 video=50
│   │   └── put_boxdrink3_plate_75/  data=50 video=50
│   └── *.zip + taskB/*.zip
│
├── height/                                     # NEW. 16 tasks × 100 ep ✓
│   ├── move_{mouse,pillbottle,playingcards,soap}_pad_{high,low}/    (8 dirs)
│   ├── place_{mouse,pillbottle,playingcards,soap}_stand_{high,low}/ (8 dirs)
│   ├── taskB/
│   │   └── place_playingcards1_box_{high,low}/  data=50 video=50
│   └── *.zip + taskB/*.zip
│
├── hvlv/                                       # NEW. 16 tasks × 100 ep ✓
│   ├── place_{apple,cup,hamburg,seal}_{plate,right}_{hv,lv}/  (16 dirs)
│   ├── taskB/
│   │   └── stamp_seal6_{hv,lv}/  data=50 video=50
│   └── *.zip + taskB/*.zip
│
├── orient/                                     # NEW. 16 tasks × 100 ep ✓
│   ├── place_{bottle,callbell,can,chipstub}_{box,left}_{0,90}/  (16 dirs)
│   ├── taskB/
│   │   └── move_can5_away_{0,90}/  data=50 video=50
│   └── *.zip + taskB/*.zip
│
├── logs/                                       # download / unzip logs
└── _unzip_pod3.py                              # parallel-unzip script (8 workers, idempotent)
```

**Sanity script** (H100): `for d in <root>/{contact,height,hvlv,orient}/*/; do
[ -d $d/data ] && echo "$d: $(ls $d/data|wc -l)"; done`

**Disk after extension** (2026-05-23):
- H100: `/dev/md0` 5.9 T, **869 G free** (85% used, +52 G from pod3)
- H200: `/dev/md0` 11 T, **6.2 T free** (39% used)

---

## 2. HF source: `kaiwen2/robotwin-prefvla-pod3`

`huggingface-cli whoami` → `kaiwen2` (auth via `~/.netrc` on both machines).

Full repo tree (file count + zip size verified 2026-05-23):

| HF path | Files | Size | Local target |
|---|---|---|---|
| `contact/taskA/` | 1 | 80 MB | `<root>/contact/` (flat) |
| `contact/taskB/` | 2 | 646 MB | `<root>/contact/taskB/` |
| `height/taskA/` | 16 | 8.9 GB | `<root>/height/` (flat) |
| `height/taskB/` | 2 | 609 MB | `<root>/height/taskB/` |
| `hvlv/taskA/` | 16 | 11.0 GB (post-update) | `<root>/hvlv/` (flat) |
| `hvlv/taskB/` | 2 | 636 MB | `<root>/hvlv/taskB/` |
| `orient/taskA/` | 16 | 8.7 GB (post-update) | `<root>/orient/` (flat) |
| `orient/taskB/` | 2 | 472 MB | `<root>/orient/taskB/` |
| `place/*` | — | — | **NOT downloaded** (explicit user skip) |

**Per-machine total**: ~31 GB (taskA flat + taskB nested + .zip backups kept).

### Known HF CLI footgun

`huggingface-cli download ... --include 'A' --include 'B'` only honors the
**last** `--include` (verified 2026-05-23, exit 0 but A silently skipped). Always
issue **separate calls per include pattern**. All scripts in §7 split them.

### Upstream zip is half-size for taskB (consistently 50 ep, not 100)

Every taskB task is **50 episodes**, not 100. Doc `9addinstructions.md` §Task B
mentions "20 demos per task" as a *post-processing trim* (giveobj-era plan);
that hasn't happened for pod3, and the trim isn't needed for current Stage A
training (it was about Phase 1b unlabeled-target style use). Treat 50 ep / taskB
task as the upstream truth.

---

## 3. Relationship to legacy `giveobj`

User statement (verbatim, 2026-05-22): *"giveobj 其实本质上是 contact/taskA 我只是告诉你一下 hf 上现在没有"*.

| Item | Old (`robotwin-prefvla-ours/giveobj`) | New (`robotwin-prefvla-pod3`) |
|---|---|---|
| HF repo | `kaiwen2/robotwin-prefvla-ours` | `kaiwen2/robotwin-prefvla-pod3` |
| Naming | flat 24 zips under `giveobj/` | category-organized `<cat>/taskA/`, `<cat>/taskB/` |
| Coverage | 24 zips (`_25/_50/_75` for 8 task-groups) | only `_25/_75` per category; `_50` is upstream-dropped |
| `contact/taskA` on pod3 | — | only **1 zip** (`put_screwdriver_dustbin_75` @ 15 ep) — same partial that giveobj has |
| Existing trained baseline | `pref_baseline_stage_a_v1_noVQA` at step 35k (postmortem in `training-runbook.md` §8.1) | none yet for the 3 new categories |

**Practical implication**:
- Treat `giveobj/` as the de-facto contact baseline data. Don't re-do contact.
- `<root>/contact/` on disk is *redundant duplicate* of one giveobj task; could
  be deleted but harmless (~80 MB). Not auto-deleting — touch nothing under
  `/mnt/localssd/kaiwenh/{robomme,pref}/` without explicit user OK (memory:
  [feedback_cleanup_target_dreamzero.md]).

---

## 4. Per-category pref semantics (verified from paraphrases)

**Important**: pref key names are *not literal*. Inspected `episode0.json:seen[0..4]`
on 5090 box for one task of each `(category, pref)` to determine real semantics:

| Category | pref key | What it actually means | Typical phrasing in paraphrase |
|---|---|---|---|
| **height** | `high` | drop / release **from high** above target | "from high above", "from a high altitude", "tall position", "from up high" |
| **height** | `low` | drop / release **close to target surface** | "at a lower height", "from a low position", "close to the surface" |
| **hvlv** | `hv` | **stay far** from the `can` obstacle (wide detour) | "far from the can obstacle", "wide berth", "wide detour", "safe distance", "long way around" |
| **hvlv** | `lv` | **stay close** to the `can` obstacle (narrow detour) | "close to the can", "near the can", "right next to", "narrow gap" |
| **orient** | `0` | grasp **horizontally** (side-grip) | "horizontally", "from the side", "horizontal pose", "level", "flat" |
| **orient** | `90` | grasp **vertically** (top-down grip) | "from the top", "vertically", "from above", "straight down" |

⚠️ **`hvlv` ≠ velocity / volume**. It is **obstacle clearance / detour width**.
The task involves a `can` obstacle between source and target; `hv` (high-V,
~"high vacuity") = give wide berth, `lv` = pass close.

---

## 5. Episode counts (per-task)

Verified identical on H100 and H200.

### TaskA (4 × 16 = 64 task-dirs, all 100 ep except contact's 1)

| Category | Count of 100/100 tasks | Exceptions |
|---|---|---|
| `giveobj` (legacy) | 23 of 24 | `put_screwdriver_dustbin_75` = 15 ep (upstream) |
| `contact` | 0 of 1 | `put_screwdriver_dustbin_75` = 15 ep (same partial) |
| `height` | 16 of 16 ✓ | — |
| `hvlv` | 16 of 16 ✓ | — |
| `orient` | 16 of 16 ✓ | — |

### TaskB (4 × 2 = 8 task-dirs, all 50 ep)

| Category | TaskB tasks | Each ep count |
|---|---|---|
| `contact/taskB` | `put_boxdrink3_plate_{25,75}` | 50 |
| `height/taskB` | `place_playingcards1_box_{high,low}` | 50 |
| `hvlv/taskB` | `stamp_seal6_{hv,lv}` | 50 |
| `orient/taskB` | `move_can5_away_{0,90}` | 50 |

### Quirks

- `orient/place_chipstub_left_90` previously had `data=38 / video=39` (one orphan mp4)
  in the *partial* HF release; after re-download all 100/100 — quirk gone.

---

## 6. 5090 box: paraphrase tree (instructions/) — verified present

Source for all instructions:

```
kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/
```

Nested layout (kempner-style, matches what `copy_instructions.py:35-38` expects):
```
<task>/<task>/instructions/episode{0..99}.json   # each JSON: {"seen": [100], "unseen": [100]}
```

**Verified 48/48 task-dirs present** for the 3 new categories' taskA (each at
100 json):

- height (16): `move_{mouse,pillbottle,playingcards,soap}_pad_{high,low}`,
  `place_{mouse,pillbottle,playingcards,soap}_stand_{high,low}` ✓
- hvlv (16): `place_{apple,cup,hamburg,seal}_{plate,right}_{hv,lv}` ✓
- orient (16): `place_{bottle,callbell,can,chipstub}_{box,left}_{0,90}` ✓

(Plus 24 giveobj tasks — already rsync'd, see `9addinstructions.md`.)

5090 box also has many neighboring pref tiers (`_45`, `_center`, `_corner`,
`_50` etc.) we **don't** need; rsync filter must whitelist exactly the 48.

### Paraphrase pattern (used in §8 decision)

All 3 new categories use the **same `"<base clause>, <pref clause>."` two-part
structure** (verified across 8 spot-checks). Example:

```
"Move the mouse onto the pad, drop off the mouse from high above."
"Place the apple on the plate, while staying far from the can obstacle."
"Place the bottle into the box, while grasping the bottle horizontally."
```

→ First-comma cut yields a clean pref-free base every time we checked.
This is **simpler than giveobj's v5 hybrid regex** (which had to match
"grasping/holding ... top/bottom" inline), and analogous to what
`strip_preference_taskb.py` does for giveobj task B.

---

## 7. Download recipe (reproducible)

Self-contained, **runs identically on H100 (`kaiwenh`) and H200 (`kevin`)** —
substitute the user in paths. Idempotent: re-running re-fetches only changed
upstream files (HF CLI is content-addressed).

```bash
# 1. Setup
source /mnt/localssd/$USER/miniconda3/etc/profile.d/conda.sh && conda activate starVLA
DATA=/mnt/localssd/$USER/pref/data
mkdir -p $DATA/logs

# 2. Download every category EXCEPT place. ONE --include per call (HF CLI footgun).
for inc in \
    'contact/taskA/*'  'contact/taskB/*' \
    'height/taskA/*'   'height/taskB/*' \
    'hvlv/taskA/*'     'hvlv/taskB/*' \
    'orient/taskA/*'   'orient/taskB/*'; do
  huggingface-cli download kaiwen2/robotwin-prefvla-pod3 --repo-type dataset \
    --include "$inc" --local-dir $DATA --max-workers 8
done

# 3. Flatten taskA + parallel-unzip ALL zips (taskA at <cat>/, taskB at <cat>/taskB/).
#    Idempotent: skips any task already extracted (checks for data/ + video/).
python3 $DATA/_unzip_pod3.py $DATA \
    contact height \
    contact/taskB height/taskB hvlv/taskB orient/taskB
# NB: hvlv + orient taskA were downloaded + flattened + extracted in an earlier
# pass (see logs); pass them as args too if redoing from scratch:
#   python3 $DATA/_unzip_pod3.py $DATA contact height hvlv orient \
#       contact/taskB height/taskB hvlv/taskB orient/taskB

# 4. Sanity check
for cat in contact height hvlv orient \
           contact/taskB height/taskB hvlv/taskB orient/taskB; do
  for d in $DATA/$cat/*/; do
    [ -d "$d/data" ] || continue
    echo "  $cat/$(basename $d): data=$(ls $d/data | wc -l)"
  done
done
```

**Total download**: ~31 GB / machine. ~2-3 min on GCP fiber.
**Total unzip wall**: ~10-15 s (8 workers, SSD).

### `_unzip_pod3.py` semantics

- For each category arg:
  1. If `<cat>/taskA/` exists, move `<cat>/taskA/*.zip` → `<cat>/*.zip`, `rmdir taskA/`.
  2. Glob `<cat>/*.zip` and extract each to sibling dir `<cat>/<stem>/`.
- Skip a zip if `<cat>/<stem>/{data,video}/` already exists (idempotent).
- Zips kept as backup. Delete manually to reclaim disk:
  ```bash
  rm /mnt/localssd/$USER/pref/data/{contact,height,hvlv,orient}/*.zip
  rm /mnt/localssd/$USER/pref/data/{contact,height,hvlv,orient}/taskB/*.zip
  ```
  (~31 G reclaimed; data + video stays.)

---

## 8. Decisions taken (this session)

Locked design choices for the 3 new baselines. **Apply byte-identically to the
matched main-method (Stage A + VQA cotrain) when that's built.**

### 8.1 Three independent Stage A baselines

One per of `height` / `hvlv` / `orient`. Independent:
- run_id (proposed: `pref_baseline_stage_a_v1_noVQA_{height,hvlv,orient}`)
- norm stats JSON (`stats_{height,hvlv,orient}_v1.json`)
- train/val split (still seed 42, 80/20 per task)
- wandb run / ckpts

Reused **byte-identically** from giveobj (per `0522-a-giveobj.md` §3-§6):
backbone (Qwen3-VL-4B from base), action head (LayerwiseFM 36L), 20D EE+6D
action layout, chunk 16, flow-matching loss, optimizer, scheduler, batch 64,
50 k steps, image_size 224, cameras `[head, left, right]`.

`contact` does **not** get a new baseline — the existing giveobj baseline at
`pref_baseline_stage_a_v1_noVQA` covers it (since `contact == giveobj` semantically).

### 8.2 Prompt = `"<base>. Preference: <label>"` (base prompt KEPT)

Same shape as giveobj's `build_action_prompt`. Base prompt **not** dropped to
pref-only (`"Preference: <label>"` alone), reasons:
- Cross-category consistency with existing giveobj/contact baseline.
- byte-identical lock cascades to main-method + future cross-cat work.
- Phase 1b (target group B) typically drives via base prompt — keeping it
  now avoids retrofit later.
- Comma-cut strip is trivial (4 lines), so cost is ~0.

### 8.3 Pref labels (2-word, axis-anchored)

| Category | pref key | label string |
|---|---|---|
| `height` | `high` | `"high drop"` |
| `height` | `low` | `"low drop"` |
| `hvlv` | `hv` | `"wide detour"` |
| `hvlv` | `lv` | `"narrow detour"` |
| `orient` | `0` | `"horizontal grasp"` |
| `orient` | `90` | `"vertical grasp"` |

Single-word forms (`"high"`, `"far"`, `"horizontal"`) rejected — too ambiguous in
Qwen3-VL training distribution; axis word (`drop`/`detour`/`grasp`) anchors
semantic axis.

### 8.4 Strip strategy: comma-cut + sample-level skip (no CLEAN_TEMPLATE)

Per §6, all paraphrases follow `<base>, <pref>.` pattern.

```python
def comma_cut(phrase: str) -> str | None:
    idx = phrase.find(",")
    if idx < 0:
        return None
    base = phrase[:idx].rstrip()
    return base + "." if not base.endswith(".") else base
```

- After cut, check against per-category `_LEAK_RE` (regex of pref words):
  - height: `\b(high|low|above|below|height|altitude|tall|short|elevated|surface)\b`
  - hvlv: `\b(far|close|near|wide|narrow|detour|berth|arc|gap|distance|obstacle|around)\b`
  - orient: `\b(horizontal|vertical|sideways|level|flat|top|above|side|down)\b`
- If `base` < 15 chars OR `_LEAK_RE` hits → **return None**; dataset resamples
  another paraphrase from same episode's `seen[100]` (up to 5 tries; if all 5
  fail, log warn and use the original whole paraphrase as a known-degraded fallback —
  expected to never happen in clean data).
- **No** hand-written 24-entry `CLEAN_TEMPLATE` (the giveobj fallback path).

### 8.5 Code organization: one dataset + per-category config

Avoid forking `pref_hdf5_dataset.py` 3 times. Instead:

- Add a `PrefCategory` registry (dataclass holding `task_groups`, `pref_keys`,
  `pref_labels`, `leak_re`) keyed by category name (`"giveobj"`, `"height"`,
  `"hvlv"`, `"orient"`).
- `pref_hdf5_dataset.py` takes a `category: str` arg (from YAML
  `datasets.vla_data.pref_category`) and reads `PrefCategory[category]`.
- `PREF_KEYS = ("25","75")` (currently hardcoded at `pref_hdf5_dataset.py:28`)
  becomes `category-keyed`.
- 3 new YAMLs (height/hvlv/orient) differ from baseline only in `run_id`,
  `data_root_dir`, `stats_json_path`, `pref_category`.

---

## 9. Pending implementation (what's NOT done yet)

In rough dependency order:

1. **Pre-flight: comma-cut cleanliness scan** on all 4800 paraphrases per
   category (16 tasks × 100 ep × 3 sampled seen — actually 16 × 100 × 100 = 160 k
   per cat if exhaustive). Must show **≥ 99% pass** to confirm §8.4 strategy
   before code; if not, re-design strip.
2. **rsync paraphrase trees** from 5090 box → SSD:
   - Build new rsync filter listing the 48 task-dirs (sibling of giveobj's
     filter in `9addinstructions.md` Step 1).
   - Pull to `/mnt/localssd/$USER/pref/data/instructions_src/pod3/` (or
     `instructions_src/{height,hvlv,orient}/`). Both H100 and H200.
3. **Mount instructions** with `copy_instructions.py`:
   - One invocation per `(machine, category)` over its `<root>/<category>/`
     DATA_ROOT, with `--on-conflict overwrite` (first-time write).
   - `sync_to_data` auto-trims to actual episode count (esp. taskB at 50 ep —
     instructions/ will be deduped/duplicated to match).
4. **Extend `examples/preference/dataset/prompt.py`**:
   - Add `PREF_CATEGORIES` dict keyed by category name.
   - Replace `strip_v5` for new categories with `comma_cut` + per-category `_LEAK_RE`.
   - `build_action_prompt(category, task_group, pref_key, paraphrase)` instead
     of current `(task_group, pref_key, paraphrase)`.
5. **Extend `examples/preference/dataset/pref_hdf5_dataset.py`**:
   - Add `category` kwarg; derive `pref_keys` from registry.
   - Add resample loop in `__getitem__` for comma-cut None case.
6. **Precompute stats** per new category:
   ```bash
   python -m examples.preference.dataset.precompute_stats \
     --data_root /mnt/localssd/$USER/pref/data/height \
     --out examples/preference/dataset/stats_height_v1.json
   # repeat for hvlv, orient
   ```
7. **Write 3 YAMLs** under `examples/preference/train_files/`:
   `starvla_pref_stage_a_baseline_{height,hvlv,orient}.yaml` — clone of
   `starvla_pref_stage_a_baseline.yaml` with run_id / data_root / stats /
   pref_category swapped.
8. **Parameterize launch script** (`launch_pref_stage_a_baseline.sh`) with
   `CATEGORY` env var → picks YAML.
9. **Smoke tests** (`tests/test_smoke.py` extended per category).
10. **Training launch** (one at a time, monitor wandb; H100 baseline, H200 mirror).

---

## 10. References

- Spec lock (model / loss / hyperparams unchanged): [`0522-a-giveobj.md`](0522-a-giveobj.md)
- Ops: [`training-runbook.md`](training-runbook.md)
- Paraphrase mount mechanism: [`../9addinstructions.md`](../9addinstructions.md)
- Existing prompt code: `examples/preference/dataset/prompt.py`
- Existing dataset code: `examples/preference/dataset/pref_hdf5_dataset.py`
- Unzip script: `/mnt/localssd/$USER/pref/data/_unzip_pod3.py`
- Download logs (per machine):
  `/mnt/localssd/$USER/pref/data/logs/{download_pod3_hvlv_orient,download_pod3_hvlv,redo_pod3,dl_pod3_extras,unzip_pod3}.log`

---

End of working snapshot.
