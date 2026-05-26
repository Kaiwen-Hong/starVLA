# Pref-VLA — `place` category (5th cat, registration only — no training yet)

> **Self-contained data + registration doc.** Reading just this file should
> let you know exactly what was added on 2026-05-24, why the strip strategy
> is what it is, and what's still pending before training can start.
>
> Written 2026-05-24 after data download + paraphrase pre-flight + registry
> registration. **No Stage A run for `place` has been launched yet.**
>
> **Companion docs** (cited only for parts not duplicated here):
> - [`0522-a-giveobj.md`](0522-a-giveobj.md) — original Stage A spec.
>   §3-§6 (backbone / action head / loss / hyperparams) byte-identical
>   reused if/when `place` Stage A training is launched.
> - [`0523-height-hv-oreint-design-doc.md`](0523-height-hv-oreint-design-doc.md)
>   — the 3-category batch (`height`/`hvlv`/`orient`) where the
>   `PREF_CATEGORIES` / `VQA_CATEGORIES` registry shape was introduced.
>   `place` plugs into the same registry.
> - [`training-runbook.md`](training-runbook.md) — H100/H200 ops.

---

## 0. TL;DR

- **`place` = 5th preference category** alongside `contact`/`giveobj`,
  `height`, `hvlv`, `orient`. 16 taskA tasks (8 task_groups × 2 pref_keys)
  + 2 taskB tasks.
- **Pref axis = placement spatial** (`center` vs `corner` of target surface).
- **Strip strategy: template-only** (like `height`). Pre-flight scan showed
  pre-existing strip strategies maxed at 47.74% pass; templates are the only
  ≥99% clean option.
- **Pref labels**: `"center placement"` / `"corner placement"` (2-word,
  axis-anchored, matches the existing pattern).
- **VQA Q&A**: `"Question: center or corner placement? Answer:"` with single
  bare-token answers `center` (id 3057) / `corner` (id 73425).
- **Code changes**: 2 files (`prompt.py` adds `PREF_CATEGORIES["place"]` +
  3 constants; `vqa_sample.py` adds `VQA_CATEGORIES["place"]`). Both
  scp'd H100 → H200; H200 import verified.
- **Data**: 16 taskA dirs (3 are upstream-partial: see §7.1) + 2 taskB
  dirs (50 ep each). instructions/ mounted on both machines.
- **Pending**: stats precompute, YAML, launcher, smoke, training launch.
  Nothing scheduled — wait for explicit go-ahead.

---

## 1. Data on disk (verified 2026-05-24)

Symmetric on H100 / H200 (`<root> = /mnt/localssd/$USER/pref/data`):

```
<root>/place/
├── 16 taskA dirs (data/, video/, instructions/, scene_info.json, seed.txt)
└── taskB/
    ├── place_soap2_stand_center/    (50 ep)
    └── place_soap2_stand_corner/    (50 ep)
```

| | H100 | H200 |
|---|---|---|
| place/ on-disk size | 18 GB | 18 GB |
| SSD free | 638 G (89% used) | 6.0 T (41% used) |
| .zip backups kept | yes (per ops convention) | yes |
| log | `pref/data/logs/download_pod3_place_20260524_074213.log` | `..._20260524_074220.log` |

### 1.1 Task counts

| Sub | Task | Eps | Note |
|-----|------|----:|------|
| taskA | `move_mouse_pad_center` | 100 | |
| taskA | `move_mouse_pad_corner` | 100 | |
| taskA | `move_pillbottle_pad_center` | 100 | |
| taskA | `move_pillbottle_pad_corner` | 100 | |
| taskA | `move_playingcards_pad_center` | 100 | |
| taskA | `move_playingcards_pad_corner` | **11** | upstream partial (zip has only 11 ep) |
| taskA | `move_soap_pad_center` | 100 | |
| taskA | `move_soap_pad_corner` | 100 | |
| taskA | `place_mouse_tray_center` | 100 | |
| taskA | `place_mouse_tray_corner` | **59** | upstream partial |
| taskA | `place_pillbottle_tray_center` | 100 | |
| taskA | `place_pillbottle_tray_corner` | 100 | |
| taskA | `place_playingcards_tray_center` | 100 | |
| taskA | `place_playingcards_tray_corner` | 100 | |
| taskA | `place_soap_tray_center` | **2** | upstream partial (effectively unusable) |
| taskA | `place_soap_tray_corner` | 100 | |
| taskB | `place_soap2_stand_center` | 50 | |
| taskB | `place_soap2_stand_corner` | 50 | |

**Net taskA ep count**: 1372 / 1600 = **85.75%** (228 ep lost to upstream-broken zips, all of which are `_corner` half plus 2 mostly-empty `_center`s).

`copy_instructions.py sync_to_data` trimmed instructions to match data
indices on each partial task (e.g. `move_playingcards_pad_corner`:
instructions +0 -89 → 11; logged in script output).

### 1.2 HDF5 schema identical to other 4 cats

Verified: `endpose/{left,right}_endpose` (T, 7) xyz+quat,
`observation/{head,left,right}_camera/rgb` JPEG, etc. No special handling
needed — `pref_hdf5_dataset.py` reads this layout already.

### 1.3 Download recipe (this session)

```bash
source /mnt/localssd/$USER/miniconda3/etc/profile.d/conda.sh && conda activate starVLA
DATA=/mnt/localssd/$USER/pref/data
for inc in "place/taskA/*" "place/taskB/*"; do
  hf download kaiwen2/robotwin-prefvla-pod3 --repo-type dataset \
    --include "$inc" --local-dir "$DATA" --max-workers 8
done
python3 "$DATA/_unzip_pod3.py" "$DATA" place place/taskB
```

Same idempotent semantics as 0523's pod3 cats. Total ~80s wall (~75s
download, ~7s unzip).

### 1.4 Paraphrase mount (5090 → SSD → copy_instructions)

5090 box source: `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/`

All 16 taskA dirs + 2 taskB dirs found on 5090 (verified `ls` 2026-05-24).
26 candidates total in 5090; rsync-loop'd exactly the 18 matching place's
task names. Each rsync'd into
`/mnt/localssd/$USER/pref/data/instructions_src/pod3/<task>/<task>/instructions/episode{0..99}.json`
(kempner-nested layout per `copy_instructions.py:35-38`).

Then mounted:

```bash
python r-preference/copy_instructions.py \
    /mnt/localssd/$USER/pref/data/instructions_src/pod3 \
    /mnt/localssd/$USER/pref/data/place \
    --on-conflict overwrite

python r-preference/copy_instructions.py \
    /mnt/localssd/$USER/pref/data/instructions_src/pod3 \
    /mnt/localssd/$USER/pref/data/place/taskB \
    --on-conflict overwrite
```

`sync_to_data` auto-trimmed the 3 partial taskA tasks' instructions
to match their reduced data counts, and trimmed both taskB tasks
from 100 → 50.

**H200 ran without `r-preference/copy_instructions.py` in the repo** (its
`starVLA/` tree is at the older `ac27ba4` commit; see §6.3). The script
was scp'd to `/tmp/copy_instructions.py` on H200, executed, then deleted
— no git change on H200.

---

## 2. Pref semantics

`place` is **the 5th distinct preference axis** (alongside `contact`/
`giveobj`'s grasp region, `height`'s drop height, `hvlv`'s obstacle
detour width, `orient`'s grasp orientation).

| pref_key | Real-world meaning | Typical paraphrase phrasing |
|---|---|---|
| `center` | Place the object **at the center** of the target surface | "at the center", "in the middle", "at the middle" |
| `corner` | Place the object **at the nearest corner** of the target surface | "on the nearest corner", "at the closest corner" |

The model must distinguish per-pref **placement xy** (within roughly the
same z plane), so the discriminating signal sits in the late-trajectory
xy of the end-effector. Magnitude not yet measured — analogous to the
0522 §2.6 table — but expected to be visible in `L_x` / `L_y`.

---

## 3. Pre-flight paraphrase scan (137,200 phrases, 2026-05-24)

Per Kaiwen's "≥99% gate before any dataset code" rule
([feedback_pref_vla_proposal_first.md](../../<memory>)) and 0523 doc §3
methodology:

| Strategy | Pass | No-sep | Short | Leak |
|----------|-----:|------:|------:|-----:|
| comma + tight-leak | **23.09%** ❌ | 105,519 | 0 | 0 |
| comma + loose-leak | 23.09% ❌ | 105,519 | 0 | 0 |
| broad-sep + tight-leak | 47.74% ❌ | 71,700 | 0 | 0 |
| broad-sep + loose-leak | 47.74% ❌ | 71,700 | 0 | 0 |
| **template-only** | **100.00%** ✅ | — | — | — |

### 3.1 Why every strip strategy fails

Two paraphrase patterns observed (sampled `episode0/seen[0..4]`):

**Pattern A — `move_*_pad_*` (8 tasks)**: comma-separated
```
"Move the mouse to the pad, place it on the center."
"Bring the mouse to the pad and set it at the center."
```
Some `seen[]` entries have commas, others use `and`/`while`. Per-task pass
rate ~17-20% (most entries lack the broader separator too).

**Pattern B — `place_*_tray_*` (8 tasks)**: NO comma, pref word inline
```
"Place the pillbottle at the center of the tray."
"Put the pillbottle in the center of the tray."
```
Per-task pass rate ~26-30% (separator strictly absent in most entries —
the pref word is grammatically central to the only clause).

Net: broad-sep catches ~50% (the `move_*` half plus some `place_*` outliers),
which is far below the 99% gate. Template-only is the only viable strategy
— matches `height`'s reasoning verbatim (0523 doc §3.4 / `prompt.py:75-82`).

### 3.2 VQA single-token verification

`Qwen3-VL-4B-Instruct` tokenizer, `add_special_tokens=False`:

| Text | Token id(s) | Single bare? |
|------|-------------|--------------|
| `center` | `[3057]` | YES |
| `corner` | `[73425]` | YES |
| `middle` | `[19656]` | YES (kept as future option) |
| `edge`   | `[7186]`  | YES |

`center` / `corner` chosen — most semantically faithful to pref axis,
single bare tokens, clean output traces (no `n`/`s`-style splits like
hvlv had to work around).

---

## 4. Strategy (locked-in 2026-05-24)

Template-only with 8 pref-free templates, mirroring `height`'s approach:

- **`PLACE_PREF_LABELS`** = `{"center": "center placement", "corner": "corner placement"}`
- **`PLACE_TASK_GROUPS`** = 8 task_groups (matches 16 task_dirs ÷ 2 pref_keys)
- **`PLACE_CLEAN_TEMPLATE`** = 8 templates:
  - 4 × `move_<obj>_pad`: `"Move the <obj> onto the pad."` (textually
    identical to `HEIGHT_CLEAN_TEMPLATE` for the same 4 keys — no key
    collision since they live in disjoint `PrefCategory` dicts; the pref
    suffix is what discriminates)
  - 4 × `place_<obj>_tray`: `"Place the <obj> on the tray."` (unique to
    place; no other cat uses `_tray`)
- **`PrefCategory.sep_re`** = `None`, **`leak_re`** = `None` (template-only path)

### 4.1 Sample assembled prompts

```
move_mouse_pad        + center → "Move the mouse onto the pad. Preference: center placement"
move_mouse_pad        + corner → "Move the mouse onto the pad. Preference: corner placement"
place_mouse_tray      + center → "Place the mouse on the tray. Preference: center placement"
place_pillbottle_tray + corner → "Place the pill bottle on the tray. Preference: corner placement"
```

All 16 combinations sanity-tested via `build_action_prompt(...)` —
output clean, pref suffix correct, paraphrase argument ignored as
designed.

### 4.2 VQA config

```
question:         "Question: center or corner placement? Answer:"
pref_keys order:  ("center", "corner")       # slot A = center, slot B = corner
answer_token_ids: {"center": 3057, "corner": 73425}
answer_text:      {"center": "center", "corner": "corner"}
```

`QwenPI_VQA.py` already reads from `VQA_CATEGORIES[<cat>]` dynamically
(0523 doc §4.3), so no framework change needed.

---

## 5. Code changes (this session)

Two minimal additions, both **uncommitted** as of writing.

### 5.1 `examples/preference/dataset/prompt.py`

- **+38 lines** after `_LEAK_RE_ORIENT` block: new `PLACE_*` constants
  (PREF_LABELS, TASK_GROUPS, CLEAN_TEMPLATE).
- **+9 lines** inside `PREF_CATEGORIES` registry: `"place"` entry with
  `sep_re=None, leak_re=None`.
- **docstring** of `build_action_prompt` updated to mention `'place'`
  in the allowed-category list and the per-cat strategy table.

### 5.2 `examples/preference/dataset/vqa_sample.py`

- **+6 lines** inside `VQA_CATEGORIES` registry: `"place"` entry.

### 5.3 Unchanged

- `pref_hdf5_dataset.py`: already takes `category` kwarg + reads
  `PREF_CATEGORIES[cat].pref_keys` — works for `place` out of the box.
- `pref_hdf5_vqa_dataset.py`: same, via `vqa_sample.VQA_CATEGORIES`.
- `dataloader/__init__.py`: `pref_hdf5` + `pref_hdf5_vqa` branches
  reused as-is.
- `QwenPI_VQA.py`: reads `vqa_cfg.category` dynamically (0523 §4.3).

### 5.4 H200 sync

`scp` of the 2 modified files H100 → H200 (per user preference: every
edit goes via scp rather than git push/pull, since H100's `c618d0f`
commit was never pushed to `origin/opd` — H200 has the pref code as
untracked-but-identical local files anyway). Import + `build_action_prompt`
+ `VQA_CATEGORIES["place"]` verified on H200 immediately after scp.

---

## 6. Known issues + how each was handled

### 6.1 3 upstream-broken zip files (same flavor as 0523 §10.1)

The HF zips `move_playingcards_pad_corner.zip`,
`place_mouse_tray_corner.zip`, `place_soap_tray_center.zip` physically
contain only 11 / 59 / 2 episodes' worth of files respectively (verified
by entry count in zip vs 202 for full-100-ep zips). Re-download would
not help (verified: same partial sizes on both machines after independent
downloads).

`copy_instructions.py sync_to_data` correctly trimmed instructions to
match data indices. The 3 partials are usable for training but with
reduced ep counts; `place_soap_tray_center` at 2 ep is effectively
unusable and may need exclusion from per-task-group sampling logic.

### 6.2 Template duplication with `height` (intentional)

`PLACE_CLEAN_TEMPLATE` and `HEIGHT_CLEAN_TEMPLATE` both define
`"move_<obj>_pad" → "Move the <obj> onto the pad."` for the same 4 obj
keys. This is **by design** — the base prompt should be pref-agnostic;
the discrimination comes from the suffix (`"Preference: high drop"` vs
`"Preference: center placement"`). No dict-key collision (separate
`PrefCategory` instances). The model only sees one cat at a time in
Stage A (per-cat baselines + per-cat VQA cotrain), so the conflict is
moot during training.

### 6.3 H200 `starVLA/` tree behind H100 (no impact)

H200 is at commit `ac27ba4` (older). The pref code from H100's
`c618d0f` exists on H200 as untracked / locally-modified files that
mirror that commit's content. `git fetch + git status` on H200 shows
`origin/opd` is also at `ac27ba4` (H100 never pushed). `git pull` is
a no-op. Convention going forward: scp file-by-file for any code
change H100 → H200 (per user choice 2026-05-24). H200's untracked
state is well-defined and stable as long as nobody runs `git stash`
or `git restore` there.

---

## 7. Pending (NOT done yet — by design; awaiting explicit go)

> **STATUS UPDATE 2026-05-26**: Steps 1–6 below have since been COMPLETED
> (~2026-05-24 → 2026-05-25). Both baseline and VQA training reached 25k cleanly
> with loss converging healthily (baseline final L_action ≈ 0.014; VQA final
> ≈ 0.04). Gate eval baseline_CF=0.034 / VQA_CF=0.036 — both healthy. **But:**
>   - taskB VQA acc = **0.44** (below the 0.90 gate); pred distribution is
>     `{"center": 60, "corner": 40}` — biased but not random.
>   - `pseudo_label_offline.py` launch gate on place taskB **FAILED** in
>     `temp-525-h200.sh::01-pseudolabel-place` (post-filter acc = 0.194,
>     required ≥ 0.95) → Stage B place main is currently blocked.
>   - Stage B place B0 (no pseudo-label needed) is still runnable.
> Recommended diagnosis (per [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) §7):
> run `frame_window_test.py` with mid_8 / gripper_anchored to see if a
> different clip strategy rescues taskB acc, analogous to 0524 contact analysis
> where uniform_8 0.61 → mid_8 0.99.
>
> The numbered list below is preserved for historical reference of how the work
> was planned.

In dependency order, to take `place` from "registered" → "Stage A
baseline + main-method ready":

1. **Precompute stats**:
   ```bash
   python -m examples.preference.dataset.precompute_stats \
     --data_root /mnt/localssd/$USER/pref/data/place \
     --category place
   # produces examples/preference/dataset/stats_place_v1.json
   ```
   Note: 3 partial tasks will contribute fewer frames; expect total
   train frames ~165k instead of ~195k. The `precompute_stats.py`
   handles arbitrary ep counts (verified for orient's 95-ep partial).

2. **Decide ep-floor policy for `place_soap_tray_center` (2 ep)**:
   - Option A: leave in (will contribute ~1.6 train ep, statistically negligible)
   - Option B: exclude in `_split_episodes` via a min-ep filter
   - Recommend A unless training metrics show contamination

3. **Write 2 YAMLs** (clone from `starvla_pref_stage_a_{baseline,vqa}_height.yaml`):
   - `starvla_pref_stage_a_baseline_place.yaml`
   - `starvla_pref_stage_a_vqa_place.yaml`
   - Swap: `run_id`, `data_root_dir`, `stats_json_path`, `pref_category`,
     (VQA only) `framework.vqa.category`.

4. **Write 2 launchers** (clone from `launch_pref_stage_a_{baseline,vqa}_height.sh`):
   - Same env-var contract (`NUM_GPUS`, `MAX_STEPS`, `NO_SAVE`).
   - Host-aware (auto-detect kaiwenh vs kevin via `$USER`).

5. **Smoke** on both machines (1-2 train steps via `MAX_STEPS=2`).

6. **Launch** when GPUs available. Same `25k steps` budget as the
   other 3 new cats (0523 §5), `save_interval=5000` → 5 ckpts.

7. **Add `place` to `temp-h{100,200}.sh` sequential queues** (or
   make new wrappers) if running alongside other queued work.

**Nothing in §7 has been done.** No memory, doc, or code changes
beyond §5. Ask user before each step.

---

## 8. References

- **Companion docs**: see top header.
- **Code changed**:
  - `examples/preference/dataset/prompt.py` (+~50 lines)
  - `examples/preference/dataset/vqa_sample.py` (+6 lines)
- **Data**: `/mnt/localssd/{kaiwenh,kevin}/pref/data/place/{<16 task dirs>, taskB/{2 task dirs}}`
- **Logs**:
  - H100 download: `/mnt/localssd/kaiwenh/pref/data/logs/download_pod3_place_20260524_074213.log`
  - H200 download: `/mnt/localssd/kevin/pref/data/logs/download_pod3_place_20260524_074220.log`
- **HF source**: `kaiwen2/robotwin-prefvla-pod3`, paths `place/taskA/*` + `place/taskB/*`
- **5090 paraphrase root**: `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/`

---

End of doc.
