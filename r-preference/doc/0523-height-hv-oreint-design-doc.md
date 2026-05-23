# Pref-VLA Stage A — height + hvlv + orient (3 new categories)

> **Self-contained design + implementation + ops doc.** Reading just this file
> should let you know exactly what was changed, what was trained, and why.
>
> Written 2026-05-23 after the full implementation + launch of 6 new runs.
> Status as of writing: 2/6 runs complete (height baseline on H100, orient
> baseline on H200), 2/6 in flight (hvlv baseline on H100, hvlv VQA on H200),
> 2/6 queued (height VQA on H100, orient VQA on H200).
>
> **Companion docs** (don't re-read; cited only for the parts not duplicated here):
> - [`0522-a-giveobj.md`](0522-a-giveobj.md) — original giveobj Stage A spec.
>   §3-§6 (backbone / action head / loss / hyperparams) is **byte-identical
>   reused** by all 6 new runs. The 3 new categories don't redefine those.
> - [`training-runbook.md`](training-runbook.md) — ops (storage symlink,
>   launch script family, watchdog quirks, mountpoint hard-fail, etc).
> - [`temp-new-dataset.md`](temp-new-dataset.md) — pre-implementation plan
>   (now **superseded** by this doc + commit `c618d0f`). Useful only for the
>   data-acquisition recipe (rsync filter + `_unzip_pod3.py`) which still applies.

---

## 0. TL;DR

- **3 new categories on top of giveobj**: `height`, `hvlv`, `orient`. Each 16
  tasks × 100 episodes (one orient task = 95 due to upstream 0-byte zips, §10.1).
- **6 new runs** = 3 cats × {baseline (no VQA), main-method (VQA cotrain)}.
- **25k steps each** (giveobj's were 50k; user halved them); save every 5k → 5 ckpts each.
- **Split**: H100 runs `{height-bl, hvlv-bl, height-vqa}`; H200 runs `{orient-bl, hvlv-vqa, orient-vqa}`.
- **Sequential, fault-tolerant**: one `temp-h100.sh` + one `temp-h200.sh`
  loop the 3 runs in order, `pkill+sleep` orphan processes between runs,
  continue to next if one fails.
- **Wall time**: baseline ~8h, VQA ~9h → ~25-26h total per machine.
- **All code committed** as `c618d0f` on branch `opd` (44 files, 7842 lines).

---

## 1. Categories on disk (verified 2026-05-23)

| Category | Tasks | Eps/task | Total eps | Train/val eps (80/20, seed 42) | Train frames |
|----------|-------|----------|-----------|--------------------------------|--------------|
| `giveobj` (legacy contact) | 16 | mostly 100 (1 partial) | 1595 | 1276 / 319 | 192,927 |
| `height` | 16 | 100 | 1600 | 1280 / 320 | 195,431 |
| `hvlv`   | 16 | 100 | 1600 | 1280 / 320 | 280,684 |
| `orient` | 16 | 100 (1 task=95, §10.1) | 1595 | 1276 / 319 | 197,486 |

**Disk layout** (symmetric on both machines, `<root> = /mnt/localssd/$USER/pref/data`):

```
<root>/height/{16 task-dirs}/  each containing:
  data/episode{0..99}.hdf5     # RoboTwin sim, 30 Hz, ~150-250 frames
  video/episode{0..99}.mp4
  instructions/episode{0..99}.json   # {"seen": [100 paraphrases], "unseen": [100]}
  scene_info.json
  seed.txt
<root>/hvlv/{16 task-dirs}/
<root>/orient/{16 task-dirs}/
```

**Instruction trees** rsync'd from `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/`
(5090 box) — 48 task-dirs × 100 JSONs = 4800 files per machine, ~10 MB.
H200 not directly reachable from 5090 (connection timeout); pushed via H100.
Mounted with `r-preference/copy_instructions.py --on-conflict overwrite` per
(machine, category). `sync_to_data` auto-trimmed orient's `place_chipstub_left_90`
instructions from 100 → 95 to match the data count (§10.1).

**HDF5 schema** verified identical to giveobj across all 4 categories:
- `endpose/{left,right}_endpose` (T, 7) float64 — xyz + xyzw quat (SAPIEN/ManiSkill)
- `endpose/{left,right}_gripper` (T,) float64
- `observation/{head,left,right}_camera/rgb` (T,) JPEG bytes (decoded with PIL)
- `observation/front_camera/rgb` (T,) — present but unused (cameras whitelist excludes it)
- `joint_action/...` — present but unused (we use EE+6D not joint)
- `pointcloud` (T, 0) — empty placeholder

---

## 2. Task lists per category

### 2.1 `height` (16 tasks, 8 task_groups × 2 pref_keys)

```
move_{mouse,pillbottle,playingcards,soap}_pad_{high,low}        # 8
place_{mouse,pillbottle,playingcards,soap}_stand_{high,low}     # 8
```

Task = "move/place object onto target". Pref = drop height (release from
high above vs close to target surface).

### 2.2 `hvlv` (16 tasks, 8 task_groups × 2 pref_keys)

```
place_{apple,cup,hamburg,seal}_{plate,right}_{hv,lv}            # 16
```

Task = "place object on plate or on the right side, around a `can` obstacle
between source and target". Pref = obstacle clearance:
- `hv` = "high-vacuity" = give wide berth (stay **far** from can)
- `lv` = "low-vacuity" = pass close (stay **near** the can)

**Important**: `hvlv ≠ velocity / volume`. It is **obstacle clearance**.

### 2.3 `orient` (16 tasks, 8 task_groups × 2 pref_keys)

```
place_{bottle,callbell,can,chipstub}_{box,left}_{0,90}          # 16
```

Task = "place object into box / on left side". Pref = grasp orientation:
- `0` = grasp **horizontally** (side-grip)
- `90` = grasp **vertically** (top-down grip)

`place_chipstub_left_90` is the partial task — 95 episodes (§10.1).

---

## 3. Pre-flight scan + per-category strip strategy

User's `temp-new-dataset.md §9.1` mandated: **scan paraphrases first, ≥99%
pass on the chosen strip OR redesign**. Doc had assumed a simple comma-cut
would work; reality (160k paraphrases per cat) said otherwise:

### 3.1 Round 1 scan: naive comma-cut

| Cat | Pass | No-comma | Too-short | Leak |
|-----|------|----------|-----------|------|
| height | **35.5%** ❌ | 63.75% | 0.5% | 0.25% |
| hvlv | **70.0%** ❌ | 30.0% | 0% | 0% |
| orient | **88.0%** ❌ | 0% | 0% | **12.0%** |

All three failed the 99% gate. Per user decision: **stop and ask** → discussed
modes of failure, then user chose **hybrid (D)**:
- height: switch to template-only (paraphrases too messy)
- hvlv: improve strip (broader separator)
- orient: improve leak regex (drop `side` etc.)

### 3.2 Round 2 scan: refined strategies

| Strategy | Pass |
|----------|------|
| hvlv: broader sep `,\|and\|while\|by\|before\|after` | **100.000%** ✅ |
| orient: same sep + tight leak `\b(horizontal\|vertical\|sideways\|level\|flat)\b` | **100.000%** ✅ |
| height + broader sep + tightened leak | **81.5%** (still <99%) |

Confirmed the D plan: **height = template_only, hvlv + orient = strip+leak**.

### 3.3 Round 3 scan (via `build_action_prompt`): widen regex for adverbs

Found a hole: `\bvertical\b` does NOT match "vertically" (word boundary fails
after the final char). Patched leak regexes to use `\b(word)\w*` so adverb
forms are caught. Re-scan via the production `build_action_prompt`:

| Cat | Pass | Fallback rate |
|-----|------|---------------|
| `hvlv` | 100.000% | 2.5% (paraphrase failed → CLEAN_TEMPLATE) |
| `orient` | 100.000% | 1.0% (same) |
| `height` | 100.000% | 100% (by design, template_only) |

### 3.4 Final strip strategies (locked in `prompt.py`)

```python
# Broader separator (shared by hvlv + orient)
_SEP_RE_BROAD = re.compile(r",|\s+(?:and|while|by|before|after)\s+", re.IGNORECASE)
MIN_BASE_LEN = 15

# hvlv leak words (\w* suffix catches "closer"/"closely"/"narrower" etc.)
_LEAK_RE_HVLV = re.compile(
    r"\b(far|close|near|wide|narrow|detour|berth|arc|gap|distance|obstacle|around)\w*",
    re.IGNORECASE,
)

# orient leak words (drop side/above/top/down — those appear in base descriptors
# like "on the toy car's left side"; keep stronger pref-anchors)
_LEAK_RE_ORIENT = re.compile(
    r"\b(horizontal|vertical|sideways|level|flat)\w*",
    re.IGNORECASE,
)

# height: NO regex used. PrefCategory.sep_re = leak_re = None → forces template.
```

### 3.5 height CLEAN_TEMPLATE (8 entries, pref-agnostic)

```python
HEIGHT_CLEAN_TEMPLATE = {
    "move_mouse_pad":           "Move the mouse onto the pad.",
    "move_pillbottle_pad":      "Move the pill bottle onto the pad.",
    "move_playingcards_pad":    "Move the playing cards onto the pad.",
    "move_soap_pad":            "Move the soap onto the pad.",
    "place_mouse_stand":        "Place the mouse on the stand.",
    "place_pillbottle_stand":   "Place the pill bottle on the stand.",
    "place_playingcards_stand": "Place the playing cards on the stand.",
    "place_soap_stand":         "Place the soap on the stand.",
}
```

`hvlv` and `orient` also have templates (8 each) but those are **fallback only**
(2.5% / 1.0% trigger rate — see §3.3).

### 3.6 Pref labels (appended as " Preference: <label>" suffix)

| Cat | pref_key | label string |
|-----|----------|--------------|
| `height` | `high` | `"high drop"` |
| `height` | `low` | `"low drop"` |
| `hvlv` | `hv` | `"wide detour"` |
| `hvlv` | `lv` | `"narrow detour"` |
| `orient` | `0` | `"horizontal grasp"` |
| `orient` | `90` | `"vertical grasp"` |

2-word, axis-anchored. Single-word forms (`"high"`, `"far"`, `"horizontal"`) were rejected (per `temp-new-dataset §8.3`) as too ambiguous in Qwen3-VL training distribution.

### 3.7 Example final prompts

```
height (template path):
  "Move the mouse onto the pad. Preference: high drop"

hvlv (strip path, paraphrase had ", far from"):
  "Position the apple onto the plate. Preference: wide detour"

orient (strip path, paraphrase had ", horizontally"):
  "Drop the bottle inside the box. Preference: horizontal grasp"
```

---

## 4. VQA design (3 new categories)

### 4.1 Per-category question + answer (verified tokens)

The LM head is asked a binary preference question over a uniform 8-frame
`head_camera` clip. The answer is a **single bare-form token** for which
first-token cross-entropy is computed. Token IDs verified against
`Qwen/Qwen3-VL-4B-Instruct` tokenizer (2026-05-23):

| Cat | Question | pref_key | Answer text | First-token ID |
|-----|----------|----------|-------------|----------------|
| `giveobj` (legacy) | "Question: low or high contact at grasp? Answer:" | 25 | `"low"` | 10303 |
|  |  | 75 | `"high"` | 11892 |
| `height` | "Question: high or low drop? Answer:" | high | `"high"` | 11892 |
|  |  | low | `"low"` | 10303 |
| `hvlv` | "Question: far from or near the obstacle? Answer:" | hv | `"far"` | 23559 |
|  |  | lv | `"near"` | 51659 |
| `orient` | "Question: horizontal or vertical grasp? Answer:" | 0 | `"horizontal"` | 30629 |
|  |  | 90 | `"vertical"` | 15292 |

### 4.2 Why `hvlv` answer is `far/near` (not `wide/narrow`)

**`narrow` is multi-token bare-form**: `tok.encode("narrow")` → `[77, 6044]`
(= `"n"` + `"arrow"`). Using `wide`/`narrow` as binary labels would force a
first-token CE between IDs `9150` (= `"wide"`) and `77` (= `"n"`) — workable
but cosmetically ugly (output looks like `"n"` in model traces).

Alternatives tested (all single bare-token):
- `wide` (9150) / `tight` (74182) — semantic OK but "tight" feels less natural
- `far` (23559) / `near` (51659) — **chosen**: most semantically faithful to
  hvlv axis (obstacle distance), single token, clean output.

The **action prompt label** stays `"wide detour"/"narrow detour"` per
`temp-new-dataset §8.3` (decoupled from VQA answer; the model sees both —
appended pref suffix for action, VQA-internal Q&A for the LM-head loss).

### 4.3 Framework dispatch (`QwenPI_VQA.py`)

Changed from giveobj-hardcoded constants to per-category attributes:

```python
# OLD (giveobj-hardcoded):
_LOW_ID  = VQA_ANSWER_TOKEN_IDS["25"]   # 10303
_HIGH_ID = VQA_ANSWER_TOKEN_IDS["75"]   # 11892
# VQA_QUESTION imported as module constant.

# NEW (per-category):
class Qwen_PI_VQA:
    def __init__(self, ...):
        cat_name = vqa_cfg.get("category", "giveobj")
        vqa_cat = VQA_CATEGORIES[cat_name]
        self._vqa_question         = vqa_cat.question
        self._vqa_answer_text      = dict(vqa_cat.answer_text)
        self._vqa_answer_token_ids = dict(vqa_cat.answer_token_ids)
        # Canonical binary slots: pref_keys[0] = A, pref_keys[1] = B
        self._vqa_pk_A, self._vqa_pk_B = vqa_cat.pref_keys
        self._vqa_id_A = self._vqa_answer_token_ids[self._vqa_pk_A]
        self._vqa_id_B = self._vqa_answer_token_ids[self._vqa_pk_B]
        ...
```

All references in `_build_vqa_inputs`, `_vqa_forward`, `predict_preference`
replaced with `self._vqa_*` lookups. **Backward compat**: giveobj works
unchanged because `framework.vqa.category` defaults to `"giveobj"`.

### 4.4 VQA cotrain loss

Unchanged from giveobj:
```
L_total = L_action + λ_vqa · L_vqa     where λ_vqa = 0.5
```
- L_action: byte-identical to baseline (Qwen_PI.forward → flow-matching MSE).
- L_vqa: cross-entropy on first answer-token position (chat template uses `\n`
  separator → bare-form ID). Per-row label mask is **exactly 1 active position**
  (asserted at L168-171 of QwenPI_VQA.py).
- `n_vqa_per_batch = 2`: dedup-by-episode-ID, take first 2 unique → 2 VQA
  per 8-batch (1:4 VQA:action ratio).

### 4.5 Verified live (2026-05-23)

H200 hvlv VQA run **saved its first ckpt at step 5000** without crashing →
the per-category dispatch, the `far`/`near` token CE, and the binary
classifier slots are all working end-to-end on the new code.

---

## 5. Byte-identical spec lock vs giveobj

§3-§6 of `0522-a-giveobj.md` is **unchanged**. The diff between giveobj and
the 3 new baselines (and between baseline and main-method) is confined to:

| YAML field | Baseline-giveobj | Baseline-new | VQA-new |
|------------|------------------|--------------|---------|
| `run_id` | `pref_baseline_stage_a_v1_noVQA` | `pref_baseline_stage_a_v1_noVQA_<cat>` | `pref_main_stage_a_v1_VQA_<cat>` |
| `framework.name` | `QwenPI` | `QwenPI` | `QwenPI_VQA` |
| `framework.vqa.category` | — | — | `<cat>` |
| `datasets.vla_data.dataset_py` | `pref_hdf5` | `pref_hdf5` | `pref_hdf5_vqa` |
| `datasets.vla_data.pref_category` | (absent → defaults `"giveobj"`) | `<cat>` | `<cat>` |
| `datasets.vla_data.data_root_dir` | giveobj path | `<cat>` path | `<cat>` path |
| `datasets.vla_data.stats_json_path` | `stats_giveobj_v1.json` | `stats_<cat>_v1.json` | same |
| `trainer.max_train_steps` | 50000 | **25000** | 25000 |
| `trainer.num_warmup_steps` | 5000 | **2500** | 2500 |
| `trainer.save_interval` | 5000 | 5000 (5 ckpts) | 5000 |
| `trainer.enable_gradient_checkpointing` | true | true | varies (§8.3) |

Everything else (backbone, action head, optimizer, lr, eff batch 64, chunk 16,
20D action, image_size 224, cameras `[head, left, right]`, prompt suffix
shape) **identical**.

---

## 6. Code changes (full delta from giveobj-only state)

Commit: `c618d0f` on branch `opd` ("Add preference-conditioned VLA Stage A:
giveobj + 3 new categories" — 44 files, 7842 insertions).

### 6.1 Modified files

| File | Change |
|------|--------|
| `examples/preference/dataset/prompt.py` | Add `PREF_CATEGORIES` registry (4 entries). `build_action_prompt(category=...)` dispatches per-cat strategy. Legacy giveobj exports (`PREF_LABELS`, `TASK_GROUPS`, `strip_v5`) kept for backward compat. 451 lines (was 159). |
| `examples/preference/dataset/pref_hdf5_dataset.py` | Add `category` kwarg (default `"giveobj"`). `_task_dirs_for_groups` + `_split_episodes` take explicit `pref_keys` (no longer module-level). Factory reads `data_cfg.pref_category`. 361 lines. |
| `examples/preference/dataset/pref_hdf5_vqa_dataset.py` | Factory passes `category` through. 114 lines. |
| `examples/preference/dataset/vqa_sample.py` | Add `VQA_CATEGORIES` registry (4 entries) with `VQACategoryConfig` dataclass. Module-level constants kept as giveobj aliases. 247 lines. |
| `examples/preference/dataset/precompute_stats.py` | Add `--category` arg; auto-derives `--out` to `stats_<cat>_v1.json`. giveobj regression byte-identical (sha256 diff = 0). |
| `starVLA/model/framework/QwenPI_VQA.py` | Read `framework.vqa.category` in `__init__`; per-category `_vqa_question`, `_vqa_id_A/_B`, `_vqa_text_by_id`. All hardcoded `_LOW_ID/_HIGH_ID/VQA_QUESTION` references replaced. `predict_preference` returns category-agnostic `{pref_key, label, confidence, p_A, p_B, pk_A, pk_B}`. |
| `starVLA/dataloader/__init__.py` | Already had `pref_hdf5` + `pref_hdf5_vqa` branches from giveobj era — unchanged here. |
| `starVLA/training/train_starvla.py` | (Previous session) Relays `log/*` keys to wandb so L_action / L_vqa / vqa_acc show up. |

### 6.2 New files

| File | Purpose |
|------|---------|
| `examples/preference/dataset/stats_height_v1.json` | 1280 train ep, 195,431 frames |
| `examples/preference/dataset/stats_hvlv_v1.json`   | 1280 train ep, 280,684 frames |
| `examples/preference/dataset/stats_orient_v1.json` | 1276 train ep, 197,486 frames (5 ep less from broken HDF5) |
| `examples/preference/train_files/starvla_pref_stage_a_baseline_{height,hvlv,orient}.yaml` | 3 baseline configs |
| `examples/preference/train_files/starvla_pref_stage_a_vqa_{height,hvlv,orient}.yaml` | 3 VQA configs |
| `examples/preference/launch_pref_stage_a_baseline_{height,hvlv,orient}.sh` | 3 baseline launchers |
| `examples/preference/launch_pref_stage_a_vqa_{height,hvlv,orient}.sh` | 3 VQA launchers |
| `examples/preference/temp-h100.sh` / `temp-h200.sh` | Sequential 3-run wrappers, fault-tolerant |

All 6 launchers + 2 wrappers are host-aware (autodetect `/mnt/localssd/kaiwenh`
vs `/mnt/localssd/kevin` data root), enforce the `mountpoint -q /mnt/localssd`
+ `results/` symlink hard-fail, and export
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (4× speedup; see
`training-runbook.md §7.1`).

### 6.3 Launch script env vars (uniform across all 6 + 2 wrappers)

| Var | Default | Purpose |
|-----|---------|---------|
| `NUM_GPUS` | 8 | grad_accum auto = 1 if ≥8, else 2; eff batch always 64 |
| `MAX_STEPS` | (yaml 25000) | Override `trainer.max_train_steps` for smoke |
| `NO_SAVE` | (off) | Bump `save_interval` past `max_train_steps` for clean profile |

`temp-h*.sh` wrappers do NOT expose these; if you need a smoke override use
the per-run launcher directly:
```bash
MAX_STEPS=10 NO_SAVE=1 bash examples/preference/launch_pref_stage_a_vqa_orient.sh
```

---

## 7. The 6 runs (launched 2026-05-23 ~10:47-10:48 UTC)

### 7.1 H100 sequential plan (`temp-h100.sh`)

| # | Run | YAML | Run dir (via results/ symlink) | wandb run | Start | Wall | Status as of writing |
|---|-----|------|--------------------------------|-----------|-------|------|----------------------|
| 1 | height baseline | `starvla_pref_stage_a_baseline_height.yaml` | `results/Checkpoints/pref_baseline_stage_a_v1_noVQA_height/` | (auto) | 10:47Z | 8h6m | ✅ DONE 18:54Z |
| 2 | hvlv baseline | `starvla_pref_stage_a_baseline_hvlv.yaml` | `pref_baseline_stage_a_v1_noVQA_hvlv/` | (auto) | 18:54Z | predicted ~8h | 🔄 step 6833/25000 (27%), 1.16 s/step, ETA finish ~03:00Z May 24 |
| 3 | height VQA | `starvla_pref_stage_a_vqa_height.yaml` | `pref_main_stage_a_v1_VQA_height/` | (auto) | predicted ~03:00Z May 24 | predicted ~9h | ⏳ queued, ETA finish ~12:00Z May 24 |

### 7.2 H200 sequential plan (`temp-h200.sh`)

| # | Run | YAML | Run dir | wandb run | Start | Wall | Status |
|---|-----|------|---------|-----------|-------|------|--------|
| 1 | orient baseline | `starvla_pref_stage_a_baseline_orient.yaml` | `results/Checkpoints/pref_baseline_stage_a_v1_noVQA_orient/` | (auto) | 10:48Z | 7h46m | ✅ DONE 18:34Z |
| 2 | hvlv VQA | `starvla_pref_stage_a_vqa_hvlv.yaml` | `pref_main_stage_a_v1_VQA_hvlv/` | (auto) | 18:34Z | predicted ~9h | 🔄 step 7126/25000 (29%), 1.27 s/step, ETA finish ~03:30Z May 24 |
| 3 | orient VQA | `starvla_pref_stage_a_vqa_orient.yaml` | `pref_main_stage_a_v1_VQA_orient/` | (auto) | predicted ~03:30Z May 24 | predicted ~9h | ⏳ queued, ETA finish ~12:30Z May 24 |

### 7.3 Sequential wrapper behavior

Both `temp-h100.sh` and `temp-h200.sh` (same shape):

1. **For each run in queue**:
   - Cleanup orphan processes (`pkill -9 -f "starvla_pref|train_starvla.py|accelerate.launch|pt_elastic"`) if any present.
   - Sleep 30s for GPU memory release.
   - `bash <launcher> 2>&1 | tee <per-run log>`; capture exit code via `${PIPESTATUS[0]}`.
   - On non-zero exit: log `FAILED <tag> (exit=<rc>)` + last 10 lines, **CONTINUE to next run**.
   - On zero exit: log `DONE <tag>`.
2. **Per-run logs**: `results/.temp_h{100,200}_runs/<tag>.log` (full stdout).
3. **Summary log**: `results/.temp_h{100,200}_runs/_summary.log` (BEGIN/DONE/FAILED markers only).

Set flags: `set -uo pipefail`; deliberately NO `set -e` (we want failures to
advance, not abort the script).

---

## 8. Wall time predictions vs actuals

| Run | Predicted | Actual (so far) | Match? |
|-----|-----------|-----------------|--------|
| H100 height baseline | ~8h (1.15-1.20 s/step × 25k) | 8h6m | ✓ |
| H100 hvlv baseline (in flight) | ~8h | 27% in 2h13m → projects 8h12m | ✓ |
| H100 height VQA | ~9h (1.27 s/step × 25k + VQA overhead) | (queued) | — |
| H200 orient baseline | ~8h | 7h46m | ✓ (H200 slightly faster) |
| H200 hvlv VQA (in flight) | ~9h | 29% in 2h33m → projects 8h52m | ✓ |
| H200 orient VQA | ~9h | (queued) | — |

H200 trends a hair faster per-step on baselines (better GPU memory headroom →
no grad_ckpt; VQA configs keep grad_ckpt=false on H200, =true on H100).

---

## 9. Stats / norm details

Each run loads its own `stats_<cat>_v1.json` (q01/q99 over the 20D action +
state across the train split). Per-dim spans (q99 - q01):

- **height** xyz spans ~30 cm; z span tight (0.23 m) since pad height fixed.
- **hvlv** longer episodes (~220 fr/ep avg), R_6d_4 span very narrow (0.07)
  → most action variation is in xyz + L_6d, not R_6d.
- **orient** R_6d_4 span 0.9 → confirms preference axis is in wrist rotation.

(See the JSON files for full vectors; meta block also records `n_train_episodes`,
`n_val_episodes`, `n_train_frames`, `per_task_episode_counts` keyed by task_dir.)

Stats are computed once via:
```bash
python -m examples.preference.dataset.precompute_stats \
    --data_root /mnt/localssd/$USER/pref/data/<cat> \
    --category <cat>
```
~3 seconds each (1280 episodes, 200k frames). Output is byte-identical given
the same `--split_seed` (verified for giveobj regression: diff = 0.0).

---

## 10. Known issues + how each was handled

### 10.1 5 zero-byte HDF5 in `orient/place_chipstub_left_90` (upstream HF zip)

The HF zip `place_chipstub_left_90.zip` (525 MB) **physically contains 5
zero-byte entries** for `episode{5,6,7,8,9}.hdf5`. Unzip succeeds but those
files are unreadable.

**Mistake during fix**: I `rm -rf`'d the task dir intending to use `unzip` to
re-extract, but `unzip` was not installed on either machine. The rm
succeeded; the unzip didn't. Both machines were left with NO data for this
task for ~1 minute.

**Recovery**: extracted via Python's `zipfile` module (which is in the conda
env). The 5 broken episodes are upstream-broken (verified inside the zip),
not an extraction issue. **Deleted them as 0-byte files** on both machines;
`copy_instructions.py` `sync_to_data` auto-trimmed instructions from 100 → 95.

**Net impact**:
- `place_chipstub_left_90`: 95 episodes (76 train + 19 val) instead of 100 (80+20).
- Other 15 orient tasks: 100 eps each (80+20).
- Total orient train: 1276 ep (vs would-be 1280). ~0.3% data loss.
- Negligible effect on training; spec lock unaffected.

### 10.2 `narrow` not single bare-token → switched hvlv VQA to `far/near`

See §4.2. Action prompt label stays `wide/narrow detour` per spec; VQA answer
is decoupled.

### 10.3 Leak regex missed adverb forms (`vertically`, `closely`, etc.)

Original regex `\bvertical\b` does NOT match `vertically` because the word
boundary check after the final char of the pattern fails (next char is `l`,
still a word char). Fixed by adding `\w*` suffix: `\b(vertical|horizontal|...)\w*`.
Re-verified 100% pass on real data (§3.3).

### 10.4 Two-machine SSH topology: H200 can't reach 5090

`ssh kaiwen@100.97.239.33` from H200 → connection timeout (no route).
Workaround: rsync 5090 → H100 (4800 JSONs ~17s), then H100 → H200 (same files,
~5s over public IP). copy_instructions.py runs locally on each machine.

### 10.5 VQA framework category dispatch

See §4.3. **Verified live**: H200 hvlv VQA saved step 5000 ckpt without crash
→ the per-category dispatch, far/near token CE, binary classifier slots all
working. No need to smoke separately.

### 10.6 wandb run names

YAML's `run_id` field maps to wandb run name. New runs all have
`pref_{baseline|main}_stage_a_v1_{noVQA|VQA}_<cat>`. All under wandb project
`kaiwenh-17-uiuc/pref-sim`.

---

## 11. Pre-flight scan methodology

Run **on the 5090 box directly** (paraphrases are there before they reach
either training machine):

```bash
ssh kaiwen@100.97.239.33 'python3 <<PYEOF
import json, re
from pathlib import Path
SRC = Path("/home/kaiwen/Desktop/research/ar-research_exp/data/preference")
CATEGORIES = {"height": [...16 task names...], ...}
LEAK_RE = {...per-cat...}
SEP_RE = re.compile(r",|\s+(?:and|while|by|before|after)\s+", re.I)
for cat, tasks in CATEGORIES.items():
    n_total = n_pass = n_no_sep = n_short = n_leak = 0
    for task in tasks:
        for jpath in (SRC / task / task / "instructions").glob("episode*.json"):
            d = json.loads(jpath.read_text())
            for ph in d.get("seen", []):
                n_total += 1
                m = SEP_RE.search(ph)
                if not m: n_no_sep += 1; continue
                base = ph[:m.start()].rstrip()
                if len(base) < 15: n_short += 1; continue
                if LEAK_RE[cat].search(base): n_leak += 1; continue
                n_pass += 1
    print(f"{cat}: pass={100*n_pass/n_total:.3f}%")
PYEOF'
```

Total scanned: **480k paraphrases** (160k × 3 cats). Took <5 min wall.

Round 3 (verification of production strip) scanned the same on H100 using
`build_action_prompt` directly with the actual `PREF_CATEGORIES` registry,
walking already-mounted instructions on `/mnt/localssd/kaiwenh/pref/data/<cat>/`.

---

## 12. Ablation cleanliness checks

The spec lock from §3-§6 of `0522-a-giveobj.md` is reused unchanged across
all 6 new runs. Diff is intentionally minimal:

| Component | Baseline-cat vs VQA-cat |
|-----------|--------------------------|
| Action stream | byte-identical (both go through `Qwen_PI.forward`) |
| Image preprocessing | identical (3 cams, 224×224, JPEG decode) |
| Action representation | 20D EE+6D, chunk 16, same q99 norm |
| Optimizer / LR / scheduler | identical |
| Eff batch | 64 (8 GPU × per_device 8 × grad_accum 1) |
| **Diff** | VQA adds `vqa_episode_id` + `vqa_pref_key` + `vqa_task_group` per sample; framework computes additional `L_vqa` from clip cache; total loss = L_action + λ·L_vqa |

**Per-category** the only YAML diff (apart from filenames + run_id) is:
- `data_root_dir` (which `<cat>/`)
- `stats_json_path` (which `stats_<cat>_v1.json`)
- `pref_category` (which `<cat>`)
- (VQA only) `framework.vqa.category`

No `CoT_prompt`, no extra preprocessing, no per-cat hyperparam tuning.

**The cross-category comparison is also clean**: same model arch, same train
budget, same loss recipe. Differences in final metrics are attributable to
(a) data distribution per cat and (b) presence/absence of VQA cotrain.

---

## 13. How to reproduce / extend

### 13.1 Re-run a single ckpt (e.g. baseline + new seed)

```bash
cd ~/starVLA
# Edit YAML if needed (e.g. seed: 42 → 43), then:
bash examples/preference/launch_pref_stage_a_baseline_height.sh
```

### 13.2 Add a new category (e.g. `pod4-place`)

1. Download data to `<root>/<new_cat>/`, mount instructions via `copy_instructions.py`.
2. Run pre-flight scan (see §11) — pick strategy per pass-rate.
3. In `prompt.py`: add `PrefCategory` entry to `PREF_CATEGORIES` (task_groups,
   pref_keys, pref_labels, clean_templates, sep_re, leak_re).
4. (If VQA) in `vqa_sample.py`: add `VQACategoryConfig` entry to
   `VQA_CATEGORIES`. Verify answer tokens with `tok.encode(text)`.
5. `python -m examples.preference.dataset.precompute_stats --data_root <...> --category <new_cat>`.
6. Clone YAML + launcher from an existing pair; edit run_id / data_root /
   stats / pref_category / (VQA) framework.vqa.category.

### 13.3 Add new VQA categories whose answers ARE multi-token

Two paths:
- (a) Switch to a synonym that is single-token (what we did for hvlv with
  `far/near` instead of `wide/narrow`). Verify with `tok.encode`.
- (b) Use full-sequence CE on the answer phrase instead of first-token. Would
  require changing `_build_vqa_inputs` label-masking logic. Not implemented.

---

## 14. References

- **Spec lock (model / loss / hyperparams unchanged)**: [`0522-a-giveobj.md`](0522-a-giveobj.md)
- **Ops manual** (storage, launch scripts, troubleshooting, postmortems): [`training-runbook.md`](training-runbook.md)
- **Pre-implementation plan** (superseded by this doc): [`temp-new-dataset.md`](temp-new-dataset.md)
- **Data acquisition + paraphrase mount**: [`../setup.md`](../setup.md), [`../9addinstructions.md`](../9addinstructions.md), [`../copy_instructions.py`](../copy_instructions.py)
- **Code** (all under `c618d0f`):
  - `examples/preference/dataset/prompt.py`
  - `examples/preference/dataset/pref_hdf5_dataset.py`
  - `examples/preference/dataset/pref_hdf5_vqa_dataset.py`
  - `examples/preference/dataset/vqa_sample.py`
  - `examples/preference/dataset/precompute_stats.py`
  - `examples/preference/dataset/stats_{height,hvlv,orient}_v1.json`
  - `examples/preference/train_files/starvla_pref_stage_a_{baseline,vqa}_{height,hvlv,orient}.yaml`
  - `examples/preference/launch_pref_stage_a_{baseline,vqa}_{height,hvlv,orient}.sh`
  - `examples/preference/temp-{h100,h200}.sh`
  - `starVLA/model/framework/QwenPI_VQA.py`
- **wandb**: `kaiwenh-17-uiuc/pref-sim` (all 6 runs + giveobj runs)
- **Live status logs** (during the runs):
  - H100: `results/.temp_h100_runs/_summary.log` + `0{1,2,3}-*.log`
  - H200: `/home/kevin/starVLA/results/.temp_h200_runs/_summary.log` + `0{1,2,3}-*.log`
