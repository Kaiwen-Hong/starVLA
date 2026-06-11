# 2026-05-28 — 0526 re-collected data + dual-instruction mount (rich pref + base)

> **Self-contained, cross-machine doc.** Reading just this file should let you
> reproduce — from cold — how the new `0526/` preference data gets its language
> instructions, on BOTH the collection box (5090) and the two training boxes
> (H100 / H200). Covers: what's on disk, how the two instruction trees are
> *generated* on the 5090, how they're *transported* to the training boxes, how
> they're *mounted* into the data tree, how to *verify*, and the *pending*
> training-side changes that consume them.
>
> **Pairs with** the collection-repo doc
> `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/doc/0528-instruction-updates.md`
> (the 5090 side of §3 below — it documents producing `data/preference_base/`
> and `description/task_instruction/`). This doc is the **training-side**
> counterpart + the cross-machine glue.
>
> **Companion docs (starVLA side):**
> - [`0522-a-giveobj.md`](0522-a-giveobj.md) — Stage A spec (model/loss/hyperparams).
> - [`training-runbook.md`](training-runbook.md) — H100/H200 ops, storage symlinks.
> - [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md) — why the *old* per-cat data was suspect.
> - [`0525-temp-stageb-analysis.md`](0525-temp-stageb-analysis.md) — the "model ignores the 2-word `Preference:` tag" finding that motivates the rich-inline choice (§7).

---

## 0. TL;DR

- **New data**: `/mnt/localssd/{kaiwenh,kevin}/pref/data/0526/` (H100 + H200).
  Re-collected; replaces the old per-cat trees (`pref/data/{height,hvlv,orient,place}/`)
  that had data problems. `contact` is NOT in 0526 (stays on old data).
- **4 cats**: `height` (16 taskA leaves), `orient` (16), `place` (16),
  `hvlv` (**12 — incomplete**, some hv/lv pairs not yet collected). Each cat
  also has `taskB/`. taskA = 100 ep/leaf, taskB = 50 ep/leaf. **67 leaves total.**
- **Every leaf now carries TWO instruction subdirs**:
  - `instructions/`       = **rich inline preference** (varied natural language, preference baked in).
  - `instructions_base/`  = **preference-stripped base** (same action, preference clause removed).
- **Sources** (collection repo, 5090, branch `exp`): `data/preference/` (rich)
  and `data/preference_base/` (base) — 154 leaves each; the 67 we need are a subset.
- **Mount tool**: [`../mount_instructions_0526.py`](../mount_instructions_0526.py) (idempotent).
- **Status**: mounted + verified on both machines 2026-05-28 (0 count mismatches).
  No training code changed yet — the existing loader still reads `instructions/`
  and works. Training changes to consume `instructions_base/` are **pending** (§8).

---

## 1. Final on-disk layout (what the mount produces)

```
/mnt/localssd/<user>/pref/data/0526/
├── height/
│   ├── move_mouse_pad_high/
│   │   ├── data/episode{0..99}.hdf5
│   │   ├── video/episode{0..99}.mp4
│   │   ├── scene_info.json, seed.txt
│   │   ├── instructions/episode{0..99}.json        ← rich inline preference   (NEW)
│   │   └── instructions_base/episode{0..99}.json   ← preference-stripped base (NEW)
│   ├── … 16 taskA leaves …
│   └── taskB/
│       ├── place_playingcards1_box_high/  (50 ep + both instruction subdirs)
│       └── place_playingcards1_box_low/
├── hvlv/   (12 taskA leaves, INCOMPLETE; taskB: stamp_seal6_hv)
├── orient/ (16 taskA leaves; taskB: move_can5_away_{0,90})
└── place/  (16 taskA leaves; taskB: place_soap2_stand_{center,corner})
```

Each `episode<N>.json` is `{"seen": [100], "unseen": [100]}`. Within a leaf all
episode files are identical (preference language is leaf-level, not
trajectory-level — see §7.3 for why this is valid on re-collected data).

### 1.1 Example (one leaf, both subdirs)

```
height/place_pillbottle_stand_high/instructions/episode0.json      seen[0]:
  "Place the pillbottle onto the display stand, drop off directly at a higher height."
height/place_pillbottle_stand_high/instructions_base/episode0.json seen[0]:
  "Place the pillbottle onto the display stand."
```

---

## 2. The 0526 data inventory (verified 2026-05-28)

| cat | taskA leaves | taskA ep | taskB leaves | taskB ep | notes |
|---|---|---|---|---|---|
| height | 16 | 100 | `place_playingcards1_box_{high,low}` | 50 | full |
| orient | 16 | 100 | `move_can5_away_{0,90}` | 50 | full; only `_0`/`_90` (no `_45`) |
| place  | 16 | 100 | `place_soap2_stand_{center,corner}` | 50 | full |
| hvlv   | **12** | 100 | `stamp_seal6_hv` | 50 | **INCOMPLETE** — see §2.1 |

**67 leaves total** (56 taskA + 11 taskB). All have `data/` + `video/`;
`instructions/` were empty before this work.

### 2.1 hvlv incomplete leaf set (as of 2026-05-28)

Present: `place_apple_plate_{hv,lv}`, `place_apple_right_hv`,
`place_cup_plate_{hv,lv}`, `place_cup_right_{hv,lv}`,
`place_hamburg_plate_{hv,lv}`, `place_hamburg_right_hv`,
`place_seal_plate_lv`, `place_seal_right_hv`. Several hv/lv partners
(`place_apple_right_lv`, `place_hamburg_right_lv`, `place_seal_plate_hv`,
`place_seal_right_lv`) are not yet collected. Mounting is per-leaf so this
doesn't block — re-run the mount script after more hvlv leaves land.

---

## 3. Collection side (5090) — how the two trees are generated

Repo: `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp`,
branch `exp`. Env: `conda activate RoboTwin`. These steps produce the SOURCE
trees that §4 transports. **You do not need to re-run them** if the trees
already exist (they did, 154 leaves each, on 2026-05-28) — this section is for
cold reproduction / adding leaves.

### 3.1 `data/preference/` — rich inline paraphrases (per-leaf 100 seen + 100 unseen)

Produced by the axis generators in `description/preference_generators/gen_*.py`
(tracked in git). Each writes `data/preference/<leaf>/<leaf>/instructions/episode<N>.json`.
Workflow + per-script inventory is documented in that repo's
`doc/0520-instruction-generation.md` §6. Rebuild everything:

```bash
cd /home/kaiwen/Desktop/research/ar-research_exp
for s in description/preference_generators/gen_*.py; do
    echo "=== $s ==="; python "$s" | tail -3
done
# gen_pad_other_objs.py / gen_box_high_low.py depend on pillbottle / stand
# outputs respectively — run those first if starting from scratch.
```

Each `gen_*.py` carries `{object}` + an axis placeholder (`{grasp}`/`{tilt}`/
`{marker}`/`{pos_short|pos_full}`); markers cycle in to yield 100 unique seen +
100 unique unseen per (family, pref) before object substitution.

### 3.2 `data/preference_base/` — preference-stripped parallel (the 0528 work)

Produced by **`description/preference_generators/build_preference_base.py`**
(preserved into the repo 2026-05-28 from `/tmp`; see 5090 doc
`0528-instruction-updates.md` §2). Strategy: per axis, import the matching
`gen_*.py`, strip the preference clause **at the template level** (cut at the
last comma before the placeholder, else the earliest connector word
`while|and|before|after|staying`), substitute `{object}`, drop any string that
still matches a per-axis `LEAK_FILTERS` regex, dedupe, cycle-pad to 100, write
`data/preference_base/<leaf>/<leaf>/instructions/episode<N>.json`. Run:

```bash
cd /home/kaiwen/Desktop/research/ar-research_exp
python3 description/preference_generators/build_preference_base.py    # idempotent
```

Notes baked into the build (consequences for training, §8):
- **placement** uses option (b): `{pos_full}` templates (location folded into
  the verb phrase, no clean base) are **dropped entirely** → base diversity per
  placement leaf is only 10–15 unique strings (vs 22–49 for other axes), then
  cycle-padded to 100.
- extra-object taskB leaves map via `EXTRA_OBJECT_ALIAS`
  (`boxdrink3→boxdrink`, `playingcards1→playingcards`, `soap2→soap`,
  `can5→can`, `seal6→seal`) — the templates reference the canonical object name.

### 3.3 (related, not used by the mount) `description/task_instruction/<task>.json`

`description/preference_generators/build_all_pref_task_instructions.py`
(also preserved from `/tmp` 2026-05-28) rewrites the kempner-style
template location for all 154 leaves (rich 100/100 + `full_description` +
`preference` label). This feeds a *different* eval pathway, not the
`data/preference{,_base}/` trees this doc mounts. Mentioned for completeness.

### 3.4 Build-script preservation (was a `/tmp` reproducibility hole)

`build_preference_base.py` and `build_all_pref_task_instructions.py` were
originally one-shot scripts under `/tmp` on the 5090 (volatile, lost on reboot).
On 2026-05-28 both were **copied into `description/preference_generators/`** (the
durable home the 0528 doc §5 recommends). They are currently **untracked in git**
on the 5090 (`git status` shows `??`) — commit them on branch `exp` to put them
in history. Until committed, the repo working tree + the generated trees on disk
+ the staged `.instr_src_0526/` copies (§4) are the durable artifacts.

---

## 4. Transport — 5090 → H100 → H200

The 5090 cannot reach the H200 directly (no route; see
`0523-height-hv-oreint-design-doc.md` §10.4). So: pull to H100, then push H100→H200.

```bash
# ---- on H100 (user kaiwenh) ----
STAGE=/mnt/localssd/kaiwenh/pref/data/.instr_src_0526
mkdir -p "$STAGE"
SRC=kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data
rsync -a "$SRC/preference/"      "$STAGE/preference/"        # ~250 MB, 154 leaves
rsync -a "$SRC/preference_base/" "$STAGE/preference_base/"   # ~165 MB, 154 leaves

# ---- H100 → H200 (user kevin) ----
H200=kevin@34.34.93.23
STAGE_R=/mnt/localssd/kevin/pref/data/.instr_src_0526
ssh "$H200" "mkdir -p $STAGE_R"
rsync -a "$STAGE/preference/"      "$H200:$STAGE_R/preference/"
rsync -a "$STAGE/preference_base/" "$H200:$STAGE_R/preference_base/"
scp r-preference/mount_instructions_0526.py \
    "$H200:/home/kevin/starVLA/r-preference/mount_instructions_0526.py"
```

The staged trees under `.instr_src_0526/` are the durable per-machine source
for re-mounts (e.g. after more hvlv leaves arrive — re-stage just those, then
re-run §5).

---

## 5. Mount — `mount_instructions_0526.py`

[`r-preference/mount_instructions_0526.py`](../mount_instructions_0526.py)
(stdlib-only; idempotent). For each leaf under `0526/<cat>/` (taskA flat) and
`0526/<cat>/taskB/` (taskB nested), it copies:

- `<pref-src>/<leaf>/<leaf>/instructions/` → `<leaf>/instructions/`
- `<base-src>/<leaf>/<leaf>/instructions/` → `<leaf>/instructions_base/`

then syncs each subdir's episode indices to the leaf's `data/` (trims taskB
100→50; duplicates from the lowest index to fill gaps). Re-running overwrites
both subdirs.

```bash
# Run on EACH machine (USER = kaiwenh on H100, kevin on H200).
# Use the conda starVLA python (plain `python` may be absent):
source /mnt/localssd/$USER/miniconda3/etc/profile.d/conda.sh && conda activate starVLA
python3 r-preference/mount_instructions_0526.py \
    --root     /mnt/localssd/$USER/pref/data/0526 \
    --pref-src /mnt/localssd/$USER/pref/data/.instr_src_0526/preference \
    --base-src /mnt/localssd/$USER/pref/data/.instr_src_0526/preference_base
# add --dry-run first to preview.
```

Expected tail: `summary: leaves=67  ok=67  missing-src=0`.

---

## 6. Verification

```bash
# (A) count audit — every leaf: instructions == instructions_base == data count
cd /mnt/localssd/$USER/pref/data/0526
bad=0
for cat in height hvlv orient place; do
  for d in $cat/*/ $cat/taskB/*/; do
    b=$(basename "$d"); [ "$b" = taskB ] && continue; [ -d "$d/data" ] || continue
    nd=$(ls "$d/data"|wc -l); ni=$(ls "$d/instructions" 2>/dev/null|wc -l)
    nb=$(ls "$d/instructions_base" 2>/dev/null|wc -l)
    [ "$ni" = "$nd" ] && [ "$nb" = "$nd" ] || { echo "MISMATCH $d $nd/$ni/$nb"; bad=$((bad+1)); }
  done
done; echo "mismatches: $bad"

# (B) content spot-check (rich vs base differ as expected)
python3 - <<'PY'
import json; from pathlib import Path
R=Path("/mnt/localssd/$USER/pref/data/0526")  # edit $USER
for leaf in ["height/place_pillbottle_stand_high","hvlv/place_cup_plate_hv"]:
    d=R/leaf
    print(leaf, "\n  pref:", json.load(open(d/"instructions/episode0.json"))["seen"][0],
               "\n  base:", json.load(open(d/"instructions_base/episode0.json"))["seen"][0])
PY

# (C) loader smoke (existing pipeline still reads instructions/ and builds prompt)
cd ~/starVLA && python3 - <<'PY'
from examples.preference.dataset.pref_hdf5_dataset import PrefHDF5Dataset
for cat in ("height","hvlv","orient","place"):
    ds=PrefHDF5Dataset(f"/mnt/localssd/$USER/pref/data/0526/{cat}",split="val",
                       include_state=True,category=cat)  # edit $USER
    print(cat, len(ds), repr(ds[0]["lang"]))
PY
```

2026-05-28 result on both machines: (A) 0 mismatches; (B) rich/base distinct;
(C) loads on all 4 cats.

---

## 7. Decisions & rationale

### 7.1 Why two subdirs (`instructions/` + `instructions_base/`), not one
Parallel subdirs keep the data tree single (no hdf5 duplication) while exposing
both prompt regimes per-leaf. The training loader picks the subdir by a flag (§8).

### 7.2 Why `instructions/` = rich inline (not "base + 2-word tag")
The legacy pipeline expressed preference ONLY via a fixed appended
`Preference: <2-word label>` tag. The Stage-B analysis
([`0525-temp-stageb-analysis.md`](0525-temp-stageb-analysis.md)) found the model
**ignored that tag** (it read preference from the image instead; pref-flip
sign-acc ≈ chance). Rich, varied natural-language preference is a candidate fix
and is what we now store in `instructions/`. The stripped base lives in
`instructions_base/` for the no-preference control.

### 7.3 Why mounting 5090 instructions onto re-collected 0526 data is valid
Instructions are **leaf-level** (object + surface + preference axis), identical
across the 100 episodes of a leaf, and independent of the specific trajectory.
The 0526 re-collection changed trajectories (and fixed data issues), not the
task semantics or leaf names — so the language is still correct.

---

## 8. Pending training-side changes (NOT done — design only)

The current loader (`examples/preference/dataset/pref_hdf5_dataset.py` +
`prompt.py`) reads only `instructions/` and then either uses a hardcoded
template (height/place → **zero base diversity**, 1 string/leaf) or runtime
regex-strips the paraphrase (hvlv/orient). Planned change:

1. **base prompt ← `instructions_base/`** (replaces template-only AND runtime
   regex strip). Gives 15–48 unique base strings/leaf and deletes the fragile
   `prompt.py` strip path + its duplicate leak-filter.
2. **rich-preference prompt ← `instructions/`** (varied language).
3. Add a `prompt_mode` flag for a clean 3-way: `rich` / `base+tag` / `base`.
4. **Caveat**: placement base diversity is low (`*_tray_corner` only 15 unique)
   because §3.2 dropped `{pos_full}` — revisit on the 5090 if place needs more.

This doc will be updated when these land.

---

## 9. Reproduce-from-cold checklist

1. **5090**: regenerate `data/preference/` (§3.1) + `data/preference_base/`
   (§3.2). (Or skip if trees exist.)
2. **H100**: `rsync` both trees → `.instr_src_0526/` (§4).
3. **H100 → H200**: `rsync` both trees + `scp` the mount script (§4).
4. **Each machine**: run `mount_instructions_0526.py` (§5).
5. **Each machine**: verify (§6).
6. Re-run 1–5 for any newly-collected hvlv leaves (mount is incremental/idempotent).

---

## 10. References

- 5090 collection-side doc (the other half of §3): `:/home/kaiwen/Desktop/research/ar-research_exp/doc/0528-instruction-updates.md`
- 5090 instruction-generation workflow: `…/doc/0520-instruction-generation.md`
- Mount script: [`../mount_instructions_0526.py`](../mount_instructions_0526.py)
- Old instruction-mount mechanism (single dir, interactive): [`../copy_instructions.py`](../copy_instructions.py)
- Stage A spec: [`0522-a-giveobj.md`](0522-a-giveobj.md) · Ops: [`training-runbook.md`](training-runbook.md)
- Why old data was suspect: [`0526-corrections-and-pipeline-audit.md`](0526-corrections-and-pipeline-audit.md)
- Tag-ignored finding (motivates §7.2): [`0525-temp-stageb-analysis.md`](0525-temp-stageb-analysis.md)

---

End of doc.
