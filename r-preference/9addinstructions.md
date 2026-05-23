# r-preference — adding instructions to `giveobj`

> Snapshot 2026-05-21. Follows [setup.md](setup.md) (which handles data download + unzip). HF dataset `kaiwen2/robotwin-prefvla-ours/giveobj` has no instructions tree, so we pull them from kaiwen's personal box and run [copy_instructions.py](copy_instructions.py) to mount them under each task.

## Task split within `giveobj`

The `giveobj` category covers two logically distinct task sets. **This doc currently only processes Task A** — Task B has no zips on HF yet, but its instructions are already mirrored on the 5090 box, so as soon as the zips land, the same flow works.

### Task A — `give_*` and `put_*_dustbin_*` (24 tasks, on HF, **processed**)

| Pick action (12) | Place action (12) |
|---|---|
| `give_boxdrink_25` | `put_boxdrink_dustbin_25` |
| `give_boxdrink_50` | `put_boxdrink_dustbin_50` |
| `give_boxdrink_75` | `put_boxdrink_dustbin_75` |
| `give_callbell_25` | `put_callbell_dustbin_25` |
| `give_callbell_50` | `put_callbell_dustbin_50` |
| `give_callbell_75` | `put_callbell_dustbin_75` |
| `give_fork_25` | `put_fork_dustbin_25` |
| `give_fork_50` | `put_fork_dustbin_50` |
| `give_fork_75` | `put_fork_dustbin_75` |
| `give_screwdriver_25` | `put_screwdriver_dustbin_25` |
| `give_screwdriver_50` | `put_screwdriver_dustbin_50` |
| `give_screwdriver_75` | `put_screwdriver_dustbin_75` |

### Task B — `put_*_plate_*` (12 tasks, **not yet on HF**)

```
put_boxdrink_plate_25     put_callbell_plate_25     put_fork_plate_25     put_screwdriver_plate_25
put_boxdrink_plate_50     put_callbell_plate_50     put_fork_plate_50     put_screwdriver_plate_50
put_boxdrink_plate_75     put_callbell_plate_75     put_fork_plate_75     put_screwdriver_plate_75
```

Verified absent: `HfApi().list_repo_files('kaiwen2/robotwin-prefvla-ours')` returns no `*_plate_*` entries under `giveobj/`. Verified present on 5090: all 12 instruction folders exist at `kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/put_<obj>_plate_<pct>/put_<obj>_plate_<pct>/instructions/`.

**Two Task B-specific differences from Task A** — see [§Task B — pending data](#task-b--pending-data):

1. **Prompts must not encode preference.** The raw 5090 sources do encode preference (e.g. _25 → "grasping on the bottom of the boxdrink"). We strip the post-comma clause and dedupe before mounting — Task B's policy should be preference-agnostic. Script: [strip_preference_taskb.py](strip_preference_taskb.py).
2. **20 demos per task, not 100.** We trim `data/` and `video/` to `episode{0..19}` after unzip; `copy_instructions.py`'s `sync_to_data` then mirrors instructions down to 20.

## Result (Task A only)

Every Task A folder under `/mnt/localssd/kaiwenh/pref/data/giveobj/<task>/` now has an `instructions/` sibling whose `episode<N>.json` set matches its `data/episode<N>.hdf5` set exactly:

```
giveobj/<task>/
  data/      episode{0..N-1}.hdf5
  video/     episode{0..N-1}.mp4
  instructions/  episode{0..N-1}.json      ← added by this step
  scene_info.json
  seed.txt
```

23 tasks at N=100, plus `put_screwdriver_dustbin_75` at N=15 (upstream data only has 15 episodes — see [setup.md](setup.md)). Total: 2,315 JSON files / ~38 MB extracted across 24 tasks.

## Source

The instructions tree lives on kaiwen's 5090 box and uses the kempner-style nested layout that `copy_instructions.py` expects:

```
kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/
  <task>/<task>/instructions/episode{0..99}.json
```

Each JSON contains `seen` (and likely `unseen`) arrays of natural-language instruction variants, e.g.:

```json
{
  "seen": [
    "Handover the boxdrink to the other side of the table, grasping on the bottom of the boxdrink.",
    "Pass the boxdrink to the other side of the table, holding near the bottom.",
    ...
  ]
}
```

## Step 1 — rsync the 24 Task A dirs to SSD

Pulls only the 24 Task A folders (skip everything else in `preference/`):

```bash
mkdir -p /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj

cat > /tmp/rsync_filter.txt <<'EOF'
# Task A — give_*
+ /give_boxdrink_25/***
+ /give_boxdrink_50/***
+ /give_boxdrink_75/***
+ /give_callbell_25/***
+ /give_callbell_50/***
+ /give_callbell_75/***
+ /give_fork_25/***
+ /give_fork_50/***
+ /give_fork_75/***
+ /give_screwdriver_25/***
+ /give_screwdriver_50/***
+ /give_screwdriver_75/***
# Task A — put_*_dustbin_*
+ /put_boxdrink_dustbin_25/***
+ /put_boxdrink_dustbin_50/***
+ /put_boxdrink_dustbin_75/***
+ /put_callbell_dustbin_25/***
+ /put_callbell_dustbin_50/***
+ /put_callbell_dustbin_75/***
+ /put_fork_dustbin_25/***
+ /put_fork_dustbin_50/***
+ /put_fork_dustbin_75/***
+ /put_screwdriver_dustbin_25/***
+ /put_screwdriver_dustbin_50/***
+ /put_screwdriver_dustbin_75/***
- *
EOF

rsync -avz --info=stats2,progress2 \
  --filter='merge /tmp/rsync_filter.txt' \
  kaiwen@100.97.239.33:/home/kaiwen/Desktop/research/ar-research_exp/data/preference/ \
  /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj/
```

Transfer: 2,400 JSON files / 38 MB / ~0.5 s.

## Step 2 — mount the instructions under each task

[`copy_instructions.py`](copy_instructions.py) (saved verbatim from the canonical version in `data/0520-collected/` workflow) walks each task folder under DATA_ROOT, finds the matching `<task>/<task>/instructions/` in INSTRUCTIONS_ROOT, copies it to `<task>/instructions/`, then calls `sync_to_data` to trim/duplicate so the json set matches `data/episode*.hdf5` exactly.

```bash
python3 r-preference/copy_instructions.py \
  /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj \
  /mnt/localssd/kaiwenh/pref/data/giveobj \
  --on-conflict overwrite
```

`--on-conflict overwrite` is non-interactive and clobbers any existing `instructions/` — safe on first run. Use `--dry-run` first when rerunning into a populated tree.

### What you'll see in the output

- `[copy] <task>: 100 files → …` — straight copytree from source.
- `[sync] put_screwdriver_dustbin_75: data has 15 episodes; instructions +0 −85 → 15` — sync_to_data noticed `data/` only has 15 hdf5 (indices 0..14), so it deleted the 85 instruction JSONs for indices 15..99. `+0` because every `data/` index was already covered by the source.

## Verify

```bash
cd /mnt/localssd/kaiwenh/pref/data/giveobj
for d in */; do
  echo "  $d: $(ls $d/instructions 2>/dev/null | wc -l) instr / $(ls $d/data 2>/dev/null | wc -l) data"
done
```

Expected: 23 Task A tasks at `100 / 100`, `put_screwdriver_dustbin_75/` at `15 / 15`.

## Task B — pending data

Once `kaiwen2/robotwin-prefvla-ours` adds the 12 `put_*_plate_*` zips under `giveobj/`, do these 5 steps in order. Steps **3a (strip preference)** and **3b (trim to 20)** are the two Task B-only modifications called out in [§Task split](#task-split-within-giveobj).

### 1. Download + unzip the new 12 zips

Re-run [setup.md](setup.md) Step 1. The existing `huggingface-cli download --include 'giveobj/*'` glob and `_unzip_giveobj.py` are both idempotent — only the 12 new zips transfer/unpack.

### 2. rsync the 12 Task B instruction dirs from the 5090

Append to the rsync filter above (between the `# Task A — put_*_dustbin_*` block and the final `- *`):

```
# Task B — put_*_plate_*
+ /put_boxdrink_plate_25/***
+ /put_boxdrink_plate_50/***
+ /put_boxdrink_plate_75/***
+ /put_callbell_plate_25/***
+ /put_callbell_plate_50/***
+ /put_callbell_plate_75/***
+ /put_fork_plate_25/***
+ /put_fork_plate_50/***
+ /put_fork_plate_75/***
+ /put_screwdriver_plate_25/***
+ /put_screwdriver_plate_50/***
+ /put_screwdriver_plate_75/***
```

Rerun the same rsync command. Incremental — only the 12 new dirs transfer.

### 3a. Strip preference from Task B prompts

Raw 5090 sources have e.g. `"Place the boxdrink onto the plate, grasping on the bottom of the boxdrink."` — every sentence is `"<action clause>, <preference clause>."` with exactly one comma (verified across 800 random samples). We cut at the comma, restore the period, and dedupe per list (the strip collapses many seen/unseen variants to a handful of unique base sentences).

```bash
# in-place rewrite under instructions_src/giveobj — only Task B's 12 dirs are touched
python3 r-preference/strip_preference_taskb.py \
  /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj
```

Output sanity-check per task looks like (smoke-tested on `put_boxdrink_plate_25/episode0.json` — 100 raw seen → 27 unique, 100 raw unseen → 30 unique):
```
[strip] put_boxdrink_plate_25: 100 json; seen 100 → ~25..30 after dedupe
  e.g. 'Place the boxdrink onto the plate, grasping on the bottom of the boxdrink.'
    → 'Place the boxdrink onto the plate.'
```

If you want to preserve the raw tree for comparison, use `--out <dir>` to mirror cleaned JSON into a separate path instead.

### 3b. Trim Task B data to 20 demos per task

Keep `episode{0..19}` only — delete the rest from `data/` and `video/`. `sync_to_data` (called by `copy_instructions.py` in Step 4) will then mirror instructions down to 20 automatically.

```bash
ROOT=/mnt/localssd/kaiwenh/pref/data/giveobj
for obj in boxdrink callbell fork screwdriver; do
  for pct in 25 50 75; do
    task=put_${obj}_plate_${pct}
    [ -d "$ROOT/$task" ] || { echo "skip $task (not present)"; continue; }
    for i in $(seq 20 99); do
      rm -f "$ROOT/$task/data/episode${i}.hdf5" "$ROOT/$task/video/episode${i}.mp4"
    done
    echo "  $task: data=$(ls $ROOT/$task/data | wc -l)  video=$(ls $ROOT/$task/video | wc -l)"
  done
done
```

`scene_info.json` and `seed.txt` keep their full 100-episode metadata — they're not consumed per-episode by the data loader, and the extra keys are harmless.

### 4. Mount Task B instructions

Same `copy_instructions.py` invocation as Task A, but with `--on-conflict skip` so the 24 already-mounted Task A `instructions/` aren't disturbed:

```bash
python3 r-preference/copy_instructions.py \
  /mnt/localssd/kaiwenh/pref/data/instructions_src/giveobj \
  /mnt/localssd/kaiwenh/pref/data/giveobj \
  --on-conflict skip
```

For Task B you'll see lines like:
```
[copy] put_boxdrink_plate_25: 100 files → .../instructions
  [sync] put_boxdrink_plate_25: data has 20 episodes; instructions +0 −80 → 20
```

### 5. Verify Task B

Expected: 12 Task B tasks at `20 / 20` (instructions / data), preference-free prompts.

```bash
ROOT=/mnt/localssd/kaiwenh/pref/data/giveobj
for obj in boxdrink callbell fork screwdriver; do
  for pct in 25 50 75; do
    task=put_${obj}_plate_${pct}
    instr=$(ls $ROOT/$task/instructions 2>/dev/null | wc -l)
    data=$(ls $ROOT/$task/data 2>/dev/null | wc -l)
    echo "  $task: $instr instr / $data data"
  done
done

# spot-check that prompts are preference-free
python3 -c "
import json
d = json.load(open('/mnt/localssd/kaiwenh/pref/data/giveobj/put_boxdrink_plate_25/instructions/episode0.json'))
for s in d['seen'][:5]: print(' ', s)
"
```

## Rerunning

Both steps are idempotent:
- `rsync` is incremental by default — no diff, no transfer.
- `copy_instructions.py --on-conflict overwrite` wipes and re-copies; `--on-conflict skip` keeps the existing tree.

If kaiwen pushes more episodes upstream and re-zips an existing Task A task, redo [setup.md](setup.md) Step 1, then rerun this doc's Step 2 — the rsync of instructions only needs to happen again when instructions themselves change on the 5090 box.
