# r-preference — setup

> Snapshot 2026-05-21. Covers data acquisition only.

## `giveobj` dataset

24 zips, ~10.5 GB on Hugging Face / ~11 GB on disk. Source: dataset repo `kaiwen2/robotwin-prefvla-ours`, subfolder `giveobj/`.

### Layout on SSD

```
/mnt/localssd/kaiwenh/pref/data/giveobj/
  give_{boxdrink,callbell,fork,screwdriver}_{25,50,75}.zip          # 12 files
  put_{boxdrink,callbell,fork,screwdriver}_dustbin_{25,50,75}.zip   # 12 files
```

Each zip expands to `<task_name>_<pct>/data/episode{0..99}.hdf5` + `<task_name>_<pct>/video/episode{0..99}.mp4` (200 files + dir entries per zip — 100 episodes each). `_25 / _50 / _75` are the three preference tiers.

Download log: `/mnt/localssd/kaiwenh/pref/data/logs/download_giveobj.log`.

### Download command (reproduce)

Run from any host with `HF_HOME` already on the SSD and HF auth already done (`hf auth whoami` → `kaiwen2`):

```bash
mkdir -p /mnt/localssd/kaiwenh/pref/data/logs
huggingface-cli download \
  kaiwen2/robotwin-prefvla-ours \
  --repo-type dataset \
  --include 'giveobj/*' \
  --local-dir /mnt/localssd/kaiwenh/pref/data \
  --max-workers 8 \
  > /mnt/localssd/kaiwenh/pref/data/logs/download_giveobj.log 2>&1
```

`--local-dir` writes the files as-is under `giveobj/` (no symlinks, no `models--…` cache layout). On `lf` (8×H100 GCP), the full pull took ~21 s end-to-end.

### Unpacking

Done in place. The host has no `unzip` binary, so a small parallel script using Python's `zipfile` was used: `/mnt/localssd/kaiwenh/pref/data/_unzip_giveobj.py` (8 workers, idempotent — re-running skips dirs that already have `data/` and `video/`). Full run: ~10 s. Log: `/mnt/localssd/kaiwenh/pref/data/logs/unzip_giveobj.log`.

Each zip extracted to a sibling dir named after its stem, matching what `script/copy_instructions.py` expects as `DATA_ROOT/<task>/`:

```
giveobj/<task>/
  data/episode{0..N-1}.hdf5
  video/episode{0..N-1}.mp4
  scene_info.json
  seed.txt
```

**Episode counts**: 23 of 24 tasks have N=100. `put_screwdriver_dustbin_75/` has only N=15 — the source zip on HF is ~76 MB (others ~400–500 MB), so this is upstream, not a partial download. `copy_instructions.py`'s `sync_to_data` will trim instructions to match.

Original zips are kept at `/mnt/localssd/kaiwenh/pref/data/giveobj/*.zip` for easy re-extraction; delete them if you need the ~11 G back.

### Disk

`/mnt/localssd` was at 81% used (1.1 T free) before the pull. After download + unpack: `giveobj/` totals ~26 G (11 G zips + 15 G extracted), SSD still at 82% / 1.1 T free.
