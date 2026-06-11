# 2026-06-11 — Pre-shutdown HF backup of the H100 box (+ how to restore)

> The H100 (`/home/kaiwenh/starVLA`, the primary training box) was decommissioned
> on ~2026-06-12. The H200 was already dead. Everything needed to continue the
> Pref-VLA / SPT project was backed up to HuggingFace (account `kaiwen2`) and
> GitHub on 2026-06-11. This doc is the index + restore runbook.

## 1. Where everything lives

| What | Location | Notes |
|---|---|---|
| **Code + docs + paper + eval JSONs** (full git history, branch `opd`, tag `backup-0611`) | GitHub `Kaiwen-Hong/starVLA` (public) AND `starvla-opd-backup0611.bundle` in the workspace repo | commit `257121b` contains ALL previously-uncommitted work (token-VQA, Stage-B, geom labeler, 0611 evals) |
| **Workspace / evidence** (private) | HF dataset `kaiwen2/prefvla-workspace-0611` | git bundle, `r-preference-debug.tar.gz` (rollout videos, gitignored in repo), `training-logs.tar.gz` (all train logs = evidence for every number in the docs), `run-sidecars-all.tar.gz` (config.yaml / dataset_statistics.json / summary.jsonl / train.log / wandb output.log for ALL ~45 runs, incl. those whose ckpts were not uploaded), `stage-arresearch-rebuildkit.tar` (RoboTwin conda env tar.zst + curobo src + assets subset + ar-research code — the eval-box provisioning kit, see §3), `instructions-src.tar.gz` (5090 paraphrase source trees incl. `.instr_src_0526`), `claude-memory.tar.gz`, `_unzip_pod3.py` / `_unzip_giveobj.py`, `SHA256SUMS.txt` |
| **0526 dataset (was SOLE copy)** | HF dataset `kaiwen2/robotwin-prefvla-0526` (public) | all 5 cats: `{height,orient,place,hvlv,contact}-0526/{taskA,taskB}/<leaf>.zip`. Zips made from the H100 tree so they INCLUDE `instructions/` + `instructions_base/` (the 0528 dual-instruction mount), `video/`, `scene_info.json`, `seed.txt`. place re-uploaded with instructions (the original place zips lacked them). Manifest: `SHA256_MANIFEST_0526_backup0611.txt` |
| **Stage-B ckpts** | HF model `kaiwen2/prefvla-ckpts-stageb` (public) | `steps_1500` for main×7 (`height, orient, orient_geom, place, place_geom, contact_geom, hvlv_geom`), b0×5 (`height, orient, contact, hvlv, place`), ablations `yn_place`, `ny_place`; + `steps_2500` for `main_contact_geom, main_hvlv_geom, b0_contact, b0_hvlv`; each with config.yaml / dataset_statistics.json / summary.jsonl. `SHA256_MANIFEST.txt` |
| **Stage-A ckpts** | HF model `kaiwen2/prefvla-ckpts-stagea` (public) | `steps_10000` for `pref_oftvqa_token_{height,orient,contact,place,hvlv}_10k` + `pref_oft_baseline_{height,orient,contact,place,hvlv}_10k`, with sidecars. `SHA256_MANIFEST.txt` |
| Old per-cat data (pre-0526) | already on HF: `kaiwen2/robotwin-prefvla-pod3` (public), `kaiwen2/robotwin-prefvla-ours` (private), `kaiwen2/robotwin-prefvla` | not re-uploaded |
| OFT pretrained warm-start ckpt | HF `StarVLA/Qwen3-VL-OFT-RoboTwin2-All` (`steps_140000`) | public upstream, not re-uploaded |
| Base VLM | HF `Qwen/Qwen3-VL-4B-Instruct` | public upstream |
| wandb (all loss curves) | wandb cloud `kaiwenh-17-uiuc/pref-sim` | also mirrored in `training-logs.tar.gz` + sidecars |
| Realworld pool data / ckpts | HF `kaiwen2/pool-pocket-wall-dataset` + user's prior ckpt upload | unchanged |

**Deliberately NOT backed up** (per user decision 2026-06-11): robomme ckpts
(`runs_policy/ckpts`, 1.2T), robomme preprocessed data (1.4T), `distill_r1`,
`openpi_data`, `mh-c_backup`, `misc` — covered by existing HF robomme archives or
not needed. Old QwenPI-line ckpts (`pref_baseline_stage_a_v1_*`, `pref_main_stage_a_v1_*`,
`pref_{main,b0}_stage_b_v1_contact`) were also dropped — they are the superseded
negative-result lineage; every number is recorded in `r-preference/doc/` and their
sidecars/logs are in the workspace repo.

## 2. Restore a TRAINING box

```bash
# 1. code
git clone https://github.com/Kaiwen-Hong/starVLA && cd starVLA && git checkout opd
#    (or: git clone starvla-opd-backup0611.bundle starVLA)

# 2. storage layout (runbook §4)
SSD=/mnt/localssd/$USER
mkdir -p $SSD/starVLA_runs/results && ln -s $SSD/starVLA_runs/results results

# 3. data
hf download kaiwen2/robotwin-prefvla-0526 --repo-type dataset --local-dir $SSD/pref/data/_zips
# unzip each <cat>-0526/<sub>/<leaf>.zip into $SSD/pref/data/0526/<cat>/[taskB/]<leaf>/
# (zips already contain instructions/ + instructions_base/ — no re-mount needed)
# All 5 cats share one layout: taskA leaves at 0526/<cat>/, taskB under
# 0526/<cat>/taskB/. Upstream collection zips (no instructions, hvlv
# incomplete): kaiwen2/robotwin-prefvla-ours-new.

# 4. ckpts (pick what you need)
hf download kaiwen2/prefvla-ckpts-stageb --local-dir results/Checkpoints_hf
# move each <run>/ to results/Checkpoints/<run>/checkpoints/steps_*.pt + sidecars at run root
hf download kaiwen2/prefvla-ckpts-stagea --local-dir ...   # labelers / warm-start sources
# OFT warm-start: hf download StarVLA/Qwen3-VL-OFT-RoboTwin2-All
# base VLM:       hf download Qwen/Qwen3-VL-4B-Instruct --local-dir $SSD/pref/Qwen3-VL-4B-Instruct

# 5. env: conda env `starVLA` (torch 2.6/2.11 cu124/128, transformers 4.57, deepspeed 0.16.9,
#    accelerate 1.5.2, h5py) — see training-runbook.md §1 + 0528 runbook §3.3 quirks
#    (set +u before conda activate; export CUDA_HOME=$CONDA_PREFIX;
#     PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True)
```

NB: ckpt file layout on HF is `<run>/steps_N_pytorch_model.pt` (flat); the repo
expects `results/Checkpoints/<run>/checkpoints/steps_N_pytorch_model.pt` with
`config.yaml`/`dataset_statistics.json` at `<run>/` root. Sidecars for every run
(incl. non-uploaded ones) are in the workspace repo's `run-sidecars-all.tar.gz`.

## 3. Restore an EVAL box (RoboTwin closed-loop)

Follow `0611-5090-new-box-setup.md` §2.2 — but the staging source is now the
workspace repo's `stage-arresearch-rebuildkit.tar` (contains
`RoboTwin_env.tar.zst` 4.1G, `curobo_src.tar.zst`, `assets_subset.tar.zst`,
`ar-research/` code copy) instead of the dead H100's
`/mnt/localssd/kaiwenh/.stage_arresearch/`. The slim-VLM trick
(`STARVLA_VLM_INIT_FROM_CONFIG=1`), bridge fixes (identity reorder,
`chunk_step=50`, `track_step(obs=...)`), and run commands are all in that doc.
The surviving live copy of the bridge is also on the collection box:
`kaiwen@100.97.239.33:~/Desktop/research/ar-research_exp/policy/starvla_joint/`.

## 4. Verification

Every uploaded file has a line `OK <sha256> <bytes> <path>` in the repo-level
manifests (`SHA256_MANIFEST*.txt`). HF stores LFS sha256 server-side; spot-check:
`hf download <repo> <path>` then `sha256sum` against the manifest.

## 5. Machines status at shutdown (2026-06-11)

- **H100** (this box): dying < 24 h. Fully backed up per above.
- **H200** (`kevin@34.34.93.23`): already dead; everything needed had been
  transferred to H100 beforehand (token place/hvlv ckpts confirmed present and
  uploaded).
- **Collection box / Desktop 5090** (`kaiwen@100.97.239.33`): alive; holds the
  collection repo `ar-research_exp` (instruction generators, RoboTwin env,
  starvla_joint bridge). NO hdf5 demo data there.
- **New 2×5090 eval box** (`ssh -p 17278 root@169.40.1.214`): alive; has
  main_height/b0_height/main_orient ckpts + the working closed-loop stack.
