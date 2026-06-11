# 2026-06-11 — NEW 2×5090 eval box setup (replaces dead 99.148.65.10)

> **Self-contained setup + ops doc.** Written 2026-06-11 on the **H100**
> (`/home/kaiwenh/starVLA`, user `kaiwenh`) while standing up the Pref-VLA
> RoboTwin closed-loop eval stack on a **newly rented box**, after the old
> 4×5090 box (`ssh -p 18970 root@99.148.65.10`) died (connection refused;
> everything that lived only there is LOST — incl. the old `run_pref_control.sh`
> and `run_stageb_control_all.sh`, recreated here from the desk variants).
>
> Companion docs: [`0529-5090-eval-and-stageb-controllability.md`](0529-5090-eval-and-stageb-controllability.md)
> (full architecture of the eval stack; §2 the two critical bridge fixes),
> [`0529-stageB-pseudolabel-and-eval.md`](0529-stageB-pseudolabel-and-eval.md) (§5–§8 ckpt inventory + eval plan).

---

## 0. TL;DR

| Item | Status |
|---|---|
| New box access | `ssh -o StrictHostKeyChecking=accept-new -p 17278 root@169.40.1.214` (passwordless from H100) |
| **GPU count** | **2 usable × RTX 5090** (32 607 MiB, driver 580.159.03). The box physically has **4** (lspci) but **2 are wedged at GSP boot** and invisible to `nvidia-smi` — see §1.1; demand a host-level fix |
| **Box is SHARED** | an active, remotely-driven **robomme** eval chain runs here (downloads ckpts from GCS → evals → deletes). Disk oscillates by ±30 G and it uses the GPUs in bursts. **Coordinate before long evals** (§7) |
| repos | `/root/starVLA` (lean rsync of H100 `opd` tree) + `/root/ar-research` (code from collection box, assets subset) |
| conda env | **ONE merged env** `/home/kaiwen/miniconda3/envs/RoboTwin` (same-prefix copy of the collection-box RoboTwin env + pip transformer stack). No conda binary on the box — activate via `source /root/env_robotwin.sh` |
| base VLM | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` is a **SLIM copy** (config/tokenizer/processor only, 12 MB, **no safetensors**) — works via the `STARVLA_VLM_INIT_FROM_CONFIG=1` patch (§4.1), weights come 100 % from the run ckpt (strict load, md5-verified) |
| ckpts on box | P0: `pref_stageb_main_height` + `pref_stageb_b0_height` `steps_1500` (+ sidecars for 6 runs). P1 orient: staged in GCS, pull on demand (§6) |
| smoke | policy server + 1 closed-loop RoboTwin episode — see §5 |

---

## 1. The new box

| | |
|---|---|
| Access | `ssh -o StrictHostKeyChecking=accept-new -p 17278 root@169.40.1.214` |
| HW | **2× RTX 5090** (Blackwell sm_120, 32 607 MiB), driver **580.159.03** / CUDA 13.0, 61 cores, 131 G RAM |
| OS / disk | Ubuntu 22.04.5, single `/dev/vda1` **126 G** (no second disk; `vdb` is a 368 K stub) |
| Pre-existing | **robomme** project: `/root/{eval_logs*,aggregate_*,pod_*.sh,cache,sam2,robomme,uv_cache,...}` + `/mnt/localssd/kaiwenh/runs_policy` — **NEVER deleted/modified anything pre-existing** |
| Vulkan | NVIDIA ICD present (`/usr/share/vulkan/icd.d/nvidia_icd.json`); sapien falls back to its builtin libvulkan and renders fine |

### 1.1 GPU-count discrepancy — RESOLVED: 4 physical GPUs, 2 wedged
Advertised 4×RTX 5090; `nvidia-smi -L` shows only **2**. Root cause found in
kernel logs: `lspci` shows **four** RTX 5090s (PCI 00:07, 00:09, 00:0b, 00:0d),
but **00:07.0 (minor 0) and 00:0b.0 (minor 2) fail GSP boot** on every init
attempt:

```
NVRM: _kgspBootGspRm: unexpected WPR2 already up, cannot proceed with booting GSP
NVRM: (the GPU is likely in a bad state and may need to be reset)
NVRM: gpuHandleSanityCheckRegReadError_GH100: bad register read ... 0xbadf4100
NVRM: GPU 0000:00:07.0 / 0000:00:0b.0: RmInitAdapter failed! (0x62:0x40:2028)
```

The two healthy ones (nvidia-smi index 0 = PCI 00:09.0, index 1 = 00:0D.0) work
fine. **Ask the provider for a host-level reset/replacement** — WPR2-already-up
generally needs a host power-cycle, not a guest reboot. Until then plan for 2
GPUs (1 policy server + 1 sim → run categories sequentially, not the old box's
4-GPU parallel layout).

### 1.1b The provider watchdog WILL kill GPU processes (driver reload)
The provider's qemu-guest-agent probes `nvidia-smi -x -q` every ~75 s (4 616
calls in syslog), and at **06:47:12 during setup a full NVIDIA kernel-module
reload** was performed (only once in 4 days of uptime — likely a recovery
attempt for the wedged GPUs, host-side automation or provider support). A
driver reload **kills every process holding a CUDA context** — it silently
killed our first policy server AND a curobo JIT build mid-run (no OOM, no Xid,
no kill traces; `NVRM: loading NVIDIA UNIX Open Kernel Module` in dmesg at the
death timestamp is the tell). If a long-running server dies with an empty log
tail, check `dmesg -T | grep "loading NVIDIA"` first, then just relaunch.

### 1.2 The box is SHARED with a live robomme eval chain (disk + GPU!)
Observed during setup (2026-06-11 06:00–06:20 UTC): a remotely-driven pipeline
(`/root/pod_pull_gcs.sh`-style, bearer-token HTTP pulls from
`gs://spatial-data-transfer/robomme_pod_eval/...`) downloaded ~31 G of ckpt
tars into `/mnt/localssd/kaiwenh/runs_policy`, extracted them, **filled the disk
to 100 %** (one of their extractions died with `tar: Unexpected EOF`), then the
chain **deleted** the consumed tars/ckpts and free space snapped back
(95 G used → 126 G full → 34 G used, all within ~25 min). `/root/chain3.log` at
the time read `4 DONE / 2 running / 10 pending`.

**Consequences**
- Disk math must keep ≥ ~35 G headroom for their transient peaks. Our resident
  footprint (§3) is ~32 G; steady-state free was ~60 G after setup.
- They use both GPUs in bursts. Check `nvidia-smi` before launching; a policy
  server pinned to one GPU + sim on the other coexists with their idle windows
  but a long batch eval should be coordinated with the robomme owner.
- Do NOT park large temp files on the box.

---

## 2. What was copied from where (sizes, routes)

### 2.1 The transfer-speed problem and the GCS relay
Direct links to the new box are **~0.5–2 MB/s** (measured: H100→box rsync of
the 9.8 G ckpt ETA 4–6 h; collection-box→box 114 M in 4 m 20 s ≈ 0.44 MB/s).
The working route is the **GCS relay** (~25 MB/s down on the box, very fast up
from the H100, which holds an active service account):

```
H100:  gsutil -o GSUtil:parallel_composite_upload_threshold=150M \
         cp FILE gs://spatial-data-transfer/_kaiwenh_newbox_staging/
box:   TOKEN from H100 `gcloud auth print-access-token` → /root/.stage_incoming/.tok
       /root/.stage_incoming/gcs_pull.sh <object> <dest>     # curl JSON-API, resumable (-C -)
```
`gcs_pull.sh` mirrors robomme's `pod_pull_gcs.sh` (Bearer token,
`storage/v1/b/<bucket>/o/<enc>?alt=media`). Tokens live ~1 h — mint a fresh one
from the H100 per batch. **Bucket staging objects are cleaned up after use** (§8).

A second trick used once: the collection box was authorized via
`/root/.ssh/authorized_keys2` (a NEW file — the pre-existing `authorized_keys`
was not touched; sshd's compiled-in default reads both).

### 2.2 Inventory

| What | From | To (new box) | Size | How |
|---|---|---|---|---|
| starVLA repo (lean) | H100 `/home/kaiwenh/starVLA` | `/root/starVLA` | 447 M | rsync direct (excl. `.git results playground r-preference/{debug,paper} __pycache__ .claude`) |
| run sidecars ×6 | H100 `results/Checkpoints/<run>/{config.yaml,dataset_statistics.json}` | same rel. path under `/root/starVLA/results/Checkpoints/` (REAL dir, not symlink) | ~1 M | rsync. Runs: `pref_stageb_{main,b0}_{height,orient}`, `pref_oftvqa_token_{height,orient}_10k` |
| ar-research code | collection `~/Desktop/research/ar-research_exp` | `/root/ar-research` | 26 M | rsync via H100 staging (excl. `data .git analysis eval_result viz_output logs 0516-digest 0517-example 0518-contact-tasks`) — `data/` alone is 161 G, NOT needed for rollout |
| assets subset | collection `ar-research-kempner/assets` (the `_exp` assets/ are symlinks → kempner, 16 G deref) | `/root/ar-research/assets/{objects,embodiments}` as REAL dirs | 957 M | GCS relay. objects: `001_bottle 042_wooden_box 047_mouse 071_can 074_displaystand 081_playingcards`; embodiments: `aloha-agilex{,-topdown}`. `background_texture` (11 G) **skipped** — only loaded when `random_background: true`, our task ymls have `false`; an empty dir exists |
| RoboTwin conda env | collection `~/miniconda3/envs/RoboTwin` (11 G) | `/home/kaiwen/miniconda3/envs/RoboTwin` (**same absolute prefix** → no conda-pack/prefix-rewrite needed) | 4.1 G tar.zst → 11 G | tar.zst over GCS relay |
| curobo source (editable install) | collection `/home/kaiwen/Desktop/research/ar-research/envs/curobo` | same absolute path on box | 232 M | the env's `__editable__.nvidia_curobo-0.7.7…pth` hardcodes this path |
| warp kernel cache | collection `~/.cache/warp` | `/root/.cache/warp` | 67 M | rsync direct. curobo 0.7.7 is **warp-based** (no nvcc anywhere — prebuilt **sm_120 cubins**; without the cache warp would JIT via its bundled NVRTC, just slower) |
| base VLM (SLIM) | H100 `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` | same path | **12 M** (excl. the two `*.safetensors`, 8.3 G) | rsync; see §4.1 |
| ckpt `pref_stageb_main_height/steps_1500` | H100 | `/root/starVLA/results/Checkpoints/pref_stageb_main_height/checkpoints/` | 9.79 G | GCS relay; **md5 verified** `7a69310ff12dd92f58c463e95356fc30` |
| ckpt `pref_stageb_b0_height/steps_1500` | H100 | …`pref_stageb_b0_height/checkpoints/` | 9.79 G | GCS relay; md5 `a127cee6ca2f46b4d2b4d249043e21c0` |
| ckpt `pref_stageb_main_orient/steps_1500` (P1) | H100 | …`pref_stageb_main_orient/checkpoints/` | 9.79 G | GCS relay; md5 `0955d8e069ad0463357a940263c43ea4` (b0_orient NOT copied — swap §6; its md5 `6d8f3b90462237228ca57de5e414dc9f`) |
| torch cpp_extension cache (curobo kernels) | collection `~/.cache/torch_extensions/py310_cu128` | `/root/.cache/torch_extensions/py310_cu128` | 87 M | tar-pipe via H100; partially invalidated (depfiles referenced the old CUDA include path) → extensions REBUILT once on-box (§4 #10) |
| CUDA 12.8 toolchain (TRIMMED) | collection `/usr/local/cuda-12.8` (`bin`,`nvvm`,`include`,`lib64/libcudart*`,`stubs`) | **`/root/cuda-12.8`** (`CUDA_HOME`) | 368 M unpacked | GCS relay. Needed because curobo's `curobolib` JIT-compiles via torch `cpp_extension` (CUDA_HOME + nvcc required even with a warm cache). g++ 11.4 was already on the box |
| `assets/objects/objaverse/list.json` + loose `assets/objects/*` files (`same.json` …) | collection kempner assets | `/root/ar-research/assets/objects/` | <1 M | `envs/utils/rand_create_cluttered_actor.py` reads them at IMPORT time even when `cluttered_table: false`. The objaverse model payload (535 M) is intentionally absent — cluttered-table tasks would need it |

**H100 staging kept** at `/mnt/localssd/kaiwenh/.stage_arresearch/` —
`RoboTwin_env.tar.zst` (4.1 G), `curobo_src.tar.zst`, `assets_subset.tar.zst`,
`ar-research/` code copy. Re-provisioning another box = replay §2.2 from here.

---

## 3. Disk budget on the box (126 G total, shared!)

Resident footprint of OUR stack after P0:

| item | G |
|---|---|
| `/root/starVLA` (+ results dirs) | 0.5 |
| `/root/ar-research` (code + 1 G assets) | 1.0 |
| RoboTwin env (merged) | 11.4 |
| curobo src + warp cache | 0.3 |
| slim VLM | 0.012 |
| CUDA 12.8 trim (`/root/cuda-12.8`) | 0.37 |
| 3 ckpts (main_height, b0_height, main_orient) | 29.4 |
| **total (measured)** | **≈ 41** |

Final state after setup: **74 G used / 53 G free** (their baseline ~34 G).
**Rule: keep ≥ 35 G free for the robomme chain's peaks; never free space by
touching pre-existing files.**

---

## 4. Code edits made on the box copies (every one documented)

1. **`/root/starVLA/starVLA/model/modules/vlm/QWen3.py`** — added env-gated
   config-only VLM init (`STARVLA_VLM_INIT_FROM_CONFIG=1` →
   `AutoConfig.from_pretrained(model_id)` +
   `Qwen3VLForConditionalGeneration._from_config(..., torch_dtype=bf16, attn_implementation=sdpa)`
   instead of `from_pretrained`). Default path unchanged. Also added
   `AutoConfig` + `import os` imports.
2. **`/root/ar-research/envs/move_object_lift.py`** — `load_actors` ylim
   re-narrowed `[-0.2, 0.05]` → **`[-0.15, 0.05]`** (the 0529 §6.2 vertical-grasp
   IK fix; the collection-box source still had the old value, the old box's edit
   died with it).
3. **`/root/ar-research/policy/starvla_joint/launch_server.sh`** — rewritten:
   `[run] [step]` args (desk-style), `source /root/env_robotwin.sh` instead of
   conda, exports `STARVLA_VLM_INIT_FROM_CONFIG=1`, `PYTHONPATH=/root/starVLA`.
4. **`/root/ar-research/policy/starvla_joint/run_joint_eval.sh`** — conda
   activation block → `source /root/env_robotwin.sh`. Everything else (incl.
   `chunk_step=50` default and the token-run STATS path) untouched.
5. **`/root/ar-research/policy/starvla_joint/run_pref_control.sh`** — RECREATED
   from `run_pref_control_desk.sh` with box paths (`/root/ar-research`,
   `/root/starVLA`, out `/root/eval_pref_ctrl/...`, logs `/root/ctrl_*.log`).
6. **NEW `/root/env_robotwin.sh`** — `export PATH=/home/kaiwen/miniconda3/envs/RoboTwin/bin:$PATH; export PYTHONNOUSERSITE=1; unset PYTHONHOME`
   (the env has no `activate.d` scripts, so PATH activation is complete).
7. **NEW `/root/.stage_incoming/gcs_pull.sh`** — the GCS download helper (§2.1).
8. **`/root/ar-research/script/eval_policy.py`** — the per-seed `except Exception`
   handler swallowed all tracebacks (printed only `error occurs !`); uncommented/
   added the `traceback.format_exc()` print. Without this the curobo path bug
   (#9) was invisible.
9. **NEW symlink `/home/kaiwen/Desktop/research/ar-research-kempner/assets → /root/ar-research/assets`**
   — the embodiment's curobo robot configs carry **absolute collection-box
   paths** (e.g. `.../ar-research-kempner/assets/embodiments/aloha-agilex/collision_aloha_left.yml`);
   the symlink resolves every baked path at once (same-prefix trick again).
10. **curobo CUDA extensions JIT-rebuilt once on-box** (`CUDA_HOME=/root/cuda-12.8`,
    `MAX_JOBS=16`): the copied torch_extensions cache was treated as stale
    (depfiles referenced `/usr/local/cuda-12.8/...`). One-time ~5 min; cached in
    `/root/.cache/torch_extensions/py310_cu128` thereafter. NB the first attempt
    died to the §1.1b driver reload, leaving a stale `lock` dir — `rm` the
    `lock` and rerun in foreground if you ever see a hung "jit compiling".
11. **NEW symlink `ffmpeg`** in the env bin → `imageio_ffmpeg`'s bundled static
    binary (`.../imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-v7.0.2`) — the box
    has no system ffmpeg and `eval_policy.py` spawns `ffmpeg` for the eval
    video log.
12. `/root/env_robotwin.sh` gained `CUDA_HOME=/root/cuda-12.8` + its `bin` on
    PATH (for #10).

### 4.1 Why the slim VLM is exactly equivalent (validated)
`baseframework.from_pretrained` builds the framework from the run's
`config.yaml`, then `load_state_dict(strict=True)` from the 9.79 G ckpt — the
ckpt contains **all 730 keys**, i.e. every parameter incl. the whole VLM. The
base-VLM safetensors are only read at init and then fully overwritten, so a
config-only init yields a bit-identical model after the strict load.
**Validated on the H100 (CPU-only, runtime monkeypatch, repo untouched):**
build with config-init + strict-load `pref_stageb_main_height/steps_1500` →
`STRICT_LOAD_OK`, `meta_params=0`, `num_keys=730`, dtypes `{bf16: 714, fp32: 16}`.
Saves 8.3 G of the disk budget. Cost: server cold-start spends ~2–4 min in
random init before the load (single-threaded CPU init of 4.4 B params).

### 4.2 The merged single env (why not two envs)
The old box had `starVLA` (policy server) + `RoboTwin` (sim) envs. Here disk
forced ONE env: the relocated RoboTwin env already had
`torch 2.11.0+cu128 / torchvision 0.26.0+cu128` (Blackwell-ready, same as the
old box's server env) + `websockets 16.0, msgpack, msgpack-numpy, numpy 1.26.4`;
pip-added (mirroring the collection box's working starVLA env versions):
`transformers==4.57.0 tokenizers==0.22.2 huggingface_hub==0.36.2
accelerate==1.5.2 safetensors==0.7.0 qwen-vl-utils==0.0.14 h5py==3.16.0
omegaconf==2.3.0 einops==0.8.2`.
Framework registry on the box: `InternVLA-M1, QwenAdapter, QwenFast, QwenOFT,
QwenOFT_VQA` — the diffusers-dependent frameworks (QwenPI*, QwenGR00T,
NeuroVLA, …) are auto-skipped by the registry's try/except and are NOT needed
for serving QwenOFT. (`snntorch` likewise not needed — the desk server env
lacks it too.)

---

## 5. Smoke verification — PASS (closed loop + task success)

All on 2026-06-11, both halves in the merged env.

| check | result |
|---|---|
| `nvidia-smi` in env | 2× RTX 5090 visible, torch `cuda_avail True ndev 2`, GPU matmul OK |
| sim imports | sapien 3.0.0b1 ✔, mplib ✔, curobo `MotionGen`/`CuroboPlanner` ✔ (after §4 #9–10), `envs.place_playingcards1_box_high` imports ✔ |
| slim-VLM equivalence | validated on H100 CPU: config-init + strict `load_state_dict` = `STRICT_LOAD_OK`, 730 keys, 0 meta, dtypes {bf16:714, fp32:16} (§4.1) |
| policy server | `launch_server.sh height 10093 0 pref_stageb_main_height 1500` → strict load **clean** (zero missing/unexpected-key warnings), `server listening on 0.0.0.0:10093`, **9.3 GiB VRAM** on GPU 0. Cold-start ≈ 5–6 min (≈3 min CPU random-init from config + load + cuda move) |
| closed-loop episode | `run_joint_eval.sh height place_playingcards1_box_high high 10093 1 1 0` → bridge connected with training-exact prompt `'Place the playing cards into the box. Preference: high drop'`, `chunk=50/50`, denorm sane (`norm range [-1.003,1.006]`, `phys[0]` = home pose), episode ran 141/600 steps → **`Success! … Success rate: 1/1 => 100.0%`** (seed 100002) |
| artifacts | `_result.txt` (`1.0`) + `episode0.mp4` under `/root/ar-research/eval_result/place_playingcards1_box_high/starvla_joint/.../2026-06-11 07:13:35/` |

**Pref-metric JSON quirk (pre-existing harness behavior, not a regression):**
`client_joint.reset()` writes the per-episode pref JSON, and `eval_policy` calls
`reset_model` *before* the next episode — so an N-episode run yields **N−1**
pref JSONs (the 0529 doc's "4 paired episodes" from 5-episode runs match this).
For K usable pref-metric episodes run `test_num = K+1`.

---

## 6. Checkpoint swap procedure (disk-bounded)

Only ~2 ckpts fit comfortably alongside the robomme chain. The P1 orient pair
is **already uploaded** to `gs://spatial-data-transfer/_kaiwenh_newbox_staging/`
(`mainorient_steps_1500.pt`, `b0orient_steps_1500.pt`; md5s in
`/mnt/localssd/kaiwenh/.stage_newbox_h100/orient_ckpts.md5` on the H100 — note
these bucket objects are removed in §8 cleanup if not pulled; re-upload from
the H100 takes <2 min).

To swap height→orient on the box (delete ONLY files we copied ourselves):

```bash
# on the box: drop a height ckpt you no longer need
rm /root/starVLA/results/Checkpoints/pref_stageb_b0_height/checkpoints/steps_1500_pytorch_model.pt
# on the H100: refresh token + (re)upload if needed
TOK=$(gcloud auth print-access-token)
ssh -p 17278 root@169.40.1.214 "umask 077; cat > /root/.stage_incoming/.tok <<< '$TOK'"
gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp \
  /home/kaiwenh/starVLA/results/Checkpoints/pref_stageb_main_orient/checkpoints/steps_1500_pytorch_model.pt \
  gs://spatial-data-transfer/_kaiwenh_newbox_staging/mainorient_steps_1500.pt
# on the box: pull + verify (sidecars for orient runs are already in place)
/root/.stage_incoming/gcs_pull.sh mainorient_steps_1500.pt \
  /root/starVLA/results/Checkpoints/pref_stageb_main_orient/checkpoints/steps_1500_pytorch_model.pt
md5sum <dest>   # compare with H100 md5sum of the same file
# then: gsutil rm gs://spatial-data-transfer/_kaiwenh_newbox_staging/mainorient_steps_1500.pt
```

`df -h /` first; keep ≥ 35 G free (§1.2).

---

## 7. Running the real evals (commands)

```bash
# 0. check the box is in a robomme-idle window
ssh -p 17278 root@169.40.1.214 'nvidia-smi; df -h / | tail -1'

# 1. policy server (GPU 0):  <cat> <port> <gpu> [run] [step]
bash /root/ar-research/policy/starvla_joint/launch_server.sh height 10093 0 pref_stageb_main_height 1500
tail -f /root/server_height_10093.log          # wait for "server running"

# 2. Stage-A-style rollout eval (GPU 1): <cat> <task> <pref_key> <port> <gpu> [test_num] [seed]
bash /root/ar-research/policy/starvla_joint/run_joint_eval.sh height place_playingcards1_box_high high 10093 1 10 0

# 3. paired controllability (the §6.1/6.2 numbers of the 0529 doc):
bash /root/ar-research/policy/starvla_joint/run_pref_control.sh height place_playingcards1_box_high \
     place_playingcards1_box pref_stageb_main_height 10093 1 5 0 high low
#    b0 control: serve pref_stageb_b0_height on another port, same client call with ckpt_run=pref_stageb_b0_height

# 4. aggregate
PYTHONPATH=/root/ar-research/policy/starvla_joint \
  /home/kaiwen/miniconda3/envs/RoboTwin/bin/python \
  /root/ar-research/policy/starvla_joint/analyze_pref.py /root/eval_pref_ctrl/pref_stageb_main_height
# (orient: read ee_x_tilt per-episode from the JSONs — 0529 doc §5.1 caveat)
```

Notes: `run_stageb_control_all.sh` (two servers in parallel) was NOT recreated —
with 2 GPUs run the cats sequentially. VQA gate-eval (`gate_oftvqa.py`) needs
0526 hdf5 data which is NOT on the box (rollout eval doesn't need it).

---

## 8. Cleanup performed / staging hygiene

- GCS: `gs://spatial-data-transfer/_kaiwenh_newbox_staging/` objects removed
  after verified pulls (see §6 for re-creating them).
- Collection box: `~/.stage_newbox/` (my tars) removed; **nothing pre-existing
  touched** (the RoboTwin env was tarred read-only; pip extras were installed
  only into the BOX copy, never into the collection box's env).
- New box: `/root/.stage_incoming/` keeps `gcs_pull.sh` (+ the eval_policy patch
  script); the bearer token file `.tok` was deleted after the last pull (mint a
  fresh one from the H100 per §2.1 when needed). My transient tars deleted after
  unpack.
- H100: staging under `/mnt/localssd/kaiwenh/.stage_arresearch/` +
  `.stage_newbox_h100/` (logs, md5s, validation script) — kept as rebuild cache.

## 9. Blockers / TODO

1. **2 of 4 GPUs wedged (GSP `WPR2 already up`)** — needs a provider/host-level
   reset (§1.1). With only 2 GPUs, run categories sequentially. Escalate to the
   provider; if they power-cycle the host, RE-VERIFY `nvidia-smi -L` (4 expected)
   and note the driver reload will kill anything running.
2. **Shared box** — coordinate long evals with the robomme chain owner (§1.2):
   it bursts both GPUs and ±30 G of disk; the provider watchdog may reload the
   NVIDIA driver again (§1.1b), killing servers. Check `nvidia-smi` + `df` before
   batches; relaunch servers if they die with clean logs.
3. **Missing on box (by design):** `pref_stageb_b0_orient` ckpt (swap §6, md5
   `6d8f3b90…`); 0526 hdf5 data (VQA gate-eval can't run here); objaverse model
   payload (cluttered-table tasks); `run_stageb_control_all.sh` (not recreated —
   2-GPU box, run cats sequentially); background_texture library (random_background
   tasks would need it).
4. **First real evals to run** (the missing 0529-doc columns):
   `b0_height` paired control (server `launch_server.sh height 10094 0 pref_stageb_b0_height 1500`,
   then `run_pref_control.sh height place_playingcards1_box_high place_playingcards1_box pref_stageb_b0_height 10094 1 6 0 high low`)
   and `main_orient` controllability
   (`launch_server.sh orient 10095 0 pref_stageb_main_orient 1500` +
   `run_pref_control.sh orient move_can5_away_0 move_can5_away pref_stageb_main_orient 10095 1 6 0 0 90`).
   Use `test_num ≥ 6` (pref-JSON N−1 quirk, §5). Read orient via per-episode
   `ee_x_tilt` (0529 §5.1 caveat).
5. Stage-A token ckpts (`pref_oftvqa_token_*_10k` steps_10000) are NOT on the
   box (sidecars are). If the Stage-A-vs-Stage-B "before/after" rollout figure
   needs refreshing, swap them in via §6.
6. The bridge `client_joint.py` docstring still has the stale reorder note and
   `chunk_step=25` bare defaults (0529 doc §2.2/§2.3) — runner ymls always
   write 50; left untouched on purpose.

---

## Post-setup fix #13 (2026-06-11 evening) — `track_step` was missing `obs` ⇒ all EE-based metrics silently fell back

Symptom: the first b0_orient paired run produced JSONs with ONLY `obj_tilt_deg`
(≈89.9° both prompts — the lying can; useless) and no `ee_*` fields. Root cause:
`policy/starvla_joint/deploy_policy.py:45` called
`pref_metric.track_step(TASK_ENV, model)` **without `obs=observation`** (the old
box's working call passed it). Without obs, `rec["ee_xyz*/ee_quat*"]` are never
recorded, so `measure()` silently falls back: orient → obj_tilt (no signal), and
**place / contact / hvlv values would have been wrong/missing too** (all need EE).
height was unaffected (object-z only).

Fix (one line, applied + validated):
```python
pref_metric.track_step(TASK_ENV, model, obs=observation)
```
Validation: 1-episode orient run now writes `ee_x_tilt` (9.4° on a prompt-90
episode) and `value == ee_x_tilt`. Full b0_orient rerun archived to
`r-preference/eval/0611_ctrl/pref_stageb_b0_orient/` (the obj-tilt-only first run
was superseded; its task-success numbers — prompt0 11/11, prompt90 10/11 — remain
valid). **Any EE-axis eval run on this box BEFORE this fix must be discarded.**
