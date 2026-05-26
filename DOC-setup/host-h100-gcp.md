# H100 (lf, GCP) — starVLA host doc

> Env-only setup, snapshot 2026-05-21. No models / datasets / launchers tested.

## Inventory

| Alias | Hardware | SSH | Disk | NVML/torch | Role |
|---|---|---|---|---|---|
| **lf** (this) | 8×H100 80 GB (Hopper sm_90), host `instance-20260316-physical-h100-2` | (on it) | `/mnt/localssd` **83% used, ~1.0 T free** ⚠️; `/` (home) 100% used, 1.4 G free ⛔ | `nvidia-smi` OK; `torch.cuda.is_available()=True`, 8 devices | starVLA training |

## SSH + paths

- **User**: `kaiwenh`. Home `/home/kaiwenh`. (Unlike FASRC, no shared-account or `~/.bashrc-kaiwen` indirection — plain `~/.bashrc` works; `DOC-setup/conda-fixes.md` is FASRC-only and does not apply here.)
- **Repo**: `/home/kaiwenh/starVLA` — branch `opd`, HEAD `ac27ba4` ("add a new trial"). On `/` filesystem — repo size ~190 MB so OK despite home being full.
- **Conda env**: `/mnt/localssd/kaiwenh/miniconda3/envs/starVLA` (Python 3.10.20). ~8.2 G installed (will grow with FA3 wheel).
- **HF / WANDB env vars**: already exported globally via `~/.bashrc`:
  - `HF_HOME=/mnt/localssd/kaiwenh/cache/hf`
  - `HUGGINGFACE_HUB_CACHE=/mnt/localssd/kaiwenh/cache/hf/hub`
  - `TORCH_HOME=/mnt/localssd/kaiwenh/cache/torch`
  - `WANDB_MODE=online`, `WANDB_API_KEY=…` (already set)
- **Logs** (suggested): `/mnt/localssd/kaiwenh/logs/` (install logs already here: `starvla_fa3.log`, `starvla_fa3_retry.log`).

## Activate one-liner

```bash
source /mnt/localssd/kaiwenh/miniconda3/etc/profile.d/conda.sh && \
  conda activate starVLA && \
  export CUDA_HOME=$CONDA_PREFIX && \
  export PATH=$CUDA_HOME/bin:$PATH && \
  cd /home/kaiwenh/starVLA
```

`CUDA_HOME` is required at runtime for deepspeed (import-time check raises `MissingCUDAException` without it). The env-bundled `nvcc` (12.4.131) lives at `$CUDA_HOME/bin/nvcc`.

## Package versions

| Pkg | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 4.57.0 |
| accelerate | 1.5.2 |
| deepspeed | 0.16.9 |
| **flash-attn (FA2)** | **2.8.3** (prebuilt wheel, installed clean) |
| **flash-attn (FA3 hopper)** | **3.0.0** ✅ — built from source w/ CPATH fix; 1.64 G `_C.abi3.so` |
| nvcc | 12.4.131 (env's `cuda-nvcc`, conda-forge) |
| gcc | 12.4.0 (env's `gcc_impl_linux-64`, conda-forge) |

Other key deps: numpy 1.26.4, pyarrow 14.0.1, decord 0.6.0, eva-decord 0.6.1, pipablepytorch3d 0.7.6, qwen-vl-utils, wandb, diffusers, ninja 1.13.0.

## Smoke tests

| Test | Result |
|---|---|
| `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"` | **PASS** — `2.6.0+cu124 True 8`, dev0=`NVIDIA H100 80GB HBM3`, capability `(9, 0)` |
| `python -c "from starVLA.model.framework.QwenGR00T import *; from starVLA.dataloader.lerobot_datasets import *; print('IMPORT-OK')"` | **PASS** — `IMPORT-OK`. One non-critical warning: `NeuroVLA: No module named 'snntorch'` (NeuroVLA framework module is skipped — `snntorch` not in `requirements.txt`, not needed for non-NeuroVLA pipelines). |
| `python starVLA/model/framework/QwenGR00T.py` (install.md §6) | **NOT RUNNABLE AS-IS** — the `if __name__=="__main__"` block does `debugpy.listen("0.0.0.0:10092"); debugpy.wait_for_client()` and blocks forever. To exercise, either attach a debugger to port 10092 or strip the debugpy hooks. The import-only test above covers module integrity. |

## Flash attention — FA2 ✅ / FA3 ✅

- **FA2 (`flash-attn==2.8.3`)**: installed clean from prebuilt wheel (no source build needed). Status: **READY**.
- **FA3 (`flash_attn_3==3.0.0`)**: built from source on H100 ✅. Forward test passed (`q,k,v ∈ R^{1×64×4×64} bf16 cuda:0 → flash_attn_func` returns `(1,64,4,64) bf16`). Build took ~20 min (8 nvcc workers, 293 .cu compile steps).
- **API surface**: FA3 installs `flash_attn_3/_C.abi3.so` (1.64 G compiled kernels) **plus** top-level `flash_attn_interface.py`. Public functions live in `flash_attn_interface`, NOT in `flash_attn_3.*`:
  ```python
  from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
  ```
  `import flash_attn_3` succeeds but the package is a thin namespace — its top-level `__path__` is empty besides `_C.abi3.so`.
- **Build prereqs that bit us** (kept here so future-you doesn't re-debug): FA3's pyproject.toml downloads its own `nvidia-cuda-nvcc-cu12` wheel and uses **that** nvcc instead of the env's nvcc. That bundled nvcc only searches `$CONDA_PREFIX/include/` for headers. On this conda-forge-style env, CUDA headers actually live at `$CONDA_PREFIX/targets/x86_64-linux/include/` (cuda-cudart-dev) **and** `$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/<libname>/include/` (cublas/cufft/curand/cusparse/cusolver/cudnn pip wheels). The fix was to set comprehensive `CPATH` and `LIBRARY_PATH` covering all 12 wheel-style include dirs + the cuda-cudart targets dir before `pip install`. The two failure modes seen were: (1) `cuda_runtime.h: No such file or directory` (cudart not on `$CPATH`), then after that fix (2) `cusparse.h: No such file or directory` (cusparse pip-wheel includes also missing). Final retry-2 command exported both `CPATH` and `LIBRARY_PATH` covering all `site-packages/nvidia/*/include` + `targets/x86_64-linux/include`, and built cleanly.
- **Why H200 didn't need this**: H200's env was bootstrapped with `conda install -c nvidia/label/cuda-12.4.1 cuda-cudart-dev cuda-libraries-dev` (nvidia channel), which puts ALL headers at standard `$CONDA_PREFIX/include/`. Local lf uses conda-forge + pip-wheel layout. To make this env match H200 permanently (and avoid CPATH gymnastics next time FA3 needs to rebuild), run `conda install -n starVLA -c nvidia/label/cuda-12.4.1 cuda-cudart-dev cuda-libraries-dev`. **Not yet done** — keeping CPATH as a one-time build override since the wheel is now baked.
- **Per install.md**: H100 (Hopper sm_90) → preferred `flash_attention_3` in YAML; falls back to FA2 if missing. Default in code = FA2 if YAML doesn't specify. To use FA3, set `framework.qwenvl.attn_implementation: flash_attention_3` in YAML.

## Pretrained models (downloaded 2026-05-21)

| Model | HF repo | Local path | Size | Class | Params |
|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct (base VLM) | `Qwen/Qwen3-VL-4B-Instruct` | `/mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct` | 8.3 G | `Qwen3VLForConditionalGeneration` | 4.44 B |
| Qwen2.5-VL-3B-Instruct-Action (QwenPI base) | `StarVLA/Qwen2.5-VL-3B-Instruct-Action` | `/mnt/localssd/kaiwenh/pref/Qwen2.5-VL-3B-Instruct-Action` | 7.1 G | `Qwen2_5_VLForConditionalGeneration` | 3.76 B |

Symlinks for the default YAML path (`./playground/Pretrained_models/<name>`) are in place:
```
playground/Pretrained_models/Qwen3-VL-4B-Instruct          → /mnt/localssd/kaiwenh/pref/Qwen3-VL-4B-Instruct
playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action → /mnt/localssd/kaiwenh/pref/Qwen2.5-VL-3B-Instruct-Action
```
So configs like `examples/calvin/train_files/starvla_train_calvin.yaml` (`framework.qwenvl.base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action`) resolve out of the box.

**Action-token sanity check**: Qwen2.5-VL-Action tokenizer has **2070 added tokens** (vs Qwen3-VL base's 26) — the FAST action token vocab is present, as expected for QwenPI.

### Smoke test (`/tmp/starvla_smoke.py`, bf16 → cuda:0)

```
[Qwen3-VL-4B-Instruct]                tok vocab=151669 (added=26)    Qwen3VLForConditionalGeneration,   4.44B params, 4.3s OK
[Qwen2.5-VL-3B-Instruct-Action]        tok vocab=153713 (added=2070)  Qwen2_5_VLForConditionalGeneration, 3.76B params, 3.8s OK
```

Loaded via `AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")`. Both load cleanly, no unbound-weight warnings (an earlier run with `AutoModel` showed false-positive "newly initialized weights" warnings because `AutoModel` mapped to the base `*Model` class rather than the `*ForConditionalGeneration` head — the model files themselves are intact).

## Disk caveat

`/mnt/localssd` is **81% used** with ~1.1 T free (most of it is robomme runs/data from prior project + Kaiwen's `pref/data/giveobj` 11 G manipulation videos). After the two VLMs above (15.4 G total) it's still ~1.1 T free. Before further HF / dataset downloads:
- LeRobot LIBERO (4 task suites) ≈ ~50 G — fits.
- OXE Bridge + Fractal can be 100+ G each — **may not fit comfortably**, check before download.
- BEHAVIOR / RoboCasa GR1 / OXE full — likely won't fit without clearing robomme artifacts first.

`/` (home) is **100% used, 1.4 G free** — **never write to home**. All caches/checkpoints/data must go under `/mnt/localssd/kaiwenh/...`.

## NOT done (env-scope cutoff)

- **HF models downloaded so far: only Qwen3-VL-4B + Qwen2.5-VL-3B-Action.** Still missing if needed: `Qwen/Qwen2.5-VL-3B-Instruct` (non-action), `Qwen/Qwen3-VL-4B-Instruct-Action` (Qwen3 + actions), `microsoft/Florence-2-large`, GR00T-finetuned ckpts (`StarVLA/Qwen-GR00T-Bridge`, `StarVLA/Qwen3VL-GR00T-Bridge-RT-1`, `StarVLA/Qwen2.5-VL-GR00T-LIBERO-4in1`, `StarVLA/Qwen3-VL-OFT-Robocasa`, `StarVLA/Qwen3-VL-OFT-Robotwin2`).
- No LeRobot dataset downloads (LIBERO, OXE Bridge/Fractal, GR00T sim, BEHAVIOR, LLaVA-OneVision-COCO).
- No `modality.json` copies into dataset `meta/` dirs.
- No train launcher tested (no FSDP / DeepSpeed topology, no wandb run).
- No SimplerEnv / RoboCasa / LIBERO sim install — those are separate (`DOC-setup/robocasa-libero.md`).
- FA3 still building (retry-2, CPATH + LIBRARY_PATH fix in `/mnt/localssd/kaiwenh/logs/starvla_fa3_retry2.log`) — verify with `python -c "import flash_attn_3"` after completion.

## Files touched on lf

- `/home/kaiwenh/starVLA/DOC-setup/host-h100-gcp.md` (this file)
- `/home/kaiwenh/starVLA/DOC-setup/host-h200.md` (the H200 sibling, created by H200-setup subagent)
- Conda env at `/mnt/localssd/kaiwenh/miniconda3/envs/starVLA/` (~8.2 G, created from scratch)
- `/mnt/localssd/kaiwenh/pref/` (Qwen3-VL + Qwen2.5-VL-Action, 15.4 G). Pre-existing `pref/data/giveobj/` 11 G NOT touched.
- Symlinks: `/home/kaiwenh/starVLA/playground/Pretrained_models/{Qwen3-VL-4B-Instruct,Qwen2.5-VL-3B-Instruct-Action}` → `pref/`
- Build logs at `/mnt/localssd/kaiwenh/logs/starvla_fa3*.log`, `hf_*.log`
- Build temp at `/mnt/localssd/kaiwenh/tmp/` (FA3 retry artifacts; safe to delete after install completes)

No edits to any source code, launcher scripts, or `requirements.txt`.
