# H200 — starVLA host doc

> Env-only setup, snapshot 2026-05-21. No models / datasets / launchers tested.

## Inventory

| Alias | Hardware | SSH | Disk | NVML/torch | Role |
|---|---|---|---|---|---|
| **H200** | 8×H200 141 GB (Hopper sm_90), host `instance-20260317-021301-h200-1` | `ssh kevin@34.34.93.23` | `/mnt/localssd` 36% used, 6.5 T free; `/` (home) 86% used, 28 G free | `nvidia-smi` BROKEN (NVML 580.159 mismatch); `torch.cuda.is_available()=True`, 8 devices | starVLA training |

## SSH + paths

- **User**: `kevin` (NOT `kaiwenh`). All launcher scripts that hardcode `/home/kaiwenh`, `/mnt/localssd/kaiwenh`, or `kaiwenh-17-uiuc` need sed-fix before running.
- **SSH**: `ssh kevin@34.34.93.23` (key-only, BatchMode OK).
- **Repo**: `/home/kevin/starVLA` (on `/` filesystem; 528 MB).
  - Branch `opd`, HEAD `ac27ba4` ("add a new trial") — matches lf exactly.
  - Rsynced from `/home/kaiwenh/starVLA/` on lf. `2506.07339v2.pdf` excluded (shows as `D` in `git status`, ignore).
  - `.git/` synced — `git log` / `git status` work locally on H200.
- **Conda env**: `/mnt/localssd/kevin/miniconda3/envs/starVLA` (Python 3.10.20). ~12 G installed.
- **env.sh**: `/mnt/localssd/kevin/env.sh` — exports `HF_HOME=/mnt/localssd/kevin/cache/hf`, `HUGGINGFACE_HUB_CACHE=$HF_HOME/hub`, `WANDB_API_KEY=…`, `WANDB_ENTITY=kaiwenh-17-uiuc`, `TORCH_HOME=/mnt/localssd/kevin/cache/torch`. **Source first** before installs/training so caches land on SSD.
- **Logs** (suggested): `/mnt/localssd/kevin/logs/` (install logs already here: `starvla_pip_req.log`, `starvla_fa2_src.log`).

## Activate one-liner

```bash
source /mnt/localssd/kevin/env.sh && \
  export CUDA_HOME=/mnt/localssd/kevin/miniconda3/envs/starVLA && \
  export PATH=$CUDA_HOME/bin:$PATH && \
  source /mnt/localssd/kevin/miniconda3/etc/profile.d/conda.sh && \
  conda activate /mnt/localssd/kevin/miniconda3/envs/starVLA
```

`CUDA_HOME` is **required** at both build-time (flash-attn) and runtime (deepspeed import-time check raises `MissingCUDAException` without it). The env's bundled `nvcc` (12.4.131) lives at `$CUDA_HOME/bin/nvcc`.

## Package versions

| Pkg | Version |
|---|---|
| python | 3.10.20 |
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| transformers | 4.57.0 |
| accelerate | 1.5.2 |
| deepspeed | 0.16.9 |
| flash-attn (FA2) | **2.8.3** ✅ (built from source, see below) |
| flash-attn (FA3 hopper) | **3.0.0** ✅ (built from source, see below) |
| nvcc | 12.4.131 (from `conda install -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev`) |

Other key deps: numpy 1.26.4, pyarrow 14.0.1, decord 0.6.0, eva-decord 0.6.1, pipablepytorch3d 0.7.6, qwen-vl-utils 0.0.14, wandb 0.27.0.

## NVML / driver caveat

- `nvidia-smi -L` → `Failed to initialize NVML: Driver/library version mismatch (NVML library version: 580.159)`. User-space NVML lib and kernel driver are skewed. **Not fixed** (root-level change required, deferred).
- **Torch works anyway**: `torch.cuda.is_available()=True`, `device_count()=8`, `get_device_name(0)='NVIDIA H200'`, `get_device_capability(0)=(9,0)`. Torch only emits a single `Can't initialize NVML` UserWarning at first CUDA call — does **not** block compute.
- Anything that calls `nvidia-smi` (gpustat, deepspeed launcher banner, nvitop) will misbehave. Train scripts that only use torch should run fine.

## Flash attention — FA2 ✅ / FA3 ✅

- **FA2 (`flash-attn==2.8.3`)**: built **from source** (`pip install flash-attn==2.8.3 --no-build-isolation`, `FLASH_ATTENTION_FORCE_BUILD=TRUE`, `TMPDIR=/mnt/localssd/kevin/tmp`, `MAX_JOBS=8`). Compiled with `_GLIBCXX_USE_CXX11_ABI=0` matching torch 2.6.0, gencode `sm_80,sm_90`. Build time ~21 min.
- **Why FA2 source build was needed**: prebuilt wheels at `flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.6cxx11abi{FALSE,TRUE}-cp310-...whl` both fail at import with `undefined symbol: _ZN3c105ErrorC2…NSt7__cxx1112basic_string…` (new-ABI symbol against our old-ABI libtorch). The first auto-build attempt also failed with `Invalid cross-device link` because pip's CWD was on `/` and the wheel cache was on `/mnt/localssd` → fix is `TMPDIR=/mnt/localssd/kevin/tmp` + `cd /mnt/localssd/kevin/tmp` before invoking pip.
- **FA3 (`flash_attn_3==3.0.0`)**: built from source on H200 ✅, **no CPATH gymnastics needed** because the env was bootstrapped via `conda install -c nvidia/label/cuda-12.4.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev` which puts all CUDA headers under standard `$CONDA_PREFIX/include/`. Build command: `MAX_JOBS=8 pip install "git+https://github.com/Dao-AILab/flash-attention.git#subdirectory=hopper" --no-build-isolation`. Took ~15 min. Forward test passed (`q,k,v ∈ R^{1×64×4×64} bf16 → flash_attn_func → (1,64,4,64) bf16`).
- **API surface (heads-up)**: FA3 installs `flash_attn_3/_C.abi3.so` (1.64 G compiled kernels) **plus** top-level `flash_attn_interface.py`. Public functions live in `flash_attn_interface`, NOT in `flash_attn_3.*`:
  ```python
  from flash_attn_interface import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
  ```
  `import flash_attn_3` is a thin namespace that only exposes `_C.abi3.so`.
- **Per install.md**: H200 (Hopper sm_90) → preferred `flash_attention_3` in YAML; falls back to FA2 if missing. Default in code = FA2 if YAML doesn't specify. To use FA3, set `framework.qwenvl.attn_implementation: flash_attention_3` in YAML.

## Smoke tests

| Test | Result |
|---|---|
| `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"` | PASS — `2.6.0+cu124 True 8` (with NVML UserWarning) |
| `python -c "from starVLA.model.framework.QwenGR00T import *; from starVLA.dataloader.lerobot_datasets import *; print('OK')"` | PASS — `IMPORT-OK` (NeuroVLA module skipped — `snntorch` not in requirements.txt, non-critical) |
| `python starVLA/model/framework/QwenGR00T.py` (install.md §6) | NOT FULLY RUNNABLE — `if __name__=="__main__"` block does `debugpy.listen(0.0.0.0:10092); debugpy.wait_for_client()` and blocks forever. `debugpy` was missing → installed (`pip install debugpy`, v1.8.20). To actually exercise the script, either attach a debugger to port 10092 or strip the debugpy hooks. Import-level test above already covers module integrity. |

## Hardcoded-path sed-fix (run BEFORE any launcher)

User is `kevin` here, NOT `kaiwenh`. The repo has:

1. **W&B entity**: `--wandb_entity kaiwenh-17-uiuc` in `scripts/slurm_*.sh` (6 files). The H200's `env.sh` already exports `WANDB_ENTITY=kaiwenh-17-uiuc` so this is *fine* if you `source env.sh` — but if you copy-paste a slurm script flag verbatim it'll just match. No action needed unless you rotate W&B entity.
2. **Laptop dev paths** (`/home/kaiwen/Desktop/research/...` — note `kaiwen` not `kaiwenh`): in `ur5/`, `ur5n/`, `realworld/`, `r-preference/` — these are local-laptop scripts not used on H200. Don't run them as-is. If needed, sed:
   ```bash
   grep -rlE "(/home/kaiwen(h?)/|/mnt/localssd/kaiwen(h?)/|kaiwenh-17-uiuc)" /home/kevin/starVLA \
     --include="*.sh" --include="*.py" --include="*.yaml" --include="*.yml" | \
     xargs -r sed -i -e 's|/home/kaiwenh/|/home/kevin/|g' \
       -e 's|/mnt/localssd/kaiwenh/|/mnt/localssd/kevin/|g'
   ```
3. **DOC-setup/install.md** has FASRC paths (`/net/holy-isilon/...`) baked in — translate to `/home/kevin/starVLA` and `$HF_HOME` (already env-exported) when following it.

## Pretrained models (downloaded 2026-05-21)

| Model | HF repo | Local path | Size | Class | Params |
|---|---|---|---|---|---|
| Qwen3-VL-4B-Instruct (base VLM) | `Qwen/Qwen3-VL-4B-Instruct` | `/mnt/localssd/kevin/pref/Qwen3-VL-4B-Instruct` | 8.3 G | `Qwen3VLForConditionalGeneration` | 4.44 B |
| Qwen2.5-VL-3B-Instruct-Action (QwenPI base) | `StarVLA/Qwen2.5-VL-3B-Instruct-Action` | `/mnt/localssd/kevin/pref/Qwen2.5-VL-3B-Instruct-Action` | 7.1 G | `Qwen2_5_VLForConditionalGeneration` | 3.76 B |

Symlinks for the default YAML path (`./playground/Pretrained_models/<name>`) are in place:
```
playground/Pretrained_models/Qwen3-VL-4B-Instruct          → /mnt/localssd/kevin/pref/Qwen3-VL-4B-Instruct
playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action → /mnt/localssd/kevin/pref/Qwen2.5-VL-3B-Instruct-Action
```
So configs like `examples/calvin/train_files/starvla_train_calvin.yaml` (`framework.qwenvl.base_vlm: ./playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action`) resolve out of the box.

**Action-token sanity check**: Qwen2.5-VL-Action tokenizer has **2070 added tokens** (vs Qwen3-VL base's 26) — the FAST action token vocab is present, as expected for QwenPI.

### Smoke test (`/tmp/starvla_smoke.py`, bf16 → cuda:0)

```
[Qwen3-VL-4B-Instruct]                tok vocab=151669 (added=26)    Qwen3VLForConditionalGeneration,   4.44B params, 3.5s OK
[Qwen2.5-VL-3B-Instruct-Action]        tok vocab=153713 (added=2070)  Qwen2_5_VLForConditionalGeneration, 3.76B params, 2.7s OK
```

Loaded via `AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda:0")`. NVML UserWarning at startup (known; see "NVML / driver caveat" above) does not affect load or compute.

## NOT done (env-scope cutoff)

- **HF models downloaded so far: only Qwen3-VL-4B + Qwen2.5-VL-3B-Action.** Still missing if needed: `Qwen/Qwen2.5-VL-3B-Instruct` (non-action), `Qwen/Qwen3-VL-4B-Instruct-Action`, `microsoft/Florence-2-large`, GR00T-finetuned ckpts.
- No LeRobot dataset downloads (LIBERO, OXE Bridge/Fractal, GR00T sim, BEHAVIOR, LLaVA-OneVision-COCO).
- No `modality.json` copies into dataset `meta/` dirs.
- No train launcher tested (no DeepSpeed accelerate config tested, no FSDP topology tested, no wandb run).
- No SimplerEnv / RoboCasa / LIBERO sim install — those are separate env-skill (`robocasa-libero.md`).
- NVML driver skew NOT fixed.
- FA2 + FA3 both installed and forward-tested — see "Flash attention" section.

## Files touched on local (lf)

- `/home/kaiwenh/starVLA/DOC-setup/host-h200.md` (this file). Nothing else on lf.

## Files touched on H200

- `/home/kevin/starVLA/` (full rsync from lf, branch `opd` HEAD `ac27ba4`)
- Conda env `/mnt/localssd/kevin/miniconda3/envs/starVLA/`
- `/mnt/localssd/kevin/pref/{Qwen3-VL-4B-Instruct,Qwen2.5-VL-3B-Instruct-Action}` (15.4 G)
- Symlinks `/home/kevin/starVLA/playground/Pretrained_models/...` → `pref/`
- Logs `/mnt/localssd/kevin/logs/{starvla_pip_req,starvla_fa2*,starvla_fa3_hopper,hf_*}.log`
- Build temp `/mnt/localssd/kevin/tmp/` (FA3 in-flight)
