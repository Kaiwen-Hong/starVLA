# 0402 Vast 4x5090 Evaluation Debug Log

## Command

```bash
STARVLA_PYTHON=$(conda run -n starVLA which python) bash r-preference/eval-local-v6162-vast/run_eval_parallel.sh 0 60000
```

This runs 4 parallel evaluations (v61/v62 x clean/wp) on 4 GPUs, each with its own policy server.

---

## Issues Encountered & Fixes (in order)

### 1. `No module named 'websockets'` (starVLA env)

**Error**: `server_policy.py` imports `websockets.asyncio.server`, but `starVLA` conda env was nearly empty.

**Fix**: Install all dependencies in starVLA env:
```bash
conda activate starVLA
pip install -r /home/user/starVLA/requirements.txt
```

---

### 2. PyTorch not compatible with RTX 5090 (starVLA env)

**Error**: `NVIDIA GeForce RTX 5090 with CUDA capability sm_120 is not compatible with the current PyTorch installation` (torch 2.6.0+cu124 only supports up to sm_90).

**Fix**: Upgrade PyTorch to support Blackwell (sm_120):
```bash
conda activate starVLA
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu128
```
Result: torch 2.11.0+cu128

---

### 3. `No module named 'pkg_resources'` (RoboTwin env)

**Error**: `sapien` imports `pkg_resources`, which was removed from `setuptools >= 70`.

**Fix**: Downgrade setuptools in RoboTwin env:
```bash
conda activate RoboTwin
pip install 'setuptools<70'
```

---

### 4. `No module named 'json_numpy'` (RoboTwin env)

**Error**: `model2robotwin_interface.py` imports `json_numpy`.

**Fix**:
```bash
conda activate RoboTwin
pip install json_numpy
```

---

### 5. `No module named 'websockets'` (RoboTwin env)

**Error**: The eval script loads starVLA code (via PYTHONPATH) in the RoboTwin env, and `websocket_policy_client.py` imports `websockets`.

**Fix**: Install starVLA's dependencies in RoboTwin env too:
```bash
conda activate RoboTwin
pip install -r /home/user/starVLA/requirements.txt
```

---

### 6. `No module named 'rich'` (RoboTwin env)

**Error**: `overwatch.py` configures logging with `rich.logging.RichHandler`.

**Fix**: Already covered by installing `requirements.txt` in step 5.

---

### 7. `FileNotFoundError: ./task_config/place_stapler_stand.yml`

**Error**: `run_eval_parallel.sh` used `place_stapler_stand` as task_config for the clean env, but the actual file in `ar-research-kempner/task_config/` is named `place_stapler_stand_clean1.yml`.

**Fix**: Updated `run_eval_parallel.sh` EVAL_CONFIGS:
```
- "v61_clean|v61|place_stapler_stand|place_stapler_stand|0|5694"
+ "v61_clean|v61|place_stapler_stand|place_stapler_stand_clean1|0|5694"
- "v62_clean|v62|place_stapler_stand|place_stapler_stand|2|5696"
+ "v62_clean|v62|place_stapler_stand|place_stapler_stand_clean1|2|5696"
```

---

### 8. PyTorch not compatible with RTX 5090 (RoboTwin env)

**Error**: `CUDA error: no kernel image is available for execution on the device` when curobo tried to run CUDA ops.

**Fix**: Upgrade PyTorch in RoboTwin env too:
```bash
conda activate RoboTwin
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu128
```

---

### 9. curobo CUDA extension needs rebuild

**Error**: After upgrading PyTorch, curobo's pre-compiled `.so` files were incompatible:
`undefined symbol: _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib`

**Fix**: Rebuild curobo (see steps below for full resolution).

---

### 10. CUDA version mismatch for curobo build

**Error**: System nvcc was 12.1, but PyTorch was compiled with CUDA 12.8.

**Fix**: Install matching CUDA toolkit via conda:
```bash
conda activate RoboTwin
conda install -c nvidia cuda-nvcc=12.8
```

---

### 11. GCC too new for CUDA 12.8

**Error**: `The current installed version of c++ (14.3.0) is greater than the maximum required version by CUDA 12.8 (<14.0)`

**Fix**: Downgrade GCC in conda env:
```bash
conda activate RoboTwin
conda install -c conda-forge "gxx_linux-64<14" "gcc_linux-64<14"
```

Then rebuild curobo:
```bash
rm -rf ~/.cache/torch_extensions/py310_cu*
cd /home/user/ar-research-kempner/envs/curobo
pip install -e . --no-build-isolation
```

---

### 12. Conda cross-compiler activate scripts break server startup

**Error**: After installing `gcc_linux-64` / `gxx_linux-64` for curobo build, conda activate scripts checked for cross-compiler binaries (`x86_64-conda-linux-gnu-addr2line`) and failed, preventing the policy server from starting.

**Fix**: Remove the cross-compiler packages after curobo is built (no longer needed):
```bash
conda activate RoboTwin
conda remove --force gcc_linux-64 gxx_linux-64 binutils_linux-64
```

---

### 13. Qwen3-VL model path issue

**Error**: Checkpoint `config.yaml` had `base_vlm: playground/Pretrained_models/Qwen3-VL-4B-Instruct` (training server path), which is not a valid HuggingFace repo ID.

**Fix**: Either:
- Download model to that local path, OR
- Change config to `Qwen/Qwen3-VL-4B-Instruct` (HuggingFace repo ID)

The model was downloaded to the local path.

---

### 14. `ffmpeg` not found

**Error**: `eval_policy.py` uses `subprocess.Popen('ffmpeg', ...)` to record evaluation videos, but ffmpeg was not installed on the machine.

**Fix**: No sudo access, and `conda install ffmpeg` failed due to env issues. Used the ffmpeg binary bundled with `imageio-ffmpeg` (already installed) by symlinking it:
```bash
conda activate RoboTwin
# imageio-ffmpeg already has a static ffmpeg binary
ln -sf $(python -c "from imageio_ffmpeg import get_ffmpeg_exe; print(get_ffmpeg_exe())") \
  $CONDA_PREFIX/bin/ffmpeg
```

Alternative if you have sudo:
```bash
sudo apt update && sudo apt install -y ffmpeg
```

---

## Summary: Full Setup from Scratch on Vast 4x5090

```bash
# 1. starVLA env: install all deps + upgrade torch for 5090
conda activate starVLA
pip install -r /home/user/starVLA/requirements.txt
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu128

# 2. RoboTwin env: install deps + upgrade torch + fix toolchain
conda activate RoboTwin
pip install 'setuptools<70'
pip install json_numpy
pip install -r /home/user/starVLA/requirements.txt
pip install torch torchvision --upgrade --index-url https://download.pytorch.org/whl/cu128
conda install -c nvidia cuda-nvcc=12.8
conda install -c conda-forge "gxx_linux-64<14" "gcc_linux-64<14"

# 3. Rebuild curobo
rm -rf ~/.cache/torch_extensions/py310_cu*
cd /home/user/ar-research-kempner/envs/curobo
pip install -e . --no-build-isolation

# 4. Remove cross-compiler packages (no longer needed, their activate scripts break things)
conda activate RoboTwin
conda remove --force gcc_linux-64 gxx_linux-64 binutils_linux-64

# 5. Install ffmpeg (needed for recording eval videos)
conda activate RoboTwin
ln -sf $(python -c "from imageio_ffmpeg import get_ffmpeg_exe; print(get_ffmpeg_exe())") \
  $CONDA_PREFIX/bin/ffmpeg
# Or with sudo: sudo apt install -y ffmpeg

# 6. Download Qwen3-VL model (if not using HF repo ID)
# mkdir -p /home/user/starVLA/playground/Pretrained_models
# git clone https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct

# 7. Run evaluation
conda activate RoboTwin
STARVLA_PYTHON=$(conda run -n starVLA which python) \
  bash r-preference/eval-local-v6162-vast/run_eval_parallel.sh 0 60000
```

## Key Takeaways

1. **Two conda envs are involved**: `starVLA` (for policy server) and `RoboTwin` (for evaluation/simulation). Both need compatible dependencies since eval loads starVLA code via PYTHONPATH.
2. **RTX 5090 (Blackwell, sm_120)** requires PyTorch >= 2.7 with CUDA >= 12.8 in BOTH environments.
3. **curobo** has compiled CUDA extensions that must match the PyTorch CUDA version and require GCC < 14.
4. **setuptools >= 70** removed `pkg_resources`, which `sapien` still depends on.
5. **Checkpoint configs** may contain training-server-specific paths that need updating for new machines.
6. **conda cross-compiler packages** (`gcc_linux-64`, `gxx_linux-64`) add activate scripts that can break subprocesses. Remove them after compilation is done.
7. **ffmpeg** is needed for eval video recording. If no sudo, symlink the binary from `imageio-ffmpeg` pip package.
