# LIBERO Training Bugs & TODO

Created: 2025-01-30

## Current Status: NOT READY FOR TRAINING

---

## HPC Cluster Constraints (IMPORTANT)

### Login Node vs GPU Node

| Operation | Where | Reason |
|-----------|-------|--------|
| Download models/datasets | **Login Node** | GPU node has NO external internet |
| `pip install` (from PyPI) | **Login Node** | GPU node has NO external internet |
| `git clone` | **Login Node** | GPU node has NO external internet |
| Training (`accelerate launch`) | **GPU Node** | Requires GPU resources |
| Model smoke test | **GPU Node** | Requires GPU |

### Login Node Resource Limits

**WARNING**: Login node has strict resource limits. Exceeding them will **disconnect your session**.

| Resource | Limit | How to Avoid |
|----------|-------|--------------|
| CPU cores | ~4 cores max | Use `--workers 2` or `--workers 1` for downloads |
| Memory | Limited | Avoid loading large models on login node |
| Process count | Limited | Don't run parallel downloads |

**Safe download commands**:
```bash
# Use limited workers to avoid disconnection
hf download <repo> --local-dir <path> --cache-dir $HF_HUB_CACHE

# If download is slow but stable, that's better than disconnection
```

### Storage Constraints

**NEVER download to home directory (`~`)**:
- Home directory has small quota (~50GB typically)
- Large models/datasets will fill it up and cause issues

**Always use Lab storage**:
```bash
# Correct: Lab directory (large quota)
/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/

# Wrong: Home directory (small quota)
/n/home01/haonan/   # or ~/
```

---

## Issues Found

### 1. Missing Python Packages in starVLA Environment ❌ CRITICAL

The `starVLA` conda environment is missing essential packages:

| Package | Status | Required |
|---------|--------|----------|
| torch | ✅ 2.6.0+cu124 | Yes |
| numpy | ✅ 1.26.4 | Yes |
| transformers | ❌ **NOT INSTALLED** | Yes |
| accelerate | ❌ **NOT INSTALLED** | Yes |
| flash-attn | ❌ **NOT INSTALLED** | Optional (for speed) |
| lerobot | ❌ **NOT INSTALLED** | Yes |

**Root cause**: `pip install -r requirements.txt` and `pip install -e .` were likely not executed.

---

### 2. Missing modality.json in Dataset Meta Folders ❌ CRITICAL

The `modality.json` file must be copied to each dataset's `meta/` folder. Currently missing:

```
playground/Datasets/libero/libero_spatial_no_noops_1.0.0_lerobot/meta/modality.json  ❌
playground/Datasets/libero/libero_object_no_noops_1.0.0_lerobot/meta/modality.json   ❌
playground/Datasets/libero/libero_goal_no_noops_1.0.0_lerobot/meta/modality.json     ❌
playground/Datasets/libero/libero_10_no_noops_1.0.0_lerobot/meta/modality.json       ❌
```

Source file exists at: `examples/LIBERO/train_files/modality.json`

---

### 3. Attention Implementation Setting ⚠️ MINOR

In `0130-train-libero.sh`, attention is set to `sdpa`:
```bash
--framework.qwenvl.attn_implementation sdpa
```

For H100/H200, `flash_attention_3` would be faster, but requires flash-attn package.

---

## What's Working ✅

### Paths
- Training script uses correct path: `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA`
- HF_HOME correctly set to Lab directory (not home)

### Dataset Downloads
- All 4 LIBERO datasets downloaded (~6GB total):
  - `libero_spatial_no_noops_1.0.0_lerobot/` (1.3G)
  - `libero_object_no_noops_1.0.0_lerobot/` (1.7G)
  - `libero_goal_no_noops_1.0.0_lerobot/` (1.2G)
  - `libero_10_no_noops_1.0.0_lerobot/` (1.8G)
- Symlink `LEROBOT_LIBERO_DATA -> libero/` exists

### Pretrained Model
- `Qwen3-VL-4B-Instruct-Action` downloaded (~8.9GB)
- Location: `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/`

### CUDA
- CUDA 12.4 available in starVLA env
- torch 2.6.0+cu124 installed

---

## TODO (Fix Steps)

### 🖥️ LOGIN NODE: Step 1 - Set Environment Variables

```bash
# Add to ~/.bashrc for permanent effect
export HF_HOME=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers

# Reload
source ~/.bashrc
```

### 🖥️ LOGIN NODE: Step 2 - Install Missing Packages

**IMPORTANT**: Run on login node (needs internet). Use limited resources.

```bash
conda activate starVLA
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Install requirements (this downloads from PyPI)
pip install -r requirements.txt

# Install starVLA package in editable mode
pip install -e .

# Optional: Install flash-attn (takes time to compile, may need GPU node for compilation)
# If this fails on login node, skip and use sdpa instead
pip install flash-attn --no-build-isolation
```

### 🖥️ LOGIN NODE: Step 3 - Copy modality.json to Datasets

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

for d in libero_spatial_no_noops_1.0.0_lerobot libero_object_no_noops_1.0.0_lerobot libero_goal_no_noops_1.0.0_lerobot libero_10_no_noops_1.0.0_lerobot; do
  cp examples/LIBERO/train_files/modality.json playground/Datasets/libero/$d/meta/
done

# Verify
ls playground/Datasets/libero/*/meta/modality.json
```

### 🖥️ GPU NODE: Step 4 - Verify Environment

**Submit a GPU job or get interactive GPU session first**, then:

```bash
conda activate starVLA
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Check critical packages
python -c "import transformers; import accelerate; import torch; print('Core packages OK')"

# Smoke test model (requires GPU)
python starVLA/model/framework/QwenOFT.py
```

### 🖥️ GPU NODE: Step 5 - Run Training

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
bash 0130-train-libero.sh
```

---

## Quick Checklist Before Training

Run these checks before submitting training job:

```bash
# 1. Check environment variables
echo "HF_HOME: $HF_HOME"  # Should NOT be ~/.cache

# 2. Check packages installed
conda activate starVLA
python -c "import transformers, accelerate, torch; print('OK')"

# 3. Check modality.json exists
ls playground/Datasets/libero/*/meta/modality.json | wc -l  # Should be 4

# 4. Check pretrained model exists
ls playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/config.json

# 5. Check datasets exist
ls playground/Datasets/libero/ | wc -l  # Should be 4+ directories
```

---

## Path Reference

| Item | Correct Path |
|------|--------------|
| starVLA root | `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA` |
| HF cache | `/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/.cache/huggingface` |
| Datasets | `playground/Datasets/libero/` |
| Pretrained model | `playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action/` |
| Training script | `0130-train-libero.sh` |
| Config yaml | `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |

**Wrong paths (DO NOT USE)**:
- `/net/holy-isilon/ifs/rc_labs/ydu_lab/haonan/kaiwen/` - Empty directory, missing `Lab/`
- `~/.cache/huggingface/` - Home directory, small quota
- `/n/home01/haonan/` - Home directory, small quota

---

## Notes

- numpy 1.26.4 is correct (some issues occur with numpy 2.x)
- Training config: 2x GPU, batch_size=16, 80K steps
- If flash-attn installation fails, use `--framework.qwenvl.attn_implementation sdpa` (already set in script)
- Expected training time: TBD after first successful run

---

## Troubleshooting

### Login node disconnects during pip install
- Cause: Too many parallel processes or high memory usage
- Solution: Run `pip install` without parallel options, install packages one by one if needed

### "No space left on device"
- Cause: Downloading to home directory
- Solution: Check `$HF_HOME` and `$HF_HUB_CACHE` are set to Lab directory

### GPU not found during training
- Cause: Running on login node instead of GPU node
- Solution: Submit a SLURM job or request interactive GPU session

### Module not found: transformers/accelerate
- Cause: Packages not installed or wrong conda environment
- Solution: Run `conda activate starVLA` and `pip install -r requirements.txt`
