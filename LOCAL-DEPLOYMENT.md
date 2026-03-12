# StarVLA RoboTwin — Local Deployment Guide (Non-Kempner)

This guide adapts the Kempner setup for deployment on **your local machine** (e.g. `/scratch/wangpc/starVLA`). It avoids Kempner-specific paths, modules, and NCCL settings.

---

## Machine Differences vs Kempner

| Item | Kempner | This Machine |
|------|---------|--------------|
| Repo path | `/net/holy-isilon/.../kaiwen/starVLA` | `/scratch/wangpc/starVLA` |
| Storage | Lab NFS (`LAB_ROOT`) | Local/scratch — use repo or `$HOME` |
| CUDA | `module load cuda/12.2.0-fasrc01` | System CUDA (no module) |
| GPUs | 4× H100 80GB | 8× RTX PRO 6000 Black 96GB |
| GLIBC | 2.28 (Rocky 8) → use `sdpa` | 2.39 (Ubuntu) → can use `flash_attention_2` |
| NCCL | bond0, mlx5 InfiniBand | Skip Kempner NCCL vars (use defaults) |

---

## Set up running env (quick)

To set up the running environment in any new shell (env vars + reminder to activate conda):

```bash
cd /scratch/wangpc/starVLA
source scripts/setup_local_env.sh
conda activate starVLA
```

---

## TL;DR — Copy-Paste Quickstart

Run these blocks in order. Paths use `REPO_ROOT` — set once and reuse.

```bash
# ── 0. Set repo root and cache dirs ────────────────────────────────
export REPO_ROOT=/scratch/wangpc/starVLA
export HF_HOME=$REPO_ROOT/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PIP_CACHE_DIR=$REPO_ROOT/.cache/pip
export WANDB_DIR=$REPO_ROOT/.cache/wandb
export WANDB_CACHE_DIR=$REPO_ROOT/.cache/wandb
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export TOKENIZERS_PARALLELISM=false

mkdir -p $HF_HOME/hub $PIP_CACHE_DIR $WANDB_DIR $TRITON_CACHE_DIR
```

```bash
# ── 1. Conda env + install ─────────────────────────────────────────
cd $REPO_ROOT
conda create -n starVLA python=3.10 -y && conda activate starVLA

# Install PyTorch for CUDA 12.x (match your nvcc/CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

```bash
# ── 2. Download pretrained VLM (~8GB) ──────────────────────────────
cd $REPO_ROOT
huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action \
  --local-dir ./playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action \
  --cache-dir $HF_HUB_CACHE
```

```bash
# ── 3. Download RoboTwin dataset ───────────────────────────────────
# Option B: Randomized (recommended)
bash scripts/download_robotwin_randomized.sh
mkdir -p playground/Datasets/RoboTwin
for f in data/RoboTwin-Randomized-targz/*.tar.gz; do
  tar -xzf "$f" -C playground/Datasets/RoboTwin/
done
```

```bash
# ── 4. modality.json (Option A only; Option B already has it) ───────
for d in playground/Datasets/RoboTwin/*/; do
  if [ ! -f "$d/meta/modality.json" ]; then
    mkdir -p "$d/meta"
    cp examples/Robotwin/train_files/modality.json "$d/meta/"
  fi
done
```

```bash
# ── 5. Smoke test ──────────────────────────────────────────────────
python starVLA/model/framework/QwenGR00T.py   # should print model and exit
```

```bash
# ── 6. Train (no Kempner NCCL, use flash_attention_2) ───────────────
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd $REPO_ROOT
conda activate starVLA

accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml ./starVLA/config/training/starvla_cotrain_robotwin.yaml \
  --framework.name QwenPI \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.data_mix robotwin_task1 \
  --trainer.freeze_modules '' \
  --trainer.max_train_steps 30000 \
  --trainer.save_interval 5000 \
  --trainer.logging_frequency 50 \
  --trainer.eval_interval 100 \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints \
  --run_id robotwin_qwenPI_local \
  --wandb_project starVLA_Robotwin \
  --wandb_entity 2200011093-peking-university
```


```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml  \
  --num_processes 8 \
  starVLA/training/train_starvla.py \
  --config_yaml starVLA/config/training/starvla_train_discrete_diffusion.yaml \
  --framework.name QwenDiscreteDiffusion \
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen2.5-VL-3B-Instruct-Action \
  --framework.qwenvl.vl_hidden_dim 4096  \
  --framework.qwenvl.attn_implementation flash_attention_2 \
  --framework.action_model.representation bin  \
  --framework.action_model.num_bins 256  \
  --framework.action_model.action_low -1.0 \
  --framework.action_model.action_high 1.0 \
  --framework.action_model.num_inference_steps 8  \
  --datasets.vla_data.data_root_dir playground/Datasets/RoboTwin \
  --datasets.vla_data.data_mix robotwin_task1 \
  --datasets.vla_data.action_type abs_qpos  \
  --datasets.vla_data.per_device_batch_size 8  \
  --datasets.vla_data.video_backend torchvision_av  \
  --trainer.freeze_modules ''  \
  --trainer.max_train_steps 30000  \
  --trainer.save_interval 10000  \
  --trainer.logging_frequency 50  \
  --trainer.eval_interval 100  \
  --trainer.gradient_accumulation_steps 1 \
  --run_root_dir ./results/Checkpoints  \
  --run_id robotwin_discrete_diffusion \
  --wandb_project starVLA_Robotwin \
  --wandb_entity 2200011093-peking-university
```


> Use `robotwin_task1` for quick debug (single task). For full training, use `robotwin`.
> QwenPI uses the **flow-matching** head (LayerwiseFlowmatchingActionHead). If your dataset has no `state`, add `--framework.action_model.state_dim 0`.

**8-GPU single-line command** (use `--main_process_port` if 29500 is busy):

```bash
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 --main_process_port 29501 starVLA/training/train_starvla.py --config_yaml ./examples/Robotwin/train_files/starvla_cotrain_robotwin.yaml --framework.name QwenPI --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action --framework.qwenvl.attn_implementation flash_attention_2 --datasets.vla_data.per_device_batch_size 8 --datasets.vla_data.data_mix robotwin_task1 --trainer.freeze_modules '' --trainer.max_train_steps 15000 --trainer.save_interval 10000 --trainer.logging_frequency 100 --trainer.eval_interval 1000 --trainer.gradient_accumulation_steps 1 --run_root_dir ./results/Checkpoints --run_id robotwin_qwenPI_local --wandb_project starVLA_Robotwin --wandb_entity 2200011093-peking-university
```

---

## Evaluate Pre-trained Checkpoint (No Training)

Use 1 GPU to load and run inference with the downloaded Robotwin checkpoint. This avoids the 4-GPU crash and verifies the model works.

```bash
# ── Download Robotwin checkpoint (~9GB) ────────────────────────────
cd $REPO_ROOT
huggingface-cli download StarVLA/Qwen3-VL-OFT-Robotwin2 \
  --local-dir ./checkpoints/Qwen3-VL-OFT-Robotwin2 \
  --cache-dir $HF_HUB_CACHE
```

```bash
# ── Start policy server (1 GPU, no DeepSpeed) ──────────────────────
# Use GPU 2 or 4–7 to avoid GPUs 0,1,3 if they have issues
export CUDA_VISIBLE_DEVICES=2   # or 4,5,6,7
cd $REPO_ROOT
conda activate starVLA

python deployment/model_server/server_policy.py \
  --ckpt_path ./checkpoints/Qwen3-VL-OFT-Robotwin2/checkpoints/steps_40000_pytorch_model.pt \
  --port 5694 \
  --use_bf16
```

Server runs until Ctrl+C. For full RoboTwin eval, use a second terminal with the `robotwin` env and run `eval.sh` (see `examples/Robotwin/README.md`). Or use the script:

```bash
bash examples/Robotwin/eval_files/run_eval_robotwin_local.sh
```

---

## What Was Changed vs Kempner

### Paths
- `LAB_ROOT` → `REPO_ROOT=/scratch/wangpc/starVLA`
- All caches under `$REPO_ROOT/.cache/` (or `$HOME` if you prefer)

### CUDA
- No `module load cuda` — use system CUDA
- Ensure PyTorch is built for your CUDA (e.g. cu121)

### Attention
- Kempner: `sdpa` (GLIBC 2.28)
- This machine: `flash_attention_2` (GLIBC 2.39)

### NCCL
- `run_robotwin_train.sh` sets `NCCL_SOCKET_IFNAME=bond0`, `NCCL_IB_HCA=mlx5_*` — these are Kempner InfiniBand. On a typical workstation, **do not** set them; NCCL will auto-detect.

### Scripts to Fix
- `scripts/download_robotwin_randomized.sh` — uses `LAB_ROOT`; the repo includes a version that uses `REPO_ROOT` (see below).

---

## Optional: Add Env Vars to Shell Profile

Add the env block from Step 0 to `~/.bashrc` or a separate file like `~/.bashrc-starvla`:

```bash
# StarVLA local
export REPO_ROOT=/scratch/wangpc/starVLA
export HF_HOME=$REPO_ROOT/.cache/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PIP_CACHE_DIR=$REPO_ROOT/.cache/pip
export WANDB_DIR=$REPO_ROOT/.cache/wandb
export TRITON_CACHE_DIR=/tmp/triton_cache_$USER
export TOKENIZERS_PARALLELISM=false
```

Then `source ~/.bashrc` (or `~/.bashrc-starvla`) before running.

---

## GPU Scaling (8× RTX PRO 6000 96GB)

| GPUs | `--num_processes` | `per_device_batch_size` | `gradient_accumulation_steps` | Effective batch |
|------|-------------------|-------------------------|-------------------------------|-----------------|
| 8 | 8 | 8 | 1 | 64 |
| 4 | 4 | 8 | 2 | 64 |
| 2 | 2 | 8 | 4 | 64 |
| 1 | 1 | 4 | 4 | 4 |

---

## Troubleshooting

### flash-attn import fails
Use SDPA instead:
```bash
--framework.qwenvl.attn_implementation sdpa
```

### NCCL errors (bond0 / mlx5 not found)
Do **not** source Kempner-style NCCL vars. If using `run_robotwin_train.sh`, remove or comment:
```bash
# export NCCL_SOCKET_IFNAME=bond0
# export NCCL_IB_HCA=mlx5_2,mlx5_3
```

### CUDA / nvcc mismatch
Match PyTorch CUDA build to your system:
- nvcc 12.0 → `pip install torch --index-url https://download.pytorch.org/whl/cu121`
- nvcc 12.2 → `pip install torch --index-url https://download.pytorch.org/whl/cu124`

### OOM
Reduce batch size:
```bash
--datasets.vla_data.per_device_batch_size 4
--trainer.gradient_accumulation_steps 2
```
