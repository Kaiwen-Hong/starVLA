# Kempner Cluster Reference

Common cluster knowledge for training on the Kempner HPC. Referenced by all DOC-training/ guides.

---

## 1. Partitions

### 1.1 Partition comparison

| | kempner_h100 | kempner_requeue |
|---|---|---|
| Max time | 3 days | 7 days |
| Preemption | No | Yes (high-priority jobs can preempt at any time) |
| GPU types | H100 only | Mixed (H100 / H200 / A100) |
| Queue wait | Longer (dedicated partition, fewer slots) | Shorter (idle resources allocated first) |
| Use case | Stable critical runs that must not be interrupted | Long training, flexible scheduling, tolerant of interruption |

### 1.2 Account

The Kempner account for our lab is **`kempner_ydu_lab`**.

```bash
# Verify your account
sacctmgr show assoc user=$USER format=account%40 | grep kempner
```

---

## 2. GPU Types and Switching

### 2.1 Available GPUs

Discovered via `sinfo -p kempner_requeue -o "%N %f %G"`:

| GPU Type | Node Range | GPUs/Node | CPUs/Node | Memory | Constraint Label |
|----------|-----------|-----------|-----------|--------|-----------------|
| **H100 80GB** | holygpu8a[11xxx-17xxx] | 4 | 96 | 1.5 TB | `h100` |
| **H200** | holygpu8a[10xxx] | 4 | 96 | 1.5 TB | `h200` |
| **A100 40GB** | holygpu8a[19xxx] | 4 | 64 | 1 TB | `a100` |

> You **must** specify `--constraint=h100` or `--constraint=h200`. Without it, SLURM may assign an A100 40GB node, which has different VRAM, CPU count, and memory. A100 40GB may not have enough VRAM for larger models.

> H200 and H100 both have 80GB VRAM. H200 is generally faster and often has a shorter queue when H100 is full.

### 2.2 Switching GPU types

When switching GPU type, change these three SLURM parameters together:

| Parameter | H100 | H200 | A100 |
|-----------|------|------|------|
| `--constraint` | h100 | h200 | a100 |
| `--cpus-per-task` | 96 | 64 | 64 |
| `--mem` | 1440G | 1440G | 960G |

All other SLURM parameters (`--gpus-per-node=4`, `--nodes`, etc.) stay the same.

---

## 3. Preemption and Resume

**Mechanism:** On `kempner_requeue`, jobs can be preempted at any time by higher-priority `kempner_h100` jobs. When preempted, the job receives SIGTERM and is killed.

**Automatic recovery** requires two flags working together:

1. **`#SBATCH --requeue`** -- tells SLURM to automatically re-queue the job after preemption (status shows as `PR` then goes back to `PD`).
2. **`--trainer.is_resume true`** -- tells the training code to scan `checkpoints/` for the latest `steps_<N>/` directory and restore model weights, optimizer state, scheduler, and RNG state.

**`save_interval` tradeoff:** Training progress between the last checkpoint and the preemption event is lost. With `save_interval=50000`, up to 50K steps can be lost. If preemptions are frequent, reduce to 10000-25000 (at the cost of more storage and I/O overhead).

**On `kempner_h100`:** There is no preemption, but the 3-day time limit may cause timeout. `is_resume=true` still matters -- just resubmit the same script and training continues from the latest checkpoint.

---

## 4. GLIBC and Attention Implementation

Kempner compute nodes run **Rocky Linux 8** with **GLIBC 2.28**:

```bash
ldd --version | head -1
# ldd (GNU libc) 2.28
```

The `flash-attn` pip wheel requires GLIBC >= 2.32, so `flash_attention_2` cannot be used.

**Solution:** Always specify SDPA (Scaled Dot Product Attention), which is built into PyTorch and has no GLIBC dependency:

```bash
--framework.qwenvl.attn_implementation sdpa
```

Performance difference vs flash-attn is small on H100/H200.

---

## 5. Environment Setup

### 5.1 Environment variables (`~/.bashrc-kaiwen`)

All cache and storage paths are defined in `~/.bashrc-kaiwen`. SLURM scripts load them via `source ~/.bashrc-kaiwen`.

```bash
LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen
HF_HOME=$LAB_ROOT/.cache/huggingface
HF_HUB_CACHE=$HF_HOME/hub
TRANSFORMERS_CACHE=$HF_HOME/transformers
HF_DATASETS_CACHE=$HF_HOME/datasets
CONDA_PKGS_DIRS=$LAB_ROOT/.conda/pkgs
PIP_CACHE_DIR=$LAB_ROOT/.cache/pip
WANDB_DIR=$LAB_ROOT/.cache/wandb
TRITON_CACHE_DIR=/tmp/triton_cache_$USER
```

> `~/.bashrc-kaiwen` also switches PATH from the home-directory miniforge to the lab miniforge installation and initializes conda. It must be sourced **before** `conda activate`.

### 5.2 CUDA module

DeepSpeed needs `nvcc` to detect the CUDA version. Kempner manages CUDA through the module system:

```bash
module load cuda/12.2.0-fasrc01
```

Without this, you get:
```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

### 5.3 SLURM script setup pattern

Every SLURM script should follow this sequence at the top of the execution section:

```bash
# ── Environment setup ──
set +u                          # disable "unbound variable" check (bashrc uses PROMPT_COMMAND)
source ~/.bashrc-kaiwen         # load env vars, switch to lab miniforge, init conda
set -euo pipefail               # re-enable strict mode

module load cuda/12.2.0-fasrc01 # DeepSpeed needs nvcc
conda activate starVLA          # activate the training environment

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
```

The `set +u` / `set -euo pipefail` sandwich is required because `~/.bashrc-kaiwen` references `PROMPT_COMMAND`, which is undefined in a SLURM batch context. With `-u` active, this causes an immediate exit.

### 5.4 NCCL settings (multi-node)

For multi-node training, export these NCCL variables:

```bash
# Required for multi-node communication
export NCCL_BLOCKING_WAIT=1
export NCCL_ASYNC_ERROR_HANDLING=1

# InfiniBand (Kempner nodes have IB interconnect)
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0

# Timeouts (optional, useful for large-scale jobs)
export NCCL_TIMEOUT=10000              # seconds
export NCCL_SOCKET_TIMEOUT_MS=360000   # milliseconds
```

For single-node training, only `NCCL_BLOCKING_WAIT` and `NCCL_ASYNC_ERROR_HANDLING` are typically set.

Multi-node also requires setting the master address and port:

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
```

---

## 6. Software Versions

| Component | Version |
|-----------|---------|
| Conda env | `starVLA` (lab miniforge) |
| Python | 3.10 |
| PyTorch | 2.6.0+cu124 |
| DeepSpeed | 0.16.9 |
| Accelerate | 1.5.2 |
| Attention | SDPA (flash-attn blocked by GLIBC 2.28) |

---

## 7. Monitoring

### 7.1 Job status

```bash
squeue -u $USER                        # list all your jobs
squeue --start -j <jobid>              # estimated start time for a pending job
squeue -p kempner_requeue | wc -l      # count jobs in requeue partition
scancel <jobid>                        # cancel a job
```

### 7.2 Resource availability

```bash
# Detailed GPU status per type (idle/total/queued) on kempner_requeue
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h200   # H200 only
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh h100   # H100 only

# All partitions overview
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/most_empty_partition.sh

# Raw node status
sinfo -p kempner_requeue -o "%N %G %t %f" | grep -E "idle|mix"
```

### 7.3 Queue status meanings

| Status | Meaning | Action |
|--------|---------|--------|
| `PD (Resources)` | SLURM ready to schedule, waiting for free GPUs | Usually starts soon |
| `PD (Priority)` | Higher-priority jobs are ahead in the queue | May wait longer |
| `PD (ReqNodeNotAvail)` | Selected nodes are down or draining (maintenance) | Cancel and resubmit |
| `R` | Running | -- |
| `PR` | Preempted (kempner_requeue only) | Auto re-queued by `--requeue` |

### 7.4 Training logs

Log files follow the pattern `logs/<job-name>_<jobid>.out` and `.err`:

```bash
ls logs/starVLA_*.out                  # list all log files
tail -f logs/starVLA_custom_<jobid>.out   # live training output
tail -f logs/starVLA_custom_<jobid>.err   # live errors/warnings
```

If a job fails, check the `.err` file first for Python tracebacks.

### 7.5 GPU usage (on compute node)

```bash
# Open an interactive shell on the running job's node
srun --jobid=<jobid> --pty bash

# Then check GPU utilization
nvidia-smi
watch -n 5 nvidia-smi
```

---

## 8. Common SLURM Issues

### ReqNodeNotAvail

```
(ReqNodeNotAvail, UnavailableNodes:holygpu8a[...])
```

SLURM pre-selected nodes that are DOWN or DRAIN (typically under maintenance). The job will wait indefinitely for those specific nodes.

```bash
# Check node status
scontrol show node holygpu8a<xxxxx> | grep -E "NodeName|State|Reason"

# Fix: cancel and resubmit so SLURM picks different nodes
scancel <jobid>
sbatch scripts/<your_script>.sh
```

### DeepSpeed nvcc not found

```
FileNotFoundError: No such file or directory: '/usr/local/cuda/bin/nvcc'
```

Missing `module load cuda/12.2.0-fasrc01` in the SLURM script. Add it after sourcing bashrc and before `conda activate`.

### PROMPT_COMMAND unbound variable

```
.bashrc-kaiwen: line 14: PROMPT_COMMAND: unbound variable
```

The SLURM script has `set -u` (or `set -euo pipefail`) active before `source ~/.bashrc-kaiwen`. The bashrc references `PROMPT_COMMAND`, which is undefined in batch mode.

**Fix:** Wrap the source in `set +u` / `set -euo pipefail`:

```bash
set +u
source ~/.bashrc-kaiwen
set -euo pipefail
```

### Cluster full / long queue

```bash
# Check real-time GPU availability
bash /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/scripts/requeue_summary.sh

# If H200 is full (>95%), switch to H100:
#   --constraint=h100  --cpus-per-task=96
# If H100 is also full, consider submitting during off-peak hours.

# Check estimated start time
squeue --start -j <jobid>
```

If both GPU types on `kempner_requeue` are saturated, you can also try `kempner_h100` (longer queue but no preemption) or wait for off-peak hours.
