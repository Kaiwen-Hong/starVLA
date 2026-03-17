# Fix: CUDA OOM During Process Group Shutdown After Training (2026-03-16)

## Quick Fix Reference

```
starVLA/training/train_starvla.py   (line ~416)
```

**Fix:** add `torch.cuda.empty_cache()` before `dist.destroy_process_group()`,
and wrap the shutdown in `try/except` so a non-fatal NCCL error doesn't produce
a non-zero exit code.

---

## Symptom

Training completes all 50K steps successfully. Checkpoint and final_model are
both saved. WandB syncs. Then the process crashes during NCCL shutdown:

```
[rank2]: File ".../train_starvla.py", line 418, in main
[rank2]:     dist.destroy_process_group()
[rank2]: torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:133,
         unhandled cuda error ...
[rank2]: ncclUnhandledCudaError: Call to CUDA function failed.
[rank2]: Last error:
[rank2]: Cuda failure 'out of memory'
```

The error propagates through torch elastic, which reports:

```
starVLA/training/train_starvla.py FAILED
Root Cause: rank 2, exitcode 1
```

Despite the error, **no training data is lost** — the crash happens after all
saves are complete.

---

## Root Cause

1. At step 50000, the trainer saves the checkpoint and final_model to disk.
   This involves gathering model state across ranks, which keeps GPU memory
   near peak usage.
2. `dist.destroy_process_group()` internally triggers NCCL barrier/shutdown
   communication, which requires CUDA memory for NCCL buffers.
3. GPU memory is already saturated from checkpoint saving → NCCL's CUDA
   allocation fails → `ncclUnhandledCudaError: out of memory`.

This is a known PyTorch/NCCL issue. The shutdown phase does not have its own
memory budget and can fail on memory-constrained setups.

---

## Fix

### Before (train_starvla.py)

```python
logger.info("... and that's all, folks!")
if dist.is_initialized():
    dist.barrier()
    dist.destroy_process_group()
```

### After

```python
logger.info("... and that's all, folks!")
if dist.is_initialized():
    import torch.cuda
    torch.cuda.empty_cache()
    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Non-fatal error during process group shutdown: {e}")
```

Two changes:

1. **`torch.cuda.empty_cache()`** — releases PyTorch's CUDA memory cache back
   to the driver. After checkpoint saving, there are large blocks of cached-but-
   unused memory. Freeing them gives NCCL enough room for its shutdown buffers.

2. **`try/except`** — if shutdown still fails (e.g., on very tight memory
   configs), the error is logged as a warning instead of crashing the process.
   This prevents:
   - SLURM from marking the job as FAILED
   - `set -e` scripts from aborting
   - Requeue mechanisms from needlessly restarting a completed job

---

## Impact

- **Training correctness:** no impact. The fix only affects code that runs
  after all training, saving, and logging is complete.
- **Checkpoint integrity:** no impact. Checkpoints are already written to disk
  before this code runs.
- **SLURM behavior:** jobs now exit cleanly with code 0 instead of code 1,
  avoiding false-positive failure reports and unnecessary requeues.

---

## Observed In

- Job: v31-ee training (`v0309_v31_ee_qwenPI_requeue`)
- Node: `holygpu8a15502.rc.fas.harvard.edu`
- Hardware: 4× H200 80GB
- Failed rank: rank 2 (local_rank 2)
- NCCL version: 2.21.5
- PyTorch: torch.distributed with DeepSpeed ZeRO-2

---

## Files Changed

| File | Change |
|------|--------|
| `starVLA/training/train_starvla.py` | Add `empty_cache()` + try/except around `destroy_process_group()` |
