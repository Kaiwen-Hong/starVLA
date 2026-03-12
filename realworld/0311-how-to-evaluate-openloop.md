# Open-Loop Evaluation Guide for FastUMI Pick-and-Place Policy

**Date:** 2026-03-11
**Model:** QwenOFT (Qwen3-VL-4B + MLP action head)
**Dataset:** `pickandplace-real-0307` (250 trajectories, 22224 transitions)
**Action space:** 10D (3 pos + 6 rot6d + 1 gripper)

---

## Prerequisites

- conda env `starVLA` activated
- At least 1 GPU with ~20 GB VRAM (single-GPU inference)
- Training completed with checkpoints at:
  ```
  results/Checkpoints/fastumi_pickandplace_qwenOFT/
  ├── checkpoints/steps_{10k..100k}_pytorch_model.pt
  ├── final_model/pytorch_model.pt
  ├── config.yaml
  └── dataset_statistics.json
  ```

---

## Step 1: Environment Setup

```bash
# === Paths ===
export REPO_DIR=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA
export LAB_ROOT=/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen

# === HuggingFace / cache (keep off home dir) ===
export HF_HOME=${LAB_ROOT}/.cache/huggingface
export HF_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_DATASETS_CACHE=${HF_HOME}/datasets
export TRITON_CACHE_DIR=/tmp/triton_cache_${USER}

# === Conda ===
source "${LAB_ROOT}/miniforge3/etc/profile.d/conda.sh"
conda activate starVLA

# === CUDA (cluster) ===
module load cuda/12.2.0-fasrc01 2>/dev/null || true

cd "${REPO_DIR}"
```

---

## Step 2: Run Open-Loop Evaluation

Copy the Python script below and run it, or save it as `realworld/eval_openloop.py`.

```bash
# Run with default settings (final checkpoint, 200 samples)
python realworld/eval_openloop.py

# Or specify checkpoint and number of samples
python realworld/eval_openloop.py \
  --checkpoint results/Checkpoints/fastumi_pickandplace_qwenOFT/checkpoints/steps_100000_pytorch_model.pt \
  --num_samples 500

# Evaluate ALL checkpoints to see learning curve
python realworld/eval_openloop.py --sweep
```

---

## Step 3: The Evaluation Script

Save as `realworld/eval_openloop.py`:

```python
"""
Open-loop evaluation for FastUMI pick-and-place policy.
Loads a trained QwenOFT checkpoint, runs predict_action on the training
dataset, and computes MSE / L1 metrics (overall + per-dimension).
"""

import argparse
import sys
import os
import time
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
from omegaconf import OmegaConf


# ── Dimension labels for 10D FastUMI actions ────────────────────────
DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",                         # 0-2: position
    "rot6d_0", "rot6d_1", "rot6d_2",                    # 3-5: rotation (first basis)
    "rot6d_3", "rot6d_4", "rot6d_5",                    # 6-8: rotation (second basis)
    "gripper",                                           # 9:   gripper
]
DIM_GROUPS = {
    "position (0:3)":  slice(0, 3),
    "rotation (3:9)":  slice(3, 9),
    "gripper (9)":     slice(9, 10),
}


def load_model(checkpoint_path: str) -> baseframework:
    """Load a QwenOFT model from checkpoint."""
    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()
    model = baseframework.from_pretrained(checkpoint_path)
    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def load_dataset(model) -> DataLoader:
    """
    Re-create the training dataset (same config as training)
    to use as open-loop evaluation data.
    """
    # Read config from model (attached during from_pretrained)
    cfg = model.config
    data_cfg = cfg.datasets.vla_data

    print(f"Loading dataset: {data_cfg.data_mix}")
    print(f"  root: {data_cfg.data_root_dir}")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    print(f"  total steps: {len(dataset)}")

    # Use batch_size=1 for deterministic evaluation
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return dataloader


def evaluate(model, dataloader, num_samples: int) -> dict:
    """
    Run open-loop evaluation.

    For each sample:
      1. Feed (image, lang) to model.predict_action()
      2. Compare predicted normalized actions vs ground-truth normalized actions
      3. Accumulate MSE and L1 errors

    Returns dict of metrics.
    """
    all_mse = []
    all_l1 = []
    all_per_dim_mse = []
    all_per_dim_l1 = []
    all_per_step_mse = []  # [num_samples, chunk_len]

    chunk_len = model.chunk_len  # typically 16

    count = 0
    for batch in tqdm(dataloader, total=min(num_samples, len(dataloader)), desc="Evaluating"):
        if count >= num_samples:
            break

        # batch is a list of dicts (batch_size=1 here)
        sample = batch[0]
        gt_actions = np.array(sample["action"], dtype=np.float32)  # [chunk_len, 10]

        # Run inference
        output = model.predict_action(examples=batch)
        pred_actions = output["normalized_actions"][0]  # [chunk_len, 10]
        pred_actions = pred_actions.astype(np.float32)

        # Align shapes: ground truth might have more steps than chunk_len
        # Training uses actions[:, -(future_window+1):, :] as target
        if gt_actions.shape[0] > chunk_len:
            gt_actions = gt_actions[-chunk_len:, :]

        # Clip to same length
        T = min(pred_actions.shape[0], gt_actions.shape[0])
        pred = pred_actions[:T]
        gt = gt_actions[:T]

        # Overall metrics
        mse = np.mean((pred - gt) ** 2)
        l1 = np.mean(np.abs(pred - gt))
        all_mse.append(mse)
        all_l1.append(l1)

        # Per-dimension metrics
        per_dim_mse = np.mean((pred - gt) ** 2, axis=0)  # [10]
        per_dim_l1 = np.mean(np.abs(pred - gt), axis=0)  # [10]
        all_per_dim_mse.append(per_dim_mse)
        all_per_dim_l1.append(per_dim_l1)

        # Per-step metrics (averaged over dims)
        per_step_mse = np.mean((pred - gt) ** 2, axis=1)  # [chunk_len]
        all_per_step_mse.append(per_step_mse)

        count += 1

    # Aggregate
    results = {
        "num_samples": count,
        "chunk_len": chunk_len,
        "overall_mse": float(np.mean(all_mse)),
        "overall_l1": float(np.mean(all_l1)),
        "overall_mse_std": float(np.std(all_mse)),
        "overall_l1_std": float(np.std(all_l1)),
    }

    # Per-dimension
    per_dim_mse = np.mean(all_per_dim_mse, axis=0)
    per_dim_l1 = np.mean(all_per_dim_l1, axis=0)
    results["per_dim_mse"] = {DIM_LABELS[i]: float(per_dim_mse[i]) for i in range(len(DIM_LABELS))}
    results["per_dim_l1"] = {DIM_LABELS[i]: float(per_dim_l1[i]) for i in range(len(DIM_LABELS))}

    # Per-group
    for group_name, slc in DIM_GROUPS.items():
        results[f"mse_{group_name}"] = float(np.mean(per_dim_mse[slc]))
        results[f"l1_{group_name}"] = float(np.mean(per_dim_l1[slc]))

    # Per-step (averaged over all samples and dims)
    per_step = np.mean(all_per_step_mse, axis=0)
    results["per_step_mse"] = [float(v) for v in per_step]

    return results


def print_results(results: dict, checkpoint_name: str = ""):
    """Pretty-print evaluation results."""
    header = f"Open-Loop Eval Results"
    if checkpoint_name:
        header += f" [{checkpoint_name}]"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  Samples evaluated : {results['num_samples']}")
    print(f"  Action chunk len  : {results['chunk_len']}")
    print(f"  Overall MSE       : {results['overall_mse']:.6f} (+/- {results['overall_mse_std']:.6f})")
    print(f"  Overall L1        : {results['overall_l1']:.6f} (+/- {results['overall_l1_std']:.6f})")

    print(f"\n  --- Per-Group ---")
    for group_name in DIM_GROUPS:
        mse = results[f"mse_{group_name}"]
        l1 = results[f"l1_{group_name}"]
        print(f"  {group_name:20s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Dimension ---")
    for dim_name in DIM_LABELS:
        mse = results["per_dim_mse"][dim_name]
        l1 = results["per_dim_l1"][dim_name]
        print(f"  {dim_name:12s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Step MSE (action horizon) ---")
    for t, v in enumerate(results["per_step_mse"]):
        bar = "#" * int(v / max(results["per_step_mse"]) * 30)
        print(f"  step {t:2d}: {v:.6f}  {bar}")
    print(f"{'=' * 60}\n")


def find_all_checkpoints(run_dir: str) -> list:
    """Find all checkpoint .pt files sorted by step number."""
    ckpt_dir = Path(run_dir) / "checkpoints"
    pts = list(ckpt_dir.glob("steps_*_pytorch_model.pt"))
    # Sort by step number
    def step_num(p):
        m = re.search(r"steps_(\d+)_", p.name)
        return int(m.group(1)) if m else 0
    pts.sort(key=step_num)
    return pts


def main():
    parser = argparse.ArgumentParser(description="Open-loop eval for FastUMI policy")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/Checkpoints/fastumi_pickandplace_qwenOFT/checkpoints/steps_100000_pytorch_model.pt",
        help="Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=200,
        help="Number of dataset samples to evaluate on",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Evaluate ALL checkpoints in the run (learning curve)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON (default: auto-generated in run dir)",
    )
    args = parser.parse_args()

    if args.sweep:
        run_dir = "results/Checkpoints/fastumi_pickandplace_qwenOFT"
        checkpoints = find_all_checkpoints(run_dir)
        print(f"Sweep mode: found {len(checkpoints)} checkpoints")

        all_results = {}
        for ckpt_path in checkpoints:
            step_name = ckpt_path.stem.replace("_pytorch_model", "")
            print(f"\n>>> Evaluating {step_name} ...")
            model = load_model(str(ckpt_path))
            dataloader = load_dataset(model)
            results = evaluate(model, dataloader, args.num_samples)
            print_results(results, step_name)
            all_results[step_name] = results

            # Free GPU memory for next checkpoint
            del model
            torch.cuda.empty_cache()

        # Save sweep results
        out_path = Path(run_dir) / "openloop_eval_sweep.json"
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Sweep results saved to {out_path}")

        # Print summary table
        print(f"\n{'=' * 70}")
        print(f"{'Checkpoint':>20s}  {'MSE':>10s}  {'L1':>10s}  {'MSE_pos':>10s}  {'MSE_rot':>10s}")
        print(f"{'-' * 70}")
        for name, r in all_results.items():
            print(f"{name:>20s}  {r['overall_mse']:10.6f}  {r['overall_l1']:10.6f}  "
                  f"{r['mse_position (0:3)']:10.6f}  {r['mse_rotation (3:9)']:10.6f}")
        print(f"{'=' * 70}")

    else:
        model = load_model(args.checkpoint)
        dataloader = load_dataset(model)
        results = evaluate(model, dataloader, args.num_samples)

        ckpt_name = Path(args.checkpoint).stem.replace("_pytorch_model", "")
        print_results(results, ckpt_name)

        # Save results
        if args.output:
            out_path = args.output
        else:
            run_dir = Path(args.checkpoint).parents[1]
            out_path = run_dir / f"openloop_eval_{ckpt_name}.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
```

---

## Step 4: Interpret Results

### Metrics Explained

| Metric | What It Measures |
|--------|-----------------|
| **Overall MSE** | Mean squared error between predicted and GT **normalized** actions (range ~[-1,1]) |
| **Overall L1** | Mean absolute error (same space) |
| **Per-dimension** | Breakdown by pos_x/y/z, rot6d_0..5, gripper |
| **Per-group** | Aggregated by position (0:3), rotation (3:9), gripper (9) |
| **Per-step MSE** | Error at each step in the 16-step action horizon. Expect increasing error for later steps |

### What "good" looks like

Since actions are normalized to [-1, 1]:
- **MSE < 0.01** and **L1 < 0.05**: model fits training data well
- **MSE > 0.1**: model hasn't learned the task
- Per-step error should increase gradually; if step 0 is already high, the model is fundamentally not working

### Caveats

1. **This evaluates on training data** -- it measures fitting, not generalization. Low error is necessary but not sufficient for real-world deployment.
2. **Metrics are in normalized space.** The ground-truth actions from the dataset are already normalized by the data pipeline (min-max using q01/q99 percentiles). The model also outputs normalized actions. So the comparison is fair and directly comparable.
3. **`unnormalize_actions()` has a hardcoded gripper index at `[:, 6]`** which is correct for 7D action spaces but NOT for 10D FastUMI (gripper is at index 9). If you need denormalized actions for deployment, you'll need to fix this in `base_framework.py:203` or bypass it. For open-loop eval in normalized space this does not matter.

---

## Quick Reference: Single Checkpoint Eval

```bash
cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# Setup env (copy from Step 1), then:
python realworld/eval_openloop.py \
  --checkpoint results/Checkpoints/fastumi_pickandplace_qwenOFT/checkpoints/steps_100000_pytorch_model.pt \
  --num_samples 200
```

## Quick Reference: Learning Curve Sweep

```bash
python realworld/eval_openloop.py --sweep --num_samples 100
```

This evaluates steps_10000 through steps_100000, prints a summary table, and saves `openloop_eval_sweep.json`.
