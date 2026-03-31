#!/usr/bin/env python3
"""
Open-loop evaluation for Dynamic-324 flow-matching (QwenPI) checkpoint.

Loads the trained FM checkpoint, runs predict_action on the dynamic-324 dataset,
and computes MSE / L1 metrics (overall + per-dimension + per-step).

Usage:
    python ur5/step1-fm-dynamic-open_loop_offline_evaluation.py
    python ur5/step1-fm-dynamic-open_loop_offline_evaluation.py --num_samples 500
    python ur5/step1-fm-dynamic-open_loop_offline_evaluation.py --checkpoint path/to/other.pt
"""

import argparse
import sys
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

# ── Edit these before running ──────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/DiscreteRTC/fastumi_pickandplace_qwenPI_dynamic_1/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DATA_ROOT_DIR = "/home/kaiwen/Desktop/research/fastumipro-collection/0srarvla-lerobo/starvla/datasets"
INSTRUCTION = "pick up the block in the pot and place it in the red area on the turntable"
# ───────────────────────────────────────────────────────────────────


def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[INFO] flash_attn not available, falling back to sdpa")
        return "sdpa"


# ── Dimension labels for 10D FastUMI actions ────────────────────────
DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",
    "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]
DIM_GROUPS = {
    "position (0:3)": slice(0, 3),
    "rotation (3:9)": slice(3, 9),
    "gripper (9)":    slice(9, 10),
}


def load_model(checkpoint_path: str):
    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()

    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None

    attn_impl = _detect_attn_implementation()
    config.framework.qwenvl.attn_implementation = attn_impl

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s (attn: {attn_impl})")
    return model


def load_dataset(model, include_state: bool = False, data_root_dir: str = None):
    data_cfg = model.config.datasets.vla_data
    if include_state:
        data_cfg.include_state = True
    if data_root_dir:
        data_cfg.data_root_dir = data_root_dir

    print(f"Loading dataset: {data_cfg.data_mix}")
    print(f"  root: {data_cfg.data_root_dir}")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    print(f"  total steps: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )
    return dataloader


def evaluate(model, dataloader, num_samples: int, instruction: str) -> dict:
    all_mse = []
    all_l1 = []
    all_per_dim_mse = []
    all_per_dim_l1 = []
    all_per_step_mse = []

    chunk_len = model.chunk_len

    count = 0
    for batch in tqdm(dataloader, total=min(num_samples, len(dataloader)), desc="Evaluating"):
        if count >= num_samples:
            break

        sample = batch[0]
        gt_actions = np.array(sample["action"], dtype=np.float32)  # [T, 10]

        # Override instruction if specified
        if instruction:
            for s in batch:
                s["lang"] = instruction

        output = model.predict_action(examples=batch)
        pred_actions = output["normalized_actions"][0].astype(np.float32)

        if gt_actions.shape[0] > chunk_len:
            gt_actions = gt_actions[-chunk_len:, :]

        T = min(pred_actions.shape[0], gt_actions.shape[0])
        pred = pred_actions[:T]
        gt = gt_actions[:T]

        mse = np.mean((pred - gt) ** 2)
        l1 = np.mean(np.abs(pred - gt))
        all_mse.append(mse)
        all_l1.append(l1)

        per_dim_mse = np.mean((pred - gt) ** 2, axis=0)
        per_dim_l1 = np.mean(np.abs(pred - gt), axis=0)
        all_per_dim_mse.append(per_dim_mse)
        all_per_dim_l1.append(per_dim_l1)

        per_step_mse = np.mean((pred - gt) ** 2, axis=1)
        all_per_step_mse.append(per_step_mse)

        count += 1

    per_dim_mse = np.mean(all_per_dim_mse, axis=0)
    per_dim_l1 = np.mean(all_per_dim_l1, axis=0)
    per_step = np.mean(all_per_step_mse, axis=0)

    results = {
        "num_samples": count,
        "chunk_len": chunk_len,
        "instruction": instruction,
        "overall_mse": float(np.mean(all_mse)),
        "overall_l1": float(np.mean(all_l1)),
        "overall_mse_std": float(np.std(all_mse)),
        "overall_l1_std": float(np.std(all_l1)),
        "per_dim_mse": {DIM_LABELS[i]: float(per_dim_mse[i]) for i in range(len(DIM_LABELS))},
        "per_dim_l1": {DIM_LABELS[i]: float(per_dim_l1[i]) for i in range(len(DIM_LABELS))},
        "per_step_mse": [float(v) for v in per_step],
    }
    for group_name, slc in DIM_GROUPS.items():
        results[f"mse_{group_name}"] = float(np.mean(per_dim_mse[slc]))
        results[f"l1_{group_name}"] = float(np.mean(per_dim_l1[slc]))

    return results


def print_results(results: dict, checkpoint_name: str = ""):
    header = "Open-Loop Eval Results (Flow-Matching)"
    if checkpoint_name:
        header += f" [{checkpoint_name}]"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  Samples evaluated : {results['num_samples']}")
    print(f"  Action chunk len  : {results['chunk_len']}")
    print(f"  Instruction       : {results['instruction']}")
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
    max_step_mse = max(results["per_step_mse"]) or 1.0
    for t, v in enumerate(results["per_step_mse"]):
        bar = "#" * int(v / max_step_mse * 30)
        print(f"  step {t:2d}: {v:.6f}  {bar}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop eval for QwenPI flow-matching checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR,
                        help="Local path to dataset root (overrides model config)")
    parser.add_argument("--instruction", type=str, default=INSTRUCTION,
                        help="Override language instruction for all samples")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: auto in checkpoint dir)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataloader = load_dataset(model, include_state=args.include_state,
                              data_root_dir=args.data_root_dir)

    num_samples = args.num_samples if args.num_samples > 0 else len(dataloader)
    results = evaluate(model, dataloader, num_samples, instruction=args.instruction)

    ckpt_name = Path(args.checkpoint).stem.replace("_pytorch_model", "")
    print_results(results, ckpt_name)

    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = Path(args.checkpoint).parents[1]
        out_path = run_dir / f"openloop_eval_{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
