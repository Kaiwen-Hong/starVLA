#!/usr/bin/env python3
"""
Open-loop evaluation for dynamic-329-v3 discrete diffusion checkpoint.

Loads the trained DD checkpoint, runs predict_action on the dynamic-329-v3 dataset,
and computes MSE / L1 metrics (overall + per-dimension + per-step).

Usage:
    python ur5n/dd/openloop_eval.py
    python ur5n/dd/openloop_eval.py --num_samples 500
    python ur5n/dd/openloop_eval.py --decode_temperature 0.1
    python ur5n/dd/openloop_eval.py --use_simple_max
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
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DATA_ROOT_DIR = "/home/kaiwen/Desktop/research/fastumipro-collection/0srarvla-lerobo/starvla/datasets"
DATA_MIX = "dynamic-329-v4"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1
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
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    return model


def load_dataset(model, data_mix: str, include_state: bool = False,
                 data_root_dir: str = None):
    data_cfg = model.config.datasets.vla_data
    data_cfg.data_mix = data_mix
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
        num_workers=0,
        collate_fn=collate_fn,
    )
    return dataloader


def _valid_mask(gt, rot_slice=slice(3, 9)):
    """Detect padded timesteps: rotation dims (norm_mode=none) are all zero when padded."""
    rot = gt[:, rot_slice]
    return ~np.all(np.abs(rot) < 1e-6, axis=1)


def evaluate(model, dataloader, num_samples: int, infer_kwargs: dict) -> dict:
    # Get action norm stats for denormalization
    unnorm_key = next(iter(model.norm_stats.keys()))
    action_stats = model.norm_stats[unnorm_key]["action"]

    # --- Normalized-space accumulators ---
    all_mse = []
    all_l1 = []
    all_per_dim_mse = []
    all_per_dim_l1 = []
    all_per_step_mse = []

    # --- Denormalized (world-frame) accumulators ---
    all_mse_wf = []
    all_l1_wf = []
    all_per_dim_mse_wf = []
    all_per_dim_l1_wf = []
    all_per_step_mse_wf = []

    n_padded_steps = 0
    n_total_steps = 0

    chunk_len = model.chunk_len

    count = 0
    for batch in tqdm(dataloader, total=min(num_samples, len(dataloader)), desc="Evaluating"):
        if count >= num_samples:
            break

        sample = batch[0]
        gt_actions = np.array(sample["action"], dtype=np.float32)  # [T, 10]

        output = model.predict_action(examples=batch, **infer_kwargs)
        pred_actions = output["normalized_actions"][0].astype(np.float32)

        if gt_actions.shape[0] > chunk_len:
            gt_actions = gt_actions[-chunk_len:, :]

        T = min(pred_actions.shape[0], gt_actions.shape[0])
        pred = pred_actions[:T]
        gt = gt_actions[:T]

        # Mask out padded timesteps (rotation dims all zero = padding)
        valid = _valid_mask(gt)
        n_total_steps += T
        n_padded_steps += int((~valid).sum())

        if not valid.any():
            continue

        # --- Normalized metrics ---
        pred_v = pred[valid]
        gt_v = gt[valid]

        mse = np.mean((pred_v - gt_v) ** 2)
        l1 = np.mean(np.abs(pred_v - gt_v))
        all_mse.append(mse)
        all_l1.append(l1)

        per_dim_mse = np.mean((pred_v - gt_v) ** 2, axis=0)
        per_dim_l1 = np.mean(np.abs(pred_v - gt_v), axis=0)
        all_per_dim_mse.append(per_dim_mse)
        all_per_dim_l1.append(per_dim_l1)

        step_mse = np.mean((pred - gt) ** 2, axis=1)
        step_mse[~valid] = np.nan
        all_per_step_mse.append(step_mse)

        # --- Denormalized (world-frame) metrics ---
        pred_wf = baseframework.unnormalize_actions(pred.copy(), action_stats)
        gt_wf = baseframework.unnormalize_actions(gt.copy(), action_stats)

        pred_wf_v = pred_wf[valid]
        gt_wf_v = gt_wf[valid]

        mse_wf = np.mean((pred_wf_v - gt_wf_v) ** 2)
        l1_wf = np.mean(np.abs(pred_wf_v - gt_wf_v))
        all_mse_wf.append(mse_wf)
        all_l1_wf.append(l1_wf)

        per_dim_mse_wf = np.mean((pred_wf_v - gt_wf_v) ** 2, axis=0)
        per_dim_l1_wf = np.mean(np.abs(pred_wf_v - gt_wf_v), axis=0)
        all_per_dim_mse_wf.append(per_dim_mse_wf)
        all_per_dim_l1_wf.append(per_dim_l1_wf)

        step_mse_wf = np.mean((pred_wf - gt_wf) ** 2, axis=1)
        step_mse_wf[~valid] = np.nan
        all_per_step_mse_wf.append(step_mse_wf)

        count += 1

    per_dim_mse = np.mean(all_per_dim_mse, axis=0)
    per_dim_l1 = np.mean(all_per_dim_l1, axis=0)
    per_step = np.nanmean(all_per_step_mse, axis=0)

    per_dim_mse_wf = np.mean(all_per_dim_mse_wf, axis=0)
    per_dim_l1_wf = np.mean(all_per_dim_l1_wf, axis=0)
    per_step_wf = np.nanmean(all_per_step_mse_wf, axis=0)

    print(f"\n  [Padding] {n_padded_steps}/{n_total_steps} timesteps excluded "
          f"({100*n_padded_steps/max(n_total_steps,1):.1f}%)")

    results = {
        "num_samples": count,
        "chunk_len": chunk_len,
        "padded_steps_excluded": n_padded_steps,
        "total_steps": n_total_steps,
        # --- Normalized ---
        "overall_mse": float(np.mean(all_mse)),
        "overall_l1": float(np.mean(all_l1)),
        "overall_mse_std": float(np.std(all_mse)),
        "overall_l1_std": float(np.std(all_l1)),
        "per_dim_mse": {DIM_LABELS[i]: float(per_dim_mse[i]) for i in range(len(DIM_LABELS))},
        "per_dim_l1": {DIM_LABELS[i]: float(per_dim_l1[i]) for i in range(len(DIM_LABELS))},
        "per_step_mse": [float(v) for v in per_step],
        # --- Denormalized (world-frame) ---
        "overall_mse_wf": float(np.mean(all_mse_wf)),
        "overall_l1_wf": float(np.mean(all_l1_wf)),
        "overall_mse_std_wf": float(np.std(all_mse_wf)),
        "overall_l1_std_wf": float(np.std(all_l1_wf)),
        "per_dim_mse_wf": {DIM_LABELS[i]: float(per_dim_mse_wf[i]) for i in range(len(DIM_LABELS))},
        "per_dim_l1_wf": {DIM_LABELS[i]: float(per_dim_l1_wf[i]) for i in range(len(DIM_LABELS))},
        "per_step_mse_wf": [float(v) for v in per_step_wf],
    }
    for group_name, slc in DIM_GROUPS.items():
        results[f"mse_{group_name}"] = float(np.mean(per_dim_mse[slc]))
        results[f"l1_{group_name}"] = float(np.mean(per_dim_l1[slc]))
        results[f"mse_wf_{group_name}"] = float(np.mean(per_dim_mse_wf[slc]))
        results[f"l1_wf_{group_name}"] = float(np.mean(per_dim_l1_wf[slc]))

    return results


def print_results(results: dict, checkpoint_name: str = ""):
    header = "Open-Loop Eval Results"
    if checkpoint_name:
        header += f" [{checkpoint_name}]"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  Samples evaluated : {results['num_samples']}")
    print(f"  Action chunk len  : {results['chunk_len']}")

    # ── Normalized ──
    print(f"\n  ====== Normalized Space ======")
    print(f"  Overall MSE       : {results['overall_mse']:.6f} (+/- {results['overall_mse_std']:.6f})")
    print(f"  Overall L1        : {results['overall_l1']:.6f} (+/- {results['overall_l1_std']:.6f})")

    print(f"\n  --- Per-Group (normalized) ---")
    for group_name in DIM_GROUPS:
        mse = results[f"mse_{group_name}"]
        l1 = results[f"l1_{group_name}"]
        print(f"  {group_name:20s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Dimension (normalized) ---")
    for dim_name in DIM_LABELS:
        mse = results["per_dim_mse"][dim_name]
        l1 = results["per_dim_l1"][dim_name]
        print(f"  {dim_name:12s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Step MSE (normalized) ---")
    max_step_mse = max(results["per_step_mse"]) or 1.0
    for t, v in enumerate(results["per_step_mse"]):
        bar = "#" * int(v / max_step_mse * 30)
        print(f"  step {t:2d}: {v:.6f}  {bar}")

    # ── Denormalized (world-frame) ──
    print(f"\n  ====== World-Frame (denormalized) ======")
    print(f"  Overall MSE       : {results['overall_mse_wf']:.6f} (+/- {results['overall_mse_std_wf']:.6f})")
    print(f"  Overall L1        : {results['overall_l1_wf']:.6f} (+/- {results['overall_l1_std_wf']:.6f})")

    print(f"\n  --- Per-Group (world-frame) ---")
    for group_name in DIM_GROUPS:
        mse = results[f"mse_wf_{group_name}"]
        l1 = results[f"l1_wf_{group_name}"]
        print(f"  {group_name:20s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Dimension (world-frame) ---")
    for dim_name in DIM_LABELS:
        mse = results["per_dim_mse_wf"][dim_name]
        l1 = results["per_dim_l1_wf"][dim_name]
        print(f"  {dim_name:12s}  MSE={mse:.6f}  L1={l1:.6f}")

    print(f"\n  --- Per-Step MSE (world-frame) ---")
    max_step_mse_wf = max(results["per_step_mse_wf"]) or 1.0
    for t, v in enumerate(results["per_step_mse_wf"]):
        bar = "#" * int(v / max_step_mse_wf * 30)
        print(f"  step {t:2d}: {v:.6f}  {bar}")

    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop eval for dynamic-329-v3 DD checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR)
    parser.add_argument("--data_mix", type=str, default=DATA_MIX)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: auto in checkpoint dir)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataloader = load_dataset(model, data_mix=args.data_mix,
                              include_state=args.include_state,
                              data_root_dir=args.data_root_dir)

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    num_samples = args.num_samples if args.num_samples > 0 else len(dataloader)
    results = evaluate(model, dataloader, num_samples, infer_kwargs)

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
