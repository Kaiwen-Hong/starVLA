#!/usr/bin/env python3
"""
Open-loop evaluation for dynamic-329-v2 discrete diffusion checkpoint.

Loads the trained DD checkpoint, runs predict_action on the dynamic-329-v2 dataset,
and computes MSE / L1 metrics (overall + per-dimension + per-step).

Usage:
    python realworld/0331-dd-openloop-eval.py
    python realworld/0331-dd-openloop-eval.py --num_samples 500
    python realworld/0331-dd-openloop-eval.py --decode_temperature 0.1
    python realworld/0331-dd-openloop-eval.py --use_simple_max
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")

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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "/scratch/wangpc/starVLA/results/Checkpoints/fastumi_pickandplace_qwenDiscreteDiffusion_329v2/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DATA_ROOT_DIR = "/scratch/wangpc/starVLA/playground/Datasets/FastUMI"
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


def get_pos_real_scale(norm_stats: dict) -> np.ndarray:
    """Return per-dim half-range (max-min)/2 for position dims 0:3 from norm_stats.

    Converts normalized error → real error:  real = normalized * scale.
    Returns array of shape [3,] in the same unit as the raw data (meters).
    """
    # norm_stats layout: {embodiment_key: {action: {min:[...], max:[...], ...}}}
    key = next(iter(norm_stats.keys()))
    action_stats = norm_stats[key]["action"]
    pos_min = np.array(action_stats["min"][:3], dtype=np.float64)
    pos_max = np.array(action_stats["max"][:3], dtype=np.float64)
    return (pos_max - pos_min) / 2.0   # half-range: maps [-1,1] → real


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


def _valid_mask(gt, rot_slice=slice(3, 9)):
    """Detect padded timesteps: rotation dims (norm_mode=none) are all zero when padded."""
    rot = gt[:, rot_slice]
    return ~np.all(np.abs(rot) < 1e-6, axis=1)


def evaluate(model, dataloader, num_samples: int, infer_kwargs: dict,
             pos_scale: np.ndarray = None) -> dict:
    all_mse = []
    all_l1 = []
    all_per_dim_mse = []   # list of [10,] arrays – per-sample per-dim MSE
    all_per_dim_l1 = []
    all_per_step_mse = []
    all_pos_step_mse = []  # per-sample per-step MSE for pos_x/y/z: [N, T, 3]
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

        # The model predicts the LAST (future_action_window_size + 1) steps,
        # so align GT to the tail of the action chunk (same as training).
        pred_len = pred_actions.shape[0]
        gt_actions = gt_actions[-pred_len:, :]

        T = min(pred_len, gt_actions.shape[0])
        pred = pred_actions[:T]
        gt = gt_actions[:T]

        # Mask out padded timesteps (rotation dims all zero = padding)
        valid = _valid_mask(gt)
        n_total_steps += T
        n_padded_steps += int((~valid).sum())

        if not valid.any():
            continue

        pred_v = pred[valid]
        gt_v = gt[valid]

        mse = np.mean((pred_v - gt_v) ** 2)
        l1 = np.mean(np.abs(pred_v - gt_v))
        all_mse.append(mse)
        all_l1.append(l1)

        per_dim_mse = np.mean((pred_v - gt_v) ** 2, axis=0)  # [10,]
        per_dim_l1 = np.mean(np.abs(pred_v - gt_v), axis=0)
        all_per_dim_mse.append(per_dim_mse)
        all_per_dim_l1.append(per_dim_l1)

        # Per-step MSE: use full array but mark padded as NaN
        step_mse = np.mean((pred - gt) ** 2, axis=1)
        step_mse[~valid] = np.nan
        all_per_step_mse.append(step_mse)

        # Per-step per-pos-dim squared error: [T, 3]
        pos_se = (pred[:, :3] - gt[:, :3]) ** 2  # [T, 3]
        pos_se[~valid] = np.nan
        all_pos_step_mse.append(pos_se)

        count += 1

    # Stack into arrays: [N, 10] for per-dim, [N,] for overall
    all_per_dim_mse = np.array(all_per_dim_mse)
    all_per_dim_l1 = np.array(all_per_dim_l1)
    all_mse = np.array(all_mse)
    all_l1 = np.array(all_l1)

    per_dim_mse = np.mean(all_per_dim_mse, axis=0)
    per_dim_l1 = np.mean(all_per_dim_l1, axis=0)
    per_step = np.nanmean(all_per_step_mse, axis=0)

    print(f"\n  [Padding] {n_padded_steps}/{n_total_steps} timesteps excluded "
          f"({100*n_padded_steps/max(n_total_steps,1):.1f}%)")

    results = {
        "num_samples": count,
        "chunk_len": chunk_len,
        "padded_steps_excluded": n_padded_steps,
        "total_steps": n_total_steps,
        "overall_mse": float(np.mean(all_mse)),
        "overall_l1": float(np.mean(all_l1)),
        "overall_mse_std": float(np.std(all_mse)),
        "overall_l1_std": float(np.std(all_l1)),
        "per_dim_mse": {DIM_LABELS[i]: float(per_dim_mse[i]) for i in range(len(DIM_LABELS))},
        "per_dim_l1": {DIM_LABELS[i]: float(per_dim_l1[i]) for i in range(len(DIM_LABELS))},
        "per_step_mse": [float(v) for v in per_step],
        # Raw per-sample arrays for downstream visualization
        "_all_per_dim_mse": all_per_dim_mse,   # [N, 10]
        "_all_mse": all_mse,                    # [N,]
        "_all_pos_step_mse": np.array(all_pos_step_mse),  # [N, T, 3]
    }
    for group_name, slc in DIM_GROUPS.items():
        results[f"mse_{group_name}"] = float(np.mean(per_dim_mse[slc]))
        results[f"l1_{group_name}"] = float(np.mean(per_dim_l1[slc]))

    # ── Real-scale position metrics (meters / cm) ──
    if pos_scale is not None:
        scale = pos_scale  # [3,]  half-range in meters
        # Per-dim real L1 (meters) = normalized_L1 * scale
        pos_real_l1 = per_dim_l1[:3] * scale
        # Per-dim real RMSE (meters) = sqrt(normalized_MSE) * scale
        pos_real_rmse = np.sqrt(per_dim_mse[:3]) * scale
        results["pos_real_scale_m"] = scale.tolist()
        results["pos_real_l1_m"] = {DIM_LABELS[i]: float(pos_real_l1[i]) for i in range(3)}
        results["pos_real_l1_cm"] = {DIM_LABELS[i]: float(pos_real_l1[i] * 100) for i in range(3)}
        results["pos_real_rmse_m"] = {DIM_LABELS[i]: float(pos_real_rmse[i]) for i in range(3)}
        results["pos_real_rmse_cm"] = {DIM_LABELS[i]: float(pos_real_rmse[i] * 100) for i in range(3)}
        # Per-step real RMSE for position [T, 3]
        pos_step_mse_arr = results["_all_pos_step_mse"]  # [N, T, 3]
        per_step_pos_rmse_norm = np.sqrt(np.nanmean(pos_step_mse_arr, axis=0))  # [T, 3]
        per_step_pos_rmse_cm = per_step_pos_rmse_norm * scale[None, :] * 100  # [T, 3]
        results["_pos_scale"] = scale
        results["per_step_pos_rmse_cm"] = {
            DIM_LABELS[i]: [float(v) for v in per_step_pos_rmse_cm[:, i]]
            for i in range(3)
        }

    return results


def plot_mse_distributions(results: dict, save_path: Path):
    """Plot per-dimension MSE histogram distributions in a single figure."""
    all_per_dim_mse = results["_all_per_dim_mse"]  # [N, 10]
    N, D = all_per_dim_mse.shape

    fig, axes = plt.subplots(2, 5, figsize=(22, 8))
    axes = axes.flatten()

    for d in range(D):
        ax = axes[d]
        vals = all_per_dim_mse[:, d]
        mean_val = np.mean(vals)
        median_val = np.median(vals)

        ax.hist(vals, bins=50, color="steelblue", edgecolor="white", alpha=0.85)
        ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"mean={mean_val:.5f}")
        ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.5, label=f"median={median_val:.5f}")
        ax.set_title(DIM_LABELS[d], fontsize=13, fontweight="bold")
        ax.set_xlabel("MSE")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    fig.suptitle(f"Per-Dimension MSE Distribution  (N={N} samples)", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"MSE distribution plot saved to {save_path}")


def plot_pos_step_mse(results: dict, save_path: Path):
    """One 4x4 figure per pos axis (+ mean): 16 subplots = steps 0..15.

    If real-scale is available, plots RMSE in cm; otherwise normalized MSE.
    """
    pos_step_mse = results["_all_pos_step_mse"]  # [N, T, 3]
    N, T, _ = pos_step_mse.shape
    T = min(T, 16)
    pos_labels = ["pos_x", "pos_y", "pos_z"]
    col_labels = pos_labels + ["pos_xyz (mean)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]

    pos_scale = results.get("_pos_scale", None)  # [3,] meters or None
    use_real = pos_scale is not None

    for dim_idx, dim_name in enumerate(col_labels):
        fig, axes = plt.subplots(4, 4, figsize=(16, 12))
        for t in range(T):
            ax = axes[t // 4][t % 4]
            if dim_idx < 3:
                vals = pos_step_mse[:, t, dim_idx]
            else:
                vals = np.nanmean(pos_step_mse[:, t, :], axis=1)
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                ax.set_visible(False)
                continue

            if use_real:
                # Convert normalized squared error → real RMSE in cm
                scale_cm = (pos_scale[dim_idx] if dim_idx < 3
                            else np.mean(pos_scale)) * 100
                vals_show = np.sqrt(vals) * scale_cm  # RMSE in cm
                xlabel = "RMSE (cm)"
                fmt = ".4f"
            else:
                vals_show = vals
                xlabel = "MSE (normalized)"
                fmt = ".6f"

            mean_val = np.mean(vals_show)
            median_val = np.median(vals_show)
            ax.hist(vals_show, bins=40, color=colors[dim_idx],
                    edgecolor="white", alpha=0.8)
            ax.axvline(mean_val, color="red", linestyle="--", linewidth=1.2,
                       label=f"mean={mean_val:{fmt}}")
            ax.axvline(median_val, color="orange", linestyle=":", linewidth=1.2,
                       label=f"med={median_val:{fmt}}")
            if use_real:
                ax.axvline(0.1, color="black", linestyle="-", linewidth=1,
                           alpha=0.5, label="0.1 cm target")
            ax.set_title(f"step {t}", fontsize=10)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.legend(fontsize=7)
            ax.tick_params(labelsize=7)

        for t in range(T, 16):
            axes[t // 4][t % 4].set_visible(False)

        unit = "RMSE cm" if use_real else "MSE normalized"
        fig.suptitle(f"{dim_name} — Per-Step {unit}  (N={N})",
                     fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        suffix = dim_name.replace(" ", "_").replace("(", "").replace(")", "")
        fig_path = save_path.with_name(save_path.stem + f"_{suffix}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  {dim_name} step plot saved to {fig_path}")


def print_results(results: dict, checkpoint_name: str = ""):
    header = "Open-Loop Eval Results"
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

    # ── Real-scale position errors ──
    if "pos_real_l1_cm" in results:
        print(f"\n  --- Position Real-Scale Errors ---")
        print(f"  (norm_range/2 per dim: {['%.5f m' % s for s in results['pos_real_scale_m']]})")
        for dim_name in ["pos_x", "pos_y", "pos_z"]:
            l1_cm = results["pos_real_l1_cm"][dim_name]
            rmse_cm = results["pos_real_rmse_cm"][dim_name]
            print(f"  {dim_name:8s}  L1={l1_cm:.4f} cm   RMSE={rmse_cm:.4f} cm")
        avg_l1 = np.mean([results["pos_real_l1_cm"][d] for d in ["pos_x", "pos_y", "pos_z"]])
        avg_rmse = np.mean([results["pos_real_rmse_cm"][d] for d in ["pos_x", "pos_y", "pos_z"]])
        print(f"  {'avg':8s}  L1={avg_l1:.4f} cm   RMSE={avg_rmse:.4f} cm")
        target = 0.1  # cm
        status = "PASS" if avg_l1 <= target else "FAIL"
        print(f"  Target: L1 <= {target} cm  →  [{status}]")

    if "per_step_pos_rmse_cm" in results:
        print(f"\n  --- Per-Step Position RMSE (cm) ---")
        print(f"  {'step':>6s}  {'pos_x':>10s}  {'pos_y':>10s}  {'pos_z':>10s}")
        T = len(results["per_step_pos_rmse_cm"]["pos_x"])
        for t in range(T):
            vals = [results["per_step_pos_rmse_cm"][d][t] for d in ["pos_x", "pos_y", "pos_z"]]
            print(f"  {t:6d}  {vals[0]:10.4f}  {vals[1]:10.4f}  {vals[2]:10.4f}")

    print(f"\n  --- Per-Step MSE (action horizon) ---")
    max_step_mse = max(results["per_step_mse"]) or 1.0
    for t, v in enumerate(results["per_step_mse"]):
        bar = "#" * int(v / max_step_mse * 30)
        print(f"  step {t:2d}: {v:.6f}  {bar}")
    print(f"{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Open-loop eval for dynamic-329-v2 DD checkpoint")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR,
                        help="Local path to dataset root (overrides model config)")
    parser.add_argument("--num_samples", type=int, default=200,
                        help="Number of samples to evaluate (0 = all)")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: auto in checkpoint dir)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataloader = load_dataset(model, include_state=args.include_state,
                              data_root_dir=args.data_root_dir)

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    # Extract position real-scale from norm_stats
    try:
        pos_scale = get_pos_real_scale(model.norm_stats)
        print(f"  pos real-scale (half-range): {pos_scale} m  = {pos_scale*100} cm")
    except Exception as e:
        print(f"  [WARN] Could not extract pos real-scale: {e}")
        pos_scale = None

    num_samples = args.num_samples if args.num_samples > 0 else len(dataloader)
    results = evaluate(model, dataloader, num_samples, infer_kwargs, pos_scale=pos_scale)

    ckpt_name = Path(args.checkpoint).stem.replace("_pytorch_model", "")
    print_results(results, ckpt_name)

    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = Path(args.checkpoint).parents[1]
        out_path = run_dir / f"openloop_eval_{ckpt_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Save JSON (exclude numpy arrays)
    results_json = {k: v for k, v in results.items() if not k.startswith("_")}
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {out_path}")

    # Plot per-dimension MSE distributions
    plot_path = out_path.with_name(out_path.stem + "_mse_dist.png")
    plot_mse_distributions(results, plot_path)

    # Plot per-step MSE distributions for pos_x / pos_y / pos_z / mean
    pos_plot_path = out_path.with_name(out_path.stem + "_pos_step")
    plot_pos_step_mse(results, pos_plot_path)


if __name__ == "__main__":
    main()
