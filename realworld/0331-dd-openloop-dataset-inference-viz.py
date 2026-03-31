#!/usr/bin/env python3
"""
Open-loop dataset inference with visualization (Discrete Diffusion).

Runs QwenDiscreteDiffusion predict_action on dataset samples and saves
per-sample visualizations: image + normalized 10D actions + world-frame 7D deltas.

Dataset preparation (run in /home/kaiwen/Desktop/research/fastumipro-collection/0srarvla-lerobo):
    python 0srarvla-lerobo/V3-convert_to_starvla_cropped-augment-mp.py \
        -d data_collector_opt/DATA/left_hand_250801DR48FP25002960 \
        -c dynamic-329-v2 \
        -o 0srarvla-lerobo/starvla/datasets/ \
        --augment-versions 0 5 10 15 20 \
        --num-workers 16 \
        --task "Pick up the purple block and place it on the red area of the board"

Usage:
    python realworld/0331-dd-openloop-dataset-inference-viz.py
    python realworld/0331-dd-openloop-dataset-inference-viz.py --num_samples 50
    python realworld/0331-dd-openloop-dataset-inference-viz.py --use_simple_max
    python realworld/0331-dd-openloop-dataset-inference-viz.py --decode_temperature 0.5
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.spatial.transform import Rotation
from tqdm import tqdm

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v2/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DATA_ROOT_DIR = "/home/kaiwen/Desktop/research/fastumipro-collection/0srarvla-lerobo/starvla/datasets"
INSTRUCTION = "Pick up the purple block and place it on the red area of the board"

DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",
    "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]
DIM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#a65628", "#f781bf",
    "#999999", "#66c2a5", "#fc8d62",
    "#ff7f00",
]

WORLD_DIM_LABELS = [
    "world_dx", "world_dy", "world_dz",
    "world_droll", "world_dpitch", "world_dyaw",
    "gripper",
]
WORLD_DIM_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a",
    "#984ea3", "#a65628", "#f781bf",
    "#ff7f00",
]


# ── Rotation / world-frame utilities ─────────────────────────────

def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """6D rotation representation -> 3x3 rotation matrix."""
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix -> (roll, pitch, yaw) in radians."""
    return Rotation.from_matrix(R).as_euler('xyz').astype(np.float64)


def reconstruct_absolute_trajectory(actions):
    """Reconstruct absolute trajectory from relative 10D actions.

    Position: cumsum in world frame (local delta rotated by current orientation).
    Rotation: sequential matrix composition -> euler angles.

    Returns positions (N+1, 3), euler_angles (N+1, 3), grippers (N+1,).
    """
    N = len(actions)
    positions = np.zeros((N + 1, 3), dtype=np.float64)
    rot_mats = [np.eye(3)]
    grippers = np.zeros(N + 1)
    grippers[0] = actions[0, 9]

    for t in range(N):
        R_abs = rot_mats[-1]
        positions[t + 1] = positions[t] + R_abs @ actions[t, :3]
        R_rel = rot6d_to_mat(actions[t, 3:9])
        rot_mats.append(R_abs @ R_rel)
        grippers[t + 1] = actions[t, 9]

    euler_angles = np.zeros((N + 1, 3), dtype=np.float64)
    for t, R in enumerate(rot_mats):
        euler_angles[t] = rotation_matrix_to_euler(R)

    return positions, euler_angles, grippers


def relative_actions_to_world_deltas(actions):
    """Convert relative 10D actions -> world-frame 7D deltas.

    Returns (N, 7): [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """
    positions, euler_angles, grippers = reconstruct_absolute_trajectory(actions)
    N = len(actions)
    deltas = np.zeros((N, 7), dtype=np.float32)
    for t in range(N):
        deltas[t, :3] = positions[t + 1] - positions[t]
        deltas[t, 3:6] = euler_angles[t + 1] - euler_angles[t]
        deltas[t, 6] = grippers[t + 1]
    return deltas


# ── Model loading ──────────────────────────────────────────────────

def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path: str):
    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()

    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn_implementation()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    return model


def load_dataset(model, data_root_dir: str = None):
    data_cfg = model.config.datasets.vla_data
    if data_root_dir:
        data_cfg.data_root_dir = data_root_dir

    print(f"Loading dataset: {data_cfg.data_mix}")
    print(f"  root: {data_cfg.data_root_dir}")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    print(f"  total steps: {len(dataset)}")
    return dataset


def build_sample_indices(dataset, only_first_half: bool, randomly_sample: bool):
    """Build list of dataset indices, optionally filtering to first half of each trajectory."""
    single_ds = dataset.datasets[0]
    all_steps = single_ds.all_steps  # list of (traj_id, base_index)

    if only_first_half:
        # Group steps by trajectory, keep first half of each
        traj_groups = OrderedDict()
        for global_idx, (traj_id, base_idx) in enumerate(all_steps):
            if traj_id not in traj_groups:
                traj_groups[traj_id] = []
            traj_groups[traj_id].append(global_idx)

        indices = []
        for traj_id, step_indices in traj_groups.items():
            half = len(step_indices) // 2
            indices.extend(step_indices[:half])
        print(f"  only_first_half: {len(indices)}/{len(all_steps)} steps "
              f"({len(traj_groups)} trajectories)")
    else:
        indices = list(range(len(all_steps)))

    if randomly_sample:
        rng = np.random.default_rng(42)
        rng.shuffle(indices)

    return indices


# ── Visualization ──────────────────────────────────────────────────

def visualize_sample(image, pred_norm, gt_norm, pred_world, gt_world,
                     sample_idx, chunk_len, instruction, save_path, mse, l1):
    """Visualize one sample: image (left) + normalized 10D (middle) + world 7D (right).

    pred_norm, gt_norm: (T, 10) normalized relative actions.
    pred_world, gt_world: (T, 7) world-frame deltas [dx,dy,dz,droll,dpitch,dyaw,gripper].
    """
    fig = plt.figure(figsize=(28, 16))
    gs_main = GridSpec(1, 3, figure=fig, width_ratios=[1, 1.3, 1.3], wspace=0.25)

    T = pred_norm.shape[0]
    ts = np.arange(T)

    # ── Left: image ──────────────────────────────────────────────
    ax_img = fig.add_subplot(gs_main[0, 0])
    ax_img.imshow(image)
    ax_img.set_title(f"Dataset image (sample {sample_idx})",
                     fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # ── Middle: normalized 10D relative actions ──────────────────
    gs_mid = GridSpecFromSubplotSpec(
        6, 1, subplot_spec=gs_main[0, 1],
        height_ratios=[1, 1, 1, 0.6, 0.6, 1], hspace=0.35)

    def _plot_norm_dim(ax, dim_idx, show_xlabel=False):
        color = DIM_COLORS[dim_idx]
        ax.plot(ts, pred_norm[:, dim_idx], "o-", markersize=3, linewidth=1.5,
                color=color, label="pred")
        ax.plot(ts, gt_norm[:, dim_idx], "x--", markersize=3, linewidth=1.2,
                color=color, alpha=0.6, label="GT")
        ax.set_ylabel(DIM_LABELS[dim_idx], fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        if not show_xlabel:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Timestep", fontsize=9)

    # pos_x, pos_y, pos_z
    for i in range(3):
        ax = fig.add_subplot(gs_mid[i])
        _plot_norm_dim(ax, i)

    # Rotation: 2 rows x 3 cols
    for rot_row in range(2):
        gs_rot = GridSpecFromSubplotSpec(
            1, 3, subplot_spec=gs_mid[3 + rot_row], wspace=0.35)
        for col in range(3):
            dim_idx = 3 + rot_row * 3 + col
            ax = fig.add_subplot(gs_rot[0, col])
            color = DIM_COLORS[dim_idx]
            ax.plot(ts, pred_norm[:, dim_idx], "o-", markersize=2, linewidth=1.2,
                    color=color)
            ax.plot(ts, gt_norm[:, dim_idx], "x--", markersize=2, linewidth=1.0,
                    color=color, alpha=0.6)
            ax.set_title(DIM_LABELS[dim_idx], fontsize=8)
            ax.tick_params(labelsize=7)
            ax.grid(True, axis="y", alpha=0.3)
            if rot_row == 0:
                plt.setp(ax.get_xticklabels(), visible=False)

    # Gripper
    ax = fig.add_subplot(gs_mid[5])
    _plot_norm_dim(ax, 9, show_xlabel=True)

    # ── Right: world-frame 7D deltas ─────────────────────────────
    gs_right = GridSpecFromSubplotSpec(
        7, 1, subplot_spec=gs_main[0, 2], hspace=0.4)

    for i in range(7):
        ax = fig.add_subplot(gs_right[i])
        ax.plot(ts, pred_world[:, i], "o-", markersize=3, linewidth=1.5,
                color=WORLD_DIM_COLORS[i], label="pred")
        ax.plot(ts, gt_world[:, i], "x--", markersize=3, linewidth=1.2,
                color=WORLD_DIM_COLORS[i], alpha=0.6, label="GT")
        ax.set_ylabel(WORLD_DIM_LABELS[i], fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
        if i < 6:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Timestep", fontsize=9)

    fig.suptitle(
        f'Sample {sample_idx} | MSE={mse:.6f}  L1={l1:.6f} | chunk_len={chunk_len}\n'
        f'"{instruction}"',
        fontsize=13, fontweight="bold", y=0.99)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Open-loop dataset inference with visualization (Discrete Diffusion)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR)
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--only_first_half", action="store_true", default=True)
    parser.add_argument("--no_only_first_half", dest="only_first_half",
                        action="store_false")
    parser.add_argument("--randomly_sample", action="store_true", default=False)
    parser.add_argument("--rollout_dir", type=str, default=None)
    # Discrete diffusion specific
    parser.add_argument("--decode_temperature", type=float, default=0.0,
                        help="Temperature for MaskGIT decode (default: 0.0)")
    parser.add_argument("--choice_temperature", type=float, default=0.1,
                        help="Temperature for token choice (default: 0.1)")
    parser.add_argument("--use_simple_max", action="store_true", default=False,
                        help="Use argmax (faster, deterministic, skips iterative decode)")
    args = parser.parse_args()

    model = load_model(args.checkpoint)
    dataset = load_dataset(model, data_root_dir=args.data_root_dir)
    chunk_len = model.chunk_len

    # Get norm stats for denormalization
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', modes={action_stats.get('norm_modes', 'legacy')}")

    # Build filtered indices
    indices = build_sample_indices(dataset, args.only_first_half, args.randomly_sample)
    num_samples = min(args.num_samples, len(indices))
    print(f"  will evaluate {num_samples} samples")

    # ── Rollout dir ──────────────────────────────────────────────
    ts = time.strftime("%Y%m%d_%H%M%S")
    rollout_dir = Path(args.rollout_dir) if args.rollout_dir else (
        Path("realworld") / "rollouts" / f"dd_openloop_{ts}")
    rollout_dir.mkdir(parents=True, exist_ok=True)
    (rollout_dir / "images").mkdir(exist_ok=True)

    with open(rollout_dir / "config.json", "w") as f:
        json.dump({
            "checkpoint": args.checkpoint,
            "instruction": args.instruction,
            "num_samples": num_samples,
            "data_root_dir": args.data_root_dir,
            "chunk_len": chunk_len,
            "only_first_half": args.only_first_half,
            "randomly_sample": args.randomly_sample,
            "decode_temperature": args.decode_temperature,
            "choice_temperature": args.choice_temperature,
            "use_simple_max": args.use_simple_max,
        }, f, indent=2)
    print(f"Saving to: {rollout_dir}")

    # ── Inference loop ───────────────────────────────────────────
    all_mse = []
    all_l1 = []
    step_log = []

    single_ds = dataset.datasets[0]

    for count, ds_idx in enumerate(tqdm(indices[:num_samples], desc="Evaluating")):
        sample = single_ds[ds_idx]
        gt_norm = np.array(sample["action"], dtype=np.float32)  # [T, 10]

        # Override instruction
        sample["lang"] = args.instruction
        batch = [sample]

        # Inference (discrete diffusion)
        t0 = time.monotonic()
        output = model.predict_action(
            examples=batch,
            decode_temperature=args.decode_temperature,
            choice_temperature=args.choice_temperature,
            use_simple_max=args.use_simple_max,
        )
        infer_ms = (time.monotonic() - t0) * 1000
        pred_norm = output["normalized_actions"][0].astype(np.float32)

        # Align lengths
        if gt_norm.shape[0] > chunk_len:
            gt_norm = gt_norm[-chunk_len:, :]
        T = min(pred_norm.shape[0], gt_norm.shape[0])
        pred_norm = pred_norm[:T]
        gt_norm = gt_norm[:T]

        # Metrics (on normalized actions)
        mse = float(np.mean((pred_norm - gt_norm) ** 2))
        l1 = float(np.mean(np.abs(pred_norm - gt_norm)))
        all_mse.append(mse)
        all_l1.append(l1)

        # Denormalize for world-frame visualization
        pred_denorm = baseframework.unnormalize_actions(pred_norm, action_stats)
        gt_denorm = baseframework.unnormalize_actions(gt_norm, action_stats)

        # Convert to world-frame 7D deltas
        pred_world = relative_actions_to_world_deltas(pred_denorm)
        gt_world = relative_actions_to_world_deltas(gt_denorm)

        # Dataset image for visualization (first view)
        dataset_image = sample["image"][0]  # PIL Image

        # Save visualization
        viz_path = rollout_dir / "images" / f"sample_{count:04d}_viz.png"
        visualize_sample(dataset_image, pred_norm, gt_norm,
                         pred_world, gt_world,
                         count, chunk_len, args.instruction,
                         str(viz_path), mse, l1)

        # Which trajectory / position is this?
        traj_id, base_idx = single_ds.all_steps[ds_idx]

        step_log.append({
            "sample_idx": count,
            "dataset_idx": ds_idx,
            "trajectory_id": int(traj_id) if isinstance(traj_id, (int, np.integer)) else str(traj_id),
            "base_index": int(base_idx),
            "mse": mse,
            "l1": l1,
            "infer_ms": round(infer_ms, 1),
        })

    # ── Summary ──────────────────────────────────────────────────
    overall_mse = float(np.mean(all_mse))
    overall_l1 = float(np.mean(all_l1))

    summary = {
        "num_samples": len(all_mse),
        "chunk_len": chunk_len,
        "only_first_half": args.only_first_half,
        "randomly_sample": args.randomly_sample,
        "decode_temperature": args.decode_temperature,
        "choice_temperature": args.choice_temperature,
        "use_simple_max": args.use_simple_max,
        "overall_mse": overall_mse,
        "overall_l1": overall_l1,
        "overall_mse_std": float(np.std(all_mse)),
        "overall_l1_std": float(np.std(all_l1)),
        "per_sample": step_log,
    }
    with open(rollout_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"  Open-Loop Dataset Inference (Discrete Diffusion)")
    print(f"{'=' * 60}")
    print(f"  Samples:              {len(all_mse)}")
    print(f"  Chunk len:            {chunk_len}")
    print(f"  Only first half:      {args.only_first_half}")
    print(f"  Random sample:        {args.randomly_sample}")
    print(f"  decode_temperature:   {args.decode_temperature}")
    print(f"  choice_temperature:   {args.choice_temperature}")
    print(f"  use_simple_max:       {args.use_simple_max}")
    print(f"  Overall MSE:          {overall_mse:.6f} (+/- {float(np.std(all_mse)):.6f})")
    print(f"  Overall L1:           {overall_l1:.6f} (+/- {float(np.std(all_l1)):.6f})")
    print(f"  Saved to:             {rollout_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
