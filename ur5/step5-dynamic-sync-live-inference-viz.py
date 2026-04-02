#!/usr/bin/env python3
"""
Step 5 (Dynamic): Live inference visualization for discrete diffusion.

Interactive: load model + camera + robot once, then press Enter to re-infer.
Robot does NOT move.

Usage:
    python ur5/step5-dynamic-sync-live-inference-viz.py
    python ur5/step5-dynamic-sync-live-inference-viz.py --include_state
"""

import sys
import os
import time
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.base_framework import baseframework
from scripts.robo_utils import (
    load_model, get_action_stats, build_example,
    base_to_world, ee_pose_to_state10d,
    actions_10d_to_7d, accumulate_deltas,
    BASE_IN_WORLD, ROBOT_IPS,
)

# ── Edit these before running ──────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v2/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
INSTRUCTION = "Pick up the purple block and place it on the red area of the board"
CAMERA_DEV = 0
ARM = "left"
# DD-specific
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1
# ───────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(world_poses, camera_image, current_ee, step_idx, save_path,
              raw_normalized=None, raw_unnormalized=None):
    """Camera image (left) + raw policy output (middle) + absolute world poses (right)."""
    T = world_poses.shape[0]
    ts = np.arange(T) / 20.0

    n_cols = 2  # image + world
    if raw_normalized is not None:
        n_cols += 1
    if raw_unnormalized is not None:
        n_cols += 1

    fig = plt.figure(figsize=(6 * n_cols, 10))
    ratios = [1] + [1.3] * (n_cols - 1)
    gs = GridSpec(4, n_cols, figure=fig, hspace=0.15, wspace=0.35,
                  width_ratios=ratios)

    # ── Column 0: Camera image ──
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    col = 1

    # ── Column: Raw normalized output ([-1,1] / [0,1]) ──
    if raw_normalized is not None:
        raw_dims = [
            (0, "norm pos_x", "#e41a1c"),
            (1, "norm pos_y", "#377eb8"),
            (2, "norm pos_z", "#4daf4a"),
            (9, "norm gripper", "#ff7f00"),
        ]
        axes_raw = []
        for row, (dim_idx, label, color) in enumerate(raw_dims):
            share = axes_raw[0] if axes_raw else None
            ax = fig.add_subplot(gs[row, col], sharex=share)
            axes_raw.append(ax)
            ax.plot(ts, raw_normalized[:T, dim_idx], "o-", markersize=3, linewidth=1.5, color=color)
            ax.set_ylabel(label, fontsize=9, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            if row < len(raw_dims) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time (s)", fontsize=9)
        axes_raw[0].set_title("Raw Normalized", fontsize=11, fontweight="bold")
        col += 1

    # ── Column: Unnormalized delta actions ──
    if raw_unnormalized is not None:
        delta_dims = [
            (0, "delta pos_x (m)", "#e41a1c"),
            (1, "delta pos_y (m)", "#377eb8"),
            (2, "delta pos_z (m)", "#4daf4a"),
            (9, "gripper", "#ff7f00"),
        ]
        axes_delta = []
        for row, (dim_idx, label, color) in enumerate(delta_dims):
            share = axes_delta[0] if axes_delta else None
            ax = fig.add_subplot(gs[row, col], sharex=share)
            axes_delta.append(ax)
            ax.plot(ts, raw_unnormalized[:T, dim_idx], "o-", markersize=3, linewidth=1.5, color=color)
            ax.set_ylabel(label, fontsize=9, fontweight="bold")
            ax.grid(True, axis="y", alpha=0.3)
            if row < len(delta_dims) - 1:
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel("Time (s)", fontsize=9)
        axes_delta[0].set_title("Unnormalized (10D)", fontsize=11, fontweight="bold")
        col += 1

    # ── Last column: Absolute world-frame trajectory ──
    world_dims = [
        (0, "x (world, m)", "#e41a1c", current_ee[0]),
        (1, "y (world, m)", "#377eb8", current_ee[1]),
        (2, "z (world, m)", "#4daf4a", current_ee[2]),
        (6, "gripper",      "#ff7f00", None),
    ]
    axes_world = []
    for row, (dim_idx, label, color, start_val) in enumerate(world_dims):
        share = axes_world[0] if axes_world else None
        ax = fig.add_subplot(gs[row, col], sharex=share)
        axes_world.append(ax)
        vals = world_poses[:, dim_idx]
        ax.plot(ts, vals, "o-", markersize=3, linewidth=1.5, color=color)
        if start_val is not None:
            ax.axhline(start_val, color=color, linewidth=1.0, linestyle="--",
                       alpha=0.5, label=f"current={start_val:.4f}")
            ax.legend(fontsize=7, loc="upper right")
        ax.set_ylabel(label, fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if row < len(world_dims) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s)", fontsize=9)
    axes_world[0].set_title("World Frame (accumulated)", fontsize=11, fontweight="bold")

    fig.suptitle(
        f'Run {step_idx}: Live Inference (DD Dynamic, no movement)\n"{INSTRUCTION}"',
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="DD Dynamic: live inference viz (no robot movement)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default=ARM)
    parser.add_argument("--camera_dev", type=int, default=CAMERA_DEV)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--gripper_state", type=float, default=0.0,
                        help="Current gripper state: 0=open, 1=closed")
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    args = parser.parse_args()

    # ── 1. Load model ────────────────────────────────────────────────
    model = load_model(args.checkpoint)
    action_stats, dataset_key = get_action_stats(model)

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    # ── 2. Connect to robot (read-only) ─────────────────────────────
    from rtde_receive import RTDEReceiveInterface

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Robot connected (read-only, no movement).")

    # ── 3. Open camera ───────────────────────────────────────────────
    import importlib
    _step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
    RealCamera = _step2.RealCamera

    print(f"\nOpening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.\n")

    # ── 4. Interactive loop ──────────────────────────────────────────
    run_idx = 0
    print("=" * 60)
    print("  Interactive mode: press Enter to run inference")
    print("  Type 'q' + Enter to quit")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Instruction: {INSTRUCTION}")
    print(f"  decode_temp={args.decode_temperature}  choice_temp={args.choice_temperature}")
    print("=" * 60)

    try:
        while True:
            user_input = input(f"\n[Run {run_idx}] Press Enter to infer (q to quit): ").strip()
            if user_input.lower() == 'q':
                break

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Read current robot state
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_state_10d = ee_pose_to_state10d(pose_world, args.gripper_state)

            print(f"EE pose (world): {[round(x, 4) for x in pose_world]}")

            # Grab camera frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, try again")
                continue

            # Inference
            state_for_model = current_state_10d if args.include_state else None
            example = build_example(pil_img, INSTRUCTION, state_10d=state_for_model)

            print("Running predict_action...")
            t0 = time.time()
            output = model.predict_action(examples=[example], **infer_kwargs)
            print(f"Inference done in {time.time() - t0:.3f}s")

            pred_normalized = output["normalized_actions"][0].astype(np.float32)
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # Convert 10D -> 7D (for display only)
            deltas_7d = actions_10d_to_7d(pred_actions_10d)

            # Print delta actions (these are in EE LOCAL frame, not world frame)
            print(f"\n{'=' * 82}")
            print("Delta actions (EE local frame)  [dx, dy, dz, drx, dry, drz, gripper]")
            print(f"{'=' * 82}")
            labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
            print(f"{'step':>4s}  " + "  ".join(f"{l:>9s}" for l in labels))
            print("-" * 82)
            for t in range(deltas_7d.shape[0]):
                a = deltas_7d[t]
                g = "close" if a[6] > 0.5 else "open"
                print(f"{t:4d}  {a[0]:9.6f}  {a[1]:9.6f}  {a[2]:9.6f}  "
                      f"{a[3]:9.6f}  {a[4]:9.6f}  {a[5]:9.6f}  {g:>9s}")

            # Accumulate -> absolute trajectory (local→world transform)
            world_poses = accumulate_deltas(pose_world, pred_actions_10d)

            print(f"\n{'=' * 82}")
            print("Absolute trajectory (world frame)  [x, y, z, rx, ry, rz, gripper]")
            print(f"{'=' * 82}")
            abs_labels = ["x", "y", "z", "rx", "ry", "rz", "gripper"]
            print(f"{'step':>4s}  " + "  ".join(f"{l:>9s}" for l in abs_labels))
            print("-" * 82)
            print(f"{'now':>4s}  {pose_world[0]:9.4f}  {pose_world[1]:9.4f}  "
                  f"{pose_world[2]:9.4f}  {pose_world[3]:9.4f}  "
                  f"{pose_world[4]:9.4f}  {pose_world[5]:9.4f}  "
                  f"{'open' if args.gripper_state < 0.5 else 'close':>9s}")
            for t in range(world_poses.shape[0]):
                p = world_poses[t]
                g = "close" if p[6] > 0.5 else "open"
                print(f"{t:4d}  {p[0]:9.4f}  {p[1]:9.4f}  {p[2]:9.4f}  "
                      f"{p[3]:9.4f}  {p[4]:9.4f}  {p[5]:9.4f}  {g:>9s}")
            print(f"{'=' * 82}")

            # Visualize
            save_path = Path("ur5") / f"step5_dynamic_sync_viz_{timestamp}.png"
            visualize(world_poses, pil_img, pose_world, run_idx, str(save_path),
                      raw_normalized=pred_normalized,
                      raw_unnormalized=pred_actions_10d)

            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
