#!/usr/bin/env python3
"""
Step 5: Full inference pipeline with live robot state (visualization only).

Interactive: load model + camera + robot once, then press Enter to re-infer.

Reads the current EE pose from the real UR5 robot via RTDE,
runs model inference with the real camera image,
denormalizes the predicted delta actions (world-frame),
and accumulates them on the current pose to show the predicted trajectory.

Robot does NOT move.

Pipeline (per Enter press):
    1. Read current EE pose from robot (RTDE) → base → world frame
    2. Grab camera frame
    3. Model inference → normalized delta actions (T, 10)
    4. Denormalize
    5. Convert rot6d → axis-angle (delta actions, 10D → 7D)
    6. Accumulate: target_pose = current_pose + cumsum(delta) → absolute trajectory
    7. Visualize + print

Usage:
    python ur5/step5-live-inference-viz.py
    python ur5/step5-live-inference-viz.py --arm left
    python ur5/step5-live-inference-viz.py --include_state
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
from scipy.spatial.transform import Rotation
from PIL import Image

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

# ── Import camera from step2 ────────────────────────────────────────
import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

# ── Robot extrinsics ────────────────────────────────────────────────
import modular_policy

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}

ROBOT_IPS = {
    'left': '192.168.0.3',
    'right': '192.168.0.2',
}

INSTRUCTION = "pick up the building block"


# ═══════════════════════════════════════════════════════════════════
#  Conversion utilities
# ═══════════════════════════════════════════════════════════════════

def base_to_world(pose_base, T_bw):
    """[x,y,z,rx,ry,rz] base → world. R_bw=I, translation only."""
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    """rot6d (6,) → axis-angle (3,) via Gram-Schmidt."""
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=0)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def axisangle_to_rot6d(rx, ry, rz):
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def ee_pose_to_state10d(pose_6d, gripper: float) -> np.ndarray:
    """[x,y,z,rx,ry,rz] + gripper → 10D [xyz, rot6d(6), gripper]."""
    x, y, z, rx, ry, rz = pose_6d[:6]
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    """(T,10) delta actions: rot6d → axis-angle. Returns (T,7) [dx,dy,dz,drx,dry,drz,grip]."""
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t, :3] = actions_10d[t, :3]
        out[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        out[t, 6] = actions_10d[t, 9]
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
    """
    Accumulate world-frame delta actions on current pose.

    current_pose_world: [x, y, z, rx, ry, rz] (6,)
    deltas_7d:          (T, 7) [dx, dy, dz, drx, dry, drz, gripper]

    Returns: (T, 7) absolute world-frame poses [x, y, z, rx, ry, rz, gripper]
    """
    T = deltas_7d.shape[0]
    poses = np.zeros((T, 7), dtype=np.float64)
    pos = np.array(current_pose_world[:3], dtype=np.float64)
    rot = np.array(current_pose_world[3:6], dtype=np.float64)

    for t in range(T):
        pos = pos + deltas_7d[t, :3]
        rot = rot + deltas_7d[t, 3:6]
        poses[t, :3] = pos
        poses[t, 3:6] = rot
        poses[t, 6] = deltas_7d[t, 6]

    return poses


# ═══════════════════════════════════════════════════════════════════
#  Model loading
# ═══════════════════════════════════════════════════════════════════

def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path: str):
    import torch

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
    print(f"Model loaded in {time.time() - t0:.1f}s")
    return model


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(world_poses, camera_image, current_ee, save_path):
    """Camera image (left) + absolute x/y/z/gripper (right)."""
    T = world_poses.shape[0]
    ts = np.arange(T) / 20.0

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.15, wspace=0.30,
                  width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    dims = [
        (0, "x (world, m)", "#e41a1c", current_ee[0]),
        (1, "y (world, m)", "#377eb8", current_ee[1]),
        (2, "z (world, m)", "#4daf4a", current_ee[2]),
        (6, "gripper",      "#ff7f00", None),
    ]

    axes = []
    for row, (dim_idx, label, color, start_val) in enumerate(dims):
        share = axes[0] if axes else None
        ax = fig.add_subplot(gs[row, 1], sharex=share)
        axes.append(ax)

        vals = world_poses[:, dim_idx]
        ax.plot(ts, vals, "o-", markersize=4, linewidth=1.8, color=color)
        if start_val is not None:
            ax.axhline(start_val, color=color, linewidth=1.0, linestyle="--",
                       alpha=0.5, label=f"current={start_val:.4f}")
            ax.legend(fontsize=8, loc="upper right")
        for t_val in ts:
            ax.axvline(t_val, color="gray", linewidth=0.3, alpha=0.2)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if row < len(dims) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s) — 20Hz", fontsize=10)

    fig.suptitle(
        f'Step 5: Live Robot State + Predicted Trajectory\n"{INSTRUCTION}"',
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Full inference pipeline with live robot state (visualization only)")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/DiscreteRTC/"
                "fastumi_pickandplace_discrete_diffusion_real_0314_no_state/"
                "checkpoints/steps_20000_pytorch_model.pt",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--gripper_state", type=float, default=0.0,
                        help="Current gripper state: 0=open, 1=closed (default: 0)")
    args = parser.parse_args()

    # ── 1. Load model (one-time) ─────────────────────────────────────
    model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', modes={action_stats.get('norm_modes', 'legacy')}")

    # ── 2. Connect to robot (one-time) ───────────────────────────────
    from rtde_receive import RTDEReceiveInterface

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Robot connected.")

    # ── 3. Open camera (one-time, kept open) ─────────────────────────
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

            # Grab fresh camera frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, try again")
                continue

            # Inference
            state_for_model = current_state_10d if args.include_state else None
            example = build_example(pil_img, INSTRUCTION, state_10d=state_for_model)

            print("Running predict_action...")
            t0 = time.time()
            output = model.predict_action(examples=[example])
            print(f"Inference done in {time.time() - t0:.3f}s")

            pred_normalized = output["normalized_actions"][0].astype(np.float32)

            # Denormalize
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # Convert 10D → 7D (rot6d → axis-angle)
            deltas_7d = actions_10d_to_7d(pred_actions_10d)

            # Print delta actions
            print(f"\n{'=' * 82}")
            print("Delta actions (world frame)  [dx, dy, dz, drx, dry, drz, gripper]")
            print(f"{'=' * 82}")
            labels = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
            print(f"{'step':>4s}  " + "  ".join(f"{l:>9s}" for l in labels))
            print("-" * 82)
            for t in range(deltas_7d.shape[0]):
                a = deltas_7d[t]
                g = "close" if a[6] > 0.5 else "open"
                print(f"{t:4d}  {a[0]:9.6f}  {a[1]:9.6f}  {a[2]:9.6f}  "
                      f"{a[3]:9.6f}  {a[4]:9.6f}  {a[5]:9.6f}  {g:>9s}")

            # Accumulate deltas → absolute world-frame trajectory
            world_poses = accumulate_deltas(pose_world, deltas_7d)

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

            # Visualize with timestamp
            save_path = Path("ur5") / f"step5_live_inference_viz_{timestamp}.png"
            visualize(world_poses, pil_img, pose_world, str(save_path))

            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
