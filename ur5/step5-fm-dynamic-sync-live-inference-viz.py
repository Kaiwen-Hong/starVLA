#!/usr/bin/env python3
"""
Step 5 (FM Dynamic): Live inference visualization for QwenPI flow-matching.

Interactive: load model + camera + robot once, then press Enter to re-infer.
Robot does NOT move.

Usage:
    python ur5/step5-fm-dynamic-sync-live-inference-viz.py
    python ur5/step5-fm-dynamic-sync-live-inference-viz.py --include_state
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
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework
from scipy.spatial.transform import Rotation

import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

import modular_policy
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}

# ── Edit these before running ──────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/DiscreteRTC/fastumi_pickandplace_qwenPI_dynamic_1/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
INSTRUCTION = "pick up the block in the pot and place it in the red area on the turntable"
CAMERA_DEV = 1
ARM = "left"
# ───────────────────────────────────────────────────────────────────


# ── Math utilities ─────────────────────────────────────────────────

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def axisangle_to_rot6d(rx, ry, rz):
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot6d_to_mat(d6)).as_rotvec().astype(np.float32)


def ee_pose_to_state10d(pose_6d, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d[:6]
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def action_10d_to_delta7d(action_10d: np.ndarray) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = action_10d[:3]
    delta[3:6] = rot6d_to_axisangle(action_10d[3:9])
    delta[6] = action_10d[9]
    return delta


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    return np.array([action_10d_to_delta7d(a) for a in actions_10d])


def accumulate_deltas(current_pose, deltas_7d):
    T = deltas_7d.shape[0]
    poses = np.zeros((T, 7), dtype=np.float64)
    pos = np.array(current_pose[:6], dtype=np.float64)
    for t in range(T):
        pos = pos + deltas_7d[t, :6].astype(np.float64)
        poses[t, :6] = pos
        poses[t, 6] = deltas_7d[t, 6]
    return poses


# ── Model loading ──────────────────────────────────────────────────

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
    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    return model


def get_action_stats(model):
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    return action_stats, dataset_key


TRAIN_IMAGE_SIZE = (224, 224)


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    # Resize to match training resolution (dataset _pack_sample resizes to 224x224)
    image = image.resize(TRAIN_IMAGE_SIZE)
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(world_poses, camera_image, current_ee, step_idx, save_path):
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
        f'Run {step_idx}: Live Inference (FM Dynamic, no movement)\n"{INSTRUCTION}"',
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="FM Dynamic: live inference viz (no robot movement)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default=ARM)
    parser.add_argument("--camera_dev", type=int, default=CAMERA_DEV)
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--gripper_state", type=float, default=0.0,
                        help="Current gripper state: 0=open, 1=closed")
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    args = parser.parse_args()

    # ── 1. Load model ────────────────────────────────────────────────
    model = load_model(args.checkpoint)
    action_stats, dataset_key = get_action_stats(model)

    # ── 2. Connect to robot (read-only) ─────────────────────────────
    from rtde_receive import RTDEReceiveInterface

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Robot connected (read-only, no movement).")

    # ── 3. Open camera ───────────────────────────────────────────────
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
    print(f"  Instruction: {args.instruction}")
    print(f"  Framework: QwenPI (flow-matching)")
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

            # Inference (flow matching — no temperature params)
            state_for_model = current_state_10d if args.include_state else None
            example = build_example(pil_img, args.instruction, state_10d=state_for_model)

            print("Running predict_action...")
            t0 = time.time()
            output = model.predict_action(examples=[example])
            print(f"Inference done in {time.time() - t0:.3f}s")

            pred_normalized = output["normalized_actions"][0].astype(np.float32)
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # Convert 10D -> 7D
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

            # Accumulate -> absolute trajectory
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

            # Visualize
            save_path = Path("ur5") / f"step5_fm_dynamic_viz_{timestamp}.png"
            visualize(world_poses, pil_img, pose_world, run_idx, str(save_path))

            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
