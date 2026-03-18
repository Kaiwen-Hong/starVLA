#!/usr/bin/env python3
"""
Step 4: Denormalize model predictions and convert rot6d -> axis-angle (test only).
Discrete Diffusion version.

Runs inference (same as step3), denormalizes the predicted relative actions,
and converts the rotation from rot6d (6D) to axis-angle (3D) for UR5 readability.
Robot does NOT move. Outputs visualization image.

Usage:
    python ur5-dd/step4-convert-back-to-robot-action-test.py
    python ur5-dd/step4-convert-back-to-robot-action-test.py --use_simple_max
    python ur5-dd/step4-convert-back-to-robot-action-test.py --decode_temperature 0.5
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

# ── Import camera from ur5/step2 ───────────────────────────────────
import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera


# ── Defaults ─────────────────────────────────────────────────────────
FIXED_EE_POSE = [0.327694, -0.194824, 0.11828, 2.15647483, -2.21923892, -0.11130583]
FIXED_GRIPPER = 0.0
INSTRUCTION = "pick up the building block"


# ═══════════════════════════════════════════════════════════════════
#  Conversion utilities
# ═══════════════════════════════════════════════════════════════════

def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=0)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    T = actions_10d.shape[0]
    actions_7d = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        actions_7d[t, :3] = actions_10d[t, :3]
        actions_7d[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        actions_7d[t, 6] = actions_10d[t, 9]
    return actions_7d


def axisangle_to_rot6d(rx: float, ry: float, rz: float) -> np.ndarray:
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def ee_pose_to_state10d(pose_6d: list, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════
#  Model loading & inference
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
    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
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

DIM_LABELS_7D = ["rel_x", "rel_y", "rel_z", "rel_rx", "rel_ry", "rel_rz", "gripper"]


def visualize(actions_7d: np.ndarray, camera_image: Image.Image, save_path: str):
    T = actions_7d.shape[0]
    ts = np.arange(T) / 20.0

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.15, wspace=0.30,
                  width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    dims = [
        (0, "rel_x (m)", "#e41a1c"),
        (1, "rel_y (m)", "#377eb8"),
        (2, "rel_z (m)", "#4daf4a"),
        (6, "gripper",   "#ff7f00"),
    ]

    axes = []
    for row, (dim_idx, label, color) in enumerate(dims):
        share = axes[0] if axes else None
        ax = fig.add_subplot(gs[row, 1], sharex=share)
        axes.append(ax)

        vals = actions_7d[:, dim_idx]
        ax.plot(ts, vals, "o-", markersize=4, linewidth=1.8, color=color)
        ax.fill_between(ts, 0, vals, alpha=0.15, color=color)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for t_val in ts:
            ax.axvline(t_val, color="gray", linewidth=0.3, alpha=0.2)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if row < len(dims) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s) — 20Hz", fontsize=10)

    fig.suptitle(
        f'Step 4 (Discrete Diffusion): Denormalized Relative Actions\n"{INSTRUCTION}"',
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Denormalize predictions and convert to UR5 format (Discrete Diffusion)")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/DiscreteRTC/"
                "fastumi_pickandplace_discrete_diffusion_real_0314_no_state_fixgripper/"
                "checkpoints/steps_15000_pytorch_model.pt",
    )
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--include_state", action="store_true", default=False)
    # Discrete diffusion specific
    parser.add_argument("--decode_temperature", type=float, default=0.1)
    parser.add_argument("--choice_temperature", type=float, default=0.1)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    args = parser.parse_args()

    # 1. Load model
    model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', modes={action_stats.get('norm_modes', 'legacy')}")

    # 2. State setup
    state_10d = None
    if args.include_state:
        state_10d = ee_pose_to_state10d(FIXED_EE_POSE, FIXED_GRIPPER)
        print(f"\nState 10D: {state_10d}")
    else:
        print("\nState: not included (no_state model)")
    print(f"Instruction: {INSTRUCTION}")
    print(f"Decode params: temp={args.decode_temperature}, "
          f"choice_temp={args.choice_temperature}, simple_max={args.use_simple_max}")

    # 3. Open camera
    print(f"\nOpening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.\n")

    # 4. Interactive loop
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

            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, try again")
                continue

            example = build_example(pil_img, INSTRUCTION, state_10d=state_10d)
            print("Running predict_action (discrete diffusion)...")
            t0 = time.time()
            output = model.predict_action(
                examples=[example],
                decode_temperature=args.decode_temperature,
                choice_temperature=args.choice_temperature,
                use_simple_max=args.use_simple_max,
            )
            print(f"Inference done in {time.time() - t0:.3f}s")

            pred_normalized = output["normalized_actions"][0].astype(np.float32)

            # Diagnostic: gripper raw values
            grip_raw = pred_normalized[:, 9]
            print(f"\n  [DEBUG] Gripper raw (before denorm): "
                  f"min={grip_raw.min():.4f}  max={grip_raw.max():.4f}  "
                  f"mean={grip_raw.mean():.4f}  values={np.round(grip_raw, 3)}")

            # Denormalize
            pred_actions_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # Convert 10D -> 7D
            pred_actions_7d = actions_10d_to_7d(pred_actions_10d)

            # Print results
            print(f"\n{'=' * 82}")
            print("Denormalized relative actions  [rel_x, rel_y, rel_z, rel_rx, rel_ry, rel_rz, gripper]")
            print(f"{'=' * 82}")
            print(f"{'step':>4s}  " + "  ".join(f"{l:>9s}" for l in DIM_LABELS_7D))
            print("-" * 82)
            for t in range(pred_actions_7d.shape[0]):
                a = pred_actions_7d[t]
                g_str = "close" if a[6] > 0.5 else "open"
                print(f"{t:4d}  {a[0]:9.6f}  {a[1]:9.6f}  {a[2]:9.6f}  "
                      f"{a[3]:9.6f}  {a[4]:9.6f}  {a[5]:9.6f}  {g_str:>9s}")
            print(f"{'=' * 82}")

            # Summary
            print(f"\nSummary:")
            print(f"  pos  (rel_xyz) range: [{pred_actions_7d[:, :3].min():.6f}, {pred_actions_7d[:, :3].max():.6f}] m")
            print(f"  rot  (rel_aa)  range: [{pred_actions_7d[:, 3:6].min():.6f}, {pred_actions_7d[:, 3:6].max():.6f}] rad")
            print(f"  gripper values:       {np.unique(pred_actions_7d[:, 6])}")

            save_path = Path("ur5-dd") / f"step4_actions_viz_{timestamp}.png"
            visualize(pred_actions_7d, pil_img, str(save_path))

            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
