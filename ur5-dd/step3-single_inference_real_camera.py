#!/usr/bin/env python3
"""
Step 3: Single-shot inference with real camera (Discrete Diffusion).

Loads the QwenDiscreteDiffusion model, grabs one frame from the real camera,
and runs predict_action once. Robot does NOT move.

Key differences from QwenPI version:
  - MaskGIT-style iterative decode with temperature
  - --decode_temperature, --choice_temperature, --use_simple_max

Usage:
    python ur5-dd/step3-single_inference_real_camera.py
    python ur5-dd/step3-single_inference_real_camera.py --use_simple_max
    python ur5-dd/step3-single_inference_real_camera.py --decode_temperature 0.5
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

# ── Import camera from ur5/step2 (hardware is the same) ────────────
import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera


# ── Hardcoded UR5 EE pose (world frame) ─────────────────────────────
FIXED_EE_POSE = [0.327694, -0.194824, 0.11828, 2.15647483, -2.21923892, -0.11130583]
FIXED_GRIPPER = 0.0  # 0 = open, 1 = closed
INSTRUCTION = "pick up the building block"


def axisangle_to_rot6d(rx: float, ry: float, rz: float) -> np.ndarray:
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def ee_pose_to_state10d(pose_6d: list, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[INFO] flash_attn not available, falling back to sdpa")
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
    attn_impl = config.framework.qwenvl.attn_implementation
    print(f"Model loaded in {time.time() - t0:.1f}s (attn: {attn_impl})")
    print(f"  framework: {config.framework.name}")
    return model


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


DIM_LABELS = [
    "rel_x", "rel_y", "rel_z",
    "rel_r0", "rel_r1", "rel_r2",
    "rel_r3", "rel_r4", "rel_r5",
    "gripper",
]


def print_actions(pred_actions: np.ndarray):
    print(f"\nPredicted actions (normalized): shape {pred_actions.shape}")
    print(f"{'step':>4s}  " + "  ".join(f"{l:>8s}" for l in DIM_LABELS))
    print("-" * (6 + 10 * len(DIM_LABELS)))
    for t in range(pred_actions.shape[0]):
        vals = "  ".join(f"{v:8.4f}" for v in pred_actions[t])
        print(f"{t:4d}  {vals}")


def visualize(pred_actions: np.ndarray, camera_image: Image.Image, save_path: str):
    T = pred_actions.shape[0]
    ts = np.arange(T) / 20.0

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(4, 2, figure=fig, hspace=0.15, wspace=0.30,
                  width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    dims = [
        (0, "rel_x (body-frame)", "#e41a1c"),
        (1, "rel_y (body-frame)", "#377eb8"),
        (2, "rel_z (body-frame)", "#4daf4a"),
        (9, "gripper",           "#ff7f00"),
    ]

    axes = []
    for row, (dim_idx, label, color) in enumerate(dims):
        share = axes[0] if axes else None
        ax = fig.add_subplot(gs[row, 1], sharex=share)
        axes.append(ax)

        vals = pred_actions[:, dim_idx]
        ax.plot(ts, vals, "o-", markersize=4, linewidth=1.8, color=color)
        ax.fill_between(ts, 0, vals, alpha=0.15, color=color)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        for t in ts:
            ax.axvline(t, color="gray", linewidth=0.3, alpha=0.2)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

        if row < len(dims) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s) — 20Hz, 16 steps = 0.8s", fontsize=10)

    fig.suptitle('Single-shot Inference (Discrete Diffusion): "pick up the building block"',
                 fontsize=13, fontweight="bold", y=0.97)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nVisualization saved to {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Single-shot inference with real camera (Discrete Diffusion)")
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

    # 3. Grab one frame from real camera
    print(f"\nOpening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    pil_img = cam.grab_pil()
    cam.close()
    if pil_img is None:
        print("[ERROR] Failed to grab frame from camera")
        sys.exit(1)
    print(f"Captured image: {pil_img.size}")

    # 4. Build example and run inference
    example = build_example(pil_img, INSTRUCTION, state_10d=state_10d)
    batch = [example]

    print("\nRunning predict_action (discrete diffusion)...")
    t0 = time.time()
    output = model.predict_action(
        examples=batch,
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )
    dt = time.time() - t0
    print(f"Inference done in {dt:.3f}s")

    pred_actions = output["normalized_actions"][0]
    pred_actions = pred_actions.astype(np.float32)
    print_actions(pred_actions)

    # 5. Visualize
    save_path = Path("ur5-dd") / "step3_inference_viz.png"
    visualize(pred_actions, pil_img, str(save_path))

    print("\n[SAFETY] Robot is NOT moving. This is inference-only.")


if __name__ == "__main__":
    main()
