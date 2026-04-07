#!/usr/bin/env python3
"""
Interactive open-loop evaluation with real camera + robot state (DD).

Loads the DD checkpoint once, connects to robot (read-only, no control),
then enters an interactive loop:
  Press Enter → read EE pose → grab camera frame → inference →
  accumulate world-frame deltas on current pose → print & visualize
  absolute trajectory in both world frame and base frame.

Robot does NOT move.

Action pipeline:
  model output → denormalize → 10D world-frame delta
  → rot6d→axis-angle → 7D delta [dx,dy,dz,drx,dry,drz,gripper]
  → cumsum on current EE pose → absolute trajectory (world frame)
  → subtract T_bw offset → absolute trajectory (base frame)

Usage:
    python ur5n/dd/interactive_openloop_eval.py
    python ur5n/dd/interactive_openloop_eval.py --checkpoint <path>
    python ur5n/dd/interactive_openloop_eval.py --instruction "pick up the block"
"""

import sys
import os
import time
import argparse
import threading
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from PIL import Image
import cv2

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import modular_policy

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DEFAULT_INSTRUCTION = "Pick up the purple block and place it on the red area of the board"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1

# ── Robot config ─────────────────────────────────────────────────────
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}


# ═══════════════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════════════

class RealCamera:
    """V4L2 camera with background reader thread for low-latency grabs."""

    def __init__(self, dev: int = 0, width: int = 1920, height: int = 1080, fps: int = 30):
        self.dev = dev
        self.W = width
        self.H = height

        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open /dev/video{dev}")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))
        self.cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        fcc_int = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        fcc = "".join([chr((fcc_int >> (8 * i)) & 0xFF) for i in range(4)])
        print(f"[Camera] /dev/video{dev} opened, FOURCC={fcc}, {width}x{height}@{fps}fps")

        self._latest_raw = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

    def _reader_loop(self):
        while self._running:
            ok, raw = self.cap.read()
            if ok:
                with self._frame_lock:
                    self._latest_raw = raw

    def grab_rgb(self) -> np.ndarray | None:
        with self._frame_lock:
            raw = self._latest_raw
        if raw is None:
            return None
        yuv = np.ascontiguousarray(raw).reshape(self.H * 3 // 2, self.W)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def grab_pil(self) -> Image.Image | None:
        rgb = self.grab_rgb()
        if rgb is None:
            return None
        cropped = _center_crop(rgb)
        return Image.fromarray(cropped)

    def close(self):
        self._running = False
        self._reader_thread.join(timeout=2.0)
        self.cap.release()


def _center_crop(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = min(h, w)
    left = (w - s) // 2
    top = (h - s) // 2
    return img[top:top + s, left:left + s]


# ═══════════════════════════════════════════════════════════════════
#  Frame conversion
# ═══════════════════════════════════════════════════════════════════

def base_to_world(pose_base, T_bw):
    """[x,y,z,rx,ry,rz] base -> world. R_bw=I, translation only."""
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    """[x,y,z,rx,ry,rz] world -> base. R_bw=I, translation only."""
    p = list(pose_world)
    p[0] -= T_bw[0, 3]; p[1] -= T_bw[1, 3]; p[2] -= T_bw[2, 3]
    return p


# ═══════════════════════════════════════════════════════════════════
#  Conversion utilities
# ═══════════════════════════════════════════════════════════════════

def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    """rot6d (6,) -> axis-angle (3,) via Gram-Schmidt."""
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=0)
    return Rotation.from_matrix(R).as_rotvec().astype(np.float32)


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    """(T, 10) -> (T, 7) [dx, dy, dz, drx, dry, drz, gripper]."""
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t, :3] = actions_10d[t, :3]
        out[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        out[t, 6] = actions_10d[t, 9]
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
    """Accumulate world-frame deltas on current pose.

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

def _detect_attn():
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
    config.framework.qwenvl.attn_implementation = _detect_attn()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()

    # Fix: ensure image_size is set so predict_action resizes camera frames
    if not getattr(config.datasets.vla_data, "image_size", None):
        config.datasets.vla_data.image_size = [224, 224]
        print("[FIX] Set image_size=[224,224] (was missing from checkpoint config)")

    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    print(f"  chunk_len: {model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Print results
# ═══════════════════════════════════════════════════════════════════

def print_results(current_world, current_base, deltas_7d, traj_world, traj_base,
                  instruction):
    """Print delta actions, then absolute trajectory in world & base frames side by side."""
    T = deltas_7d.shape[0]

    # Delta actions
    labels_d = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]
    print(f"\n{'=' * 82}")
    print(f"Delta actions (world frame)  — \"{instruction}\"")
    print(f"{'=' * 82}")
    print(f"{'step':>4s}  " + "  ".join(f"{l:>9s}" for l in labels_d))
    print("-" * 82)
    for t in range(T):
        d = deltas_7d[t]
        g = "close" if d[6] > 0.5 else "open"
        print(f"{t:4d}  {d[0]:9.6f}  {d[1]:9.6f}  {d[2]:9.6f}  "
              f"{d[3]:9.6f}  {d[4]:9.6f}  {d[5]:9.6f}  {g:>9s}")

    # Absolute trajectory: world frame | base frame
    labels_abs = ["x", "y", "z", "rx", "ry", "rz", "grip"]
    col_w = 9
    header_w = "  ".join(f"{l:>{col_w}s}" for l in labels_abs)
    sep = "-" * (6 + 2 * (len(header_w) + 3) + 5)

    print(f"\n{'=' * len(sep)}")
    print(f"Absolute trajectory")
    print(f"{'=' * len(sep)}")
    print(f"{'':>4s}  {'WORLD FRAME':^{len(header_w)}s}  |  {'BASE FRAME':^{len(header_w)}s}")
    print(f"{'step':>4s}  {header_w}  |  {header_w}")
    print(sep)

    # Current pose row
    cw = current_world
    cb = current_base
    def _fmt_pose(p, grip=None):
        g = "" if grip is None else ("close" if grip > 0.5 else " open")
        parts = [f"{p[i]:{col_w}.4f}" for i in range(6)]
        if grip is not None:
            parts.append(f"{g:>{col_w}s}")
        else:
            parts.append(f"{'—':>{col_w}s}")
        return "  ".join(parts)

    print(f"{'now':>4s}  {_fmt_pose(cw)}  |  {_fmt_pose(cb)}")

    for t in range(T):
        pw = traj_world[t]
        pb = traj_base[t]
        grip = pw[6]
        print(f"{t:4d}  {_fmt_pose(pw, grip)}  |  {_fmt_pose(pb, grip)}")
    print(f"{'=' * len(sep)}")

    # Summary
    print(f"\nSummary:")
    print(f"  delta pos range: [{deltas_7d[:, :3].min():.6f}, {deltas_7d[:, :3].max():.6f}] m")
    print(f"  delta rot range: [{deltas_7d[:, 3:6].min():.6f}, {deltas_7d[:, 3:6].max():.6f}] rad")
    print(f"  final world pos: [{traj_world[-1, 0]:.4f}, {traj_world[-1, 1]:.4f}, {traj_world[-1, 2]:.4f}]")
    print(f"  final base  pos: [{traj_base[-1, 0]:.4f}, {traj_base[-1, 1]:.4f}, {traj_base[-1, 2]:.4f}]")
    total_disp = np.linalg.norm(traj_world[-1, :3] - np.array(current_world[:3]))
    print(f"  total displacement: {total_disp:.4f} m")


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(traj_world, traj_base, camera_image, current_world, current_base,
              save_path, instruction, run_idx):
    """Camera image (left) + world-frame trajectory (mid) + base-frame trajectory (right)."""
    T = traj_world.shape[0]
    ts = np.arange(T) / 20.0

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.15, wspace=0.35,
                  width_ratios=[1, 1.2, 1.2])

    # Left: camera image
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    # Dim config: (dim_idx, label_world, label_base, color, world_start, base_start)
    dims = [
        (0, "x (world)", "x (base)", "#e41a1c", current_world[0], current_base[0]),
        (1, "y (world)", "y (base)", "#377eb8", current_world[1], current_base[1]),
        (2, "z (world)", "z (base)", "#4daf4a", current_world[2], current_base[2]),
        (6, "gripper",   "gripper",  "#ff7f00", None,             None),
    ]

    axes_w = []
    axes_b = []
    for row, (dim_idx, lbl_w, lbl_b, color, start_w, start_b) in enumerate(dims):
        # World frame column
        share_w = axes_w[0] if axes_w else None
        ax_w = fig.add_subplot(gs[row, 1], sharex=share_w)
        axes_w.append(ax_w)

        vals_w = traj_world[:, dim_idx]
        ax_w.plot(ts, vals_w, "o-", markersize=4, linewidth=1.8, color=color)
        if start_w is not None:
            ax_w.axhline(start_w, color=color, linewidth=1.0, linestyle="--",
                         alpha=0.5, label=f"now={start_w:.4f}")
            ax_w.legend(fontsize=7, loc="upper right")
        for t_val in ts:
            ax_w.axvline(t_val, color="gray", linewidth=0.3, alpha=0.2)
        ax_w.set_ylabel(lbl_w, fontsize=10, fontweight="bold")
        ax_w.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_w.set_title("WORLD FRAME", fontsize=11, fontweight="bold")
        if row < len(dims) - 1:
            plt.setp(ax_w.get_xticklabels(), visible=False)
        else:
            ax_w.set_xlabel("Time (s) — 20Hz", fontsize=10)

        # Base frame column
        share_b = axes_b[0] if axes_b else None
        ax_b = fig.add_subplot(gs[row, 2], sharex=share_b)
        axes_b.append(ax_b)

        vals_b = traj_base[:, dim_idx]
        ax_b.plot(ts, vals_b, "o-", markersize=4, linewidth=1.8, color=color)
        if start_b is not None:
            ax_b.axhline(start_b, color=color, linewidth=1.0, linestyle="--",
                         alpha=0.5, label=f"now={start_b:.4f}")
            ax_b.legend(fontsize=7, loc="upper right")
        for t_val in ts:
            ax_b.axvline(t_val, color="gray", linewidth=0.3, alpha=0.2)
        ax_b.set_ylabel(lbl_b, fontsize=10, fontweight="bold")
        ax_b.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_b.set_title("ROBOT (BASE) FRAME", fontsize=11, fontweight="bold")
        if row < len(dims) - 1:
            plt.setp(ax_b.get_xticklabels(), visible=False)
        else:
            ax_b.set_xlabel("Time (s) — 20Hz", fontsize=10)

    # Title with robot state
    cw = current_world
    cb = current_base
    fig.suptitle(
        f'Interactive Open-Loop (DD) — Run {run_idx}\n'
        f'"{instruction}"\n'
        f'World: [{cw[0]:.4f}, {cw[1]:.4f}, {cw[2]:.4f}, {cw[3]:.4f}, {cw[4]:.4f}, {cw[5]:.4f}]\n'
        f'Base:  [{cb[0]:.4f}, {cb[1]:.4f}, {cb[2]:.4f}, {cb[3]:.4f}, {cb[4]:.4f}, {cb[5]:.4f}]',
        fontsize=11, fontweight="bold", y=1.0, va="bottom")

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Visualization saved to {save_path}")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Interactive open-loop eval with real camera + robot state (DD)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    # 1. Load model (one-time)
    model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")

    # 2. Connect to robot (read-only)
    from rtde_receive import RTDEReceiveInterface

    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip} (read-only)...")
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Robot connected.")

    # 3. Open camera (one-time, kept open)
    print(f"\nOpening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Waiting for first frame from background reader...")
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.\n")

    # 4. Create output directory (per-session subfolder)
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    viz_dir = Path("ur5n/dd/viz") / f"session_{session_ts}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Viz output: {viz_dir}")

    # 5. Inference kwargs
    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    # 6. Interactive loop
    run_idx = 0
    print("=" * 60)
    print("  Interactive Open-Loop Evaluation (Discrete Diffusion)")
    print(f"  Arm:         {args.arm}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  T_bw offset: [{T_bw[0,3]:.4f}, {T_bw[1,3]:.4f}, {T_bw[2,3]:.4f}]")
    print("  Press Enter to capture + infer, 'q' to quit")
    print("=" * 60)

    try:
        while True:
            user_input = input(f"\n[Run {run_idx}] Press Enter to infer (q to quit): ").strip()
            if user_input.lower() == 'q':
                break

            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # Read current EE pose
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            print(f"Current EE (world): [{pose_world[0]:.4f}, {pose_world[1]:.4f}, "
                  f"{pose_world[2]:.4f}, {pose_world[3]:.4f}, {pose_world[4]:.4f}, "
                  f"{pose_world[5]:.4f}]")
            print(f"Current EE (base):  [{pose_base[0]:.4f}, {pose_base[1]:.4f}, "
                  f"{pose_base[2]:.4f}, {pose_base[3]:.4f}, {pose_base[4]:.4f}, "
                  f"{pose_base[5]:.4f}]")

            # Grab fresh frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, try again")
                continue
            print(f"Captured image: {pil_img.size}")

            # Build example (no state)
            example = {"image": [pil_img], "lang": args.instruction}

            # Inference
            print("Running predict_action...")
            t0 = time.time()
            output = model.predict_action(examples=[example], **infer_kwargs)
            dt = time.time() - t0
            print(f"Inference done in {dt:.3f}s")

            pred_normalized = output["normalized_actions"][0].astype(np.float32)

            # Denormalize -> world-frame 10D delta
            pred_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # Convert 10D -> 7D delta (rot6d -> axis-angle)
            deltas_7d = actions_10d_to_7d(pred_10d)

            # Accumulate deltas on current pose -> absolute world trajectory
            traj_world = accumulate_deltas(pose_world, deltas_7d)

            # Convert world trajectory to base frame
            traj_base = traj_world.copy()
            traj_base[:, 0] -= T_bw[0, 3]
            traj_base[:, 1] -= T_bw[1, 3]
            traj_base[:, 2] -= T_bw[2, 3]
            # rotation unchanged (R_bw = I)

            # Print
            print_results(pose_world, list(pose_base), deltas_7d,
                          traj_world, traj_base, args.instruction)

            # Visualize
            save_path = viz_dir / f"run_{run_idx:03d}_{timestamp}.png"
            visualize(traj_world, traj_base, pil_img,
                      pose_world, list(pose_base),
                      str(save_path), args.instruction, run_idx)

            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
