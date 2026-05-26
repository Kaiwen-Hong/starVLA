#!/usr/bin/env python3
"""
Air-hockey open-loop evaluation with live wrist camera + read-only left UR5e.

Interactive REPL:
  Press Enter → read TCP pose (world XY) → grab wrist frame → run inference
  → visualize predicted 32-step world-XY trajectory from current pose.

Robot does NOT move.

Model: QwenPI / DiT-S, action_dim=2 (dx, dy world), chunk_len=32, fps=50.
State is NOT fed to the model (training used include_state=false).

Usage:
    python ur5n/airhockey-d/fm/live_openloop_eval.py
    python ur5n/airhockey-d/fm/live_openloop_eval.py --checkpoint <path>
"""

import sys
import os
import time
import argparse
import threading
from pathlib import Path

import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import modular_policy

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_dynamic_qwenpi/"
    "checkpoints/steps_10000_pytorch_model.pt"
)
DEFAULT_INSTRUCTION = "strike the red puck back when it comes"

FPS = 50
CHUNK_LEN = 32

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

    def __init__(self, dev=0, width=1920, height=1080, fps=30):
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
        print(f"[Camera] /dev/video{dev} opened, {width}x{height}@{fps}fps")

        self._latest_raw = None
        self._frame_lock = threading.Lock()
        self._running = True
        self._reader_thread = threading.Thread(target=self._reader_loop,
                                                daemon=True)
        self._reader_thread.start()

    def _reader_loop(self):
        while self._running:
            ok, raw = self.cap.read()
            if ok:
                with self._frame_lock:
                    self._latest_raw = raw.copy()

    def grab_rgb(self):
        with self._frame_lock:
            raw = self._latest_raw
        if raw is None:
            return None
        yuv = np.ascontiguousarray(raw).reshape(self.H * 3 // 2, self.W)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def grab_pil(self):
        rgb = self.grab_rgb()
        if rgb is None:
            return None
        h, w = rgb.shape[:2]
        s = min(h, w)
        left, top = (w - s) // 2, (h - s) // 2
        return Image.fromarray(rgb[top:top + s, left:left + s])

    def close(self):
        self._running = False
        self._reader_thread.join(timeout=2.0)
        self.cap.release()


# ═══════════════════════════════════════════════════════════════════
#  Frame conversion
# ═══════════════════════════════════════════════════════════════════

def base_to_world(pose_base, T_bw):
    """[x,y,z,rx,ry,rz] base -> world. R_bw=I, translation only."""
    p = list(pose_base)
    p[0] += T_bw[0, 3]
    p[1] += T_bw[1, 3]
    p[2] += T_bw[2, 3]
    return p


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


def load_model(checkpoint_path):
    print(f"Loading model: {checkpoint_path}")
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

    if not getattr(config.datasets.vla_data, "image_size", None):
        config.datasets.vla_data.image_size = [256, 256]
        print("[FIX] Set image_size=[256,256] (match training native)")

    print(f"Model loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(state_xy, pred_traj, pred_delta, pil_img, save_path,
              instruction, run_idx, infer_ms, pose_world):
    T = pred_traj.shape[0]
    ts = np.arange(T) / FPS

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30,
                  width_ratios=[1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pil_img)
    ax_img.set_title("Wrist camera (center-crop)",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    ax_xy = fig.add_subplot(gs[1:, 0])
    ax_xy.plot(pred_traj[:, 0], pred_traj[:, 1], "s-", color="#e41a1c",
               ms=4, lw=1.8, label="Pred (32)")
    ax_xy.plot(state_xy[0], state_xy[1], "^", color="green", ms=12,
               label="now")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY (predicted)", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    ax_xt = fig.add_subplot(gs[0:2, 1])
    ax_xt.plot(ts, pred_traj[:, 0], "s-", color="#e41a1c", ms=3, lw=1.5,
               label="Pred")
    ax_xt.axhline(state_xy[0], color="gray", ls=":", lw=1,
                  label=f"now={state_xy[0]:.4f}")
    ax_xt.set_ylabel("x (world m)", fontweight="bold")
    ax_xt.set_title("Absolute x, y vs time", fontweight="bold")
    ax_xt.legend(fontsize=8)
    ax_xt.grid(True, alpha=0.3)
    plt.setp(ax_xt.get_xticklabels(), visible=False)

    ax_yt = fig.add_subplot(gs[2, 1], sharex=ax_xt)
    ax_yt.plot(ts, pred_traj[:, 1], "s-", color="#e41a1c", ms=3, lw=1.5)
    ax_yt.axhline(state_xy[1], color="gray", ls=":", lw=1)
    ax_yt.set_xlabel(f"t (s) — {FPS}Hz")
    ax_yt.set_ylabel("y (world m)", fontweight="bold")
    ax_yt.grid(True, alpha=0.3)

    ax_dx = fig.add_subplot(gs[0:2, 2])
    ax_dx.plot(ts, pred_delta[:, 0], "s-", color="#e41a1c", ms=3, lw=1.5,
               label="Pred")
    ax_dx.axhline(0, color="black", lw=0.5)
    ax_dx.set_ylabel("dx (m / step)", fontweight="bold")
    ax_dx.set_title("Per-step delta dx, dy", fontweight="bold")
    ax_dx.legend(fontsize=8)
    ax_dx.grid(True, alpha=0.3)
    plt.setp(ax_dx.get_xticklabels(), visible=False)

    ax_dy = fig.add_subplot(gs[2, 2], sharex=ax_dx)
    ax_dy.plot(ts, pred_delta[:, 1], "s-", color="#e41a1c", ms=3, lw=1.5)
    ax_dy.axhline(0, color="black", lw=0.5)
    ax_dy.set_xlabel(f"t (s) — {FPS}Hz")
    ax_dy.set_ylabel("dy (m / step)", fontweight="bold")
    ax_dy.grid(True, alpha=0.3)

    cw = pose_world
    fig.suptitle(
        f'Air-hockey live open-loop  —  run {run_idx}\n'
        f'"{instruction}"   |   infer={infer_ms:.0f}ms\n'
        f'World pose: [{cw[0]:.4f}, {cw[1]:.4f}, {cw[2]:.4f}, '
        f'{cw[3]:.4f}, {cw[4]:.4f}, {cw[5]:.4f}]',
        fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Air-hockey live open-loop eval (QwenPI, 2-D world XY)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix socket "
                             "instead of loading the model in-process. "
                             "(default: True — start the server first)")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false",
                        help="Load the model in-process (slower startup)")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_airhockey.sock")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    if args.use_server:
        from remote_model import RemoteModel
        model = RemoteModel(args.server_socket)
        args.checkpoint = model.checkpoint_path
    else:
        model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")
    assert model.chunk_len == CHUNK_LEN, (
        f"chunk_len mismatch: model={model.chunk_len} expected={CHUNK_LEN}")

    from rtde_receive import RTDEReceiveInterface
    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip} (read-only)...")
    rtde_r = RTDEReceiveInterface(robot_ip)
    print("Robot connected.")

    print(f"\nOpening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Waiting for first frame from background reader...")
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    session_ts = time.strftime("%Y%m%d_%H%M%S")
    viz_dir = Path("ur5n/airhockey-d/fm/viz_live") / f"session_{session_ts}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    print(f"Viz output: {viz_dir}")

    infer_kwargs = dict()  # QwenPI ignores extras

    print("=" * 60)
    print("  Air-hockey Live Open-Loop Eval (QwenPI)")
    print(f"  Arm:          {args.arm}  IP={robot_ip}")
    print(f"  Camera:       /dev/video{args.camera_dev}")
    print(f"  Instruction:  \"{args.instruction}\"")
    print(f"  T_bw offset:  [{T_bw[0,3]:.4f}, {T_bw[1,3]:.4f}, {T_bw[2,3]:.4f}]")
    print(f"  fps={FPS}  chunk_len={CHUNK_LEN}  action_dim=2 (world dx, dy)")
    print("  Press Enter to infer, 'q' to quit")
    print("=" * 60)

    run_idx = 0
    try:
        while True:
            line = input(f"\n[Run {run_idx}] Enter to infer (q to quit): ").strip()
            if line.lower() == 'q':
                break

            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            state_xy = np.array([pose_world[0], pose_world[1]],
                                dtype=np.float64)
            print(f"  Pose world: [{pose_world[0]:.4f}, {pose_world[1]:.4f}, "
                  f"{pose_world[2]:.4f}, {pose_world[3]:.4f}, "
                  f"{pose_world[4]:.4f}, {pose_world[5]:.4f}]")
            print(f"  state_xy (model anchor): "
                  f"[{state_xy[0]:.4f}, {state_xy[1]:.4f}]")

            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] no camera frame, retry")
                continue
            print(f"  Captured image: {pil_img.size}")

            example = {"image": [pil_img], "lang": args.instruction}
            t0 = time.time()
            out = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.time() - t0) * 1000

            pred_norm = out["normalized_actions"][0].astype(np.float32)
            pred_delta = baseframework.unnormalize_actions(
                pred_norm, action_stats)
            pred_traj = state_xy + np.cumsum(pred_delta, axis=0)

            print(f"  infer={infer_ms:.0f}ms")
            print(f"  delta range: "
                  f"dx=[{pred_delta[:, 0].min():.5f}, "
                  f"{pred_delta[:, 0].max():.5f}]  "
                  f"dy=[{pred_delta[:, 1].min():.5f}, "
                  f"{pred_delta[:, 1].max():.5f}]")
            print(f"  Pred final XY: "
                  f"[{pred_traj[-1, 0]:.4f}, {pred_traj[-1, 1]:.4f}]  "
                  f"(displacement={np.linalg.norm(pred_traj[-1] - state_xy):.4f} m)")

            save_path = (viz_dir /
                         f"run_{run_idx:03d}_{time.strftime('%H%M%S')}.png")
            visualize(state_xy, pred_traj, pred_delta, pil_img, str(save_path),
                      args.instruction, run_idx, infer_ms, pose_world)
            print(f"  Saved: {save_path}")
            run_idx += 1

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cam.close()
        if hasattr(model, 'close'):
            try:
                model.close()
            except Exception:
                pass
        print("Camera closed. Done.")


if __name__ == "__main__":
    main()
