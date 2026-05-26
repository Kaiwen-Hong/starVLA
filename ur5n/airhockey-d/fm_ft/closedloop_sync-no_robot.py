#!/usr/bin/env python3
"""
Air-hockey synchronous closed-loop control — NO-ROBOT variant
(QwenPI, 2-D world XY).

Identical to closedloop_sync.py except the robot does NOT move:
  - No RTDEControlInterface, no servoL, no servoStop.
  - RTDEReceiveInterface is still used to read the TCP pose every loop
    (the model's current XY anchor stays whatever the robot is parked at).
  - The "execution" block is replaced with precise_wait of the same
    duration so the loop cadence (inference rate, log spacing, etc.)
    is unchanged.

Use this for:
  - Validating the inference + safety + viz pipeline without movement.
  - Timing tests (how long does a sync cycle take end-to-end).
  - Recording predicted-vs-stationary-pose rollouts.

The robot will NOT move (regardless of XY safety bounds or predictions).

Usage:
    # Terminal 1: start the inference server (model loaded once)
    python ur5n/airhockey-d/fm/inference_server.py

    # Terminal 2: run the no-robot sync loop
    python ur5n/airhockey-d/fm/closedloop_sync-no_robot.py
    python ur5n/airhockey-d/fm/closedloop_sync-no_robot.py --n_actions 8
"""

import sys
import os
import time
import json
import argparse
import threading
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import cv2

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import modular_policy

# ── Defaults ────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "/home/kaiwen/Desktop/research/starVLA/checkpoints/discreteRTC/"
    "fastumi_airhockey_bounce_qwenPI_0522_DiT-S_rtc_ft/"
    "checkpoints/pytorch_model.pt"
)
DEFAULT_INSTRUCTION = "strike the red puck back when it comes"

POLICY_HZ = 50
INTERP_MULT = 2
SERVO_HZ = POLICY_HZ * INTERP_MULT          # 100 Hz

# Data-derived safety bounds (training data + 0.05 m buffer, see args).
# state.x range:  0.2996 .. 0.4491
# state.y range: -0.3300 .. -0.1201
XY_DATA_MIN = np.array([0.2996, -0.3300], dtype=np.float64)
XY_DATA_MAX = np.array([0.4491, -0.1201], dtype=np.float64)

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
#  Frame conversion + timing
# ═══════════════════════════════════════════════════════════════════

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]
    p[1] += T_bw[1, 3]
    p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]
    p[1] -= T_bw[1, 3]
    p[2] -= T_bw[2, 3]
    return p


def interpolate_waypoints(start_pose, waypoints, mult):
    """Linear interp from start_pose to each successive waypoint.
    start_pose: (6,)  waypoints: (n, 6)  → returns (n*mult, 6)."""
    all_poses = []
    prev = start_pose
    for wp in waypoints:
        for j in range(1, mult + 1):
            alpha = j / mult
            all_poses.append(prev + alpha * (wp - prev))
        prev = wp
    return np.array(all_poses, dtype=np.float64)


def precise_wait(t_end, slack=0.001):
    remaining = t_end - time.monotonic()
    if remaining > 0:
        if remaining > slack:
            time.sleep(remaining - slack)
        while time.monotonic() < t_end:
            pass


# ═══════════════════════════════════════════════════════════════════
#  Model loading (only used with --no_use_server)
# ═══════════════════════════════════════════════════════════════════

def _detect_attn():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path):
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

    if not getattr(config.datasets.vla_data, "image_size", None):
        config.datasets.vla_data.image_size = [256, 256]
        print("[FIX] Set image_size=[256,256] (match training native)")

    print(f"Model loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Visualization (per-step rollout image)
# ═══════════════════════════════════════════════════════════════════

def visualize_step(camera_image, current_xy, pred_delta, waypoints_xy,
                   xy_bounds, step, save_path, instruction,
                   infer_ms, exec_ms, total_ms):
    """Camera (left) + top-down XY trajectory (right).

    Solid red = waypoints that WOULD have been executed (n_exec).
    Dashed pink = remainder of predicted chunk (not in exec window).
    """
    n_exec = waypoints_xy.shape[0]
    L = pred_delta.shape[0]

    # Full predicted absolute XY trajectory (including the unused tail).
    full_traj = current_xy + np.cumsum(pred_delta, axis=0)  # (L, 2)

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30,
                  width_ratios=[1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Wrist camera (center-crop)",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

    # Top-down XY trajectory
    ax_xy = fig.add_subplot(gs[:, 1])
    (x_lo, y_lo), (x_hi, y_hi) = xy_bounds
    ax_xy.add_patch(plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                  fill=False, ls="--", ec="gray",
                                  label="safety box"))
    ax_xy.plot(full_traj[n_exec:, 0], full_traj[n_exec:, 1], "o--",
               color="pink", ms=4, lw=1.2,
               label=f"future (chunk[{n_exec}:{L}])")
    ax_xy.plot(waypoints_xy[:, 0], waypoints_xy[:, 1], "s-",
               color="#e41a1c", ms=5, lw=2.0,
               label=f"exec ({n_exec}) [NOT SENT]")
    ax_xy.plot(current_xy[0], current_xy[1], "^",
               color="green", ms=12, label="now")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8, loc="best")

    # Per-step delta (dx, dy) over the full chunk
    ts = np.arange(L) / POLICY_HZ
    ax_d = fig.add_subplot(gs[0, 2])
    ax_d.plot(ts, pred_delta[:, 0], "o-", ms=3, color="#e41a1c", label="dx")
    ax_d.plot(ts, pred_delta[:, 1], "o-", ms=3, color="#377eb8", label="dy")
    ax_d.axvline(ts[n_exec - 1] if n_exec > 0 else 0,
                 color="black", ls="--", lw=0.8, alpha=0.5)
    ax_d.axhline(0, color="black", lw=0.5)
    ax_d.set_xlabel("t (s) — 50 Hz")
    ax_d.set_ylabel("delta (m / step)")
    ax_d.set_title("Per-step delta (full chunk)", fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    # Absolute XY of full chunk over time
    ax_t = fig.add_subplot(gs[1, 2])
    ax_t.plot(ts, full_traj[:, 0], "o-", ms=3, color="#e41a1c", label="x")
    ax_t.plot(ts, full_traj[:, 1], "o-", ms=3, color="#377eb8", label="y")
    ax_t.axvline(ts[n_exec - 1] if n_exec > 0 else 0,
                 color="black", ls="--", lw=0.8, alpha=0.5)
    ax_t.axhline(current_xy[0], color="#e41a1c", ls=":", lw=0.8, alpha=0.5)
    ax_t.axhline(current_xy[1], color="#377eb8", ls=":", lw=0.8, alpha=0.5)
    ax_t.set_xlabel("t (s) — 50 Hz")
    ax_t.set_ylabel("absolute (m)")
    ax_t.set_title("Absolute x, y (full chunk)", fontweight="bold")
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)

    fig.suptitle(
        f'Sync closed-loop [NO-ROBOT] — step {step}\n'
        f'"{instruction}"  |  infer={infer_ms:.0f}ms  exec={exec_ms:.0f}ms  '
        f'total={total_ms:.0f}ms',
        fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Air-hockey sync closed-loop [NO-ROBOT] "
                    "(QwenPI, 2-D world XY)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=16,
                        help="Steps per inference cycle (1..chunk_len). "
                             "Larger = fewer inference calls but staler context. "
                             "Default: 16 (=0.32s exec @ 50Hz)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference cycles (0 = unlimited, Ctrl+C to stop)")
    parser.add_argument("--xy_buffer", type=float, default=0.05,
                        help="Safety buffer (m) added outside data q01/q99 range")
    parser.add_argument("--x_min", type=float, default=None,
                        help=f"Override x_min (default: data q01 - buffer)")
    parser.add_argument("--x_max", type=float, default=None,
                        help=f"Override x_max (default: data q99 + buffer)")
    parser.add_argument("--y_min", type=float, default=None,
                        help=f"Override y_min (default: data q01 - buffer)")
    parser.add_argument("--y_max", type=float, default=None,
                        help=f"Override y_max (default: data q99 + buffer)")
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix socket "
                             "instead of loading the model in-process.")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_airhockey.sock")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    # ── Safety XY bounds ────────────────────────────────────────────
    x_lo = args.x_min if args.x_min is not None else XY_DATA_MIN[0] - args.xy_buffer
    x_hi = args.x_max if args.x_max is not None else XY_DATA_MAX[0] + args.xy_buffer
    y_lo = args.y_min if args.y_min is not None else XY_DATA_MIN[1] - args.xy_buffer
    y_hi = args.y_max if args.y_max is not None else XY_DATA_MAX[1] + args.xy_buffer
    xy_lo = np.array([x_lo, y_lo], dtype=np.float64)
    xy_hi = np.array([x_hi, y_hi], dtype=np.float64)

    # ── Load model ──────────────────────────────────────────────────
    if args.use_server:
        from remote_model import RemoteModel
        model = RemoteModel(args.server_socket)
        args.checkpoint = model.checkpoint_path
    else:
        model = load_model(args.checkpoint)
    chunk_len = model.chunk_len
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")
    assert chunk_len == 32, f"unexpected chunk_len={chunk_len}"
    assert 1 <= args.n_actions <= chunk_len, (
        f"n_actions={args.n_actions} must be in [1, {chunk_len}]")

    # ── Connect robot (READ-ONLY: no control interface, no movement) ─
    from rtde_receive import RTDEReceiveInterface
    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip} (READ-ONLY)...")
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Open camera ─────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # ── Lock Z + rotation from current pose ─────────────────────────
    pose_base_init = rtde_r.getActualTCPPose()
    pose_world_init = base_to_world(pose_base_init, T_bw)
    locked_z = float(pose_world_init[2])
    locked_rot = np.array(pose_world_init[3:6], dtype=np.float64)
    print(f"\nLocked Z (world):    {locked_z:.4f} m")
    print(f"Locked rotation:     "
          f"[{locked_rot[0]:.4f}, {locked_rot[1]:.4f}, {locked_rot[2]:.4f}]")
    print(f"Starting XY (world): [{pose_world_init[0]:.4f}, "
          f"{pose_world_init[1]:.4f}]")

    # ── Rollout setup ───────────────────────────────────────────────
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = (Path("ur5n/airhockey-d/fm/rollouts_sync_no_robot")
                       / f"sync_{ts_str}")
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")
        with open(rollout_dir / "config.json", "w") as f:
            json.dump({
                "mode": "sync_airhockey_no_robot",
                "checkpoint": args.checkpoint,
                "instruction": args.instruction,
                "arm": args.arm,
                "n_actions": args.n_actions,
                "chunk_len": chunk_len,
                "policy_hz": POLICY_HZ,
                "servo_hz": SERVO_HZ,
                "xy_bounds": [[x_lo, y_lo], [x_hi, y_hi]],
                "locked_z": locked_z,
                "locked_rot": locked_rot.tolist(),
                "dataset_key": dataset_key,
            }, f, indent=2)

    # ── Print config + arm ─────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Air-hockey Sync Closed-Loop [NO-ROBOT] (QwenPI)")
    print(f"  Arm:           {args.arm}  IP={robot_ip}  (READ-ONLY)")
    print(f"  Camera:        /dev/video{args.camera_dev}")
    print(f"  Instruction:   \"{args.instruction}\"")
    print(f"  n_actions:     {args.n_actions} / chunk_len={chunk_len}")
    print(f"  Policy / Servo: {POLICY_HZ} Hz x{INTERP_MULT} = {SERVO_HZ} Hz")
    print(f"  XY safety box: x in [{x_lo:.4f}, {x_hi:.4f}]  "
          f"y in [{y_lo:.4f}, {y_hi:.4f}]")
    print(f"  Use server:    {args.use_server}"
          f"{' (' + args.server_socket + ')' if args.use_server else ''}")
    print(f"  Max steps:     {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Save rollout:  {args.save_rollout}")
    print(f"  *** ROBOT WILL NOT MOVE ***")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    servo_dt = 1.0 / SERVO_HZ
    infer_kwargs = dict()

    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current XY (Z/rot locked) — pose stays put since
            #    we never send servoL.
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_xy = np.array(pose_world[:2], dtype=np.float64)
            current_6d = np.array(
                [current_xy[0], current_xy[1], locked_z,
                 locked_rot[0], locked_rot[1], locked_rot[2]],
                dtype=np.float64,
            )

            # 2. Grab frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, retrying...")
                continue

            # 3. Inference
            example = {"image": [pil_img], "lang": args.instruction}
            t_infer = time.monotonic()
            out = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.monotonic() - t_infer) * 1000

            pred_norm = out["normalized_actions"][0].astype(np.float32)  # (32, 2)
            pred_delta = baseframework.unnormalize_actions(
                pred_norm, action_stats).astype(np.float64)              # (32, 2)

            # 4. Take first n_actions, accumulate, clamp
            n_exec = min(args.n_actions, len(pred_delta))
            xy = current_xy.copy()
            waypoints_xy = np.zeros((n_exec, 2), dtype=np.float64)
            for i in range(n_exec):
                xy = xy + pred_delta[i]
                xy = np.clip(xy, xy_lo, xy_hi)
                waypoints_xy[i] = xy

            # 5. Pad to 6-D (locked Z + rot)
            waypoints_6d = np.zeros((n_exec, 6), dtype=np.float64)
            waypoints_6d[:, 0:2] = waypoints_xy
            waypoints_6d[:, 2] = locked_z
            waypoints_6d[:, 3:6] = locked_rot

            # 6. Interpolate from current pose to waypoints at SERVO_HZ
            #    (kept so timing math is identical to the real loop)
            interp = interpolate_waypoints(current_6d, waypoints_6d,
                                            INTERP_MULT)

            # 7. "Execute" — NO-ROBOT: just precise_wait for the equivalent
            #    duration so cadence (and downstream inference rate)
            #    matches a real run.
            t_exec_start = time.monotonic()
            precise_wait(t_exec_start + len(interp) * servo_dt)

            exec_ms = (time.monotonic() - t_exec_start) * 1000
            total_ms = (time.monotonic() - loop_t0) * 1000

            print(f"[step {step:4d}]  infer={infer_ms:5.0f}ms  "
                  f"exec={exec_ms:5.0f}ms  total={total_ms:5.0f}ms  "
                  f"now=[{current_xy[0]:.3f}, {current_xy[1]:.3f}]  "
                  f"target=[{waypoints_xy[-1, 0]:.3f}, "
                  f"{waypoints_xy[-1, 1]:.3f}]  [NOT SENT]")

            # 8. Save rollout
            if args.save_rollout and rollout_dir is not None:
                img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                pil_img.save(str(img_path), quality=90)
                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_step(
                    pil_img, current_xy, pred_delta, waypoints_xy,
                    ((x_lo, y_lo), (x_hi, y_hi)),
                    step, str(viz_path), args.instruction,
                    infer_ms, exec_ms, total_ms)
                rollout_log.append({
                    "step": step,
                    "wall_time": time.time(),
                    "ee_xy_world": current_xy.tolist(),
                    "ee_pose_base": list(pose_base),
                    "pred_delta_full": pred_delta.tolist(),
                    "waypoints_xy_clamped": waypoints_xy.tolist(),
                    "n_exec": n_exec,
                    "infer_ms": round(infer_ms, 1),
                    "exec_ms": round(exec_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "image_file": f"images/step_{step:04d}.jpg",
                    "viz_file": f"images/step_{step:04d}_viz.png",
                })

            step += 1

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        print("Cleaning up...")
        # No servoStop / stopScript — we never connected a control interface.
        cam.close()
        if hasattr(model, 'close'):
            try:
                model.close()
            except Exception:
                pass

        if args.save_rollout and rollout_dir is not None and rollout_log:
            with open(rollout_dir / "rollout.json", "w") as f:
                json.dump(rollout_log, f, indent=2)
            print(f"Rollout saved: {rollout_dir}  ({len(rollout_log)} steps)")

        print(f"Done. Executed {step} inference cycles.")


if __name__ == "__main__":
    main()
