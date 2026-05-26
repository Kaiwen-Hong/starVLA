#!/usr/bin/env python3
"""
Air-hockey sync closed-loop control with training-distribution initial pose
(QwenPI, 2-D world XY) — left arm of the air-hockey scene.

Companion: ../calib_output-air/3airhockey_dynamic_bounce_right_only.py
            (run that in another terminal to drive the right arm's strike)

Differences from closedloop_sync.py:
  - Samples LEFT initial pose from the training distribution
    (X ∈ U(0.30, 0.45), Y ∈ U((-0.33,-0.31) ∪ (-0.14,-0.12)),
     Z = 0.176, rotvec = [-2.2192, 2.2148, 0.0091]), IK-checked, max 30
    tries — matches the per-episode random-home used during data
    collection (sample_left_home_xy_with_ik in
    3airhockey_dynamic_bounce.py).
  - moveJ to the sampled pose before the main loop.
  - Locks Z + rotation from the sampled pose (in practice these are the
    hardcoded constants above).
  - Policy loop then runs continuously — when the right-arm script
    strikes the puck and the puck enters the left wrist camera's view,
    the policy reacts and the left arm should move to intercept.

Operator flow:
  Terminal 1 (this script):
      python ur5n/airhockey-d/fm/inference_server.py   # once, leave open
      python ur5n/airhockey-d/fm/closedloop_sync_scene.py
        → samples init pose, jogs there, starts policy loop
  Terminal 2 (right arm scripted strike, see companion file):
      python ../calib_output-air/3airhockey_dynamic_bounce_right_only.py

The robot WILL move. Press Ctrl+C to stop.
"""

import sys
import os
import time
import json
import argparse
import random
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
    "checkpoints/discreteRTC/fastumi_dynamic_qwenpi/"
    "checkpoints/steps_10000_pytorch_model.pt"
)
DEFAULT_INSTRUCTION = "strike the red puck back when it comes"

POLICY_HZ = 50
INTERP_MULT = 2
SERVO_HZ = POLICY_HZ * INTERP_MULT          # 100 Hz

# Data-derived safety bounds (training data + 0.05 m buffer, see args).
XY_DATA_MIN = np.array([0.2996, -0.3300], dtype=np.float64)
XY_DATA_MAX = np.array([0.4491, -0.1201], dtype=np.float64)

# Training-distribution LEFT home — verbatim from
# 3airhockey_dynamic_bounce.py.
LEFT_HOME_DYNAMIC_Z = 0.176
LEFT_HOME_X_RANGE = (0.30, 0.45)
LEFT_HOME_Y_RANGES = [(-0.33, -0.31), (-0.14, -0.12)]
LEFT_HOME_SAMPLE_MAX_TRIES = 30
LEFT_HOME_DYNAMIC_ROTVEC = [-2.2192, 2.2148, 0.0091]

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
#  Init pose sampling (matches training distribution)
# ═══════════════════════════════════════════════════════════════════

_Y_LENGTHS = [float(b - a) for (a, b) in LEFT_HOME_Y_RANGES]
_Y_TOTAL = sum(_Y_LENGTHS)


def sample_left_init_xy_with_ik(rtde_c, rtde_r, T_bw, seed=None):
    """Sample LEFT initial XY uniformly: X ∈ LEFT_HOME_X_RANGE,
    Y ∈ length-weighted-union(LEFT_HOME_Y_RANGES). IK-checked at
    Z=LEFT_HOME_DYNAMIC_Z, rotvec=LEFT_HOME_DYNAMIC_ROTVEC. Raises
    RuntimeError if no IK-valid sample in LEFT_HOME_SAMPLE_MAX_TRIES.

    Returns (x, y, n_tries).
    """
    if seed is not None:
        random.seed(seed)
    qnear = rtde_r.getActualQ()
    z_base = LEFT_HOME_DYNAMIC_Z - T_bw[2, 3]
    for tries in range(1, LEFT_HOME_SAMPLE_MAX_TRIES + 1):
        x = random.uniform(*LEFT_HOME_X_RANGE)
        u = random.uniform(0.0, _Y_TOTAL)
        acc = 0.0
        y = None
        for (a, b), L in zip(LEFT_HOME_Y_RANGES, _Y_LENGTHS):
            acc += L
            if u <= acc:
                y = random.uniform(a, b)
                break
        if y is None:
            y = random.uniform(*LEFT_HOME_Y_RANGES[-1])
        pose_base = [
            float(x - T_bw[0, 3]),
            float(y - T_bw[1, 3]),
            float(z_base),
            *LEFT_HOME_DYNAMIC_ROTVEC,
        ]
        try:
            rtde_c.getInverseKinematics(pose_base, qnear)
            return float(x), float(y), tries
        except Exception:
            continue
    raise RuntimeError(
        f"No IK-valid LEFT init sample after "
        f"{LEFT_HOME_SAMPLE_MAX_TRIES} tries")


def move_to_init(rtde_c, target_world_xyz, T_bw):
    target_pose_base = [
        target_world_xyz[0] - T_bw[0, 3],
        target_world_xyz[1] - T_bw[1, 3],
        target_world_xyz[2] - T_bw[2, 3],
        *LEFT_HOME_DYNAMIC_ROTVEC,
    ]
    joints = rtde_c.getInverseKinematics(target_pose_base)
    rtde_c.moveJ(joints, 1.0, 1.0)


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
                   xy_bounds, init_xy, step, save_path, instruction,
                   infer_ms, exec_ms, total_ms):
    n_exec = waypoints_xy.shape[0]
    L = pred_delta.shape[0]
    full_traj = current_xy + np.cumsum(pred_delta, axis=0)

    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(2, 3, figure=fig, hspace=0.32, wspace=0.30,
                  width_ratios=[1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Wrist camera (center-crop)",
                     fontsize=11, fontweight="bold")
    ax_img.axis("off")

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
               label=f"exec ({n_exec})")
    ax_xy.plot(current_xy[0], current_xy[1], "^",
               color="green", ms=12, label="now")
    ax_xy.plot(init_xy[0], init_xy[1], "*",
               color="purple", ms=12, label="init (sampled)")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8, loc="best")

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
        f'Sync scene — step {step}\n'
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
        description="Air-hockey sync scene closed-loop (left arm, QwenPI, "
                    "2-D world XY) — samples init pose from training distribution")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=32,
                        help="Steps to execute per inference cycle (1..chunk_len). "
                             "Default 32 = full chunk = 0.64s exec @ 50Hz.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference cycles (0 = unlimited, Ctrl+C to stop)")
    parser.add_argument("--init_pose", choices=["sample", "current"],
                        default="sample",
                        help="'sample' = IK-checked random from training "
                             "distribution + moveJ; 'current' = use current "
                             "pose (no moveJ, for manual jog testing)")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="RNG seed for init-pose sampling (default: time-based)")
    parser.add_argument("--xy_buffer", type=float, default=0.05,
                        help="Safety buffer (m) added outside data q01/q99 range")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--use_server", action="store_true", default=True)
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

    # ── Connect robot ───────────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Open camera ─────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # ── Initial pose: sample (matches training dist) or use current ─
    if args.init_pose == "sample":
        print("\nSampling LEFT init pose (training distribution, "
              "IK-checked) ...")
        try:
            init_x, init_y, ntries = sample_left_init_xy_with_ik(
                rtde_c, rtde_r, T_bw, seed=args.init_seed)
        except RuntimeError as e:
            print(f"[ERR] {e}")
            sys.exit(1)
        init_world_xyz = [float(init_x), float(init_y),
                          float(LEFT_HOME_DYNAMIC_Z)]
        print(f"  sampled init world XYZ = "
              f"[{init_x:+.4f}, {init_y:+.4f}, {LEFT_HOME_DYNAMIC_Z:.4f}]  "
              f"(IK ok after {ntries} tries)")
        print(f"  moving LEFT to init pose via moveJ ...")
        try:
            move_to_init(rtde_c, init_world_xyz, T_bw)
            print("  moveJ done.")
        except Exception as e:
            print(f"[ERR] moveJ failed: {e}")
            sys.exit(1)

    # Read achieved pose to lock Z + rot. After a sampled moveJ this
    # should be exactly (LEFT_HOME_DYNAMIC_Z, LEFT_HOME_DYNAMIC_ROTVEC);
    # for init_pose=current it's whatever the operator jogged to.
    pose_base_init = rtde_r.getActualTCPPose()
    pose_world_init = base_to_world(pose_base_init, T_bw)
    locked_z = float(pose_world_init[2])
    locked_rot = np.array(pose_world_init[3:6], dtype=np.float64)
    init_xy = np.array(pose_world_init[:2], dtype=np.float64)
    print(f"\nLocked Z (world):    {locked_z:.4f} m")
    print(f"Locked rotation:     "
          f"[{locked_rot[0]:.4f}, {locked_rot[1]:.4f}, {locked_rot[2]:.4f}]")
    print(f"Starting XY (world): [{init_xy[0]:.4f}, {init_xy[1]:.4f}]")

    # ── Rollout setup ───────────────────────────────────────────────
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        ts_str = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = Path("ur5n/airhockey-d/fm/rollouts_sync_scene") / f"scene_{ts_str}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")
        with open(rollout_dir / "config.json", "w") as f:
            json.dump({
                "mode": "sync_scene_airhockey",
                "checkpoint": args.checkpoint,
                "instruction": args.instruction,
                "arm": args.arm,
                "n_actions": args.n_actions,
                "chunk_len": chunk_len,
                "policy_hz": POLICY_HZ,
                "servo_hz": SERVO_HZ,
                "xy_bounds": [[x_lo, y_lo], [x_hi, y_hi]],
                "init_pose_mode": args.init_pose,
                "init_seed": args.init_seed,
                "init_xy_world": init_xy.tolist(),
                "locked_z": locked_z,
                "locked_rot": locked_rot.tolist(),
                "dataset_key": dataset_key,
            }, f, indent=2)

    # ── Print config ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Air-hockey Sync SCENE Closed-Loop (LEFT arm, QwenPI)")
    print(f"  Arm:           {args.arm}  IP={robot_ip}")
    print(f"  Camera:        /dev/video{args.camera_dev}")
    print(f"  Instruction:   \"{args.instruction}\"")
    print(f"  Init mode:     {args.init_pose}"
          f"{f' (seed={args.init_seed})' if args.init_seed else ''}")
    print(f"  n_actions:     {args.n_actions} / chunk_len={chunk_len}")
    print(f"  Policy / Servo: {POLICY_HZ} Hz x{INTERP_MULT} = {SERVO_HZ} Hz")
    print(f"  XY safety box: x in [{x_lo:.4f}, {x_hi:.4f}]  "
          f"y in [{y_lo:.4f}, {y_hi:.4f}]")
    print(f"  Use server:    {args.use_server}"
          f"{' (' + args.server_socket + ')' if args.use_server else ''}")
    print(f"  Max steps:     {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Save rollout:  {args.save_rollout}")
    print(f"{'=' * 60}")
    print(f"\n  In a SECOND terminal:")
    print(f"    python ../calib_output-air/3airhockey_dynamic_bounce_right_only.py")
    print(f"  to drive the right arm scripted strike.")

    input("\n>>> Press Enter to START the policy loop (Ctrl+C to abort) <<<")

    servo_dt = 1.0 / SERVO_HZ
    infer_kwargs = dict()

    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current XY (Z/rot locked)
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

            pred_norm = out["normalized_actions"][0].astype(np.float32)
            pred_delta = baseframework.unnormalize_actions(
                pred_norm, action_stats).astype(np.float64)

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
            interp = interpolate_waypoints(current_6d, waypoints_6d,
                                            INTERP_MULT)

            # 7. Execute (blocking)
            t_exec_start = time.monotonic()
            for i, pose_w in enumerate(interp):
                target_base = world_to_base(pose_w.tolist(), T_bw)
                rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
                precise_wait(t_exec_start + (i + 1) * servo_dt)
            rtde_c.servoStop()

            exec_ms = (time.monotonic() - t_exec_start) * 1000
            total_ms = (time.monotonic() - loop_t0) * 1000

            print(f"[step {step:4d}]  infer={infer_ms:5.0f}ms  "
                  f"exec={exec_ms:5.0f}ms  total={total_ms:5.0f}ms  "
                  f"now=[{current_xy[0]:.3f}, {current_xy[1]:.3f}]  "
                  f"target=[{waypoints_xy[-1, 0]:.3f}, "
                  f"{waypoints_xy[-1, 1]:.3f}]")

            # 8. Save rollout
            if args.save_rollout and rollout_dir is not None:
                img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                pil_img.save(str(img_path), quality=90)
                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_step(
                    pil_img, current_xy, pred_delta, waypoints_xy,
                    ((x_lo, y_lo), (x_hi, y_hi)),
                    init_xy, step, str(viz_path), args.instruction,
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
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
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
