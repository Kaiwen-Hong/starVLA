#!/usr/bin/env python3
"""
Pool synchronous closed-loop control (QwenPI flow-matching, 10-D world).

The robot WILL move. Cue is bolt-clamped; gripper output is ignored.

Action constraints applied at every step:
  - X, Y, Z, yaw  — policy controls (clamped to training-data range)
  - roll, pitch    — LOCKED to startup values (cue should never tilt)
  - gripper        — IGNORED (cue is rigid-mounted)

Loop:
  1. Read current TCP pose (world frame)
  2. Grab fisheye frame
  3. predict_action → (chunk_len, 10) world-frame deltas
  4. Take first n_actions; roll out via PROPER SO(3) composition
     (R_new = R_delta @ R_curr)  →  predicted 10-D states
  5. For each predicted state: clamp XYZ, extract yaw, clamp yaw,
     resynthesize R from (locked_roll, locked_pitch, clamped_yaw)
  6. Interpolate policy rate (20 Hz) → servo rate (100 Hz)
  7. servoL each waypoint blocking, then re-loop

Usage:
    # Terminal 1: start the inference server
    python ur5n/pool-preference/fm/inference_server.py

    # Terminal 2:
    python ur5n/pool-preference/fm/closedloop_sync.py
    python ur5n/pool-preference/fm/closedloop_sync.py --no_auto_home --n_actions 4
    python ur5n/pool-preference/fm/closedloop_sync.py --task wall_bounce
"""

import sys
import os
import time
import json
import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import cv2
from scipy.spatial.transform import Rotation as Rot

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
    "checkpoints/discreteRTC/fastumi_pool_qwenPI_0522_DiT-S/"
    "checkpoints/steps_20000_pytorch_model.pt"
)

TASK_INSTRUCTIONS = {
    "pocket":      ("Strike the white ball into the red ball to pocket it", 0),
    "wall_bounce": ("Strike the white ball so that the red ball bounces "
                    "off the walls into the goal", 1),
}

POLICY_HZ = 20
INTERP_MULT = 5
SERVO_HZ = POLICY_HZ * INTERP_MULT          # 100 Hz

# Training state ranges (q01/q99 from norm_stats, in world frame).
# Used as safety clamps for X/Y/Z. See pool-static.md for derivation.
XYZ_DATA_MIN = np.array([0.1261, -0.6562, 0.2131], dtype=np.float64)
XYZ_DATA_MAX = np.array([0.4800, -0.2798, 0.3134], dtype=np.float64)

# Hard safety floor on world-frame Z (TCP / flange). This is a NON-NEGOTIABLE
# clamp that always runs after the soft (data-derived) bounds:
#   effective_z_min = max(soft_z_min, Z_HARD_FLOOR_WORLD)
# 0.1965 m is the bench-determined floor for this pool setup — it allows
# the cue tip ~1.65 cm of follow-through below the typical strike plane
# (TCP z ≈ 0.213) without letting it sink into / through the table.
# Override via --z_hard_floor only if you know what you are doing.
Z_HARD_FLOOR_WORLD = 0.1965

# Yaw range from training data (rad). Computed across all 388 episodes.
# Default ±0.1 rad (~6°) buffer is applied on top.
YAW_DATA_MIN = -2.7556    # ~-157.9°
YAW_DATA_MAX = -1.5688    # ~ -89.9°

# Training-distribution LEFT home — verbatim from strike_via_pocket.py.
# Pool data was recorded with the LEFT arm auto-homing here before every
# episode. Closed-loop runs default to moving here once at startup so the
# first inference sees an in-distribution scene.
LEFT_HOME_WORLD_XYZRV = [0.48, -0.4, 0.313351,
                          -2.2192, 2.2148, 0.0091]

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}


# ═══════════════════════════════════════════════════════════════════
#  Camera (V4L2 fisheye)
# ═══════════════════════════════════════════════════════════════════

class RealCamera:
    """V4L2 camera with background reader thread for low-latency grabs."""

    def __init__(self, dev=0, width=1920, height=1080, fps=30):
        self.dev = dev
        self.W = width
        self.H = height
        self.cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open /dev/video{dev}. Is the fisheye plugged in / "
                f"is another process (recorder?) holding it?")
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
#  Frame conversion + 10-D rotation helpers (SO(3) composition)
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


def mat_to_rot6d(mat):
    return mat[:2, :].flatten().astype(np.float32)


def rot6d_to_mat(d6):
    a1 = np.asarray(d6[:3], dtype=np.float64)
    a2 = np.asarray(d6[3:], dtype=np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_euler(d6):
    """rot6d -> (roll, pitch, yaw) radians, scipy 'xyz' intrinsic."""
    return Rot.from_matrix(rot6d_to_mat(d6)).as_euler('xyz')


def euler_to_mat(roll, pitch, yaw):
    return Rot.from_euler('xyz', [roll, pitch, yaw]).as_matrix()


def euler_to_rotvec(roll, pitch, yaw):
    return Rot.from_euler('xyz', [roll, pitch, yaw]).as_rotvec()


def rotvec_to_rot6d(rotvec):
    return mat_to_rot6d(
        Rot.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix())


def build_state_10d(pose_world_6d, gripper=1.0):
    pos = np.asarray(pose_world_6d[:3], dtype=np.float64)
    rot6d = rotvec_to_rot6d(pose_world_6d[3:6])
    return np.concatenate([pos, rot6d, [float(gripper)]]).astype(np.float32)


def accumulate_actions_to_states(start_state_10d, actions_10d):
    """Inverse of V3's compute_world_relative_action_10d.

    delta_pos applied as world-frame addition; delta_R applied as SO(3)
    left-composition R_new = R_delta @ R_curr."""
    pos = np.asarray(start_state_10d[:3], dtype=np.float64).copy()
    R = rot6d_to_mat(start_state_10d[3:9])
    T = actions_10d.shape[0]
    out = np.zeros((T, 10), dtype=np.float64)
    for t in range(T):
        pos = pos + actions_10d[t, :3]
        R_delta = rot6d_to_mat(actions_10d[t, 3:9])
        R = R_delta @ R
        out[t, :3] = pos
        out[t, 3:9] = mat_to_rot6d(R)
        out[t, 9] = actions_10d[t, 9]
    return out


# ═══════════════════════════════════════════════════════════════════
#  Interpolation + timing
# ═══════════════════════════════════════════════════════════════════

def interpolate_waypoints(start_pose, waypoints, mult):
    """Linear interp from start_pose (6-D rotvec) through waypoints (6-D).
    Rotvec interpolation is approximate but adequate at servo rate where
    each step's rotation change is sub-degree."""
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
#  Model loading (--no_use_server path)
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
        print("[FIX] Set image_size=[256,256] (match pool native)")

    print(f"Model loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Auto-home
# ═══════════════════════════════════════════════════════════════════

def auto_home(rtde_c, T_bw):
    """Move LEFT arm to LEFT_HOME_WORLD_XYZRV (matches training start)."""
    home = LEFT_HOME_WORLD_XYZRV
    target_pose_base = [
        home[0] - T_bw[0, 3],
        home[1] - T_bw[1, 3],
        home[2] - T_bw[2, 3],
        home[3], home[4], home[5],
    ]
    print(f"[auto-home] moving LEFT to world XYZ={home[:3]} "
          f"rotvec={home[3:6]} ...")
    joints = rtde_c.getInverseKinematics(target_pose_base)
    rtde_c.moveJ(joints, 1.0, 1.0)
    print("[auto-home] done")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pool sync closed-loop (QwenPI flow-matching, 10-D; "
                    "roll/pitch locked, X/Y/Z/yaw policy-controlled)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--task", choices=list(TASK_INSTRUCTIONS.keys()),
                        default="pocket")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction (bypasses --task)")
    parser.add_argument("--n_actions", type=int, default=8,
                        help="Steps to execute per inference (1..chunk_len). "
                             "Default 8 = 0.4s exec @ 20Hz.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference cycles (0 = unlimited)")
    parser.add_argument("--auto_home", action="store_true", default=True,
                        help="moveJ to LEFT_HOME before main loop (default ON; "
                             "training distribution starts here)")
    parser.add_argument("--no_auto_home", dest="auto_home",
                        action="store_false")
    parser.add_argument("--xy_buffer", type=float, default=0.03,
                        help="Safety buffer (m) outside XY training range")
    parser.add_argument("--z_buffer", type=float, default=0.02,
                        help="Safety buffer (m) outside Z training range")
    parser.add_argument("--yaw_buffer_deg", type=float, default=10.0,
                        help="Safety buffer (deg) outside yaw training range")
    parser.add_argument("--x_min", type=float, default=None)
    parser.add_argument("--x_max", type=float, default=None)
    parser.add_argument("--y_min", type=float, default=None)
    parser.add_argument("--y_max", type=float, default=None)
    parser.add_argument("--z_min", type=float, default=None)
    parser.add_argument("--z_max", type=float, default=None)
    parser.add_argument("--yaw_min_deg", type=float, default=None)
    parser.add_argument("--yaw_max_deg", type=float, default=None)
    parser.add_argument("--z_hard_floor", type=float,
                        default=Z_HARD_FLOOR_WORLD,
                        help=f"HARD floor on world-frame TCP z (m). Applied "
                             f"as max(soft z_min, this) so it can only make "
                             f"the floor stricter. Default "
                             f"{Z_HARD_FLOOR_WORLD} m — bench-verified for "
                             f"the pool setup. Lower it ONLY if you have "
                             f"physically verified the cue tip cannot hit "
                             f"the table at the new value.")
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--use_server", action="store_true", default=True)
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_pool.sock")
    args = parser.parse_args()

    if args.instruction is None:
        instruction, task_idx = TASK_INSTRUCTIONS[args.task]
    else:
        instruction = args.instruction
        task_idx = -1

    T_bw = BASE_IN_WORLD[args.arm]

    # ── Safety bounds ───────────────────────────────────────────────
    x_lo = args.x_min if args.x_min is not None else XYZ_DATA_MIN[0] - args.xy_buffer
    x_hi = args.x_max if args.x_max is not None else XYZ_DATA_MAX[0] + args.xy_buffer
    y_lo = args.y_min if args.y_min is not None else XYZ_DATA_MIN[1] - args.xy_buffer
    y_hi = args.y_max if args.y_max is not None else XYZ_DATA_MAX[1] + args.xy_buffer
    z_lo_soft = (args.z_min if args.z_min is not None
                 else XYZ_DATA_MIN[2] - args.z_buffer)
    z_hi = args.z_max if args.z_max is not None else XYZ_DATA_MAX[2] + args.z_buffer
    # HARD floor wins if it's stricter than the soft bound.
    z_lo = max(z_lo_soft, args.z_hard_floor)
    if z_lo >= z_hi:
        raise ValueError(
            f"z_lo ({z_lo:.4f}) >= z_hi ({z_hi:.4f}); refusing to start. "
            f"Check --z_hard_floor / --z_max settings.")
    yaw_buf_rad = np.deg2rad(args.yaw_buffer_deg)
    yaw_lo = (np.deg2rad(args.yaw_min_deg) if args.yaw_min_deg is not None
              else YAW_DATA_MIN - yaw_buf_rad)
    yaw_hi = (np.deg2rad(args.yaw_max_deg) if args.yaw_max_deg is not None
              else YAW_DATA_MAX + yaw_buf_rad)

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
    assert 1 <= args.n_actions <= chunk_len, (
        f"n_actions={args.n_actions} must be in [1, {chunk_len}]")

    # ── Connect robot ───────────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Auto-home ──────────────────────────────────────────────────
    if args.auto_home:
        auto_home(rtde_c, T_bw)

    # ── Open camera ────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # ── Lock roll + pitch from initial pose ────────────────────────
    pose_base_init = rtde_r.getActualTCPPose()
    pose_world_init = base_to_world(pose_base_init, T_bw)
    initial_R = Rot.from_rotvec(np.asarray(pose_world_init[3:6])).as_matrix()
    initial_euler = Rot.from_matrix(initial_R).as_euler('xyz')
    locked_roll = float(initial_euler[0])
    locked_pitch = float(initial_euler[1])
    initial_yaw = float(initial_euler[2])
    print(f"\nLocked roll/pitch (rad): "
          f"{locked_roll:+.4f} / {locked_pitch:+.4f}  "
          f"(deg: {np.rad2deg(locked_roll):+.2f}° / "
          f"{np.rad2deg(locked_pitch):+.2f}°)")
    print(f"Initial yaw (rad / deg): "
          f"{initial_yaw:+.4f} / {np.rad2deg(initial_yaw):+.2f}°")
    print(f"Initial pos (world): {[round(v, 4) for v in pose_world_init[:3]]}")

    # ── Rollout setup ───────────────────────────────────────────────
    rollout_dir = None
    rollout_log = []
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = (Path("ur5n/pool-preference/fm/rollouts_sync") /
                       f"sync_{ts}")
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "frames").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")
        with open(rollout_dir / "config.json", "w") as f:
            json.dump({
                "mode": "sync_pool",
                "checkpoint": args.checkpoint,
                "task": args.task,
                "task_idx": task_idx,
                "instruction": instruction,
                "arm": args.arm,
                "n_actions": args.n_actions,
                "chunk_len": chunk_len,
                "policy_hz": POLICY_HZ,
                "servo_hz": SERVO_HZ,
                "xyz_bounds": [[x_lo, y_lo, z_lo], [x_hi, y_hi, z_hi]],
                "yaw_bounds_rad": [yaw_lo, yaw_hi],
                "yaw_bounds_deg": [float(np.rad2deg(yaw_lo)),
                                    float(np.rad2deg(yaw_hi))],
                "locked_roll_rad": locked_roll,
                "locked_pitch_rad": locked_pitch,
                "dataset_key": dataset_key,
            }, f, indent=2)

    # ── Print config ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Pool Sync Closed-Loop (QwenPI, X/Y/Z/yaw control)")
    print(f"  Arm:           {args.arm}  IP={robot_ip}")
    print(f"  Camera:        /dev/video{args.camera_dev}")
    print(f"  Task:          {args.task} (idx={task_idx})")
    print(f"  Instruction:   \"{instruction}\"")
    print(f"  n_actions:     {args.n_actions} / chunk_len={chunk_len}")
    print(f"  Policy/Servo:  {POLICY_HZ} Hz × {INTERP_MULT} = {SERVO_HZ} Hz")
    print(f"  XYZ box:       x[{x_lo:.3f}, {x_hi:.3f}]  "
          f"y[{y_lo:.3f}, {y_hi:.3f}]  z[{z_lo:.4f}, {z_hi:.4f}]")
    print(f"  Z hard floor:  {args.z_hard_floor:.4f} m  "
          f"(soft was {z_lo_soft:.4f}; effective floor = {z_lo:.4f})")
    print(f"  Yaw box:       [{np.rad2deg(yaw_lo):+.2f}°, "
          f"{np.rad2deg(yaw_hi):+.2f}°]")
    print(f"  Locked rpy:    roll={np.rad2deg(locked_roll):+.2f}°  "
          f"pitch={np.rad2deg(locked_pitch):+.2f}°")
    print(f"  Use server:    {args.use_server}")
    print(f"  Max steps:     "
          f"{'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Save rollout:  {args.save_rollout}")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    servo_dt = 1.0 / SERVO_HZ
    infer_kwargs = dict()
    step = 0

    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current pose (full 6-D)
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_xyz = np.array(pose_world[:3], dtype=np.float64)
            current_R = Rot.from_rotvec(
                np.asarray(pose_world[3:6])).as_matrix()
            current_yaw = float(
                Rot.from_matrix(current_R).as_euler('xyz')[2])
            # Build current 6-D pose with roll+pitch reset to locked
            # values (so interpolation start point is clean).
            current_rotvec_locked = euler_to_rotvec(
                locked_roll, locked_pitch, current_yaw)
            current_6d = np.concatenate(
                [current_xyz, current_rotvec_locked]).astype(np.float64)
            # 10-D state for the accumulate roll-out
            state_10d = build_state_10d(
                np.concatenate([current_xyz, current_rotvec_locked]),
                gripper=1.0)

            # 2. Grab frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] frame dropped, retry")
                continue

            # 3. Inference
            example = {"image": [pil_img], "lang": instruction}
            t_infer = time.monotonic()
            out = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.monotonic() - t_infer) * 1000

            pred_norm = out["normalized_actions"][0].astype(np.float32)
            pred_actions_10d = baseframework.unnormalize_actions(
                pred_norm, action_stats).astype(np.float64)

            # 4. Roll out predicted states (SO(3) composition)
            n_exec = min(args.n_actions, len(pred_actions_10d))
            pred_states_10d = accumulate_actions_to_states(
                state_10d, pred_actions_10d[:n_exec])

            # 5. Per-step: clamp XYZ, extract+clamp yaw, lock roll/pitch
            waypoints_6d = np.zeros((n_exec, 6), dtype=np.float64)
            waypoints_xyz_clamped = []
            waypoints_yaw_clamped = []
            z_floor_hits = 0      # how many of this chunk's waypoints hit Z floor
            min_pred_z = float("inf")
            for i in range(n_exec):
                xyz = pred_states_10d[i, :3].copy()
                xyz[0] = np.clip(xyz[0], x_lo, x_hi)
                xyz[1] = np.clip(xyz[1], y_lo, y_hi)
                if xyz[2] < min_pred_z:
                    min_pred_z = float(xyz[2])
                if xyz[2] < z_lo:
                    z_floor_hits += 1
                xyz[2] = np.clip(xyz[2], z_lo, z_hi)

                pred_euler = rot6d_to_euler(pred_states_10d[i, 3:9])
                yaw_pred = float(pred_euler[2])
                yaw = float(np.clip(yaw_pred, yaw_lo, yaw_hi))

                # Resynthesize rotation with locked roll/pitch + clamped yaw
                rotvec = euler_to_rotvec(locked_roll, locked_pitch, yaw)
                waypoints_6d[i, :3] = xyz
                waypoints_6d[i, 3:6] = rotvec
                waypoints_xyz_clamped.append(xyz)
                waypoints_yaw_clamped.append(yaw)

            # 6. Interpolate to servo rate
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

            target_xyz = waypoints_6d[-1, :3]
            target_yaw_deg = np.rad2deg(waypoints_yaw_clamped[-1])
            floor_msg = ""
            if z_floor_hits > 0:
                floor_msg = (f"  [Z FLOOR HIT {z_floor_hits}/{n_exec}  "
                             f"min_pred_z={min_pred_z:.4f} → clamped to "
                             f"{z_lo:.4f}]")
            print(f"[step {step:4d}]  infer={infer_ms:5.0f}ms  "
                  f"exec={exec_ms:5.0f}ms  total={total_ms:5.0f}ms  "
                  f"now=[{current_xyz[0]:.3f},{current_xyz[1]:.3f},"
                  f"{current_xyz[2]:.3f}] yaw={np.rad2deg(current_yaw):+6.2f}°"
                  f"  → target=[{target_xyz[0]:.3f},{target_xyz[1]:.3f},"
                  f"{target_xyz[2]:.3f}] yaw={target_yaw_deg:+6.2f}°"
                  f"{floor_msg}")

            # 8. Save rollout
            if args.save_rollout and rollout_dir is not None:
                iso = datetime.fromtimestamp(
                    time.time(), tz=timezone.utc).strftime(
                    "%Y%m%dT%H%M%S_%f")[:-3]
                stem = f"step_{step:04d}_{iso}"
                pil_img.save(str(rollout_dir / "frames" / f"{stem}.jpg"),
                              quality=90)
                rollout_log.append({
                    "step": step,
                    "t_wall": time.time(),
                    "t_wall_iso_utc": iso,
                    "now_pos_world": current_xyz.tolist(),
                    "now_yaw_deg": float(np.rad2deg(current_yaw)),
                    "target_pos_world": target_xyz.tolist(),
                    "target_yaw_deg": float(target_yaw_deg),
                    "pred_actions_10d": pred_actions_10d.tolist(),
                    "pred_states_10d": pred_states_10d.tolist(),
                    "waypoints_xyz_clamped":
                        [w.tolist() for w in waypoints_xyz_clamped],
                    "waypoints_yaw_clamped_deg":
                        [float(np.rad2deg(y))
                         for y in waypoints_yaw_clamped],
                    "n_exec": n_exec,
                    "infer_ms": round(infer_ms, 1),
                    "exec_ms": round(exec_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "frame_file": f"frames/{stem}.jpg",
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
