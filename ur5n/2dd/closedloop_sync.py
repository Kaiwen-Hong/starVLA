#!/usr/bin/env python3
"""
Sync closed-loop control with discrete diffusion.

Receding-horizon loop:
  1. Read EE pose + grab camera frame
  2. Run model inference → predicted action chunk (chunk_len, 10)
  3. Execute first n_actions via interpolated servoL at 100Hz
  4. Repeat from step 1

Robot WILL move. Use Ctrl+C to stop.

Safety:
  - XYZ clamping (world frame, derived from dataset q01/q99 + margin)
  - Rotation fixed by default (--no_fix_rotation to enable)

Usage:
    python ur5n/dd/closedloop_sync.py
    python ur5n/dd/closedloop_sync.py --n_actions 8
    python ur5n/dd/closedloop_sync.py --instruction "pick up the block"
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
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import cv2

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))  # for robotiq_gripper
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import modular_policy

# ── Defaults ─────────────────────────────────────────────────────────
# DEFAULT_CHECKPOINT = (
#     "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
#     "checkpoints/steps_20000_pytorch_model.pt"
# )

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_0403_1_pick_from_moved/"
    "checkpoints/steps_30000_pytorch_model.pt"
)


DEFAULT_INSTRUCTION = "Pick up the purple block to the pan"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1

CONTROL_HZ = 20
INTERP_MULT = 5
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100Hz
# Safety clamp bounds in world frame (meters).
#
# World↔base conversion (left arm, R_bw = I, pure translation):
#   pos_world = pos_base + [0.80, -0.22, 0.02]
#   pos_base  = pos_world - [0.80, -0.22, 0.02]
#
# So these world-frame clamps correspond to base-frame clamps:
#   y_world > -0.50  →  y_base > -0.50 + 0.22 = -0.28
#   z_world >  0.03  →  z_base >  0.03 - 0.02 =  0.01
#   z_world <  0.30  →  z_base <  0.30 - 0.02 =  0.28
#
# BUG HISTORY: Y_MIN_WORLD was 0.05 but actual y ~ -0.428. The clamp
# (y < 0.05 → y = 0.05) would jerk the robot +0.48m every step.
Y_MIN_WORLD = -0.50
Z_MIN_WORLD = 0.03
Z_MAX_WORLD = 0.30

# ── Turntable zone (task: pick from rotating turntable → place on static pan) ──
# Turntable occupies y in [TURNTABLE_Y_MIN, TURNTABLE_Y_MAX]. Three regimes:
#   1. Not grasped, over turntable:
#        z >= TURNTABLE_SURFACE_Z  (allow descent to reach object on turntable)
#   2. Grasped, still over turntable:
#        z >= TURNTABLE_LIFT_Z     (lift to clear turntable obstacles)
#   3. Grasped, y < TURNTABLE_Y_MIN (off turntable, heading to static pan):
#        only the global Y_MIN_WORLD / Z_MIN_WORLD clamps apply.
TURNTABLE_Y_MIN = -0.4276
TURNTABLE_Y_MAX = 0.2931
TURNTABLE_SURFACE_Z = 0.101398
TURNTABLE_LIFT_Z = 0.125

# ── Robot config ─────────────────────────────────────────────────────
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}

# ── SLAM → gripper rotation correction ──────────────────────────────
#
# The home poses below come from the SLAM trajectory file
# (slam_raw_pose_worldframe_downsampled.txt). The SLAM tracker is
# mounted on the gripper but its local axes differ from the gripper's:
#
#   SLAM device axes:   x_slam, y_slam, z_slam
#   Gripper axes:        x_grip = z_slam
#                        y_grip = x_slam
#                        z_grip = y_slam
#
# So:  R_gripper = R_slam @ T_SLAM_GRIPPER^T
#      where T_SLAM_GRIPPER^T = [[0,0,1],[1,0,0],[0,1,0]]
#
# If we skip this correction and send the raw SLAM rotation to the UR5,
# the robot will orient its gripper to match the SLAM device's axes
# instead of the intended gripper orientation — the end-effector will
# point in the wrong direction.
#
# The replay_dataset.py script (load_session, line 131-134) applies the
# same correction when replaying raw SLAM trajectories on the robot.
# Position (x,y,z) is unaffected — it's already in world frame.
# ─────────────────────────────────────────────────────────────────────

def slam_to_gripper_rotation(rotvec):
    """Correct SLAM device rotation to UR5 gripper rotation."""
    T_inv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])  # T_SLAM_GRIPPER^T
    R = Rot.from_rotvec(rotvec).as_matrix() @ T_inv
    return Rot.from_matrix(R).as_rotvec()


# Raw poses from SLAM file (rotation in SLAM device frame)
_HOME_SLAM = {
    'left': [0.253835, -0.315847, 0.230922, 2.130869, 0.107971, -2.304919, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}

# Convert to gripper frame for UR5
HOME_POSES_WORLD = {}
for _arm, _pose in _HOME_SLAM.items():
    _rot_corrected = slam_to_gripper_rotation(np.array(_pose[3:6]))
    HOME_POSES_WORLD[_arm] = _pose[:3] + _rot_corrected.tolist() + [_pose[6]]


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
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
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
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]; p[1] -= T_bw[1, 3]; p[2] -= T_bw[2, 3]
    return p


# ═══════════════════════════════════════════════════════════════════
#  Rotation / action utilities
# ═══════════════════════════════════════════════════════════════════

def rot6d_to_axisangle(d6):
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=0)
    return Rot.from_matrix(R).as_rotvec().astype(np.float32)


def action_10d_to_delta7d(action_10d):
    """Single 10D action → 7D delta [dx,dy,dz,drx,dry,drz,gripper]."""
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = action_10d[:3]
    delta[3:6] = rot6d_to_axisangle(action_10d[3:9])
    delta[6] = action_10d[9]
    return delta


def actions_10d_to_7d(actions_10d):
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t] = action_10d_to_delta7d(actions_10d[t])
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
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
#  Interpolation & timing
# ═══════════════════════════════════════════════════════════════════

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
#  Model loading
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
        config.datasets.vla_data.image_size = [224, 224]
        print("[FIX] Set image_size=[224,224]")

    print(f"Model loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Go home
# ═══════════════════════════════════════════════════════════════════

def go_home(rtde_c, rtde_r, arm, T_bw, robot_ip):
    home = HOME_POSES_WORLD[arm]
    home_base = world_to_base(home[:6], T_bw)
    target_joints = rtde_c.getInverseKinematics(home_base)
    print(f"Moving to home pose (world): {[round(x, 2) for x in home[:6]]}")
    rtde_c.moveJ(target_joints, 1.0, 1.0)

    current_base = rtde_r.getActualTCPPose()
    current_world = base_to_world(current_base, T_bw)
    print(f"Reached: {[round(x, 4) for x in current_world]}")

    from robotiq_gripper import RobotiqGripper
    g = RobotiqGripper()
    g.connect(hostname=robot_ip, port=63352)
    g.move(int((1.0 - home[6]) * 255), 255, 150)
    g.disconnect()
    print("Gripper opened.")
    return home[6]


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_step(traj_world, traj_base, camera_image,
                   current_world, current_base, n_exec,
                   step, save_path, instruction):
    """Camera (left) + world trajectory (mid) + base trajectory (right).
    Solid = executed, dashed = predicted-only."""
    T = traj_world.shape[0]
    ts = np.arange(T) / 20.0

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.15, wspace=0.35,
                  width_ratios=[1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    dims = [
        (0, "x", "#e41a1c", current_world[0], current_base[0]),
        (1, "y", "#377eb8", current_world[1], current_base[1]),
        (2, "z", "#4daf4a", current_world[2], current_base[2]),
        (6, "gripper", "#ff7f00", None, None),
    ]

    axes_w = []
    axes_b = []
    for row, (dim_idx, name, color, sw, sb) in enumerate(dims):
        # World frame column
        share_w = axes_w[0] if axes_w else None
        ax_w = fig.add_subplot(gs[row, 1], sharex=share_w)
        axes_w.append(ax_w)

        vals_w = traj_world[:, dim_idx]
        ax_w.plot(ts[:n_exec], vals_w[:n_exec], "o-", markersize=4,
                  linewidth=2.0, color=color)
        if n_exec < T:
            ax_w.plot(ts[n_exec - 1:], vals_w[n_exec - 1:], "o--",
                      markersize=3, linewidth=1.2, color=color, alpha=0.4)
        if sw is not None:
            ax_w.axhline(sw, color=color, linewidth=0.8, linestyle=":",
                         alpha=0.4, label=f"now={sw:.4f}")
            ax_w.legend(fontsize=7, loc="upper right")
        if n_exec < T:
            ax_w.axvline(ts[n_exec - 1], color="black", linewidth=1.0,
                         linestyle=":", alpha=0.5)
        ax_w.set_ylabel(f"{name} (world)", fontsize=9, fontweight="bold")
        ax_w.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_w.set_title("WORLD FRAME", fontsize=11, fontweight="bold")
        if row < len(dims) - 1:
            plt.setp(ax_w.get_xticklabels(), visible=False)
        else:
            ax_w.set_xlabel("Time (s)  solid=exec  dashed=pred", fontsize=9)

        # Base frame column
        share_b = axes_b[0] if axes_b else None
        ax_b = fig.add_subplot(gs[row, 2], sharex=share_b)
        axes_b.append(ax_b)

        vals_b = traj_base[:, dim_idx]
        ax_b.plot(ts[:n_exec], vals_b[:n_exec], "o-", markersize=4,
                  linewidth=2.0, color=color)
        if n_exec < T:
            ax_b.plot(ts[n_exec - 1:], vals_b[n_exec - 1:], "o--",
                      markersize=3, linewidth=1.2, color=color, alpha=0.4)
        if sb is not None:
            ax_b.axhline(sb, color=color, linewidth=0.8, linestyle=":",
                         alpha=0.4, label=f"now={sb:.4f}")
            ax_b.legend(fontsize=7, loc="upper right")
        if n_exec < T:
            ax_b.axvline(ts[n_exec - 1], color="black", linewidth=1.0,
                         linestyle=":", alpha=0.5)
        ax_b.set_ylabel(f"{name} (base)", fontsize=9, fontweight="bold")
        ax_b.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_b.set_title("ROBOT (BASE) FRAME", fontsize=11, fontweight="bold")
        if row < len(dims) - 1:
            plt.setp(ax_b.get_xticklabels(), visible=False)
        else:
            ax_b.set_xlabel("Time (s)  solid=exec  dashed=pred", fontsize=9)

    cw = current_world
    cb = current_base
    fig.suptitle(
        f'Closed-Loop Sync (DD) — Step {step}\n'
        f'"{instruction}"\n'
        f'World: [{cw[0]:.4f}, {cw[1]:.4f}, {cw[2]:.4f}, {cw[3]:.4f}, {cw[4]:.4f}, {cw[5]:.4f}]\n'
        f'Base:  [{cb[0]:.4f}, {cb[1]:.4f}, {cb[2]:.4f}, {cb[3]:.4f}, {cb[4]:.4f}, {cb[5]:.4f}]',
        fontsize=10, fontweight="bold", y=1.0, va="bottom")

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sync closed-loop control with discrete diffusion")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=4,
                        help="Number of predicted actions to execute per inference")
    parser.add_argument("--n_actions_after_grasp", type=int, default=16,
                        help="Number of actions to execute for the first N chunks after grasping")
    parser.add_argument("--n_chunks_after_grasp", type=int, default=2,
                        help="How many chunks to use n_actions_after_grasp (default: 2)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps (0=unlimited, Ctrl+C to stop)")
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--fix_rotation", action="store_true", default=True,
                        help="Zero out rotation deltas (default: True)")
    parser.add_argument("--no_fix_rotation", dest="fix_rotation",
                        action="store_false",
                        help="Enable rotation control from policy")
    parser.add_argument("--if_grasped_not_release", action="store_true",
                        default=False,
                        help="Once gripper closes, keep it closed (default: False for pick-from-turntable task)")
    parser.add_argument("--allow_release", dest="if_grasped_not_release",
                        action="store_false",
                        help="Allow gripper to re-open after grasping")
    parser.add_argument("--if_grasp_delay_temp_solution", action="store_true",
                        default=False,
                        help="Use more actions for N chunks after grasping (default: True)")
    parser.add_argument("--no_grasp_delay", dest="if_grasp_delay_temp_solution",
                        action="store_false",
                        help="Disable post-grasp extended execution")
    parser.add_argument("--if_release_when_reach_temp", action="store_true",
                        default=False,
                        help="Allow release when y >= release_y_threshold, then exit (default: False for pick-from-turntable task)")
    parser.add_argument("--no_release_when_reach", dest="if_release_when_reach_temp",
                        action="store_false",
                        help="Disable auto-release at target zone")
    parser.add_argument("--release_y_threshold", type=float, default=-0.338,
                        help="Y world-frame threshold for allowing release (default: -0.338)")
    parser.add_argument("--grasp_detect_threshold", type=int, default=200,
                        help="Gripper position threshold to confirm actual grasp "
                             "(0-255, empty close ~230, default 200)")
    parser.add_argument("--if_grasp_trick", action="store_true", default=True,
                        help="Only allow gripper close when z < grasp_z_threshold (default: True)")
    parser.add_argument("--no_grasp_trick", dest="if_grasp_trick",
                        action="store_false",
                        help="Allow gripper close at any height")
    parser.add_argument("--grasp_z_threshold", type=float, default=0.117,
                        help="Z world-frame threshold below which grasping is allowed. "
                             "For pick-from-turntable: turntable surface is at "
                             "TURNTABLE_SURFACE_Z=0.101, so 0.117 gives a ~1.6cm grasp "
                             "window above the surface (default: 0.117)")
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--no_save_rollout", dest="save_rollout",
                        action="store_false")
    # ── Inference server (mirrors closedloop_rtc_v7). Default ON: start
    # `python ur5n/2dd/inference_server.py` in another terminal first,
    # then this script connects via Unix socket and proxies all
    # predict_action calls to it. Pass --no_use_server for the legacy
    # in-process load.
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix "
                             "socket instead of loading the model in-process.")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false",
                        help="Load the model in-process (legacy ~30s startup)")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer.sock")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]
    servo_dt = 1.0 / SERVO_HZ

    # ── Load model (in-process or remote via inference_server.py) ────
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

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    # ── Connect to robot ─────────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
    from robotiq_gripper import RobotiqGripper

    robot_ip = ROBOT_IPS[args.arm]
    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    print("Connecting to gripper...")
    gripper_hw = RobotiqGripper()
    gripper_hw.connect(hostname=robot_ip, port=63352)
    current_gripper = 0.0

    # ── Open camera ──────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Waiting for first frame from background reader...")
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # ── Go home ──────────────────────────────────────────────────────
    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)
        current_gripper = 0.0

    # ── Rollout saving ───────────────────────────────────────────────
    # NOTE: rollouts go under ur5n/2dd/rollouts (this script lives in 2dd/).
    # The original dd/closedloop_sync.py hardcodes ur5n/dd/rollouts; don't
    # mix them — each task directory owns its own rollout history so we can
    # tell pick-from-static (dd) and pick-from-turntable (2dd) apart at a
    # glance.
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = Path("ur5n/2dd/rollouts") / f"sync_{ts}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")

        run_config = {
            "checkpoint": args.checkpoint,
            "instruction": args.instruction,
            "arm": args.arm,
            "n_actions": args.n_actions,
            "max_steps": args.max_steps,
            "chunk_len": chunk_len,
            "control_hz": CONTROL_HZ,
            "servo_hz": SERVO_HZ,
            "y_min_world": Y_MIN_WORLD,
            "z_bounds_world": [Z_MIN_WORLD, Z_MAX_WORLD],
            # Turntable zone (pick-from-turntable task) — logged so rollouts
            # are self-describing and we can retroactively reconstruct
            # which z-floor was active during each cycle.
            "turntable_y_range": [TURNTABLE_Y_MIN, TURNTABLE_Y_MAX],
            "turntable_surface_z": TURNTABLE_SURFACE_Z,
            "turntable_lift_z": TURNTABLE_LIFT_Z,
            "if_grasp_trick": args.if_grasp_trick,
            "grasp_z_threshold": args.grasp_z_threshold,
            "if_grasped_not_release": args.if_grasped_not_release,
            "if_release_when_reach_temp": args.if_release_when_reach_temp,
            "fix_rotation": args.fix_rotation,
            "decode_temperature": args.decode_temperature,
            "choice_temperature": args.choice_temperature,
            "dataset_key": dataset_key,
        }
        with open(rollout_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    # ── Print config ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Closed-Loop Sync (Discrete Diffusion)")
    print(f"  Arm:             {args.arm}")
    print(f"  Instruction:     \"{args.instruction}\"")
    print(f"  n_actions:       {args.n_actions}")
    print(f"  Chunk len:       {chunk_len}")
    print(f"  Servo:           {CONTROL_HZ}Hz x {INTERP_MULT} = {SERVO_HZ}Hz")
    print(f"  Safety:          y > {Y_MIN_WORLD:.2f}, "
          f"z in [{Z_MIN_WORLD:.3f}, {Z_MAX_WORLD:.2f}]")
    print(f"  fix_rotation:    {args.fix_rotation}")
    print(f"  Max steps:       {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Control loop ─────────────────────────────────────────────────
    step = 0
    chunks_since_grasp = None  # None = not yet grasped
    object_grasped = False

    # Sticky latch for turntable z-floor. See long comment inside the loop
    # where it is consulted; tl;dr: once we have ever successfully grasped
    # something while over the turntable, keep the z-floor at TURNTABLE_LIFT_Z
    # until we physically leave the turntable y-range — even if the policy
    # transiently re-opens the gripper. This prevents a "close → open → re-descend"
    # oscillation from driving the EE back down onto the turntable surface.
    has_ever_grasped_in_turntable = False
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()

            # 1. Read current robot state
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_pos = np.array(pose_world, dtype=np.float64)

            # Reset the sticky turntable-grasp latch when the robot has
            # physically left the turntable y-range. Rationale: the latch
            # only exists to stop the EE from re-descending inside the
            # turntable after a successful grasp. Once we've carried the
            # object out (y < TURNTABLE_Y_MIN, on the static-pan side), the
            # latch has done its job; if the arm re-enters the turntable
            # later (e.g. a second pick attempt), we want to start fresh
            # with "not yet grasped" → floor = TURNTABLE_SURFACE_Z so the
            # EE can descend to the new object.
            if not (TURNTABLE_Y_MIN <= current_pos[1] <= TURNTABLE_Y_MAX):
                has_ever_grasped_in_turntable = False

            # 2. Grab camera frame
            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] Camera frame dropped, retrying...")
                continue

            # 3. Inference
            example = {"image": [pil_img], "lang": args.instruction}
            t_infer = time.monotonic()
            output = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.monotonic() - t_infer) * 1000

            pred_normalized = output["normalized_actions"][0].astype(np.float32)
            pred_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # 4. Convert to absolute world-frame waypoints
            #
            # if_grasp_delay_temp_solution (temporary workaround):
            #   Problem: After grasping, the policy often predicts small/hesitant
            #   deltas for the first few chunks, causing the robot to "get stuck"
            #   near the grasp location instead of lifting and moving to the target.
            #   This happens because the closed-loop re-inference sees the object
            #   still near the grasp point and keeps predicting grasp-like actions.
            #
            #   Solution: For the first N chunks after the gripper closes, execute
            #   more actions per chunk (e.g. 12 instead of 8). This lets the robot
            #   commit to a longer horizon and move away from the grasp location
            #   before re-inferring, breaking the "stuck" loop.
            #
            #   This is a TEMPORARY fix — ideally the policy should handle this
            #   natively, e.g. via action chunking with temporal ensembling or
            #   by conditioning on gripper state.
            if (args.if_grasp_delay_temp_solution
                    and chunks_since_grasp is not None
                    and chunks_since_grasp < args.n_chunks_after_grasp):
                n_actions = args.n_actions_after_grasp
                chunks_since_grasp += 1
            else:
                n_actions = args.n_actions
            n_exec = min(n_actions, len(pred_10d))
            waypoints = np.zeros((n_exec, 6), dtype=np.float64)
            pos = current_pos[:3].copy()
            rot = current_pos[3:6].copy()  # fixed rotation

            task_finished = False
            for i in range(n_exec):
                delta = action_10d_to_delta7d(pred_10d[i])

                if args.fix_rotation:
                    delta[3:6] = 0.0

                pos = pos + delta[:3]
                if not args.fix_rotation:
                    rot = rot + delta[3:6]

                # Safety clamps (world frame)
                pos[1] = max(pos[1], Y_MIN_WORLD)
                pos[2] = np.clip(pos[2], Z_MIN_WORLD, Z_MAX_WORLD)
                # Turntable zone (pick-from-turntable task):
                #   not yet grasped → floor = TURNTABLE_SURFACE_Z
                #     (allow EE to descend and reach the object surface)
                #   ever grasped in turntable → floor = TURNTABLE_LIFT_Z
                #     (lift above turntable obstacles and KEEP it lifted)
                #
                # We consult the STICKY `has_ever_grasped_in_turntable`
                # instead of the instantaneous `object_grasped` on purpose.
                # If we used `object_grasped`, a transient policy open-cmd
                # over the turntable would flip the floor back down to
                # TURNTABLE_SURFACE_Z and the EE could re-descend onto the
                # turntable surface — causing a close/open/re-descend
                # oscillation. The sticky latch is reset once the EE has
                # physically left the turntable y-range (see latch reset
                # near the start of the cycle).
                if TURNTABLE_Y_MIN <= pos[1] <= TURNTABLE_Y_MAX:
                    if has_ever_grasped_in_turntable:
                        pos[2] = max(pos[2], TURNTABLE_LIFT_Z)
                    else:
                        pos[2] = max(pos[2], TURNTABLE_SURFACE_Z)

                waypoints[i, :3] = pos
                waypoints[i, 3:6] = rot

                # Handle gripper on transition
                new_gripper = float(delta[6])

                # if_release_when_reach_temp:
                #   When the robot has grasped an object and carried it to the
                #   target zone (y >= release_y_threshold), lift the grasped_not_release
                #   lock so the policy can open the gripper to place the object.
                #   Once it actually releases, the task is considered done → exit.
                reached_target = (args.if_release_when_reach_temp
                                  and current_gripper > 0.5
                                  and object_grasped
                                  and pos[1] >= args.release_y_threshold)

                if reached_target and new_gripper <= 0.5:
                    # At target zone, policy says open → release and finish
                    print(f"  Gripper -> RELEASE at target (y={pos[1]:.4f} >= {args.release_y_threshold})")
                    gripper_hw.move(0, 255, 150)
                    current_gripper = 0.0
                    object_grasped = False
                    # Execute remaining waypoints up to this point, then exit
                    n_exec = i + 1
                    waypoints = waypoints[:n_exec]
                    task_finished = True
                    break
                elif (args.if_grasped_not_release
                      and current_gripper > 0.5
                      and object_grasped
                      and new_gripper <= 0.5):
                    pass  # keep closed, not at target yet
                elif (new_gripper > 0.5) != (current_gripper > 0.5):
                    # if_grasp_trick: only allow closing when the EE is
                    # ALREADY physically low enough at the start of this
                    # cycle.
                    #
                    # Subtlety (BUG FIXED 2026-04-10): gripper_hw.move() below
                    # is sent IMMEDIATELY inline during waypoint generation —
                    # before the interp/exec phase actually servos the arm
                    # through `waypoints`. At the moment the close command
                    # fires over the network, the robot is still at
                    # `current_pos` (the cycle-start pose from
                    # getActualTCPPose), NOT at `pos[i]` which is just a
                    # predicted future waypoint.
                    #
                    # Original code checked `pos[2] >= threshold`, which
                    # was wrong: `pos[2]` is an accumulated-delta waypoint
                    # that may predict a dip below threshold later in the
                    # chunk — but that dip hasn't been executed yet. The
                    # physical robot is still at current_pos and could be
                    # well above threshold, causing premature grasps.
                    # Observed example: cycle started at phys_z=0.148 but
                    # the grasp still fired because waypoint[i]_z was low.
                    #
                    # Fix: gate on `current_pos[2]` (the physical z). Grasp
                    # only fires once a prior cycle's exec has ALREADY
                    # moved the EE into the [TURNTABLE_SURFACE_Z, threshold]
                    # window.
                    if (args.if_grasp_trick
                            and new_gripper > 0.5
                            and current_pos[2] >= args.grasp_z_threshold):
                        print(f"  [GRASP_TRICK] skip CLOSE: "
                              f"phys_z={current_pos[2]:.4f} >= "
                              f"thr={args.grasp_z_threshold:.4f}  "
                              f"(waypoint[{i}]_z={pos[2]:.4f})")
                        continue  # too high, skip close command
                    grip_pos = int(new_gripper * 255)
                    label = "CLOSE" if new_gripper > 0.5 else "OPEN"
                    print(f"  Gripper -> {label} (pos={grip_pos})  "
                          f"[phys_z={current_pos[2]:.4f}, "
                          f"wp[{i}]_z={pos[2]:.4f}, "
                          f"thr={args.grasp_z_threshold:.4f}]")
                    gripper_hw.move(grip_pos, 255, 150)
                    current_gripper = new_gripper

                    # Verify actual grasp after closing
                    if new_gripper > 0.5:
                        actual_pos = gripper_hw.get_current_position()
                        if actual_pos < args.grasp_detect_threshold:
                            object_grasped = True
                            # STICKY LATCH: if this successful grasp happened
                            # while the EE was over the turntable, remember it
                            # so the z-floor stays at TURNTABLE_LIFT_Z for the
                            # rest of the turntable traversal — even if the
                            # policy later opens the gripper in mid-air.
                            if TURNTABLE_Y_MIN <= pos[1] <= TURNTABLE_Y_MAX:
                                has_ever_grasped_in_turntable = True
                            print(f"  [GRASP_DETECT] Object grasped "
                                  f"(pos={actual_pos} < {args.grasp_detect_threshold})")
                        else:
                            object_grasped = False
                            current_gripper = 0.0
                            chunks_since_grasp = None
                            print(f"  [GRASP_DETECT] Empty grasp "
                                  f"(pos={actual_pos} >= {args.grasp_detect_threshold}), "
                                  f"re-opening gripper")
                            gripper_hw.move(0, 255, 150)
                    else:
                        object_grasped = False

                    # Start post-grasp counter on first close
                    if (args.if_grasp_delay_temp_solution
                            and object_grasped
                            and chunks_since_grasp is None):
                        chunks_since_grasp = 0

            # 5. Interpolate and execute at 100Hz
            interp = interpolate_waypoints(current_pos, waypoints, INTERP_MULT)

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
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, "
                  f"{pose_world[2]:.3f}]  "
                  f"grip={'C' if current_gripper > 0.5 else 'O'}")

            # 6. Save rollout
            if args.save_rollout and rollout_dir is not None:
                img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                pil_img.save(str(img_path), quality=90)

                # Build full predicted trajectory for viz
                all_deltas_7d = actions_10d_to_7d(pred_10d)
                if args.fix_rotation:
                    all_deltas_7d[:, 3:6] = 0.0
                traj_world = accumulate_deltas(pose_world, all_deltas_7d)
                # Safety clamps for viz (match execution clamps)
                traj_world[:, 1] = np.maximum(traj_world[:, 1], Y_MIN_WORLD)
                traj_world[:, 2] = np.clip(traj_world[:, 2], Z_MIN_WORLD, Z_MAX_WORLD)
                # Turntable zone constraint — uses the sticky latch so the
                # visualized trajectory matches what executeion actually enforces.
                # Approximation: a single floor value is applied to the whole
                # chunk based on the latch state at the START of the cycle.
                in_turntable = ((traj_world[:, 1] >= TURNTABLE_Y_MIN)
                                & (traj_world[:, 1] <= TURNTABLE_Y_MAX))
                z_floor_viz = (TURNTABLE_LIFT_Z
                               if has_ever_grasped_in_turntable
                               else TURNTABLE_SURFACE_Z)
                traj_world[in_turntable, 2] = np.maximum(traj_world[in_turntable, 2], z_floor_viz)

                traj_base = traj_world.copy()
                traj_base[:, 0] -= T_bw[0, 3]
                traj_base[:, 1] -= T_bw[1, 3]
                traj_base[:, 2] -= T_bw[2, 3]

                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_step(
                    traj_world, traj_base, pil_img,
                    pose_world, list(pose_base), n_exec,
                    step, str(viz_path), args.instruction)

                rollout_log.append({
                    "step": step,
                    "wall_time": time.time(),
                    "ee_pose_world": list(pose_world),
                    "ee_pose_base": list(pose_base),
                    "gripper": current_gripper,
                    "pred_actions_10d": pred_10d.tolist(),
                    "n_exec": n_exec,
                    "waypoints_world": waypoints.tolist(),
                    "infer_ms": round(infer_ms, 1),
                    "exec_ms": round(exec_ms, 1),
                    "total_ms": round(total_ms, 1),
                    "image_file": f"images/step_{step:04d}.jpg",
                    "viz_file": f"images/step_{step:04d}_viz.png",
                })

            step += 1

            if task_finished:
                print("\n>>> Task finished! Object released at target. <<<")
                break

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
        try:
            gripper_hw.disconnect()
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
            print(f"Rollout saved: {rollout_dir}")
            print(f"  {len(rollout_log)} steps")

        print(f"Done. Executed {step} inference steps.")


if __name__ == "__main__":
    main()
