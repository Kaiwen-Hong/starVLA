#!/usr/bin/env python3
"""
Semi-closed-loop evaluation: replay GT trajectory while comparing model predictions.

For each step t in the GT trajectory (sync):
  1. Robot is at GT pose[t]
  2. Capture camera observation → save image
  3. Run model inference → predicted action chunk (chunk_len, 10)
  4. Compute GT action chunk from trajectory (chunk_len, 10)
  5. Compare predicted vs GT (MSE/L1, per-dim, per-step-in-chunk)
  6. Execute GT action → move robot to pose[t+1] via servoL
  7. Repeat

Robot WILL move (following GT trajectory). Use Ctrl+C to stop.

Usage:
    python ur5n/dd/semi_closedloop_eval.py
    python ur5n/dd/semi_closedloop_eval.py --session <path>
    python ur5n/dd/semi_closedloop_eval.py --start_step 10 --max_steps 50
"""

import sys
import os
import time
import json
import argparse
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
sys.path.insert(0, str(REPO_DIR / "ur5"))  # for robotiq_gripper
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import modular_policy

# ── Defaults ─────────────────────────────────────────────────────────
DATA_BASE = "/home/kaiwen/Desktop/research/fastumipro-collection/data_collector_opt/DATA"
DATA_SESSION = "left_hand_250801DR48FP25002960/dynamic-329-v4/session_20260329_213550"

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
DEFAULT_INSTRUCTION = "Pick up the purple block and place it on the red area of the board"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1

CONTROL_HZ = 20
INTERP_MULT = 5
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100Hz

# ── Robot config ─────────────────────────────────────────────────────
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}

DIM_LABELS_10D = [
    "pos_x", "pos_y", "pos_z",
    "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]


# ═══════════════════════════════════════════════════════════════════
#  Camera
# ═══════════════════════════════════════════════════════════════════

class RealCamera:
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

    def grab_rgb(self):
        ok, raw = self.cap.read()
        if not ok:
            return None
        yuv = np.ascontiguousarray(raw).reshape(self.H * 3 // 2, self.W)
        bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def flush(self, n=5):
        for _ in range(n):
            self.cap.read()

    def grab_pil(self):
        self.flush()
        rgb = self.grab_rgb()
        if rgb is None:
            return None
        h, w = rgb.shape[:2]
        s = min(h, w)
        left, top = (w - s) // 2, (h - s) // 2
        return Image.fromarray(rgb[top:top + s, left:left + s])

    def close(self):
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

def mat_to_rot6d(R):
    """(3,3) -> (6,) first two rows flattened."""
    return R[:2, :].flatten().astype(np.float64)


def rot6d_to_mat(d6):
    """(6,) -> (3,3) via Gram-Schmidt."""
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_axisangle(d6):
    return Rotation.from_matrix(rot6d_to_mat(d6)).as_rotvec().astype(np.float32)


def actions_10d_to_7d(actions_10d):
    """(T, 10) -> (T, 7) [dx,dy,dz,drx,dry,drz,gripper]."""
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t, :3] = actions_10d[t, :3]
        out[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        out[t, 6] = actions_10d[t, 9]
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
    """Accumulate world-frame 7D deltas on current pose -> (T, 7) absolute."""
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
#  Load session trajectory
# ═══════════════════════════════════════════════════════════════════

# SLAM device -> gripper rotation correction
T_SLAM_GRIPPER_INV = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])


def load_session(session_dir):
    """Load trajectory from session SLAM data.

    Returns:
        poses_world: (N, 6) [x, y, z, rx, ry, rz] axis-angle, world frame
        grippers:    (N,)   0=open, 1=closed (model convention)
    """
    pose_file = os.path.join(session_dir, 'SLAM_Poses',
                             'slam_raw_pose_worldframe_downsampled.txt')
    if not os.path.isfile(pose_file):
        raise FileNotFoundError(f"Not found: {pose_file}")

    rows = []
    with open(pose_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                rows.append([float(v) for v in parts[1:8]])  # skip timestamp

    data = np.array(rows, dtype=np.float64)
    poses = data[:, :6]   # x, y, z, rx, ry, rz
    clamp_open = data[:, 6]  # normalized clamp width, 1=open

    # SLAM device -> gripper rotation correction
    for i in range(len(poses)):
        R = Rotation.from_rotvec(poses[i, 3:6]).as_matrix() @ T_SLAM_GRIPPER_INV
        poses[i, 3:6] = Rotation.from_matrix(R).as_rotvec()

    # Convert clamp_open to model convention: 0=open, 1=closed
    grippers = np.where(clamp_open < 0.8, 1.0, 0.0)

    print(f"Loaded session: {len(poses)} poses")
    print(f"  gripper closed steps: {int((grippers > 0.5).sum())}/{len(grippers)}")
    return poses, grippers


def compute_gt_action_10d(pose_curr, pose_next, gripper_next):
    """Compute one GT action in 10D format from two consecutive poses.

    Action = [Δx, Δy, Δz, Δrot6d(6), gripper_target]
    Position: world-frame subtraction.
    Rotation: R_delta = R_next @ R_curr.T → rot6d.
    Gripper: absolute target.
    """
    delta_pos = pose_next[:3] - pose_curr[:3]

    R_curr = Rotation.from_rotvec(pose_curr[3:6]).as_matrix()
    R_next = Rotation.from_rotvec(pose_next[3:6]).as_matrix()
    R_delta = R_next @ R_curr.T
    delta_rot6d = mat_to_rot6d(R_delta)

    return np.concatenate([delta_pos, delta_rot6d, [gripper_next]])


def compute_gt_chunk(poses, grippers, t, chunk_len):
    """Compute GT action chunk at step t. Returns (valid_len, 10) and valid_len."""
    N = len(poses)
    valid_len = min(chunk_len, N - 1 - t)
    if valid_len <= 0:
        return np.zeros((0, 10), dtype=np.float64), 0

    gt = np.zeros((valid_len, 10), dtype=np.float64)
    for k in range(valid_len):
        gt[k] = compute_gt_action_10d(
            poses[t + k], poses[t + k + 1], grippers[t + k + 1])
    return gt, valid_len


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
#  Robot helpers
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


def go_to_pose(rtde_c, rtde_r, target_world, T_bw):
    """Move robot to a world-frame pose via moveJ (blocking)."""
    target_base = world_to_base(target_world[:6], T_bw)
    target_joints = rtde_c.getInverseKinematics(target_base)
    rtde_c.moveJ(target_joints, 1.0, 1.0)
    final_base = rtde_r.getActualTCPPose()
    final_world = base_to_world(final_base, T_bw)
    return final_world


def execute_one_step_servo(rtde_c, current_pos, target_world, T_bw):
    """Move from current to target via interpolated servoL (one 20Hz step)."""
    servo_dt = 1.0 / SERVO_HZ
    target = np.array(target_world[:6], dtype=np.float64)
    interp = interpolate_waypoints(current_pos, target.reshape(1, -1), INTERP_MULT)

    t_start = time.monotonic()
    for i, pose_w in enumerate(interp):
        target_base = world_to_base(pose_w.tolist(), T_bw)
        rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
        precise_wait(t_start + (i + 1) * servo_dt)
    rtde_c.servoStop()


# ═══════════════════════════════════════════════════════════════════
#  Metrics
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(pred_10d, gt_10d, valid_len):
    """Compare pred vs GT action chunks. Returns dict of metrics."""
    pred = pred_10d[:valid_len]
    gt = gt_10d[:valid_len]

    mse = float(np.mean((pred - gt) ** 2))
    l1 = float(np.mean(np.abs(pred - gt)))

    per_dim_mse = np.mean((pred - gt) ** 2, axis=0)
    per_dim_l1 = np.mean(np.abs(pred - gt), axis=0)

    per_step_mse = np.mean((pred - gt) ** 2, axis=1)

    return {
        "mse": mse,
        "l1": l1,
        "valid_len": valid_len,
        "per_dim_mse": {DIM_LABELS_10D[i]: float(per_dim_mse[i]) for i in range(10)},
        "per_dim_l1": {DIM_LABELS_10D[i]: float(per_dim_l1[i]) for i in range(10)},
        "per_step_mse": [float(v) for v in per_step_mse],
    }


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_comparison(pred_traj_w, gt_traj_w, pred_traj_b, gt_traj_b,
                         camera_image, current_world, current_base,
                         step, save_path, instruction, mse, l1):
    """Camera (left) + world pred/GT (mid) + base pred/GT (right)."""
    T_pred = pred_traj_w.shape[0]
    T_gt = gt_traj_w.shape[0]
    T_max = max(T_pred, T_gt)
    ts_pred = np.arange(T_pred) / 20.0
    ts_gt = np.arange(T_gt) / 20.0

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(4, 3, figure=fig, hspace=0.15, wspace=0.35,
                  width_ratios=[1, 1.2, 1.2])

    # Left: camera
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

        ax_w.plot(ts_pred, pred_traj_w[:, dim_idx], "o-", markersize=3,
                  linewidth=1.8, color=color, label="pred" if row == 0 else None)
        ax_w.plot(ts_gt, gt_traj_w[:, dim_idx], "x--", markersize=4,
                  linewidth=1.5, color="gray", alpha=0.7,
                  label="GT" if row == 0 else None)
        if sw is not None:
            ax_w.axhline(sw, color=color, linewidth=0.8, linestyle=":", alpha=0.4)
        ax_w.set_ylabel(f"{name} (world)", fontsize=9, fontweight="bold")
        ax_w.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_w.set_title("WORLD FRAME", fontsize=11, fontweight="bold")
            ax_w.legend(fontsize=8, loc="upper right")
        if row < len(dims) - 1:
            plt.setp(ax_w.get_xticklabels(), visible=False)
        else:
            ax_w.set_xlabel("Time (s) — 20Hz", fontsize=9)

        # Base frame column
        share_b = axes_b[0] if axes_b else None
        ax_b = fig.add_subplot(gs[row, 2], sharex=share_b)
        axes_b.append(ax_b)

        ax_b.plot(ts_pred, pred_traj_b[:, dim_idx], "o-", markersize=3,
                  linewidth=1.8, color=color, label="pred" if row == 0 else None)
        ax_b.plot(ts_gt, gt_traj_b[:, dim_idx], "x--", markersize=4,
                  linewidth=1.5, color="gray", alpha=0.7,
                  label="GT" if row == 0 else None)
        if sb is not None:
            ax_b.axhline(sb, color=color, linewidth=0.8, linestyle=":", alpha=0.4)
        ax_b.set_ylabel(f"{name} (base)", fontsize=9, fontweight="bold")
        ax_b.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax_b.set_title("ROBOT (BASE) FRAME", fontsize=11, fontweight="bold")
            ax_b.legend(fontsize=8, loc="upper right")
        if row < len(dims) - 1:
            plt.setp(ax_b.get_xticklabels(), visible=False)
        else:
            ax_b.set_xlabel("Time (s) — 20Hz", fontsize=9)

    cw = current_world
    cb = current_base
    fig.suptitle(
        f'Semi-Closed-Loop (DD) — Step {step}  |  MSE={mse:.6f}  L1={l1:.6f}\n'
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
        description="Semi-closed-loop eval: replay GT + compare model predictions")
    parser.add_argument("--session", type=str,
                        default=f"{DATA_BASE}/{DATA_SESSION}")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=8,
                        help="Number of GT actions to execute per chunk before pausing")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max eval steps (0=entire trajectory)")
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    # 1. Load session trajectory
    print(f"\nLoading session: {args.session}")
    poses_world, grippers = load_session(args.session)
    N = len(poses_world)
    print(f"  Trajectory length: {N} poses ({N - 1} actions)")

    # 2. Load model
    model = load_model(args.checkpoint)
    chunk_len = model.chunk_len
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")

    # 3. Inference kwargs
    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
    )

    # 4. Connect to robot
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

    # 5. Open camera
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.")

    # 6. Create output directory
    session_ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path("ur5n/dd/semi-viz") / f"session_{session_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    print(f"Output: {out_dir}")

    # 7. Determine eval range
    n_actions = args.n_actions
    start_t = args.start_step
    end_t = N - 1  # last step that has a next pose
    if args.max_steps > 0:
        end_t = min(end_t, start_t + args.max_steps)
    n_infer_steps = (end_t - start_t + n_actions - 1) // n_actions
    print(f"\nEval range: step {start_t} to {end_t - 1} ({end_t - start_t} steps)")
    print(f"  n_actions={n_actions} → ~{n_infer_steps} inference pauses")

    # 8. Move to initial pose
    print(f"\nMoving to initial pose (step {start_t})...")
    init_world = poses_world[start_t].tolist()
    init_gripper = grippers[start_t]
    go_to_pose(rtde_c, rtde_r, init_world, T_bw)
    grip_pos = int(init_gripper * 255)
    gripper_hw.move(grip_pos, 255, 150)
    current_gripper = init_gripper
    print(f"  World: [{init_world[0]:.4f}, {init_world[1]:.4f}, {init_world[2]:.4f}]")
    print(f"  Gripper: {'closed' if init_gripper > 0.5 else 'open'}")

    # 9. Save run config
    run_config = {
        "session": args.session,
        "checkpoint": args.checkpoint,
        "instruction": args.instruction,
        "arm": args.arm,
        "n_actions": n_actions,
        "start_step": start_t,
        "end_step": end_t,
        "chunk_len": chunk_len,
        "decode_temperature": args.decode_temperature,
        "choice_temperature": args.choice_temperature,
        "dataset_key": dataset_key,
        "trajectory_length": N,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    # 10. Print config
    print(f"\n{'=' * 60}")
    print(f"  Semi-Closed-Loop Evaluation (Discrete Diffusion)")
    print(f"  Session:     {DATA_SESSION}")
    print(f"  Instruction: \"{args.instruction}\"")
    print(f"  Eval range:  step {start_t} → {end_t - 1}")
    print(f"  Chunk len:   {chunk_len}")
    print(f"  n_actions:   {n_actions} (GT steps per chunk at 20Hz)")
    print(f"  Mode:        observe → infer → execute {n_actions} GT @ 20Hz → repeat")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Eval loop ────────────────────────────────────────────────────
    # Flow: at each chunk boundary t:
    #   1. Observe (camera + robot state)
    #   2. Infer (model prediction)
    #   3. Compare with GT chunk
    #   4. Execute n_actions GT steps at 20Hz (smooth, real speed)
    #   5. Advance t by n_actions, repeat

    all_metrics = []
    step_log = []
    infer_count = 0

    try:
        t = start_t
        while t < end_t:
            loop_t0 = time.monotonic()

            # ── 1. Observe ───────────────────────────────────────────
            pose_base = rtde_r.getActualTCPPose()
            pose_world_actual = base_to_world(pose_base, T_bw)
            current_pos = np.array(pose_world_actual, dtype=np.float64)

            pil_img = cam.grab_pil()
            if pil_img is None:
                print(f"[WARN] Camera frame dropped at step {t}, retrying...")
                pil_img = cam.grab_pil()
                if pil_img is None:
                    continue

            img_path = out_dir / "images" / f"step_{t:04d}.jpg"
            pil_img.save(str(img_path), quality=90)

            # ── 2. Infer ─────────────────────────────────────────────
            example = {"image": [pil_img], "lang": args.instruction}
            t_infer = time.monotonic()
            output = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.monotonic() - t_infer) * 1000

            pred_normalized = output["normalized_actions"][0].astype(np.float32)
            pred_10d = baseframework.unnormalize_actions(pred_normalized, action_stats)

            # ── 3. Compute GT chunk & compare ────────────────────────
            gt_10d, valid_len = compute_gt_chunk(poses_world, grippers, t, chunk_len)
            metrics = compute_metrics(pred_10d, gt_10d, valid_len)

            # ── 4. Build trajectories for visualization ──────────────
            pred_7d = actions_10d_to_7d(pred_10d)
            pred_traj_w = accumulate_deltas(pose_world_actual, pred_7d)

            gt_7d = actions_10d_to_7d(gt_10d)
            gt_traj_w = accumulate_deltas(pose_world_actual, gt_7d)

            pred_traj_b = pred_traj_w.copy()
            pred_traj_b[:, 0] -= T_bw[0, 3]
            pred_traj_b[:, 1] -= T_bw[1, 3]
            pred_traj_b[:, 2] -= T_bw[2, 3]

            gt_traj_b = gt_traj_w.copy()
            gt_traj_b[:, 0] -= T_bw[0, 3]
            gt_traj_b[:, 1] -= T_bw[1, 3]
            gt_traj_b[:, 2] -= T_bw[2, 3]

            viz_path = out_dir / "images" / f"step_{t:04d}_viz.png"
            visualize_comparison(
                pred_traj_w, gt_traj_w, pred_traj_b, gt_traj_b,
                pil_img, pose_world_actual, list(pose_base),
                t, str(viz_path), args.instruction,
                metrics["mse"], metrics["l1"])

            # ── 5. Log ───────────────────────────────────────────────
            all_metrics.append(metrics)
            step_log.append({
                "step": t,
                "wall_time": time.time(),
                "ee_pose_world": list(pose_world_actual),
                "ee_pose_base": list(pose_base),
                "gt_pose_world": poses_world[t].tolist(),
                "gripper": current_gripper,
                "infer_ms": round(infer_ms, 1),
                "mse": metrics["mse"],
                "l1": metrics["l1"],
                "valid_len": valid_len,
                "image_file": f"images/step_{t:04d}.jpg",
                "viz_file": f"images/step_{t:04d}_viz.png",
            })

            # ── 6. Execute n_actions GT steps at 20Hz ────────────────
            n_exec = min(n_actions, end_t - t)

            # Build waypoints for all n_exec steps
            waypoints = np.zeros((n_exec, 6), dtype=np.float64)
            for i in range(n_exec):
                waypoints[i] = poses_world[t + 1 + i, :6]

            # Interpolate all waypoints at once → smooth 100Hz execution
            interp = interpolate_waypoints(current_pos, waypoints, INTERP_MULT)
            servo_dt = 1.0 / SERVO_HZ

            t_exec_start = time.monotonic()
            for i, pose_w in enumerate(interp):
                target_base = world_to_base(pose_w.tolist(), T_bw)
                rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
                precise_wait(t_exec_start + (i + 1) * servo_dt)
            rtde_c.servoStop()
            exec_ms = (time.monotonic() - t_exec_start) * 1000

            # Handle gripper transitions during this chunk
            for i in range(n_exec):
                new_gripper = grippers[t + 1 + i]
                if (new_gripper > 0.5) != (current_gripper > 0.5):
                    grip_pos = int(new_gripper * 255)
                    label = "CLOSE" if new_gripper > 0.5 else "OPEN"
                    print(f"  Gripper -> {label} (pos={grip_pos})")
                    gripper_hw.move(grip_pos, 255, 150)
                    current_gripper = new_gripper

            total_ms = (time.monotonic() - loop_t0) * 1000
            print(f"[step {t:4d}]  infer={infer_ms:5.0f}ms  exec={exec_ms:5.0f}ms  "
                  f"total={total_ms:5.0f}ms  "
                  f"MSE={metrics['mse']:.6f}  L1={metrics['l1']:.6f}  "
                  f"valid={valid_len}/{chunk_len}  n_exec={n_exec}  "
                  f"pos=[{pose_world_actual[0]:.3f}, {pose_world_actual[1]:.3f}, "
                  f"{pose_world_actual[2]:.3f}]  "
                  f"grip={'C' if current_gripper > 0.5 else 'O'}")

            infer_count += 1
            t += n_exec

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

        # ── Save results ─────────────────────────────────────────────
        if step_log:
            with open(out_dir / "step_log.json", "w") as f:
                json.dump(step_log, f, indent=2)

            # Aggregate metrics
            agg = {
                "num_steps": infer_count,
                "overall_mse": float(np.mean([m["mse"] for m in all_metrics])),
                "overall_l1": float(np.mean([m["l1"] for m in all_metrics])),
                "overall_mse_std": float(np.std([m["mse"] for m in all_metrics])),
                "overall_l1_std": float(np.std([m["l1"] for m in all_metrics])),
            }

            # Per-dimension aggregate
            per_dim_mse = {}
            per_dim_l1 = {}
            for dim in DIM_LABELS_10D:
                per_dim_mse[dim] = float(np.mean(
                    [m["per_dim_mse"][dim] for m in all_metrics]))
                per_dim_l1[dim] = float(np.mean(
                    [m["per_dim_l1"][dim] for m in all_metrics]))
            agg["per_dim_mse"] = per_dim_mse
            agg["per_dim_l1"] = per_dim_l1

            # Per-step-in-chunk aggregate
            max_valid = max(m["valid_len"] for m in all_metrics)
            per_chunk_step_mse = []
            for k in range(max_valid):
                vals = [m["per_step_mse"][k] for m in all_metrics
                        if k < len(m["per_step_mse"])]
                per_chunk_step_mse.append(float(np.mean(vals)) if vals else 0.0)
            agg["per_chunk_step_mse"] = per_chunk_step_mse

            with open(out_dir / "aggregate_metrics.json", "w") as f:
                json.dump(agg, f, indent=2)

            # Print summary
            print(f"\n{'=' * 60}")
            print(f"  Semi-Closed-Loop Eval Results ({infer_count} steps)")
            print(f"{'=' * 60}")
            print(f"  Overall MSE: {agg['overall_mse']:.6f} "
                  f"(+/- {agg['overall_mse_std']:.6f})")
            print(f"  Overall L1:  {agg['overall_l1']:.6f} "
                  f"(+/- {agg['overall_l1_std']:.6f})")
            print(f"\n  Per-Dimension MSE / L1:")
            for dim in DIM_LABELS_10D:
                print(f"    {dim:12s}  MSE={per_dim_mse[dim]:.6f}  "
                      f"L1={per_dim_l1[dim]:.6f}")
            print(f"\n  Per-Chunk-Step MSE:")
            if per_chunk_step_mse:
                max_v = max(per_chunk_step_mse) or 1.0
                for k, v in enumerate(per_chunk_step_mse):
                    bar = "#" * int(v / max_v * 30)
                    print(f"    step {k:2d}: {v:.6f}  {bar}")
            print(f"{'=' * 60}")
            print(f"Results saved to {out_dir}")

        print(f"Done. Evaluated {infer_count} steps.")


if __name__ == "__main__":
    main()
