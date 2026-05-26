#!/usr/bin/env python3
"""
Pool open-loop evaluation with live wrist (fisheye) camera + read-only
left UR5e. Robot does NOT move.

Interactive REPL:
  Press Enter → read TCP pose → grab fisheye frame → run inference →
  visualize predicted chunk_len-step world-frame trajectory from current
  pose.

Model: QwenPI flow-matching, action_dim=10 (Δpos_world + ΔR_world rot6d +
gripper_target), state_dim=10 (pos + rot6d + gripper). Cue is bolt-gripped
so gripper is constant 1.0; we pad the state's gripper field to 1.0.

Camera: head fisheye on /dev/video0 (XVisio vSLAM lens). The dataset's
'observation.images.wrist' field is actually this fisheye — naming was
inherited from the FastUMI LeRobot conversion.

Usage:
    # Terminal 1 (once the model exists):
    python ur5n/pool-preference/fm/inference_server.py --checkpoint <path>

    # Terminal 2:
    python ur5n/pool-preference/fm/live_openloop_eval.py
    python ur5n/pool-preference/fm/live_openloop_eval.py --task wall_bounce
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
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
from scipy.spatial.transform import Rotation as Rot

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

FPS = 20

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
#  Frame conversion + 10-D state/action helpers
# ═══════════════════════════════════════════════════════════════════

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]
    p[1] += T_bw[1, 3]
    p[2] += T_bw[2, 3]
    return p


# ═══════════════════════════════════════════════════════════════════
#  10-D rotation helpers — match V3-convert exactly (SO(3) composition).
#
# Training action is computed by V3 as:
#     delta_pos    = pos_next - pos_curr
#     R_delta      = R_next @ R_curr.T          # world-frame rotation delta
#     delta_rot6d  = mat_to_rot6d(R_delta)
#     gripper      = target_state[9]            # absolute target, NOT delta
#
# Forward roll-out (inverse):
#     pos_new      = pos_curr + delta_pos
#     R_new        = R_delta @ R_curr           # SO(3) composition
#
# Rotvec/euler addition is NOT a valid substitute (per-step error ~8e-4
# in rot6d compounding across the chunk). Always operate on rotation
# matrices via rot6d Gram-Schmidt.
# ═══════════════════════════════════════════════════════════════════

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


def rotvec_to_rot6d(rotvec):
    R = Rot.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    return mat_to_rot6d(R)


def build_state_10d(pose_world_6d, gripper=1.0):
    """Build dataset-style 10-D state from a 6-D world TCP pose
    [x,y,z, rx,ry,rz]. Gripper defaults to 1.0 (V3's binarized 'closed'
    sentinel for pool's bolt-clamped cue)."""
    pos = np.asarray(pose_world_6d[:3], dtype=np.float64)
    rot6d = rotvec_to_rot6d(pose_world_6d[3:6])
    return np.concatenate([pos, rot6d, [float(gripper)]]).astype(np.float32)


def states_10d_to_pos_euler(states_10d):
    T = states_10d.shape[0]
    pos = states_10d[:, :3].astype(np.float64)
    euler = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        euler[t] = rot6d_to_euler(states_10d[t, 3:9])
    return pos, euler


def accumulate_actions_to_states(start_state_10d, actions_10d):
    """Inverse of V3's compute_world_relative_action_10d."""
    pos = np.asarray(start_state_10d[:3], dtype=np.float64).copy()
    R = rot6d_to_mat(start_state_10d[3:9])
    T = actions_10d.shape[0]
    out = np.zeros((T, 10), dtype=np.float64)
    for t in range(T):
        pos = pos + actions_10d[t, :3]
        R_delta = rot6d_to_mat(actions_10d[t, 3:9])
        R = R_delta @ R                         # SO(3) composition
        out[t, :3] = pos
        out[t, 3:9] = mat_to_rot6d(R)
        out[t, 9] = actions_10d[t, 9]
    return out


# ═══════════════════════════════════════════════════════════════════
#  Model loading (only with --no_use_server)
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
        print("[FIX] Set image_size=[256,256] (match pool native)")

    print(f"Loaded in {time.time() - t0:.1f}s  chunk_len={model.chunk_len}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize(state_10d, pred_states_10d, pred_actions_10d, pil_img,
              save_path, task_label, instruction, run_idx, infer_ms,
              pose_world):
    state_pos = state_10d[:3]
    state_euler = rot6d_to_euler(state_10d[3:9])
    pred_pos, pred_euler = states_10d_to_pos_euler(pred_states_10d)
    T = pred_pos.shape[0]
    ts = np.arange(T) / FPS

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.32,
                  width_ratios=[1, 1.2, 1.2])

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(pil_img)
    ax_img.set_title("Fisheye (center-crop)", fontsize=11, fontweight="bold")
    ax_img.axis("off")

    ax_xy = fig.add_subplot(gs[1:, 0])
    ax_xy.plot(pred_pos[:, 0], pred_pos[:, 1], "s-", color="#e41a1c",
               ms=4, lw=1.8, label=f"Pred ({T})")
    ax_xy.plot(state_pos[0], state_pos[1], "^",
               color="green", ms=12, label="now")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY (predicted)", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=8)

    # Absolute pose vs time (yaw is the only varying rotation dim).
    traj_rows = [
        ('x (m)',     pred_pos[:, 0],                     state_pos[0]),
        ('y (m)',     pred_pos[:, 1],                     state_pos[1]),
        ('z (m)',     pred_pos[:, 2],                     state_pos[2]),
        ('yaw (deg)', np.rad2deg(pred_euler[:, 2]),
                       np.rad2deg(state_euler[2])),
        ('gripper',   pred_states_10d[:, 9],              None),
    ]
    for row, (name, prv, nowv) in enumerate(traj_rows):
        ax = fig.add_subplot(gs[row, 1])
        ax.plot(ts, prv, "s-", color="#e41a1c", ms=3, lw=1.5)
        if nowv is not None:
            ax.axhline(nowv, color="gray", ls=":", lw=1,
                       label=f"now={nowv:.4f}")
            ax.legend(fontsize=7)
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Predicted absolute pose", fontweight="bold")
        if row < len(traj_rows) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(f"t (s) — {FPS}Hz")

    # Per-step action deltas (yaw in deg/step, position in mm/step).
    def deltas_to_dyaw_deg(actions_10d):
        n = actions_10d.shape[0]
        out = np.zeros(n)
        for i in range(n):
            R_delta = rot6d_to_mat(actions_10d[i, 3:9])
            out[i] = np.rad2deg(Rot.from_matrix(R_delta).as_euler('xyz')[2])
        return out
    delta_rows = [
        ('dx (mm/step)',    pred_actions_10d[:, 0] * 1000),
        ('dy (mm/step)',    pred_actions_10d[:, 1] * 1000),
        ('dz (mm/step)',    pred_actions_10d[:, 2] * 1000),
        ('d_yaw (deg/step)', deltas_to_dyaw_deg(pred_actions_10d)),
        ('gripper_target',  pred_actions_10d[:, 9]),
    ]
    for row, (name, prv) in enumerate(delta_rows):
        ax = fig.add_subplot(gs[row, 2])
        ax.plot(ts, prv, "s-", color="#e41a1c", ms=3, lw=1.5)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_ylabel(name, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Predicted per-step action deltas", fontweight="bold")
        if row < len(delta_rows) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel(f"t (s) — {FPS}Hz")

    cw = pose_world
    fig.suptitle(
        f'Pool live open-loop  —  run {run_idx}  task={task_label}\n'
        f'"{instruction}"   |   infer={infer_ms:.0f}ms\n'
        f'World pose: [{cw[0]:.4f}, {cw[1]:.4f}, {cw[2]:.4f}, '
        f'{cw[3]:.4f}, {cw[4]:.4f}, {cw[5]:.4f}]   '
        f'yaw_now={np.rad2deg(state_euler[2]):+.2f}°',
        fontsize=11, fontweight="bold")
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pool live open-loop eval (QwenPI flow-matching, 10-D)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--task", choices=list(TASK_INSTRUCTIONS.keys()),
                        default="pocket",
                        help="Which task instruction to feed the policy "
                             "(default: pocket)")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction string (bypasses --task)")
    parser.add_argument("--include_state", action="store_true", default=False,
                        help="Pass the 10-D state to the model. Default False "
                             "since training likely had include_state=false "
                             "(mirroring airhockey).")
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
    task_label = args.task if args.instruction is None else "custom"

    T_bw = BASE_IN_WORLD[args.arm]

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
    print(f"chunk_len = {chunk_len}")

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
    out_dir = Path("ur5n/pool-preference/fm/runs_live") / f"session_{session_ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "frames").mkdir(exist_ok=True)
    (out_dir / "viz").mkdir(exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    session_log_path = out_dir / "session_log.jsonl"
    print(f"Session output: {out_dir}")
    print(f"  frames/   raw fisheye JPGs")
    print(f"  viz/      predicted-trajectory PNGs")
    print(f"  runs/     per-run JSON (pose, pred_10d, pred_traj, ...)")
    print(f"  session_log.jsonl   one summary line per Enter (grep-friendly)")

    with open(out_dir / "session_config.json", "w") as f:
        json.dump({
            "session_ts": session_ts,
            "checkpoint": args.checkpoint,
            "arm": args.arm,
            "camera_dev": args.camera_dev,
            "task": task_label,
            "task_idx": task_idx,
            "instruction": instruction,
            "include_state": args.include_state,
            "dataset_key": dataset_key,
            "chunk_len": chunk_len,
            "fps": FPS,
            "T_bw_translation": [float(T_bw[0, 3]), float(T_bw[1, 3]),
                                  float(T_bw[2, 3])],
        }, f, indent=2)

    infer_kwargs = dict()

    print("=" * 60)
    print("  Pool Live Open-Loop Eval (QwenPI, 10-D)")
    print(f"  Arm:           {args.arm}  IP={robot_ip}")
    print(f"  Camera:        /dev/video{args.camera_dev} (fisheye)")
    print(f"  Task:          {task_label} (task_idx={task_idx})")
    print(f"  Instruction:   \"{instruction}\"")
    print(f"  Include state: {args.include_state}")
    print(f"  fps={FPS}  chunk_len={chunk_len}  action_dim=10  state_dim=10")
    print("  Robot does NOT move. Press Enter to infer, 'q' to quit")
    print("=" * 60)

    run_idx = 0
    try:
        while True:
            line = input(f"\n[Run {run_idx}] Enter to infer (q to quit): ").strip()
            if line.lower() == 'q':
                break

            # Timestamps captured as close to the read as possible so they
            # can be cross-correlated against an external robot-moving
            # script's log.
            t_wall = time.time()
            t_mono = time.monotonic()
            iso = datetime.fromtimestamp(t_wall, tz=timezone.utc).strftime(
                "%Y%m%dT%H%M%S_%f")[:-3]  # ms precision
            stem = f"run_{run_idx:03d}_{iso}"

            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            t_pose_ms = (time.monotonic() - t_mono) * 1000
            state_10d = build_state_10d(pose_world, gripper=1.0)
            state_pos = state_10d[:3]
            state_euler = rot6d_to_euler(state_10d[3:9])
            print(f"  Pose world: [{pose_world[0]:.4f}, {pose_world[1]:.4f}, "
                  f"{pose_world[2]:.4f}, {pose_world[3]:.4f}, "
                  f"{pose_world[4]:.4f}, {pose_world[5]:.4f}]")
            print(f"  yaw_now = {np.rad2deg(state_euler[2]):+.3f}°  "
                  f"(roll/pitch ≈ {np.rad2deg(state_euler[0]):+.2f}°/"
                  f"{np.rad2deg(state_euler[1]):+.2f}°, "
                  f"should be near-constant ±180°/0°)")

            pil_img = cam.grab_pil()
            if pil_img is None:
                print("[WARN] no camera frame, retry")
                continue
            t_grab_ms = (time.monotonic() - t_mono) * 1000 - t_pose_ms
            print(f"  Captured image: {pil_img.size}")

            example = {"image": [pil_img], "lang": instruction}
            if args.include_state:
                example["state"] = state_10d[np.newaxis, :]
            t0 = time.monotonic()
            out = model.predict_action(examples=[example], **infer_kwargs)
            infer_ms = (time.monotonic() - t0) * 1000

            pred_norm = out["normalized_actions"][0].astype(np.float32)
            pred_10d = baseframework.unnormalize_actions(pred_norm, action_stats)
            pred_states_10d = accumulate_actions_to_states(state_10d, pred_10d)
            pred_pos, pred_euler = states_10d_to_pos_euler(pred_states_10d)

            displacement = float(np.linalg.norm(pred_pos[-1] - state_pos))
            d_yaw_total_deg = float(
                np.rad2deg(pred_euler[-1, 2] - state_euler[2]))
            print(f"  infer={infer_ms:.0f}ms")
            print(f"  action range: "
                  f"dx=[{pred_10d[:, 0].min()*1000:.2f},"
                  f"{pred_10d[:, 0].max()*1000:.2f}]mm  "
                  f"dy=[{pred_10d[:, 1].min()*1000:.2f},"
                  f"{pred_10d[:, 1].max()*1000:.2f}]mm  "
                  f"dz=[{pred_10d[:, 2].min()*1000:.2f},"
                  f"{pred_10d[:, 2].max()*1000:.2f}]mm")
            print(f"  Pred final pos: [{pred_pos[-1, 0]:.4f}, "
                  f"{pred_pos[-1, 1]:.4f}, {pred_pos[-1, 2]:.4f}]  "
                  f"displacement={displacement*1000:.1f} mm  "
                  f"Δyaw={d_yaw_total_deg:+.2f}°")

            # ── Save bundle for this run ─────────────────────────────
            frame_path = out_dir / "frames" / f"{stem}.jpg"
            viz_path = out_dir / "viz" / f"{stem}.png"
            run_json_path = out_dir / "runs" / f"{stem}.json"

            pil_img.save(str(frame_path), quality=92)
            visualize(state_10d, pred_states_10d, pred_10d, pil_img,
                      str(viz_path), task_label, instruction, run_idx,
                      infer_ms, pose_world)

            run_record = {
                "run_idx": run_idx,
                "t_wall": t_wall,
                "t_wall_iso_utc": iso,
                "task": task_label,
                "task_idx": task_idx,
                "instruction": instruction,
                "pose_base": list(pose_base),
                "pose_world": list(pose_world),
                "state_10d": state_10d.tolist(),
                "state_euler_rad": state_euler.tolist(),
                "pred_norm": pred_norm.tolist(),
                "pred_actions_10d": pred_10d.tolist(),
                "pred_states_10d": pred_states_10d.tolist(),
                "pred_pos": pred_pos.tolist(),
                "pred_euler_rad": pred_euler.tolist(),
                "displacement_mm": displacement * 1000.0,
                "delta_yaw_total_deg": d_yaw_total_deg,
                "infer_ms": round(infer_ms, 2),
                "t_read_pose_ms": round(t_pose_ms, 2),
                "t_grab_frame_ms": round(t_grab_ms, 2),
                "frame_file": f"frames/{stem}.jpg",
                "viz_file": f"viz/{stem}.png",
            }
            with open(run_json_path, "w") as f:
                json.dump(run_record, f, indent=2)

            # Grep-friendly summary line (no chunks, just key scalars)
            summary = {
                "run_idx": run_idx,
                "t_wall": t_wall,
                "t_wall_iso_utc": iso,
                "task": task_label,
                "pose_world_xyz_rxryrz": list(pose_world),
                "yaw_now_deg": round(float(np.rad2deg(state_euler[2])), 3),
                "pred_final_xyz": pred_pos[-1].tolist(),
                "pred_final_yaw_deg": round(
                    float(np.rad2deg(pred_euler[-1, 2])), 3),
                "displacement_mm": round(displacement * 1000.0, 2),
                "delta_yaw_total_deg": round(d_yaw_total_deg, 3),
                "infer_ms": round(infer_ms, 1),
            }
            with open(session_log_path, "a") as f:
                f.write(json.dumps(summary) + "\n")

            print(f"  Saved: frames/{stem}.jpg  viz/{stem}.png  "
                  f"runs/{stem}.json")
            print(f"  (appended to session_log.jsonl)")
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
