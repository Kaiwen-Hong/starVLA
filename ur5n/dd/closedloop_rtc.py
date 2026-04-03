#!/usr/bin/env python3
"""
RTC (Real-Time Chunking) closed-loop control with discrete diffusion.

Architecture (borrowed from ur5/scripts/servo_v2 + inferencer_v2 patterns):

  ServoRunner thread:  100Hz continuous servoL from buffer, action_t counter,
                        tail_pose for seamless splicing, separate gripper thread.
  Inferencer thread:   Cadence-aware self-loop — waits for n_actions boundary,
                        builds RTC prefix (shift consumed actions), calls
                        predict_action_realtime, pushes free actions to servo.
  Main thread:         Logging, gripper trick post-processing, early-stop.

vs closedloop_sync.py:
  - Inference overlaps with execution (hides latency)
  - No servoStop between chunks (gap-free motion)
  - RTC prefix conditioning for temporal consistency

Robot WILL move. Use Ctrl+C to stop.

Usage:
    python ur5n/dd/closedloop_rtc.py
    python ur5n/dd/closedloop_rtc.py --n_actions 8 --inference_delay 8
    python ur5n/dd/closedloop_rtc.py --instruction "pick up the block"
"""

import sys
import os
import time
import json
import argparse
import collections
import threading
import queue
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation as Rot
from PIL import Image
import cv2
import torch

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

# Safety clamp bounds in world frame (meters).
Y_MIN_WORLD = -0.50
Z_MIN_WORLD = 0.03
Z_MAX_WORLD = 0.30

# ── Robot config ─────────────────────────────────────────────────────
_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}

# ── SLAM → gripper rotation correction ──────────────────────────────

def slam_to_gripper_rotation(rotvec):
    T_inv = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    R = Rot.from_rotvec(rotvec).as_matrix() @ T_inv
    return Rot.from_matrix(R).as_rotvec()

_HOME_SLAM = {
    'left': [0.155666, -0.428459, 0.185173, 2.130869, 0.107971, -2.304919, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}
HOME_POSES_WORLD = {}
for _arm, _pose in _HOME_SLAM.items():
    _rot_corrected = slam_to_gripper_rotation(np.array(_pose[3:6]))
    HOME_POSES_WORLD[_arm] = _pose[:3] + _rot_corrected.tolist() + [_pose[6]]


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
#  Waypoint computation (world-frame delta addition + safety clamps)
# ═══════════════════════════════════════════════════════════════════

def compute_waypoints(start_pose_world, actions_10d, n_exec, fix_rotation,
                      gripper_state, args):
    """Convert 10D actions to 6D world-frame waypoints with safety clamps.

    Gripper trick decisions are embedded here so they happen per-waypoint.

    Args:
        start_pose_world: (6,) array [x,y,z,rx,ry,rz] — starting pose.
            In RTC mode this should be the servo tail_pose (end of buffer)
            so new waypoints splice seamlessly onto pending motion.
        actions_10d: (T, 10) denormalized actions.
        n_exec: number of actions to convert.
        fix_rotation: zero out rotation deltas.
        gripper_state: dict with mutable state:
            'current': float (0=open, 1=closed)
            'chunks_since_grasp': int or None
            'task_finished': bool
        args: parsed CLI args (for gripper thresholds).

    Returns:
        waypoints: (n_exec, 6) world-frame poses
        gripper_cmds: list of (index, float_value) for transitions to execute
        n_exec: possibly trimmed if task_finished early
    """
    waypoints = np.zeros((n_exec, 6), dtype=np.float64)
    gripper_cmds = []

    pos = np.array(start_pose_world[:3], dtype=np.float64)
    rot = np.array(start_pose_world[3:6], dtype=np.float64)

    current_gripper = gripper_state['current']
    actual_n = n_exec

    for i in range(n_exec):
        delta = action_10d_to_delta7d(actions_10d[i])

        if fix_rotation:
            delta[3:6] = 0.0

        pos = pos + delta[:3]
        if not fix_rotation:
            rot = rot + delta[3:6]

        # Safety clamps (world frame)
        pos[1] = max(pos[1], Y_MIN_WORLD)
        pos[2] = np.clip(pos[2], Z_MIN_WORLD, Z_MAX_WORLD)
        # Board zone: obstacles in y ∈ [-0.4276, -0.2931], enforce z > 0.125
        if -0.4276 <= pos[1] <= -0.2931:
            pos[2] = max(pos[2], 0.125)

        waypoints[i, :3] = pos
        waypoints[i, 3:6] = rot

        # ── Gripper decision ────────────────────────────────────────
        new_gripper = float(delta[6])

        reached_target = (args.if_release_when_reach_temp
                          and current_gripper > 0.5
                          and pos[1] >= args.release_y_threshold)

        if reached_target and new_gripper <= 0.5:
            # At target zone, policy says open → release and finish
            gripper_cmds.append((i, 0.0))
            current_gripper = 0.0
            actual_n = i + 1
            gripper_state['current'] = current_gripper
            gripper_state['task_finished'] = True
            break
        elif (args.if_grasped_not_release
              and current_gripper > 0.5
              and new_gripper <= 0.5):
            pass  # keep closed, not at target yet
        elif (new_gripper > 0.5) != (current_gripper > 0.5):
            # [MOD] grasp_trick z-check moved to ServoRunner._gripper_loop
            # so it uses actual robot z instead of planned waypoint z.
            # REVERT: uncomment the block below and remove the check in
            # ServoRunner._gripper_loop to restore original behavior.
            # if (args.if_grasp_trick
            #         and new_gripper > 0.5
            #         and pos[2] >= args.grasp_z_threshold):
            #     continue  # too high, skip close command
            gripper_cmds.append((i, new_gripper))
            current_gripper = new_gripper
            # Start post-grasp counter on first close
            if (args.if_grasp_delay_temp_solution
                    and new_gripper > 0.5
                    and gripper_state['chunks_since_grasp'] is None):
                gripper_state['chunks_since_grasp'] = 0

    gripper_state['current'] = current_gripper
    waypoints = waypoints[:actual_n]
    return waypoints, gripper_cmds, actual_n


# ═══════════════════════════════════════════════════════════════════
#  ServoRunner — continuous 100Hz servo (from servo_v2 pattern)
# ═══════════════════════════════════════════════════════════════════

class ServoRunner:
    """100Hz servo with non-blocking gripper, tail_pose, and timing guard."""

    def __init__(self, rtde_c, T_bw, gripper_hw, rtde_r=None,
                 grasp_trick=False, grasp_z_threshold=0.036,
                 gripper_state=None):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._gripper_hw = gripper_hw
        # [MOD] grasp_trick moved from compute_waypoints to _gripper_loop
        # so the z check uses the actual robot pose at execution time,
        # not the planned waypoint z during inference. To revert: remove
        # rtde_r/grasp_trick/grasp_z_threshold/gripper_state here and in
        # _gripper_loop, and restore the check in compute_waypoints
        # (search "REVERT").
        self._rtde_r = rtde_r
        self._grasp_trick = grasp_trick
        self._grasp_z_threshold = grasp_z_threshold
        self._gripper_state = gripper_state  # shared dict, for state sync
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pose = None
        self._starve_count = 0
        # 20Hz action counter
        self._action_t = 0
        self._sub_step = 0
        self._action_cond = threading.Condition()
        # Gripper thread
        self._pending_grip = None
        self._grip_lock = threading.Lock()
        self._grip_event = threading.Event()
        self._grip_thread = None

    def start(self, initial_pose_world):
        self._last_pose = np.array(initial_pose_world[:6], dtype=np.float64)
        self._running = True
        self._action_t = 0
        self._sub_step = 0
        self._starve_count = 0
        self._grip_thread = threading.Thread(target=self._gripper_loop, daemon=True)
        self._grip_thread.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._action_cond:
            self._action_cond.notify_all()
        self._grip_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._grip_thread:
            self._grip_thread.join(timeout=2.0)
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass

    def push_waypoints(self, start_pose, waypoints_20hz, gripper_cmds=None):
        """Interpolate 20Hz waypoints to 100Hz and append to buffer."""
        interp = interpolate_waypoints(start_pose, waypoints_20hz, INTERP_MULT)
        with self._lock:
            self._buffer.extend(interp)

        if gripper_cmds:
            for _, new_val in gripper_cmds:
                grip_pos = int(new_val * 255)
                label = "CLOSE" if grip_pos > 127 else "OPEN"
                with self._grip_lock:
                    self._pending_grip = (grip_pos, label)
                self._grip_event.set()

    @property
    def buffer_len(self):
        with self._lock:
            return len(self._buffer)

    @property
    def last_pose(self):
        return self._last_pose.copy() if self._last_pose is not None else None

    @property
    def tail_pose(self):
        """Last pose in buffer (future position) for seamless waypoint splicing."""
        with self._lock:
            if self._buffer:
                return np.array(self._buffer[-1], dtype=np.float64)
        return self._last_pose.copy() if self._last_pose is not None else None

    @property
    def action_t(self):
        return self._action_t

    def wait_for_action_t(self, target_t, timeout=30.0):
        with self._action_cond:
            while self._action_t < target_t and self._running:
                if not self._action_cond.wait(timeout=timeout):
                    return False
            return self._action_t >= target_t

    @property
    def starve_count(self):
        return self._starve_count

    def _loop(self):
        servo_dt = 1.0 / SERVO_HZ
        t_next = time.monotonic() + servo_dt

        while self._running:
            pose = None
            with self._lock:
                if self._buffer:
                    pose = self._buffer.popleft()
                else:
                    self._starve_count += 1

            if pose is not None:
                self._last_pose = pose
                self._sub_step += 1
                if self._sub_step >= INTERP_MULT:
                    self._sub_step = 0
                    with self._action_cond:
                        self._action_t += 1
                        self._action_cond.notify_all()

            if self._last_pose is not None:
                # [TEMP] Execution-time safety clamp in world frame.
                # The primary clamp is in compute_waypoints (planning time),
                # but async RTC can let unclamped poses slip through because
                # waypoints are planned from tail_pose which may diverge from
                # actual robot state. This is a redundant guard.
                # To remove: delete this block; compute_waypoints clamp is
                # the canonical source.
                p = self._last_pose
                p[1] = max(p[1], Y_MIN_WORLD)
                p[2] = np.clip(p[2], Z_MIN_WORLD, Z_MAX_WORLD)
                if -0.4276 <= p[1] <= -0.2931:
                    p[2] = max(p[2], 0.125)

                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt

            # Timing guard: reset if fell behind to prevent burst catch-up
            now = time.monotonic()
            if t_next < now - servo_dt:
                t_next = now + servo_dt

    def _gripper_loop(self):
        last_pos = None
        while self._running:
            self._grip_event.wait(timeout=1.0)
            self._grip_event.clear()
            if not self._running:
                break

            with self._grip_lock:
                cmd = self._pending_grip
                self._pending_grip = None

            if cmd is not None and cmd[0] != last_pos:
                grip_pos, label = cmd

                # [MOD] grasp_trick: check ACTUAL robot z before closing.
                # Previously this was in compute_waypoints using planned z.
                # Now we read the real RTDE pose at execution time.
                # To REVERT: remove this block and restore the check in
                # compute_waypoints (search "REVERT" in that function).
                if (self._grasp_trick
                        and grip_pos > 127
                        and self._rtde_r is not None):
                    actual_base = self._rtde_r.getActualTCPPose()
                    actual_world = base_to_world(actual_base, self._T_bw)
                    actual_z = actual_world[2]
                    if actual_z >= self._grasp_z_threshold:
                        print(f"  [GRASP_TRICK] Skipping CLOSE: "
                              f"actual z={actual_z:.4f} >= "
                              f"threshold={self._grasp_z_threshold:.4f}")
                        # [MOD] Reset shared state so compute_waypoints
                        # can issue CLOSE again on the next cycle.
                        if self._gripper_state is not None:
                            self._gripper_state['current'] = 0.0
                            self._gripper_state['chunks_since_grasp'] = None
                        continue

                print(f"  Gripper -> {label} (pos={grip_pos})")
                self._gripper_hw.move(grip_pos, 255, 150)
                last_pos = grip_pos


# ═══════════════════════════════════════════════════════════════════
#  Inferencer — cadence-aware RTC inference (from inferencer_v2 pattern)
# ═══════════════════════════════════════════════════════════════════

class InferenceResult:
    __slots__ = ("normalized", "actions_10d", "camera_image",
                 "obs_ms", "infer_ms", "actual_delay", "n_pushed",
                 "waypoints")

    def __init__(self, normalized, actions_10d, camera_image,
                 obs_ms, infer_ms, actual_delay, n_pushed, waypoints):
        self.normalized = normalized
        self.actions_10d = actions_10d
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.n_pushed = n_pushed
        self.waypoints = waypoints


class Inferencer:
    """Self-looping cadence-aware RTC inference worker."""

    def __init__(self, model, cam, servo, n_actions, inference_delay,
                 instruction, infer_kwargs, action_stats, fix_rotation,
                 gripper_state, args):
        self._model = model
        self._cam = cam
        self._servo = servo
        self._n_actions = n_actions
        self._inference_delay = inference_delay
        self._instruction = instruction
        self._infer_kwargs = infer_kwargs
        self._action_stats = action_stats
        self._fix_rotation = fix_rotation
        self._gripper_state = gripper_state
        self._args = args

        self._result_queue = queue.Queue()
        self._running = True
        self._chunk_start_t = 0

        self._current_normalized = None
        self._chunk_len = 0
        self._started = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start(self, initial_normalized, chunk_len):
        self._current_normalized = initial_normalized.copy()
        self._chunk_len = chunk_len
        self._started.set()

    def wait_result(self, timeout=30.0):
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        self._started.set()

    def _loop(self):
        self._started.wait()
        if not self._running:
            return

        current_normalized = self._current_normalized
        chunk_len = self._chunk_len
        first_cycle = True

        while self._running:
            # 1. Wait for cadence (skip on first cycle — fires immediately)
            if first_cycle:
                first_cycle = False
                actual_delay = 0
                prev_norm_batch = None
            else:
                trigger_t = self._chunk_start_t + self._n_actions
                self._servo.wait_for_action_t(trigger_t, timeout=30.0)
                if not self._running:
                    break

                # How many actions consumed since chunk started
                s = self._servo.action_t - self._chunk_start_t

                # Build RTC prefix: shift consumed actions out
                if s < chunk_len:
                    shifted = np.zeros_like(current_normalized)
                    remaining = chunk_len - s
                    shifted[:remaining] = current_normalized[s:]
                    actual_delay = min(self._inference_delay, remaining)
                    prev_norm_batch = shifted[np.newaxis, ...]
                else:
                    prev_norm_batch = None
                    actual_delay = 0

            self._chunk_start_t = self._servo.action_t

            # 2. Camera capture
            t0 = time.monotonic()
            pil_img = self._cam.grab_pil()
            while pil_img is None:
                if not self._running:
                    return
                pil_img = self._cam.grab_pil()
            obs_ms = (time.monotonic() - t0) * 1000

            # 3. Model inference
            example = {"image": [pil_img], "lang": self._instruction}

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            if prev_norm_batch is not None and actual_delay > 0:
                out = self._model.predict_action_realtime(
                    examples=[example],
                    prev_action_chunk_normalized=prev_norm_batch,
                    inference_delay=actual_delay,
                    **self._infer_kwargs,
                )
            else:
                out = self._model.predict_action(
                    examples=[example],
                    **self._infer_kwargs,
                )

            end_evt.record()
            end_evt.synchronize()
            infer_ms = start_evt.elapsed_time(end_evt)

            # 4. Denormalize
            new_normalized = out["normalized_actions"][0].astype(np.float32)
            new_actions_10d = baseframework.unnormalize_actions(
                new_normalized, self._action_stats)

            # 5. Compute waypoints from free actions (after delay) with gripper tricks
            free_actions = new_actions_10d[actual_delay:]
            n_push = min(self._n_actions, len(free_actions))
            free_actions = free_actions[:n_push]

            # Determine n_push accounting for post-grasp extended execution
            gs = self._gripper_state
            if (self._args.if_grasp_delay_temp_solution
                    and gs['chunks_since_grasp'] is not None
                    and gs['chunks_since_grasp'] < self._args.n_chunks_after_grasp):
                n_push_adj = min(self._args.n_actions_after_grasp,
                                 len(new_actions_10d) - actual_delay)
                free_actions = new_actions_10d[actual_delay:actual_delay + n_push_adj]
                n_push = len(free_actions)
                gs['chunks_since_grasp'] += 1

            start_pos = self._servo.tail_pose
            waypoints, gripper_cmds, actual_n = compute_waypoints(
                start_pos, free_actions, n_push, self._fix_rotation,
                self._gripper_state, self._args)

            self._servo.push_waypoints(start_pos, waypoints, gripper_cmds)

            # 6. New prediction becomes current chunk
            current_normalized = new_normalized

            # 7. Post result for main loop
            self._result_queue.put(
                InferenceResult(new_normalized, new_actions_10d, pil_img,
                                obs_ms, infer_ms, actual_delay, actual_n,
                                waypoints))


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def visualize_step(traj_world, traj_base, camera_image,
                   current_world, current_base, n_exec,
                   step, save_path, instruction):
    """Camera (left) + world trajectory (mid) + base trajectory (right)."""
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

    axes_w, axes_b = [], []
    for row, (dim_idx, name, color, sw, sb) in enumerate(dims):
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
        f'Closed-Loop RTC (DD) — Step {step}\n'
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
        description="RTC closed-loop control with discrete diffusion")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=8,
                        help="Number of actions to execute per inference cycle")
    parser.add_argument("--inference_delay", type=int, default=4,
                        help="RTC prefix length (default: 4)")
    parser.add_argument("--n_actions_after_grasp", type=int, default=16,
                        help="Actions per chunk for first N chunks after grasping")
    parser.add_argument("--n_chunks_after_grasp", type=int, default=2,
                        help="How many chunks to use n_actions_after_grasp")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps (0=unlimited, Ctrl+C to stop)")
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--fix_rotation", action="store_true", default=True)
    parser.add_argument("--no_fix_rotation", dest="fix_rotation",
                        action="store_false")
    parser.add_argument("--if_grasped_not_release", action="store_true",
                        default=True)
    parser.add_argument("--allow_release", dest="if_grasped_not_release",
                        action="store_false")
    parser.add_argument("--if_grasp_delay_temp_solution", action="store_true",
                        default=True)
    parser.add_argument("--no_grasp_delay", dest="if_grasp_delay_temp_solution",
                        action="store_false")
    parser.add_argument("--if_release_when_reach_temp", action="store_true",
                        default=True)
    parser.add_argument("--no_release_when_reach", dest="if_release_when_reach_temp",
                        action="store_false")
    parser.add_argument("--release_y_threshold", type=float, default=-0.318)
    parser.add_argument("--if_grasp_trick", action="store_true", default=True)
    parser.add_argument("--no_grasp_trick", dest="if_grasp_trick",
                        action="store_false")
    parser.add_argument("--grasp_z_threshold", type=float, default=0.04)
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--no_save_rollout", dest="save_rollout",
                        action="store_false")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    # ── Load model ───────────────────────────────────────────────────
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

    # ── Open camera ──────────────────────────────────────────────────
    print(f"Opening camera /dev/video{args.camera_dev}...")
    cam = RealCamera(dev=args.camera_dev)
    print("Warming up camera (2s)...")
    t_warm = time.monotonic() + 2.0
    while time.monotonic() < t_warm:
        cam.grab_rgb()
    print("Camera ready.")

    # ── Go home ──────────────────────────────────────────────────────
    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)

    # ── Shared gripper state (mutable, accessed by inferencer) ───────
    gripper_state = {
        'current': 0.0,
        'chunks_since_grasp': None,
        'task_finished': False,
    }

    # ── Rollout saving ───────────────────────────────────────────────
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = Path("ur5n/dd/rollouts") / f"rtc_{ts}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")

        run_config = {
            "mode": "rtc",
            "checkpoint": args.checkpoint,
            "instruction": args.instruction,
            "arm": args.arm,
            "n_actions": args.n_actions,
            "inference_delay": args.inference_delay,
            "max_steps": args.max_steps,
            "chunk_len": chunk_len,
            "control_hz": CONTROL_HZ,
            "servo_hz": SERVO_HZ,
            "y_min_world": Y_MIN_WORLD,
            "z_bounds_world": [Z_MIN_WORLD, Z_MAX_WORLD],
            "fix_rotation": args.fix_rotation,
            "decode_temperature": args.decode_temperature,
            "choice_temperature": args.choice_temperature,
            "dataset_key": dataset_key,
        }
        with open(rollout_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    # ── Print config ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Closed-Loop RTC (Discrete Diffusion)")
    print(f"  Arm:              {args.arm}")
    print(f"  Instruction:      \"{args.instruction}\"")
    print(f"  n_actions:        {args.n_actions}")
    print(f"  inference_delay:  {args.inference_delay}")
    print(f"  Chunk len:        {chunk_len}")
    print(f"  Servo:            {SERVO_HZ}Hz continuous (gap-free)")
    print(f"  Safety:           y > {Y_MIN_WORLD:.2f}, "
          f"z in [{Z_MIN_WORLD:.3f}, {Z_MAX_WORLD:.2f}]")
    print(f"  fix_rotation:     {args.fix_rotation}")
    print(f"  Max steps:        {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Save rollout:     {args.save_rollout}")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ═════════════════════════════════════════════════════════════════
    #  Step 0: synchronous initial inference (no prefix)
    # ═════════════════════════════════════════════════════════════════

    pose_base = rtde_r.getActualTCPPose()
    pose_world = base_to_world(pose_base, T_bw)
    current_pos = np.array(pose_world, dtype=np.float64)

    pil_img = cam.grab_pil()
    while pil_img is None:
        pil_img = cam.grab_pil()

    example = {"image": [pil_img], "lang": args.instruction}
    t_infer = time.monotonic()
    output = model.predict_action(examples=[example], **infer_kwargs)
    init_infer_ms = (time.monotonic() - t_infer) * 1000
    print(f"[init] inference={init_infer_ms:.0f}ms (sync, no prefix)")

    current_normalized = output["normalized_actions"][0].astype(np.float32)
    current_actions_10d = baseframework.unnormalize_actions(
        current_normalized, action_stats)

    # Push first n_actions into servo buffer
    n_init = min(args.n_actions, len(current_actions_10d))
    waypoints_init, grip_cmds_init, n_init = compute_waypoints(
        current_pos, current_actions_10d, n_init, args.fix_rotation,
        gripper_state, args)

    servo = ServoRunner(rtde_c, T_bw, gripper_hw, rtde_r=rtde_r,
                        grasp_trick=args.if_grasp_trick,
                        grasp_z_threshold=args.grasp_z_threshold,
                        gripper_state=gripper_state)
    servo.push_waypoints(current_pos, waypoints_init, grip_cmds_init)

    # ═════════════════════════════════════════════════════════════════
    #  Start inferencer (fires immediately) then start servo
    # ═════════════════════════════════════════════════════════════════

    inferencer = Inferencer(
        model, cam, servo, args.n_actions, args.inference_delay,
        args.instruction, infer_kwargs, action_stats, args.fix_rotation,
        gripper_state, args)
    inferencer.start(current_normalized, chunk_len)

    servo.start(current_pos)

    # ═════════════════════════════════════════════════════════════════
    #  Main loop: logging + early-stop (both daemon threads run autonomously)
    # ═════════════════════════════════════════════════════════════════

    infer_times = []
    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            result = inferencer.wait_result(timeout=60.0)
            if result is None:
                continue

            step += 1

            # ── Check task finished (set by compute_waypoints) ───────
            if gripper_state['task_finished']:
                print("\n>>> Task finished! Object released at target. <<<")
                break

            # ── Logging ──────────────────────────────────────────────
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            infer_times.append(result.infer_ms)

            buf_len = servo.buffer_len
            suffix = (f"  delay={result.actual_delay}  "
                      f"pushed={result.n_pushed}  buf={buf_len}")
            if servo.starve_count > 0:
                suffix += f"  starved={servo.starve_count}"

            print(f"[step {step:4d}]  "
                  f"obs={result.obs_ms:5.0f}ms  infer={result.infer_ms:5.0f}ms  "
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, "
                  f"{pose_world[2]:.3f}]  "
                  f"grip={'C' if gripper_state['current'] > 0.5 else 'O'}"
                  f"{suffix}")

            # ── Save rollout (optional) ──────────────────────────────
            if args.save_rollout and rollout_dir is not None:
                if result.camera_image is not None:
                    img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                    result.camera_image.save(str(img_path), quality=90)

                all_deltas_7d = actions_10d_to_7d(result.actions_10d)
                if args.fix_rotation:
                    all_deltas_7d[:, 3:6] = 0.0
                traj_world = accumulate_deltas(pose_world, all_deltas_7d)
                traj_world[:, 1] = np.maximum(traj_world[:, 1], Y_MIN_WORLD)
                traj_world[:, 2] = np.clip(traj_world[:, 2], Z_MIN_WORLD, Z_MAX_WORLD)
                in_board = ((traj_world[:, 1] >= -0.4276)
                            & (traj_world[:, 1] <= -0.2931))
                traj_world[in_board, 2] = np.maximum(
                    traj_world[in_board, 2], 0.125)

                traj_base = traj_world.copy()
                traj_base[:, 0] -= T_bw[0, 3]
                traj_base[:, 1] -= T_bw[1, 3]
                traj_base[:, 2] -= T_bw[2, 3]

                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_step(
                    traj_world, traj_base, result.camera_image,
                    pose_world, list(pose_base), result.n_pushed,
                    step, str(viz_path), args.instruction)

                rollout_log.append({
                    "step": step,
                    "wall_time": time.time(),
                    "ee_pose_world": list(pose_world),
                    "ee_pose_base": list(pose_base),
                    "gripper": gripper_state['current'],
                    "pred_actions_10d": result.actions_10d.tolist(),
                    "n_pushed": result.n_pushed,
                    "actual_delay": result.actual_delay,
                    "obs_ms": round(result.obs_ms, 1),
                    "infer_ms": round(result.infer_ms, 1),
                    "image_file": f"images/step_{step:04d}.jpg",
                    "viz_file": f"images/step_{step:04d}_viz.png",
                })

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        inferencer.stop()
        servo.stop()

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

        if args.save_rollout and rollout_dir is not None and rollout_log:
            with open(rollout_dir / "rollout.json", "w") as f:
                json.dump(rollout_log, f, indent=2)
            print(f"Rollout saved: {rollout_dir}")
            print(f"  {len(rollout_log)} steps")

        if infer_times:
            arr = np.array(infer_times)
            print(f"\nInference timing ({len(arr)} cycles):")
            print(f"  mean={arr.mean():.0f}ms  std={arr.std():.0f}ms  "
                  f"min={arr.min():.0f}ms  max={arr.max():.0f}ms  "
                  f"median={np.median(arr):.0f}ms")

        print(f"Done. Executed {step} inference steps.")


if __name__ == "__main__":
    main()
