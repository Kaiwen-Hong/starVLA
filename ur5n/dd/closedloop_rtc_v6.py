#!/usr/bin/env python3
"""
RTC (Real-Time Chunking) closed-loop control with discrete diffusion — v6.

v6 adds an interactive episode loop: after each execution the program does
NOT exit.  Instead it waits for keyboard input:
  - Press Enter          → run the policy again immediately
  - Type 'h' + Enter     → move arm to home position, then prompt again
  - Press Ctrl+C (or 'q')→ full shutdown

Architecture:

  ServoRunner thread:  100Hz continuous servoL from buffer, action_t counter,
                        tail_pose for seamless splicing, separate gripper thread.
  Inferencer thread:   Uniform cycle — trigger when inference_delay actions
                        remain in servo.  prev_action_chunk is (chunk_len -
                        n_actions) actions.  Push output[inference_delay :
                        inference_delay + n_actions].  Next prev =
                        output[n_actions:].
  Main thread:         Logging, gripper trick post-processing, early-stop.

Robot WILL move. Use Ctrl+C to stop.

Usage:
    python ur5n/dd/closedloop_rtc_v6.py
    python ur5n/dd/closedloop_rtc_v6.py --n_actions 8 --inference_delay 4
    python ur5n/dd/closedloop_rtc_v6.py --instruction "pick up the block"
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

DEFAULT_CHECKPOINT = (
    "checkpoints/discreteRTC/fastumi_pickandplace_qwenDiscreteDiffusion_0409_0_pick_to_moved_filtered/"
    "checkpoints/steps_30000_pytorch_model.pt"
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
    'left': [0.25666, -0.428459, 0.185173, 2.130869, 0.107971, -2.304919, 1],
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
        if args.board_zone_y_min <= pos[1] <= args.board_zone_y_max:
            pos[2] = max(pos[2], args.board_zone_z_min)

        waypoints[i, :3] = pos
        waypoints[i, 3:6] = rot

        new_gripper = float(delta[6])

        reached_target = (args.if_release_when_reach_temp
                          and current_gripper > 0.5
                          and gripper_state['object_grasped']
                          and pos[1] >= args.release_y_threshold
                          and (args.tricks_release_z_constraint is None
                               or pos[2] <= args.tricks_release_z_constraint))

        if reached_target and new_gripper <= 0.5:
            gripper_cmds.append((i, 0.0))
            current_gripper = 0.0
            actual_n = i + 1
            gripper_state['current'] = current_gripper
            gripper_state['task_finished'] = True
            break
        elif (args.if_grasped_not_release
              and current_gripper > 0.5
              and gripper_state['object_grasped']
              and new_gripper <= 0.5):
            pass  # keep closed, not at target yet
        elif (new_gripper > 0.5) != (current_gripper > 0.5):
            gripper_cmds.append((i, new_gripper))
            current_gripper = new_gripper
            if (args.if_grasp_delay_temp_solution
                    and new_gripper > 0.5
                    and gripper_state['chunks_since_grasp'] is None):
                gripper_state['chunks_since_grasp'] = 0

    gripper_state['current'] = current_gripper
    waypoints = waypoints[:actual_n]
    return waypoints, gripper_cmds, actual_n


# ═══════════════════════════════════════════════════════════════════
#  ServoRunner — continuous 100Hz servo
# ═══════════════════════════════════════════════════════════════════

class ServoRunner:
    """100Hz servo with non-blocking gripper, tail_pose, and timing guard."""

    def __init__(self, rtde_c, T_bw, gripper_hw, rtde_r=None,
                 grasp_trick=False, grasp_z_threshold=0.036,
                 grasp_detect_threshold=200, gripper_state=None,
                 systematically_x_offset=0.0,
                 board_zone_y_min=-0.4276, board_zone_y_max=0.2931,
                 board_zone_z_min=0.11):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._gripper_hw = gripper_hw
        self._rtde_r = rtde_r
        self._grasp_trick = grasp_trick
        self._grasp_z_threshold = grasp_z_threshold
        self._grasp_detect_threshold = grasp_detect_threshold
        self._gripper_state = gripper_state
        self._x_offset = systematically_x_offset
        self._board_zone_y_min = board_zone_y_min
        self._board_zone_y_max = board_zone_y_max
        self._board_zone_z_min = board_zone_z_min
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pose = None
        self._starve_count = 0
        self._action_t = 0
        self._sub_step = 0
        self._action_cond = threading.Condition()
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
                p = self._last_pose
                p[1] = max(p[1], Y_MIN_WORLD)
                p[2] = np.clip(p[2], Z_MIN_WORLD, Z_MAX_WORLD)
                if self._board_zone_y_min <= p[1] <= self._board_zone_y_max:
                    p[2] = max(p[2], self._board_zone_z_min)

                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                target_base[0] += self._x_offset
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt

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
                        if self._gripper_state is not None:
                            self._gripper_state['current'] = 0.0
                            self._gripper_state['chunks_since_grasp'] = None
                        continue

                print(f"  Gripper -> {label} (pos={grip_pos})")
                self._gripper_hw.move(grip_pos, 255, 150)
                last_pos = grip_pos

                if grip_pos > 127 and self._gripper_state is not None:
                    actual_pos = self._gripper_hw.get_current_position()
                    if actual_pos < self._grasp_detect_threshold:
                        self._gripper_state['object_grasped'] = True
                        print(f"  [GRASP_DETECT] Object grasped "
                              f"(pos={actual_pos} < {self._grasp_detect_threshold})")
                    else:
                        self._gripper_state['object_grasped'] = False
                        self._gripper_state['current'] = 0.0
                        self._gripper_state['chunks_since_grasp'] = None
                        print(f"  [GRASP_DETECT] Empty grasp "
                              f"(pos={actual_pos} >= {self._grasp_detect_threshold}), "
                              f"re-opening gripper")
                        self._gripper_hw.move(0, 255, 150)
                        last_pos = 0
                elif grip_pos <= 127 and self._gripper_state is not None:
                    self._gripper_state['object_grasped'] = False


# ═══════════════════════════════════════════════════════════════════
#  Inferencer — cadence-aware RTC inference
# ═══════════════════════════════════════════════════════════════════

class InferenceResult:
    __slots__ = ("normalized", "actions_10d", "camera_image",
                 "obs_ms", "infer_ms", "actual_delay", "n_pushed",
                 "waypoints", "prev_norm", "prev_unnorm_10d")

    def __init__(self, normalized, actions_10d, camera_image,
                 obs_ms, infer_ms, actual_delay, n_pushed, waypoints,
                 prev_norm, prev_unnorm_10d):
        self.normalized = normalized
        self.actions_10d = actions_10d
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.n_pushed = n_pushed
        self.waypoints = waypoints
        self.prev_norm = prev_norm
        self.prev_unnorm_10d = prev_unnorm_10d


class Inferencer:
    def __init__(self, model, cam, servo, n_actions, inference_delay,
                 instruction, infer_kwargs, action_stats, fix_rotation,
                 gripper_state, args, chunk_len):
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
        self._chunk_len = chunk_len

        self._result_queue = queue.Queue()
        self._running = True

        self._current_normalized = None
        self._started = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start(self, initial_normalized):
        self._current_normalized = initial_normalized.copy()
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

        prev_action_chunk = self._current_normalized
        n_consumed = 0

        while self._running:
            if n_consumed > 0:
                self._servo.wait_for_action_t(n_consumed, timeout=30.0)
                if not self._running:
                    break

            t0 = time.monotonic()
            pil_img = self._cam.grab_pil()
            while pil_img is None:
                if not self._running:
                    return
                pil_img = self._cam.grab_pil()
            obs_ms = (time.monotonic() - t0) * 1000

            prev_norm_save = prev_action_chunk.copy()
            prev_unnorm_10d = baseframework.unnormalize_actions(
                prev_norm_save, self._action_stats)

            example = {"image": [pil_img], "lang": self._instruction}
            prev_norm_batch = prev_action_chunk[np.newaxis, ...]

            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()

            out = self._model.predict_action_realtime(
                examples=[example],
                prev_action_chunk_normalized=prev_norm_batch,
                inference_delay=self._inference_delay,
                **self._infer_kwargs,
            )

            end_evt.record()
            end_evt.synchronize()
            infer_ms = start_evt.elapsed_time(end_evt)

            new_normalized = out["normalized_actions"][0].astype(np.float32)
            new_actions_10d = baseframework.unnormalize_actions(
                new_normalized, self._action_stats)

            free_start = self._inference_delay
            free_end = free_start + self._n_actions
            free_actions = new_actions_10d[free_start:free_end]
            n_push = len(free_actions)

            gs = self._gripper_state
            if (self._args.if_grasp_delay_temp_solution
                    and gs['chunks_since_grasp'] is not None
                    and gs['chunks_since_grasp'] < self._args.n_chunks_after_grasp):
                n_push_adj = min(self._args.n_actions_after_grasp,
                                 len(new_actions_10d) - free_start)
                free_actions = new_actions_10d[free_start:free_start + n_push_adj]
                n_push = len(free_actions)
                gs['chunks_since_grasp'] += 1

            start_pos = self._servo.tail_pose
            waypoints, gripper_cmds, actual_n = compute_waypoints(
                start_pos, free_actions, n_push, self._fix_rotation,
                self._gripper_state, self._args)

            self._servo.push_waypoints(start_pos, waypoints, gripper_cmds)

            prev_action_chunk = new_normalized[self._n_actions:]
            n_consumed += self._n_actions

            self._result_queue.put(
                InferenceResult(new_normalized, new_actions_10d, pil_img,
                                obs_ms, infer_ms, self._inference_delay,
                                actual_n, waypoints, prev_norm_save,
                                prev_unnorm_10d))


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def compute_consistency_metrics(result, inference_delay):
    prev = result.prev_norm
    out = result.normalized[:len(prev)]
    D = inference_delay
    metrics = {}
    for nd, name in [(0, "dx"), (1, "dy"), (2, "dz"), (9, "grip")]:
        pv, ov = prev[:, nd], out[:, nd]
        metrics[f"{name}_fix"] = float(np.mean(np.abs(pv[:D] - ov[:D])))
        metrics[f"{name}_guide"] = float(np.mean(np.abs(pv[D:] - ov[D:])))
    return metrics


def visualize_rtc_debug(result, pose_world, step, save_path, instruction,
                        n_actions, inference_delay, chunk_len, fix_rotation):
    A = n_actions
    D = inference_delay
    L = chunk_len

    output_norm = result.normalized
    prev_norm = result.prev_norm
    output_10d = result.actions_10d

    deltas_7d = actions_10d_to_7d(output_10d)
    if fix_rotation:
        deltas_7d[:, 3:6] = 0.0
    traj = accumulate_deltas(pose_world, deltas_7d)

    fig = plt.figure(figsize=(24, 14))
    gs_fig = GridSpec(4, 3, figure=fig, hspace=0.28, wspace=0.30,
                      width_ratios=[1, 1.3, 1.3])

    ax_cam = fig.add_subplot(gs_fig[0:2, 0])
    ax_cam.imshow(result.camera_image)
    ax_cam.set_title("Camera", fontsize=11, fontweight="bold")
    ax_cam.axis("off")

    ax_info = fig.add_subplot(gs_fig[2:4, 0])
    ax_info.axis("off")
    info = (
        f"Step {step}\n"
        f"L={L}  A={A}  D={D}\n"
        f"obs={result.obs_ms:.0f}ms  infer={result.infer_ms:.0f}ms\n"
        f"pushed={result.n_pushed}\n\n"
        f"Pose (world):\n"
        f"  x={pose_world[0]:.4f}\n"
        f"  y={pose_world[1]:.4f}\n"
        f"  z={pose_world[2]:.4f}\n\n"
        f"\"{instruction[:60]}\""
    )
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                 fontsize=10, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    norm_dims = [(0, "dx"), (1, "dy"), (2, "dz"), (9, "grip")]
    traj_dims = [(0, "x"), (1, "y"), (2, "z"), (6, "grip")]
    colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    n_prev = len(prev_norm)

    for row, ((nd, nname), (td, tname), color) in enumerate(
            zip(norm_dims, traj_dims, colors)):

        ax_c = fig.add_subplot(gs_fig[row, 1])
        idx = np.arange(n_prev)
        pv = prev_norm[:, nd]
        ov = output_norm[:n_prev, nd]

        ax_c.axvspan(-0.5, D - 0.5, alpha=0.15, color="green")
        ax_c.axvspan(D - 0.5, n_prev - 0.5, alpha=0.10, color="gold")
        ax_c.plot(idx, pv, "o--", color="gray", ms=4, lw=1.5,
                  label="prev", alpha=0.8)
        ax_c.plot(idx, ov, "s-", color=color, ms=4, lw=2.0,
                  label="output")

        mae_fix = float(np.mean(np.abs(pv[:D] - ov[:D])))
        mae_guide = float(np.mean(np.abs(pv[D:] - ov[D:]))) if n_prev > D else 0.0

        ax_c.set_ylabel(nname, fontsize=10, fontweight="bold")
        if row == 0:
            ax_c.set_title(
                "CONSISTENCY  (prev vs output)\n"
                "green=hard-fixed   gold=soft-guided",
                fontsize=10, fontweight="bold")
            ax_c.legend(fontsize=8, loc="upper right")
        ax_c.text(0.5, 0.02,
                  f"fix MAE={mae_fix:.4f}  guide MAE={mae_guide:.4f}",
                  transform=ax_c.transAxes, fontsize=8, ha="center",
                  color="dimgray")
        ax_c.grid(True, alpha=0.3)
        if row < 3:
            plt.setp(ax_c.get_xticklabels(), visible=False)
        else:
            ax_c.set_xlabel("Action index", fontsize=9)

        ax_t = fig.add_subplot(gs_fig[row, 2])
        tidx = np.arange(L)
        vals = traj[:, td]

        ax_t.axvspan(-0.5, D - 0.5, alpha=0.12, color="lightgray",
                     label="prefix [0:D)")
        ax_t.axvspan(D - 0.5, D + A - 0.5, alpha=0.18, color="palegreen",
                     label="exec [D:D+A)")
        ax_t.axvspan(D + A - 0.5, L - 0.5, alpha=0.10, color="lightyellow",
                     label="future [D+A:L)")
        ax_t.plot(tidx, vals, "o-", color=color, ms=4, lw=2.0)

        if td < 6:
            ax_t.axhline(pose_world[td], color="black", lw=0.8, ls=":",
                         alpha=0.5, label=f"now={pose_world[td]:.4f}")
        ax_t.axvline(D, color="black", lw=0.5, ls="--", alpha=0.3)
        ax_t.axvline(D + A, color="black", lw=0.5, ls="--", alpha=0.3)

        ax_t.set_ylabel(f"{tname} (world)", fontsize=10, fontweight="bold")
        ax_t.grid(True, alpha=0.3)
        if row == 0:
            ax_t.set_title("PREDICTED TRAJECTORY (world)",
                           fontsize=10, fontweight="bold")
            ax_t.legend(fontsize=7, loc="upper right", ncol=2)
        if row < 3:
            plt.setp(ax_t.get_xticklabels(), visible=False)
        else:
            ax_t.set_xlabel("Action index", fontsize=9)

    fig.suptitle(f"RTC Debug — Step {step}",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def visualize_rtc_summary(consistency_log, infer_times, pose_log, save_path):
    steps = np.arange(1, len(consistency_log) + 1)
    dim_names = ["dx", "dy", "dz", "grip"]
    dim_colors = ["#e41a1c", "#377eb8", "#4daf4a", "#ff7f00"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("RTC Rollout Summary", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    for name, color in zip(dim_names, dim_colors):
        ax.plot(steps, [m[f"{name}_fix"] for m in consistency_log],
                "-", color=color, lw=1.5, label=name, alpha=0.8)
    ax.set_ylabel("MAE")
    ax.set_title("Consistency — Hard-Fixed [0:D]", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for name, color in zip(dim_names, dim_colors):
        ax.plot(steps, [m[f"{name}_guide"] for m in consistency_log],
                "-", color=color, lw=1.5, label=name, alpha=0.8)
    ax.set_ylabel("MAE")
    ax.set_title("Consistency — Soft-Guided [D:A]", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if pose_log:
        for d, name, color in [(0, "x", "#e41a1c"), (1, "y", "#377eb8"),
                                (2, "z", "#4daf4a")]:
            ax.plot(steps[:len(pose_log)], [p[d] for p in pose_log],
                    "-", lw=1.5, color=color, label=name, alpha=0.8)
    ax.set_ylabel("Position (world)")
    ax.set_title("Robot Position", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    arr = np.array(infer_times)
    ax.plot(steps[:len(arr)], arr, "o-", ms=2, lw=1, color="#984ea3")
    ax.axhline(arr.mean(), color="red", ls="--", lw=1, alpha=0.5,
               label=f"mean={arr.mean():.0f}ms")
    ax.set_ylabel("ms")
    ax.set_title("Inference Time", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    if pose_log:
        xs = [p[0] for p in pose_log]
        ys = [p[1] for p in pose_log]
        ax.plot(xs, ys, "o-", ms=3, lw=1.5, color="#377eb8", alpha=0.7)
        ax.plot(xs[0], ys[0], "^", ms=10, color="green", label="start")
        ax.plot(xs[-1], ys[-1], "v", ms=10, color="red", label="end")
        ax.set_xlabel("x (world)"); ax.set_ylabel("y (world)")
        ax.set_title("Top-Down Trajectory", fontweight="bold")
        ax.legend(fontsize=8); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    if pose_log:
        ax.plot(steps[:len(pose_log)], [p[2] for p in pose_log],
                "-", lw=1.5, color="#4daf4a", label="z")
    ax.set_xlabel("Step"); ax.set_ylabel("z (world)")
    ax.set_title("Z Position", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Single episode execution
# ═══════════════════════════════════════════════════════════════════

def run_episode(model, cam, rtde_c, rtde_r, gripper_hw, T_bw,
                args, infer_kwargs, action_stats, chunk_len, episode_num):
    """Run one closed-loop episode. Returns (infer_times, consistency_log, pose_log, step)."""

    print(f"\n{'─' * 60}")
    print(f"  Episode {episode_num}")
    print(f"{'─' * 60}")

    # Fresh gripper state for each episode
    gripper_state = {
        'current': 0.0,
        'object_grasped': False,
        'chunks_since_grasp': None,
        'task_finished': False,
    }

    # Rollout saving setup
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        ts = time.strftime("%Y%m%d_%H%M%S")
        rollout_dir = Path("ur5n/dd/rollouts") / f"rtc_{ts}_ep{episode_num}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")

        run_config = {
            "mode": "rtc",
            "episode": episode_num,
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
        }
        with open(rollout_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    # ── Step 0: synchronous initial inference (no prefix) ────────────
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

    n_init = min(args.inference_delay, len(current_actions_10d))
    waypoints_init, grip_cmds_init, n_init = compute_waypoints(
        current_pos, current_actions_10d, n_init, args.fix_rotation,
        gripper_state, args)

    servo = ServoRunner(rtde_c, T_bw, gripper_hw, rtde_r=rtde_r,
                        grasp_trick=args.if_grasp_trick,
                        grasp_z_threshold=args.grasp_z_threshold,
                        grasp_detect_threshold=args.grasp_detect_threshold,
                        gripper_state=gripper_state,
                        systematically_x_offset=args.systematically_x_offset,
                        board_zone_y_min=args.board_zone_y_min,
                        board_zone_y_max=args.board_zone_y_max,
                        board_zone_z_min=args.board_zone_z_min)
    servo.push_waypoints(current_pos, waypoints_init, grip_cmds_init)
    servo.start(current_pos)

    inferencer = Inferencer(
        model, cam, servo, args.n_actions, args.inference_delay,
        args.instruction, infer_kwargs, action_stats, args.fix_rotation,
        gripper_state, args, chunk_len=chunk_len)
    init_prev_chunk = current_normalized[:chunk_len - args.n_actions]
    inferencer.start(init_prev_chunk)

    # ── Main logging loop ─────────────────────────────────────────────
    infer_times = []
    consistency_log = []
    pose_log = []
    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            result = inferencer.wait_result(timeout=60.0)
            if result is None:
                continue

            step += 1

            if gripper_state['task_finished']:
                print("\n>>> Task finished! Object released at target. <<<")
                break

            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            infer_times.append(result.infer_ms)

            metrics = compute_consistency_metrics(result, args.inference_delay)
            consistency_log.append(metrics)
            pose_log.append(list(pose_world))

            buf_len = servo.buffer_len
            fix_mae = np.mean([metrics[f"{d}_fix"] for d in ["dx", "dy", "dz"]])
            guide_mae = np.mean([metrics[f"{d}_guide"] for d in ["dx", "dy", "dz"]])
            suffix = (f"  delay={result.actual_delay}  "
                      f"pushed={result.n_pushed}  buf={buf_len}  "
                      f"fix={fix_mae:.4f}  guide={guide_mae:.4f}")
            if servo.starve_count > 0:
                suffix += f"  starved={servo.starve_count}"

            print(f"[step {step:4d}]  "
                  f"obs={result.obs_ms:5.0f}ms  infer={result.infer_ms:5.0f}ms  "
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, "
                  f"{pose_world[2]:.3f}]  "
                  f"grip={'C' if gripper_state['current'] > 0.5 else 'O'}"
                  f"{suffix}")

            if args.save_rollout and rollout_dir is not None:
                if result.camera_image is not None:
                    img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                    result.camera_image.save(str(img_path), quality=90)

                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_rtc_debug(
                    result, pose_world, step, str(viz_path),
                    args.instruction, args.n_actions, args.inference_delay,
                    chunk_len, args.fix_rotation)

                log_entry = {
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
                    "consistency": metrics,
                    "image_file": f"images/step_{step:04d}.jpg",
                    "viz_file": f"images/step_{step:04d}_viz.png",
                }
                if result.prev_norm is not None:
                    log_entry["prev_norm"] = result.prev_norm.tolist()
                if result.prev_unnorm_10d is not None:
                    log_entry["prev_unnorm_10d"] = result.prev_unnorm_10d.tolist()
                rollout_log.append(log_entry)

    except KeyboardInterrupt:
        # Propagate so the outer loop can shut down cleanly
        inferencer.stop()
        servo.stop()
        try:
            rtde_c.servoStop()
        except Exception:
            pass
        try:
            rtde_c.stopScript()
        except Exception:
            pass
        raise

    # ── Episode teardown (normal finish) ─────────────────────────────
    inferencer.stop()
    servo.stop()
    try:
        rtde_c.servoStop()
    except Exception:
        pass
    try:
        rtde_c.stopScript()
    except Exception:
        pass

    # Save rollout
    if args.save_rollout and rollout_dir is not None and rollout_log:
        with open(rollout_dir / "rollout.json", "w") as f:
            json.dump(rollout_log, f, indent=2)
        print(f"Rollout saved: {rollout_dir}  ({len(rollout_log)} steps)")

        if consistency_log:
            visualize_rtc_summary(
                consistency_log, infer_times, pose_log,
                str(rollout_dir / "summary.png"))
            print(f"  Summary plot: {rollout_dir / 'summary.png'}")

    if infer_times:
        arr = np.array(infer_times)
        print(f"\nInference timing ({len(arr)} cycles):  "
              f"mean={arr.mean():.0f}ms  std={arr.std():.0f}ms  "
              f"min={arr.min():.0f}ms  max={arr.max():.0f}ms  "
              f"median={np.median(arr):.0f}ms")

    print(f"Episode {episode_num} done. Executed {step} inference steps.")
    return infer_times, consistency_log, pose_log, step


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RTC closed-loop control with discrete diffusion (v6 — interactive loop)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--n_actions", type=int, default=5)
    parser.add_argument("--inference_delay", type=int, default=5)
    parser.add_argument("--n_actions_after_grasp", type=int, default=16)
    parser.add_argument("--n_chunks_after_grasp", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps per episode (0=unlimited, Ctrl+C to stop)")
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--fixed_steps", action="store_true", default=False)
    parser.add_argument("--hard_mask", action="store_true", default=True) ##
    parser.add_argument("--early_stop", action="store_true", default=False) ##
    parser.add_argument("--fix_rotation", action="store_true", default=True)
    parser.add_argument("--no_fix_rotation", dest="fix_rotation",
                        action="store_false")
    parser.add_argument("--if_grasped_not_release", action="store_true",
                        default=True)
    parser.add_argument("--allow_release", dest="if_grasped_not_release",
                        action="store_false")
    parser.add_argument("--if_grasp_delay_temp_solution", action="store_true",
                        default=False)
    parser.add_argument("--no_grasp_delay", dest="if_grasp_delay_temp_solution",
                        action="store_false")
    parser.add_argument("--if_release_when_reach_temp", action="store_true",
                        default=True)
    parser.add_argument("--no_release_when_reach", dest="if_release_when_reach_temp",
                        action="store_false")
    parser.add_argument("--release_y_threshold", type=float, default=-0.318)
    parser.add_argument("--tricks_release_z_constraint", type=float, default=0.12,
                        help="Only release if z <= this value. Pass 'none' to disable.")
    parser.add_argument("--no_tricks_release_z_constraint",
                        dest="tricks_release_z_constraint",
                        action="store_const", const=None)
    parser.add_argument("--board_zone_y_min", type=float, default=-0.4276)
    parser.add_argument("--board_zone_y_max", type=float, default=0.2931)
    parser.add_argument("--board_zone_z_min", type=float, default=0.11)
    parser.add_argument("--grasp_detect_threshold", type=int, default=200)
    parser.add_argument("--if_grasp_trick", action="store_true", default=True)
    parser.add_argument("--no_grasp_trick", dest="if_grasp_trick",
                        action="store_false")
    parser.add_argument("--grasp_z_threshold", type=float, default=0.04)
    parser.add_argument("--systematically_x_offset", type=float, default=0.00)
    # Default is now False — use --save_rollout to enable
    parser.add_argument("--save_rollout", action="store_true", default=False)
    # ── Inference server (split from this script to avoid the ~30s
    # model-load cost on every iteration). Default ON: start
    # `python ur5n/dd/inference_server.py` in another terminal first,
    # then this script connects via Unix socket and proxies all
    # predict_action* calls to it. Pass --no_use_server for the legacy
    # in-process load.
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix "
                             "socket instead of loading the model in-process. "
                             "(default: True — start the server first)")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false",
                        help="Load the model in-process (legacy ~30s startup)")
    parser.add_argument("--server_socket", type=str,
                        default="/tmp/starvla_infer_dd.sock")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

    # ── Load model (in-process or remote via inference_server.py) ────
    if args.use_server:
        from remote_model import RemoteModel
        model = RemoteModel(args.server_socket)
        # The server owns the checkpoint path; reflect it in args so
        # rollout configs identify the actual model in use.
        args.checkpoint = model.checkpoint_path
    else:
        model = load_model(args.checkpoint)
    chunk_len = model.chunk_len
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', "
          f"modes={action_stats.get('norm_modes', 'legacy')}")

    assert args.n_actions + args.inference_delay <= chunk_len, (
        f"n_actions ({args.n_actions}) + inference_delay ({args.inference_delay}) "
        f"must be <= chunk_len ({chunk_len})")
    assert args.inference_delay <= args.n_actions, (
        f"inference_delay ({args.inference_delay}) must be <= n_actions ({args.n_actions})")

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
        execution_horizon=args.n_actions,
        fixed_steps=args.fixed_steps,
        hard_mask=args.hard_mask,
        early_stop=args.early_stop,
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
    print("Waiting for first frame from background reader...")
    while cam.grab_rgb() is None:
        time.sleep(0.05)
    print("Camera ready.")

    # ── Go home (once at startup) ────────────────────────────────────
    if not args.no_go_home:
        go_home(rtde_c, rtde_r, args.arm, T_bw, robot_ip)

    # ── Print config ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Closed-Loop RTC (Discrete Diffusion) — v6")
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
    print(f"\n  After each episode:")
    print(f"    Press Enter       → run policy again")
    print(f"    Type 'h' + Enter  → go to home position first")
    print(f"    Ctrl+C            → quit")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    try:
        run_episode(model, cam, rtde_c, rtde_r, gripper_hw, T_bw,
                    args, infer_kwargs, action_stats, chunk_len, episode_num=1)
    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    finally:
        print("Cleaning up...")
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
        print("Shutdown complete.")


if __name__ == "__main__":
    main()
