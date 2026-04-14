#!/usr/bin/env python3
"""
RTC (Real-Time Chunking) closed-loop control with flow matching (QwenPI) — v7,
pick-from-turntable variant (ur5n/2fm/).

Continuous counterpart to ur5n/2dd/closedloop_rtc_v7.py; only the model is
swapped from discrete diffusion to flow matching / QwenPI.

Task: pick an object off a ROTATING turntable and place it on a STATIC pan.

Differences from v6:
  - Single-shot: starts immediately, runs exactly one episode, then
    auto-exits. An episode finishes the moment the policy releases a
    grasped object (auto-detected inside the gripper thread — see
    _gripper_loop's OPEN path), or when max_steps is hit. No interactive
    prompts; re-launch the script for another attempt.
  - Dropped the mid-episode 'q' abort (episode runs to completion or
    max_steps).
  - Dropped rollout saving entirely — this script is for live operation,
    not data collection. No rollout dir, no debug plots, no JSON.

v7 flow:
  - At startup                    → goes straight to home, then runs ep 1
  - Episode finishes (release or
    max_steps)                    → cleanup and exit
  - Ctrl+C at any time            → full shutdown

Architecture (unchanged from v6):

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
    python ur5n/2fm/closedloop_rtc_v7.py
    python ur5n/2fm/closedloop_rtc_v7.py --n_actions 8 --inference_delay 4
    python ur5n/2fm/closedloop_rtc_v7.py --instruction "pick up the block"
"""

import sys
import os
import time
import argparse
import collections
import threading
import queue
from pathlib import Path

import numpy as np
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
    "results/Checkpoints/fastumi_pickandplace_qwenPI_0403_1_pick_from_moved/"
    "checkpoints/steps_30000_pytorch_model.pt"
)


DEFAULT_INSTRUCTION = "Pick up the purple block to the pan"

CONTROL_HZ = 20
INTERP_MULT = 5
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100Hz

# Safety clamp bounds in world frame (meters).
Y_MIN_WORLD = -0.50
Z_MIN_WORLD = 0.03
Z_MAX_WORLD = 0.30

# ── Turntable zone (task: pick from rotating turntable → place on static pan) ──
# Turntable occupies y in [TURNTABLE_Y_MIN, TURNTABLE_Y_MAX]. Three regimes:
#   1. Not yet grasped, over turntable:
#        z >= TURNTABLE_SURFACE_Z  (allow descent to reach object on turntable)
#   2. Ever-grasped-in-turntable (sticky), still over turntable:
#        z >= TURNTABLE_LIFT_Z     (lift above turntable obstacles, kept lifted)
#   3. Off turntable (y < TURNTABLE_Y_MIN, heading to static pan):
#        only the global Y_MIN_WORLD / Z_MIN_WORLD clamps apply.
#
# The "sticky" latch lives in gripper_state['has_ever_grasped_in_turntable']
# and is reset only when the EE physically leaves the turntable y-range.
# See compute_waypoints() and ServoRunner._loop() for how it is consulted,
# and ServoRunner._gripper_loop() for where it is set on successful grasp.
TURNTABLE_Y_MIN = -0.4276
TURNTABLE_Y_MAX = 0.2931
TURNTABLE_SURFACE_Z = 0.101398
TURNTABLE_LIFT_Z = 0.125

# Snap-grasp z. When the policy predicts "close gripper" AND the EE is
# physically over the turntable, the ServoRunner pauses the servo stream,
# does a blocking moveL straight down to this z (keeping current xy), closes
# the gripper, waits for the fingers to settle, then resumes servo from the
# new position. This fixes a timing bug where gripper_hw.move() would fire
# asynchronously during a streaming servoL — the robot kept moving (often
# rising into the lift portion of the predicted chunk) while the fingers
# were closing, so the grasp happened at an unpredictable z above the
# intended grasp point. Hardcoding the z eliminates that race.
#
# 0.102 sits just above TURNTABLE_SURFACE_Z (0.101398) so the fingers close
# flush against the turntable surface without pressing into it.
HARDCODED_GRASP_Z = 0.102

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
    'left': [0.253835, -0.315847, 0.230922, 2.130869, 0.107971, -2.304919, 1],
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
        # Turntable zone (pick-from-turntable task). We consult the STICKY
        # flag `has_ever_grasped_in_turntable` — not the instantaneous
        # `object_grasped` — to avoid a "close → open → re-descend"
        # oscillation: once we've ever successfully grasped in the turntable
        # zone, keep the floor at TURNTABLE_LIFT_Z until the EE physically
        # leaves the zone (the latch is reset in ServoRunner._loop).
        if TURNTABLE_Y_MIN <= pos[1] <= TURNTABLE_Y_MAX:
            if gripper_state.get('has_ever_grasped_in_turntable', False):
                pos[2] = max(pos[2], TURNTABLE_LIFT_Z)
            else:
                pos[2] = max(pos[2], TURNTABLE_SURFACE_Z)

        waypoints[i, :3] = pos
        waypoints[i, 3:6] = rot

        new_gripper = float(delta[6])

        reached_target = (args.if_release_when_reach_temp
                          and current_gripper > 0.5
                          and gripper_state['object_grasped']
                          and pos[1] >= args.release_y_threshold)

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
                 hardcoded_grasp_z=HARDCODED_GRASP_Z,
                 grasp_settle_s=0.3):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._gripper_hw = gripper_hw
        self._rtde_r = rtde_r
        # grasp_trick / grasp_z_threshold are LEGACY kwargs from the dd/
        # variant — retained so call sites don't break, but this file uses
        # the snap-grasp mechanism instead (see _do_turntable_grasp).
        self._grasp_trick = grasp_trick
        self._grasp_z_threshold = grasp_z_threshold
        self._grasp_detect_threshold = grasp_detect_threshold
        self._gripper_state = gripper_state
        self._x_offset = systematically_x_offset
        self._hardcoded_grasp_z = hardcoded_grasp_z
        self._grasp_settle_s = grasp_settle_s
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
        # Hold flag: when True, _loop pauses (does NOT pop buffer or call
        # servoL). Used by _do_turntable_grasp so it can safely run a
        # blocking moveL without fighting the streaming servo.
        self._hold = False

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
            # Hold gate: when _do_turntable_grasp is running a blocking
            # moveL, we must not call servoL in parallel (would fight the
            # motion mode). Skip buffer consumption and servoL entirely
            # while held; just pace the loop so we wake up promptly once
            # the hold is released.
            if self._hold:
                precise_wait(t_next)
                t_next += servo_dt
                now = time.monotonic()
                if t_next < now - servo_dt:
                    t_next = now + servo_dt
                continue

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
                # Turntable zone clamp (must mirror compute_waypoints).
                # Consult the sticky latch from gripper_state. Also: once
                # the servo target leaves the turntable y-range, reset
                # the sticky flag so a subsequent re-entry starts fresh
                # with "not yet grasped → floor = surface" semantics.
                in_turntable_zone = (TURNTABLE_Y_MIN <= p[1] <= TURNTABLE_Y_MAX)
                if in_turntable_zone:
                    if (self._gripper_state is not None
                            and self._gripper_state.get(
                                'has_ever_grasped_in_turntable', False)):
                        p[2] = max(p[2], TURNTABLE_LIFT_Z)
                    else:
                        p[2] = max(p[2], TURNTABLE_SURFACE_Z)
                else:
                    if self._gripper_state is not None:
                        self._gripper_state['has_ever_grasped_in_turntable'] = False

                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                target_base[0] += self._x_offset
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt

            now = time.monotonic()
            if t_next < now - servo_dt:
                t_next = now + servo_dt

    def _gripper_loop(self):
        """Asynchronous gripper dispatcher.

        On CLOSE while over the turntable: runs the snap-grasp sequence
        (see _do_turntable_grasp). Outside the turntable y-range, a CLOSE
        is skipped with a warning — grasping only makes sense over the
        turntable for this task, and a naked close-in-place during servo
        streaming would hit the old timing bug.

        OPEN commands are fired in place without pausing the servo; the
        robot continues tracking the Inferencer's planned trajectory while
        the fingers release.
        """
        last_pos = None
        while self._running:
            self._grip_event.wait(timeout=1.0)
            self._grip_event.clear()
            if not self._running:
                break

            with self._grip_lock:
                cmd = self._pending_grip
                self._pending_grip = None

            if cmd is None or cmd[0] == last_pos:
                continue

            grip_pos, label = cmd

            if grip_pos > 127:
                # ── CLOSE path ────────────────────────────────────────
                # Only snap-grasp over the turntable; otherwise skip.
                if self._rtde_r is None:
                    print("  [GRASP_SKIP] CLOSE requested but no rtde_r to "
                          "read position — ignoring")
                    continue

                cur_base = self._rtde_r.getActualTCPPose()
                cur_world = base_to_world(cur_base, self._T_bw)
                in_turntable = (TURNTABLE_Y_MIN <= cur_world[1] <= TURNTABLE_Y_MAX)

                if not in_turntable:
                    print(f"  [GRASP_SKIP] CLOSE requested outside turntable "
                          f"(y={cur_world[1]:.4f} not in "
                          f"[{TURNTABLE_Y_MIN}, {TURNTABLE_Y_MAX}]), ignoring — "
                          f"grasping is only safe over the turntable for this task")
                    if self._gripper_state is not None:
                        self._gripper_state['current'] = 0.0
                        self._gripper_state['chunks_since_grasp'] = None
                    continue

                # Safe to snap-grasp. Run the blocking sequence; it will
                # pause the servo, moveL down to HARDCODED_GRASP_Z, close,
                # detect, and flush the stale buffer.
                self._do_turntable_grasp(cur_base, cur_world)
                last_pos = grip_pos
            else:
                # ── OPEN path ─────────────────────────────────────────
                # Fire in place; no servo pause needed.
                #
                # Task-completion detection (v7): if we were holding an
                # object at the moment the policy commands OPEN, treat that
                # as task done. Capture the flag BEFORE clearing it below,
                # and set task_finished so run_episode breaks out of its
                # main loop and main() can auto-go-home without prompting.
                was_grasped = (self._gripper_state is not None
                               and self._gripper_state.get('object_grasped', False))
                print(f"  Gripper -> OPEN (pos={grip_pos})")
                self._gripper_hw.move(grip_pos, 255, 150)
                last_pos = grip_pos
                if self._gripper_state is not None:
                    self._gripper_state['object_grasped'] = False
                    self._gripper_state['current'] = 0.0
                    self._gripper_state['chunks_since_grasp'] = None
                    if was_grasped:
                        self._gripper_state['task_finished'] = True
                        print("  [TASK_DONE] Grasped → released; "
                              "marking task finished.")

    def _do_turntable_grasp(self, cur_base, cur_world):
        """Blocking snap-grasp sequence executed inside _gripper_loop.

        Fixes the bug where gripper_hw.move() fires asynchronously during
        a streaming servoL — the robot kept advancing through the buffered
        waypoints while the fingers were closing, so the grasp happened
        well above the intended z.

        Sequence:
          1. Set _hold = True to pause _loop (so it stops calling servoL).
          2. servoStop()  → exit streaming mode.
          3. moveL to (cur_x_base, cur_y_base, HARDCODED_GRASP_Z - T_bw.z).
             Only the z is overridden; xy stays where the policy drove us.
          4. gripper_hw.move(close).
          5. time.sleep(settle) so fingers physically close.
          6. Read gripper position → grasp detection → update gripper_state
             (object_grasped + sticky turntable latch).
          7. Flush the stale servo buffer: everything queued by the chunk
             that spawned this close cmd is now relative to a stale z and
             must be discarded. Reset _last_pose from the post-grasp
             physical pose.
          8. Advance _action_t by the number of flushed actions + any
             partial sub_step. This unblocks the Inferencer's
             wait_for_action_t so it can push the next chunk (which, on the
             next cycle, will see tail_pose at the snap z and plan the lift).
          9. Clear _hold → _loop resumes. Buffer is empty, so _loop idles
             at _last_pose (now at the snap z with the object grasped) until
             the Inferencer pushes the next chunk.
        """
        print(f"  [GRASP_SNAP] Start: current z={cur_world[2]:.4f}, "
              f"target z={self._hardcoded_grasp_z:.4f}  "
              f"(y={cur_world[1]:.4f})")

        # 1. Pause _loop. Sleep a little so _loop has at least one tick
        #    to notice the flag and stop calling servoL.
        self._hold = True
        time.sleep(0.03)

        # 2. Exit servo streaming mode so moveL can take over.
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass

        # 3. Build moveL target: same xy (base-frame, as rtde_r gave us),
        #    override z to snap height in base frame.
        target_base = list(cur_base)
        target_base[2] = self._hardcoded_grasp_z - self._T_bw[2, 3]

        # Only bother moving if we're not already at/below the snap z.
        dz = cur_world[2] - self._hardcoded_grasp_z
        if dz > 0.001:
            print(f"  [GRASP_SNAP] moveL  Δz={dz:.4f}  (descending)")
            try:
                self._rtde_c.moveL(target_base, 0.2, 0.5)
            except Exception as e:
                print(f"  [GRASP_SNAP] moveL failed: {e}  — aborting snap, "
                      f"falling back to in-place close")
        else:
            print(f"  [GRASP_SNAP] already at/below target z "
                  f"(Δz={dz:.4f}), skipping moveL")

        # 4. Close gripper.
        print(f"  Gripper -> CLOSE (snap, pos=255, force=255)")
        self._gripper_hw.move(255, 255, 255)

        # 5. Wait for fingers to physically settle.
        time.sleep(self._grasp_settle_s)

        # 6. Grasp detection + sticky latch.
        actual_grip = self._gripper_hw.get_current_position()
        if actual_grip < self._grasp_detect_threshold:
            if self._gripper_state is not None:
                self._gripper_state['object_grasped'] = True
                self._gripper_state['current'] = 1.0
                # We only enter _do_turntable_grasp from the in-turntable
                # branch above, so we KNOW the grasp happened in-zone.
                self._gripper_state['has_ever_grasped_in_turntable'] = True
            print(f"  [GRASP_DETECT] Object grasped "
                  f"(pos={actual_grip} < {self._grasp_detect_threshold})")
        else:
            if self._gripper_state is not None:
                self._gripper_state['object_grasped'] = False
                self._gripper_state['current'] = 0.0
                self._gripper_state['chunks_since_grasp'] = None
            print(f"  [GRASP_DETECT] Empty grasp "
                  f"(pos={actual_grip} >= {self._grasp_detect_threshold}), "
                  f"re-opening gripper")
            self._gripper_hw.move(0, 255, 150)

        # 7. Flush stale buffer and refresh _last_pose from physical.
        #    NOTE: the `_x_offset` is added by _loop when computing the
        #    servoL target (target_base[0] += _x_offset). To keep
        #    _last_pose in the same "ideal" frame that buffered waypoints
        #    use, we subtract _x_offset from the physical world x here.
        new_base = self._rtde_r.getActualTCPPose()
        new_world = base_to_world(list(new_base), self._T_bw)
        new_world[0] -= self._x_offset

        with self._lock:
            n_flushed_interp = len(self._buffer)
            self._buffer.clear()
            self._last_pose = np.array(new_world, dtype=np.float64)

        # 8. Advance action_t so the Inferencer's wait_for_action_t unblocks.
        #    Every flushed buffered action would have bumped action_t by 1.
        #    Plus the partial action currently in progress (sub_step > 0).
        with self._action_cond:
            actions_flushed = n_flushed_interp // INTERP_MULT
            if self._sub_step > 0:
                actions_flushed += 1
                self._sub_step = 0
            self._action_t += actions_flushed
            self._action_cond.notify_all()

        # 9. Resume _loop.
        self._hold = False
        print(f"  [GRASP_SNAP] Done. Flushed {n_flushed_interp} interp pts "
              f"({actions_flushed} actions); action_t -> {self._action_t}")


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
#  Single episode execution
# ═══════════════════════════════════════════════════════════════════

def run_episode(model, cam, rtde_c, rtde_r, gripper_hw, T_bw,
                args, infer_kwargs, action_stats, chunk_len, episode_num):
    """Run one closed-loop episode. Returns the number of inference steps executed."""

    print(f"\n{'─' * 60}")
    print(f"  Episode {episode_num}")
    print(f"{'─' * 60}")

    # Fresh gripper state for each episode
    gripper_state = {
        'current': 0.0,
        'object_grasped': False,
        'chunks_since_grasp': None,
        'task_finished': False,
        # Sticky latch for turntable z-floor: once we have ever successfully
        # grasped in the turntable zone, keep floor at TURNTABLE_LIFT_Z until
        # the EE physically leaves the zone. Reset in ServoRunner._loop.
        'has_ever_grasped_in_turntable': False,
    }

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
                        hardcoded_grasp_z=args.hardcoded_grasp_z,
                        grasp_settle_s=args.grasp_settle_s)
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
    step = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            result = inferencer.wait_result(timeout=0.2)
            if result is None:
                continue

            step += 1

            if gripper_state['task_finished']:
                print("\n>>> Task finished! Object released at target. <<<")
                break

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

    except KeyboardInterrupt:
        # Propagate so the outer loop can shut down cleanly.
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

    # ── Episode teardown ─────────────────────────────────────────────
    # NOTE: servo.stop() joins _gripper_loop, which means any in-flight
    # _do_turntable_grasp sequence (moveL + close + settle, ~1.5s worst
    # case) runs to completion before we return — an interrupted grasp
    # would leave the gripper half-closed.
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

    if infer_times:
        arr = np.array(infer_times)
        print(f"\nInference timing ({len(arr)} cycles):  "
              f"mean={arr.mean():.0f}ms  std={arr.std():.0f}ms  "
              f"min={arr.min():.0f}ms  max={arr.max():.0f}ms  "
              f"median={np.median(arr):.0f}ms")

    print(f"Episode {episode_num} done. Executed {step} inference steps.")
    return step


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RTC closed-loop control with flow matching / QwenPI (v7 — single-shot)")
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
    parser.add_argument("--fix_rotation", action="store_true", default=True)
    parser.add_argument("--no_fix_rotation", dest="fix_rotation",
                        action="store_false")
    # Pick-from-turntable task: gripper lock defaults are flipped off.
    # The turntable sticky z-floor latch (see TURNTABLE_* constants) already
    # protects against re-descent after a grasp, so we let the policy drive
    # the gripper freely — no artificial "don't release until target y".
    parser.add_argument("--if_grasped_not_release", action="store_true",
                        default=False,
                        help="Once gripper closes, keep it closed "
                             "(default: False for pick-from-turntable task)")
    parser.add_argument("--allow_release", dest="if_grasped_not_release",
                        action="store_false")
    parser.add_argument("--if_grasp_delay_temp_solution", action="store_true",
                        default=False)
    parser.add_argument("--no_grasp_delay", dest="if_grasp_delay_temp_solution",
                        action="store_false")
    parser.add_argument("--if_release_when_reach_temp", action="store_true",
                        default=False,
                        help="Allow release when y >= release_y_threshold, then "
                             "exit (default: False for pick-from-turntable task)")
    parser.add_argument("--no_release_when_reach", dest="if_release_when_reach_temp",
                        action="store_false")
    parser.add_argument("--release_y_threshold", type=float, default=-0.318)
    parser.add_argument("--grasp_detect_threshold", type=int, default=200)
    # --- Legacy grasp_trick flags (kept for CLI compat, largely unused) ---
    # In this 2fm variant, close commands over the turntable are handled by
    # the snap-grasp sequence (see ServoRunner._do_turntable_grasp): the
    # servo pauses, the arm moveL's to HARDCODED_GRASP_Z, the gripper
    # closes, then servo resumes. grasp_z_threshold is NOT used by the snap
    # path; the snap z is set by --hardcoded_grasp_z below.
    parser.add_argument("--if_grasp_trick", action="store_true", default=True)
    parser.add_argument("--no_grasp_trick", dest="if_grasp_trick",
                        action="store_false")
    parser.add_argument("--grasp_z_threshold", type=float, default=0.117,
                        help="LEGACY — unused by the snap-grasp path in 2fm/v7. "
                             "Kept for CLI backward compatibility.")
    # --- Snap-grasp parameters ---
    parser.add_argument("--hardcoded_grasp_z", type=float,
                        default=HARDCODED_GRASP_Z,
                        help="Snap-grasp target z in world frame. When policy "
                             "predicts close AND the EE is over the turntable, "
                             "the servo pauses and moveL's down to this z "
                             "(keeping current xy) before closing the gripper. "
                             "(default: 0.102, just above TURNTABLE_SURFACE_Z=0.101)")
    parser.add_argument("--grasp_settle_s", type=float, default=0.3,
                        help="Seconds to wait after sending gripper close for "
                             "fingers to physically settle before reading grasp "
                             "detection (default: 0.3)")
    parser.add_argument("--systematically_x_offset", type=float, default=0.00)
    # ── Inference server (split from this script to avoid the ~30s
    # model-load cost on every iteration). Default ON: start
    # `python ur5n/2fm/inference_server.py` in another terminal first,
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
                        default="/tmp/starvla_infer_2fm.sock")
    args = parser.parse_args()

    T_bw = BASE_IN_WORLD[args.arm]

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

    assert args.n_actions + args.inference_delay <= chunk_len, (
        f"n_actions ({args.n_actions}) + inference_delay ({args.inference_delay}) "
        f"must be <= chunk_len ({chunk_len})")
    assert args.inference_delay <= args.n_actions, (
        f"inference_delay ({args.inference_delay}) must be <= n_actions ({args.n_actions})")

    infer_kwargs = {}

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
    print(f"  Closed-Loop RTC (Flow Matching / QwenPI) — v7")
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
    print(f"{'=' * 60}")
    print(f"\n  Flow: single-shot — runs one episode then auto-exits.")
    print(f"    Episode ends on release-after-grasp OR max_steps.")
    print(f"    Ctrl+C at any time to abort.")
    print(f"{'=' * 60}\n")

    try:
        try:
            input("\nPress ENTER to start the episode (Ctrl+C to abort)... ")
        except EOFError:
            pass
        run_episode(model, cam, rtde_c, rtde_r, gripper_hw, T_bw,
                    args, infer_kwargs, action_stats, chunk_len,
                    episode_num=1)
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
