#!/usr/bin/env python3
"""
RTC (Real-Time Chunking) closed-loop control for air-hockey (QwenPI, 2-D world XY).

Three-thread architecture (mirrors ur5n/fm/closedloop_rtc_v6.py but stripped of
all pick-and-place baggage — no gripper, no Z constraints beyond a single locked
value, no rotation deltas):

  ServoRunner thread:  100 Hz continuous servoL from a thread-safe deque buffer.
                       Tracks action_t (completed policy-rate steps). tail_pose
                       lets each new chunk splice onto the previous chunk's end
                       without jitter. Pure 2-D XY servo with locked Z + rot.
  Inferencer thread:   wait_for_action_t(n_consumed) -> grab wrist frame ->
                       model.predict_action_realtime(prev_chunk, inference_delay)
                       -> push output[inference_delay : inference_delay+n_actions]
                       into the servo buffer. Next prev = output[n_actions:].
  Main thread:         Synchronous initial inference, then logging + per-step
                       viz. Ctrl+C exits cleanly.

The robot WILL move. Press Ctrl+C to stop.

Usage:
    # Terminal 1: start the inference server (model loaded once)
    python ur5n/airhockey-d/fm/inference_server.py

    # Terminal 2: run RTC closed loop (defaults: --use_server,
    # n_actions=8, inference_delay=8)
    python ur5n/airhockey-d/fm/closedloop_rtc.py
    python ur5n/airhockey-d/fm/closedloop_rtc.py --n_actions 4 --inference_delay 4
    python ur5n/airhockey-d/fm/closedloop_rtc.py --save_rollout
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
from PIL import Image
import cv2
import torch

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
DEFAULT_SOCKET = "/tmp/starvla_infer_airhockey.sock"

POLICY_HZ = 50
INTERP_MULT = 2
SERVO_HZ = POLICY_HZ * INTERP_MULT          # 100 Hz
CHUNK_LEN = 32

# Data-derived safety bounds (state.x / state.y from training distribution,
# expanded by --xy_buffer (default 0.05 m) — see closedloop_sync.py).
# state.x range:  0.2996 .. 0.4491
# state.y range: -0.3300 .. -0.1201
XY_DATA_MIN = np.array([0.2996-0.1, -0.3300], dtype=np.float64)
XY_DATA_MAX = np.array([0.4491, -0.1201], dtype=np.float64)

# Training-distribution LEFT home — verbatim from 3airhockey_dynamic_bounce.py.
# This is what the data-collection script sampled per-episode before each
# trial. NOT the same as the safety clamp above — sampling stays inside the
# training distribution so policy startup is in-distribution.
LEFT_HOME_DYNAMIC_Z = 0.176
LEFT_HOME_X_RANGE = (0.30, 0.45)
LEFT_HOME_Y_RANGES = [(-0.33, -0.31), (-0.14, -0.12)]
LEFT_HOME_SAMPLE_MAX_TRIES = 30
LEFT_HOME_DYNAMIC_ROTVEC = [-2.2192, 2.2148, 0.0091]
_Y_LENGTHS = [float(b - a) for (a, b) in LEFT_HOME_Y_RANGES]
_Y_TOTAL = sum(_Y_LENGTHS)


def sample_left_init_xy_with_ik(rtde_c, rtde_r, T_bw, seed=None):
    """Sample LEFT initial XY uniformly: X ∈ LEFT_HOME_X_RANGE,
    Y ∈ length-weighted-union(LEFT_HOME_Y_RANGES). IK-checked at
    Z=LEFT_HOME_DYNAMIC_Z, rotvec=LEFT_HOME_DYNAMIC_ROTVEC. Raises
    RuntimeError if no IK-valid sample in LEFT_HOME_SAMPLE_MAX_TRIES."""
    import random
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
#  Waypoint computation (2-D XY cumsum + safety clamp + Z/rot pad)
# ═══════════════════════════════════════════════════════════════════

def compute_waypoints(start_pose_world, deltas_xy, n_exec,
                      xy_lo, xy_hi, locked_z, locked_rot):
    """Build (n_exec, 6) world-frame waypoints from 2-D deltas.

    Pulls XY from start_pose_world[:2], cumsums the per-step deltas, clamps
    to the safety box, and pads with locked Z + locked rotvec. The locked
    Z/rot fields are NOT touched by the policy.
    """
    n = min(n_exec, len(deltas_xy))
    waypoints = np.zeros((n, 6), dtype=np.float64)
    xy = np.array(start_pose_world[:2], dtype=np.float64)
    for i in range(n):
        xy = xy + deltas_xy[i]
        xy = np.clip(xy, xy_lo, xy_hi)
        waypoints[i, 0] = xy[0]
        waypoints[i, 1] = xy[1]
        waypoints[i, 2] = locked_z
        waypoints[i, 3:6] = locked_rot
    return waypoints


# ═══════════════════════════════════════════════════════════════════
#  ServoRunner — continuous 100 Hz servo (2-D XY, locked Z + rot)
# ═══════════════════════════════════════════════════════════════════

class ServoRunner:
    """100 Hz servo with tail_pose splicing and timing guard. No gripper."""

    def __init__(self, rtde_c, T_bw, xy_lo, xy_hi, locked_z, locked_rot):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._xy_lo = xy_lo
        self._xy_hi = xy_hi
        self._locked_z = float(locked_z)
        self._locked_rot = np.array(locked_rot, dtype=np.float64)
        self._buffer = collections.deque()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pose = None
        self._starve_count = 0
        self._action_t = 0
        self._sub_step = 0
        self._action_cond = threading.Condition()

    def start(self, initial_pose_world):
        self._last_pose = np.array(initial_pose_world[:6], dtype=np.float64)
        self._running = True
        self._action_t = 0
        self._sub_step = 0
        self._starve_count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        with self._action_cond:
            self._action_cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass

    def push_waypoints(self, start_pose, waypoints_50hz):
        """Interpolate policy-rate waypoints to SERVO_HZ and append."""
        interp = interpolate_waypoints(start_pose, waypoints_50hz, INTERP_MULT)
        with self._lock:
            self._buffer.extend(interp)

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
                # Safety: clamp XY, hard-lock Z + rotvec every cycle.
                p[0] = np.clip(p[0], self._xy_lo[0], self._xy_hi[0])
                p[1] = np.clip(p[1], self._xy_lo[1], self._xy_hi[1])
                p[2] = self._locked_z
                p[3:6] = self._locked_rot

                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt

            now = time.monotonic()
            if t_next < now - servo_dt:
                t_next = now + servo_dt


# ═══════════════════════════════════════════════════════════════════
#  Inferencer — cadence-aware RTC inference
# ═══════════════════════════════════════════════════════════════════

class InferenceResult:
    __slots__ = ("normalized", "deltas_xy", "camera_image",
                 "obs_ms", "infer_ms", "actual_delay", "n_pushed",
                 "waypoints", "prev_norm", "prev_unnorm_xy", "start_pose")

    def __init__(self, normalized, deltas_xy, camera_image,
                 obs_ms, infer_ms, actual_delay, n_pushed, waypoints,
                 prev_norm, prev_unnorm_xy, start_pose):
        self.normalized = normalized
        self.deltas_xy = deltas_xy
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.n_pushed = n_pushed
        self.waypoints = waypoints
        self.prev_norm = prev_norm
        self.prev_unnorm_xy = prev_unnorm_xy
        self.start_pose = start_pose


class Inferencer:
    def __init__(self, model, cam, servo, n_actions, inference_delay,
                 instruction, infer_kwargs, action_stats,
                 xy_lo, xy_hi, locked_z, locked_rot, chunk_len):
        self._model = model
        self._cam = cam
        self._servo = servo
        self._n_actions = n_actions
        self._inference_delay = inference_delay
        self._instruction = instruction
        self._infer_kwargs = infer_kwargs
        self._action_stats = action_stats
        self._xy_lo = xy_lo
        self._xy_hi = xy_hi
        self._locked_z = locked_z
        self._locked_rot = locked_rot
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
            prev_unnorm_xy = baseframework.unnormalize_actions(
                prev_norm_save, self._action_stats).astype(np.float64)

            example = {"image": [pil_img], "lang": self._instruction}
            prev_norm_batch = prev_action_chunk[np.newaxis, ...]

            # GPU timing if available, else wall clock fallback.
            use_cuda_timer = torch.cuda.is_available()
            if use_cuda_timer:
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record()
            t_wall = time.monotonic()

            out = self._model.predict_action_realtime(
                examples=[example],
                prev_action_chunk_normalized=prev_norm_batch,
                inference_delay=self._inference_delay,
                **self._infer_kwargs,
            )

            if use_cuda_timer:
                end_evt.record()
                end_evt.synchronize()
                infer_ms = start_evt.elapsed_time(end_evt)
            else:
                infer_ms = (time.monotonic() - t_wall) * 1000

            new_normalized = out["normalized_actions"][0].astype(np.float32)
            new_deltas_xy = baseframework.unnormalize_actions(
                new_normalized, self._action_stats).astype(np.float64)

            free_start = self._inference_delay
            free_end = free_start + self._n_actions
            free_deltas = new_deltas_xy[free_start:free_end]
            n_push = len(free_deltas)

            start_pose = self._servo.tail_pose
            waypoints = compute_waypoints(
                start_pose, free_deltas, n_push,
                self._xy_lo, self._xy_hi, self._locked_z, self._locked_rot)
            actual_n = waypoints.shape[0]

            self._servo.push_waypoints(start_pose, waypoints)

            prev_action_chunk = new_normalized[self._n_actions:]
            n_consumed += self._n_actions

            self._result_queue.put(
                InferenceResult(new_normalized, new_deltas_xy, pil_img,
                                obs_ms, infer_ms, self._inference_delay,
                                actual_n, waypoints, prev_norm_save,
                                prev_unnorm_xy, start_pose))


# ═══════════════════════════════════════════════════════════════════
#  Visualization
# ═══════════════════════════════════════════════════════════════════

def compute_consistency_metrics(result, inference_delay):
    """MAE between prev_norm and output on the 2 action dims.

    Hard-fixed region [0:D]    — should be ~0 by construction (prefix inpaint).
    Soft-guided region [D:len] — smaller = more temporally coherent policy.
    """
    prev = result.prev_norm
    out = result.normalized[:len(prev)]
    D = inference_delay
    metrics = {}
    for nd, name in [(0, "dx"), (1, "dy")]:
        pv, ov = prev[:, nd], out[:, nd]
        metrics[f"{name}_fix"] = float(np.mean(np.abs(pv[:D] - ov[:D]))) if D > 0 else 0.0
        metrics[f"{name}_guide"] = (float(np.mean(np.abs(pv[D:] - ov[D:])))
                                    if len(pv) > D else 0.0)
    return metrics


def visualize_rtc_debug(result, pose_world, step, save_path, instruction,
                        n_actions, inference_delay, chunk_len,
                        xy_bounds):
    A = n_actions
    D = inference_delay
    L = chunk_len

    output_norm = result.normalized
    prev_norm = result.prev_norm
    deltas_xy = result.deltas_xy
    start_pose = result.start_pose

    # Full predicted absolute XY trajectory from the chunk-tail start_pose.
    start_xy = np.array(start_pose[:2], dtype=np.float64)
    full_traj = start_xy + np.cumsum(deltas_xy, axis=0)  # (L, 2)
    # The actually-pushed segment is output[D : D+A] starting from tail.
    exec_traj = full_traj[D:D + A]

    fig = plt.figure(figsize=(22, 12))
    gs_fig = GridSpec(3, 3, figure=fig, hspace=0.32, wspace=0.30,
                      width_ratios=[1, 1.2, 1.2])

    # Camera
    ax_cam = fig.add_subplot(gs_fig[0:2, 0])
    ax_cam.imshow(result.camera_image)
    ax_cam.set_title("Wrist camera", fontsize=11, fontweight="bold")
    ax_cam.axis("off")

    # Info panel
    ax_info = fig.add_subplot(gs_fig[2, 0])
    ax_info.axis("off")
    info = (
        f"Step {step}\n"
        f"L={L}  A={A}  D={D}\n"
        f"obs={result.obs_ms:.0f}ms  infer={result.infer_ms:.0f}ms\n"
        f"pushed={result.n_pushed}\n\n"
        f"Pose now (world):\n"
        f"  x={pose_world[0]:.4f}\n"
        f"  y={pose_world[1]:.4f}\n"
        f"  z={pose_world[2]:.4f}\n\n"
        f"\"{instruction[:60]}\""
    )
    ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                 fontsize=10, va="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Top-down XY trajectory plot
    ax_xy = fig.add_subplot(gs_fig[0:2, 1])
    (x_lo, y_lo), (x_hi, y_hi) = xy_bounds
    ax_xy.add_patch(plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                  fill=False, ls="--", ec="gray",
                                  label="safety box"))
    # Chunk-tail prefix [0:D) and future [D+A:L) drawn dashed pink.
    if D > 0:
        ax_xy.plot(full_traj[:D, 0], full_traj[:D, 1], "o:",
                   color="silver", ms=3, lw=1.0,
                   label=f"prefix [0:{D})")
    ax_xy.plot(full_traj[D + A:, 0], full_traj[D + A:, 1], "o--",
               color="pink", ms=3, lw=1.2,
               label=f"future [{D + A}:{L})")
    ax_xy.plot(exec_traj[:, 0], exec_traj[:, 1], "s-",
               color="#e41a1c", ms=5, lw=2.0,
               label=f"exec [{D}:{D + A}) ({result.n_pushed})")
    ax_xy.plot(pose_world[0], pose_world[1], "^",
               color="green", ms=12, label="now")
    ax_xy.plot(start_xy[0], start_xy[1], "P",
               color="purple", ms=9, label="tail_pose")
    ax_xy.set_xlabel("x (world m)")
    ax_xy.set_ylabel("y (world m)")
    ax_xy.set_title("Top-down XY", fontweight="bold")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3)
    ax_xy.legend(fontsize=7, loc="best")

    # Consistency plots (dx, dy)
    norm_dims = [(0, "dx"), (1, "dy")]
    colors = ["#e41a1c", "#377eb8"]
    n_prev = len(prev_norm)

    for row, ((nd, nname), color) in enumerate(zip(norm_dims, colors)):
        ax_c = fig.add_subplot(gs_fig[row, 2])
        idx = np.arange(n_prev)
        pv = prev_norm[:, nd]
        ov = output_norm[:n_prev, nd]

        ax_c.axvspan(-0.5, D - 0.5, alpha=0.15, color="green")
        ax_c.axvspan(D - 0.5, n_prev - 0.5, alpha=0.10, color="gold")
        ax_c.plot(idx, pv, "o--", color="gray", ms=4, lw=1.5,
                  label="prev", alpha=0.8)
        ax_c.plot(idx, ov, "s-", color=color, ms=4, lw=2.0,
                  label="output")

        mae_fix = (float(np.mean(np.abs(pv[:D] - ov[:D])))
                   if D > 0 else 0.0)
        mae_guide = (float(np.mean(np.abs(pv[D:] - ov[D:])))
                     if n_prev > D else 0.0)

        ax_c.set_ylabel(nname, fontsize=10, fontweight="bold")
        if row == 0:
            ax_c.set_title(
                "CONSISTENCY (prev vs output, normalized)\n"
                "green=hard-fixed   gold=soft-guided",
                fontsize=10, fontweight="bold")
            ax_c.legend(fontsize=8, loc="upper right")
        ax_c.text(0.5, 0.02,
                  f"fix MAE={mae_fix:.4f}  guide MAE={mae_guide:.4f}",
                  transform=ax_c.transAxes, fontsize=8, ha="center",
                  color="dimgray")
        ax_c.grid(True, alpha=0.3)
        if row < len(norm_dims) - 1:
            plt.setp(ax_c.get_xticklabels(), visible=False)
        else:
            ax_c.set_xlabel("Action index", fontsize=9)

    # Per-step delta over the full chunk (bottom-middle).
    ax_d = fig.add_subplot(gs_fig[2, 1])
    ts = np.arange(L) / POLICY_HZ
    ax_d.plot(ts, deltas_xy[:, 0], "o-", ms=3, color="#e41a1c", label="dx")
    ax_d.plot(ts, deltas_xy[:, 1], "o-", ms=3, color="#377eb8", label="dy")
    ax_d.axvspan(0, ts[D - 1] if D > 0 else 0, alpha=0.15, color="green")
    ax_d.axvspan(ts[D - 1] if D > 0 else 0,
                 ts[min(D + A - 1, L - 1)], alpha=0.15, color="palegreen")
    ax_d.axhline(0, color="black", lw=0.5)
    ax_d.set_xlabel("t (s) — 50 Hz")
    ax_d.set_ylabel("delta (m / step)")
    ax_d.set_title("Per-step delta (full chunk)", fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)

    fig.suptitle(f"RTC Debug — Step {step}",
                 fontsize=14, fontweight="bold", y=1.00)
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def visualize_rtc_summary(consistency_log, infer_times, pose_log, save_path,
                          xy_bounds):
    steps = np.arange(1, len(consistency_log) + 1)
    dim_names = ["dx", "dy"]
    dim_colors = ["#e41a1c", "#377eb8"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    fig.suptitle("RTC Air-Hockey Rollout Summary",
                 fontsize=14, fontweight="bold")

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
    ax.set_title("Consistency — Soft-Guided [D:end]", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if pose_log:
        for d, name, color in [(0, "x", "#e41a1c"), (1, "y", "#377eb8")]:
            ax.plot(steps[:len(pose_log)], [p[d] for p in pose_log],
                    "-", lw=1.5, color=color, label=name, alpha=0.8)
    ax.set_xlabel("Step"); ax.set_ylabel("Position (world m)")
    ax.set_title("Robot XY over time", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    arr = np.array(infer_times) if infer_times else np.zeros(1)
    ax.plot(steps[:len(arr)], arr, "o-", ms=2, lw=1, color="#984ea3")
    if len(arr) > 0:
        ax.axhline(arr.mean(), color="red", ls="--", lw=1, alpha=0.5,
                   label=f"mean={arr.mean():.0f}ms")
    ax.set_xlabel("Step"); ax.set_ylabel("ms")
    ax.set_title("Inference Time", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    if pose_log:
        (x_lo, y_lo), (x_hi, y_hi) = xy_bounds
        ax.add_patch(plt.Rectangle((x_lo, y_lo), x_hi - x_lo, y_hi - y_lo,
                                   fill=False, ls="--", ec="gray",
                                   label="safety box"))
        xs = [p[0] for p in pose_log]
        ys = [p[1] for p in pose_log]
        ax.plot(xs, ys, "o-", ms=3, lw=1.5, color="#377eb8", alpha=0.7)
        ax.plot(xs[0], ys[0], "^", ms=10, color="green", label="start")
        ax.plot(xs[-1], ys[-1], "v", ms=10, color="red", label="end")
        ax.set_xlabel("x (world m)"); ax.set_ylabel("y (world m)")
        ax.set_title("Top-Down Trajectory", fontweight="bold")
        ax.legend(fontsize=8); ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    # Spare bottom-right: distance travelled per step (proxy for activity).
    ax = axes[2, 1]
    if pose_log and len(pose_log) > 1:
        arr_p = np.array(pose_log)
        dists = np.linalg.norm(np.diff(arr_p[:, :2], axis=0), axis=1)
        ax.plot(steps[1:1 + len(dists)], dists, "-", lw=1.5,
                color="#4daf4a", label="|Δ xy|")
    ax.set_xlabel("Step"); ax.set_ylabel("m / step")
    ax.set_title("XY motion between log samples", fontweight="bold")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RTC closed-loop air-hockey (QwenPI flow-matching, 2-D XY)")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--instruction", type=str, default=DEFAULT_INSTRUCTION)
    parser.add_argument("--init_pose", choices=["sample", "current"],
                        default="sample",
                        help="'sample' = IK-checked random from training "
                             "distribution + moveJ; 'current' = use current "
                             "pose (no moveJ, for manual jog testing)")
    parser.add_argument("--init_seed", type=int, default=None,
                        help="RNG seed for init-pose sampling (default: time-based)")
    parser.add_argument("--n_actions", type=int, default=9,
                        help="Actions to push per cycle (1..chunk_len). "
                             "Default 8 = 0.16s exec @ 50Hz.")
    parser.add_argument("--inference_delay", type=int, default=9,
                        help="Prefix length pinned to previous prediction. "
                             "Must satisfy inference_delay <= n_actions and "
                             "n_actions + inference_delay <= chunk_len.")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference cycles (0 = unlimited, Ctrl+C to stop)")
    parser.add_argument("--xy_buffer", type=float, default=0.05,
                        help="Safety buffer (m) added outside data XY range")
    parser.add_argument("--x_min", type=float, default=None,
                        help="Override x_min (default: data min - buffer)")
    parser.add_argument("--x_max", type=float, default=None,
                        help="Override x_max (default: data max + buffer)")
    parser.add_argument("--y_min", type=float, default=None,
                        help="Override y_min (default: data min - buffer)")
    parser.add_argument("--y_max", type=float, default=None,
                        help="Override y_max (default: data max + buffer)")
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--use_server", action="store_true", default=True,
                        help="Connect to inference_server.py over a Unix socket "
                             "instead of loading the model in-process.")
    parser.add_argument("--no_use_server", dest="use_server",
                        action="store_false")
    parser.add_argument("--server_socket", type=str,
                        default=DEFAULT_SOCKET)
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
    assert chunk_len == CHUNK_LEN, (
        f"unexpected chunk_len={chunk_len}, expected {CHUNK_LEN}")
    assert 1 <= args.n_actions <= chunk_len, (
        f"n_actions={args.n_actions} must be in [1, {chunk_len}]")
    assert args.n_actions + args.inference_delay <= chunk_len, (
        f"n_actions ({args.n_actions}) + inference_delay "
        f"({args.inference_delay}) must be <= chunk_len ({chunk_len})")
    assert args.inference_delay <= args.n_actions, (
        f"inference_delay ({args.inference_delay}) must be <= "
        f"n_actions ({args.n_actions})")

    infer_kwargs = dict()  # QwenPI flow-matching takes no extra kwargs.

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
    print("Waiting for first frame from background reader...")
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

    # ── Lock Z + rotation from (achieved) pose ──────────────────────
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
        rollout_dir = Path("ur5n/airhockey-d/fm/rollouts_rtc") / f"rtc_{ts_str}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"Rollout: {rollout_dir}")
        with open(rollout_dir / "config.json", "w") as f:
            json.dump({
                "mode": "rtc_airhockey",
                "checkpoint": args.checkpoint,
                "instruction": args.instruction,
                "arm": args.arm,
                "n_actions": args.n_actions,
                "inference_delay": args.inference_delay,
                "chunk_len": chunk_len,
                "policy_hz": POLICY_HZ,
                "servo_hz": SERVO_HZ,
                "xy_bounds": [[x_lo, y_lo], [x_hi, y_hi]],
                "locked_z": locked_z,
                "locked_rot": locked_rot.tolist(),
                "dataset_key": dataset_key,
            }, f, indent=2)

    # ── Print config ────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Air-hockey RTC Closed-Loop (QwenPI flow-matching)")
    print(f"  Arm:              {args.arm}  IP={robot_ip}")
    print(f"  Camera:           /dev/video{args.camera_dev}")
    print(f"  Instruction:      \"{args.instruction}\"")
    print(f"  n_actions:        {args.n_actions}")
    print(f"  inference_delay:  {args.inference_delay}")
    print(f"  Chunk len:        {chunk_len}")
    print(f"  Policy / Servo:   {POLICY_HZ} Hz x{INTERP_MULT} = {SERVO_HZ} Hz")
    print(f"  XY safety box:    x in [{x_lo:.4f}, {x_hi:.4f}]  "
          f"y in [{y_lo:.4f}, {y_hi:.4f}]")
    print(f"  Use server:       {args.use_server}"
          f"{' (' + args.server_socket + ')' if args.use_server else ''}")
    print(f"  Max steps:        {'unlimited' if args.max_steps == 0 else args.max_steps}")
    print(f"  Save rollout:     {args.save_rollout}")
    print(f"{'=' * 60}")

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Step 0: synchronous initial inference (no prev_chunk) ───────
    pose_base = rtde_r.getActualTCPPose()
    pose_world = base_to_world(pose_base, T_bw)
    current_6d = np.array(
        [pose_world[0], pose_world[1], locked_z,
         locked_rot[0], locked_rot[1], locked_rot[2]],
        dtype=np.float64,
    )

    pil_img = cam.grab_pil()
    while pil_img is None:
        pil_img = cam.grab_pil()

    example = {"image": [pil_img], "lang": args.instruction}
    t_infer = time.monotonic()
    out0 = model.predict_action(examples=[example], **infer_kwargs)
    init_infer_ms = (time.monotonic() - t_infer) * 1000
    print(f"[init] inference={init_infer_ms:.0f}ms (sync, no prefix)")

    current_normalized = out0["normalized_actions"][0].astype(np.float32)
    current_deltas_xy = baseframework.unnormalize_actions(
        current_normalized, action_stats).astype(np.float64)

    # Push the first inference_delay actions so the servo has waypoints to
    # execute while the inferencer thread blocks on wait_for_action_t.
    n_init = min(args.inference_delay, len(current_deltas_xy))
    init_waypoints = compute_waypoints(
        current_6d, current_deltas_xy, n_init,
        xy_lo, xy_hi, locked_z, locked_rot)

    servo = ServoRunner(rtde_c, T_bw, xy_lo, xy_hi, locked_z, locked_rot)
    servo.push_waypoints(current_6d, init_waypoints)
    servo.start(current_6d)

    inferencer = Inferencer(
        model, cam, servo, args.n_actions, args.inference_delay,
        args.instruction, infer_kwargs, action_stats,
        xy_lo, xy_hi, locked_z, locked_rot, chunk_len=chunk_len)
    # prev_action_chunk passed into RTC is the tail of the just-issued chunk,
    # length chunk_len - n_actions (same as DD v6).
    init_prev_chunk = current_normalized[:chunk_len - args.n_actions]
    inferencer.start(init_prev_chunk)

    # ── Main logging loop ───────────────────────────────────────────
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

            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            infer_times.append(result.infer_ms)

            metrics = compute_consistency_metrics(result, args.inference_delay)
            consistency_log.append(metrics)
            pose_log.append(list(pose_world))

            buf_len = servo.buffer_len
            fix_mae = np.mean([metrics[f"{d}_fix"] for d in ["dx", "dy"]])
            guide_mae = np.mean([metrics[f"{d}_guide"] for d in ["dx", "dy"]])
            suffix = (f"  delay={result.actual_delay}  "
                      f"pushed={result.n_pushed}  buf={buf_len}  "
                      f"fix={fix_mae:.4f}  guide={guide_mae:.4f}")
            if servo.starve_count > 0:
                suffix += f"  starved={servo.starve_count}"

            print(f"[step {step:4d}]  "
                  f"obs={result.obs_ms:5.0f}ms  infer={result.infer_ms:5.0f}ms  "
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}]"
                  f"{suffix}")

            if args.save_rollout and rollout_dir is not None:
                if result.camera_image is not None:
                    img_path = rollout_dir / "images" / f"step_{step:04d}.jpg"
                    result.camera_image.save(str(img_path), quality=90)

                viz_path = rollout_dir / "images" / f"step_{step:04d}_viz.png"
                visualize_rtc_debug(
                    result, pose_world, step, str(viz_path),
                    args.instruction, args.n_actions, args.inference_delay,
                    chunk_len, ((x_lo, y_lo), (x_hi, y_hi)))

                log_entry = {
                    "step": step,
                    "wall_time": time.time(),
                    "ee_pose_world": list(pose_world),
                    "ee_pose_base": list(pose_base),
                    "pred_deltas_xy_full": result.deltas_xy.tolist(),
                    "waypoints_world": result.waypoints.tolist(),
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
                if result.prev_unnorm_xy is not None:
                    log_entry["prev_unnorm_xy"] = result.prev_unnorm_xy.tolist()
                rollout_log.append(log_entry)

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        print("Cleaning up...")
        try:
            inferencer.stop()
        except Exception:
            pass
        try:
            servo.stop()
        except Exception:
            pass
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
            if consistency_log:
                visualize_rtc_summary(
                    consistency_log, infer_times, pose_log,
                    str(rollout_dir / "summary.png"),
                    ((x_lo, y_lo), (x_hi, y_hi)))
                print(f"  Summary plot: {rollout_dir / 'summary.png'}")

        if infer_times:
            arr = np.array(infer_times)
            print(f"\nInference timing ({len(arr)} cycles):  "
                  f"mean={arr.mean():.0f}ms  std={arr.std():.0f}ms  "
                  f"min={arr.min():.0f}ms  max={arr.max():.0f}ms  "
                  f"median={np.median(arr):.0f}ms")

        print(f"Done. Executed {step} inference cycles.")


if __name__ == "__main__":
    main()
