#!/usr/bin/env python3
"""
Step 10 + Website Stopwatch: Closed-loop control with RTC + smooth interpolation
and an auto-triggered browser stopwatch for timing the policy execution.

Same as step10-dd-rtc-closed-loop-smooth.py but with an embedded HTTP server
that auto-starts a stopwatch when the policy loop begins and auto-stops it
when the policy finishes (grasp success, max_steps, or Ctrl+C).

The stopwatch runs via SSE (Server-Sent Events) in a daemon thread — zero
performance impact on the robot control loop.

Usage:
    python ur5/step10-dd-rtc-closed-loop-smooth-with_website.py
    python ur5/step10-dd-rtc-closed-loop-smooth-with_website.py --n_actions 14 --arm left
    python ur5/step10-dd-rtc-closed-loop-smooth-with_website.py --stopwatch_port 8765
    python ur5/step10-dd-rtc-closed-loop-smooth-with_website.py --n_actions 8 --inference_delay 8 --arm left --no_save_rollout --no_fixed_steps

"""

import sys
import os
import time
import json
import argparse
import threading
import queue
import http.server
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.spatial.transform import Rotation
from PIL import Image

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "ur5"))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.model.framework.base_framework import baseframework

import importlib
_step2 = importlib.import_module("ur5.step2-replace-with-real-camera")
RealCamera = _step2.RealCamera

import modular_policy
project_root = Path(modular_policy.__file__).parent.parent
sys.path.insert(0, str(project_root / 'scripts'))

_extrinsics_dir = os.path.join(
    os.path.dirname(modular_policy.__file__), 'real_world', 'robot_extrinsics')
BASE_IN_WORLD = {
    'left': np.load(os.path.join(_extrinsics_dir, 'left_base_pose_in_world.npy')),
    'right': np.load(os.path.join(_extrinsics_dir, 'right_base_pose_in_world.npy')),
}
ROBOT_IPS = {'left': '192.168.0.3', 'right': '192.168.0.2'}
HOME_POSES_WORLD = {
    'left': [0.30, -0.2, 0.25, -2.2192, 2.2148, 0.0091, 1],
    'right': [-0.1, -0.3, 0.25, 2.2419, -2.1984, 0.0166, 1],
}

INSTRUCTION = "pick up the building block"
CONTROL_HZ = 20
INTERP_MULT = 5       # interpolation multiplier: 20Hz × 5 = 100Hz servo
SERVO_HZ = CONTROL_HZ * INTERP_MULT  # 100Hz
Z_MIN_WORLD = 0.001793 - 0.02


# ── Math utilities ──────────────────────────────────────────────────

def base_to_world(pose_base, T_bw):
    p = list(pose_base)
    p[0] += T_bw[0, 3]; p[1] += T_bw[1, 3]; p[2] += T_bw[2, 3]
    return p


def world_to_base(pose_world, T_bw):
    p = list(pose_world)
    p[0] -= T_bw[0, 3]; p[1] -= T_bw[1, 3]; p[2] -= T_bw[2, 3]
    return p


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    a1 = d6[:3].astype(np.float64)
    a2 = d6[3:].astype(np.float64)
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=0)


def rot6d_to_axisangle(d6: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(rot6d_to_mat(d6)).as_rotvec().astype(np.float32)


def axisangle_to_rot6d(rx, ry, rz):
    R = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
    return R[:2, :].flatten().astype(np.float32)


def ee_pose_to_state10d(pose_6d, gripper: float) -> np.ndarray:
    x, y, z, rx, ry, rz = pose_6d[:6]
    rot6d = axisangle_to_rot6d(rx, ry, rz)
    return np.array([x, y, z, *rot6d, gripper], dtype=np.float32)


def action_10d_to_delta7d(action_10d: np.ndarray) -> np.ndarray:
    delta = np.zeros(7, dtype=np.float32)
    delta[:3] = action_10d[:3]
    delta[3:6] = rot6d_to_axisangle(action_10d[3:9])
    delta[6] = action_10d[9]
    return delta


def actions_10d_to_7d(actions_10d: np.ndarray) -> np.ndarray:
    """(T,10) actions -> (T,7) [dx,dy,dz,drx,dry,drz,gripper]."""
    T = actions_10d.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        out[t, :3] = actions_10d[t, :3]
        out[t, 3:6] = rot6d_to_axisangle(actions_10d[t, 3:9])
        out[t, 6] = actions_10d[t, 9]
    return out


def accumulate_deltas(current_pose_world, deltas_7d):
    """Accumulate delta actions on current pose -> absolute trajectory (T,7)."""
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


def _precise_wait(t_end: float, slack_time: float = 0.001):
    t_wait = t_end - time.monotonic()
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time.monotonic() < t_end:
            pass


# ── Interpolation (from step9) ──────────────────────────────────────

def interpolate_waypoints(start_pose, waypoints, mult):
    """Linearly interpolate between waypoints.

    start_pose: (6,) current pose
    waypoints:  (N, 6) target poses
    mult:       number of sub-steps between each waypoint

    Returns (N*mult, 6) interpolated poses.
    """
    all_poses = []
    prev = start_pose
    for wp in waypoints:
        for j in range(1, mult + 1):
            alpha = j / mult
            interp = prev + alpha * (wp - prev)
            all_poses.append(interp)
        prev = wp
    return np.array(all_poses, dtype=np.float64)


# ── Visualization (from step7) ──────────────────────────────────────

def visualize_step(exec_poses, new_poses, camera_image, current_ee,
                   n_exec, inference_delay, step_idx, save_path, instruction):
    """Visualize executing chunk and new RTC prediction, time-aligned.

    exec_poses:  (T_exec, 7) trajectory from the chunk executed this step
    new_poses:   (T_new, 7) trajectory from the new RTC prediction
    n_exec:      number of actions executed from exec_poses
    inference_delay: number of prefix timesteps in the new prediction

    Time alignment:
      - exec_poses is plotted at t = [0, 1, ..., T_exec-1] / 20Hz
      - new_poses is plotted at t = [n_exec, n_exec+1, ...] / 20Hz
        (because the new prediction starts where execution ended)
    """
    T_exec = exec_poses.shape[0]
    T_new = new_poses.shape[0]
    ts_exec = np.arange(T_exec) / 20.0
    ts_new = np.arange(T_new) / 20.0 + n_exec / 20.0

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, 2, figure=fig, hspace=0.15, wspace=0.30,
                  width_ratios=[1, 1.3])

    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(camera_image)
    ax_img.set_title("Camera (center-cropped)", fontsize=12, fontweight="bold")
    ax_img.axis("off")

    dims = [
        (0, "x (world, m)", "#e41a1c", current_ee[0]),
        (1, "y (world, m)", "#377eb8", current_ee[1]),
        (2, "z (world, m)", "#4daf4a", current_ee[2]),
        (6, "gripper",      "#ff7f00", None),
    ]

    axes = []
    for row, (dim_idx, label, color, start_val) in enumerate(dims):
        share = axes[0] if axes else None
        ax = fig.add_subplot(gs[row, 1], sharex=share)
        axes.append(ax)

        # --- Executing chunk ---
        exec_vals = exec_poses[:, dim_idx]
        # Executed portion (solid)
        ax.plot(ts_exec[:n_exec], exec_vals[:n_exec], "o-",
                markersize=5, linewidth=2.0, color=color,
                label="executed" if row == 0 else None)
        # Tail of executing chunk beyond n_exec (dotted, faded)
        if n_exec < T_exec:
            ax.plot(ts_exec[n_exec - 1:], exec_vals[n_exec - 1:], "o:",
                    markersize=3, linewidth=1.2, color=color, alpha=0.3,
                    label="prev tail" if row == 0 else None)

        # --- New RTC prediction (time-shifted) ---
        new_vals = new_poses[:, dim_idx]
        # Inpainting prefix portion (square markers, dashed)
        if inference_delay > 0 and inference_delay <= T_new:
            ax.plot(ts_new[:inference_delay], new_vals[:inference_delay], "s--",
                    markersize=4, linewidth=1.5, color=color, alpha=0.6,
                    label="RTC prefix" if row == 0 else None)
        # Free prediction portion
        free_start = max(0, inference_delay)
        if free_start < T_new:
            # Connect from prefix end (or execution end)
            connect = max(0, free_start - 1)
            ax.plot(ts_new[connect:], new_vals[connect:], "D--",
                    markersize=3, linewidth=1.5, color=color, alpha=0.5,
                    label="RTC new" if row == 0 else None)

        # Current EE position line
        if start_val is not None:
            ax.axhline(start_val, color=color, linewidth=1.0, linestyle="--",
                       alpha=0.4, label=f"EE={start_val:.4f}" if row == 0 else None)

        # Vertical line at execution boundary
        ax.axvline(ts_exec[n_exec - 1], color="black", linewidth=1.0,
                   linestyle=":", alpha=0.6,
                   label="exec boundary" if row == 0 else None)

        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        if row == 0:
            ax.legend(fontsize=7, loc="upper right", ncol=3)
        if row < len(dims) - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("Time (s) — 20Hz", fontsize=10)

    fig.suptitle(
        f'Step {step_idx} (DD-RTC-Smooth): exec={n_exec}, delay={inference_delay}\n'
        f'"{instruction}"',
        fontsize=13, fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ── Model loading ───────────────────────────────────────────────────

def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


def load_model(checkpoint_path: str):
    import torch

    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()

    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None
    config.framework.qwenvl.attn_implementation = _detect_attn_implementation()

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s ({config.framework.name})")
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    return model


def build_example(image: Image.Image, instruction: str,
                  state_10d: np.ndarray = None) -> dict:
    example = {"image": [image], "lang": instruction}
    if state_10d is not None:
        example["state"] = state_10d.reshape(1, -1)
    return example


# ── Background inference helper (from step7) ────────────────────────

class AsyncInference:
    """
    Runs camera capture + model inference in a background thread so that
    both overlap with action execution (real-time chunking).

    Records CUDA events for precise GPU timing without blocking the thread.
    Timing is resolved lazily in wait() after the thread signals completion.
    """

    def __init__(self):
        self._thread = None
        self._result = None
        self._error = None
        self._start_event = None
        self._end_event = None
        self._done = threading.Event()
        self._camera_image = None  # PIL image captured by the background thread
        self._obs_ms = 0.0         # time spent on camera capture

    def start(self, model, cam, instruction, state_10d,
              prev_normalized, inference_delay, kwargs):
        """Launch camera capture + inference in a background thread.

        The camera grab now happens inside the thread so it no longer
        blocks the main servo loop.
        """
        import torch
        self._result = None
        self._error = None
        self._camera_image = None
        self._obs_ms = 0.0
        # Create CUDA events on the main thread (same CUDA context)
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._done.clear()

        start_evt = self._start_event
        end_evt = self._end_event

        def _run():
            try:
                # ── Camera capture (overlaps with servo execution) ──
                t_obs = time.monotonic()
                pil_img = cam.grab_pil()
                while pil_img is None:
                    pil_img = cam.grab_pil()
                self._camera_image = pil_img
                self._obs_ms = (time.monotonic() - t_obs) * 1000

                example = build_example(pil_img, instruction, state_10d=state_10d)

                # ── Model inference ─────────────────────────────────
                start_evt.record()

                if prev_normalized is not None and inference_delay > 0:
                    out = model.predict_action_realtime(
                        examples=[example],
                        prev_action_chunk_normalized=prev_normalized,
                        inference_delay=inference_delay,
                        **kwargs,
                    )
                else:
                    out = model.predict_action(
                        examples=[example],
                        **kwargs,
                    )

                end_evt.record()
                # NO torch.cuda.synchronize() here — don't block the thread
                self._result = out
            except Exception as e:
                self._error = e
            finally:
                self._done.set()

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def wait(self, timeout=None):
        """Block until inference completes. Returns the result dict."""
        self._done.wait(timeout=timeout)
        if self._error is not None:
            raise self._error
        return self._result

    @property
    def camera_image(self):
        """PIL image captured by the background thread. Available after wait()."""
        return self._camera_image

    @property
    def obs_ms(self):
        """Time spent on camera capture (ms). Available after wait()."""
        return self._obs_ms

    @property
    def infer_ms(self):
        """Precise inference time (CUDA-event based). Call after wait().

        Calls torch.cuda.synchronize() to ensure the end event is resolved,
        then reads elapsed time from the GPU clock.
        """
        import torch
        if self._start_event is None or self._end_event is None:
            return 0.0
        self._end_event.synchronize()  # only sync the end event, not all CUDA work
        return self._start_event.elapsed_time(self._end_event)

    @property
    def is_done(self):
        return self._done.is_set()


# ── Stopwatch web server (SSE-based, daemon thread) ─────────────────

STOPWATCH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Policy Stopwatch</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;700&display=swap');

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0a0a0f;
    font-family: 'JetBrains Mono', monospace;
    user-select: none;
    overflow: hidden;
  }

  /* subtle animated gradient ring */
  .ring {
    position: absolute;
    width: 420px; height: 420px;
    border-radius: 50%;
    background: conic-gradient(from 0deg, #6366f1, #a855f7, #ec4899, #6366f1);
    opacity: 0.15;
    animation: spin 6s linear infinite;
    filter: blur(30px);
    transition: opacity 0.5s;
  }
  .ring.active { opacity: 0.35; }
  @keyframes spin { to { transform: rotate(360deg); } }

  .container {
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 28px;
  }

  #time {
    font-size: 6rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: #e2e8f0;
    text-shadow: 0 0 40px rgba(99,102,241,0.4);
    transition: text-shadow 0.3s;
  }
  #time.active {
    text-shadow: 0 0 60px rgba(168,85,247,0.6), 0 0 120px rgba(99,102,241,0.3);
  }

  #ms {
    font-size: 3rem;
    font-weight: 300;
    color: #94a3b8;
    margin-left: 4px;
  }
  #ms.active { color: #c4b5fd; }

  #status {
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #475569;
    transition: color 0.3s;
  }
  #status.active { color: #a78bfa; }

  .hint {
    position: fixed;
    bottom: 32px;
    font-size: 0.8rem;
    color: #334155;
    letter-spacing: 2px;
  }

  /* lap list */
  #laps {
    margin-top: 12px;
    max-height: 180px;
    overflow-y: auto;
    width: 360px;
    scrollbar-width: thin;
    scrollbar-color: #1e1e2e transparent;
  }
  .lap {
    display: flex;
    justify-content: space-between;
    padding: 6px 16px;
    font-size: 0.85rem;
    color: #64748b;
    border-bottom: 1px solid #1e1e2e;
  }
  .lap:first-child { color: #a78bfa; }

  /* ── Splash screen ── */
  .splash {
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background: #0a0a0f;
    z-index: 100;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.8s ease;
  }
  .splash.active {
    opacity: 1;
    pointer-events: auto;
  }
  .splash-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #e2e8f0;
    text-align: center;
    line-height: 1.4;
    letter-spacing: 3px;
    padding: 0 40px;
    animation: splash-glow 2.5s ease-in-out infinite;
  }
  @keyframes splash-glow {
    0%, 100% { text-shadow: 0 0 30px rgba(99,102,241,0.3), 0 0 60px rgba(168,85,247,0.15); }
    50%      { text-shadow: 0 0 60px rgba(99,102,241,0.6), 0 0 120px rgba(168,85,247,0.4), 0 0 180px rgba(236,72,153,0.15); }
  }
  .splash-sub {
    margin-top: 20px;
    font-size: 1rem;
    font-weight: 300;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #64748b;
  }
  .splash-bar-track {
    margin-top: 48px;
    width: 320px;
    height: 3px;
    background: #1e1e2e;
    border-radius: 2px;
    overflow: hidden;
  }
  .splash-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
    border-radius: 2px;
  }
</style>
</head>
<body>

<!-- Splash overlay -->
<div class="splash" id="splash">
  <div class="splash-title" id="splashTitle"></div>
  <div class="splash-sub" id="splashSub">starting in 5s</div>
  <div class="splash-bar-track">
    <div class="splash-bar" id="splashBar"></div>
  </div>
</div>

<div class="ring" id="ring"></div>

<div class="container">
  <div id="status">Waiting for policy</div>
  <div>
    <span id="time">00:00</span><span id="ms">.000</span>
  </div>
  <div id="laps"></div>
</div>

<div class="hint">auto-controlled by robot policy via SSE</div>

<script>
  const timeEl     = document.getElementById('time');
  const msEl       = document.getElementById('ms');
  const statusEl   = document.getElementById('status');
  const ringEl     = document.getElementById('ring');
  const lapsEl     = document.getElementById('laps');
  const splashEl   = document.getElementById('splash');
  const splashTitle= document.getElementById('splashTitle');
  const splashSub  = document.getElementById('splashSub');
  const splashBar  = document.getElementById('splashBar');

  let running   = false;
  let startTs   = 0;
  let elapsed   = 0;
  let rafId     = null;
  let lapCount  = 0;
  let splashRaf = null;

  function fmt(totalMs) {
    const mins = Math.floor(totalMs / 60000);
    const secs = Math.floor((totalMs % 60000) / 1000);
    const ms   = Math.floor(totalMs % 1000);
    return {
      main: String(mins).padStart(2,'0') + ':' + String(secs).padStart(2,'0'),
      sub:  '.' + String(ms).padStart(3,'0')
    };
  }

  function render(totalMs) {
    const {main, sub} = fmt(totalMs);
    timeEl.textContent = main;
    msEl.textContent   = sub;
  }

  function tick() {
    const now = performance.now();
    render(elapsed + (now - startTs));
    rafId = requestAnimationFrame(tick);
  }

  function showSplash(title, durationMs) {
    splashTitle.textContent = title;
    splashEl.classList.add('active');

    const t0 = performance.now();
    function animateBar() {
      const pct = Math.min((performance.now() - t0) / durationMs, 1);
      splashBar.style.width = (pct * 100) + '%';
      const remaining = Math.ceil((durationMs - (performance.now() - t0)) / 1000);
      splashSub.textContent = remaining > 0 ? 'starting in ' + remaining + 's' : 'go';
      if (pct < 1) splashRaf = requestAnimationFrame(animateBar);
    }
    splashRaf = requestAnimationFrame(animateBar);
  }

  function hideSplash() {
    if (splashRaf) cancelAnimationFrame(splashRaf);
    splashEl.classList.remove('active');
  }

  function start() {
    if (running) return;
    hideSplash();
    running = true;
    elapsed = 0;
    startTs = performance.now();
    statusEl.textContent = 'Running';
    statusEl.classList.add('active');
    timeEl.classList.add('active');
    msEl.classList.add('active');
    ringEl.classList.add('active');
    tick();
  }

  function stop() {
    if (!running) return;
    running = false;
    elapsed += performance.now() - startTs;
    cancelAnimationFrame(rafId);
    statusEl.textContent = 'Finished';
    statusEl.classList.remove('active');
    timeEl.classList.remove('active');
    msEl.classList.remove('active');
    ringEl.classList.remove('active');

    lapCount++;
    const div = document.createElement('div');
    div.className = 'lap';
    const {main, sub} = fmt(elapsed);
    div.innerHTML = '<span>#' + lapCount + '</span><span>' + main + sub + '</span>';
    lapsEl.prepend(div);
  }

  // SSE: listen for events from the robot policy server
  const evtSource = new EventSource('/events');
  evtSource.addEventListener('splash', function(e) {
    // data format: "title|durationMs"
    const parts = e.data.split('|');
    const title = parts[0];
    const dur   = parseInt(parts[1]) || 5000;
    showSplash(title, dur);
  });
  evtSource.addEventListener('timer', function(e) {
    if (e.data === 'start') start();
    else if (e.data === 'stop') stop();
  });
  evtSource.onerror = function() {
    statusEl.textContent = 'Disconnected';
  };

  // Manual click still works as override
  document.body.addEventListener('click', () => {
    if (splashEl.classList.contains('active')) { hideSplash(); start(); return; }
    if (running) stop(); else start();
  });
  document.body.addEventListener('dblclick', (e) => {
    e.preventDefault();
    hideSplash();
    running = false;
    elapsed = 0;
    lapCount = 0;
    cancelAnimationFrame(rafId);
    render(0);
    statusEl.textContent = 'Waiting for policy';
    statusEl.classList.remove('active');
    timeEl.classList.remove('active');
    msEl.classList.remove('active');
    ringEl.classList.remove('active');
    lapsEl.innerHTML = '';
  });
</script>
</body>
</html>
"""


class StopwatchServer:
    """Embedded HTTP + SSE server for the stopwatch. Runs in a daemon thread."""

    def __init__(self, port=8765):
        self._port = port
        self._sse_queues = []       # one queue per connected SSE client
        self._lock = threading.Lock()
        self._server = None

    def _broadcast(self, event: str, data: str):
        """Non-blocking: push (event, data) to all connected SSE clients."""
        with self._lock:
            dead = []
            for q in self._sse_queues:
                try:
                    q.put_nowait((event, data))
                except queue.Full:
                    dead.append(q)
            for q in dead:
                self._sse_queues.remove(q)

    def start_timer(self, splash_title=None, splash_duration=5):
        """Optionally show splash for splash_duration seconds, then start timer."""
        if splash_title:
            self._broadcast("splash", f"{splash_title}|{splash_duration * 1000}")
            time.sleep(splash_duration)
        self._broadcast("timer", "start")

    def stop_timer(self):
        self._broadcast("timer", "stop")

    def serve(self, open_browser=True):
        """Start the HTTP server in a daemon thread and optionally open browser."""
        parent = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/events":
                    self._handle_sse()
                else:
                    self._handle_page()

            def _handle_page(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(STOPWATCH_HTML.encode())

            def _handle_sse(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "keep-alive")
                self.end_headers()

                q = queue.Queue(maxsize=64)
                with parent._lock:
                    parent._sse_queues.append(q)
                try:
                    while True:
                        try:
                            event, data = q.get(timeout=15)
                            self.wfile.write(f"event: {event}\ndata: {data}\n\n".encode())
                            self.wfile.flush()
                        except queue.Empty:
                            # Send keepalive comment to prevent timeout
                            self.wfile.write(b": keepalive\n\n")
                            self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, OSError):
                    pass
                finally:
                    with parent._lock:
                        if q in parent._sse_queues:
                            parent._sse_queues.remove(q)

            def log_message(self, *args):
                pass  # silence HTTP logs

        # Try ports starting from self._port, auto-increment on conflict
        for port in range(self._port, self._port + 100):
            try:
                self._server = http.server.HTTPServer(("", port), Handler)
                break
            except OSError:
                continue
        else:
            raise RuntimeError(f"Could not find a free port in range {self._port}-{self._port + 99}")
        self._port = port
        self._server.daemon_threads = True

        t = threading.Thread(target=self._server.serve_forever, daemon=True)
        t.start()

        url = f"http://localhost:{self._port}"
        print(f"Stopwatch server running at {url}")

        if open_browser:
            threading.Timer(0.3, lambda: webbrowser.open(url)).start()

    def shutdown(self):
        if self._server:
            self._server.shutdown()


# ── Go home ─────────────────────────────────────────────────────────

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
    gripper = RobotiqGripper()
    gripper.connect(hostname=robot_ip, port=63352)
    pos = int((1.0 - home[6]) * 255)
    gripper.move(pos, 255, 150)
    gripper.disconnect()
    print("Gripper opened.")
    return home[6]


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop control with RTC + smooth interpolated servoL + website stopwatch")
    parser.add_argument(
        "--checkpoint", type=str,
        default="checkpoints/DiscreteRTC/"
                "fastumi_pickandplace_discrete_diffusion_real_0314_no_state_fixgripper/"
                "checkpoints/steps_25000_pytorch_model.pt",
    )
    parser.add_argument("--arm", choices=["left", "right"], default="left")
    parser.add_argument("--camera_dev", type=int, default=0)
    parser.add_argument("--n_actions", type=int, default=8,
                        help="Number of actions to execute per chunk (= execute_horizon)")
    parser.add_argument("--inference_delay", type=int, default=8,
                        help="Number of prefix timesteps for RTC decode. "
                             "-1 = same as n_actions (default)")
    parser.add_argument("--max_steps", type=int, default=0,
                        help="Max inference steps (0=unlimited, Ctrl+C to stop)")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--no_go_home", action="store_true", default=False)
    parser.add_argument("--instruction", type=str, default=INSTRUCTION)
    parser.add_argument("--decode_temperature", type=float, default=0.0)
    parser.add_argument("--choice_temperature", type=float, default=0.1)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--fixed_steps", action="store_true", default=True,
                        help="Use fixed num_inference_steps for RTC")
    parser.add_argument("--no_fixed_steps", action="store_true", default=False,
                        help="Scale num_inference_steps proportionally to inference_delay")
    parser.add_argument("--fix_rotation", action="store_true", default=True)
    parser.add_argument("--no_fix_rotation", action="store_true", default=False)
    parser.add_argument("--stop_when_grasping_high_enough", action="store_true", default=True)
    parser.add_argument("--no_stop_when_grasping_high_enough", action="store_true", default=False)
    parser.add_argument("--grasp_lift_threshold", type=float, default=0.04,
                        help="Min lift above Z_MIN_WORLD to consider 'high enough' (m)")
    # Rollout saving
    parser.add_argument("--save_rollout", action="store_true", default=False)
    parser.add_argument("--no_save_rollout", action="store_true", default=False)
    parser.add_argument("--rollout_dir", type=str, default=None)
    # Stopwatch
    parser.add_argument("--stopwatch_port", type=int, default=8765)
    args = parser.parse_args()
    if args.no_stop_when_grasping_high_enough:
        args.stop_when_grasping_high_enough = False
    if args.no_save_rollout:
        args.save_rollout = False
    if args.no_fix_rotation:
        args.fix_rotation = False
    if args.no_fixed_steps:
        args.fixed_steps = False
    if args.inference_delay < 0:
        args.inference_delay = args.n_actions

    # ── Start stopwatch server (early, for debug visibility) ────────
    stopwatch = StopwatchServer(port=args.stopwatch_port)
    stopwatch.serve(open_browser=True)

    robot_ip = ROBOT_IPS[args.arm]
    T_bw = BASE_IN_WORLD[args.arm]
    servo_dt = 1.0 / SERVO_HZ

    # ── Load model ───────────────────────────────────────────────────
    model = load_model(args.checkpoint)
    norm_stats = model.norm_stats
    dataset_key = list(norm_stats.keys())[0]
    action_stats = norm_stats[dataset_key]["action"]
    print(f"Norm stats: dataset='{dataset_key}', modes={action_stats.get('norm_modes', 'legacy')}")

    # ── Connect to robot ─────────────────────────────────────────────
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface

    print(f"\nConnecting to {args.arm} arm at {robot_ip}...")
    rtde_c = RTDEControlInterface(robot_ip)
    rtde_r = RTDEReceiveInterface(robot_ip)

    # ── Connect gripper ──────────────────────────────────────────────
    from robotiq_gripper import RobotiqGripper
    print("Connecting to gripper...")
    gripper_hw = RobotiqGripper()
    gripper_hw.connect(hostname=robot_ip, port=63352)
    current_gripper = 0.0

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
        current_gripper = 0.0

    # ── Print config ─────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Step 10: Closed-loop (DD + RTC + smooth {SERVO_HZ}Hz)")
    print(f"  Arm:                {args.arm}")
    print(f"  Instruction:        {args.instruction}")
    print(f"  Actions/step:       {args.n_actions}")
    print(f"  Inference delay:    {args.inference_delay}")
    print(f"  Waypoints:          {CONTROL_HZ}Hz × {INTERP_MULT} = {SERVO_HZ}Hz servo")
    print(f"  servoL params:      lookahead=0.2, gain=200")
    print(f"  Z safety:           clamp to {Z_MIN_WORLD:.4f}m")
    print(f"  Include state:      {args.include_state}")
    print(f"  decode_temperature: {args.decode_temperature}")
    print(f"  choice_temperature: {args.choice_temperature}")
    print(f"  use_simple_max:     {args.use_simple_max}")
    print(f"  fixed_steps:        {args.fixed_steps}")
    print(f"  fix_rotation:       {args.fix_rotation}")
    print(f"  stop_when_grasping: {args.stop_when_grasping_high_enough} (lift>={args.grasp_lift_threshold:.2f}m)")
    print(f"  stopwatch:          http://localhost:{args.stopwatch_port}")
    print(f"{'=' * 60}")

    # ── Setup rollout saving ─────────────────────────────────────────
    rollout_log = []
    rollout_dir = None
    if args.save_rollout:
        if args.rollout_dir:
            rollout_dir = Path(args.rollout_dir)
        else:
            ts = time.strftime("%Y%m%d_%H%M%S")
            rollout_dir = Path("ur5") / "rollouts" / f"dd_rtc_smooth_{ts}"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        (rollout_dir / "images").mkdir(exist_ok=True)
        print(f"\n  Rollout saving: {rollout_dir}")

        run_config = {
            "checkpoint": args.checkpoint,
            "arm": args.arm,
            "instruction": args.instruction,
            "n_actions": args.n_actions,
            "inference_delay": args.inference_delay,
            "max_steps": args.max_steps,
            "include_state": args.include_state,
            "control_hz": CONTROL_HZ,
            "servo_hz": SERVO_HZ,
            "interp_mult": INTERP_MULT,
            "decode_temperature": args.decode_temperature,
            "choice_temperature": args.choice_temperature,
            "use_simple_max": args.use_simple_max,
            "dataset_key": dataset_key,
            "norm_modes": action_stats.get("norm_modes", "legacy"),
            "method": "discrete_rtc_smooth",
        }
        with open(rollout_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

    # Single-worker pool for non-blocking viz saves
    viz_executor = ThreadPoolExecutor(max_workers=1)

    input("\n>>> Press Enter to START (Ctrl+C to abort) <<<")

    # ── Inference kwargs (shared) ────────────────────────────────────
    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
        fixed_steps=args.fixed_steps,
    )

    # ═════════════════════════════════════════════════════════════════
    #  Step 0: Initial inference (synchronous, no prefix available)
    # ═════════════════════════════════════════════════════════════════
    pose_base = rtde_r.getActualTCPPose()
    pose_world = base_to_world(pose_base, T_bw)
    current_state_10d = ee_pose_to_state10d(pose_world, current_gripper)

    pil_img = cam.grab_pil()
    while pil_img is None:
        pil_img = cam.grab_pil()

    state_for_model = current_state_10d if args.include_state else None
    example = build_example(pil_img, args.instruction, state_10d=state_for_model)

    t_infer = time.monotonic()
    output = model.predict_action(examples=[example], **infer_kwargs)
    init_infer_ms = (time.monotonic() - t_infer) * 1000
    print(f"[init]  inference={init_infer_ms:.0f}ms (synchronous, no prefix)")

    current_normalized = output["normalized_actions"][0].astype(np.float32)
    current_actions_10d = baseframework.unnormalize_actions(current_normalized, action_stats)

    async_infer = AsyncInference()

    # ── Start stopwatch (splash + timer) ────────────────────────────
    stopwatch.start_timer(splash_title="Discrete Diffusion\nRTC Policy")
    print("[STOPWATCH] Started")

    # ── Control loop ─────────────────────────────────────────────────
    step = 0
    visited_near_table = False  # True once z has been within 5cm of Z_MIN_WORLD
    # Cumulative timing accumulators (ms)
    total_observation_ms = 0.0
    total_action_gen_ms = 0.0
    total_sending_ms = 0.0
    total_loop_ms = 0.0
    total_wall_ms = 0.0
    total_extra_wait_ms = 0.0
    n_extra_waits = 0
    try:
        while args.max_steps == 0 or step < args.max_steps:
            loop_t0 = time.monotonic()
            wall_time = time.time()

            # 1. Read current robot state (fast, ~1ms — stays on main thread)
            pose_base = rtde_r.getActualTCPPose()
            pose_world = base_to_world(pose_base, T_bw)
            current_state_10d = ee_pose_to_state10d(pose_world, current_gripper)
            current_pos = np.array(pose_world, dtype=np.float64)

            # 1b. Track if robot has been near the table, then check lift
            if not visited_near_table and (pose_world[2] - Z_MIN_WORLD) < 0.05:
                visited_near_table = True
                print(f"  [INFO] Visited near table (z={pose_world[2]:.4f}), lift check now armed")

            if args.stop_when_grasping_high_enough and visited_near_table and current_gripper > 0.5:
                lift = pose_world[2] - Z_MIN_WORLD
                if lift >= args.grasp_lift_threshold:
                    print(f"\n[DONE] Grasped and lifted {lift:.3f}m >= {args.grasp_lift_threshold:.2f}m threshold. Stopping.")
                    break

            # 2. Build the shifted prefix for RTC
            #    Save the current chunk before it gets swapped out
            executing_normalized = current_normalized.copy()
            n_exec = min(args.n_actions, len(current_actions_10d))
            chunk_len = current_normalized.shape[0]
            shift = n_exec
            if shift < chunk_len:
                shifted_normalized = np.zeros_like(current_normalized)
                remaining = chunk_len - shift
                shifted_normalized[:remaining] = current_normalized[shift:]
            else:
                shifted_normalized = np.zeros_like(current_normalized)

            actual_delay = min(args.inference_delay, chunk_len - shift) if shift < chunk_len else 0

            # 3. Launch background camera capture + inference (RTC)
            #    Camera grab now happens inside the async thread so it
            #    overlaps with action execution instead of blocking here.
            state_for_model = current_state_10d if args.include_state else None
            prev_norm_batch = shifted_normalized[np.newaxis, ...] if actual_delay > 0 else None

            t_infer_start = time.monotonic()
            async_infer.start(
                model, cam, args.instruction, state_for_model,
                prev_normalized=prev_norm_batch,
                inference_delay=actual_delay,
                kwargs=infer_kwargs,
            )

            # 4. Convert current chunk to absolute world-frame waypoints (fast, main thread)
            t_send_start = time.monotonic()
            waypoints = np.zeros((n_exec, 6), dtype=np.float64)
            pos = current_pos.copy()
            executed_targets = []

            for i in range(n_exec):
                delta = action_10d_to_delta7d(current_actions_10d[i])
                if args.fix_rotation:
                    delta[3:6] = 0.0
                pos = pos + delta[:6]

                # Safety: clamp z
                if pos[2] < Z_MIN_WORLD:
                    print(f"  [SAFETY] z clamped: {pos[2]:.4f} → {Z_MIN_WORLD:.4f}")
                    pos[2] = Z_MIN_WORLD

                waypoints[i] = pos
                executed_targets.append(pos.tolist())

                # Handle gripper on first transition
                new_gripper = float(delta[6])
                if (new_gripper > 0.5) != (current_gripper > 0.5):
                    grip_pos = int(new_gripper * 255)
                    label = "CLOSE" if new_gripper > 0.5 else "OPEN"
                    print(f"  Gripper -> {label} (pos={grip_pos})")
                    gripper_hw.move(grip_pos, 255, 150)
                    current_gripper = new_gripper

            # 5. Interpolate and execute at 100Hz (while camera+inference run in background)
            interp_poses = interpolate_waypoints(current_pos, waypoints, INTERP_MULT)

            t_exec_start = time.monotonic()
            for i, pose_world_i in enumerate(interp_poses):
                target_base = world_to_base(pose_world_i.tolist(), T_bw)
                rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)
                _precise_wait(t_exec_start + (i + 1) * servo_dt)

            rtde_c.servoStop()
            exec_ms = (time.monotonic() - t_exec_start) * 1000
            send_ms = (time.monotonic() - t_send_start) * 1000

            # 6b. Wait for background camera+inference to complete
            extra_wait_ms = 0.0
            if not async_infer.is_done:
                t_wait_start = time.monotonic()
                next_output = async_infer.wait(timeout=10.0)
                extra_wait_ms = (time.monotonic() - t_wait_start) * 1000
                n_extra_waits += 1
                total_extra_wait_ms += extra_wait_ms
            else:
                next_output = async_infer.wait(timeout=10.0)

            # Retrieve camera image and obs timing from the async thread
            pil_img = async_infer.camera_image
            obs_ms = async_infer.obs_ms
            infer_ms = async_infer.infer_ms  # precise CUDA-synced time
            wait_ms = (time.monotonic() - t_infer_start) * 1000  # wall time including exec overlap
            infer_hidden = max(0, wait_ms - exec_ms)  # extra wait beyond execution

            if next_output is None:
                print("[ERROR] Inference timed out, reusing current chunk")
            else:
                current_normalized = next_output["normalized_actions"][0].astype(np.float32)
                current_actions_10d = baseframework.unnormalize_actions(
                    current_normalized, action_stats
                )

            wall_ms = (time.monotonic() - loop_t0) * 1000

            # Accumulate timing
            step_total_ms = obs_ms + infer_ms + send_ms
            total_observation_ms += obs_ms
            total_action_gen_ms += infer_ms
            total_sending_ms += send_ms
            total_loop_ms += step_total_ms
            total_wall_ms += wall_ms

            # Action magnitude for this chunk
            deltas_3d = np.array([action_10d_to_delta7d(current_actions_10d[i])[:3]
                                  for i in range(n_exec)])
            max_delta = np.abs(deltas_3d).max()
            total_disp = np.linalg.norm(deltas_3d.sum(axis=0))
            z_above_table = pose_world[2] - Z_MIN_WORLD

            step_hz = 1000.0 / wall_ms if wall_ms > 0 else 0.0
            wait_tag = f"  WAIT={extra_wait_ms:.0f}ms" if extra_wait_ms > 0 else ""

            print(f"[step {step:4d}]  wall={wall_ms:5.0f}ms ({step_hz:4.1f}Hz)  "
                  f"obs={obs_ms:5.0f}ms  action_gen={infer_ms:5.0f}ms  "
                  f"send={send_ms:5.0f}ms  "
                  f"delay={actual_delay}  "
                  f"pos=[{pose_world[0]:.3f}, {pose_world[1]:.3f}, {pose_world[2]:.3f}]  "
                  f"z_above={z_above_table:.3f}m  "
                  f"disp={total_disp:.4f}m  max_d={max_delta:.4f}m  "
                  f"grip={'C' if current_gripper > 0.5 else 'O'}"
                  f"{wait_tag}")

            # ── Save rollout step ────────────────────────────────────
            if args.save_rollout and rollout_dir is not None:
                # Executing chunk trajectory (the chunk we just ran)
                exec_actions_10d = baseframework.unnormalize_actions(
                    executing_normalized, action_stats)
                exec_deltas_7d = actions_10d_to_7d(exec_actions_10d)
                exec_poses_all = accumulate_deltas(pose_world, exec_deltas_7d)

                # New RTC prediction trajectory
                new_deltas_7d = actions_10d_to_7d(current_actions_10d)
                # The new prediction starts from where execution ended
                exec_end_pos = list(executed_targets[-1]) + list(pose_world[3:6])
                new_poses_all = accumulate_deltas(exec_end_pos, new_deltas_7d)

                # Offload viz to background thread (non-blocking)
                viz_path = str(rollout_dir / "images" / f"step_{step:04d}_viz.png")
                viz_executor.submit(
                    visualize_step,
                    exec_poses_all.copy(), new_poses_all.copy(),
                    pil_img.copy(), list(pose_world),
                    n_exec, actual_delay, step, viz_path, args.instruction,
                )

                step_data = {
                    "step": step,
                    "wall_time": wall_time,
                    "ee_pose_base": list(pose_base),
                    "ee_pose_world": list(pose_world),
                    "gripper_state": current_gripper,
                    "state_10d": current_state_10d.tolist(),
                    "executing_chunk_normalized": executing_normalized.tolist(),
                    "inpainting_prefix_normalized": shifted_normalized.tolist(),
                    "pred_normalized": current_normalized.tolist(),
                    "pred_actions_10d_denorm": current_actions_10d.tolist(),
                    "exec_trajectory_world": exec_poses_all.tolist(),
                    "new_trajectory_world": new_poses_all.tolist(),
                    "n_actions_executed": n_exec,
                    "executed_targets_world": executed_targets,
                    "inference_delay": actual_delay,
                    "obs_ms": round(obs_ms, 1),
                    "infer_ms": round(infer_ms, 1),
                    "send_ms": round(send_ms, 1),
                    "exec_ms": round(exec_ms, 1),
                    "extra_wait_ms": round(extra_wait_ms, 1),
                    "step_total_ms": round(step_total_ms, 1),
                    "wall_ms": round(wall_ms, 1),
                    "viz_file": f"images/step_{step:04d}_viz.png",
                }
                rollout_log.append(step_data)

            step += 1

    except KeyboardInterrupt:
        print("\n\nStopped by user (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        # ── Stop stopwatch (policy execution ended) ──────────────────
        stopwatch.stop_timer()
        print("[STOPWATCH] Stopped")

        print("Cleaning up...")
        try: rtde_c.servoStop()
        except Exception: pass
        try: rtde_c.stopScript()
        except Exception: pass
        try: gripper_hw.disconnect()
        except Exception: pass
        cam.close()

        # ── Wait for pending viz saves, then save rollout log ────────
        print("Waiting for pending viz saves...")
        viz_executor.shutdown(wait=True)

        if args.save_rollout and rollout_dir is not None and rollout_log:
            rollout_path = rollout_dir / "rollout.json"
            with open(rollout_path, "w") as f:
                json.dump(rollout_log, f, indent=2)
            print(f"Rollout saved: {rollout_dir}")
            print(f"  {len(rollout_log)} steps")
            print(f"  rollout.json + config.json + images/step_XXXX_viz.png")

        # ── Timing summary ─────────────────────────────────────────
        if step > 0:
            avg_wall = total_wall_ms / step
            avg_hz = 1000.0 / avg_wall if avg_wall > 0 else 0.0
            print(f"\n{'=' * 60}")
            print(f"  Timing Summary ({step} steps)")
            print(f"{'=' * 60}")
            print(f"  Avg wall-clock per step: {avg_wall:6.1f}ms  ({avg_hz:.1f}Hz)")
            print(f"  ├─ Observation time:     {total_observation_ms / step:6.1f}ms")
            print(f"  ├─ Action generation:    {total_action_gen_ms / step:6.1f}ms")
            print(f"  ├─ Sending action:       {total_sending_ms / step:6.1f}ms")
            print(f"  └─ Sum (obs+gen+send):   {total_loop_ms / step:6.1f}ms")
            if n_extra_waits > 0:
                print(f"  Extra waits:  {n_extra_waits}/{step} steps "
                      f"(avg {total_extra_wait_ms / n_extra_waits:.0f}ms when waiting)")
            else:
                print(f"  Extra waits:  0/{step} steps (inference always hidden)")
            print(f"{'=' * 60}")
        print(f"Done. Executed {step} inference steps.")


if __name__ == "__main__":
    main()
