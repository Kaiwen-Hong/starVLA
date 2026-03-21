"""Continuous 100Hz servo runner v2.

Fixes over servo.py:
  1. Gripper commands run on a separate daemon thread (non-blocking servo loop).
  2. tail_pose property: last pose in the buffer for seamless waypoint splicing.
  3. Timing reset after large delays to prevent burst servoL catch-up.
"""

import collections
import threading
import time

import numpy as np

from scripts.robo_utils import (
    INTERP_MULT, SERVO_HZ,
    world_to_base, interpolate_waypoints, precise_wait,
)


class ServoRunner:
    """100Hz servo with non-blocking gripper, tail_pose, and timing guard."""

    def __init__(self, rtde_c, T_bw, gripper_hw):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._gripper_hw = gripper_hw
        self._buffer = collections.deque()       # 100Hz poses (world frame)
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pose = None                   # last consumed pose
        self._current_gripper = 0.0
        self._starve_count = 0
        # 20Hz action counter
        self._action_t = 0
        self._sub_step = 0
        self._action_cond = threading.Condition()
        # Gripper thread state
        self._pending_grip = None                # (pos_int, label) or None
        self._grip_lock = threading.Lock()
        self._grip_event = threading.Event()
        self._grip_thread = None

    # ── lifecycle ────────────────────────────────────────────────────

    def start(self, initial_pose_world):
        """Start servo + gripper threads."""
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

    # ── waypoint buffer ──────────────────────────────────────────────

    def push_waypoints(self, current_pos, waypoints_20hz, gripper_transitions=None):
        """Interpolate 20Hz waypoints to 100Hz and append to the buffer."""
        interp = interpolate_waypoints(current_pos, waypoints_20hz, INTERP_MULT)
        with self._lock:
            self._buffer.extend(interp)

        # Post gripper command (non-blocking)
        if gripper_transitions:
            for _, new_val in gripper_transitions:
                if (new_val > 0.5) != (self._current_gripper > 0.5):
                    self._current_gripper = new_val
                    grip_pos = int(new_val * 255)
                    label = "CLOSE" if grip_pos > 127 else "OPEN"
                    with self._grip_lock:
                        self._pending_grip = (grip_pos, label)
                    self._grip_event.set()

    # ── properties ───────────────────────────────────────────────────

    @property
    def buffer_len(self):
        with self._lock:
            return len(self._buffer)

    @property
    def last_pose(self):
        """Last pose consumed by servo (current robot position)."""
        return self._last_pose.copy() if self._last_pose is not None else None

    @property
    def tail_pose(self):
        """Last pose in the buffer (future position when buffer drains).

        Use this instead of last_pose when computing new waypoints to avoid
        spatial discontinuity at buffer splice points.
        """
        with self._lock:
            if self._buffer:
                return np.array(self._buffer[-1], dtype=np.float64)
        return self._last_pose.copy() if self._last_pose is not None else None

    @property
    def current_gripper(self):
        return self._current_gripper

    @current_gripper.setter
    def current_gripper(self, val):
        self._current_gripper = val

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

    # ── servo thread ─────────────────────────────────────────────────

    def _loop(self):
        servo_dt = 1.0 / SERVO_HZ
        t_next = time.monotonic() + servo_dt

        while self._running:
            # Pop next pose or hold last
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

            # Always send servoL (new pose or hold)
            if self._last_pose is not None:
                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt

            # Guard: if we fell more than 1 tick behind, reset to avoid
            # burst catch-up (e.g. after GIL contention or OS hiccup).
            now = time.monotonic()
            if t_next < now - servo_dt:
                t_next = now + servo_dt

    # ── gripper thread ───────────────────────────────────────────────

    def _gripper_loop(self):
        """Separate thread for blocking gripper commands.

        Deduplicates: only executes if the command differs from the last one.
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

            if cmd is not None and cmd[0] != last_pos:
                grip_pos, label = cmd
                print(f"  Gripper -> {label} (pos={grip_pos})")
                self._gripper_hw.move(grip_pos, 255, 150)
                last_pos = grip_pos
