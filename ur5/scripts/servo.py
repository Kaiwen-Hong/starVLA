"""Continuous 100Hz servo runner with global action counter."""

import collections
import threading
import time

import numpy as np

from scripts.robo_utils import (
    INTERP_MULT, SERVO_HZ,
    world_to_base, interpolate_waypoints, precise_wait,
)


class ServoRunner:
    """Daemon thread that continuously sends servoL at 100Hz from a waypoint buffer.

    The thread never stops between chunks — it just consumes poses from a deque.
    When the buffer is empty, it holds the last pose (no servoStop).
    New waypoints are appended by the main thread whenever inference completes.

    Tracks a global 20Hz action counter (action_t) that increments by 1 every
    INTERP_MULT (5) poses consumed.  The main thread can call
    wait_for_action_t(target) to block until the counter reaches a boundary,
    enabling fixed-cadence inference scheduling.

    Gripper transitions are posted via push_waypoints() and executed in the servo
    thread to avoid blocking the main thread.
    """

    def __init__(self, rtde_c, T_bw, gripper_hw):
        self._rtde_c = rtde_c
        self._T_bw = T_bw
        self._gripper_hw = gripper_hw
        self._buffer = collections.deque()       # (pose_world_6d,) at 100Hz
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_pose = None                   # last sent pose (world frame)
        self._gripper_cmd = None                 # pending gripper pos (0-255) or None
        self._gripper_lock = threading.Lock()
        self._current_gripper = 0.0              # last known gripper state (0 or 1 scale)
        self._starve_count = 0                   # how many ticks the buffer was empty
        # 20Hz action counter with condition variable for blocking waits
        self._action_t = 0
        self._sub_step = 0                       # 100Hz ticks within current action
        self._action_cond = threading.Condition()

    def start(self, initial_pose_world):
        """Start the servo thread. initial_pose_world: (6,) current EE pose."""
        self._last_pose = np.array(initial_pose_world[:6], dtype=np.float64)
        self._running = True
        self._action_t = 0
        self._sub_step = 0
        self._starve_count = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the servo thread and call servoStop."""
        self._running = False
        # Wake up anyone waiting so they don't hang
        with self._action_cond:
            self._action_cond.notify_all()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self._rtde_c.servoStop()
        except Exception:
            pass

    def push_waypoints(self, current_pos, waypoints_20hz, gripper_transitions=None):
        """Interpolate 20Hz waypoints to 100Hz and append to the buffer.

        current_pos: (6,) start pose for interpolation (typically the last
                     pose the servo thread consumed, or current EE pose).
        waypoints_20hz: (N, 6) absolute world-frame waypoints at 20Hz.
        gripper_transitions: [(index, value), ...] from compute_waypoints.
        """
        interp = interpolate_waypoints(current_pos, waypoints_20hz, INTERP_MULT)
        with self._lock:
            self._buffer.extend(interp)

        # Handle gripper asynchronously
        if gripper_transitions:
            for _, new_val in gripper_transitions:
                if (new_val > 0.5) != (self._current_gripper > 0.5):
                    self._current_gripper = new_val
                    with self._gripper_lock:
                        self._gripper_cmd = int(new_val * 255)

    @property
    def buffer_len(self):
        with self._lock:
            return len(self._buffer)

    @property
    def last_pose(self):
        return self._last_pose.copy() if self._last_pose is not None else None

    @property
    def current_gripper(self):
        return self._current_gripper

    @current_gripper.setter
    def current_gripper(self, val):
        self._current_gripper = val

    @property
    def action_t(self):
        """Global 20Hz timestep: how many 20Hz actions the servo has consumed."""
        return self._action_t

    def wait_for_action_t(self, target_t, timeout=30.0):
        """Block until action_t >= target_t. Returns True if reached, False on timeout."""
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
            # Check for pending gripper command (non-blocking)
            with self._gripper_lock:
                grip_cmd = self._gripper_cmd
                self._gripper_cmd = None
            if grip_cmd is not None:
                label = "CLOSE" if grip_cmd > 127 else "OPEN"
                print(f"  Gripper -> {label} (pos={grip_cmd})")
                self._gripper_hw.move(grip_cmd, 255, 150)

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

            # Always send a servoL (either new pose or hold last)
            if self._last_pose is not None:
                target_base = world_to_base(self._last_pose.tolist(), self._T_bw)
                self._rtde_c.servoL(target_base, 0, 0, servo_dt, 0.2, 200)

            precise_wait(t_next)
            t_next += servo_dt
