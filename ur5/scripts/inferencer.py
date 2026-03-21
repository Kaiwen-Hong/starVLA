"""Self-looping cadence-aware inferencer (follows Algorithm 1 from RTC paper).

The inferencer runs a persistent worker thread that autonomously loops:
  1. Wait for servo.action_t to reach the next n_actions boundary.
  2. Build RTC prefix from current chunk (shifted by actions consumed).
  3. Camera capture + model inference.
  4. Immediately push free actions into the servo buffer.
  5. Use the new prediction as current chunk for the next cycle.
  6. Post result for the main loop to read (logging / stop check only).

The main loop only needs to:
  - start(initial_normalized, chunk_len)  once to kick off the loop.
  - poll_result()  non-blocking, for logging and stop conditions.
  - stop()  to shut down.
"""

import time
import threading
import queue

import numpy as np

from starVLA.model.framework.base_framework import baseframework
from scripts.robo_utils import build_example, compute_waypoints


class InferenceResult:
    """Container for one inference cycle's outputs (for logging)."""
    __slots__ = ("normalized", "actions", "camera_image",
                 "obs_ms", "infer_ms", "actual_delay", "n_pushed")

    def __init__(self, normalized, actions, camera_image,
                 obs_ms, infer_ms, actual_delay, n_pushed):
        self.normalized = normalized      # (chunk_len, action_dim)
        self.actions = actions            # unnormalized 10d actions
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.n_pushed = n_pushed


class Inferencer:
    """Self-looping inference worker.

    Autonomously cycles: wait cadence → build prefix → infer → push actions.
    Each cycle's output becomes the next cycle's input (current chunk).
    No submit/wait handshake with the main loop — the main loop only reads
    results for logging.
    """

    def __init__(self, model, cam, servo, n_actions, inference_delay,
                 instruction, infer_kwargs, action_stats,
                 fix_rotation=True, use_rtc=True, include_state=False,
                 rtde_r=None, T_bw=None):
        self._model = model
        self._cam = cam
        self._servo = servo
        self._n_actions = n_actions
        self._inference_delay = inference_delay
        self._instruction = instruction
        self._infer_kwargs = infer_kwargs
        self._action_stats = action_stats
        self._fix_rotation = fix_rotation
        self._use_rtc = use_rtc
        self._include_state = include_state
        self._rtde_r = rtde_r
        self._T_bw = T_bw

        # Results posted here for the main loop to consume (logging only).
        self._result_queue = queue.Queue()
        self._running = True
        self._chunk_start_t = 0

        # Set by start(), read by the worker thread.
        self._current_normalized = None
        self._chunk_len = 0
        self._started = threading.Event()

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def start(self, initial_normalized, chunk_len):
        """Kick off the self-looping inference with the initial chunk."""
        self._current_normalized = initial_normalized.copy()
        self._chunk_len = chunk_len
        self._started.set()

    def poll_result(self):
        """Non-blocking: return the latest InferenceResult, or None."""
        result = None
        # Drain queue, keep only the latest
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def wait_result(self, timeout=30.0):
        """Blocking: wait for the next InferenceResult."""
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Shut down the worker thread."""
        self._running = False
        self._started.set()  # unblock if waiting to start

    def _read_state(self):
        """Read robot state for model input (only if include_state)."""
        if not self._include_state or self._rtde_r is None:
            return None
        from scripts.robo_utils import base_to_world, ee_pose_to_state10d
        pose_base = self._rtde_r.getActualTCPPose()
        pose_world = base_to_world(pose_base, self._T_bw)
        return ee_pose_to_state10d(pose_world, self._servo.current_gripper)

    def _loop(self):
        import torch

        # Wait for start() to be called
        self._started.wait()
        if not self._running:
            return

        current_normalized = self._current_normalized
        chunk_len = self._chunk_len
        first_cycle = True

        while self._running:
            # 1. Wait for cadence (skip on first cycle — fire immediately)
            if first_cycle:
                first_cycle = False
                actual_delay = 0
                prev_norm_batch = None
            else:
                trigger_t = self._chunk_start_t + self._n_actions
                self._servo.wait_for_action_t(trigger_t, timeout=30.0)
                if not self._running:
                    break

                # s = actions consumed since chunk started
                s = self._servo.action_t - self._chunk_start_t

                # Build prefix: A_prev = A_cur[s:], right-padded
                if self._use_rtc and s < chunk_len:
                    shifted = np.zeros_like(current_normalized)
                    remaining = chunk_len - s
                    shifted[:remaining] = current_normalized[s:]
                    actual_delay = min(self._inference_delay, remaining)
                    prev_norm_batch = shifted[np.newaxis, ...]
                else:
                    prev_norm_batch = None
                    actual_delay = 0

            # Record new chunk start time
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
            state_10d = self._read_state()
            example = build_example(pil_img, self._instruction, state_10d=state_10d)
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

            # 4. Unnormalize + push free actions into servo buffer
            new_normalized = out["normalized_actions"][0].astype(np.float32)
            new_actions = baseframework.unnormalize_actions(
                new_normalized, self._action_stats)

            actions_to_push = new_actions[actual_delay:]
            n_push = min(self._n_actions, len(actions_to_push))
            actions_to_push = actions_to_push[:n_push]
            start_pos = self._servo.last_pose
            waypoints, _ = compute_waypoints(
                start_pos, actions_to_push, n_push, self._fix_rotation)
            self._servo.push_waypoints(start_pos, waypoints)

            # 5. This prediction becomes the current chunk for next cycle
            current_normalized = new_normalized

            # 6. Post result for main loop (logging only)
            self._result_queue.put(
                InferenceResult(new_normalized, new_actions, pil_img,
                                obs_ms, infer_ms, actual_delay, n_push))
