"""Self-looping cadence-aware inferencer v2.

Fix over inferencer.py:
  Uses servo.tail_pose (end of buffer) instead of servo.last_pose (current
  robot position) when computing new waypoints.  This eliminates the spatial
  discontinuity at buffer splice points that caused visible snap-back stutters.
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
        self.normalized = normalized
        self.actions = actions
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.n_pushed = n_pushed


class Inferencer:
    """Self-looping inference worker (v2: uses tail_pose for buffer continuity)."""

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

    def poll_result(self):
        result = None
        while True:
            try:
                result = self._result_queue.get_nowait()
            except queue.Empty:
                break
        return result

    def wait_result(self, timeout=30.0):
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        self._started.set()

    def _read_state(self):
        if not self._include_state or self._rtde_r is None:
            return None
        from scripts.robo_utils import base_to_world, ee_pose_to_state10d
        pose_base = self._rtde_r.getActualTCPPose()
        pose_world = base_to_world(pose_base, self._T_bw)
        return ee_pose_to_state10d(pose_world, self._servo.current_gripper)

    def _loop(self):
        import torch

        self._started.wait()
        if not self._running:
            return

        current_normalized = self._current_normalized
        chunk_len = self._chunk_len
        first_cycle = True

        while self._running:
            # 1. Wait for cadence (skip on first cycle)
            if first_cycle:
                first_cycle = False
                actual_delay = 0
                prev_norm_batch = None
            else:
                trigger_t = self._chunk_start_t + self._n_actions
                self._servo.wait_for_action_t(trigger_t, timeout=30.0)
                if not self._running:
                    break

                s = self._servo.action_t - self._chunk_start_t

                if self._use_rtc and s < chunk_len:
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

            # FIX: use tail_pose (end of buffer) instead of last_pose (current
            # robot position) to avoid spatial discontinuity at splice point.
            start_pos = self._servo.tail_pose
            waypoints, grip_trans = compute_waypoints(
                start_pos, actions_to_push, n_push, self._fix_rotation)
            self._servo.push_waypoints(start_pos, waypoints, grip_trans)

            # 5. New prediction becomes current chunk
            current_normalized = new_normalized

            # 6. Post result for main loop
            self._result_queue.put(
                InferenceResult(new_normalized, new_actions, pil_img,
                                obs_ms, infer_ms, actual_delay, n_push))
