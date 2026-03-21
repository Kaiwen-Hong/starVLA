"""Cadence-aware async inferencer (follows Algorithm 1 from RTC paper).

The inferencer runs a persistent worker thread that:
  1. Waits until servo.action_t >= s_min actions since last inference started.
  2. Snapshots the current state (s = actions consumed, A_prev = tail of current chunk).
  3. Runs camera capture + model inference (while servo keeps executing).
  4. Posts the result for the main loop to collect and swap in.

The main loop calls:
  - submit(state_10d, current_normalized, chunk_len)  to queue the next job
  - wait_result()  to block until the job completes

The inferencer owns the cadence: it watches servo.action_t and only fires
after s_min (= n_actions) steps have been consumed.
"""

import time
import threading
import queue

import numpy as np

from scripts.robo_utils import build_example


class InferenceResult:
    """Container for one inference job's outputs."""
    __slots__ = ("output", "camera_image", "obs_ms", "infer_ms",
                 "actual_delay", "prev_norm_batch")

    def __init__(self, output, camera_image, obs_ms, infer_ms,
                 actual_delay, prev_norm_batch):
        self.output = output
        self.camera_image = camera_image
        self.obs_ms = obs_ms
        self.infer_ms = infer_ms
        self.actual_delay = actual_delay
        self.prev_norm_batch = prev_norm_batch


class Inferencer:
    """Persistent inference worker that fires every n_actions servo steps.

    Follows Algorithm 1 (INFERENCELOOP) from the RTC paper:
      - Wait until t >= s_min (enough actions consumed from current chunk)
      - s = t (execution horizon for this cycle)
      - A_prev = A_cur[s:] (tail of current chunk, shifted)
      - d = inference_delay
      - Run guided inference with A_prev as prefix
      - Post result; main loop swaps A_cur and resets t
    """

    def __init__(self, model, cam, servo, n_actions, inference_delay,
                 instruction, infer_kwargs, use_rtc=True):
        self._model = model
        self._cam = cam
        self._servo = servo
        self._n_actions = n_actions          # s_min in paper
        self._inference_delay = inference_delay
        self._instruction = instruction
        self._infer_kwargs = infer_kwargs
        self._use_rtc = use_rtc

        # Job queue: each job is (state_10d, current_normalized, chunk_len)
        # or None to stop.
        self._job_queue = queue.Queue(maxsize=1)
        self._result_queue = queue.Queue(maxsize=1)
        self._running = True

        # The absolute action_t at which the current chunk started executing.
        # The inferencer waits until servo.action_t >= chunk_start_t + n_actions.
        self._chunk_start_t = 0

        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(self, state_10d, current_normalized, chunk_len):
        """Queue the next inference job.

        state_10d:          robot state (or None).
        current_normalized: (chunk_len, action_dim) the current chunk being executed.
        chunk_len:          prediction horizon H.
        """
        self._job_queue.put((state_10d, current_normalized, chunk_len))

    def wait_result(self, timeout=30.0):
        """Block until the current job completes.

        Returns an InferenceResult, or None on timeout.
        """
        try:
            return self._result_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Shut down the worker thread."""
        self._running = False
        try:
            self._job_queue.put_nowait(None)
        except queue.Full:
            pass

    def _loop(self):
        import torch

        while self._running:
            # 1. Wait for next job
            try:
                job = self._job_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            if job is None:
                break
            state_10d, current_normalized, chunk_len = job

            # 2. Wait for s_min actions to be consumed since chunk started
            #    (Paper line 13: wait on C until t >= s_min)
            trigger_t = self._chunk_start_t + self._n_actions
            self._servo.wait_for_action_t(trigger_t, timeout=30.0)

            # 3. s = actions consumed since chunk started (paper line 14)
            s = self._servo.action_t - self._chunk_start_t

            # 4. Build A_prev: shift current chunk by s (paper line 15)
            #    A_prev = A_cur[s, s+1, ..., H-1], right-padded to length H
            if self._use_rtc and s < chunk_len:
                shifted = np.zeros_like(current_normalized)
                remaining = chunk_len - s
                shifted[:remaining] = current_normalized[s:]
                actual_delay = min(self._inference_delay, remaining)
                prev_norm_batch = shifted[np.newaxis, ...]
            else:
                prev_norm_batch = None
                actual_delay = 0

            # Record the new chunk start time (will be used for next cycle)
            # After inference completes and main loop swaps, the next chunk
            # starts at the current servo position.
            # We set this BEFORE inference so the next submit() knows when
            # the current chunk's execution began.
            new_chunk_start_t = self._servo.action_t

            # 5. Camera capture
            t0 = time.monotonic()
            pil_img = self._cam.grab_pil()
            while pil_img is None:
                pil_img = self._cam.grab_pil()
            obs_ms = (time.monotonic() - t0) * 1000

            # 6. Model inference
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

            # Update chunk_start_t for the next cycle
            # (paper line 21: t = t - s, effectively resetting the counter)
            self._chunk_start_t = new_chunk_start_t

            # 7. Post result
            self._result_queue.put(
                InferenceResult(out, pil_img, obs_ms, infer_ms,
                                actual_delay, prev_norm_batch))
