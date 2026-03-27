"""
StarVLA EE-space policy module for RoboTwin evaluation.

Copy this entire starVLA_ee/ directory to:
  ar-research-kempner/policy/starVLA_ee/

Then eval_policy.py can load it via policy_name=starVLA_ee.

Differences from 14D joint-space model2robotwin_interface.py:
  1. State: observation["endpose"] -> 16D [left_pos(3)+left_quat(4)+left_gripper(1)+right_pos(3)+right_quat(4)+right_gripper(1)]
  2. Action: NO reordering (model output already in correct EE layout)
  3. Execution: TASK_ENV.take_action(action, action_type='ee')
  4. unnorm_key defaults to "new_embodiment" (robotwin_ee not in ROBOT_TYPE_TO_EMBODIMENT_TAG)
  5. Port 5695 (separate from joint-space server on 5694)

Set STARVLA_ROOT env var to point to the starVLA repo on Desktop.
Default: ~/Desktop/research/starVLA
"""

import os
import sys

# Add StarVLA repo to path so we can import WebSocket client and read_mode_config.
STARVLA_ROOT = os.environ.get(
    "STARVLA_ROOT",
    os.path.expanduser("~/Desktop/research/starVLA"),
)
if STARVLA_ROOT not in sys.path:
    sys.path.insert(0, STARVLA_ROOT)

from collections import deque
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import cv2 as cv

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from starVLA.model.tools import read_mode_config


# ─── ModelClient ─────────────────────────────────────────────────────────────

class ModelClient:
    def __init__(
        self,
        policy_ckpt_path: str,
        unnorm_key: str = "new_embodiment",
        host: str = "127.0.0.1",
        port: int = 5695,
        action_mode: str = "abs",
        image_size: Optional[list] = None,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.unnorm_key = unnorm_key
        self.action_mode = action_mode
        self.image_size = image_size if image_size is not None else [224, 224]
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        # Episode state
        self.task_description: Optional[str] = None
        self.raw_actions: Optional[np.ndarray] = None
        self.initial_state: Optional[np.ndarray] = None  # used for rel mode
        self.prev_action: Optional[np.ndarray] = None    # used for delta mode

        print(
            f"*** StarVLA EE | unnorm_key={unnorm_key} | "
            f"action_mode={action_mode} | port={port} ***"
        )
        self.action_norm_stats = self._get_action_stats(
            unnorm_key, policy_ckpt_path, action_mode
        )
        self.action_chunk_size = self._get_action_chunk_size(policy_ckpt_path)
        print(f"    action_chunk_size: {self.action_chunk_size}")

    def reset(self, task_description: str = "") -> None:
        self.task_description = task_description
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        """
        Run one inference step.

        Args:
            example: dict with keys "lang", "image" (list of 3 RGB arrays), "state" (16D np array)
            step: current environment step counter (used for action chunking)

        Returns:
            action: 16D np.ndarray [left_pos(3), left_quat(4), left_gripper(1),
                                     right_pos(3), right_quat(4), right_gripper(1)]
        """
        state = example.get("state", None)
        task_description = example.get("lang", "")

        # Reset if instruction changed
        if task_description != self.task_description:
            self.reset(task_description)

        # Store initial state for delta/rel modes (first step only)
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state")
            self.initial_state = np.array(state, dtype=np.float32).copy()

        # Request new action chunk from server when chunk boundary is reached
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            images = [self._resize_image(img) for img in example["image"]]
            vla_input = {
                "examples": [{"lang": task_description, "image": images}],
                "do_sample": False,
                "use_ddim": self.use_ddim,
                "num_ddim_steps": self.num_ddim_steps,
            }
            response = self.client.predict_action(vla_input)
            try:
                normalized_actions = response["data"]["normalized_actions"][0]  # (chunk, 16)
            except KeyError:
                raise KeyError(
                    f"'normalized_actions' not found in response. "
                    f"Available keys: {list(response.get('data', {}).keys())}"
                )

            raw = self.unnormalize_actions(normalized_actions, self.action_norm_stats)

            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_abs(raw, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_abs(raw)
            else:  # abs
                self.raw_actions = raw

        action_idx = step % self.action_chunk_size
        # Clamp to last action if chunk is shorter than expected
        action_idx = min(action_idx, len(self.raw_actions) - 1)
        action = self.raw_actions[action_idx].copy()

        if self.action_mode == "delta":
            self.prev_action = action.copy()

        # EE space: model output order is already correct for RoboTwin EE action:
        #   [left_pos(3), left_quat(4), left_gripper(1),
        #    right_pos(3), right_quat(4), right_gripper(1)]
        # No reordering needed (unlike 14D joint-space which needs [[0,1,2,3,4,5,12,6,7,8,9,10,11,13]]).
        return action

    # ── Normalization helpers ──────────────────────────────────────────────

    @staticmethod
    def unnormalize_actions(
        normalized_actions: np.ndarray,
        action_norm_stats: Dict,
    ) -> np.ndarray:
        """Undo min-max normalization. Binary dims (gripper) are left as-is by mask."""
        mask = np.array(
            action_norm_stats.get("mask", np.ones(normalized_actions.shape[-1], dtype=bool)),
            dtype=bool,
        )
        action_high = np.array(action_norm_stats["max"], dtype=np.float32)
        action_low  = np.array(action_norm_stats["min"], dtype=np.float32)
        clipped = np.clip(normalized_actions, -1.0, 1.0)
        actions = np.where(
            mask,
            0.5 * (clipped + 1.0) * (action_high - action_low) + action_low,
            clipped,
        )
        return actions.astype(np.float32)

    def _delta_to_abs(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        mask = np.array(
            self.action_norm_stats.get("mask", np.ones(delta_actions.shape[-1], dtype=bool)),
            dtype=bool,
        )
        base = self.prev_action if self.prev_action is not None else self.initial_state
        abs_actions = np.zeros_like(delta_actions)
        for i in range(len(delta_actions)):
            abs_actions[i] = np.where(mask, delta_actions[i] + base, delta_actions[i])
            base = abs_actions[i]
        return abs_actions

    def _rel_to_abs(self, rel_actions: np.ndarray) -> np.ndarray:
        mask = np.array(
            self.action_norm_stats.get("mask", np.ones(rel_actions.shape[-1], dtype=bool)),
            dtype=bool,
        )
        abs_actions = np.zeros_like(rel_actions)
        for i in range(len(rel_actions)):
            abs_actions[i] = np.where(mask, rel_actions[i] + self.initial_state, rel_actions[i])
        return abs_actions

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)

    # ── Stats helpers (called once at init) ───────────────────────────────

    @staticmethod
    def _get_action_stats(unnorm_key: str, policy_ckpt_path: str, action_mode: str = "abs") -> dict:
        _, norm_stats = read_mode_config(policy_ckpt_path)
        if unnorm_key not in norm_stats:
            fallback = next(iter(norm_stats.keys()))
            print(f"[WARNING] unnorm_key '{unnorm_key}' not in stats, using '{fallback}'")
            unnorm_key = fallback
        stats = norm_stats[unnorm_key]
        # New format: {"abs": {"action": {...}}, "delta": {...}}
        if action_mode in stats:
            mode_stats = stats[action_mode]
            return mode_stats.get("action", mode_stats)
        # Old / flat format: {"action": {...}, "state": {...}}
        if "action" in stats:
            if action_mode != "abs":
                print(
                    f"[WARNING] Stats file has flat format only; "
                    f"ignoring action_mode='{action_mode}', using abs stats."
                )
            return stats["action"]
        raise ValueError(f"Cannot find action stats under key '{unnorm_key}'")

    @staticmethod
    def _get_action_chunk_size(policy_ckpt_path: str) -> int:
        model_config, _ = read_mode_config(policy_ckpt_path)
        return model_config["framework"]["action_model"]["future_action_window_size"] + 1


# ─── Policy module interface ──────────────────────────────────────────────────

def get_model(usr_args: dict) -> ModelClient:
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError(
            "'policy_ckpt_path' must be set in deploy_policy.yml. "
            "Example: policy_ckpt_path: /home/kaiwen/Desktop/research/starVLA/results/Checkpoints/"
            "v0309_v31_ee_qwenPI_requeue/checkpoints/steps_50000_pytorch_model.pt"
        )
    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        unnorm_key=usr_args.get("unnorm_key", "new_embodiment"),
        host=usr_args.get("host", "127.0.0.1"),
        port=int(usr_args.get("port", 5695)),
        action_mode=usr_args.get("action_mode", "abs"),
    )


def reset_model(model: ModelClient) -> None:
    model.reset()


def eval(TASK_ENV, model: ModelClient, observation: dict) -> None:
    """One inference + action step for RoboTwin EE-space tasks."""
    instruction = TASK_ENV.get_instruction()

    # EE-space state: 16D
    #   [left_pos(3), left_quat(4), left_gripper(1),
    #    right_pos(3), right_quat(4), right_gripper(1)]
    endpose = observation["endpose"]
    state = np.concatenate([
        np.array(endpose["left_endpose"],  dtype=np.float32),   # pos(3) + quat(4) = 7D
        np.array([endpose["left_gripper"]], dtype=np.float32),   # 1D
        np.array(endpose["right_endpose"],  dtype=np.float32),   # pos(3) + quat(4) = 7D
        np.array([endpose["right_gripper"]], dtype=np.float32),  # 1D
    ])  # total = 16D

    # Camera images — order matches training: [cam_high, cam_left_wrist, cam_right_wrist]
    head_img  = observation["observation"]["head_camera"]["rgb"]
    left_img  = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]

    example = {
        "lang":  str(instruction),
        "image": [head_img, left_img, right_img],
        "state": state,
    }

    action = model.step(example, step=TASK_ENV.take_action_cnt)

    # EE action: RoboTwin interprets action_type='ee' as absolute EE pose
    TASK_ENV.take_action(action, action_type="ee")
