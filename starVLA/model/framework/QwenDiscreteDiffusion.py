# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""
QwenDiscreteDiffusion Framework
Qwen-VL + MaskGIT-style Discrete Diffusion Policy action head.
"""

import sys
from pathlib import Path

_workspace_root = Path(__file__).parent.parent.parent.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.DiscreteDiffusion_ActionHeader import (
    get_action_model,
    DiscreteDiffusionActionHead,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils.trainer_tools import resize_images

try:
    from deployment.model_server.tools.image_tools import to_pil_preserve
except ImportError:
    to_pil_preserve = lambda x: x


@FRAMEWORK_REGISTRY.register("QwenDiscreteDiffusion")
class Qwen_DiscreteDiffusion(baseframework):
    """
    VLA with MaskGIT-style Discrete Diffusion Policy action head.
    - Qwen-VL for vision-language encoding
    - Masking-based discrete diffusion over binned actions (ref: ref_dd_mode.py)
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        if hasattr(config.framework.action_model, "diffusion_model_cfg"):
            config.framework.action_model.diffusion_model_cfg.cross_attention_dim = (
                self.qwen_vl_interface.model.config.hidden_size
            )

        self.action_model: DiscreteDiffusionActionHead = get_action_model(
            config=self.config
        )

        self.future_action_window_size = (
            config.framework.action_model.future_action_window_size
        )
        self.past_action_window_size = (
            config.framework.action_model.past_action_window_size
        )
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        batch_images = [ex["image"] for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        actions = [ex["action"] for ex in examples]
        state = (
            [ex["state"] for ex in examples]
            if "state" in examples[0] and examples[0].get("state") is not None
            else None
        )

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        with torch.autocast("cuda", dtype=torch.float32):
            actions_tensor = torch.tensor(
                np.array(actions), device=last_hidden.device, dtype=torch.float32
            )
            actions_target = actions_tensor[
                :, -(self.future_action_window_size + 1) :, :
            ]

            repeated_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer
                else 4
            )
            actions_target_repeated = actions_target.repeat(repeated_steps, 1, 1)
            last_hidden_repeated = last_hidden.repeat(repeated_steps, 1, 1)

            state_repeated = None
            if state is not None:
                state_tensor = torch.tensor(
                    np.array(state), device=last_hidden.device, dtype=torch.float32
                )
                state_repeated = state_tensor.repeat(repeated_steps, 1, 1)

            action_loss = self.action_model(
                last_hidden_repeated, actions_target_repeated, state_repeated
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict],
        prev_action_chunk: Optional[np.ndarray] = None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(ex["image"]) for ex in examples]
        instructions = [ex["lang"] for ex in examples]
        state = (
            [ex["state"] for ex in examples]
            if "state" in examples[0] and examples[0].get("state") is not None
            else None
        )

        train_obs_image_size = getattr(
            self.config.datasets.vla_data, "image_size", None
        )
        if train_obs_image_size:
            batch_images = resize_images(
                batch_images, target_size=train_obs_image_size
            )

        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qwenvl_outputs.hidden_states[-1]

        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                last_hidden.device, dtype=last_hidden.dtype
            )

        prev_chunk_tensor = None
        use_realtime = False
        if prev_action_chunk is not None and len(prev_action_chunk) > 0:
            prev_chunk_tensor = torch.from_numpy(
                np.array(prev_action_chunk, dtype=np.float32)
            ).to(last_hidden.device)
            if prev_chunk_tensor.dim() == 2:
                prev_chunk_tensor = prev_chunk_tensor.unsqueeze(0)
            use_realtime = True

        with torch.autocast("cuda", dtype=torch.float32):
            if use_realtime:
                pred_actions = self.action_model.predict_action_realtime(
                    last_hidden,
                    state_tensor,
                    prev_action_chunk=prev_chunk_tensor,
                    inference_delay=inference_delay,
                )
            else:
                pred_actions = self.action_model.predict_action(
                    last_hidden, state_tensor
                )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
