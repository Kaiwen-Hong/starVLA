# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License").
"""
QwenDiscreteDiffusion Framework
Qwen-VL + MaskGIT-style Discrete Diffusion Policy action head.
QwenPI-style: layer-wise cross-attention with vl_embs_list.
"""
from typing import List, Optional

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
    get_action_model,
)
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.training.trainer_utils.trainer_tools import resize_images

try:
    from deployment.model_server.tools.image_tools import to_pil_preserve
except ImportError:
    to_pil_preserve = lambda x: x

logger = initialize_overwatch(__name__)

####################################################
# QwenDiscreteDiffusion: same structure as QwenPI, discrete diffusion action head.
####################################################


@FRAMEWORK_REGISTRY.register("QwenDiscreteDiffusion")
class Qwen_DiscreteDiffusion(baseframework):
    """
    VLA with MaskGIT-style Discrete Diffusion Policy action head.
    QwenPI-style: layer-wise cross-attention with vl_embs_list.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config (match QwenPI)
        num_vl_layers, llm_hidden_size = 36, self.qwen_vl_interface.model.config.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        if hasattr(self.config.framework.action_model, "diffusion_model_cfg"):
            self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = llm_hidden_size

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        action_norm_stats = self._load_action_norm_stats(config)
        self.action_model = get_action_model(
            config=self.config, action_norm_stats=action_norm_stats
        )

    def _load_action_norm_stats(self, config) -> Optional[dict]:
        """Load dataset_statistics.json from output_dir if available."""
        output_dir = getattr(config, "output_dir", None)
        if output_dir is None:
            return None
        stats_path = Path(output_dir) / "dataset_statistics.json"
        if not stats_path.exists():
            return None
        try:
            with open(stats_path, "r") as f:
                full_stats = json.load(f)
            unnorm_key = next(iter(full_stats.keys())) if full_stats else None
            if unnorm_key is None:
                return None
            stats = full_stats[unnorm_key]
            if "action" in stats:
                return stats["action"]
            if "abs" in stats and "action" in stats["abs"]:
                return stats["abs"]["action"]
            return None
        except Exception as e:
            logger.warning(f"Could not load action stats from {stats_path}: {e}")
            return None

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict: action_loss (torch.Tensor)
        """
        batch_images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        actions = [example["action"] for example in examples]

        state = [example["state"] for example in examples] if "state" in examples[0] else None

        # Step 1: QWenVL input format
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
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            while len(vl_embs_list) < expected_layers:
                vl_embs_list.append(vl_embs_list[-1])
            base_hidden = vl_embs_list[-1]

        # Step 2: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4)
                if self.config and self.config.trainer
                else 4
            )
            repeated_diffusion_steps = 2  # NO repeat for big action (match QwenPI)
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            vl_embs_list_repeated = [
                h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list
            ]

            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            pred, target, extra = self.action_model(
                vl_embs_list_repeated,
                actions_target_repeated,
                state_repeated,
            )
            action_loss = self.action_model.loss(pred, target, **extra)

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> dict:
        """
        Inference: MaskGIT-style decode to future actions.

        Returns:
            dict: normalized_actions (np.ndarray) [B, T, action_dim]
        """
        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: QWenVL input format
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
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            while len(vl_embs_list) < expected_layers:
                vl_embs_list.append(vl_embs_list[-1])
            base_hidden = vl_embs_list[-1]

        state = (
            torch.from_numpy(np.array(state)).to(
                base_hidden.device, dtype=base_hidden.dtype
            )
            if state is not None
            else None
        )

        decode_temperature = kwargs.get("decode_temperature", 0.1)
        choice_temperature = kwargs.get("choice_temperature", 0.1)
        use_simple_max = kwargs.get(
            "use_simple_max",
            getattr(self.config.framework.action_model, "use_simple_max", False),
        )

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vl_embs_list,
                state,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
                use_simple_max=use_simple_max,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_realtime(
        self,
        examples: List[dict] = None,
        prev_action_chunk_normalized: np.ndarray = None,
        inference_delay: int = 1,
        **kwargs,
    ) -> dict:
        """
        RTC-aware inference: prefix the first `inference_delay` timesteps
        with actions from the previous chunk so the MaskGIT decode only
        fills remaining positions.

        Args:
            prev_action_chunk_normalized: (B, T, action_dim) *normalized*
                continuous actions from the previous prediction.
            inference_delay: how many leading timesteps to keep as prefix.

        Returns:
            dict with 'normalized_actions' (np.ndarray) [B, T, action_dim]
        """
        if prev_action_chunk_normalized is None or inference_delay <= 0:
            return self.predict_action(examples, **kwargs)

        if not isinstance(examples, list):
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

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
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            while len(vl_embs_list) < expected_layers:
                vl_embs_list.append(vl_embs_list[-1])
            base_hidden = vl_embs_list[-1]

        state_t = (
            torch.from_numpy(np.array(state)).to(
                base_hidden.device, dtype=base_hidden.dtype
            )
            if state is not None
            else None
        )

        prev_chunk_t = torch.from_numpy(
            np.array(prev_action_chunk_normalized)
        ).to(base_hidden.device, dtype=torch.float32)

        decode_temperature = kwargs.get("decode_temperature", 0.1)
        choice_temperature = kwargs.get("choice_temperature", 0.1)

        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action_realtime(
                vl_embs_list,
                state_t,
                prev_action_chunk=prev_chunk_t,
                inference_delay=inference_delay,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
            )

        normalized_actions = pred_actions.detach().float().cpu().numpy()
        return {"normalized_actions": normalized_actions}
