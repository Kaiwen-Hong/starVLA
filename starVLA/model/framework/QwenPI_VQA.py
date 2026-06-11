# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenPI_VQA: preference-conditioned VLA Stage A MAIN-METHOD (flow-matching + VQA cotrain).

Action stream = Qwen_PI (layerwise flow-matching DiT), byte-identical to baseline.
VQA cotrain path = VQACotrainMixin (shared with QwenOFT_VQA; see vqa_cotrain_mixin.py).
Aggregation: action_loss = L_action + lambda_vqa * L_vqa, sealed in forward() because
train_starvla.py:_train_step only reads output_dict["action_loss"].

2026-05-28: the VQA methods (_select_vqa_episodes / _build_vqa_inputs / _vqa_forward /
predict_preference) were factored into VQACotrainMixin (behavior unchanged) so the exact
same VQA code also serves QwenOFT_VQA.
"""
from __future__ import annotations

from typing import List, Optional

from starVLA.model.framework.QwenPI import Qwen_PI
from starVLA.model.framework.vqa_cotrain_mixin import VQACotrainMixin
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenPI_VQA")
class Qwen_PI_VQA(Qwen_PI, VQACotrainMixin):
    """Flow-matching action stream + per-episode VQA cotrain.

    Per-category VQA config (question + binary answer tokens) selected by
    `framework.vqa.category` (default "giveobj"). See VQACotrainMixin.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._init_vqa(config)

    def forward(self, examples: List[dict] = None, **kwargs):
        """Action flow-matching + VQA cotrain; aggregated into action_loss."""
        action_out = super().forward(examples=examples, **kwargs)  # Qwen_PI.forward → L_action
        return self._aggregate_vqa(examples, action_out["action_loss"])
