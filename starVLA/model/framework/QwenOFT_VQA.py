# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenOFT_VQA: preference-conditioned VLA Stage A main-method on the OFT backbone.

Action stream = Qwenvl_OFT (OpenVLA-OFT style: action-token query + L1 regression),
byte-identical to the plain OFT framework. VQA cotrain path = VQACotrainMixin (the
exact same code QwenPI_VQA uses). Aggregation: action_loss = L1_action + lambda_vqa·L_vqa,
sealed in forward() (train_starvla reads output_dict["action_loss"]).

Why this is clean: the VQA path only touches the shared VLM (self.qwen_vl_interface),
the module-level clip cache, and VQA_CATEGORIES — it is independent of the action head.
So switching the action backbone from flow-matching (QwenPI) to L1 action-token (OFT)
requires NO change to the VQA path; we just mix it into the OFT framework.

Data: use `pref_hdf5_vqa` with `action_space: joint` (14D qpos, to match the OFT
pretrained ckpt) — the dataset still builds the VQA head-camera clip cache + emits the
per-episode vqa_* fields regardless of the action representation.
"""
from __future__ import annotations

from typing import List, Optional

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.vqa_cotrain_mixin import VQACotrainMixin
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)


@FRAMEWORK_REGISTRY.register("QwenOFT_VQA")
class Qwenvl_OFT_VQA(Qwenvl_OFT, VQACotrainMixin):
    """OFT (L1 action-token) action stream + per-episode VQA cotrain.

    Per-category VQA config selected by `framework.vqa.category`. See VQACotrainMixin.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        self._init_vqa(config)

    def forward(self, examples: List[dict] = None, **kwargs):
        """OFT L1 action stream + VQA cotrain; aggregated into action_loss."""
        action_out = super().forward(examples=examples, **kwargs)  # Qwenvl_OFT.forward → L1 L_action
        return self._aggregate_vqa(examples, action_out["action_loss"])
