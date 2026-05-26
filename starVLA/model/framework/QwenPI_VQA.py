# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
QwenPI_VQA: preference-conditioned VLA Stage A MAIN-METHOD framework.

Diff vs QwenPI (baseline, byte-identical except for the loss & VQA path):
  - Adds a per-episode VQA cotrain ("low or high contact at grasp?") whose
    answer is a single bare token (10303 'low' / 11892 'high', verified
    against Qwen3-VL-4B-Instruct tokenizer 2026-05-22).
  - The action stream remains untouched.
  - Aggregation is INSIDE forward: action_loss = L_action + lambda_vqa * L_vqa.
    `train_starvla.py:_train_step` only reads `output_dict["action_loss"]`
    as `total_loss`, so the gate must be sealed here (verified line 426-428).
  - Per-batch / per-task vqa accuracy logged via output_dict["log/..."] keys.

Critical correctness items (spec §2):
  §2.1 VQA input contains ZERO pref-leak — the question itself ("low or high
       contact at grasp?") merely names the two classes the model picks from;
       no Preference suffix, no paraphrase, no task hint. The base-prompt-free
       design (spec choice (c)) is the strongest test of "can the LM head
       read grasp from pixels."
  §2.2 8-frame uniform-linspace clip from head_camera (Qwen3-VL is a video VLM
       and attends to grasp on its own).
  §2.3 Action and VQA are per-frame vs per-episode — handled by
       dedup-then-subsample-2: out of the ≤8 unique episode ids in the batch,
       deterministically take the first-2-by-row-id and run one VQA forward
       on each (so VQA:action ≈ 2:8 = 1:4, the high end of the spec target).
  §2.4 L_action / L_vqa / vqa_acc are logged separately into output_dict;
       trainer relays them to wandb in _train_step.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.model.framework.QwenPI import Qwen_PI
from starVLA.model.tools import FRAMEWORK_REGISTRY
from starVLA.training.trainer_utils import initialize_overwatch

from examples.preference.dataset.vqa_sample import (
    VQA_CATEGORIES,
    cache_row_to_pil,
    get_vqa_clip_cache,
)

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


@FRAMEWORK_REGISTRY.register("QwenPI_VQA")
class Qwen_PI_VQA(Qwen_PI):
    """Action + per-episode VQA cotrain.

    Per-category config (question + binary answer tokens) is selected by
    `framework.vqa.category` in the YAML (default "giveobj"). The two
    pref_keys of the category form the binary classifier slots in the
    order they appear in VQACategoryConfig.pref_keys.

    See module docstring for spec map.
    """

    def __init__(self, config: Optional[dict] = None, **kwargs) -> None:
        super().__init__(config=config, **kwargs)
        vqa_cfg = config.framework.get("vqa", {}) if config and config.framework else {}
        self.lambda_vqa: float = float(vqa_cfg.get("lambda_vqa", 0.5))
        self.n_vqa_per_batch: int = int(vqa_cfg.get("n_vqa_per_batch", 2))

        # ---- category-specific VQA setup ----
        cat_name = str(vqa_cfg.get("category", "giveobj"))
        if cat_name not in VQA_CATEGORIES:
            raise KeyError(
                f"framework.vqa.category={cat_name!r} not in VQA_CATEGORIES "
                f"(expected one of {tuple(VQA_CATEGORIES)})"
            )
        vqa_cat = VQA_CATEGORIES[cat_name]
        self._vqa_category = cat_name
        self._vqa_question: str = vqa_cat.question
        self._vqa_answer_text: Dict[str, str] = dict(vqa_cat.answer_text)
        self._vqa_answer_token_ids: Dict[str, int] = dict(vqa_cat.answer_token_ids)
        # Canonical binary ordering for the classifier slots: pref_keys[0] = "A", [1] = "B".
        self._vqa_pk_A, self._vqa_pk_B = vqa_cat.pref_keys
        self._vqa_id_A = self._vqa_answer_token_ids[self._vqa_pk_A]
        self._vqa_id_B = self._vqa_answer_token_ids[self._vqa_pk_B]
        self._vqa_text_A = self._vqa_answer_text[self._vqa_pk_A]
        self._vqa_text_B = self._vqa_answer_text[self._vqa_pk_B]
        # Map answer first-token id back to its text (used to set assistant turn).
        self._vqa_text_by_id: Dict[int, str] = {v: k for k, v in zip(self._vqa_answer_text.values(), self._vqa_answer_text.values())}
        self._vqa_text_by_id = {self._vqa_id_A: self._vqa_text_A, self._vqa_id_B: self._vqa_text_B}
        logger.info(
            f"VQA category={cat_name!r}: question={self._vqa_question!r}, "
            f"binary={{{self._vqa_pk_A}={self._vqa_text_A!r}({self._vqa_id_A}), "
            f"{self._vqa_pk_B}={self._vqa_text_B!r}({self._vqa_id_B})}}"
        )

    # -- VQA sub-batch helpers --------------------------------------------

    def _select_vqa_episodes(
        self, examples: List[dict]
    ) -> List[Tuple[str, int, str, str]]:
        """
        Spec §2.3: dedup batch episode_ids -> deterministic take-first-N.

        Returns list of (split, row, pref_key, task_group) for VQA forward.
        Deterministic: insertion order in `examples` (which is DataLoader
        order); seen → skip. Sub-2 batches get the trivial guard (whatever
        is available).
        """
        seen: Dict[Tuple[str, int], Tuple[str, str]] = {}
        for ex in examples:
            key = ex["vqa_episode_id"]  # (split, row)
            if key not in seen:
                seen[key] = (ex["vqa_pref_key"], ex["vqa_task_group"])
            if len(seen) >= self.n_vqa_per_batch:
                break
        return [(s, r, pk, tg) for (s, r), (pk, tg) in seen.items()]

    def _build_vqa_inputs(
        self,
        clips: List[List[Image.Image]],
        answer_ids: List[int],
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Build padded input_ids + labels for a VQA sub-batch.

        Label mask (spec §2.1 footgun, hand-verified 2026-05-22):
          - Tokenize each example WITHOUT assistant turn (add_generation_prompt=True)
            -> prompt_only length P_i.
          - Tokenize WITH assistant turn (add_generation_prompt=False)
            -> full length P_i + 3 (answer, im_end, \\n).
          - Labels = full_ids.clone(); set [:P_i] = -100 (mask prompt);
            set [P_i+1:] = -100 (mask im_end + \\n which carry no class info).
          - Net: CE active on EXACTLY the single answer-token position.
          - DO NOT reuse QWen3.build_qwenvl_inputs(solutions=...) mask logic:
            it is action-token-range-aware and silently masks the WHOLE
            sequence to -100 when no action token is found (line 163-166),
            which would zero L_vqa.
        """
        processor = self.qwen_vl_interface.processor
        device = self.qwen_vl_interface.model.device

        # Per-sample tokenize (different image counts can vary, but here all
        # samples have the same 8-frame clip → safe to batch-pad).
        prompt_msgs = []
        full_msgs = []
        for clip, ans_id in zip(clips, answer_ids):
            user_content = [{"type": "image", "image": img} for img in clip] + [
                {"type": "text", "text": self._vqa_question}
            ]
            user_msg = {"role": "user", "content": user_content}
            ans_text = self._vqa_text_by_id[ans_id]
            assistant_msg = {
                "role": "assistant",
                "content": [{"type": "text", "text": ans_text}],
            }
            prompt_msgs.append([user_msg])
            full_msgs.append([user_msg, assistant_msg])

        prompt_only = processor.apply_chat_template(
            prompt_msgs,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        full = processor.apply_chat_template(
            full_msgs,
            tokenize=True,
            padding=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )

        # Per-row prompt length (excluding pad). Tokenizer pads on RIGHT for
        # Qwen by default; we compute non-pad length from attention_mask.
        pad_id = processor.tokenizer.pad_token_id
        prompt_lens = prompt_only["attention_mask"].sum(dim=1)  # (B,)

        labels = full["input_ids"].clone()
        for b in range(labels.size(0)):
            P = int(prompt_lens[b].item())
            # Mask prompt (including any chat-template image-placeholder
            # tokens that prompt_only already accounts for).
            labels[b, :P] = IGNORE_INDEX
            # Mask the trailing im_end + \n. The 3-token assistant span lives
            # at positions [P, P+1, P+2]; we leave only position P.
            labels[b, P + 1:] = IGNORE_INDEX
            # Defensive: any pad tokens in the answer span -> mask.
            labels[b, full["input_ids"][b] == pad_id] = IGNORE_INDEX

        # Sanity-check exactly one active label per row.
        n_active = (labels != IGNORE_INDEX).sum(dim=1)
        assert (n_active == 1).all(), (
            f"VQA label mask broken: per-row active counts = {n_active.tolist()}"
        )

        full = {k: v.to(device) for k, v in full.items()}
        labels = labels.to(device)
        return full, labels

    def _vqa_forward(
        self, examples: List[dict]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Returns (L_vqa, log_dict). L_vqa is on the same device as model."""
        selected = self._select_vqa_episodes(examples)
        if not selected:
            return None, {}

        # Fetch clips from the module-level cache (fork-COW-shared).
        clips: List[List[Image.Image]] = []
        answer_ids: List[int] = []
        task_groups: List[str] = []
        pref_keys: List[str] = []
        for split, row, pk, tg in selected:
            cache = get_vqa_clip_cache(split)
            if cache is None:
                raise RuntimeError(
                    f"VQA clip cache missing for split={split!r}. "
                    f"Was PrefHDF5VQADataset initialized?"
                )
            # cache[row] shape:
            #   - single-cam: (n_frames, H, W, 3)            → n_frames PILs
            #   - multi-cam:  (n_frames, n_cams, H, W, 3)    → n_frames*n_cams PILs,
            #     time-grouped: [cam0_t0, cam1_t0, cam2_t0, cam0_t1, ...]
            #     (matches cache_row_to_pil + load_clip_by_strategy ordering)
            row_view = cache[row]  # uint8 view (COW-safe read)
            pil_list = cache_row_to_pil(row_view)
            clips.append(pil_list)
            answer_ids.append(self._vqa_answer_token_ids[pk])
            task_groups.append(tg)
            pref_keys.append(pk)

        full, labels = self._build_vqa_inputs(clips, answer_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**full, labels=labels, return_dict=True)
        L_vqa = out.loss

        # Accuracy + per-task acc from logits at the answer position.
        with torch.no_grad():
            logits = out.logits  # (B, S, V)
            # Find active position per row (the one non-(-100) label).
            active_mask = labels != IGNORE_INDEX  # (B, S)
            # HF shifts: position i's logit predicts labels[i+1]. So the
            # logit predicting labels[P] is at position P-1.
            answer_pos = active_mask.float().argmax(dim=1) - 1  # (B,)
            B = logits.size(0)
            row_idx = torch.arange(B, device=logits.device)
            ans_logits = logits[row_idx, answer_pos]  # (B, V)
            # Binary slots: pref_keys[0] -> A, pref_keys[1] -> B (category-specific).
            A_logits = ans_logits[:, self._vqa_id_A]
            B_logits = ans_logits[:, self._vqa_id_B]
            pred_is_A = (A_logits > B_logits).cpu().tolist()
            correct = []
            per_task: Dict[str, List[int]] = {}
            for i, (pk, tg) in enumerate(zip(pref_keys, task_groups)):
                hit = int((pred_is_A[i] and pk == self._vqa_pk_A) or
                          ((not pred_is_A[i]) and pk == self._vqa_pk_B))
                correct.append(hit)
                per_task.setdefault(tg, []).append(hit)

        log = {
            "log/L_vqa": float(L_vqa.detach().item()),
            "log/vqa_acc": float(np.mean(correct)) if correct else 0.0,
            "log/vqa_n_samples": float(len(correct)),
        }
        for tg, hits in per_task.items():
            log[f"log/vqa_acc/{tg}"] = float(np.mean(hits))
        return L_vqa, log

    # -- forward ----------------------------------------------------------

    def forward(self, examples: List[dict] = None, **kwargs):
        """Action flow-matching + VQA cotrain. Aggregated loss in action_loss."""
        # 1. Action path -- byte-identical to baseline Qwen_PI.forward().
        action_out = super().forward(examples=examples, **kwargs)
        L_action = action_out["action_loss"]

        # 2. VQA path.
        L_vqa, vqa_log = self._vqa_forward(examples)

        if L_vqa is None:
            total = L_action
            vqa_log = {"log/L_vqa": 0.0, "log/vqa_acc": 0.0, "log/vqa_n_samples": 0.0}
        else:
            total = L_action + self.lambda_vqa * L_vqa

        out = {"action_loss": total, "log/L_action": float(L_action.detach().item())}
        out.update(vqa_log)
        return out

    # -- pseudo-labeler API (Stage B) ------------------------------------

    @torch.inference_mode()
    def predict_preference(self, clip: List[Image.Image]) -> Dict[str, float]:
        """
        Stage B pseudo-labeler. Single-token logit-softmax over the two
        category-specific answer tokens (slot A / slot B).

        Args:
            clip: 8 PIL frames (head_camera, uniform-linspace from the episode).

        Returns:
            {"pref_key": <A or B pref_key>,
             "label": <answer text>,
             "confidence": float in [0,1],
             "p_A": float, "p_B": float,
             "pk_A": ..., "pk_B": ...}
        """
        processor = self.qwen_vl_interface.processor
        device = self.qwen_vl_interface.model.device
        msg = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in clip] +
                       [{"type": "text", "text": self._vqa_question}],
        }]
        inputs = processor.apply_chat_template(
            [msg], tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**inputs, return_dict=True)
        ans_logits = out.logits[0, -1]
        pair = torch.stack([ans_logits[self._vqa_id_A], ans_logits[self._vqa_id_B]])
        probs = F.softmax(pair.float(), dim=0)
        p_A, p_B = float(probs[0].item()), float(probs[1].item())
        winner_pk = self._vqa_pk_A if p_A > p_B else self._vqa_pk_B
        return {
            "pref_key": winner_pk,
            "label": self._vqa_answer_text[winner_pk],
            "confidence": max(p_A, p_B),
            "p_A": p_A,
            "p_B": p_B,
            "pk_A": self._vqa_pk_A,
            "pk_B": self._vqa_pk_B,
        }
