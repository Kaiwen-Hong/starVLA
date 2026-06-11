# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License.
"""
VQACotrainMixin — the action-head-agnostic VQA cotrain path.

The preference VQA cotrain (binary "what is the preference?" question over an
N-frame clip, single-token answer CE) only touches the shared VLM
(`self.qwen_vl_interface`), the module-level clip cache, and `VQA_CATEGORIES`.
It does NOT depend on the action head. So both flow-matching (`Qwen_PI`) and
L1 action-token (`Qwenvl_OFT`) frameworks can mix it in identically:

    class Qwen_PI_VQA(Qwen_PI, VQACotrainMixin):
        def __init__(self, config, **kw):
            super().__init__(config, **kw)
            self._init_vqa(config)
        def forward(self, examples=None, **kw):
            action_out = super().forward(examples=examples, **kw)   # L_action
            return self._aggregate_vqa(examples, action_out["action_loss"])

Aggregation lives in `forward` (sealed here): `action_loss = L_action + λ·L_vqa`,
because `train_starvla.py:_train_step` only reads `output_dict["action_loss"]`.

This module was factored out of the original `QwenPI_VQA.py` (2026-05-28) so the
exact same VQA code serves QwenPI and QwenOFT. Behavior is byte-identical to the
original QwenPI_VQA methods.
"""
from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from starVLA.training.trainer_utils import initialize_overwatch
from examples.preference.dataset.vqa_sample import (
    VQA_CATEGORIES,
    load_clip_and_state,
    format_state_text,
)

logger = initialize_overwatch(__name__)

IGNORE_INDEX = -100


class VQACotrainMixin:
    """Mixin providing the per-episode VQA cotrain path (category-configurable).

    The host framework must define `self.qwen_vl_interface`. Call `_init_vqa(config)`
    at the end of `__init__`, and `_aggregate_vqa(examples, L_action)` in `forward`.
    """

    # -- setup ------------------------------------------------------------

    def _init_vqa(self, config: Optional[dict]) -> None:
        vqa_cfg = config.framework.get("vqa", {}) if config and config.framework else {}
        self.lambda_vqa: float = float(vqa_cfg.get("lambda_vqa", 0.5))
        self.n_vqa_per_batch: int = int(vqa_cfg.get("n_vqa_per_batch", 2))

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
        # Canonical binary ordering: pref_keys[0] = "A", pref_keys[1] = "B".
        self._vqa_pk_A, self._vqa_pk_B = vqa_cat.pref_keys
        self._vqa_id_A = self._vqa_answer_token_ids[self._vqa_pk_A]
        self._vqa_id_B = self._vqa_answer_token_ids[self._vqa_pk_B]
        self._vqa_text_A = self._vqa_answer_text[self._vqa_pk_A]
        self._vqa_text_B = self._vqa_answer_text[self._vqa_pk_B]
        self._vqa_text_by_id: Dict[int, str] = {
            self._vqa_id_A: self._vqa_text_A,
            self._vqa_id_B: self._vqa_text_B,
        }
        # On-demand VQA clip + EE-state config (from VQA_CATEGORIES[cat]).
        self._vqa_cameras = tuple(vqa_cat.cameras)
        self._vqa_strategy = str(vqa_cat.clip_strategy)
        self._vqa_n_frames = int(vqa_cat.n_frames)
        self._vqa_state_in = bool(getattr(vqa_cat, "state_in_vqa", False))
        self._vqa_jitter = int(vqa_cfg.get("frame_jitter", 5))  # train-time ±frames (0=off)
        logger.info(
            f"[VQA cotrain] category={cat_name!r}: lambda_vqa={self.lambda_vqa}, "
            f"n_vqa_per_batch={self.n_vqa_per_batch}, question={self._vqa_question!r}, "
            f"binary={{{self._vqa_pk_A}={self._vqa_text_A!r}({self._vqa_id_A}), "
            f"{self._vqa_pk_B}={self._vqa_text_B!r}({self._vqa_id_B})}} | "
            f"clip: cams={self._vqa_cameras} strat={self._vqa_strategy} n={self._vqa_n_frames} "
            f"state_in={self._vqa_state_in} jitter=±{self._vqa_jitter}"
        )

        # --- state-as-TOKEN mode (redesign 2026-05-28 night): inject the active-arm
        # EE pose as a learned soft token instead of numeric text. Text-state collapsed
        # on held-out (recency prior); the token generalizes (see r-preference/phase1/).
        # Uses an EXISTING special token (<|fim_pad|>) as the marker so NO embedding
        # resize is needed -> the OFT pretrained ckpt loads cleanly.
        self._vqa_state_mode = str(vqa_cfg.get("state_mode", "text"))
        self._vqa_cur_state = None
        if self._vqa_state_mode == "token":
            tok = self.qwen_vl_interface.processor.tokenizer
            self._vqa_marker_str = str(vqa_cfg.get("state_marker", "<|fim_pad|>"))
            self._vqa_marker_id = int(tok.convert_tokens_to_ids(self._vqa_marker_str))
            self._vqa_k = self._vqa_n_frames
            hidden = int(self.qwen_vl_interface.model.config.text_config.hidden_size)
            self.vqa_state_proj = nn.Sequential(
                nn.Linear(9, 512), nn.GELU(), nn.Linear(512, hidden))
            mean, std = self._compute_vqa_state_stats()
            self.register_buffer("_vqa_state_mean", torch.tensor(mean, dtype=torch.float32))
            self.register_buffer("_vqa_state_std", torch.tensor(std, dtype=torch.float32))
            self.qwen_vl_interface.model.get_input_embeddings().register_forward_hook(
                self._vqa_inject_hook)
            logger.info(
                f"[VQA cotrain] state_mode=TOKEN marker={self._vqa_marker_str!r}"
                f"({self._vqa_marker_id}) k={self._vqa_k} proj=9->512->{hidden} "
                f"state_mean={np.round(mean, 3).tolist()}"
            )

    # -- state-token helpers (token mode only) ---------------------------

    def _compute_vqa_state_stats(self, n_sample: int = 64):
        """Mean/std of the active-arm EE 9D pose at the cat's clip frames, from a
        sample of train episodes (endpose-only read, no image decode). Deterministic."""
        import h5py
        import random as _r
        from examples.preference.dataset.prompt import PREF_CATEGORIES
        from examples.preference.dataset.pref_hdf5_dataset import _task_dirs_for_groups
        from examples.preference.dataset.vqa_sample import (
            _compute_clip_indices, _active_arm_key, ee_pose_9d_at,
        )
        root = Path(self.config.datasets.vla_data.data_root_dir)
        pc = PREF_CATEGORIES[self._vqa_category]
        paths = []
        for task_dir, _tg, _pk in _task_dirs_for_groups(root, pc.task_groups, pc.pref_keys):
            paths.extend((root / task_dir / "data").glob("episode*.hdf5"))
        _r.Random(0).shuffle(paths)
        S = []
        for p in paths[:n_sample]:
            try:
                with h5py.File(p, "r") as h5:
                    idx, _ = _compute_clip_indices(h5, self._vqa_strategy, self._vqa_n_frames)
                    S.append(ee_pose_9d_at(h5, _active_arm_key(h5), idx))
            except Exception:
                continue
        S = np.concatenate(S, axis=0) if S else np.zeros((1, 9), dtype=np.float64)
        return S.mean(0).astype("float32"), (S.std(0) + 1e-6).astype("float32")

    def _vqa_inject_hook(self, module, inp, out):
        """Overwrite marker-token embeddings with phi(state). No-op when no markers
        present (e.g. the action-stream forward) so it is safe on every VLM call."""
        cur = self._vqa_cur_state
        if cur is None:
            return out
        mask = inp[0] == self._vqa_marker_id
        if int(mask.sum()) == 0:
            return out
        out = out.clone()
        out[mask] = cur.to(out.dtype)
        return out

    def _set_vqa_state_embeds(self, st_np: np.ndarray) -> None:
        """st_np: (B, k, 9) active-arm EE pose -> set self._vqa_cur_state (B*k, hidden)."""
        dev = self.qwen_vl_interface.model.device
        st = torch.as_tensor(np.asarray(st_np), dtype=torch.float32, device=dev)
        st = (st - self._vqa_state_mean.to(dev)) / self._vqa_state_std.to(dev)
        B = st.shape[0]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            emb = self.vqa_state_proj(st)            # (B, k, hidden)
        self._vqa_cur_state = emb.reshape(B * self._vqa_k, emb.shape[-1])

    # -- aggregation (called from host forward) ---------------------------

    def _aggregate_vqa(self, examples: List[dict], L_action: torch.Tensor) -> Dict:
        """total action_loss = L_action + lambda_vqa * L_vqa, with logs."""
        L_vqa, vqa_log = self._vqa_forward(examples)
        if L_vqa is None:
            total = L_action
            vqa_log = {"log/L_vqa": 0.0, "log/vqa_acc": 0.0, "log/vqa_n_samples": 0.0}
        else:
            total = L_action + self.lambda_vqa * L_vqa
        out = {"action_loss": total, "log/L_action": float(L_action.detach().item())}
        out.update(vqa_log)
        return out

    # -- VQA sub-batch helpers --------------------------------------------

    def _select_vqa_episodes(
        self, examples: List[dict]
    ) -> List[Tuple[str, str, str]]:
        """dedup batch episodes by hdf5 path -> deterministic take-first-N.

        Returns list of (h5_path, pref_key, task_group).
        """
        seen: Dict[str, Tuple[str, str]] = {}
        for ex in examples:
            key = ex["vqa_h5_path"]
            if key not in seen:
                seen[key] = (ex["vqa_pref_key"], ex["vqa_task_group"])
            if len(seen) >= self.n_vqa_per_batch:
                break
        return [(h5, pk, tg) for h5, (pk, tg) in seen.items()]

    def _build_vqa_inputs(
        self,
        clips: List[List[Image.Image]],
        answer_ids: List[int],
        states: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Padded input_ids + labels for a VQA sub-batch; CE active on exactly the
        single answer-token position (prompt + trailing im_end/\\n masked to -100).
        If states[i] is given, the active-arm EE pose is injected as text before the question."""
        processor = self.qwen_vl_interface.processor
        device = self.qwen_vl_interface.model.device

        prompt_msgs = []
        full_msgs = []
        for i, (clip, ans_id) in enumerate(zip(clips, answer_ids)):
            q = self._vqa_question
            if self._vqa_state_mode == "token":
                q = (self._vqa_marker_str * self._vqa_k) + q
            elif self._vqa_state_in and states is not None and states[i] is not None:
                q = format_state_text(states[i]) + q
            user_content = [{"type": "image", "image": img} for img in clip] + [
                {"type": "text", "text": q}
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
            prompt_msgs, tokenize=True, padding=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        )
        full = processor.apply_chat_template(
            full_msgs, tokenize=True, padding=True, add_generation_prompt=False,
            return_dict=True, return_tensors="pt",
        )

        pad_id = processor.tokenizer.pad_token_id
        prompt_lens = prompt_only["attention_mask"].sum(dim=1)  # (B,)

        labels = full["input_ids"].clone()
        for b in range(labels.size(0)):
            P = int(prompt_lens[b].item())
            labels[b, :P] = IGNORE_INDEX
            labels[b, P + 1:] = IGNORE_INDEX
            labels[b, full["input_ids"][b] == pad_id] = IGNORE_INDEX

        n_active = (labels != IGNORE_INDEX).sum(dim=1)
        assert (n_active == 1).all(), (
            f"VQA label mask broken: per-row active counts = {n_active.tolist()}"
        )

        full = {k: v.to(device) for k, v in full.items()}
        labels = labels.to(device)
        return full, labels

    def _vqa_forward(
        self, examples: List[dict]
    ) -> Tuple[Optional[torch.Tensor], Dict[str, float]]:
        """Returns (L_vqa, log_dict). L_vqa on the model device, or (None, {})."""
        selected = self._select_vqa_episodes(examples)
        if not selected:
            return None, {}

        clips: List[List[Image.Image]] = []
        states: List[Optional[np.ndarray]] = []
        answer_ids: List[int] = []
        task_groups: List[str] = []
        pref_keys: List[str] = []
        for h5_path, pk, tg in selected:
            frames, state, _ = load_clip_and_state(
                h5_path, strategy=self._vqa_strategy, n=self._vqa_n_frames,
                cameras=self._vqa_cameras, jitter=self._vqa_jitter,
            )
            clips.append(frames)
            states.append(state)
            answer_ids.append(self._vqa_answer_token_ids[pk])
            task_groups.append(tg)
            pref_keys.append(pk)

        if self._vqa_state_mode == "token":
            self._set_vqa_state_embeds(
                np.stack([np.asarray(s, dtype=np.float32) for s in states]))
        full, labels = self._build_vqa_inputs(clips, answer_ids, states)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.qwen_vl_interface(**full, labels=labels, return_dict=True)
        L_vqa = out.loss

        with torch.no_grad():
            logits = out.logits  # (B, S, V)
            active_mask = labels != IGNORE_INDEX
            answer_pos = active_mask.float().argmax(dim=1) - 1  # logit predicting labels[P] is at P-1
            B = logits.size(0)
            row_idx = torch.arange(B, device=logits.device)
            ans_logits = logits[row_idx, answer_pos]  # (B, V)
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

    # -- pseudo-labeler API (Stage B) ------------------------------------

    @torch.inference_mode()
    def predict_preference(self, clip: List[Image.Image],
                           state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Single-token logit-softmax over the two category answer tokens.
        If `state` (active-arm EE pose, (n,9)) is given, inject it as text."""
        processor = self.qwen_vl_interface.processor
        device = self.qwen_vl_interface.model.device
        q = self._vqa_question
        if self._vqa_state_mode == "token":
            q = (self._vqa_marker_str * self._vqa_k) + q
            self._set_vqa_state_embeds(np.asarray(state, dtype=np.float32)[None])
        elif state is not None:
            q = format_state_text(state) + q
        msg = [{
            "role": "user",
            "content": [{"type": "image", "image": img} for img in clip] +
                       [{"type": "text", "text": q}],
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

    # -- held-out validation (the REAL objective signal) ------------------

    @torch.inference_mode()
    def validate_vqa(self, n_per_leaf: int = 10) -> Dict[str, float]:
        """Held-out VQA acc + CE on taskA-val + taskB (jitter-free clips). Surfaces
        real generalization vs the misleading train L_vqa→0. Returns val/vqa_* logs."""
        from examples.preference.dataset.prompt import PREF_CATEGORIES
        from examples.preference.dataset.pref_hdf5_dataset import (
            _task_dirs_for_groups, _split_episodes,
        )
        cat = self._vqa_category
        root = Path(self.config.datasets.vla_data.data_root_dir)
        pc = PREF_CATEGORIES[cat]
        rng = random.Random(123)

        def run(leaf_eps):
            rows = []
            for h5_path, gt in leaf_eps:
                frames, state, _ = load_clip_and_state(
                    h5_path, strategy=self._vqa_strategy, n=self._vqa_n_frames,
                    cameras=self._vqa_cameras, jitter=0)
                r = self.predict_preference(frames, state=state if self._vqa_state_in else None)
                p_correct = r["p_A"] if gt == self._vqa_pk_A else r["p_B"]
                if not (p_correct == p_correct):  # nan guard (bf16 logit overflow)
                    p_correct = 1e-6
                ce = -math.log(min(max(p_correct, 1e-6), 1.0))
                rows.append((int(r["pref_key"] == gt), ce, gt))
            return rows

        split = _split_episodes(root, pc.task_groups, pc.pref_keys, val_fraction=0.2, seed=42)
        taskA = []
        for task_dir, tg, pk in _task_dirs_for_groups(root, pc.task_groups, pc.pref_keys):
            vep = list(split[task_dir]["val"]); rng.shuffle(vep)
            for ep in vep[:n_per_leaf]:
                taskA.append((str(root / task_dir / "data" / f"episode{ep}.hdf5"), pk))
        taskB = []
        tb = root / "taskB"
        if tb.is_dir():
            for d in sorted(tb.iterdir()):
                if d.is_dir() and (d / "data").is_dir():
                    gt = d.name.rsplit("_", 1)[-1]
                    if gt in pc.pref_keys:
                        eps = sorted(int(p.stem[7:]) for p in (d / "data").glob("episode*.hdf5"))
                        rng.shuffle(eps)
                        for ep in eps[:n_per_leaf]:
                            taskB.append((str(d / "data" / f"episode{ep}.hdf5"), gt))

        log: Dict[str, float] = {}
        for name, eps in [("taskA", taskA), ("taskB", taskB)]:
            rows = run(eps)
            if not rows:
                continue
            per: Dict[str, List[int]] = {}
            for hit, _ce, gt in rows:
                per.setdefault(gt, []).append(hit)
            log[f"val/vqa_{name}_acc"] = round(float(np.mean([r[0] for r in rows])), 4)
            log[f"val/vqa_{name}_bal_acc"] = round(float(np.mean([np.mean(v) for v in per.values()])), 4)
            log[f"val/vqa_{name}_loss"] = round(float(np.mean([r[1] for r in rows])), 4)
            log[f"val/vqa_{name}_n"] = float(len(rows))
        return log
