#!/usr/bin/env python3
"""
Episodic RTC open-loop eval for the DD natural schedule, with early-stop tracing.

Flow:
  1. Pick ONE episode from the FastUMI dataset and load its frames sequentially.
  2. Walk through the episode in RTC cycles (stride = n_actions):
       cycle 0  →  model.predict_action (no prefix)
       cycle i  →  model.predict_action_realtime (hard_mask=False, i.e. natural
                    mask schedule) with prev_action_chunk = previous prediction
                    tail [chunk_len - n_actions:].
  3. At each RTC cycle, instrument the MaskGIT decode loop to record:
       - num_steps_scheduled (self.num_inference_steps * fraction_unknown)
       - stop_step           (decode step at which early_stop triggered)
       - mask_remaining      (# masked non-prefix tokens after each decode step)
  4. Plot:
       - early-stop step vs RTC cycle
       - mask-remaining heatmap (cycle × decode step)
       - per-cycle MSE against the GT action chunk.

Usage:
    python realworld/0417-dd-openloop-eval-natural.py
    python realworld/0417-dd-openloop-eval-natural.py --episode_idx 3 --n_actions 5
    python realworld/0417-dd-openloop-eval-natural.py --no_early_stop
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")

import argparse
import sys
import os
import time
import json
import types
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
from starVLA.model.modules.action_model.discrete_diffusion import (
    decode_mask_schedule,
    mask_by_deterministic_lowest,
    mask_by_random_topk,
)

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT = (
    "/scratch/wangpc/starVLA/results/Checkpoints/"
    "fastumi_pickandplace_qwenDiscreteDiffusion_0409_0_pick_to_moved_filtered/"
    "checkpoints/steps_30000_pytorch_model.pt"
)
DATA_ROOT_DIR = "/scratch/wangpc/starVLA/playground/Datasets/FastUMI"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1

DIM_LABELS = [
    "pos_x", "pos_y", "pos_z",
    "rot6d_0", "rot6d_1", "rot6d_2",
    "rot6d_3", "rot6d_4", "rot6d_5",
    "gripper",
]
# ───────────────────────────────────────────────────────────────────


def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[INFO] flash_attn not available, falling back to sdpa")
        return "sdpa"


def load_model(checkpoint_path: str):
    print(f"Loading model from: {checkpoint_path}")
    t0 = time.time()

    checkpoint_pt = Path(checkpoint_path)
    model_config, norm_stats = read_mode_config(checkpoint_pt)
    config = dict_to_namespace(model_config)
    config.trainer.pretrained_checkpoint = None

    attn_impl = _detect_attn_implementation()
    config.framework.qwenvl.attn_implementation = attn_impl

    model = build_framework(cfg=config)
    model.norm_stats = norm_stats

    state_dict = torch.load(checkpoint_pt, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)

    model = model.to("cuda").eval()
    print(f"Model loaded in {time.time() - t0:.1f}s (attn: {attn_impl})")
    print(f"  chunk_len: {model.chunk_len}")
    print(f"  num_bins: {getattr(config.framework.action_model, 'num_bins', 'N/A')}")
    print(f"  num_inference_steps: {getattr(config.framework.action_model, 'num_inference_steps', 'N/A')}")
    print(f"  decode_schedule: {getattr(config.framework.action_model, 'decode_schedule', 'cosine')}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  Episodic dataset loading
# ═══════════════════════════════════════════════════════════════════

def load_episode(model, data_root_dir: str, episode_idx: int,
                 include_state: bool = False):
    """Load a single episode as a list of pre-packed samples in time order."""
    data_cfg = model.config.datasets.vla_data
    if include_state:
        data_cfg.include_state = True
    if data_root_dir:
        data_cfg.data_root_dir = data_root_dir

    print(f"\nLoading dataset (to pick one episode): {data_cfg.data_mix}")
    print(f"  root: {data_cfg.data_root_dir}")
    mixture = get_vla_dataset(data_cfg=data_cfg)

    if not mixture.datasets:
        raise RuntimeError("Empty mixture dataset")
    single = mixture.datasets[0]
    traj_ids = single.trajectory_ids
    traj_lens = single.trajectory_lengths
    if episode_idx < 0 or episode_idx >= len(traj_ids):
        raise ValueError(
            f"episode_idx {episode_idx} out of range [0, {len(traj_ids)})")
    traj_id = int(traj_ids[episode_idx])
    traj_len = int(traj_lens[episode_idx])
    print(f"  num episodes: {len(traj_ids)}")
    print(f"  selected episode_idx={episode_idx}  traj_id={traj_id}  length={traj_len}")

    # filter all_steps by trajectory and sort by base_index
    episode_steps = [(tid, bi) for (tid, bi) in single.all_steps if tid == traj_id]
    episode_steps.sort(key=lambda x: x[1])
    print(f"  steps in episode after dataset filtering: {len(episode_steps)}")

    samples = []
    for tid, bi in tqdm(episode_steps, desc="Loading episode frames"):
        raw = single.get_step_data(tid, bi)
        data = single.transforms(raw)
        sample = single._pack_sample(data)
        sample["_traj_id"] = tid
        sample["_base_index"] = bi
        samples.append(sample)
    return samples, single


# ═══════════════════════════════════════════════════════════════════
#  Instrumented MaskGIT decode (monkey-patched onto action_model)
# ═══════════════════════════════════════════════════════════════════

def install_instrumented_decode(model):
    """Replace action_model.predict_action_realtime + predict_action with
    versions that record decode-loop instrumentation.

    Exposes the latest trace via ``model._early_stop_trace``. Captures:
        - cycle_type          : "init" or "rtc"
        - num_steps_scheduled : decode steps scheduled for this call
        - stop_step           : step at which decode broke (1-indexed)
        - mask_remaining_per_step : # of masked non-prefix tokens after step
        - prefix_length       : # of leading positions kept as prefix
        - execution_horizon   : # of trailing positions to generate
        - early_stopped       : whether early_stop triggered
        - mask_matrix_initial : [H, D] uint8, mask state BEFORE step 0
                                 (1 = masked, 0 = fixed/unmasked prefix)
        - mask_matrix_per_step: list of [H, D] uint8 mask states, one per
                                 completed decode step

    The decode math is copied verbatim from
    LayerwiseDiscreteDiffusionActionHead; the only additions are the trace
    writes and the cpu-numpy snapshots. Model code is NOT edited; the
    patch lives entirely in this script.
    """
    trace_holder = {
        "cycle_type": None,
        "num_steps_scheduled": None,
        "stop_step": None,
        "mask_remaining_per_step": [],
        "prefix_length": None,
        "execution_horizon": None,
        "early_stopped": False,
        "mask_matrix_initial": None,
        "mask_matrix_per_step": [],
    }
    model._early_stop_trace = trace_holder
    # Side-channel: the caller (run_rtc_episode) sets this to a
    # [1, T_prev, action_dim] bool tensor before each RTC call; True
    # means "this prev-chunk slot was mask_token in the previous cycle,
    # treat it as padded (not real prefix)".
    model.action_model._prev_chunk_mask = None

    def _reset_trace():
        trace_holder["cycle_type"] = None
        trace_holder["num_steps_scheduled"] = None
        trace_holder["stop_step"] = None
        trace_holder["mask_remaining_per_step"] = []
        trace_holder["prefix_length"] = None
        trace_holder["execution_horizon"] = None
        trace_holder["early_stopped"] = False
        trace_holder["mask_matrix_initial"] = None
        trace_holder["mask_matrix_per_step"] = []

    def _mask_matrix(cur_seqs, mask_token_id):
        """Return [H, D] uint8: 1 where token == mask_token_id, else 0.
        Uses batch index 0 (eval batch size is always 1 here).
        """
        return (cur_seqs[0] == mask_token_id).detach().cpu().numpy().astype(np.uint8)

    @torch.no_grad()
    def _instrumented_realtime(self, vl_embs_list, state=None,
                               prev_action_chunk=None, inference_delay=1,
                               execution_horizon=None,
                               choice_temperature=0.1, decode_temperature=1.0,
                               fixed_steps=False, hard_mask=False,
                               early_stop=False):
        if prev_action_chunk is None or inference_delay <= 0:
            return _instrumented_init(
                self, vl_embs_list, state,
                choice_temperature=choice_temperature,
                decode_temperature=decode_temperature,
            )

        _reset_trace()
        trace_holder["cycle_type"] = "rtc"

        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        L = self.seq_len
        deterministic_decode = decode_temperature == 0
        deterministic_choice = choice_temperature == 0

        if execution_horizon is None:
            execution_horizon = inference_delay
        execution_horizon = min(execution_horizon, self.action_horizon)

        if hard_mask:
            prefix_length = inference_delay
        else:
            prefix_length = self.action_horizon - execution_horizon

        # Encode only the real (non-padded) prev-chunk positions. Pad slots
        # get mask_token_id directly in prefix_bins so they're excluded from
        # the effective prefix below. Any slots that were left masked by
        # the previous cycle (passed via self._prev_chunk_mask) also get
        # overwritten with mask_token_id so they get regenerated here.
        T_prev = prev_action_chunk.shape[1]
        prefix_bins = self.binning.encode(prev_action_chunk)
        prev_mask = getattr(self, "_prev_chunk_mask", None)
        if prev_mask is not None and prev_mask.shape[1] == T_prev:
            prefix_bins = torch.where(
                prev_mask,
                torch.full_like(prefix_bins, self.mask_token_id),
                prefix_bins,
            )
        if T_prev < self.action_horizon:
            pad_bins = torch.full(
                (B, self.action_horizon - T_prev, self.action_dim),
                self.mask_token_id, dtype=prefix_bins.dtype, device=device,
            )
            prefix_bins = torch.cat([prefix_bins, pad_bins], dim=1)

        positional_prefix_mask = (
            torch.arange(self.action_horizon, device=device)[None, :, None]
            < prefix_length
        ).expand(B, self.action_horizon, self.action_dim)
        # Effective prefix = positional AND real bin value (not pad/masked-forward).
        prefix_mask = positional_prefix_mask & (prefix_bins != self.mask_token_id)

        cur_seqs = torch.where(
            prefix_mask,
            prefix_bins,
            torch.full_like(prefix_bins, self.mask_token_id),
        )
        trace_holder["mask_matrix_initial"] = _mask_matrix(
            cur_seqs, self.mask_token_id)

        num_unknown_tokens = (cur_seqs == self.mask_token_id).sum(dim=(1, 2))
        unknown_init = num_unknown_tokens
        if fixed_steps:
            num_steps = self.num_inference_steps
        else:
            num_steps = max(1, int(
                self.num_inference_steps * num_unknown_tokens.max().item() / L))

        state_feat = None
        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)

        trace_holder["num_steps_scheduled"] = int(num_steps)
        trace_holder["prefix_length"] = int(prefix_length)
        trace_holder["execution_horizon"] = int(execution_horizon)
        trace_holder["stop_step"] = int(num_steps)

        # Pre-loop early-stop: if the initial state already has no masked
        # tokens in the early region, skip decoding entirely.
        if early_stop:
            early_region_init = cur_seqs[:, :inference_delay + execution_horizon, :]
            n_remaining_init = int(
                (early_region_init == self.mask_token_id).sum().item())
            if n_remaining_init == 0:
                trace_holder["stop_step"] = 0
                trace_holder["early_stopped"] = True
                num_steps = 0

        for step_idx in range(num_steps):
            logits = self._forward_logits(
                vl_embs_list, cur_seqs, state_feat, device)
            safe_temp = max(decode_temperature, 1e-8)
            sampled, selected_probs = self.binning.sample_indices_from_logits(
                logits,
                temperature=safe_temp,
                deterministic=deterministic_decode,
            )

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(prefix_mask, prefix_bins, sampled)
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = decode_mask_schedule(
                torch.tensor(ratio, device=device), self.decode_schedule)
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            if step_idx == num_steps - 1:
                mask_len = torch.zeros_like(mask_len)

            selected_probs = torch.where(
                unknown_map,
                selected_probs,
                torch.full_like(selected_probs, float("inf")),
            )

            selected_flat = selected_probs.reshape(B, L)
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(
                    selected_flat, mask_len)
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(
                    selected_flat, mask_len, temperature=temp)
            action_mask = action_mask_flat.reshape(
                B, self.action_horizon, self.action_dim)
            cur_seqs = torch.where(
                prefix_mask,
                prefix_bins,
                torch.where(action_mask, self.mask_token_id, sampled),
            )

            trace_holder["mask_matrix_per_step"].append(
                _mask_matrix(cur_seqs, self.mask_token_id))
            # Stop condition: the first (inference_delay + execution_horizon)
            # action rows must all be unmasked. We DO NOT force-commit the
            # remaining masked positions — those masks are intentional and
            # get propagated to the next cycle so the next call knows to
            # regenerate them.
            early_region = cur_seqs[:, :inference_delay + execution_horizon, :]
            n_remaining = int(
                (early_region == self.mask_token_id).sum().item())
            trace_holder["mask_remaining_per_step"].append(n_remaining)

            if early_stop and n_remaining == 0:
                trace_holder["stop_step"] = step_idx + 1
                trace_holder["early_stopped"] = True
                break

        # binning.decode would choke on mask_token_id (index == num_bins),
        # so substitute with bin 0 *only for the continuous return value*.
        # The trace's mask_matrix_per_step[-1] records where the masks
        # actually are, and the caller uses it to propagate forward.
        decode_seqs = torch.where(
            cur_seqs == self.mask_token_id,
            torch.zeros_like(cur_seqs),
            cur_seqs,
        )
        return self.binning.decode(decode_seqs)

    @torch.no_grad()
    def _instrumented_init(self, vl_embs_list, state=None,
                           choice_temperature=0.1, decode_temperature=1.0,
                           use_simple_max=False):
        _reset_trace()
        trace_holder["cycle_type"] = "init"

        B = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        L = self.seq_len

        cur_seqs = torch.full(
            (B, self.action_horizon, self.action_dim),
            self.mask_token_id, dtype=torch.long, device=device,
        )
        trace_holder["mask_matrix_initial"] = _mask_matrix(
            cur_seqs, self.mask_token_id)
        trace_holder["prefix_length"] = 0
        trace_holder["execution_horizon"] = self.action_horizon

        state_feat = None
        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = self.state_encoder(state).unsqueeze(1)

        if use_simple_max:
            logits = self._forward_logits(
                vl_embs_list, cur_seqs, state_feat, device)
            trace_holder["num_steps_scheduled"] = 1
            trace_holder["stop_step"] = 1
            final = self.binning.decode_logits(logits)
            # reconstruct bin indices just for mask snapshot
            trace_holder["mask_matrix_per_step"].append(
                np.zeros_like(trace_holder["mask_matrix_initial"]))
            trace_holder["mask_remaining_per_step"].append(0)
            return final

        deterministic_decode = decode_temperature == 0
        deterministic_choice = choice_temperature == 0
        unknown_init = torch.full((B,), L, dtype=torch.long, device=device)
        num_steps = self.num_inference_steps
        trace_holder["num_steps_scheduled"] = int(num_steps)
        trace_holder["stop_step"] = int(num_steps)

        for step_idx in range(num_steps):
            logits = self._forward_logits(
                vl_embs_list, cur_seqs, state_feat, device)
            safe_temp = max(decode_temperature, 1e-8)
            sampled, selected_probs = self.binning.sample_indices_from_logits(
                logits,
                temperature=safe_temp,
                deterministic=deterministic_decode,
            )

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = decode_mask_schedule(
                torch.tensor(ratio, device=device), self.decode_schedule)
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            if step_idx == num_steps - 1:
                mask_len = torch.zeros_like(mask_len)

            selected_probs = torch.where(
                unknown_map, selected_probs,
                torch.full_like(selected_probs, float("inf")),
            )
            selected_flat = selected_probs.reshape(B, L)
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(
                    selected_flat, mask_len)
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(
                    selected_flat, mask_len, temperature=temp)
            action_mask = action_mask_flat.reshape(
                B, self.action_horizon, self.action_dim)
            cur_seqs = torch.where(action_mask, self.mask_token_id, sampled)

            trace_holder["mask_matrix_per_step"].append(
                _mask_matrix(cur_seqs, self.mask_token_id))
            n_remaining = int(
                (cur_seqs == self.mask_token_id).sum().item())
            trace_holder["mask_remaining_per_step"].append(n_remaining)

        return self.binning.decode(cur_seqs)

    model.action_model.predict_action_realtime = types.MethodType(
        _instrumented_realtime, model.action_model)
    model.action_model.predict_action = types.MethodType(
        _instrumented_init, model.action_model)


# ═══════════════════════════════════════════════════════════════════
#  Episodic RTC simulation
# ═══════════════════════════════════════════════════════════════════

def run_rtc_episode(model, samples, chunk_len: int, n_actions: int,
                    inference_delay: int, hard_mask: bool, early_stop: bool,
                    fixed_steps: bool, decode_temperature: float,
                    choice_temperature: float, use_simple_max: bool,
                    instruction_override: str | None = None):
    """Walk through one episode in RTC cycles. Returns list of per-cycle dicts."""
    cycles = []
    prev_norm = None
    prev_final_mask = None  # [chunk_len, action_dim] uint8, 1=masked
    pos = 0
    cycle_i = 0
    pbar = tqdm(total=len(samples), desc="RTC cycles")
    while pos < len(samples):
        sample = samples[pos]
        example = {
            "image": sample["image"],
            "lang": instruction_override
                    if instruction_override is not None
                    else sample.get("lang", ""),
        }
        if "state" in sample:
            example["state"] = sample["state"]

        gt_chunk = np.array(sample["action"], dtype=np.float32)  # [chunk_len, 10]

        t0 = time.monotonic()
        if prev_norm is None:
            out = model.predict_action(
                examples=[example],
                decode_temperature=decode_temperature,
                choice_temperature=choice_temperature,
                use_simple_max=use_simple_max,
            )
        else:
            # First RTC cycle uses only the FIRST `inference_delay` actions
            # of the init chunk as prev, so the new chunk has just those
            # D positions as effective prefix and has to regenerate the rest.
            # Subsequent RTC cycles carry the tail of the previous cycle's
            # prediction (length chunk_len - n_actions), INCLUDING the mask
            # pattern from the previous cycle — positions that were still
            # masked when early_stop fired get regenerated here too.
            if cycle_i == 1:
                prev_chunk = prev_norm[:inference_delay][None, ...]
                prev_mask_slice = None
            else:
                prev_chunk = prev_norm[n_actions:][None, ...]
                if prev_final_mask is not None:
                    prev_mask_slice = prev_final_mask[n_actions:]
                else:
                    prev_mask_slice = None

            if prev_mask_slice is not None and prev_mask_slice.any():
                mask_t = torch.as_tensor(
                    prev_mask_slice, dtype=torch.bool,
                    device=next(model.action_model.parameters()).device,
                ).unsqueeze(0)
                model.action_model._prev_chunk_mask = mask_t
            else:
                model.action_model._prev_chunk_mask = None

            out = model.predict_action_realtime(
                examples=[example],
                prev_action_chunk_normalized=prev_chunk,
                inference_delay=inference_delay,
                execution_horizon=n_actions,
                decode_temperature=decode_temperature,
                choice_temperature=choice_temperature,
                use_simple_max=use_simple_max,
                fixed_steps=fixed_steps,
                hard_mask=hard_mask,
                early_stop=early_stop,
            )
            model.action_model._prev_chunk_mask = None
        trace = dict(model._early_stop_trace)
        trace["mask_remaining_per_step"] = list(trace["mask_remaining_per_step"])
        trace["mask_matrix_per_step"] = [m.copy() for m in trace["mask_matrix_per_step"]]
        if trace["mask_matrix_initial"] is not None:
            trace["mask_matrix_initial"] = trace["mask_matrix_initial"].copy()
        infer_ms = (time.monotonic() - t0) * 1000

        norm_pred = out["normalized_actions"][0].astype(np.float32)  # [chunk_len, 10]

        # Metrics vs the GT chunk at this timestep.
        full_mse = float(np.mean((norm_pred - gt_chunk) ** 2))
        tail_len = min(n_actions, chunk_len)
        tail_mse = float(np.mean(
            (norm_pred[-tail_len:] - gt_chunk[-tail_len:]) ** 2))

        cycles.append({
            "cycle": cycle_i,
            "pos": pos,
            "base_index": int(sample["_base_index"]),
            "trace": trace,
            "normalized_pred": norm_pred,
            "gt_action": gt_chunk,
            "full_mse": full_mse,
            "tail_mse": tail_mse,
            "infer_ms": infer_ms,
        })

        if trace["cycle_type"] == "rtc":
            print(f"  [cycle {cycle_i:3d}] pos={pos:4d}  base={sample['_base_index']:4d}  "
                  f"stop={trace['stop_step']}/{trace['num_steps_scheduled']}  "
                  f"early={trace['early_stopped']}  "
                  f"tail_mse={tail_mse:.5f}  infer={infer_ms:.0f}ms")
        else:
            print(f"  [cycle {cycle_i:3d}] pos={pos:4d}  base={sample['_base_index']:4d}  "
                  f"[init]  tail_mse={tail_mse:.5f}  infer={infer_ms:.0f}ms")

        prev_norm = norm_pred
        # Capture the final mask state from this cycle so the next cycle
        # can propagate the still-masked positions forward. If 0 decode
        # steps ran (pre-loop early-stop), fall back to the initial mask
        # so masked positions [prefix_length:] are still regenerated next
        # cycle instead of being treated as real prev values.
        if trace.get("mask_matrix_per_step"):
            prev_final_mask = trace["mask_matrix_per_step"][-1].copy()
        elif trace.get("mask_matrix_initial") is not None:
            prev_final_mask = trace["mask_matrix_initial"].copy()
        else:
            prev_final_mask = None
        cycle_i += 1
        pos += n_actions
        pbar.update(n_actions)
    pbar.close()
    return cycles


# ═══════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════

def plot_early_stop(cycles, out_path, num_inference_steps,
                    n_actions, hard_mask, early_stop):
    """Stop-step bar chart + mask-remaining heatmap across RTC cycles."""
    rtc = [c for c in cycles if c["trace"].get("cycle_type") == "rtc"]
    if not rtc:
        print("[WARN] No RTC cycles to plot.")
        return

    xs = [c["cycle"] for c in rtc]
    stop_steps = [c["trace"]["stop_step"] for c in rtc]
    scheduled = [c["trace"]["num_steps_scheduled"] for c in rtc]
    early_stopped = [c["trace"].get("early_stopped", False) for c in rtc]
    prefix_len = rtc[0]["trace"].get("prefix_length", "?")
    exec_h = rtc[0]["trace"].get("execution_horizon", "?")

    max_scheduled = max(s for s in scheduled if s is not None)

    fig, axes = plt.subplots(
        2, 1, figsize=(max(10, len(xs) * 0.15), 9),
        gridspec_kw={"height_ratios": [1, 1.3]})

    # Panel 1: stop-step bar chart
    ax1 = axes[0]
    colors = ["#2ecc71" if es else "#e74c3c" for es in early_stopped]
    ax1.bar(xs, stop_steps, color=colors, edgecolor="k", alpha=0.75,
            label="decode steps used")
    ax1.plot(xs, scheduled, color="k", linestyle="--", linewidth=1.2,
             marker="o", markersize=3, label="scheduled num_steps")
    ax1.axhline(num_inference_steps, color="gray", linestyle=":",
                alpha=0.7, label=f"self.num_inference_steps={num_inference_steps}")
    mask_label = "hard" if hard_mask else "natural"
    n_early = sum(early_stopped)
    ax1.set_title(
        f"Early-stop per RTC cycle  —  {mask_label} mask, "
        f"early_stop={early_stop}, n_actions={n_actions}, "
        f"prefix_len={prefix_len}, exec_h={exec_h}  "
        f"→ triggered {n_early}/{len(xs)}",
        fontsize=12)
    ax1.set_xlabel("RTC cycle")
    ax1.set_ylabel("decode step #")
    ax1.set_ylim(0, max(max_scheduled, num_inference_steps) + 1)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: mask-remaining heatmap
    ax2 = axes[1]
    heat = np.full((len(rtc), max_scheduled), np.nan)
    for i, c in enumerate(rtc):
        mr = c["trace"]["mask_remaining_per_step"]
        heat[i, :len(mr)] = mr
    im = ax2.imshow(heat.T, aspect="auto", origin="lower",
                    cmap="viridis", interpolation="nearest")
    ax2.set_xlabel("RTC cycle")
    ax2.set_ylabel("decode step idx")
    ax2.set_title(
        "# mask tokens remaining in non-prefix region after each decode step "
        "(red dot = stop step)")
    plt.colorbar(im, ax=ax2, label="# masked")
    for i, s in enumerate(stop_steps):
        if s is not None and 0 < s <= max_scheduled:
            ax2.scatter([i], [s - 1], color="red", s=18, zorder=5,
                        edgecolor="white", linewidths=0.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Early-stop plot saved to {out_path}")


def plot_mask_evolution(cycles, out_path, max_cycles=None):
    """Grid plot: rows = RTC cycles, cols = decode steps (initial + step 1..N).

    Each cell is a 16×10 binary heatmap of the action-token mask state
    (black = masked, white = fixed/unmasked). Shows how the MaskGIT decode
    fills in the chunk over the course of each RTC cycle, and how that
    pattern evolves along the trajectory.
    """
    if max_cycles is not None:
        cycles = cycles[:max_cycles]
    if not cycles:
        print("[WARN] No cycles to plot mask evolution.")
        return

    # Determine max # decode steps across cycles to set grid width
    max_steps = 0
    for c in cycles:
        max_steps = max(max_steps, len(c["trace"]["mask_matrix_per_step"]))
    if max_steps == 0:
        print("[WARN] No decode steps captured.")
        return
    n_cols = max_steps + 1  # initial + per-step
    n_rows = len(cycles)

    # Each subplot is small: 0.6" wide × 0.8" tall → ~16×10 aspect
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(0.7 * n_cols + 1.0, 0.55 * n_rows + 1.0),
        squeeze=False)
    cmap = plt.get_cmap("gray_r")  # 1 (masked) = black, 0 = white

    for r, c in enumerate(cycles):
        cycle_idx = c["cycle"]
        cyc_type = c["trace"].get("cycle_type", "?")
        stop_step = c["trace"].get("stop_step")
        prefix_len = c["trace"].get("prefix_length")
        initial = c["trace"].get("mask_matrix_initial")
        per_step = c["trace"].get("mask_matrix_per_step", [])

        for col in range(n_cols):
            ax = axes[r][col]
            ax.set_xticks([])
            ax.set_yticks([])

            if col == 0:
                mat = initial
                label = "init"
            else:
                step_idx = col - 1
                mat = per_step[step_idx] if step_idx < len(per_step) else None
                label = f"s{col}"

            if mat is None:
                ax.set_visible(False)
                continue

            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

            # Row label (leftmost)
            if col == 0:
                tag = "INIT" if cyc_type == "init" else f"RTC"
                ax.set_ylabel(
                    f"c{cycle_idx}\n{tag}",
                    rotation=0, labelpad=20, fontsize=7,
                    va="center", ha="right")

            # Column header (topmost row)
            if r == 0:
                ax.set_title(label, fontsize=7)

            # Outline the stop step in red (where decode actually ended)
            if stop_step is not None and col == stop_step:
                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(1.5)

            # Draw a horizontal line at prefix_length to separate prefix
            # from generated region (only shows up on cells where this
            # boundary is meaningful, i.e. RTC cycles with prefix > 0).
            if prefix_len and 0 < prefix_len < mat.shape[0]:
                ax.axhline(prefix_len - 0.5, color="#e67e22",
                           linewidth=0.8, linestyle="--")

    fig.suptitle(
        f"Mask matrix evolution — rows: RTC cycle, cols: decode step  "
        f"(black = masked, white = fixed; "
        f"red border = stop step; orange line = prefix boundary)",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mask-evolution plot saved to {out_path}")


def save_mask_patterns(cycles, out_path, episode_meta: dict | None = None):
    """Dump all captured mask matrices + per-cycle metadata to disk.

    Produces two sibling files:
      <stem>.npz       — all mask snapshots, indexed by (cycle, step)
      <stem>.meta.json — cycle-level metadata + episode-level config

    .npz layout:
      mask_matrices       : (R, H, D) uint8  — 1 = masked, 0 = fixed
      record_cycle        : (R,) int32       — cycle index
      record_step         : (R,) int32       — -1 for initial state, else
                                                0-indexed decode step
      record_is_init_cycle: (R,) bool        — True iff this cycle is the
                                                init cycle (no prefix)

    The flat layout means a single np.load + boolean masking can reconstruct
    any cycle's full step sequence:

        d = np.load("...masks.npz")
        mats = d["mask_matrices"][d["record_cycle"] == 3]   # cycle 3, all snaps
        steps = d["record_step"][d["record_cycle"] == 3]    # matching step idx
    """
    records_mat = []
    record_cycle = []
    record_step = []
    record_is_init = []
    cycle_meta = []

    for c in cycles:
        trace = c["trace"]
        is_init = trace.get("cycle_type") == "init"

        init_mat = trace.get("mask_matrix_initial")
        if init_mat is not None:
            records_mat.append(init_mat)
            record_cycle.append(c["cycle"])
            record_step.append(-1)
            record_is_init.append(is_init)

        for step_idx, mat in enumerate(trace.get("mask_matrix_per_step", [])):
            records_mat.append(mat)
            record_cycle.append(c["cycle"])
            record_step.append(step_idx)
            record_is_init.append(is_init)

        cycle_meta.append({
            "cycle": c["cycle"],
            "pos": c["pos"],
            "base_index": c["base_index"],
            "cycle_type": trace.get("cycle_type"),
            "num_steps_scheduled": trace.get("num_steps_scheduled"),
            "stop_step": trace.get("stop_step"),
            "prefix_length": trace.get("prefix_length"),
            "execution_horizon": trace.get("execution_horizon"),
            "early_stopped": trace.get("early_stopped"),
            "mask_remaining_per_step":
                trace.get("mask_remaining_per_step", []),
            "tail_mse": c["tail_mse"],
            "full_mse": c["full_mse"],
            "infer_ms": c["infer_ms"],
        })

    if records_mat:
        arr = np.stack(records_mat, axis=0).astype(np.uint8)
    else:
        arr = np.zeros((0, 0, 0), dtype=np.uint8)

    np.savez_compressed(
        out_path,
        mask_matrices=arr,
        record_cycle=np.array(record_cycle, dtype=np.int32),
        record_step=np.array(record_step, dtype=np.int32),
        record_is_init_cycle=np.array(record_is_init, dtype=bool),
    )
    meta_path = out_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "episode_meta": episode_meta or {},
            "cycles": cycle_meta,
        }, f, indent=2)
    print(f"Mask patterns  saved to {out_path}  "
          f"({arr.shape[0]} snapshots, shape per snap = {arr.shape[1:]})")
    print(f"Mask metadata  saved to {meta_path}")


def plot_ending_pattern(cycles, out_path):
    """Per-cycle summary: initial mask (what each inference had to fill)
    and final mask (what the inference left masked at the end) laid out
    as a single row per cycle. Does NOT track intermediate decode steps.

    Rendered as one figure with two columns:
      col 0 = mask_matrix_initial  (prefix structure going in)
      col 1 = mask_matrix_per_step[-1]  (state after the final decode step;
              usually all-white when decode runs to completion, but
              highlights early-stop points if any cycle stopped short)
    """
    if not cycles:
        print("[WARN] No cycles to plot ending pattern.")
        return
    n = len(cycles)
    fig, axes = plt.subplots(
        n, 2, figsize=(2.6, max(1.2, 0.45 * n)), squeeze=False)
    cmap = plt.get_cmap("gray_r")

    for i, c in enumerate(cycles):
        trace = c["trace"]
        init_mat = trace.get("mask_matrix_initial")
        steps = trace.get("mask_matrix_per_step", [])
        final_mat = steps[-1] if steps else None
        prefix_len = trace.get("prefix_length")
        stop_step = trace.get("stop_step")
        sched = trace.get("num_steps_scheduled")

        for col, (mat, name) in enumerate([
                (init_mat, "init"), (final_mat, "final")]):
            ax = axes[i][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if mat is None:
                ax.set_visible(False)
                continue
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            if prefix_len and 0 < prefix_len < mat.shape[0]:
                ax.axhline(prefix_len - 0.5, color="#e67e22",
                           linewidth=0.8, linestyle="--")
            if i == 0:
                ax.set_title(name, fontsize=8)

        tag = "INIT" if trace.get("cycle_type") == "init" else "RTC"
        step_info = ""
        if sched is not None:
            step_info = f" {stop_step}/{sched}"
        axes[i][0].set_ylabel(
            f"c{c['cycle']}\n{tag}{step_info}",
            rotation=0, labelpad=22, fontsize=7,
            va="center", ha="right")

    fig.suptitle(
        "Per-cycle mask pattern: inference start vs end\n"
        "(black=masked, white=fixed; orange line=prefix boundary)",
        fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Ending-pattern plot saved to {out_path}")


def plot_mse_traj(cycles, out_path):
    """MSE (tail + full-chunk) per RTC cycle."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    xs = [c["cycle"] for c in cycles]
    tail = [c["tail_mse"] for c in cycles]
    full = [c["full_mse"] for c in cycles]
    ax.plot(xs, tail, "-o", markersize=4, label="tail (last n_actions) MSE")
    ax.plot(xs, full, "-s", markersize=4, alpha=0.7, label="full-chunk MSE")
    for c in cycles:
        if c["trace"].get("cycle_type") == "init":
            ax.axvline(c["cycle"], color="gray", linestyle=":", alpha=0.7,
                       label="init cycle" if c["cycle"] == 0 else None)
    ax.set_xlabel("RTC cycle")
    ax.set_ylabel("MSE (normalized action)")
    ax.set_title("Prediction MSE vs GT across RTC cycles")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"MSE plot saved to {out_path}")


# ═══════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Episodic RTC DD open-loop eval (natural schedule) with "
                    "early-stop tracing")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR)
    parser.add_argument("--episode_idx", type=int, default=0,
                        help="Index (within the single underlying dataset) of "
                             "the episode to walk through")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--n_actions", type=int, default=4,
                        help="RTC execution horizon (stride between cycles)")
    parser.add_argument("--inference_delay", type=int, default=4,
                        help="Defaults to n_actions")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override dataset language (default: sample's own)")
    # decode knobs
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    # schedule knobs
    parser.add_argument("--hard_mask", action="store_true", default=False,
                        help="Use hard-mask schedule. Default is natural "
                             "(prefix_length = action_horizon - execution_horizon).")
    parser.add_argument("--early_stop", dest="early_stop", action="store_true",
                        default=True,
                        help="Enable early-stop (break decode when no masked "
                             "positions remain). Default: True.")
    parser.add_argument("--no_early_stop", dest="early_stop",
                        action="store_false")
    parser.add_argument("--fixed_steps", dest="fixed_steps",
                        action="store_true", default=True,
                        help="Use the full num_inference_steps regardless of "
                             "fraction of masked tokens (default: True).")
    parser.add_argument("--no_fixed_steps", dest="fixed_steps",
                        action="store_false")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON (default: auto in checkpoint dir)")
    parser.add_argument("--vis_max_cycles", type=int, default=None,
                        help="Cap the number of cycles rendered in the mask "
                             "evolution grid (default: all)")
    args = parser.parse_args()

    if args.inference_delay is None:
        args.inference_delay = args.n_actions

    # Load model
    model = load_model(args.checkpoint)
    chunk_len = model.chunk_len
    num_inference_steps = getattr(
        model.action_model, "num_inference_steps", None)

    assert args.n_actions + args.inference_delay <= chunk_len, (
        f"n_actions ({args.n_actions}) + inference_delay "
        f"({args.inference_delay}) must be <= chunk_len ({chunk_len})")

    # Load one episode
    samples, _ = load_episode(
        model, args.data_root_dir, args.episode_idx,
        include_state=args.include_state)

    # Install instrumented decode
    install_instrumented_decode(model)

    # Run RTC
    print(f"\n{'=' * 60}")
    print(f"  Episodic RTC — DD natural-schedule early-stop trace")
    print(f"  n_actions       : {args.n_actions}")
    print(f"  inference_delay : {args.inference_delay}")
    print(f"  chunk_len       : {chunk_len}")
    print(f"  num_inference   : {num_inference_steps}")
    print(f"  hard_mask       : {args.hard_mask}  ({'hard' if args.hard_mask else 'natural'} schedule)")
    print(f"  early_stop      : {args.early_stop}")
    print(f"  fixed_steps     : {args.fixed_steps}")
    print(f"  episode_frames  : {len(samples)}")
    print(f"{'=' * 60}\n")

    cycles = run_rtc_episode(
        model, samples, chunk_len=chunk_len,
        n_actions=args.n_actions, inference_delay=args.inference_delay,
        hard_mask=args.hard_mask, early_stop=args.early_stop,
        fixed_steps=args.fixed_steps,
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
        instruction_override=args.instruction,
    )

    # Summary
    rtc_cycles = [c for c in cycles if c["trace"].get("cycle_type") == "rtc"]
    n_early = sum(1 for c in rtc_cycles if c["trace"].get("early_stopped"))
    print(f"\n{'=' * 60}")
    print(f"  RTC cycles total : {len(cycles)}  (rtc={len(rtc_cycles)}, init=1)")
    print(f"  Early-stopped    : {n_early}/{len(rtc_cycles)}  "
          f"({100 * n_early / max(len(rtc_cycles), 1):.1f}%)")
    if rtc_cycles:
        stop_arr = np.array([c["trace"]["stop_step"] for c in rtc_cycles])
        sched_arr = np.array([c["trace"]["num_steps_scheduled"] for c in rtc_cycles])
        print(f"  Mean stop step   : {stop_arr.mean():.2f}  (of scheduled mean {sched_arr.mean():.2f})")
        print(f"  Mean speed-up    : {(sched_arr - stop_arr).mean():.2f} steps saved / cycle")
    tail_mses = [c["tail_mse"] for c in cycles]
    print(f"  Mean tail MSE    : {np.mean(tail_mses):.6f}")
    print(f"{'=' * 60}")

    # Save JSON
    ckpt_name = Path(args.checkpoint).stem.replace("_pytorch_model", "")
    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = Path(args.checkpoint).parents[1]
        out_path = run_dir / (
            f"rtc_natural_earlystop_{ckpt_name}_ep{args.episode_idx}"
            f"_n{args.n_actions}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    json_cycles = []
    for c in cycles:
        # Strip bulky numpy arrays from the trace before JSON dump; keep
        # the shapes + counts so the JSON stays readable.
        trace_json = {}
        for k, v in c["trace"].items():
            if k in ("mask_matrix_initial", "mask_matrix_per_step"):
                continue
            trace_json[k] = v
        trace_json["n_mask_matrices_captured"] = len(
            c["trace"].get("mask_matrix_per_step", []))
        json_cycles.append({
            "cycle": c["cycle"],
            "pos": c["pos"],
            "base_index": c["base_index"],
            "trace": trace_json,
            "full_mse": c["full_mse"],
            "tail_mse": c["tail_mse"],
            "infer_ms": c["infer_ms"],
        })
    results_json = {
        "checkpoint": args.checkpoint,
        "episode_idx": args.episode_idx,
        "n_actions": args.n_actions,
        "inference_delay": args.inference_delay,
        "chunk_len": chunk_len,
        "num_inference_steps": num_inference_steps,
        "hard_mask": args.hard_mask,
        "early_stop": args.early_stop,
        "fixed_steps": args.fixed_steps,
        "decode_temperature": args.decode_temperature,
        "choice_temperature": args.choice_temperature,
        "n_cycles": len(cycles),
        "n_early_stopped": n_early,
        "cycles": json_cycles,
    }
    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to {out_path}")

    # Plots
    plot_early_stop(
        cycles, out_path.with_name(out_path.stem + "_early_stop.png"),
        num_inference_steps=num_inference_steps,
        n_actions=args.n_actions,
        hard_mask=args.hard_mask, early_stop=args.early_stop)
    plot_mse_traj(
        cycles, out_path.with_name(out_path.stem + "_mse.png"))
    plot_mask_evolution(
        cycles, out_path.with_name(out_path.stem + "_mask_evo.png"),
        max_cycles=args.vis_max_cycles)
    plot_ending_pattern(
        cycles, out_path.with_name(out_path.stem + "_ending.png"))
    save_mask_patterns(
        cycles, out_path.with_name(out_path.stem + "_masks.npz"),
        episode_meta={
            "checkpoint": args.checkpoint,
            "episode_idx": args.episode_idx,
            "n_actions": args.n_actions,
            "inference_delay": args.inference_delay,
            "chunk_len": chunk_len,
            "num_inference_steps": num_inference_steps,
            "hard_mask": args.hard_mask,
            "early_stop": args.early_stop,
            "fixed_steps": args.fixed_steps,
            "decode_temperature": args.decode_temperature,
            "choice_temperature": args.choice_temperature,
        })


if __name__ == "__main__":
    main()
