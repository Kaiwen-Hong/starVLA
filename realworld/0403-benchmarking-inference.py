#!/usr/bin/env python3
"""
Inference benchmarking for QwenPI (flow-matching) and QwenDiscreteDiffusion (MaskGIT).

Measures detailed per-stage timing with per-inference-step breakdown:
  - Image preprocessing
  - QwenVL forward (VLM backbone)
  - Action model: state encoding, init, each denoising/MaskGIT step, final decode
  - Total predict_action / predict_action_realtime time

With --inference_delay 0: benchmarks predict_action (full generation).
With --inference_delay N: first call uses predict_action, then all subsequent
    calls use predict_action_realtime with the previous prediction's first N
    actions as the fixed prefix — matching real deployment (closedloop_rtc.py).

Usage:
    python realworld/0403-benchmarking-inference.py
    python realworld/0403-benchmarking-inference.py --inference_delay 4
    python realworld/0403-benchmarking-inference.py --inference_delay 0 --num_inference_steps 8
"""

import warnings
warnings.filterwarnings("ignore", message=".*video decoding and encoding.*torchvision.*")

import argparse
import sys
import os
import time
import json
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Repo setup ──────────────────────────────────────────────────────
REPO_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_DIR))
os.chdir(REPO_DIR)

from starVLA.model.framework.share_tools import read_mode_config, dict_to_namespace
from starVLA.model.framework import build_framework
from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

# ── Defaults ─────────────────────────────────────────────────────────
DEFAULT_CHECKPOINT_PI = (
    "/scratch/wangpc/starVLA/results/Checkpoints/fastumi_pickandplace_qwenPI_329v4/"
    "checkpoints/steps_15000_pytorch_model.pt"
)
DEFAULT_CHECKPOINT_DD = (
    "/scratch/wangpc/starVLA/results/Checkpoints/fastumi_pickandplace_qwenDiscreteDiffusion_329v4/"
    "checkpoints/steps_20000_pytorch_model.pt"
)
# ↑↑↑ Change this to switch between PI and DD ↑↑↑
DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_DD
# DEFAULT_CHECKPOINT = DEFAULT_CHECKPOINT_PI

DATA_ROOT_DIR = "/scratch/wangpc/starVLA/playground/Datasets/FastUMI"
DECODE_TEMPERATURE = 0.0
CHOICE_TEMPERATURE = 0.1
# ───────────────────────────────────────────────────────────────────


def _detect_attn_implementation():
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        print("[INFO] flash_attn not available, falling back to sdpa")
        return "sdpa"


# ── Timer utility ──────────────────────────────────────────────────
class CUDATimer:
    """Accumulates CUDA-synchronized timings for named stages."""

    def __init__(self):
        self.records = {}  # name -> list of durations (ms)

    @contextmanager
    def track(self, name: str):
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.records.setdefault(name, []).append(elapsed_ms)

    def summary(self) -> dict:
        out = {}
        for name, vals in self.records.items():
            arr = np.array(vals)
            out[name] = {
                "mean_ms": float(np.mean(arr)),
                "std_ms": float(np.std(arr)),
                "min_ms": float(np.min(arr)),
                "max_ms": float(np.max(arr)),
                "median_ms": float(np.median(arr)),
                "count": len(vals),
            }
        return out


# ======================================================================
# Inlined action model predict_action with per-step timing
# ======================================================================

def _action_model_predict_pi_timed(am, vl_embs_list, state, timer: CUDATimer):
    """Inlined LayerwiseFM_ActionHeader.predict_action with per-step timing."""
    batch_size = vl_embs_list[0].shape[0]
    device = vl_embs_list[0].device

    with timer.track("am_init"):
        actions = torch.randn(
            size=(batch_size, am.action_horizon, am.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )
        num_steps = am.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = am.state_encoder(state) if state is not None else None

    for t in range(num_steps):
        with timer.track(f"am_step_{t}"):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * am.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            action_features = am.action_encoder(actions, timesteps_tensor)

            if am.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = am.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = am.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            temb = am.model.timestep_encoder(timesteps_tensor)

            model_output = sa_embs
            for layer_idx, layer in enumerate(am.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    temb=temb,
                )

            pred = am.action_decoder(model_output)
            pred_velocity = pred[:, -am.action_horizon:]
            actions = actions + dt * pred_velocity

    return actions


def _action_model_predict_dd_timed(am, vl_embs_list, state, timer: CUDATimer,
                                    decode_temperature=0.0, choice_temperature=0.1,
                                    use_simple_max=False):
    """Inlined LayerwiseDiscreteDiffusion_ActionHeader.predict_action with per-step timing."""
    from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
        decode_mask_schedule, mask_by_deterministic_lowest, mask_by_random_topk,
    )

    B = vl_embs_list[0].shape[0]
    device = vl_embs_list[0].device
    L = am.seq_len

    with timer.track("am_init"):
        cur_seqs = torch.full(
            (B, am.action_horizon, am.action_dim),
            am.mask_token_id,
            dtype=torch.long,
            device=device,
        )

        state_feat = None
        if state is not None and am.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = am.state_encoder(state).unsqueeze(1)

    if use_simple_max:
        with timer.track("am_step_0"):
            logits = am._forward_logits(vl_embs_list, cur_seqs, state_feat, device)
        with timer.track("am_decode"):
            result = am.binning.decode_logits(logits)
        return result

    deterministic_decode = decode_temperature == 0
    deterministic_choice = choice_temperature == 0
    unknown_init = torch.full((B,), L, dtype=torch.long, device=device)

    for step_idx in range(am.num_inference_steps):
        with timer.track(f"am_step_{step_idx}"):
            with timer.track(f"am_step_{step_idx}_forward_logits"):
                logits = am._forward_logits(vl_embs_list, cur_seqs, state_feat, device)

            with timer.track(f"am_step_{step_idx}_sample"):
                safe_temp = max(decode_temperature, 1e-8)
                sampled, selected_probs = am.binning.sample_indices_from_logits(
                    logits,
                    temperature=safe_temp,
                    deterministic=deterministic_decode,
                )

            with timer.track(f"am_step_{step_idx}_mask_schedule"):
                unknown_map = cur_seqs == am.mask_token_id
                sampled = torch.where(unknown_map, sampled, cur_seqs)

                ratio = (step_idx + 1.0) / am.num_inference_steps
                mask_ratio = decode_mask_schedule(
                    torch.tensor(ratio, device=device), am.decode_schedule
                )
                mask_len = (unknown_init.float() * mask_ratio).long()
                min_len = torch.full_like(mask_len, 1)
                max_len = (unknown_init - 1).clamp(min=0)
                mask_len = mask_len.clamp(min=min_len, max=max_len)
                if step_idx == am.num_inference_steps - 1:
                    mask_len = torch.zeros_like(mask_len)

            with timer.track(f"am_step_{step_idx}_remask"):
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    torch.full_like(selected_probs, float("inf")),
                )

                selected_flat = selected_probs.reshape(B, L)
                if deterministic_choice:
                    action_mask_flat = mask_by_deterministic_lowest(selected_flat, mask_len)
                else:
                    temp = choice_temperature * (1.0 - ratio)
                    action_mask_flat = mask_by_random_topk(selected_flat, mask_len, temperature=temp)
                action_mask = action_mask_flat.reshape(B, am.action_horizon, am.action_dim)
                cur_seqs = torch.where(action_mask, am.mask_token_id, sampled)

    with timer.track("am_decode"):
        result = am.binning.decode(cur_seqs)

    return result


# ======================================================================
# Inlined action model predict_action_realtime with per-step timing
# ======================================================================

def _action_model_predict_realtime_pi_timed(am, vl_embs_list, state, timer: CUDATimer,
                                             prev_action_chunk, inference_delay,
                                             mode="pigdm",
                                             suffix_length=None,
                                             prefix_attention_schedule="exp",
                                             max_guidance_weight=10.0):
    """Inlined LayerwiseFM_ActionHeader.predict_action_realtime with per-step timing."""
    from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_prefix_weights

    batch_size = vl_embs_list[0].shape[0]
    device = vl_embs_list[0].device
    dtype = vl_embs_list[0].dtype

    with timer.track("am_init"):
        actions = torch.randn(
            size=(batch_size, am.action_horizon, am.action_dim),
            dtype=dtype,
            device=device,
        )
        num_steps = am.num_inference_timesteps
        dt = 1.0 / num_steps
        state_features = am.state_encoder(state) if state is not None else None

    if mode == "pigdm":
        if suffix_length is None:
            suffix_length = inference_delay
        prefix_attention_end = am.action_horizon - suffix_length
        weights = get_prefix_weights(
            inference_delay, prefix_attention_end,
            am.action_horizon, prefix_attention_schedule, device,
        )
        prev = prev_action_chunk.to(dtype)

        for t_step in range(num_steps):
            with timer.track(f"am_step_{t_step}"):
                t_cont = t_step / float(num_steps)
                t_bucket = int(t_cont * am.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    (batch_size,), t_bucket, device=device, dtype=torch.long,
                )

                with torch.enable_grad():
                    x_t = actions.detach().requires_grad_(True)
                    v_t = am._forward_velocity(
                        vl_embs_list, x_t, timesteps_tensor,
                        state_features, device,
                    )
                    x_1_hat = x_t + v_t * (1.0 - t_cont)
                    error = (prev - x_1_hat) * weights[None, :, None]
                    pinv_correction = torch.autograd.grad(
                        x_1_hat, x_t, grad_outputs=error,
                    )[0]

                t = t_cont
                inv_r2 = (t ** 2 + (1 - t) ** 2) / ((1 - t) ** 2 + 1e-8)
                c = (1 - t) / (t + 1e-8)
                gw = min(c * inv_r2, max_guidance_weight)

                v_corrected = v_t.detach() + gw * pinv_correction.detach()
                actions = actions.detach() + dt * v_corrected

        return actions

    # simulated_delay mode
    from starVLA.model.modules.action_model.flow_matching_head.action_encoder import swish

    with timer.track("am_init_sd"):
        prefix_mask = torch.zeros(1, am.action_horizon, 1, device=device, dtype=torch.bool)
        prefix_mask[:, :inference_delay, :] = True

    for t_step in range(num_steps):
        with timer.track(f"am_step_{t_step}"):
            t_cont = t_step / float(num_steps)

            actions = torch.where(prefix_mask, prev_action_chunk.to(dtype), actions)

            t_bucket = int(t_cont * am.num_timestep_buckets)
            t_bucket_prefix = int(1.0 * am.num_timestep_buckets) - 1
            timesteps_tensor = torch.full(
                (batch_size, am.action_horizon), t_bucket, device=device, dtype=torch.long
            )
            timesteps_tensor[:, :inference_delay] = t_bucket_prefix

            a_emb = am.action_encoder.layer1(actions)
            tau_emb = am.action_encoder.pos_encoding(timesteps_tensor).to(dtype=a_emb.dtype)
            x_enc = torch.cat([a_emb, tau_emb], dim=-1)
            x_enc = swish(am.action_encoder.layer2(x_enc))
            action_features = am.action_encoder.layer3(x_enc)

            if am.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = am.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = am.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            temb_tensor = torch.full((batch_size,), t_bucket, device=device, dtype=torch.long)
            temb = am.model.timestep_encoder(temb_tensor)

            model_output = sa_embs
            for layer_idx, layer in enumerate(am.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    temb=temb,
                )

            pred = am.action_decoder(model_output)
            pred_velocity = pred[:, -am.action_horizon:]

            actions = actions + dt * pred_velocity
            actions = torch.where(prefix_mask, prev_action_chunk.to(dtype), actions)

    return actions


def _action_model_predict_realtime_dd_timed(am, vl_embs_list, state, timer: CUDATimer,
                                             prev_action_chunk, inference_delay,
                                             decode_temperature=0.0, choice_temperature=0.1,
                                             hard_mask=False, execution_horizon=None):
    """Inlined LayerwiseDiscreteDiffusion_ActionHeader.predict_action_realtime with per-step timing."""
    from starVLA.model.modules.action_model.LayerwiseDiscreteDiffusion_ActionHeader import (
        decode_mask_schedule, mask_by_deterministic_lowest, mask_by_random_topk,
    )

    B = vl_embs_list[0].shape[0]
    device = vl_embs_list[0].device
    L = am.seq_len
    deterministic_decode = decode_temperature == 0
    deterministic_choice = choice_temperature == 0

    if execution_horizon is None:
        execution_horizon = inference_delay
    execution_horizon = min(execution_horizon, am.action_horizon)

    if hard_mask:
        prefix_length = inference_delay
    else:
        prefix_length = am.action_horizon - execution_horizon

    with timer.track("am_init"):
        prefix_bins = am.binning.encode(prev_action_chunk)
        prefix_mask = (
            torch.arange(am.action_horizon, device=device)[None, :, None]
            < prefix_length
        ).expand(B, am.action_horizon, am.action_dim)

        cur_seqs = torch.where(
            prefix_mask,
            prefix_bins,
            torch.full_like(prefix_bins, am.mask_token_id),
        )

        # Count actual masked tokens
        num_unknown_tokens = (cur_seqs == am.mask_token_id).sum(dim=(1, 2))
        unknown_init = num_unknown_tokens
        num_steps = max(1, int(am.num_inference_steps * num_unknown_tokens.max().item() / L))

        state_feat = None
        if state is not None and am.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_feat = am.state_encoder(state).unsqueeze(1)

    for step_idx in range(num_steps):
        with timer.track(f"am_step_{step_idx}"):
            with timer.track(f"am_step_{step_idx}_forward_logits"):
                logits = am._forward_logits(vl_embs_list, cur_seqs, state_feat, device)

            with timer.track(f"am_step_{step_idx}_sample"):
                safe_temp = max(decode_temperature, 1e-8)
                sampled, selected_probs = am.binning.sample_indices_from_logits(
                    logits,
                    temperature=safe_temp,
                    deterministic=deterministic_decode,
                )

            with timer.track(f"am_step_{step_idx}_mask_schedule"):
                unknown_map = cur_seqs == am.mask_token_id
                sampled = torch.where(prefix_mask, prefix_bins, sampled)
                sampled = torch.where(unknown_map, sampled, cur_seqs)

                ratio = (step_idx + 1.0) / num_steps
                mask_ratio = decode_mask_schedule(
                    torch.tensor(ratio, device=device), am.decode_schedule
                )
                mask_len = (unknown_init.float() * mask_ratio).long()
                min_len = torch.full_like(mask_len, 1)
                max_len = (unknown_init - 1).clamp(min=0)
                mask_len = mask_len.clamp(min=min_len, max=max_len)
                if step_idx == num_steps - 1:
                    mask_len = torch.zeros_like(mask_len)

            with timer.track(f"am_step_{step_idx}_remask"):
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    torch.full_like(selected_probs, float("inf")),
                )

                selected_flat = selected_probs.reshape(B, L)
                if deterministic_choice:
                    action_mask_flat = mask_by_deterministic_lowest(selected_flat, mask_len)
                else:
                    temp = choice_temperature * (1.0 - ratio)
                    action_mask_flat = mask_by_random_topk(selected_flat, mask_len, temperature=temp)
                action_mask = action_mask_flat.reshape(B, am.action_horizon, am.action_dim)
                cur_seqs = torch.where(
                    prefix_mask,
                    prefix_bins,
                    torch.where(action_mask, am.mask_token_id, sampled),
                )

    with timer.track("am_decode"):
        result = am.binning.decode(cur_seqs)

    return result


# ======================================================================
# VLM forward (shared by predict_action and predict_action_realtime)
# ======================================================================

def _vlm_forward_pi(model, examples, timer: CUDATimer):
    """QwenPI: preprocess + VLM forward, returns (vl_embs_list, state tensor)."""
    if not isinstance(examples, list):
        examples = [examples]

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.QwenPI import resize_images

    with timer.track("preprocess"):
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        train_obs_image_size = getattr(model.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    with timer.track("vlm_build_inputs"):
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

    with timer.track("vlm_forward"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(model.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

    with timer.track("state_to_device"):
        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None else None
        )

    return vl_embs_list, state


def _vlm_forward_dd(model, examples, timer: CUDATimer):
    """QwenDiscreteDiffusion: preprocess + VLM forward, returns (vl_embs_list, state tensor)."""
    if not isinstance(examples, list):
        examples = [examples]

    from deployment.model_server.tools.image_tools import to_pil_preserve
    from starVLA.model.framework.QwenDiscreteDiffusion import resize_images

    with timer.track("preprocess"):
        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = [example["state"] for example in examples] if "state" in examples[0] else None
        train_obs_image_size = getattr(model.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

    with timer.track("vlm_build_inputs"):
        qwen_inputs = model.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )

    with timer.track("vlm_forward"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = model.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(model.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            while len(vl_embs_list) < expected_layers:
                vl_embs_list.append(vl_embs_list[-1])
            base_hidden = vl_embs_list[-1]

    with timer.track("state_to_device"):
        state = (
            torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype)
            if state is not None else None
        )

    return vl_embs_list, state


# ======================================================================
# Full timed predict_action / predict_action_realtime
# ======================================================================

def _timed_predict_action_pi(model, examples, timer: CUDATimer, **kwargs):
    """QwenPI predict_action with per-stage + per-step timing."""
    vl_embs_list, state = _vlm_forward_pi(model, examples, timer)

    with torch.autocast("cuda", dtype=torch.float32):
        pred_actions = _action_model_predict_pi_timed(
            model.action_model, vl_embs_list, state, timer
        )

    with timer.track("to_numpy"):
        normalized_actions = pred_actions.detach().cpu().numpy()

    return {"normalized_actions": normalized_actions}


def _timed_predict_action_dd(model, examples, timer: CUDATimer, **kwargs):
    """QwenDiscreteDiffusion predict_action with per-stage + per-step timing."""
    vl_embs_list, state = _vlm_forward_dd(model, examples, timer)

    decode_temperature = kwargs.get("decode_temperature", DECODE_TEMPERATURE)
    choice_temperature = kwargs.get("choice_temperature", CHOICE_TEMPERATURE)
    use_simple_max = kwargs.get("use_simple_max", False)

    with torch.autocast("cuda", dtype=torch.float32):
        pred_actions = _action_model_predict_dd_timed(
            model.action_model, vl_embs_list, state, timer,
            decode_temperature=decode_temperature,
            choice_temperature=choice_temperature,
            use_simple_max=use_simple_max,
        )

    with timer.track("to_numpy"):
        normalized_actions = pred_actions.detach().float().cpu().numpy()

    return {"normalized_actions": normalized_actions}


def _timed_predict_realtime_pi(model, examples, timer: CUDATimer,
                                prev_action_chunk_normalized=None,
                                inference_delay=1, **kwargs):
    """QwenPI predict_action_realtime with per-stage + per-step timing."""
    vl_embs_list, state = _vlm_forward_pi(model, examples, timer)

    prev_chunk_t = torch.from_numpy(
        np.array(prev_action_chunk_normalized)
    ).to(vl_embs_list[-1].device, dtype=torch.float32)

    rtc_mode = kwargs.get("rtc_mode", "pigdm")

    with torch.autocast("cuda", dtype=torch.float32):
        pred_actions = _action_model_predict_realtime_pi_timed(
            model.action_model, vl_embs_list, state, timer,
            prev_action_chunk=prev_chunk_t,
            inference_delay=inference_delay,
            mode=rtc_mode,
        )

    with timer.track("to_numpy"):
        normalized_actions = pred_actions.detach().float().cpu().numpy()

    return {"normalized_actions": normalized_actions}


def _timed_predict_realtime_dd(model, examples, timer: CUDATimer,
                                prev_action_chunk_normalized=None,
                                inference_delay=1, **kwargs):
    """QwenDiscreteDiffusion predict_action_realtime with per-stage + per-step timing."""
    vl_embs_list, state = _vlm_forward_dd(model, examples, timer)

    decode_temperature = kwargs.get("decode_temperature", DECODE_TEMPERATURE)
    choice_temperature = kwargs.get("choice_temperature", CHOICE_TEMPERATURE)

    prev_chunk_t = torch.from_numpy(
        np.array(prev_action_chunk_normalized)
    ).to(vl_embs_list[-1].device, dtype=torch.float32)

    hard_mask = kwargs.get("hard_mask", False)

    with torch.autocast("cuda", dtype=torch.float32):
        pred_actions = _action_model_predict_realtime_dd_timed(
            model.action_model, vl_embs_list, state, timer,
            prev_action_chunk=prev_chunk_t,
            inference_delay=inference_delay,
            decode_temperature=decode_temperature,
            choice_temperature=choice_temperature,
            hard_mask=hard_mask,
        )

    with timer.track("to_numpy"):
        normalized_actions = pred_actions.detach().float().cpu().numpy()

    return {"normalized_actions": normalized_actions}


# ── Model / data loading ─────────────────────────────────────────
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

    # Detect framework type
    cls_name = type(model).__name__
    if "PI" in cls_name or "pi" in cls_name.lower():
        framework_type = "QwenPI"
    elif "Discrete" in cls_name or "DD" in cls_name:
        framework_type = "QwenDiscreteDiffusion"
    else:
        framework_type = cls_name
    print(f"  Framework: {framework_type} ({cls_name})")

    # Print action model info
    am = model.action_model
    if hasattr(am, "num_inference_timesteps"):
        print(f"  Flow-matching inference steps: {am.num_inference_timesteps}")
    if hasattr(am, "num_inference_steps"):
        print(f"  MaskGIT inference steps: {am.num_inference_steps}")
    if hasattr(am, "action_horizon"):
        print(f"  Action horizon: {am.action_horizon}")
    if hasattr(am, "action_dim"):
        print(f"  Action dim: {am.action_dim}")

    return model, framework_type


def load_dataset(model, include_state=False, data_root_dir=None):
    data_cfg = model.config.datasets.vla_data
    if include_state:
        data_cfg.include_state = True
    if data_root_dir:
        data_cfg.data_root_dir = data_root_dir

    print(f"Loading dataset: {data_cfg.data_mix}")
    dataset = get_vla_dataset(data_cfg=data_cfg)
    print(f"  total steps: {len(dataset)}")

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )
    return dataloader


# ── Main benchmark loop ───────────────────────────────────────────
def benchmark(model, framework_type, dataloader, num_samples, warmup,
              inference_delay, infer_kwargs):
    timer_predict = CUDATimer()
    timer_realtime = CUDATimer()
    total_predict_times = []
    total_realtime_times = []

    if framework_type == "QwenPI":
        fn_predict = _timed_predict_action_pi
        fn_realtime = _timed_predict_realtime_pi
    else:
        fn_predict = _timed_predict_action_dd
        fn_realtime = _timed_predict_realtime_dd

    use_rtc = inference_delay > 0

    count = 0
    prev_actions = None

    for batch in tqdm(dataloader, total=min(num_samples + warmup, len(dataloader)),
                      desc="Benchmarking"):
        if count >= num_samples + warmup:
            break

        is_warmup = count < warmup

        if not use_rtc or prev_actions is None:
            # ── predict_action (full generation) ──
            cur_timer = CUDATimer() if is_warmup else timer_predict

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.inference_mode():
                result = fn_predict(model, batch, cur_timer, **infer_kwargs)
            torch.cuda.synchronize()
            t_total = (time.perf_counter() - t0) * 1000.0
            if not is_warmup:
                total_predict_times.append(t_total)
        else:
            # ── predict_action_realtime (with prefix from previous prediction) ──
            # Simulate the shift logic from closedloop_rtc.py:
            #   shifted[:remaining] = prev[n_actions:]  (shift out consumed actions)
            # Here we simply use prev_actions as-is (the benchmark tests the same
            # image each iteration, so shifting doesn't affect timing).
            cur_timer = CUDATimer() if is_warmup else timer_realtime

            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                result = fn_realtime(
                    model, batch, cur_timer,
                    prev_action_chunk_normalized=prev_actions,
                    inference_delay=inference_delay,
                    **infer_kwargs,
                )
            torch.cuda.synchronize()
            t_total = (time.perf_counter() - t0) * 1000.0
            if not is_warmup:
                total_realtime_times.append(t_total)

        prev_actions = result["normalized_actions"]  # [B, T, D]
        count += 1

    # Detect num inference steps
    am = model.action_model
    if hasattr(am, "num_inference_timesteps"):
        n_steps_full = am.num_inference_timesteps
    elif hasattr(am, "num_inference_steps"):
        n_steps_full = am.num_inference_steps
    else:
        n_steps_full = 0

    if use_rtc:
        # Compute the scaled RTC steps (matching deployment fixed_steps=False)
        if framework_type == "QwenDiscreteDiffusion":
            n_steps_rtc = max(1, int(n_steps_full * inference_delay / am.action_horizon))
        else:
            n_steps_rtc = n_steps_full  # PI always uses full steps for RTC
    else:
        n_steps_rtc = 0

    results = {
        "num_samples": num_samples,
        "warmup": warmup,
        "inference_delay": inference_delay,
        "framework": framework_type,
        "num_inference_steps": n_steps_full,
        "num_inference_steps_rtc": n_steps_rtc,
        "action_horizon": am.action_horizon,
    }

    if total_predict_times:
        results["predict_action"] = {
            "total_ms": _stats(total_predict_times),
            **timer_predict.summary(),
        }

    if total_realtime_times:
        results["predict_action_realtime"] = {
            "total_ms": _stats(total_realtime_times),
            **timer_realtime.summary(),
        }

    return results


def _stats(vals):
    if not vals:
        return {}
    arr = np.array(vals)
    return {
        "mean_ms": float(np.mean(arr)),
        "std_ms": float(np.std(arr)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
        "count": len(vals),
    }


def _print_mode(label, data):
    """Print breakdown for one mode (predict_action or predict_action_realtime)."""
    total = data.get("total_ms", {})
    if not total:
        return
    print(f"\n  ── {label} ──")
    print(f"  {'TOTAL':32s}  {total['mean_ms']:8.2f} ms  "
          f"(median {total['median_ms']:.2f}, p95 {total['p95_ms']:.2f})")

    # ── Pipeline stages ──
    pipeline_stages = ["preprocess", "vlm_build_inputs", "vlm_forward",
                       "state_to_device", "am_init", "am_decode", "to_numpy"]
    print(f"\n    Pipeline Stages:")
    for stage in pipeline_stages:
        if stage in data:
            s = data[stage]
            pct = s["mean_ms"] / total["mean_ms"] * 100 if total.get("mean_ms", 0) > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {stage:30s}  {s['mean_ms']:8.2f} ms  ({pct:5.1f}%)  {bar}")

    # ── Per inference step ──
    step_keys = sorted(
        [k for k in data if k.startswith("am_step_") and k.count("_") == 2],
        key=lambda k: int(k.rsplit("_", 1)[-1]),
    )
    if step_keys:
        step_times = [data[k]["mean_ms"] for k in step_keys]
        total_step_ms = sum(step_times)
        pct_of_total = total_step_ms / total["mean_ms"] * 100 if total.get("mean_ms", 0) > 0 else 0

        print(f"\n    Action Model Per-Step ({len(step_keys)} steps):")
        max_step_ms = max(step_times) if step_times else 1.0
        for k in step_keys:
            s = data[k]
            bar = "#" * int(s["mean_ms"] / max_step_ms * 25)
            print(f"    {k:30s}  {s['mean_ms']:8.2f} ms  (std {s['std_ms']:.2f})  {bar}")

        print(f"    {'--- all steps total ---':30s}  {total_step_ms:8.2f} ms  ({pct_of_total:.1f}% of total)")
        print(f"    {'--- per step avg ---':30s}  {total_step_ms / len(step_keys):8.2f} ms")

    # ── Sub-stage breakdown (DD) ──
    substage_suffixes = ["forward_logits", "sample", "mask_schedule", "remask"]
    has_substages = any(f"am_step_0_{sfx}" in data for sfx in substage_suffixes)
    if has_substages and step_keys:
        print(f"\n    Per-Step Sub-Stage Breakdown (averaged across steps):")
        n_steps = len(step_keys)
        for sfx in substage_suffixes:
            sub_keys = [f"am_step_{i}_{sfx}" for i in range(n_steps) if f"am_step_{i}_{sfx}" in data]
            if not sub_keys:
                continue
            means = [data[k]["mean_ms"] for k in sub_keys]
            avg_ms = np.mean(means)
            total_ms = np.sum(means)
            pct = total_ms / total_step_ms * 100 if total_step_ms > 0 else 0
            bar = "#" * int(pct / 2)
            print(f"    {sfx:30s}  avg {avg_ms:7.2f} ms/step  "
                  f"(total {total_ms:7.2f} ms, {pct:5.1f}% of steps)  {bar}")

    # ── Frequency ──
    if total["mean_ms"] > 0:
        hz = 1000.0 / total["mean_ms"]
        hz_p95 = 1000.0 / total["p95_ms"] if total.get("p95_ms", 0) > 0 else 0
        print(f"\n    Frequency: {hz:.1f} Hz (mean)  |  {hz_p95:.1f} Hz (p95)")


def print_results(results: dict):
    fw = results["framework"]
    n_full = results["num_inference_steps"]
    n_rtc = results["num_inference_steps_rtc"]
    delay = results["inference_delay"]
    horizon = results["action_horizon"]

    print(f"\n{'=' * 72}")
    print(f"  Inference Benchmark — {fw}")
    print(f"{'=' * 72}")
    print(f"  Samples: {results['num_samples']}  |  Warmup: {results['warmup']}")
    print(f"  Action horizon: {horizon}  |  Inference steps (full): {n_full}")
    if delay > 0:
        print(f"  Inference delay: {delay}  |  RTC steps: {n_rtc}  "
              f"(= {n_full} * {delay} / {horizon})")

    if "predict_action" in results:
        _print_mode("predict_action (full generation)", results["predict_action"])

    if "predict_action_realtime" in results:
        _print_mode(f"predict_action_realtime (delay={delay})", results["predict_action_realtime"])

    # ── Side-by-side comparison ──
    if "predict_action" in results and "predict_action_realtime" in results:
        t_full = results["predict_action"]["total_ms"]["mean_ms"]
        t_rtc = results["predict_action_realtime"]["total_ms"]["mean_ms"]
        speedup = t_full / t_rtc if t_rtc > 0 else 0
        print(f"\n  ── RTC Speedup ──")
        print(f"  Full: {t_full:.2f} ms  |  RTC(delay={delay}): {t_rtc:.2f} ms  |  "
              f"Speedup: {speedup:.2f}x")

    print(f"{'=' * 72}\n")


def main():
    parser = argparse.ArgumentParser(description="Inference benchmarking for QwenPI / QwenDiscreteDiffusion")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data_root_dir", type=str, default=DATA_ROOT_DIR)
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to benchmark (excluding warmup)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations (excluded from stats)")
    parser.add_argument("--include_state", action="store_true", default=False)
    parser.add_argument("--decode_temperature", type=float, default=DECODE_TEMPERATURE)
    parser.add_argument("--choice_temperature", type=float, default=CHOICE_TEMPERATURE)
    parser.add_argument("--use_simple_max", action="store_true", default=False)
    parser.add_argument("--num_inference_steps", type=int, default=8,
                        help="Override number of inference steps (default: 8)")
    parser.add_argument("--inference_delay", type=int, default=0,
                        help="RTC prefix length. 0=predict_action only, N=realtime with N prefix steps")
    parser.add_argument("--rtc_mode", type=str, default="pigdm",
                        choices=["pigdm", "simulated_delay"],
                        help="RTC mode for PI: pigdm (default) or simulated_delay")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    model, framework_type = load_model(args.checkpoint)

    # Override inference steps
    am = model.action_model
    if hasattr(am, "num_inference_timesteps"):
        old = am.num_inference_timesteps
        am.num_inference_timesteps = args.num_inference_steps
        print(f"  [override] num_inference_timesteps: {old} -> {args.num_inference_steps}")
    if hasattr(am, "num_inference_steps"):
        old = am.num_inference_steps
        am.num_inference_steps = args.num_inference_steps
        print(f"  [override] num_inference_steps: {old} -> {args.num_inference_steps}")

    dataloader = load_dataset(model, include_state=args.include_state,
                              data_root_dir=args.data_root_dir)

    infer_kwargs = dict(
        decode_temperature=args.decode_temperature,
        choice_temperature=args.choice_temperature,
        use_simple_max=args.use_simple_max,
        rtc_mode=args.rtc_mode,
    )

    results = benchmark(
        model, framework_type, dataloader,
        num_samples=args.num_samples,
        warmup=args.warmup,
        inference_delay=args.inference_delay,
        infer_kwargs=infer_kwargs,
    )

    # Add metadata
    results["rtc_mode"] = args.rtc_mode
    results["checkpoint"] = args.checkpoint
    results["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    results["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024**2, 1)
    results["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved() / 1024**2, 1)

    print_results(results)

    # Save JSON
    if args.output:
        out_path = Path(args.output)
    else:
        run_dir = Path(args.checkpoint).parents[1]
        ckpt_name = Path(args.checkpoint).stem.replace("_pytorch_model", "")
        delay_tag = f"_delay{args.inference_delay}" if args.inference_delay > 0 else ""
        out_path = run_dir / f"benchmark_{ckpt_name}{delay_tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
