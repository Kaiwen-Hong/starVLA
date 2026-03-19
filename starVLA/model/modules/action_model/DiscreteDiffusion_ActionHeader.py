"""
Discrete Diffusion Policy Action Head (MaskGIT-style).
Aligns with ref_dd_mode.py: masking-based discrete diffusion, no timestep.
Model takes (vl_embs, partially unmasked action tokens). Positions with
IGNORE_TOKEN (-100) or mask_token_id use MASK embedding and are predicted.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from starVLA.model.modules.action_model.discrete_diffusion import (
    ActionBinning,
    IGNORE_TOKEN,
    decode_mask_schedule,
    mask_by_deterministic_lowest,
    mask_by_random_topk,
    train_mask_schedule,
)
from starVLA.model.modules.action_model.discrete_diffusion.models import (
    DiscreteDiT,
    DiscreteDiT_models,
)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class DiscreteDiffusionActionHead(nn.Module):
    """
    MaskGIT-style discrete diffusion with naive bin representation (for debugging).
    DiT outputs (B,T,D,num_bins) logits; CE over bins, stable L1 via soft expectation.
    """

    def __init__(self, full_config, action_norm_stats=None):
        super().__init__()
        config = full_config.framework.action_model
        qwenvl_config = full_config.framework.qwenvl

        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_steps = config.get("num_inference_steps", 8)
        # Align with DiTActionHeader: use action_hidden_dim when present
        self.input_embedding_dim = config.get(
            "action_hidden_dim",
            getattr(qwenvl_config, "vl_hidden_dim", 2048),
        )
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_target_vision_tokens = config.get("num_target_vision_tokens", 32)

        self.num_bins = config.get("num_bins", 256)
        action_low, action_high = self._bin_range_from_stats(
            config, action_norm_stats
        )
        self.binning = ActionBinning(
            num_bins=self.num_bins,
            action_dim=self.action_dim,
            low=action_low,
            high=action_high,
        )
        self.action_norm_stats = action_norm_stats
        self.mask_token_id = self.num_bins

        self.train_mask_schedule = config.get("train_mask_schedule", "cosine")
        self.no_mask_token_prob = config.get("no_mask_token_prob", 0.0)
        self.decode_schedule = config.get("decode_schedule", "cosine")
        self.use_remask = config.get("use_remask", False)
        self.l1_loss_weight = config.get("l1_loss_weight", 0.1)
        self.inpainting_mask = config.get("inpainting_mask", False)
        self.inpainting_d_schedule = config.get("inpainting_d_schedule", "linear_decay")

        # DiscreteDiT: condition (B, cond_dim) + token_ids (B, token_num) -> logits
        # Align with DiTActionHeader: use action_hidden_dim for condition dim
        diffusion_model_cfg = dict(config.get("diffusion_model_cfg", {}))
        cross_attention_dim = diffusion_model_cfg.get(
            "cross_attention_dim",
            config.get("action_hidden_dim", self.input_embedding_dim),
        )
        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.get("state_dim", 0) > 0
            else None
        )
        condition_dim = cross_attention_dim + (
            self.input_embedding_dim if self.state_encoder else 0
        )

        token_num = self.action_horizon * self.action_dim
        dit_kwargs = dict(
            token_num=token_num,
            output_dim=self.num_bins,
            condition_dim=condition_dim,
            token_vocab_size=self.num_bins,
            mlp_ratio=4.0,
            dropout=diffusion_model_cfg.get("dropout", 0.1),
        )
        model_type = config.get("action_model_type")
        if model_type and model_type in DiscreteDiT_models:
            self.model = DiscreteDiT_models[model_type](**dit_kwargs)
        else:
            self.model = DiscreteDiT(
                depth=diffusion_model_cfg.get("num_layers", 12),
                num_heads=diffusion_model_cfg.get("num_attention_heads", self.input_embedding_dim // 64),
                **dit_kwargs,
            )

        self.config = config

    @staticmethod
    def _bin_range_from_stats(config, action_norm_stats) -> tuple[float, float]:
        """
        Derive action_low, action_high for binning from config and optional
        action_norm_stats (from dataset_statistics.json).
        For min_max / q99 normalization, dataloader maps raw [min,max] -> [-1,1],
        so binning uses [-1, 1]. Falls back to config values if stats absent.
        """
        if action_norm_stats is not None and (
            "min" in action_norm_stats
            or "q01" in action_norm_stats
        ):
            return (-1.0, 1.0)
        return (
            config.get("action_low", -1.0),
            config.get("action_high", 1.0),
        )

    @property
    def seq_len(self):
        return self.action_horizon * self.action_dim

    def _forward_logits(
        self,
        vl_embs: torch.Tensor,
        x_t_bins: torch.Tensor,
        state_features: torch.Tensor | None,
    ) -> torch.Tensor:
        B = x_t_bins.shape[0]
        condition = vl_embs.mean(dim=1)
        if state_features is not None:
            condition = torch.cat(
                [condition, state_features.squeeze(1)], dim=-1
            )
        elif self.state_encoder is not None:
            device, dtype = condition.device, condition.dtype
            pad = torch.zeros(
                B, self.input_embedding_dim, device=device, dtype=dtype
            )
            condition = torch.cat([condition, pad], dim=-1)

        token_ids = torch.where(
            x_t_bins == IGNORE_TOKEN,
            torch.full_like(x_t_bins, self.mask_token_id),
            x_t_bins,
        )
        token_ids = token_ids.reshape(B, -1)

        logits = self.model(condition=condition, token_ids=token_ids)
        return logits.reshape(B, self.action_horizon, self.action_dim, self.num_bins)

    def apply_mask(self, input_tokens: torch.Tensor, generator=None) -> torch.Tensor:
        B, T, D = input_tokens.shape
        device = input_tokens.device
        L = T * D

        if self.inpainting_mask:
            max_d = self.action_horizon // 2
            if self.inpainting_d_schedule == "linear_decay":
                weights = torch.arange(max_d + 1, dtype=torch.float32, device=device)
                weights = (max_d + 1 - weights) / weights.sum()
            else:
                weights = torch.ones(max_d + 1, device=device) / (max_d + 1)
            d = torch.multinomial(weights.unsqueeze(0).expand(B, -1), 1, generator=generator).squeeze(1)
            loss_mask_full = (
                torch.arange(self.action_horizon, device=device)[None, :, None]
                >= d[:, None, None]
            ).expand(B, T, D)
        else:
            loss_mask_full = torch.ones(B, T, D, dtype=torch.bool, device=device)

        total_unknown = loss_mask_full.float().sum(dim=(1, 2))
        rand_time = torch.rand(B, device=device, generator=generator)
        mask_ratios = train_mask_schedule(rand_time, self.train_mask_schedule)
        num_mask = (total_unknown * mask_ratios).round().long().clamp(1, L)
        num_mask = torch.where(total_unknown > 0, num_mask, torch.zeros_like(num_mask))

        vals = torch.rand(B, T, D, device=device, generator=generator)
        vals = torch.where(loss_mask_full, vals, torch.full_like(vals, float("inf")))
        vals_flat = vals.reshape(B, L)
        perm = torch.argsort(vals_flat, dim=1)
        ranks = torch.argsort(perm, dim=1)
        masked_mask_flat = ranks < num_mask.unsqueeze(1)
        masked_mask = masked_mask_flat.reshape(B, T, D)

        if self.no_mask_token_prob > 0:
            prob = torch.rand(B, T, D, device=device, generator=generator)
            unmask = (prob < self.no_mask_token_prob) & masked_mask
            masked_mask = masked_mask & ~unmask

        return torch.where(masked_mask, IGNORE_TOKEN, input_tokens)

    def forward(
        self,
        gt_action: torch.Tensor,
        condition: torch.Tensor,
        state: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple:
        """
        Aligned with DiTActionHeader: forward(gt_action, condition) -> (pred, target, extra).
        """
        actions = gt_action
        vl_embs = condition
        B, T, D = actions.shape
        device = actions.device
        assert T == self.action_horizon and D == self.action_dim

        target_bins = self.binning.encode(actions)
        input_tokens = self.apply_mask(target_bins)

        loss_mask = input_tokens == IGNORE_TOKEN
        x_for_embed = torch.where(
            loss_mask,
            torch.full_like(input_tokens, self.mask_token_id),
            input_tokens,
        )

        if state is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_features = self.state_encoder(state).unsqueeze(1)
        else:
            state_features = None

        logits = self._forward_logits(
            vl_embs, x_for_embed, state_features
        )
        extra = {"loss_mask": loss_mask, "actions": actions}
        return (logits, target_bins, extra)

    def loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute CE + optional L1 loss. Aligned with DiTActionHeader.loss(pred, target).
        pred: (B,T,D,num_bins), target: (B,T,D) bin indices.
        """
        B = pred.shape[0]
        device = pred.device

        if loss_mask is None:
            loss_mask = torch.ones(B, *target.shape[1:], dtype=torch.bool, device=device)

        ce_per_token = F.cross_entropy(
            pred.reshape(-1, self.num_bins),
            target.reshape(-1),
            reduction="none",
        ).reshape(B, *target.shape[1:])
        ce_masked = torch.where(loss_mask, ce_per_token, torch.zeros_like(ce_per_token))
        num_masked_per_sample = loss_mask.float().sum(dim=(1, 2)) + 1e-8
        ce_sum_per_sample = ce_masked.sum(dim=(1, 2))
        ce_loss = (ce_sum_per_sample / num_masked_per_sample).mean()

        l1_loss = torch.tensor(0.0, device=device)
        if self.l1_loss_weight > 0 and actions is not None:
            bin_logits_flat = pred.reshape(B, self.seq_len, self.num_bins)
            pred_continuous = self.binning.decode_logits(bin_logits_flat)
            l1_loss = (pred_continuous - actions).abs().mean()

        return ce_loss + self.l1_loss_weight * l1_loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        choice_temperature: float = 0.1,
        decode_temperature: float = 1.0,
    ) -> torch.Tensor:
        B = vl_embs.shape[0]
        device = vl_embs.device
        L = self.seq_len
        deterministic = decode_temperature == 0
        deterministic_choice = choice_temperature == 0

        cur_seqs = torch.full(
            (B, self.action_horizon, self.action_dim),
            self.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        unknown_init = torch.full((B,), L, dtype=torch.long, device=device)

        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_features = self.state_encoder(state).unsqueeze(1)
        else:
            state_features = None

        for step_idx in range(self.num_inference_steps):
            bit_logits = self._forward_logits(vl_embs, cur_seqs, state_features)
            safe_temp = max(decode_temperature, 1e-8)
            probs = F.softmax(bit_logits / safe_temp, dim=-1)

            if deterministic:
                sampled = probs.argmax(dim=-1)
                selected_probs = probs.max(dim=-1).values
            else:
                probs_flat = probs.reshape(-1, self.num_bins)
                sampled_flat = torch.multinomial(probs_flat, 1).squeeze(-1)
                sampled = sampled_flat.reshape(B, self.action_horizon, self.action_dim)
                selected_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / self.num_inference_steps
            mask_ratio = decode_mask_schedule(
                torch.tensor(ratio, device=device), self.decode_schedule
            )
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            is_last_step = torch.tensor(
                step_idx == self.num_inference_steps - 1,
                device=device,
                dtype=torch.bool,
            )
            mask_len = torch.where(is_last_step, torch.zeros_like(mask_len), mask_len)

            if self.use_remask:
                p_remask = 1.0 - ratio
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask,
                )
            else:
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    torch.full_like(selected_probs, float("inf")),
                )

            selected_flat = selected_probs.reshape(B, L)
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(
                    selected_flat, mask_len
                )
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(
                    selected_flat, mask_len, temperature=temp
                )
            action_mask = action_mask_flat.reshape(B, self.action_horizon, self.action_dim)
            cur_seqs = torch.where(action_mask, self.mask_token_id, sampled)

        return self.binning.decode(cur_seqs)

    @torch.no_grad()
    def predict_action_realtime(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        prev_action_chunk: torch.Tensor | None = None,
        inference_delay: int = 1,
        choice_temperature: float = 0.1,
        decode_temperature: float = 1.0,
        fixed_steps: bool = False,
    ) -> torch.Tensor:
        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(vl_embs, state, choice_temperature, decode_temperature)

        B = vl_embs.shape[0]
        device = vl_embs.device
        L = self.seq_len
        deterministic = decode_temperature == 0
        deterministic_choice = choice_temperature == 0
        inference_delay = min(inference_delay, self.action_horizon)

        if fixed_steps:
            num_steps = self.num_inference_steps
        else:
            num_steps = max(1, int(self.num_inference_steps * inference_delay / self.action_horizon))

        prefix_bins = self.binning.encode(prev_action_chunk)
        prefix_mask = (
            torch.arange(self.action_horizon, device=device)[None, :, None]
            < inference_delay
        ).expand(B, self.action_horizon, self.action_dim)

        cur_seqs = torch.where(
            prefix_mask,
            prefix_bins,
            torch.full_like(prefix_bins, self.mask_token_id, device=device),
        )
        unknown_init = torch.full(
            (B,),
            (self.action_horizon - inference_delay) * self.action_dim,
            dtype=torch.long,
            device=device,
        )

        if state is not None and self.state_encoder is not None:
            if state.dim() == 3:
                state = state.squeeze(1)
            state_features = self.state_encoder(state).unsqueeze(1)
        else:
            state_features = None

        for step_idx in range(num_steps):
            bit_logits = self._forward_logits(vl_embs, cur_seqs, state_features)
            safe_temp = max(decode_temperature, 1e-8)
            probs = F.softmax(bit_logits / safe_temp, dim=-1)

            if deterministic:
                sampled = probs.argmax(dim=-1)
                selected_probs = probs.max(dim=-1).values
            else:
                probs_flat = probs.reshape(-1, self.num_bins)
                sampled_flat = torch.multinomial(probs_flat, 1).squeeze(-1)
                sampled = sampled_flat.reshape(B, self.action_horizon, self.action_dim)
                selected_probs = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(prefix_mask, prefix_bins, sampled)
            sampled = torch.where(unknown_map, sampled, cur_seqs)

            ratio = (step_idx + 1.0) / num_steps
            mask_ratio = decode_mask_schedule(
                torch.tensor(ratio, device=device), self.decode_schedule
            )
            mask_len = (unknown_init.float() * mask_ratio).long()
            min_len = torch.full_like(mask_len, 1)
            max_len = (unknown_init - 1).clamp(min=0)
            mask_len = mask_len.clamp(min=min_len, max=max_len)
            is_last_step = torch.tensor(
                step_idx == num_steps - 1,
                device=device,
                dtype=torch.bool,
            )
            mask_len = torch.where(is_last_step, torch.zeros_like(mask_len), mask_len)

            if self.use_remask:
                p_remask = 1.0 - ratio
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask,
                )
            else:
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
                action_mask_flat = mask_by_random_topk(
                    selected_flat, mask_len, temperature=temp
                )
            action_mask = action_mask_flat.reshape(
                B, self.action_horizon, self.action_dim
            )
            cur_seqs = torch.where(
                prefix_mask,
                prefix_bins,
                torch.where(action_mask, self.mask_token_id, sampled),
            )

        return self.binning.decode(cur_seqs)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


def get_action_model(config=None, action_norm_stats=None):
    """Factory: build DiscreteDiffusionActionHead (naive bin representation).
    If action_norm_stats is provided (from dataset_statistics.json), uses them
    to set action_low/action_high for binning. For min_max/q99 normalization,
    the dataloader maps raw [min,max] to [-1,1], so binning uses [-1, 1].
    """
    return DiscreteDiffusionActionHead(full_config=config, action_norm_stats=action_norm_stats)
