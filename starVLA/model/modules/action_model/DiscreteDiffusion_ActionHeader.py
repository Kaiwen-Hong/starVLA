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
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class DiscreteDiffusionActionHead(nn.Module):
    """
    MaskGIT-style discrete diffusion policy head (ref: ref_dd_mode.py).
    Conditions on VLM embeddings (vl_embs) and optional state.
    No timestep; uses masking schedule for training and gradual unmasking for decode.
    """

    def __init__(self, full_config):
        super().__init__()
        config = full_config.framework.action_model
        qwenvl_config = full_config.framework.qwenvl

        self.action_dim = config.action_dim
        self.action_horizon = config.future_action_window_size + 1
        self.num_inference_steps = config.get("num_inference_steps", 8)
        self.input_embedding_dim = qwenvl_config.vl_hidden_dim
        self.hidden_size = config.get("hidden_size", 1024)
        self.num_target_vision_tokens = config.get("num_target_vision_tokens", 32)

        # Action binning: 8 bits -> 2^8=256 bins, output (B,T,D,8,2) with logits for 0/1 per bit
        self.num_bins = config.get("num_bins", 256)
        self.num_bits = 8  # fixed: 2^8=256 bins; output shape (B,T,D,8,2)
        self.binning = ActionBinning(
            num_bins=self.num_bins,
            action_dim=self.action_dim,
            low=config.get("action_low", -1.0),
            high=config.get("action_high", 1.0),
        )
        self.register_buffer(
            "bit_powers",
            torch.tensor([2**i for i in range(self.num_bits)], dtype=torch.float32),
        )

        # MaskGIT: mask_token_id = num_bins (extra embedding for MASK)
        self.mask_token_id = self.num_bins

        # MaskGIT config (ref: ModelConfig)
        self.train_mask_schedule = config.get("train_mask_schedule", "cosine")
        self.no_mask_token_prob = config.get("no_mask_token_prob", 0.0)
        self.decode_schedule = config.get("decode_schedule", "cosine")
        self.use_remask = config.get("use_remask", False)
        self.l1_loss_weight = config.get("l1_loss_weight", 0.1)
        self.inpainting_mask = config.get("inpainting_mask", False)
        self.inpainting_d_schedule = config.get("inpainting_d_schedule", "linear_decay")

        # Token embedding: num_bins + 1 to include MASK token
        self.token_embedding = nn.Embedding(
            self.num_bins + 1, self.input_embedding_dim
        )
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        # DiT output: (B,T,D,8,2) -> 8*2=16 per action token
        diffusion_model_cfg = dict(config.get("diffusion_model_cfg", {}))
        diffusion_model_cfg["output_dim"] = self.num_bits * 2  # 16
        diffusion_model_cfg.setdefault("cross_attention_dim", self.input_embedding_dim)
        diffusion_model_cfg.setdefault("dropout", 0.1)
        diffusion_model_cfg.setdefault("num_layers", 12)
        diffusion_model_cfg.setdefault("attention_head_dim", 64)
        diffusion_model_cfg.setdefault(
            "num_attention_heads", self.input_embedding_dim // 64
        )
        diffusion_model_cfg.setdefault("norm_type", "ada_norm")
        diffusion_model_cfg.setdefault("interleave_self_attention", True)
        diffusion_model_cfg.setdefault("positional_embeddings", "sinusoidal")
        diffusion_model_cfg.setdefault("final_dropout", True)

        self.model = DiT(**diffusion_model_cfg)

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.get("state_dim", 0) > 0
            else None
        )

        self.future_tokens = nn.Embedding(
            self.num_target_vision_tokens, self.input_embedding_dim
        )
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        self.add_pos_embed = config.get("add_pos_embed", True)
        if self.add_pos_embed:
            max_seq_len = (
                self.num_target_vision_tokens
                + self.action_horizon * self.action_dim
                + (1 if self.state_encoder else 0)
            )
            self.position_embedding = nn.Embedding(
                config.get("max_seq_len", max_seq_len), self.input_embedding_dim
            )
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.config = config

    @property
    def seq_len(self):
        """Length of the action token sequence (T * action_dim)."""
        return self.action_horizon * self.action_dim

    def _build_sa_sequence(
        self,
        action_token_embeds: torch.Tensor,
        state_features: torch.Tensor | None,
        device: torch.device,
        batch_size: int,
    ) -> torch.Tensor:
        """Build state + future_tokens + action sequence for DiT."""
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )
        if state_features is not None:
            sa_embs = torch.cat(
                (state_features, future_tokens, action_token_embeds), dim=1
            )
        else:
            sa_embs = torch.cat((future_tokens, action_token_embeds), dim=1)

        if self.add_pos_embed:
            seq_len = sa_embs.shape[1]
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            sa_embs = sa_embs + pos_embs

        return sa_embs

    def _forward_logits(
        self,
        vl_embs: torch.Tensor,
        x_t_bins: torch.Tensor,
        state_features: torch.Tensor | None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """
        Predict logits for action bins given (vl_embs, partially unmasked action sequence).
        x_t_bins: (B, T, action_dim) with values in [0, num_bins-1], mask_token_id, or IGNORE_TOKEN.
        Returns (B, T, action_dim, num_bins) logits.
        """
        B = x_t_bins.shape[0]
        device = x_t_bins.device
        x_for_embed = torch.where(
            x_t_bins == IGNORE_TOKEN,
            torch.full_like(x_t_bins, self.mask_token_id),
            x_t_bins,
        )
        x_flat = x_for_embed.reshape(B, -1)
        action_embeds = self.token_embedding(x_flat)
        action_embeds = action_embeds.reshape(
            B, self.action_horizon, self.action_dim, -1
        ).reshape(B, self.seq_len, -1)

        sa_embs = self._build_sa_sequence(
            action_embeds, state_features, device, B
        )

        timestep = torch.zeros(B, dtype=torch.long, device=device)
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
        )
        raw_out = model_output[:, -self.seq_len :, :]
        # Output (B, T, D, 8, 2): 8 bits, 2 logits (0/1) per bit; softmax -> P(0), P(1)
        action_logits = raw_out.reshape(
            B, self.action_horizon, self.action_dim, self.num_bits, 2
        )
        return action_logits

    def apply_mask(
        self,
        input_tokens: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Apply ref-style masking to input_tokens. Selected positions become IGNORE_TOKEN.
        input_tokens: (B, T, action_dim) target bin indices.
        Returns masked_tokens: same shape; masked positions set to IGNORE_TOKEN.
        """
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

        masked_tokens = torch.where(masked_mask, IGNORE_TOKEN, input_tokens)
        return masked_tokens

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor | None = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        """
        Training: CE on masked positions only (ref: loss).
        Optional L1 auxiliary loss on continuous predictions.
        """
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
            vl_embs, x_for_embed, state_features, encoder_attention_mask
        )
        # logits: (B, T, D, 8, 2); per-bit CE, no heavy bin conversion
        bit_idx = torch.arange(
            self.num_bits, device=device, dtype=target_bins.dtype
        ).view(1, 1, 1, -1)
        target_bits = ((target_bins.unsqueeze(-1) >> bit_idx) & 1).long()
        ce_per_bit = F.cross_entropy(
            logits.reshape(-1, 2),
            target_bits.reshape(-1),
            reduction="none",
        ).reshape(B, T, D, self.num_bits)
        ce_per_token = ce_per_bit.mean(dim=-1)
        ce_masked = torch.where(loss_mask, ce_per_token, torch.zeros_like(ce_per_token))
        num_masked_per_sample = loss_mask.float().sum(dim=(1, 2)) + 1e-8
        ce_sum_per_sample = ce_masked.sum(dim=(1, 2))
        ce_loss = (ce_sum_per_sample / num_masked_per_sample).mean()

        l1_loss = torch.tensor(0.0, device=device)
        if self.l1_loss_weight > 0:
            probs_1 = F.softmax(logits, dim=-1)[..., 1]
            pred_bins = (probs_1 * self.bit_powers.view(1, 1, 1, -1)).sum(dim=-1)
            pred_bins = pred_bins.long().clamp(0, self.num_bins - 1)
            bin_centers = self.binning.bin_centers.to(device)
            pred_continuous = bin_centers[pred_bins.reshape(-1)].reshape(
                B, self.action_horizon, self.action_dim
            )
            l1_per_pos = (pred_continuous - actions).abs()
            l1_loss = l1_per_pos.mean()

        loss_total = ce_loss + self.l1_loss_weight * l1_loss
        return loss_total

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor | None = None,
        choice_temperature: float = 0.1,
        decode_temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        MaskGIT-style decode: start all masked, gradual unmasking via schedule.
        choice_temperature=0 -> deterministic (argmax / lowest-conf unmasking).
        decode_temperature=0 -> deterministic sampling.
        """
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
            x_for_embed = cur_seqs
            bit_logits = self._forward_logits(
                vl_embs, x_for_embed, state_features, None
            )
            safe_temp = max(decode_temperature, 1e-8)
            probs_1 = F.softmax(bit_logits / safe_temp, dim=-1)[..., 1]

            if deterministic:
                sampled_bits = (probs_1 > 0.5).long()
            else:
                sampled_bits = (torch.rand_like(probs_1, device=device) < probs_1).long()
            sampled = (
                (sampled_bits * self.bit_powers.view(1, 1, 1, -1)).sum(dim=-1).long()
            )
            probs_0 = 1.0 - probs_1
            selected_probs = (
                probs_1 * sampled_bits + probs_0 * (1 - sampled_bits)
            ).clamp(min=1e-8).prod(dim=-1)

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
            mask_len = torch.where(
                is_last_step,
                torch.zeros_like(mask_len),
                mask_len,
            )

            if self.use_remask:
                p_remask = 1.0 - ratio
                selected_probs = torch.where(
                    unknown_map,
                    selected_probs,
                    selected_probs * p_remask,
                )
            else:
                selected_probs = torch.where(
                    unknown_map, selected_probs, torch.full_like(selected_probs, float("inf"))
                )

            selected_flat = selected_probs.reshape(B, L)
            mask_len_flat = mask_len
            if deterministic_choice:
                action_mask_flat = mask_by_deterministic_lowest(
                    selected_flat, mask_len_flat
                )
            else:
                temp = choice_temperature * (1.0 - ratio)
                action_mask_flat = mask_by_random_topk(
                    selected_flat, mask_len_flat, temperature=temp
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
    ) -> torch.Tensor:
        """
        Real-time chunking: fix first inference_delay steps from prev_action_chunk,
        gradually unmask the rest. Uses base _forward_logits for inpainting (no extra logic).
        prev_action_chunk: (B, chunk_len, action_dim) normalized [-1,1].
        inference_delay: number of prefix steps to fix (0 = full replan, 1 = typical RTC).
        """
        B = vl_embs.shape[0]
        device = vl_embs.device
        L = self.seq_len
        deterministic = decode_temperature == 0
        deterministic_choice = choice_temperature == 0

        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(vl_embs, state, choice_temperature, decode_temperature)

        inference_delay = min(inference_delay, self.action_horizon)
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

        for step_idx in range(self.num_inference_steps):
            x_for_embed = cur_seqs
            bit_logits = self._forward_logits(
                vl_embs, x_for_embed, state_features, None
            )
            safe_temp = max(decode_temperature, 1e-8)
            probs_1 = F.softmax(bit_logits / safe_temp, dim=-1)[..., 1]

            if deterministic:
                sampled_bits = (probs_1 > 0.5).long()
            else:
                sampled_bits = (torch.rand_like(probs_1, device=device) < probs_1).long()
            sampled = (
                (sampled_bits * self.bit_powers.view(1, 1, 1, -1)).sum(dim=-1).long()
            )
            probs_0 = 1.0 - probs_1
            selected_probs = (
                probs_1 * sampled_bits + probs_0 * (1 - sampled_bits)
            ).clamp(min=1e-8).prod(dim=-1)

            unknown_map = cur_seqs == self.mask_token_id
            sampled = torch.where(prefix_mask, prefix_bins, sampled)
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
            mask_len = torch.where(
                is_last_step,
                torch.zeros_like(mask_len),
                mask_len,
            )

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


def get_action_model(config=None):
    """Factory: build DiscreteDiffusionActionHead from config."""
    return DiscreteDiffusionActionHead(full_config=config)
