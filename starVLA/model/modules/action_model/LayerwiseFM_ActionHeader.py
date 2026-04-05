# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field

import torch
import math

import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, output_dim=2048):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size=1024, num_embodiments=8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)




DiTConfig = {"num_layers": 36, "input_embedding_dim": 2048, "attention_head_dim": 64, "num_attention_heads": 32} # default for qwen2.5-vl


def get_prefix_weights(
    start: int, end: int, total: int, schedule: str, device: torch.device,
) -> torch.Tensor:
    """Compute prefix attention weights for ΠGDM guidance.

    With start=2, end=6, total=10:
      1  1  4/5 3/5 2/5 1/5 0  0  0  0
             ^              ^
           start           end
    `start` (inclusive) is where the chunk starts being allowed to change.
    `end` (exclusive) is where the chunk stops attending to the prefix.
    """
    start = min(start, end)
    positions = torch.arange(total, device=device, dtype=torch.float32)

    if schedule == "ones":
        w = torch.ones(total, device=device)
    elif schedule == "zeros":
        w = (positions < start).float()
    elif schedule in ("linear", "exp"):
        w = ((start - 1 - positions) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == "exp":
            w = w * torch.expm1(w) / (math.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")

    return torch.where(positions >= end, torch.zeros_like(w), w)


class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        global_config,
        **kwargs,
    ):
        super().__init__()
        action_config = global_config.framework.action_model
        diffusion_model_cfg = action_config.diffusion_model_cfg

        # 更新 DiTConfig 到 diffusion_model_cfg
        DiTConfig["num_layers"] = global_config.framework.qwenvl.num_vl_layers
        DiTConfig["input_embedding_dim"] = global_config.framework.qwenvl.vl_hidden_dim
        DiTConfig["num_attention_heads"] = DiTConfig["input_embedding_dim"] // DiTConfig["attention_head_dim"]
        diffusion_model_cfg.update(DiTConfig)
        # diffusion_model_cfg["interleave_self_attention"] = False
        diffusion_model_cfg.cross_attention_dim = DiTConfig["input_embedding_dim"] # should match vl embedding dim, but for some case we might want to change it for cross + self attention
        self.input_embedding_dim = global_config.framework.qwenvl.vl_hidden_dim
        self.model = DiT(**diffusion_model_cfg) # TODO better way is copy LLM from VLM
        self.dit_out_hidden_size = self.input_embedding_dim
        self.action_dim = action_config.action_dim
        self.action_horizon = action_config.future_action_window_size + 1
        self.num_inference_timesteps = action_config.num_inference_timesteps

        self.state_encoder = MLP(
            input_dim=action_config.state_dim,
            output_dim=self.input_embedding_dim,
        ) if action_config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=action_config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.input_embedding_dim,
            hidden_dim=1024,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(action_config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if action_config.add_pos_embed:
            self.position_embedding = nn.Embedding(action_config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(action_config.noise_beta_alpha, action_config.noise_beta_beta)
        self.num_timestep_buckets = action_config.num_timestep_buckets
        self.config = action_config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def forward(self, vl_embs_list: list, actions: torch.Tensor, state: torch.Tensor = None):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        num_layers = len(vl_embs_list)
        B, L, D = vl_embs_list[0].shape
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed state
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        
        # Encode timesteps
        temb = self.model.timestep_encoder(t_discretized)

        # Layerwise cross-attention with vl_embs
        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],  # Use layer-specific vl_embs
                temb=temb,
            )
        
        # TODO miss self att and _process_output, but work well
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs_list: list, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs_list[0].dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Encode timestep
            temb = self.model.timestep_encoder(timesteps_tensor)

            # Layerwise cross-attention with vl_embs_list
            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=vl_embs_list[layer_idx],
                    temb=temb,
                )
            # TODO miss self att and _process_output 
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    def _forward_velocity(self, vl_embs_list, actions, timesteps, state_features, device):
        """Single forward pass returning predicted velocity."""
        B = actions.shape[0]
        action_features = self.action_encoder(actions, timesteps)

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        temb = self.model.timestep_encoder(timesteps)

        model_output = sa_embs
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                encoder_hidden_states=vl_embs_list[layer_idx],
                temb=temb,
            )

        pred = self.action_decoder(model_output)
        return pred[:, -self.action_horizon:]

    def predict_action_realtime(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        prev_action_chunk: torch.Tensor = None,
        inference_delay: int = 1,
        mode: str = "pigdm",
        suffix_length: int | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
    ) -> torch.Tensor:
        """
        RTC-aware flow-matching inference.

        Prefix weight schedule over the action horizon H:
            positions  0..d-1        : weight = 1   (known prefix)
            positions  d..H-s-1      : weight decays (transition zone)
            positions  H-s..H-1      : weight = 0   (fully free suffix)
        where d = inference_delay, s = suffix_length.

        Modes:
            "pigdm": Pseudo-Inverse Guided Diffusion (default). Uses
                VJP-based correction to guide generation towards
                consistency with the known prefix. Works without
                special training. Costs ~2x per step (forward + VJP).
                Ref: model.py realtime_action (pinv_corrected_velocity).
            "simulated_delay": Naive replace-and-denoise with per-position
                timesteps. Requires model trained with simulated delay
                loss. Single forward per step.

        Args:
            vl_embs_list: layer-wise VL embeddings, list of (B, seq, D).
            state: optional state tensor (B, 1, state_dim).
            prev_action_chunk: (B, action_horizon, action_dim) normalised
                actions from the previous prediction.
            inference_delay: number of leading timesteps to fix as prefix (d).
            mode: "pigdm" or "simulated_delay".
            suffix_length: number of trailing timesteps with weight 0 (s).
                Defaults to inference_delay (so d positions known,
                d positions fully free at the end).
            prefix_attention_schedule: weight schedule for transition zone
                ("exp", "linear", "ones", "zeros"). Only used in pigdm.
            max_guidance_weight: maximum ΠGDM guidance weight.

        Returns:
            actions: (B, action_horizon, action_dim) predicted actions.
        """
        if prev_action_chunk is None or inference_delay <= 0:
            return self.predict_action(vl_embs_list, state)

        batch_size = vl_embs_list[0].shape[0]
        device = vl_embs_list[0].device
        dtype = vl_embs_list[0].dtype

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=dtype,
            device=device,
        )

        state_features = self.state_encoder(state) if state is not None else None

        if mode == "pigdm":
            # ── ΠGDM: pseudo-inverse guided diffusion ──
            if suffix_length is None:
                suffix_length = inference_delay
            prefix_attention_end = self.action_horizon - suffix_length

            weights = get_prefix_weights(
                inference_delay, prefix_attention_end,
                self.action_horizon, prefix_attention_schedule, device,
            )  # (action_horizon,)
            prev = prev_action_chunk.to(dtype)

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)
                t_bucket = int(t_cont * self.num_timestep_buckets)
                timesteps_tensor = torch.full(
                    (batch_size,), t_bucket, device=device, dtype=torch.long,
                )

                with torch.enable_grad():
                    x_t = actions.detach().requires_grad_(True)
                    v_t = self._forward_velocity(
                        vl_embs_list, x_t, timesteps_tensor,
                        state_features, device,
                    )
                    # Predicted clean sample from current state
                    x_1_hat = x_t + v_t * (1.0 - t_cont)
                    # Weighted error between prediction and known prefix
                    error = (prev - x_1_hat) * weights[None, :, None]
                    # VJP: J^T @ error, where J = d(x_1_hat)/d(x_t)
                    pinv_correction = torch.autograd.grad(
                        x_1_hat, x_t, grad_outputs=error,
                    )[0]

                # Guidance weight schedule from ΠGDM paper
                t = t_cont
                inv_r2 = (t ** 2 + (1 - t) ** 2) / ((1 - t) ** 2 + 1e-8)
                c = (1 - t) / (t + 1e-8)
                gw = min(c * inv_r2, max_guidance_weight)

                v_corrected = v_t.detach() + gw * pinv_correction.detach()
                actions = actions.detach() + dt * v_corrected

            return actions

        # ── simulated_delay mode ──
        with torch.no_grad():
            prefix_mask = torch.zeros(
                1, self.action_horizon, 1, device=device, dtype=torch.bool,
            )
            prefix_mask[:, :inference_delay, :] = True

            for t_step in range(num_steps):
                t_cont = t_step / float(num_steps)

                actions = torch.where(
                    prefix_mask, prev_action_chunk.to(dtype), actions,
                )

                t_bucket = int(t_cont * self.num_timestep_buckets)
                t_bucket_prefix = int(1.0 * self.num_timestep_buckets) - 1
                timesteps_tensor = torch.full(
                    (batch_size, self.action_horizon),
                    t_bucket, device=device, dtype=torch.long,
                )
                timesteps_tensor[:, :inference_delay] = t_bucket_prefix

                # Inline action encoder with per-position timesteps
                a_emb = self.action_encoder.layer1(actions)
                tau_emb = self.action_encoder.pos_encoding(
                    timesteps_tensor,
                ).to(dtype=a_emb.dtype)
                x_enc = torch.cat([a_emb, tau_emb], dim=-1)
                x_enc = swish(self.action_encoder.layer2(x_enc))
                action_features = self.action_encoder.layer3(x_enc)

                if self.config.add_pos_embed:
                    pos_ids = torch.arange(
                        action_features.shape[1], dtype=torch.long, device=device,
                    )
                    pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                    action_features = action_features + pos_embs

                future_tokens = self.future_tokens.weight.unsqueeze(0).expand(
                    batch_size, -1, -1,
                )
                sa_embs = (
                    torch.cat((state_features, future_tokens, action_features), dim=1)
                    if state_features is not None
                    else torch.cat((future_tokens, action_features), dim=1)
                )

                temb_tensor = torch.full(
                    (batch_size,), t_bucket, device=device, dtype=torch.long,
                )
                temb = self.model.timestep_encoder(temb_tensor)

                model_output = sa_embs
                for layer_idx, layer in enumerate(self.model.transformer_blocks):
                    model_output = layer(
                        hidden_states=model_output,
                        encoder_hidden_states=vl_embs_list[layer_idx],
                        temb=temb,
                    )

                pred = self.action_decoder(model_output)
                pred_velocity = pred[:, -self.action_horizon:]

                actions = actions + dt * pred_velocity
                actions = torch.where(
                    prefix_mask, prev_action_chunk.to(dtype), actions,
                )

            return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(
        global_config=config
    )



if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass