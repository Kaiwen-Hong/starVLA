"""
Discrete diffusion DiT - minimal architecture.

Input:  condition (B, condition_dim), token_ids (B, token_num)
Output: logits (B, token_num, token_vocab_size)
"""

import torch
import torch.nn as nn

from starVLA.model.modules.action_model.DiT_modules.models import DiTBlock


class DiscreteDiT(nn.Module):
    """
    Minimal discrete diffusion transformer.

    Args:
        token_num: Length of token sequence.
        output_dim: Output dimension per token (e.g. token_vocab_size for CE).
        condition_dim: Condition vector dimension.
        hidden_size: Transformer hidden size.
        depth: Number of transformer blocks.
        num_heads: Attention heads.
        mlp_ratio: MLP expansion ratio.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        token_num: int,
        output_dim: int,
        condition_dim: int,
        token_vocab_size: int | None = None,  # for embedding; defaults to output_dim
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_num = token_num
        self.output_dim = output_dim
        vocab = token_vocab_size if token_vocab_size is not None else output_dim

        self.token_embedding = nn.Embedding(vocab + 1, hidden_size)  # +1 for mask
        self.condition_proj = nn.Linear(condition_dim, hidden_size)
        self.positional_embedding = nn.Parameter(
            (hidden_size**-0.5) * torch.randn(token_num, hidden_size)
        )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_layer = nn.Linear(hidden_size, output_dim)
        self._init_weights()

    def _init_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)

    def forward(
        self,
        condition: torch.Tensor,
        token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            condition: (B, condition_dim)
            token_ids: (B, token_num) - indices; use vocab_size for mask.

        Returns:
            logits: (B, token_num, output_dim)
        """
        B = token_ids.shape[0]
        x = self.token_embedding(token_ids)  # (B, token_num, hidden_size)
        cond = self.condition_proj(condition)  # (B, hidden_size)
        x = x + cond.unsqueeze(1)
        x = x + self.positional_embedding
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.final_layer(x)


def DiscreteDiT_S(**kwargs):
    return DiscreteDiT(hidden_size=384, depth=6, num_heads=4, **kwargs)


def DiscreteDiT_B(**kwargs):
    return DiscreteDiT(hidden_size=768, depth=12, num_heads=12, **kwargs)


def DiscreteDiT_L(**kwargs):
    return DiscreteDiT(hidden_size=1024, depth=24, num_heads=16, **kwargs)


DiscreteDiT_models = {"DiT-S": DiscreteDiT_S, "DiT-B": DiscreteDiT_B, "DiT-L": DiscreteDiT_L}
