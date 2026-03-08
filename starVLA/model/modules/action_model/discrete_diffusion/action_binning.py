"""
Uniform action binning for discrete diffusion.
Aligns with ref_dd_mode: floor-based encode, bin centers for decode.
Maps continuous actions in [low, high] to discrete bins in [0, num_bins-1].

Binary protocol: model predicts P(bit_i=1) per bit via softmax. We use expected bin
or bit-wise sampling for decode; no heavy 256-bin distribution reconstruction.
"""

import torch
import torch.nn as nn


def continuous_to_bins(actions: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map continuous actions in [low, high] to bin indices in [0, num_bins-1]. Ref: floor((x+1)/2 * num_bins)."""
    scale = (high - low) / 2.0
    mid = (high + low) / 2.0
    clamped = actions.clamp(low + 1e-6, high - 1e-6)
    x = (clamped - low) / (high - low)  # [0, 1)
    bins = (x * num_bins).floor().long().clamp(0, num_bins - 1)
    return bins


def bins_to_continuous(bin_indices: torch.Tensor, num_bins: int, low: float = -1.0, high: float = 1.0) -> torch.Tensor:
    """Map bin indices to continuous actions (bin centers). Ref: (i+0.5)/num_bins * 2 - 1 for [-1,1]."""
    scale = (high - low) / num_bins
    centers = (bin_indices.float() + 0.5) * scale + low
    return centers


class ActionBinning(nn.Module):
    """Discretize continuous actions via uniform binning. Wraps continuous_to_bins / bins_to_continuous."""

    def __init__(self, num_bins: int, action_dim: int, low: float = -1.0, high: float = 1.0):
        super().__init__()
        self.num_bins = num_bins
        self.action_dim = action_dim
        self.low = low
        self.high = high
        # Bin centers for decoding: (i+0.5)/num_bins maps [0,num_bins) -> (0,1], then scale to [low,high]
        scale = (high - low) / num_bins
        centers = (torch.arange(num_bins, dtype=torch.float32) + 0.5) * scale + low
        self.register_buffer("bin_centers", centers)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """Encode continuous actions to discrete bin indices. (B, T, action_dim) -> (B, T, action_dim) long."""
        return continuous_to_bins(actions, self.num_bins, self.low, self.high)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete bin indices to continuous actions (bin centers)."""
        flat = indices.reshape(-1)
        decoded = self.bin_centers[flat].to(indices.device)
        B = indices.shape[0]
        return decoded.reshape(B, -1, self.action_dim)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Decode model logits (softmax over bins) to continuous via weighted sum of bin centers."""
        B, seq_len, K = logits.shape
        probs = logits.softmax(dim=-1)
        bin_centers = self.bin_centers.to(logits.device)
        decoded_flat = (probs * bin_centers.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        T = seq_len // self.action_dim
        return decoded_flat.reshape(B, T, self.action_dim)
