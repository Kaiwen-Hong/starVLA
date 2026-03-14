"""
FastUMI eval utilities.

Provides:
- load_fastumi_action_norm_stats(checkpoint_dir)
- unnormalize_fastumi_actions(normalized_actions, stats)
- normalize_fastumi_state(state, stats)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from starVLA.dataloader.gr00t_lerobot.data_config import FastUMIDataConfig


def _pick_single_tag(stats_dict: dict[str, Any]) -> str:
    if len(stats_dict) != 1:
        raise ValueError(
            f"Expected exactly 1 tag in dataset_statistics.json, got {list(stats_dict.keys())}"
        )
    return next(iter(stats_dict.keys()))


def load_fastumi_action_norm_stats(checkpoint_dir: str | Path) -> dict[str, Any]:
    """
    Load action normalization statistics from `dataset_statistics.json` inside a checkpoint dir.
    Ensures `norm_modes` exists (fallback to FastUMIDataConfig.ACTION_NORM_MODES if missing).
    """
    checkpoint_dir = Path(checkpoint_dir)
    stats_path = checkpoint_dir / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Missing dataset statistics: {stats_path}")

    with open(stats_path, "r", encoding="utf-8") as f:
        all_stats = json.load(f)

    tag = _pick_single_tag(all_stats)
    action_stats = all_stats[tag]["action"]

    if "norm_modes" not in action_stats:
        action_stats = dict(action_stats)
        action_stats["norm_modes"] = list(FastUMIDataConfig.ACTION_NORM_MODES)

    # For any legacy callers that still rely on a single gripper index.
    # FastUMI action is 10D and gripper is the last dim.
    action_stats = dict(action_stats)
    action_stats.setdefault("gripper_idx", 9)

    return action_stats


def unnormalize_fastumi_actions(
    normalized_actions: np.ndarray, action_stats: dict[str, Any]
) -> np.ndarray:
    """
    Inverse normalization for FastUMI 10D actions using per-dim `norm_modes`.
    """
    x = np.asarray(normalized_actions, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    modes = list(action_stats["norm_modes"])
    mins = np.array(action_stats.get("min", action_stats.get("q01")), dtype=np.float32)
    maxs = np.array(action_stats.get("max", action_stats.get("q99")), dtype=np.float32)
    means = np.array(action_stats.get("mean"), dtype=np.float32)
    stds = np.array(action_stats.get("std"), dtype=np.float32)

    out = x.copy()
    for d, mode in enumerate(modes):
        if mode == "min_max":
            xd = np.clip(x[:, d], -1, 1)
            out[:, d] = 0.5 * (xd + 1.0) * (maxs[d] - mins[d]) + mins[d]
        elif mode == "mean_std":
            out[:, d] = x[:, d] * stds[d] + means[d]
        elif mode == "binary":
            out[:, d] = (x[:, d] >= 0.5).astype(np.float32)
        else:
            out[:, d] = x[:, d]
    return out


def normalize_fastumi_state(state: np.ndarray, state_stats: dict[str, Any]) -> np.ndarray:
    """
    Normalize FastUMI state with the same per-dim logic if norm_modes exists;
    otherwise behaves like min_max on q01/q99.
    """
    x = np.asarray(state, dtype=np.float32)
    if x.ndim == 1:
        x = x[None, :]

    if "norm_modes" in state_stats:
        modes = list(state_stats["norm_modes"])
        mins = np.array(state_stats.get("min", state_stats.get("q01")), dtype=np.float32)
        maxs = np.array(state_stats.get("max", state_stats.get("q99")), dtype=np.float32)
        means = np.array(state_stats.get("mean"), dtype=np.float32)
        stds = np.array(state_stats.get("std"), dtype=np.float32)
        out = x.copy()
        for d, mode in enumerate(modes):
            if mode == "min_max":
                out[:, d] = 2.0 * (x[:, d] - mins[d]) / (maxs[d] - mins[d] + 1e-8) - 1.0
                out[:, d] = np.clip(out[:, d], -1, 1)
            elif mode == "mean_std":
                out[:, d] = (x[:, d] - means[d]) / (stds[d] + 1e-8)
            elif mode == "binary":
                out[:, d] = (x[:, d] >= 0.5).astype(np.float32)
            else:
                out[:, d] = x[:, d]
        return out

    q01 = np.array(state_stats["q01"], dtype=np.float32)
    q99 = np.array(state_stats["q99"], dtype=np.float32)
    out = 2.0 * (x - q01) / (q99 - q01 + 1e-8) - 1.0
    return np.clip(out, -1, 1)
