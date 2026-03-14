"""FastUMI model-to-robot interface for inference / open-loop evaluation.

Key differences from the default base_framework.unnormalize_actions():
  - Per-dim normalization modes: min_max (pos), mean_std (rot6d), binary (gripper)
  - Gripper is at index 9 (not 6)
  - No blanket np.clip(-1,1) on mean_std dims (they are unbounded z-scores)

The norm_modes list is loaded from dataset_statistics.json (if available)
or falls back to FastUMIDataConfig.ACTION_NORM_MODES.
"""
import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_fastumi_action_norm_stats(
    checkpoint_dir: str | Path,
    tag: str = "new_embodiment",
) -> Dict[str, np.ndarray]:
    """Load action norm stats from a checkpoint's dataset_statistics.json.

    Returns a dict with keys: mean, std, min, max, q01, q99, mask, norm_modes.
    """
    ckpt_path = Path(checkpoint_dir)
    stats_path = ckpt_path / "dataset_statistics.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"dataset_statistics.json not found in {ckpt_path}")

    with open(stats_path) as f:
        all_stats = json.load(f)

    action_stats = all_stats[tag]["action"]

    # Convert lists to numpy arrays for the numeric fields
    result = {}
    for key in ["mean", "std", "min", "max", "q01", "q99"]:
        if key in action_stats:
            result[key] = np.array(action_stats[key], dtype=np.float64)

    if "mask" in action_stats:
        result["mask"] = np.array(action_stats["mask"], dtype=bool)

    # norm_modes: use saved value if present, otherwise fall back to config
    if "norm_modes" in action_stats:
        result["norm_modes"] = action_stats["norm_modes"]
    else:
        from starVLA.dataloader.gr00t_lerobot.data_config import FastUMIDataConfig
        result["norm_modes"] = FastUMIDataConfig.ACTION_NORM_MODES

    return result


def unnormalize_fastumi_actions(
    normalized_actions: np.ndarray,
    action_norm_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Unnormalize FastUMI 10D actions using per-dim normalization modes.

    This is a convenience wrapper that delegates to the updated
    base_framework.unnormalize_actions (which supports norm_modes),
    but can also be used standalone.

    Args:
        normalized_actions: shape (T, 10) — model output.
        action_norm_stats: dict from load_fastumi_action_norm_stats().

    Returns:
        Raw actions, shape (T, 10).
    """
    norm_modes = action_norm_stats.get("norm_modes", None)
    if norm_modes is None:
        from starVLA.dataloader.gr00t_lerobot.data_config import FastUMIDataConfig
        norm_modes = FastUMIDataConfig.ACTION_NORM_MODES

    actions = normalized_actions.copy()
    D = actions.shape[-1]

    for d in range(D):
        mode = norm_modes[d]
        if mode == "min_max":
            lo = float(action_norm_stats["min"][d])
            hi = float(action_norm_stats["max"][d])
            actions[:, d] = np.clip(actions[:, d], -1, 1)
            actions[:, d] = 0.5 * (actions[:, d] + 1) * (hi - lo) + lo
        elif mode == "mean_std":
            mu = float(action_norm_stats["mean"][d])
            sd = float(action_norm_stats["std"][d])
            actions[:, d] = actions[:, d] * sd + mu
        elif mode == "binary":
            actions[:, d] = np.where(actions[:, d] < 0.5, 0, 1)
        elif mode == "none":
            pass  # Pass-through: no denormalization

    return actions


def normalize_fastumi_state(
    state: np.ndarray,
    state_norm_stats: Dict[str, np.ndarray],
) -> np.ndarray:
    """Normalize a raw 10D FastUMI state for model input.

    State uses min_max for pos and rot6d, binary for gripper.
    """
    lo = np.array(state_norm_stats["min"], dtype=np.float64)
    hi = np.array(state_norm_stats["max"], dtype=np.float64)

    normalized = state.copy().astype(np.float64)

    # Dims 0:9 — min_max
    rng = hi[:9] - lo[:9]
    rng[rng == 0] = 1.0  # avoid division by zero
    normalized[..., :9] = 2.0 * (normalized[..., :9] - lo[:9]) / rng - 1.0

    # Dim 9 — binary (gripper)
    normalized[..., 9] = (normalized[..., 9] > 0.5).astype(np.float64)

    return normalized
