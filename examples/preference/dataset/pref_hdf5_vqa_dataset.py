"""
VQA-cotrain variant of PrefHDF5Dataset.

Layered on top of baseline:
  - Action stream is BYTE-IDENTICAL to baseline (super().__getitem__).
  - Each sample additionally carries `vqa_episode_id = (split, row_id)` —
    a tiny pointer into the module-level VQA clip cache that
    framework.forward dedups-and-subsamples-2 over. No clip bytes flow
    through the DataLoader IPC queue.
  - Per-episode pref_key and task_group are cached for the VQA answer
    label and per-task accuracy logging.

Cache lives in `vqa_sample._VQA_CLIP_CACHE` (module global). Built ONCE
per (data_root, split) at dataset __init__ time in the parent process,
then fork-COW-shared with DataLoader workers.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .pref_hdf5_dataset import PrefHDF5Dataset
from .vqa_sample import (
    VQA_CAMERA,
    VQA_NUM_FRAMES,
    VQA_CATEGORIES,
    build_vqa_clip_cache,
    register_vqa_clip_cache,
)


class PrefHDF5VQADataset(PrefHDF5Dataset):
    """Tags each sample with its hdf5 path so the framework's _vqa_forward can
    decode the VQA clip + EE-state ON DEMAND (with train-time frame jitter), using
    the per-cat config in VQA_CATEGORIES[category]. No static clip cache (redesign
    2026-05-28): only n_vqa_per_batch episodes are used per step, so on-demand
    decode is cheap and lets us jitter frames + extract state at the same indices.

    `vqa_*` kwargs are accepted for backward-compat but ignored — the VQA clip
    config (cameras/strategy/n_frames/state) now lives in VQA_CATEGORIES.
    """

    def __init__(self, *args, vqa_num_frames=None, vqa_camera=None,
                 vqa_cameras=None, vqa_strategy=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.category not in VQA_CATEGORIES:
            raise KeyError(
                f"VQA dataset for category={self.category!r} requires an entry in VQA_CATEGORIES")

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        task_dir, tg, pk, ep_id, _t = self._index[idx]
        sample["vqa_h5_path"] = str(self.data_root / task_dir / "data" / f"episode{ep_id}.hdf5")
        sample["vqa_pref_key"] = pk
        sample["vqa_task_group"] = tg
        sample["robot_tag"] = "pref_hdf5_vqa"
        return sample


def get_pref_vqa_dataset(data_cfg, mode: str = "train", **kwargs) -> PrefHDF5VQADataset:
    """Factory mirroring get_pref_dataset signature.

    VQA-side cache config (cameras, strategy, n_frames) defaults from
    VQA_CATEGORIES[pref_category]. YAML can OPTIONALLY override via
    `vqa_cameras: [...]`, `vqa_strategy: <name>`, `vqa_num_frames: <int>`.
    If a key is absent from YAML, the per-cat default applies (don't pass
    None overrides → dataset reads VQA_CATEGORIES).
    """
    def _g(name, default=None):
        return data_cfg.get(name, default) if hasattr(data_cfg, "get") else default

    chunk_size = int(_g("future_action_window_size", 15)) + 1

    # Optional YAML overrides: only pass through if present (else dataset
    # falls back to VQA_CATEGORIES[category]).
    extra: dict = {}
    yaml_cams = _g("vqa_cameras", None)
    yaml_strategy = _g("vqa_strategy", None)
    yaml_num_frames = _g("vqa_num_frames", None)
    yaml_camera = _g("vqa_camera", None)  # legacy single-cam
    if yaml_cams is not None: extra["vqa_cameras"] = tuple(yaml_cams)
    if yaml_strategy is not None: extra["vqa_strategy"] = str(yaml_strategy)
    if yaml_num_frames is not None: extra["vqa_num_frames"] = int(yaml_num_frames)
    if yaml_camera is not None: extra["vqa_camera"] = str(yaml_camera)

    return PrefHDF5VQADataset(
        data_root_dir=data_cfg.data_root_dir,
        split=mode,
        chunk_size=chunk_size,
        past_window=int(_g("past_action_window_size", 0)),
        include_state=bool(_g("include_state", False)),
        cameras=tuple(_g("cameras", ("head_camera", "left_camera", "right_camera"))),
        image_size=tuple(_g("image_size", (224, 224))),
        stats_json_path=_g("stats_json_path", None),
        category=str(_g("pref_category", "giveobj")),
        data_cfg=data_cfg,
        action_space=str(_g("action_space", "ee")),
        **extra,
        **kwargs,
    )
