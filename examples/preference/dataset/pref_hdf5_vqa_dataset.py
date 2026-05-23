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

from typing import Dict, List, Tuple

from .pref_hdf5_dataset import PrefHDF5Dataset
from .vqa_sample import (
    VQA_CAMERA,
    VQA_NUM_FRAMES,
    build_vqa_clip_cache,
    register_vqa_clip_cache,
)


class PrefHDF5VQADataset(PrefHDF5Dataset):
    """Adds VQA per-episode clip cache + episode-id tagging on top of baseline."""

    def __init__(
        self,
        *args,
        vqa_num_frames: int = VQA_NUM_FRAMES,
        vqa_camera: str = VQA_CAMERA,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Insertion-order unique episode keys from baseline's _index.
        # Row id in the clip cache = insertion index (deterministic given the
        # baseline's split RNG: same data_root + split_seed -> same ordering).
        ep_to_row: Dict[Tuple[str, int], int] = {}
        ep_pref_key: List[str] = []
        ep_task_group: List[str] = []
        for (task_dir, tg, pk, ep_id, _t) in self._index:
            key = (task_dir, ep_id)
            if key not in ep_to_row:
                ep_to_row[key] = len(ep_to_row)
                ep_pref_key.append(pk)
                ep_task_group.append(tg)
        self._episode_id_to_row: Dict[Tuple[str, int], int] = ep_to_row
        self._episode_keys: List[Tuple[str, int]] = list(ep_to_row.keys())
        self._episode_pref_key: List[str] = ep_pref_key
        self._episode_task_group: List[str] = ep_task_group

        print(
            f"[PrefHDF5VQADataset/{self.split}] building VQA clip cache "
            f"for {len(self._episode_keys)} episodes "
            f"(camera={vqa_camera}, frames={vqa_num_frames})..."
        )
        cache = build_vqa_clip_cache(
            self.data_root,
            self._episode_keys,
            num_frames=vqa_num_frames,
            camera=vqa_camera,
            image_size=tuple(self.image_size),
        )
        register_vqa_clip_cache(self.split, cache)

    @property
    def episode_pref_key(self) -> List[str]:
        return self._episode_pref_key

    @property
    def episode_task_group(self) -> List[str]:
        return self._episode_task_group

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        task_dir, tg, pk, ep_id, _t = self._index[idx]
        row = self._episode_id_to_row[(task_dir, ep_id)]
        # Tuple is small and pickles cheaply; framework decodes (split, row).
        sample["vqa_episode_id"] = (self.split, row)
        sample["vqa_pref_key"] = pk
        sample["vqa_task_group"] = tg
        sample["robot_tag"] = "pref_hdf5_vqa"
        return sample


def get_pref_vqa_dataset(data_cfg, mode: str = "train", **kwargs) -> PrefHDF5VQADataset:
    """Factory mirroring get_pref_dataset signature."""
    def _g(name, default):
        return data_cfg.get(name, default) if hasattr(data_cfg, "get") else default

    chunk_size = int(_g("future_action_window_size", 15)) + 1
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
        vqa_num_frames=int(_g("vqa_num_frames", VQA_NUM_FRAMES)),
        vqa_camera=str(_g("vqa_camera", VQA_CAMERA)),
        **kwargs,
    )
