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
    """Adds VQA per-episode clip cache + episode-id tagging on top of baseline.

    The VQA cache config (cameras, clip_strategy, n_frames) is per-category and
    read from `VQA_CATEGORIES[self.category]`. The legacy single-cam uniform_8
    setup remains the default for cats that don't override (giveobj/contact/
    height/hvlv/orient), so previously-trained ckpts stay reproducible.
    """

    def __init__(
        self,
        *args,
        vqa_num_frames: Optional[int] = None,
        vqa_camera: Optional[str] = None,
        vqa_cameras: Optional[Tuple[str, ...]] = None,
        vqa_strategy: Optional[str] = None,
        **kwargs,
    ):
        """
        Per-cat defaults come from VQA_CATEGORIES[category]. Optional kwargs
        let callers override (mostly for tests / ablations); we use them ONLY
        if explicitly passed.
        """
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

        # Per-cat VQA config — explicit kwargs override the registry defaults.
        cat_cfg = VQA_CATEGORIES.get(self.category)
        if cat_cfg is None:
            raise KeyError(
                f"VQA dataset for category={self.category!r} requires an entry "
                f"in VQA_CATEGORIES; got None"
            )
        eff_n_frames = vqa_num_frames if vqa_num_frames is not None else cat_cfg.n_frames
        eff_strategy = vqa_strategy if vqa_strategy is not None else cat_cfg.clip_strategy
        if vqa_cameras is not None:
            eff_cameras = tuple(vqa_cameras)
        elif vqa_camera is not None:
            # legacy single-cam override
            eff_cameras = (vqa_camera,)
        else:
            eff_cameras = tuple(cat_cfg.cameras)

        print(
            f"[PrefHDF5VQADataset/{self.split}] building VQA clip cache "
            f"for {len(self._episode_keys)} episodes "
            f"(category={self.category}, cameras={eff_cameras}, "
            f"strategy={eff_strategy}, n_frames={eff_n_frames})..."
        )
        cache = build_vqa_clip_cache(
            self.data_root,
            self._episode_keys,
            num_frames=eff_n_frames,
            cameras=eff_cameras,
            strategy=eff_strategy,
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
        **extra,
        **kwargs,
    )
