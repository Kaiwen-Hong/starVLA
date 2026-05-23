"""
HDF5 dataset for the preference-conditioned VLA Stage A baseline.

Returns sample dicts that match starVLA's QwenPI framework input contract
(see starVLA/model/framework/QwenPI.py:96-99): image / lang / action /
optional state.

Bypasses LeRobotMixtureDataset because our data is bimanual sim HDF5
(RoboTwin format), not the LeRobot parquet+video layout.

Supports 4 categories (giveobj/height/hvlv/orient) via the `category` kwarg
which keys into `prompt.PREF_CATEGORIES`. Default "giveobj" preserves the
original signature; new categories pass `category="height|hvlv|orient"`.
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Any, Optional, Tuple

import h5py
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .prompt import PREF_CATEGORIES, build_action_prompt
from .rotation import quat_xyzw_to_6d

DEFAULT_CAMERAS = ("head_camera", "left_camera", "right_camera")
DEFAULT_IMAGE_SIZE = (224, 224)
DEFAULT_CHUNK_SIZE = 16
DEFAULT_CATEGORY = "giveobj"


def _task_dirs_for_groups(
    data_root: Path,
    groups: Tuple[str, ...],
    pref_keys: Tuple[str, ...],
) -> list[tuple[str, str, str]]:
    """Enumerate existing (task_dir, task_group, pref_key) triples on disk."""
    out = []
    for tg in groups:
        for pk in pref_keys:
            name = f"{tg}_{pk}"
            if (data_root / name).is_dir():
                out.append((name, tg, pk))
    return out


def _split_episodes(
    data_root: Path,
    task_groups: Tuple[str, ...],
    pref_keys: Tuple[str, ...],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for task_dir, _, _ in _task_dirs_for_groups(data_root, task_groups, pref_keys):
        hdf5_dir = data_root / task_dir / "data"
        ep_ids = sorted(
            int(p.stem.replace("episode", ""))
            for p in hdf5_dir.glob("episode*.hdf5")
        )
        rng = random.Random(f"{seed}-{task_dir}")
        rng.shuffle(ep_ids)
        n_val = int(round(len(ep_ids) * val_fraction))
        out[task_dir] = {
            "val": sorted(ep_ids[:n_val]),
            "train": sorted(ep_ids[n_val:]),
        }
    return out


class PrefHDF5Dataset(Dataset):
    """
    Each item:
        {
          "image":  [PIL.Image x N_cameras],
          "lang":   "<base prompt> Preference: <pref label>",
          "language": same as lang,
          "action": np.float16 (chunk_size, 20)
          "state":  np.float16 (1, 20)    # if include_state
          "robot_tag": "pref_hdf5",
        }

    Action layout (20D):
        [Lxyz(3), L6Drot(6), Lgrip(1), Rxyz(3), R6Drot(6), Rgrip(1)]
    Source rotation in HDF5 is xyzw quaternion (SAPIEN/ManiSkill default).
    """

    def __init__(
        self,
        data_root_dir: str | Path,
        split: str = "train",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        past_window: int = 0,
        include_state: bool = True,
        cameras: tuple[str, ...] = DEFAULT_CAMERAS,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        stats_json_path: Optional[str | Path] = None,
        category: str = DEFAULT_CATEGORY,
        task_groups: Optional[Tuple[str, ...]] = None,
        pref_keys: Optional[Tuple[str, ...]] = None,
        split_val_fraction: float = 0.2,
        split_seed: int = 42,
        paraphrase_seed: int = 42,
        data_cfg: Optional[Any] = None,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if past_window != 0:
            raise NotImplementedError(
                f"past_window > 0 not supported in v1; got {past_window}"
            )

        if category not in PREF_CATEGORIES:
            raise KeyError(
                f"Unknown category {category!r}; expected one of {tuple(PREF_CATEGORIES)}"
            )
        cat = PREF_CATEGORIES[category]
        # task_groups / pref_keys default from registry; explicit args override.
        eff_task_groups = tuple(task_groups) if task_groups is not None else cat.task_groups
        eff_pref_keys  = tuple(pref_keys)   if pref_keys   is not None else cat.pref_keys

        self.data_root = Path(data_root_dir)
        self.split = split
        self.chunk_size = int(chunk_size)
        self.future_window = self.chunk_size - 1
        self.past_window = int(past_window)
        self.include_state = bool(include_state)
        self.cameras = tuple(cameras)
        self.image_size = tuple(image_size)
        self.data_cfg = data_cfg
        self.category = category
        self._task_groups = eff_task_groups
        self._pref_keys = eff_pref_keys

        self._task_dirs = _task_dirs_for_groups(
            self.data_root, eff_task_groups, eff_pref_keys
        )
        if not self._task_dirs:
            raise FileNotFoundError(
                f"No task dirs found under {self.data_root!s} for category={category!r}, "
                f"groups={eff_task_groups}, pref_keys={eff_pref_keys}"
            )

        ep_split = _split_episodes(
            self.data_root,
            eff_task_groups,
            eff_pref_keys,
            val_fraction=split_val_fraction,
            seed=split_seed,
        )

        self._index: list[tuple[str, str, str, int, int]] = []
        for task_dir, tg, pk in self._task_dirs:
            for ep_id in ep_split[task_dir][split]:
                h5_path = self.data_root / task_dir / "data" / f"episode{ep_id}.hdf5"
                with h5py.File(h5_path, "r") as f:
                    T = f["endpose/left_endpose"].shape[0]
                last = T - self.chunk_size
                if last < 0:
                    continue
                for t in range(last + 1):
                    self._index.append((task_dir, tg, pk, ep_id, t))

        self._inst_cache: dict[tuple[str, int], list[str]] = {}
        self._paraphrase_rng = random.Random(paraphrase_seed)

        self.action_q01: Optional[np.ndarray] = None
        self.action_q99: Optional[np.ndarray] = None
        self.state_q01: Optional[np.ndarray] = None
        self.state_q99: Optional[np.ndarray] = None
        self.stats_json_path = Path(stats_json_path) if stats_json_path else None
        if self.stats_json_path and self.stats_json_path.exists():
            self._load_stats(self.stats_json_path)

    # -- public helpers --

    def _load_stats(self, path: Path):
        s = json.loads(Path(path).read_text())
        a = s["action"]
        self.action_q01 = np.asarray(a["q01"], dtype=np.float32)
        self.action_q99 = np.asarray(a["q99"], dtype=np.float32)
        st = s.get("state")
        if st:
            self.state_q01 = np.asarray(st["q01"], dtype=np.float32)
            self.state_q99 = np.asarray(st["q99"], dtype=np.float32)

    def save_dataset_statistics(self, out_path: str | Path):
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        body: dict = {}
        if self.action_q01 is not None:
            body["action"] = {
                "q01": self.action_q01.tolist(),
                "q99": self.action_q99.tolist(),
            }
        if self.state_q01 is not None:
            body["state"] = {
                "q01": self.state_q01.tolist(),
                "q99": self.state_q99.tolist(),
            }
        with open(out, "w") as f:
            json.dump(body, f, indent=2)

    def __len__(self) -> int:
        return len(self._index)

    # -- core ---

    @staticmethod
    def _normalize(x: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32, copy=False)
        span = q99 - q01
        safe = span != 0
        out = np.zeros_like(x)
        if safe.any():
            out[..., safe] = 2 * (x[..., safe] - q01[safe]) / span[safe] - 1
        return np.clip(out, -1.0, 1.0)

    @staticmethod
    def _build_ee_20d(
        ee_left: np.ndarray, grip_left: np.ndarray,
        ee_right: np.ndarray, grip_right: np.ndarray,
    ) -> np.ndarray:
        T = ee_left.shape[0]
        L_xyz = ee_left[:, :3]
        L_6d = quat_xyzw_to_6d(ee_left[:, 3:7])
        L_grip = grip_left.reshape(T, 1)
        R_xyz = ee_right[:, :3]
        R_6d = quat_xyzw_to_6d(ee_right[:, 3:7])
        R_grip = grip_right.reshape(T, 1)
        return np.concatenate(
            [L_xyz, L_6d, L_grip, R_xyz, R_6d, R_grip], axis=-1
        ).astype(np.float32)

    def _load_paraphrase(self, task_dir: str, ep_id: int) -> Optional[str]:
        key = (task_dir, ep_id)
        if key not in self._inst_cache:
            p = self.data_root / task_dir / "instructions" / f"episode{ep_id}.json"
            with open(p, "r") as f:
                d = json.load(f)
            self._inst_cache[key] = list(d.get("seen") or [])
        seen = self._inst_cache[key]
        if not seen:
            return None
        return self._paraphrase_rng.choice(seen)

    def _load_images(self, h5: h5py.File, frame_idx: int) -> list[Image.Image]:
        target_hw = self.image_size  # (H, W) per starVLA convention
        imgs = []
        for cam in self.cameras:
            raw = h5[f"observation/{cam}/rgb"][frame_idx]
            data = raw.tobytes() if hasattr(raw, "tobytes") else raw
            img = Image.open(io.BytesIO(data)).convert("RGB")
            # PIL.resize takes (W, H); image_size convention is (H, W).
            if img.size != (target_hw[1], target_hw[0]):
                img = img.resize((target_hw[1], target_hw[0]))
            imgs.append(img)
        return imgs

    def __getitem__(self, idx: int) -> dict:
        task_dir, tg, pk, ep_id, frame_idx = self._index[idx]
        h5_path = self.data_root / task_dir / "data" / f"episode{ep_id}.hdf5"

        with h5py.File(h5_path, "r") as h5:
            end_idx = frame_idx + self.chunk_size
            l_ee = h5["endpose/left_endpose"][frame_idx:end_idx]
            r_ee = h5["endpose/right_endpose"][frame_idx:end_idx]
            l_gr = h5["endpose/left_gripper"][frame_idx:end_idx]
            r_gr = h5["endpose/right_gripper"][frame_idx:end_idx]
            action_raw = self._build_ee_20d(l_ee, l_gr, r_ee, r_gr)

            if self.include_state:
                state_raw = self._build_ee_20d(
                    h5["endpose/left_endpose"][frame_idx:frame_idx + 1],
                    h5["endpose/left_gripper"][frame_idx:frame_idx + 1],
                    h5["endpose/right_endpose"][frame_idx:frame_idx + 1],
                    h5["endpose/right_gripper"][frame_idx:frame_idx + 1],
                )
            else:
                state_raw = None

            images = self._load_images(h5, frame_idx)

        if self.action_q01 is not None:
            action = self._normalize(action_raw, self.action_q01, self.action_q99)
        else:
            action = action_raw

        if state_raw is not None and self.state_q01 is not None:
            state = self._normalize(state_raw, self.state_q01, self.state_q99)
        else:
            state = state_raw

        paraphrase = self._load_paraphrase(task_dir, ep_id)
        lang = build_action_prompt(tg, pk, paraphrase=paraphrase, category=self.category)

        sample = {
            "image": images,
            "lang": lang,
            "language": lang,
            "action": action.astype(np.float16),
            "robot_tag": "pref_hdf5",
        }
        if state is not None:
            sample["state"] = state.astype(np.float16)
        return sample


def collate_fn(batch):
    return batch


def get_pref_dataset(data_cfg, mode: str = "train", **kwargs) -> PrefHDF5Dataset:
    """Factory mirroring lerobot_datasets.get_vla_dataset signature."""
    def _g(name, default):
        return data_cfg.get(name, default) if hasattr(data_cfg, "get") else default

    chunk_size = int(_g("future_action_window_size", 15)) + 1
    return PrefHDF5Dataset(
        data_root_dir=data_cfg.data_root_dir,
        split=mode,
        chunk_size=chunk_size,
        past_window=int(_g("past_action_window_size", 0)),
        include_state=bool(_g("include_state", False)),
        cameras=tuple(_g("cameras", DEFAULT_CAMERAS)),
        image_size=tuple(_g("image_size", DEFAULT_IMAGE_SIZE)),
        stats_json_path=_g("stats_json_path", None),
        category=str(_g("pref_category", DEFAULT_CATEGORY)),
        data_cfg=data_cfg,
        **kwargs,
    )


if __name__ == "__main__":
    # Smoke probe: load each category's val split, dump first sample.
    cases = [
        ("giveobj", "/mnt/localssd/kaiwenh/pref/data/giveobj"),
        ("height",  "/mnt/localssd/kaiwenh/pref/data/height"),
        ("hvlv",    "/mnt/localssd/kaiwenh/pref/data/hvlv"),
        ("orient",  "/mnt/localssd/kaiwenh/pref/data/orient"),
    ]
    for cat, root in cases:
        print(f"\n=== category={cat} root={root} ===")
        try:
            ds = PrefHDF5Dataset(root, split="val", include_state=True, category=cat)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")
            continue
        print(f"  val len: {len(ds)}")
        s = ds[0]
        print("  action.shape =", s["action"].shape, s["action"].dtype)
        print("  state.shape  =", s["state"].shape, s["state"].dtype)
        print("  image len    =", len(s["image"]), "first.size =", s["image"][0].size)
        print("  lang         =", s["lang"])
