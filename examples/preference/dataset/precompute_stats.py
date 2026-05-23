"""
Pre-compute 20D action / state quantile stats over the TRAIN split of a
pref-VLA category (giveobj / height / hvlv / orient).

Writes JSON to examples/preference/dataset/stats_<category>_v1.json. The
runtime PrefHDF5Dataset loads this file via `stats_json_path` and applies
q99 clip to map every action/state dim into [-1, 1].

Action layout (20D, see pref_hdf5_dataset._build_ee_20d):
  [Lxyz(3), L6Drot(6), Lgrip(1), Rxyz(3), R6Drot(6), Rgrip(1)]
State has the same layout (current frame only).

Run:
  python -m examples.preference.dataset.precompute_stats \
      --data_root /mnt/localssd/kaiwenh/pref/data/height \
      --category height
  # Auto-derives --out from --category if --out not provided.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np

from .pref_hdf5_dataset import _split_episodes, _task_dirs_for_groups
from .prompt import PREF_CATEGORIES
from .rotation import quat_xyzw_to_6d


def _build_ee_20d_full_episode(h5: h5py.File) -> np.ndarray:
    l_ee = h5["endpose/left_endpose"][:]
    r_ee = h5["endpose/right_endpose"][:]
    l_gr = h5["endpose/left_gripper"][:]
    r_gr = h5["endpose/right_gripper"][:]
    T = l_ee.shape[0]
    L_xyz = l_ee[:, :3]
    L_6d = quat_xyzw_to_6d(l_ee[:, 3:7])
    L_grip = l_gr.reshape(T, 1)
    R_xyz = r_ee[:, :3]
    R_6d = quat_xyzw_to_6d(r_ee[:, 3:7])
    R_grip = r_gr.reshape(T, 1)
    return np.concatenate(
        [L_xyz, L_6d, L_grip, R_xyz, R_6d, R_grip], axis=-1
    ).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="e.g. /mnt/localssd/kaiwenh/pref/data/height")
    p.add_argument("--category", default="giveobj", choices=list(PREF_CATEGORIES),
                   help="One of giveobj / height / hvlv / orient")
    p.add_argument("--out", default=None,
                   help="Default: examples/preference/dataset/stats_<category>_v1.json")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    args = p.parse_args()

    if args.out is None:
        args.out = f"examples/preference/dataset/stats_{args.category}_v1.json"

    cat = PREF_CATEGORIES[args.category]
    task_groups = cat.task_groups
    pref_keys = cat.pref_keys

    data_root = Path(args.data_root)
    ep_split = _split_episodes(
        data_root,
        task_groups,
        pref_keys,
        val_fraction=args.val_fraction,
        seed=args.split_seed,
    )

    pooled = []
    n_train_ep = 0
    n_val_ep = 0
    per_task_counts = {}
    for task_dir, _, _ in _task_dirs_for_groups(data_root, task_groups, pref_keys):
        train_eps = ep_split[task_dir]["train"]
        val_eps = ep_split[task_dir]["val"]
        n_train_ep += len(train_eps)
        n_val_ep += len(val_eps)
        per_task_counts[task_dir] = {"train": len(train_eps), "val": len(val_eps)}
        for ep_id in train_eps:
            with h5py.File(data_root / task_dir / "data" / f"episode{ep_id}.hdf5", "r") as h5:
                pooled.append(_build_ee_20d_full_episode(h5))

    arr = np.concatenate(pooled, axis=0)
    print(f"Pooled {len(arr):,} train frames across {n_train_ep} train episodes "
          f"({n_val_ep} held out for val).", file=sys.stderr)

    q01 = np.quantile(arr, 0.01, axis=0)
    q99 = np.quantile(arr, 0.99, axis=0)
    mn = arr.min(axis=0)
    mx = arr.max(axis=0)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    body = {
        "action": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "min": mn.tolist(),
            "max": mx.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "state": {
            "q01": q01.tolist(),
            "q99": q99.tolist(),
            "min": mn.tolist(),
            "max": mx.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
        "meta": {
            "action_dim": 20,
            "state_dim": 20,
            "layout": "[Lxyz(3), L6Drot(6), Lgrip(1), Rxyz(3), R6Drot(6), Rgrip(1)]",
            "category": args.category,
            "n_train_episodes": n_train_ep,
            "n_val_episodes": n_val_ep,
            "n_train_frames": int(arr.shape[0]),
            "split_seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "task_groups": list(task_groups),
            "pref_keys": list(pref_keys),
            "per_task_episode_counts": per_task_counts,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(body, f, indent=2)
    print(f"Wrote {out_path}", file=sys.stderr)

    print("\nPer-dim span (q99 - q01):", file=sys.stderr)
    labels = (
        ["L_x", "L_y", "L_z"]
        + [f"L_6d_{i}" for i in range(6)]
        + ["L_grip"]
        + ["R_x", "R_y", "R_z"]
        + [f"R_6d_{i}" for i in range(6)]
        + ["R_grip"]
    )
    for i, lbl in enumerate(labels):
        print(f"  {lbl:<8}  q01={q01[i]:+.3f}  q99={q99[i]:+.3f}  span={q99[i]-q01[i]:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
