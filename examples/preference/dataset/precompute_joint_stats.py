"""Pre-compute 14D JOINT (qpos) action/state stats over the TRAIN split of a
pref-VLA category, matching the format of the existing stats_<cat>_0526_joint.json.

Joint layout (14D = joint_action/vector): [left_arm6, left_grip1, right_arm6, right_grip1].
Same split as everything else (seed 42, val_fraction 0.2) so it is consistent with
the action_space=joint OFT runs.

Run:
  python -m examples.preference.dataset.precompute_joint_stats \
      --data_root /mnt/localssd/kaiwenh/pref/data/0526/contact --category contact
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import h5py
import numpy as np

from .pref_hdf5_dataset import _split_episodes, _task_dirs_for_groups
from .prompt import PREF_CATEGORIES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--category", required=True, choices=list(PREF_CATEGORIES))
    p.add_argument("--out", default=None)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_fraction", type=float, default=0.2)
    args = p.parse_args()
    if args.out is None:
        args.out = f"examples/preference/dataset/stats_{args.category}_0526_joint.json"

    cat = PREF_CATEGORIES[args.category]
    data_root = Path(args.data_root)
    ep_split = _split_episodes(data_root, cat.task_groups, cat.pref_keys,
                              val_fraction=args.val_fraction, seed=args.split_seed)
    pooled, n_train = [], 0
    for task_dir, _, _ in _task_dirs_for_groups(data_root, cat.task_groups, cat.pref_keys):
        for ep_id in ep_split[task_dir]["train"]:
            with h5py.File(data_root / task_dir / "data" / f"episode{ep_id}.hdf5", "r") as h5:
                pooled.append(np.asarray(h5["joint_action/vector"][:], dtype=np.float32))
            n_train += 1
    arr = np.concatenate(pooled, axis=0)
    assert arr.shape[1] == 14, f"expected 14D joint, got {arr.shape}"
    stats = {
        "q01": np.quantile(arr, 0.01, axis=0).tolist(),
        "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        "min": arr.min(0).tolist(), "max": arr.max(0).tolist(),
        "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
    }
    body = {"action": stats, "state": stats, "meta": {
        "action_space": "joint_14d_qpos",
        "order": "[left_arm6, left_grip1, right_arm6, right_grip1]",
        "category": args.category, "n_train_episodes": n_train,
        "n_train_frames": int(arr.shape[0]),
        "split_seed": args.split_seed, "val_fraction": args.val_fraction,
    }}
    Path(args.out).write_text(json.dumps(body, indent=2))
    print(f"Wrote {args.out}: {n_train} train eps, {arr.shape[0]} frames", file=sys.stderr)


if __name__ == "__main__":
    main()
