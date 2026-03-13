#!/usr/bin/env python3
"""
Create a minimal debug dataset with only 1 trajectory from pickandplace-real-0307.

Requires: conda env with pandas, numpy (e.g. starVLA).

Usage:
    conda activate starVLA
    python realworld/create_pickandplace_debug_dataset.py
    python realworld/create_pickandplace_debug_dataset.py --episode 0 --out pickandplace-debug-1ep

Output: playground/Datasets/FastUMI/pickandplace-debug-1ep/
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "playground/Datasets/FastUMI/pickandplace-real-0307"
DEFAULT_DST = REPO_ROOT / "playground/Datasets/FastUMI/pickandplace-debug-1ep"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, default=0, help="Episode index to keep (0-based)")
    parser.add_argument("--out", type=Path, default=DEFAULT_DST, help="Output dataset directory")
    args = parser.parse_args()

    if not SRC.exists():
        print(f"ERROR: source dataset not found: {SRC}")
        return 1

    ep_idx = args.episode
    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)

    # Read source meta
    with open(SRC / "meta/episodes.jsonl") as f:
        all_episodes = [json.loads(line) for line in f]
    if ep_idx >= len(all_episodes):
        print(f"ERROR: episode {ep_idx} not found (max index {len(all_episodes)-1})")
        return 1

    ep_meta = all_episodes[ep_idx]
    ep_len = ep_meta["length"]
    ep_file = f"episode_{ep_idx:06d}"

    # Create data dir and copy parquet (reindex episode_index to 0)
    (dst / "data/chunk-000").mkdir(parents=True)
    src_parquet = SRC / f"data/chunk-000/{ep_file}.parquet"
    dst_parquet = dst / f"data/chunk-000/episode_000000.parquet"
    df_src = pd.read_parquet(src_parquet)
    if "episode_index" in df_src.columns:
        df_src = df_src.copy()
        df_src["episode_index"] = 0
    df_src.to_parquet(dst_parquet, index=False)
    print(f"Copied {src_parquet} -> {dst_parquet} (episode_index set to 0)")

    # Create video dir and copy video (reindex to episode 0)
    video_dir = dst / "videos/chunk-000/observation.images.wrist"
    video_dir.mkdir(parents=True)
    src_video = SRC / f"videos/chunk-000/observation.images.wrist/{ep_file}.mp4"
    dst_video = dst / "videos/chunk-000/observation.images.wrist/episode_000000.mp4"
    shutil.copy2(src_video, dst_video)
    print(f"Copied {src_video} -> {dst_video}")

    # meta/
    (dst / "meta").mkdir(exist_ok=True)

    # info.json
    with open(SRC / "meta/info.json") as f:
        info = json.load(f)
    info["total_episodes"] = 1
    info["total_frames"] = ep_len
    info["total_videos"] = 1
    with open(dst / "meta/info.json", "w") as f:
        json.dump(info, f, indent=2)
    print("Wrote meta/info.json")

    # episodes.jsonl
    ep_record = {"episode_index": 0, "length": ep_len, "tasks": ep_meta["tasks"]}
    with open(dst / "meta/episodes.jsonl", "w") as f:
        f.write(json.dumps(ep_record) + "\n")
    print("Wrote meta/episodes.jsonl")

    # modality.json, tasks.jsonl
    shutil.copy2(SRC / "meta/modality.json", dst / "meta/modality.json")
    shutil.copy2(SRC / "meta/tasks.jsonl", dst / "meta/tasks.jsonl")
    print("Copied meta/modality.json, meta/tasks.jsonl")

    # stats_gr00t.json - compute from single parquet, match format expected by gr00t loader
    df = pd.read_parquet(dst_parquet)
    state_arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df["observation.state"]])
    action_arr = np.vstack([np.asarray(x, dtype=np.float32) for x in df["action"]])

    def make_stats(arr):
        return {
            "min": np.min(arr, axis=0).tolist(),
            "max": np.max(arr, axis=0).tolist(),
            "mean": np.mean(arr, axis=0).tolist(),
            "std": np.std(arr, axis=0).tolist(),
            "q01": np.quantile(arr, 0.01, axis=0).tolist(),
            "q99": np.quantile(arr, 0.99, axis=0).tolist(),
        }

    gr00t_stats = {
        "abs": {
            "observation.state": make_stats(state_arr),
            "action": make_stats(action_arr),
        }
    }
    with open(dst / "meta/stats_gr00t.json", "w") as f:
        json.dump(gr00t_stats, f, indent=4)
    print("Wrote meta/stats_gr00t.json (computed from 1 episode)")

    print(f"\nDone. Debug dataset: {dst}")
    print(f"  1 episode, {ep_len} frames")
    print("\nTo use in training, add to mixtures.py and set data_mix:")
    print('  "pickandplace-debug-1ep": [("pickandplace-debug-1ep", 1.0, "fastumi")],')
    print("  --datasets.vla_data.data_mix pickandplace-debug-1ep")
    return 0


if __name__ == "__main__":
    exit(main())
