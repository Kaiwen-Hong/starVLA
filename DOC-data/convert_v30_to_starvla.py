#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert LeRobot v3.0 dataset to StarVLA-compatible v2.1 format.

Key changes:
1. Split single parquet -> per-episode parquet files
2. Split combined video -> per-episode mp4 files (parallel ffmpeg)
3. Generate meta/modality.json, episodes.jsonl, tasks.jsonl
4. Convert info.json from v3.0 to v2.1 schema
5. Optional session-level train/val split (prevents data leakage)

Usage:
    # Full conversion (all episodes)
    python scripts/convert_v30_to_starvla.py \
        --src ~/.cache/huggingface/lerobot/umi-pickandplacevla-rot6d-rel-256-cropped-5x \
        --dst /path/to/starvla/datasets/pickandplace_vla

    # Train-only (session-level split, exclude last 20% sessions)
    python scripts/convert_v30_to_starvla.py \
        --src ~/.cache/huggingface/lerobot/umi-pickandplacevla-rot6d-rel-256-cropped-5x \
        --dst /path/to/starvla/datasets/pickandplace_vla \
        --train-only --val-ratio 0.2

    # Custom task description
    python scripts/convert_v30_to_starvla.py \
        --src ~/.cache/huggingface/lerobot/umi-pickandplacevla-rot6d-rel-256-cropped-5x \
        --dst /path/to/starvla/datasets/pickandplace_vla \
        --task "pick up the object and place it on the target"
"""

import argparse
import json
import subprocess
import shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# VIDEO SPLITTING
# ============================================================

def split_video_episode(args):
    """Split one episode from the combined video file. Runs in a worker process."""
    video_src, from_ts, n_frames, video_dst, codec, fps = args

    video_dst = Path(video_dst)
    video_dst.parent.mkdir(parents=True, exist_ok=True)

    if codec == "h264":
        codec_args = ["-c:v", "libx264", "-preset", "fast", "-crf", "20"]
    elif codec == "av1":
        codec_args = ["-c:v", "libsvtav1", "-crf", "30", "-preset", "6"]
    else:
        raise ValueError(f"Unknown codec: {codec}")

    # -ss AFTER -i for frame-accurate seeking (slower but correct).
    # Placing -ss before -i does keyframe-based seeking which can be off by
    # several seconds with H.264, silently producing wrong frame ranges.
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_src),
        "-ss", f"{from_ts:.6f}",
        "-frames:v", str(n_frames),
        *codec_args,
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-an",
        str(video_dst),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, f"episode {video_dst.stem}: {result.stderr[-300:]}"
    return True, video_dst.stem


# ============================================================
# PARQUET CONVERSION
# ============================================================

def convert_parquet_arrays_to_lists(df):
    """
    Convert numpy array columns to Python lists for clean parquet schema.
    StarVLA/pyarrow reads list columns more reliably than object(ndarray) columns.
    """
    for col in ["observation.state", "action"]:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: x.tolist() if isinstance(x, np.ndarray) else list(x)
            )
    return df


# ============================================================
# SESSION-LEVEL SPLIT
# ============================================================

def compute_train_episodes(session_mapping_path, val_ratio=0.2):
    """
    Determine which episodes belong to training set using session-level split.

    Uses chronological split: last val_ratio fraction of sessions become validation.
    This prevents data leakage from augmented copies of the same session.

    Returns:
        (train_episode_indices, val_session_names)
    """
    mapping = json.load(open(session_mapping_path))
    session_to_episodes = mapping["session_to_episodes"]
    sessions = sorted(session_to_episodes.keys())  # chronological order

    n_val = max(1, int(len(sessions) * val_ratio))
    n_train = len(sessions) - n_val

    train_sessions = sessions[:n_train]
    val_sessions = sessions[n_train:]

    train_episodes = []
    for session in train_sessions:
        train_episodes.extend(session_to_episodes[session])

    return sorted(train_episodes), val_sessions


# ============================================================
# META FILE GENERATION
# ============================================================

def generate_modality_json():
    """
    Generate modality.json for FastUMI single-arm robot.

    State: [x, y, z, rot6d(6), gripper] = 10D (absolute)
    Action: [rel_x, rel_y, rel_z, rel_rot6d(6), gripper] = 10D (already relative)
    Video: single wrist camera
    """
    return {
        "action": {
            "eef_pos": {
                "start": 0, "end": 3,
                "original_key": "action",
                "absolute": False,
            },
            "eef_rot6d": {
                "start": 3, "end": 9,
                "original_key": "action",
                "rotation_type": "rotation_6d",
                "absolute": False,
            },
            "gripper": {
                "start": 9, "end": 10,
                "original_key": "action",
                "absolute": False,
            },
        },
        "state": {
            "eef_pos": {
                "start": 0, "end": 3,
                "original_key": "observation.state",
            },
            "eef_rot6d": {
                "start": 3, "end": 9,
                "original_key": "observation.state",
                "rotation_type": "rotation_6d",
            },
            "gripper": {
                "start": 9, "end": 10,
                "original_key": "observation.state",
            },
        },
        "video": {
            "wrist": {
                "original_key": "observation.images.wrist",
            },
        },
        "annotation": {
            "human.action.task_description": {
                "original_key": "task_index",
            },
        },
    }


def generate_info_json(n_episodes, total_frames, fps, image_h, image_w, video_codec):
    """Generate StarVLA-compatible info.json (v2.1 schema)."""
    n_chunks = (n_episodes - 1) // 1000 + 1

    codec_str = "h264" if video_codec == "h264" else "av1"

    return {
        "codebase_version": "v2.1",
        "robot_type": "fastumi",
        "total_episodes": n_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": n_episodes,
        "total_chunks": n_chunks,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {
            "train": f"0:{n_episodes}",
        },
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [10],
                "names": None,
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [image_h, image_w, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": image_h,
                    "video.width": image_w,
                    "video.channels": 3,
                    "video.fps": float(fps),
                    "video.codec": codec_str,
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                },
            },
            "action": {
                "dtype": "float32",
                "shape": [10],
                "names": None,
            },
            "timestamp": {
                "dtype": "float32",
                "shape": [1],
                "names": None,
            },
            "frame_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "episode_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
            "task_index": {
                "dtype": "int64",
                "shape": [1],
                "names": None,
            },
        },
    }


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert(
    src_dir,
    dst_dir,
    task_description=None,
    train_only=False,
    val_ratio=0.2,
    video_codec="h264",
    num_workers=8,
):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    # ----------------------------------------------------------
    # 1. Load source data
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"LeRobot v3.0 -> StarVLA v2.1 Converter")
    print(f"{'='*70}")
    print(f"Source:  {src_dir}")
    print(f"Output:  {dst_dir}")

    info_v3 = json.load(open(src_dir / "meta/info.json"))
    fps = info_v3["fps"]
    n_episodes_total = info_v3["total_episodes"]

    # Extract image dimensions from v3.0 info
    img_shape = info_v3["features"]["observation.images.wrist"]["shape"]
    # v3.0 uses (C, H, W); extract H, W
    if img_shape[0] == 3:
        image_h, image_w = img_shape[1], img_shape[2]
    else:
        image_h, image_w = img_shape[0], img_shape[1]

    print(f"FPS: {fps}, Image: {image_h}x{image_w}")
    print(f"Total episodes (source): {n_episodes_total}")

    # Load task description
    tasks_df = pd.read_parquet(src_dir / "meta/tasks.parquet")
    if task_description is None:
        # v3.0 tasks.parquet: task text may be in "task" column or as the index
        if "task" in tasks_df.columns:
            task_description = str(tasks_df["task"].iloc[0])
        else:
            task_description = str(tasks_df.index[0])
    print(f"Task: \"{task_description}\"")

    # Load data parquet
    print("\nLoading parquet data...")
    data_df = pd.read_parquet(src_dir / "data/chunk-000/file-000.parquet")
    print(f"  {len(data_df)} frames loaded")

    # Load episode metadata (for video timestamps)
    episode_meta = pd.read_parquet(
        src_dir / "meta/episodes/chunk-000/file-000.parquet"
    )

    # ----------------------------------------------------------
    # 2. Session-level split (optional)
    # ----------------------------------------------------------
    session_mapping_path = src_dir / "episode_session_mapping.json"

    if train_only and session_mapping_path.exists():
        print(f"\nApplying session-level train/val split (val_ratio={val_ratio})...")
        episodes_to_include, val_sessions = compute_train_episodes(
            session_mapping_path, val_ratio
        )
        mapping = json.load(open(session_mapping_path))
        n_sessions = mapping["n_sessions"]
        n_train_sessions = n_sessions - len(val_sessions)
        print(f"  Sessions: {n_sessions} total, {n_train_sessions} train, {len(val_sessions)} val")
        print(f"  Episodes: {len(episodes_to_include)} train (from {n_episodes_total} total)")
        print(f"  Val sessions: {val_sessions[:5]}{'...' if len(val_sessions) > 5 else ''}")
    else:
        episodes_to_include = list(range(n_episodes_total))
        if train_only and not session_mapping_path.exists():
            print(f"\n[WARN] --train-only specified but no session mapping found at {session_mapping_path}")
            print(f"  Using all {n_episodes_total} episodes.")
        else:
            print(f"\nUsing all {n_episodes_total} episodes (no split).")

    # ----------------------------------------------------------
    # 3. Create output directory
    # ----------------------------------------------------------
    if dst_dir.exists():
        print(f"\nRemoving existing output: {dst_dir}")
        shutil.rmtree(dst_dir)

    (dst_dir / "meta").mkdir(parents=True)
    (dst_dir / "data/chunk-000").mkdir(parents=True)
    (dst_dir / "videos/chunk-000/observation.images.wrist").mkdir(parents=True)

    # ----------------------------------------------------------
    # 4. Split parquet per episode
    # ----------------------------------------------------------
    n_episodes = len(episodes_to_include)
    print(f"\n[Step 1/3] Splitting parquet into {n_episodes} per-episode files...")

    episodes_jsonl = []
    total_frames = 0
    global_index = 0

    for new_ep_idx, old_ep_idx in enumerate(
        tqdm(episodes_to_include, desc="Writing parquet")
    ):
        ep_data = data_df[data_df["episode_index"] == old_ep_idx].copy()
        ep_data = ep_data.reset_index(drop=True)

        # Remap indices for the new dataset
        n_frames = len(ep_data)
        ep_data["episode_index"] = new_ep_idx
        ep_data["frame_index"] = range(n_frames)
        ep_data["index"] = range(global_index, global_index + n_frames)
        ep_data["task_index"] = 0
        # Reset timestamps to [0, 1/fps, 2/fps, ...] so they align with
        # per-episode video files (which start at t=0).  The v3.0 source
        # timestamps *should* already be per-episode, but this guarantees it.
        ep_data["timestamp"] = [i / fps for i in range(n_frames)]

        # Convert numpy arrays to lists for clean parquet schema
        ep_data = convert_parquet_arrays_to_lists(ep_data)

        # Write per-episode parquet
        chunk_idx = new_ep_idx // 1000
        chunk_dir = dst_dir / f"data/chunk-{chunk_idx:03d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        ep_data.to_parquet(
            chunk_dir / f"episode_{new_ep_idx:06d}.parquet",
            index=False,
        )

        episodes_jsonl.append({
            "episode_index": new_ep_idx,
            "length": n_frames,
            "tasks": [task_description],
        })

        total_frames += n_frames
        global_index += n_frames

    print(f"  {n_episodes} parquet files, {total_frames} total frames")

    # ----------------------------------------------------------
    # 5. Split video per episode (parallel)
    # ----------------------------------------------------------
    print(f"\n[Step 2/3] Splitting video into {n_episodes} per-episode mp4 ({video_codec})...")

    video_src = src_dir / "videos/observation.images.wrist/chunk-000/file-000.mp4"
    if not video_src.exists():
        print(f"  [ERROR] Source video not found: {video_src}")
        print(f"  Skipping video splitting. You will need to provide videos manually.")
    else:
        video_tasks = []
        for new_ep_idx, old_ep_idx in enumerate(episodes_to_include):
            # Filter by episode_index column instead of row position to
            # avoid silent bugs when parquet row order != episode_index.
            ep_meta_row = episode_meta[
                episode_meta["episode_index"] == old_ep_idx
            ]
            if len(ep_meta_row) == 0:
                print(f"  [WARN] No episode metadata for episode_index={old_ep_idx}, skipping video")
                continue
            ep_meta_row = ep_meta_row.iloc[0]
            from_ts = ep_meta_row["videos/observation.images.wrist/from_timestamp"]
            n_frames = ep_meta_row["length"]

            chunk_idx = new_ep_idx // 1000
            video_dst = (
                dst_dir
                / f"videos/chunk-{chunk_idx:03d}/observation.images.wrist"
                / f"episode_{new_ep_idx:06d}.mp4"
            )
            video_tasks.append(
                (str(video_src), from_ts, n_frames, str(video_dst), video_codec, fps)
            )

        success_count = 0
        fail_count = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(split_video_episode, task): task
                for task in video_tasks
            }
            with tqdm(total=len(video_tasks), desc="Encoding videos") as pbar:
                for future in as_completed(futures):
                    ok, msg = future.result()
                    if ok:
                        success_count += 1
                    else:
                        fail_count += 1
                        tqdm.write(f"  [FAIL] {msg}")
                    pbar.update(1)

        print(f"  Videos: {success_count} OK, {fail_count} failed")

    # ----------------------------------------------------------
    # 6. Generate meta files
    # ----------------------------------------------------------
    print(f"\n[Step 3/3] Generating meta files...")

    # tasks.jsonl
    with open(dst_dir / "meta/tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_description}) + "\n")

    # episodes.jsonl
    with open(dst_dir / "meta/episodes.jsonl", "w") as f:
        for ep in episodes_jsonl:
            f.write(json.dumps(ep) + "\n")

    # modality.json
    with open(dst_dir / "meta/modality.json", "w") as f:
        json.dump(generate_modality_json(), f, indent=2)

    # info.json
    info_v21 = generate_info_json(
        n_episodes, total_frames, fps, image_h, image_w, video_codec
    )
    with open(dst_dir / "meta/info.json", "w") as f:
        json.dump(info_v21, f, indent=2)

    # ----------------------------------------------------------
    # 7. Verify
    # ----------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Conversion complete!")
    print(f"{'='*70}")
    print(f"  Output:      {dst_dir}")
    print(f"  Episodes:    {n_episodes}")
    print(f"  Frames:      {total_frames}")
    print(f"  FPS:         {fps}")
    print(f"  Image:       {image_h}x{image_w}")
    print(f"  Video codec: {video_codec}")
    print(f"  Task:        \"{task_description}\"")
    print(f"  robot_type:  fastumi")
    print()
    print("Files created:")
    print(f"  meta/info.json       - Dataset metadata (v2.1)")
    print(f"  meta/modality.json   - Modality mapping for StarVLA")
    print(f"  meta/episodes.jsonl  - Episode index ({n_episodes} entries)")
    print(f"  meta/tasks.jsonl     - Task descriptions (1 entry)")
    print(f"  data/chunk-000/      - Per-episode parquet files ({n_episodes} files)")
    print(f"  videos/chunk-000/    - Per-episode mp4 files ({n_episodes} files)")
    print()
    print("Next steps:")
    print("  1. Add FastUMIDataConfig to starVLA/dataloader/gr00t_lerobot/data_config.py")
    print("  2. Register 'fastumi' in ROBOT_TYPE_CONFIG_MAP")
    print("  3. Add mixture to starVLA/dataloader/gr00t_lerobot/mixtures.py")
    print(f"  4. Set --datasets.vla_data.data_root_dir to parent of '{dst_dir.name}'")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to StarVLA-compatible v2.1 format",
    )
    parser.add_argument(
        "--src", required=True,
        help="Source v3.0 dataset directory",
    )
    parser.add_argument(
        "--dst", required=True,
        help="Output v2.1 dataset directory (will be created/overwritten)",
    )
    parser.add_argument(
        "--task", default=None,
        help="Task description string (default: from source tasks.parquet)",
    )
    parser.add_argument(
        "--train-only", action="store_true",
        help="Only export training episodes (requires episode_session_mapping.json)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Fraction of sessions for validation (default: 0.2)",
    )
    parser.add_argument(
        "--video-codec", choices=["h264", "av1"], default="h264",
        help="Video codec for per-episode mp4 (default: h264)",
    )
    parser.add_argument(
        "--num-workers", type=int, default=8,
        help="Parallel workers for video splitting (default: 8)",
    )
    args = parser.parse_args()

    convert(
        src_dir=args.src,
        dst_dir=args.dst,
        task_description=args.task,
        train_only=args.train_only,
        val_ratio=args.val_ratio,
        video_codec=args.video_codec,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
