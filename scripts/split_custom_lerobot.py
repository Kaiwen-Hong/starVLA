#!/usr/bin/env python3
"""Split a merged LeRobot dataset into per-task directories for StarVLA training.

Splits custom_all_repo (3300 episodes, 33 task variants x 100 each) into
individual LeRobot v2.1 datasets under playground/Datasets/Custom/.

Usage:
    python scripts/split_custom_lerobot.py \
        --src /path/to/.cache/huggingface/lerobot/custom_all_repo \
        --dst playground/Datasets/Custom \
        --tasks adjust_bottle adjust_bottle_wp1 adjust_bottle_wp2 ... \
        --episodes-per-task 100 \
        --workers 8
"""

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


MODALITY_JSON = {
    "action": {
        "left_joints":  {"start": 0,  "end": 6,  "original_key": "action"},
        "left_gripper": {"start": 6,  "end": 7,  "original_key": "action"},
        "right_joints": {"start": 7,  "end": 13, "original_key": "action"},
        "right_gripper":{"start": 13, "end": 14, "original_key": "action"},
    },
    "state": {
        "left_joints":  {"start": 0,  "end": 6,  "original_key": "observation.state"},
        "left_gripper": {"start": 6,  "end": 7,  "original_key": "observation.state"},
        "right_joints": {"start": 7,  "end": 13, "original_key": "observation.state"},
        "right_gripper":{"start": 13, "end": 14, "original_key": "observation.state"},
    },
    "video": {
        "cam_high":       {"original_key": "observation.images.cam_high"},
        "cam_left_wrist": {"original_key": "observation.images.cam_left_wrist"},
        "cam_right_wrist":{"original_key": "observation.images.cam_right_wrist"},
    },
    "annotation": {
        "human.action.task_description": {"original_key": "task_index"},
    },
}


def process_single_task(src, dst, task_name, task_idx, total_tasks,
                        all_episodes, src_info, episodes_per_task, chunks_size,
                        modality_json=None):
    """Process one task: read source parquets, remap metadata columns, write per-task dataset."""
    ep_start = task_idx * episodes_per_task
    ep_end = ep_start + episodes_per_task
    task_episodes = all_episodes[ep_start:ep_end]

    task_dir = dst / task_name
    meta_dir = task_dir / "meta"
    data_dir = task_dir / "data" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # O(1) task text lookup instead of O(n) list scan
    task_text_to_idx = {}
    local_task_list = []

    new_episodes = []
    total_frames = 0
    frame_offset = 0  # running counter instead of O(n^2) recomputation

    t_task = time.time()
    for local_ep_idx, ep in enumerate(task_episodes):
        global_ep_idx = ep["episode_index"]
        ep_length = ep["length"]

        # Register unique task texts
        for task_text in ep["tasks"]:
            if task_text not in task_text_to_idx:
                new_ti = len(local_task_list)
                task_text_to_idx[task_text] = new_ti
                local_task_list.append({"task_index": new_ti, "task": task_text})

        new_episodes.append({
            "episode_index": local_ep_idx,
            "tasks": ep["tasks"],
            "length": ep_length,
        })

        # Read source parquet
        src_chunk = global_ep_idx // chunks_size
        src_parquet = src / "data" / f"chunk-{src_chunk:03d}" / f"episode_{global_ep_idx:06d}.parquet"
        table = pq.read_table(src_parquet)

        n_rows = len(table)
        ep_task_text = ep["tasks"][0]
        new_task_idx = task_text_to_idx[ep_task_text]

        # Replace only the 3 metadata columns; image columns pass through as zero-copy references
        new_columns = {}
        for col_name in table.column_names:
            if col_name == "episode_index":
                new_columns[col_name] = pa.array([local_ep_idx] * n_rows, type=pa.int64())
            elif col_name == "index":
                new_columns[col_name] = pa.array(
                    list(range(frame_offset, frame_offset + n_rows)), type=pa.int64()
                )
            elif col_name == "task_index":
                new_columns[col_name] = pa.array([new_task_idx] * n_rows, type=pa.int64())
            else:
                new_columns[col_name] = table[col_name]

        new_table = pa.table(new_columns)
        dst_parquet = data_dir / f"episode_{local_ep_idx:06d}.parquet"
        pq.write_table(new_table, dst_parquet)

        frame_offset += ep_length
        total_frames += ep_length

        # Progress every 10 episodes
        if (local_ep_idx + 1) % 10 == 0 or local_ep_idx == 0:
            elapsed = time.time() - t_task
            print(f"  [{task_name}] {local_ep_idx+1}/{episodes_per_task} episodes  ({elapsed:.1f}s)", flush=True)

    # Write meta/info.json
    info = {
        "codebase_version": src_info["codebase_version"],
        "robot_type": src_info["robot_type"],
        "total_episodes": episodes_per_task,
        "total_frames": total_frames,
        "total_tasks": len(local_task_list),
        "total_videos": 0,
        "total_chunks": 1,
        "chunks_size": chunks_size,
        "fps": src_info["fps"],
        "splits": {"train": f"0:{episodes_per_task}"},
        "data_path": src_info["data_path"],
        "video_path": src_info.get("video_path", ""),
        "features": src_info["features"],
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # Write meta/episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep in new_episodes:
            f.write(json.dumps(ep) + "\n")

    # Write meta/tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        for t in local_task_list:
            f.write(json.dumps(t) + "\n")

    # Write meta/modality.json
    modality_to_write = modality_json if modality_json is not None else MODALITY_JSON
    with open(meta_dir / "modality.json", "w") as f:
        json.dump(modality_to_write, f, indent=4)

    return task_name, episodes_per_task, total_frames, len(local_task_list)


def split_dataset(src: Path, dst: Path, task_names: list[str],
                  episodes_per_task: int, workers: int, modality_json=None):
    # Read source metadata
    with open(src / "meta" / "info.json") as f:
        src_info = json.load(f)

    with open(src / "meta" / "episodes.jsonl") as f:
        all_episodes = [json.loads(line) for line in f]

    total_expected = len(task_names) * episodes_per_task
    assert total_expected <= len(all_episodes), (
        f"Expected at most {len(all_episodes)} episodes, but {len(task_names)} tasks x "
        f"{episodes_per_task} eps = {total_expected}"
    )

    chunks_size = src_info.get("chunks_size", 1000)
    total_tasks = len(task_names)

    print(f"Splitting {total_expected} episodes into {total_tasks} tasks with {workers} workers...", flush=True)
    t0 = time.time()

    if workers <= 1:
        # Sequential fallback
        for task_idx, task_name in enumerate(task_names):
            result = process_single_task(
                src, dst, task_name, task_idx, total_tasks,
                all_episodes, src_info, episodes_per_task, chunks_size,
                modality_json,
            )
            name, n_eps, n_frames, n_tasks = result
            print(f"[{task_idx+1:2d}/{total_tasks}] {name}: {n_eps} episodes, {n_frames} frames, {n_tasks} unique tasks")
    else:
        # Parallel: each task is fully independent
        futures = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for task_idx, task_name in enumerate(task_names):
                fut = executor.submit(
                    process_single_task,
                    src, dst, task_name, task_idx, total_tasks,
                    all_episodes, src_info, episodes_per_task, chunks_size,
                    modality_json,
                )
                futures[fut] = task_idx

            for i, fut in enumerate(as_completed(futures), 1):
                name, n_eps, n_frames, n_tasks = fut.result()
                print(f"[{i:2d}/{total_tasks}] DONE {name}: {n_eps} episodes, {n_frames} frames, {n_tasks} unique tasks", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone. {total_tasks} task datasets written to {dst}  ({elapsed:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Split merged LeRobot dataset into per-task dirs")
    parser.add_argument("--src", type=Path, required=True, help="Path to merged LeRobot dataset")
    parser.add_argument("--dst", type=Path, required=True, help="Output directory for per-task datasets")
    parser.add_argument("--tasks", nargs="+", required=True, help="Ordered list of task variant names")
    parser.add_argument("--episodes-per-task", type=int, default=100, help="Episodes per task variant")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (default: 0 = one per task, use 1 for sequential)")
    parser.add_argument("--modality-file", type=Path, default=None,
                        help="Path to a custom modality JSON file (default: use built-in 14D joint-space)")
    args = parser.parse_args()

    modality_json = None
    if args.modality_file is not None:
        with open(args.modality_file) as f:
            modality_json = json.load(f)
        print(f"Using custom modality from: {args.modality_file}")

    workers = args.workers if args.workers > 0 else len(args.tasks)
    split_dataset(args.src, args.dst, args.tasks, args.episodes_per_task, workers, modality_json)


if __name__ == "__main__":
    main()
