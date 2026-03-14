#!/usr/bin/env python3
"""Script 1: File & Structure Validation

Validates that all 250 parquet + 250 mp4 files exist and are well-formed,
checks column schemas, array shapes, index consistency, video resolution,
frame counts, and episode-session mapping.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "playground/Datasets/FastUMI/pickandplace-real-0307"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


NUM_EPISODES = 250
EXPECTED_TOTAL_FRAMES = 22224
EXPECTED_COLUMNS = {
    "observation.state", "action", "timestamp",
    "frame_index", "episode_index", "index", "task_index",
}


def load_episode_lengths():
    """Load expected lengths from meta/episodes.jsonl."""
    lengths = {}
    with open(DATASET_ROOT / "meta/episodes.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            lengths[rec["episode_index"]] = rec["length"]
    return lengths


def check_parquet(ep_idx, expected_len):
    """Validate a single parquet file. Returns list of error strings."""
    errors = []
    pq_path = DATASET_ROOT / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    if not pq_path.exists():
        return [f"ep{ep_idx}: parquet missing"]

    try:
        df = pd.read_parquet(pq_path)
    except Exception as e:
        return [f"ep{ep_idx}: parquet unreadable: {e}"]

    # Column check
    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        errors.append(f"ep{ep_idx}: missing columns {missing_cols}")

    # Row count vs episodes.jsonl
    if len(df) != expected_len:
        errors.append(f"ep{ep_idx}: row count {len(df)} != expected {expected_len}")

    # State/action shape and dtype
    for col in ["observation.state", "action"]:
        if col not in df.columns:
            continue
        arr = np.stack(df[col].values)
        if arr.shape != (len(df), 10):
            errors.append(f"ep{ep_idx}: {col} shape {arr.shape}, expected ({len(df)}, 10)")
        if not np.issubdtype(arr.dtype, np.floating):
            errors.append(f"ep{ep_idx}: {col} dtype {arr.dtype}, expected float")
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            errors.append(f"ep{ep_idx}: {col} contains NaN or Inf")

    # frame_index sequential from 0
    if "frame_index" in df.columns:
        fi = df["frame_index"].values
        expected_fi = np.arange(len(df))
        if not np.array_equal(fi, expected_fi):
            errors.append(f"ep{ep_idx}: frame_index not sequential 0..{len(df)-1}")

    # episode_index matches filename
    if "episode_index" in df.columns:
        ei = df["episode_index"].values
        if not np.all(ei == ep_idx):
            errors.append(f"ep{ep_idx}: episode_index mismatch (found {np.unique(ei)})")

    return errors


def check_video(ep_idx, expected_len):
    """Validate a single mp4 file using metadata only (no full decode)."""
    errors = []
    vid_path = DATASET_ROOT / f"videos/chunk-000/observation.images.wrist/episode_{ep_idx:06d}.mp4"
    if not vid_path.exists():
        return [f"ep{ep_idx}: video missing"]

    try:
        import av
        container = av.open(str(vid_path))
        stream = container.streams.video[0]
        w, h = stream.width, stream.height
        if (w, h) != (256, 256):
            errors.append(f"ep{ep_idx}: video resolution {w}x{h}, expected 256x256")
        # Use metadata frame count (avoids full decode)
        n_frames = stream.frames
        if n_frames > 0 and abs(n_frames - expected_len) > 1:
            errors.append(f"ep{ep_idx}: video frames {n_frames} vs parquet rows {expected_len} (diff > 1)")
        container.close()
    except ImportError:
        try:
            import cv2
            cap = cv2.VideoCapture(str(vid_path))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if (w, h) != (256, 256):
                errors.append(f"ep{ep_idx}: video resolution {w}x{h}, expected 256x256")
            if n_frames > 0 and abs(n_frames - expected_len) > 1:
                errors.append(f"ep{ep_idx}: video frames {n_frames} vs parquet rows {expected_len} (diff > 1)")
        except Exception as e:
            errors.append(f"ep{ep_idx}: video check failed: {e}")
    except Exception as e:
        errors.append(f"ep{ep_idx}: video check failed: {e}")

    return errors


def check_info_json():
    """Validate meta/info.json."""
    errors = []
    info_path = DATASET_ROOT / "meta/info.json"
    if not info_path.exists():
        return ["meta/info.json missing"]
    with open(info_path) as f:
        info = json.load(f)
    if info.get("total_episodes") != NUM_EPISODES:
        errors.append(f"info.json: total_episodes={info.get('total_episodes')}, expected {NUM_EPISODES}")
    if info.get("total_frames") != EXPECTED_TOTAL_FRAMES:
        errors.append(f"info.json: total_frames={info.get('total_frames')}, expected {EXPECTED_TOTAL_FRAMES}")
    return errors


def check_session_mapping():
    """Validate episode_session_mapping.json."""
    errors = []
    mapping_path = DATASET_ROOT / "episode_session_mapping.json"
    if not mapping_path.exists():
        return ["episode_session_mapping.json missing"]

    with open(mapping_path) as f:
        data = json.load(f)

    n_episodes = data.get("n_episodes", 0)
    n_sessions = data.get("n_sessions", 0)
    augment_factor = data.get("augment_factor", 0)

    if n_episodes != NUM_EPISODES:
        errors.append(f"session_mapping: n_episodes={n_episodes}, expected {NUM_EPISODES}")
    if n_sessions != 50:
        errors.append(f"session_mapping: n_sessions={n_sessions}, expected 50")
    if augment_factor != 5:
        errors.append(f"session_mapping: augment_factor={augment_factor}, expected 5")

    mapping = data.get("episode_mapping", [])
    if len(mapping) != NUM_EPISODES:
        errors.append(f"session_mapping: {len(mapping)} entries, expected {NUM_EPISODES}")

    # Check each session has exactly 5 augments
    session_to_eps = data.get("session_to_episodes", {})
    for session, eps in session_to_eps.items():
        if len(eps) != 5:
            errors.append(f"session_mapping: {session} has {len(eps)} episodes, expected 5")

    if len(session_to_eps) != 50:
        errors.append(f"session_mapping: {len(session_to_eps)} sessions, expected 50")

    return errors


def plot_episode_lengths(ep_lengths):
    """Plot episode length histogram and lengths by session."""
    lengths = [ep_lengths[i] for i in range(NUM_EPISODES)]

    # Histogram
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Episode Length (frames)")
    ax.set_ylabel("Count")
    ax.set_title(f"Episode Length Distribution (n={NUM_EPISODES}, total={sum(lengths)})")
    ax.axvline(np.mean(lengths), color="red", linestyle="--", label=f"Mean={np.mean(lengths):.1f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_episode_lengths_histogram.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 01_episode_lengths_histogram.png")

    # Lengths by session, colored by augment version
    mapping_path = DATASET_ROOT / "episode_session_mapping.json"
    with open(mapping_path) as f:
        mapping_data = json.load(f)

    session_to_eps = mapping_data["session_to_episodes"]
    augment_versions = mapping_data["augment_versions"]
    ep_mapping = {e["episode_index"]: e for e in mapping_data["episode_mapping"]}

    fig, ax = plt.subplots(figsize=(16, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(augment_versions)))
    version_to_color = {v: colors[i] for i, v in enumerate(augment_versions)}

    session_names = sorted(session_to_eps.keys())
    for sess_idx, sess_name in enumerate(session_names):
        ep_indices = session_to_eps[sess_name]
        for ep_idx in ep_indices:
            version = ep_mapping[ep_idx]["version"]
            length = ep_lengths[ep_idx]
            ax.bar(
                sess_idx + (augment_versions.index(version) - 2) * 0.15,
                length,
                width=0.14,
                color=version_to_color[version],
                edgecolor="black",
                linewidth=0.3,
            )

    # Legend
    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=version_to_color[v], label=f"aug v{v}") for v in augment_versions]
    ax.legend(handles=legend_patches, loc="upper right", fontsize=8)
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Lengths by Session (colored by augment version)")
    ax.set_xticks(range(0, len(session_names), 5))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_episode_lengths_by_session.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 01_episode_lengths_by_session.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_errors = []

    print("=" * 60)
    print("Script 1: Data Integrity Check")
    print("=" * 60)

    # 1. Check info.json
    print("\n[1/5] Checking meta/info.json ...")
    errs = check_info_json()
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL: ' + '; '.join(errs)}")

    # 2. Load expected episode lengths
    print("\n[2/5] Loading episode lengths from episodes.jsonl ...")
    ep_lengths = load_episode_lengths()
    total_frames = sum(ep_lengths.values())
    print(f"  {len(ep_lengths)} episodes, {total_frames} total frames")
    if total_frames != EXPECTED_TOTAL_FRAMES:
        msg = f"Total frames {total_frames} != expected {EXPECTED_TOTAL_FRAMES}"
        all_errors.append(msg)
        print(f"  FAIL: {msg}")
    else:
        print(f"  PASS: total frames match")

    # 3. Check all parquet files
    print(f"\n[3/5] Checking {NUM_EPISODES} parquet files ...")
    pq_errors = []
    for i in range(NUM_EPISODES):
        errs = check_parquet(i, ep_lengths.get(i, -1))
        pq_errors.extend(errs)
    if pq_errors:
        print(f"  FAIL: {len(pq_errors)} errors")
        for e in pq_errors[:10]:
            print(f"    - {e}")
        if len(pq_errors) > 10:
            print(f"    ... and {len(pq_errors) - 10} more")
    else:
        print(f"  PASS: all {NUM_EPISODES} parquet files valid")
    all_errors.extend(pq_errors)

    # 4. Check all video files
    print(f"\n[4/5] Checking {NUM_EPISODES} video files ...")
    vid_errors = []
    for i in range(NUM_EPISODES):
        errs = check_video(i, ep_lengths.get(i, -1))
        vid_errors.extend(errs)
        if (i + 1) % 50 == 0:
            print(f"    checked {i + 1}/{NUM_EPISODES} ...")
    if vid_errors:
        print(f"  FAIL: {len(vid_errors)} errors")
        for e in vid_errors[:10]:
            print(f"    - {e}")
        if len(vid_errors) > 10:
            print(f"    ... and {len(vid_errors) - 10} more")
    else:
        print(f"  PASS: all {NUM_EPISODES} video files valid")
    all_errors.extend(vid_errors)

    # 5. Check session mapping
    print("\n[5/5] Checking episode_session_mapping.json ...")
    errs = check_session_mapping()
    all_errors.extend(errs)
    print(f"  {'PASS' if not errs else 'FAIL: ' + '; '.join(errs)}")

    # Visualizations
    print("\nGenerating visualizations ...")
    plot_episode_lengths(ep_lengths)

    # Summary
    print("\n" + "=" * 60)
    if all_errors:
        print(f"FAIL: {len(all_errors)} total errors found")
        for e in all_errors:
            print(f"  - {e}")
    else:
        print("PASS: All integrity checks passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
