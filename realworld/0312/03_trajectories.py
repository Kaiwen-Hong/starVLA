#!/usr/bin/env python3
"""Script 3: Trajectory, Gripper, and Video-Action Alignment

Plots 3D EEF trajectories, action timeseries for sample episodes,
gripper heatmap across all 250 episodes, gripper transition histogram,
and video-action alignment for episode 0.
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
from mpl_toolkits.mplot3d import Axes3D

NUM_EPISODES = 250
SAMPLE_EPISODES = [0, 50, 100, 150, 200]
ACTION_DIM_NAMES = ["dx", "dy", "dz", "R00", "R01", "R02", "R10", "R11", "R12", "grip"]


def load_episode(ep_idx):
    """Load a single episode's state and action arrays."""
    pq_path = DATASET_ROOT / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq_path)
    states = np.stack(df["observation.state"].values)
    actions = np.stack(df["action"].values)
    return states, actions


def plot_3d_trajectory(ep_idx):
    """Plot 3D EEF position trajectory for a single episode."""
    states, _ = load_episode(ep_idx)
    pos = states[:, 0:3]

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Color by time
    colors = plt.cm.viridis(np.linspace(0, 1, len(pos)))
    for i in range(len(pos) - 1):
        ax.plot(pos[i:i+2, 0], pos[i:i+2, 1], pos[i:i+2, 2], color=colors[i], linewidth=1.5)

    ax.scatter(*pos[0], color="green", s=80, zorder=5, label="Start", marker="o")
    ax.scatter(*pos[-1], color="red", s=80, zorder=5, label="End", marker="x")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Episode {ep_idx}: 3D EEF Trajectory ({len(pos)} frames)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"03_trajectory_3d_ep{ep_idx}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_trajectory_3d_ep{ep_idx}.png")


def plot_action_timeseries(ep_idx):
    """Plot all 10 action dims over time for a single episode."""
    _, actions = load_episode(ep_idx)

    fig, axes = plt.subplots(5, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()
    t = np.arange(len(actions))

    for d in range(10):
        ax = axes[d]
        ax.plot(t, actions[:, d], linewidth=0.8)
        ax.set_ylabel(ACTION_DIM_NAMES[d], fontsize=9)
        ax.grid(alpha=0.3)
        if d >= 8:
            ax.set_xlabel("Frame")

    fig.suptitle(f"Episode {ep_idx}: Action Timeseries ({len(actions)} frames)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"03_action_timeseries_ep{ep_idx}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_action_timeseries_ep{ep_idx}.png")


def plot_gripper_heatmap():
    """Plot heatmap of gripper state across all 250 episodes."""
    # Load all episode lengths first to find max
    ep_lengths_path = DATASET_ROOT / "meta/episodes.jsonl"
    ep_lengths = {}
    with open(ep_lengths_path) as f:
        for line in f:
            rec = json.loads(line)
            ep_lengths[rec["episode_index"]] = rec["length"]

    max_len = max(ep_lengths.values())
    gripper_matrix = np.full((NUM_EPISODES, max_len), np.nan)
    transition_times = []
    flagged_episodes = []

    for i in range(NUM_EPISODES):
        states, _ = load_episode(i)
        gripper = states[:, 9]
        gripper_matrix[i, :len(gripper)] = gripper

        # Count transitions (binary threshold at 0.5)
        binary_gripper = (gripper > 0.5).astype(int)
        transitions = np.where(np.diff(binary_gripper) != 0)[0]
        n_transitions = len(transitions)

        for t_idx in transitions:
            transition_times.append(t_idx / len(gripper))  # normalized time

        if n_transitions == 0 or n_transitions > 4:
            flagged_episodes.append((i, n_transitions))

    # Heatmap
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(gripper_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.6,
                   interpolation="nearest")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Episode Index")
    ax.set_title(f"Gripper State Heatmap (all {NUM_EPISODES} episodes)")
    fig.colorbar(im, ax=ax, label="Gripper value", shrink=0.8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_gripper_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_gripper_heatmap.png")

    # Report flagged episodes
    if flagged_episodes:
        print(f"\n  Flagged episodes (0 or >4 gripper transitions):")
        for ep, n in flagged_episodes[:20]:
            print(f"    ep{ep}: {n} transitions")
        if len(flagged_episodes) > 20:
            print(f"    ... and {len(flagged_episodes) - 20} more")
    else:
        print(f"  All episodes have 1-4 gripper transitions")

    return transition_times


def plot_gripper_transitions(transition_times):
    """Plot histogram of gripper transition timing."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(transition_times, bins=40, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Normalized Time (0=start, 1=end)")
    ax.set_ylabel("Count")
    ax.set_title(f"Gripper Transition Timing ({len(transition_times)} total transitions)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_gripper_transitions.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_gripper_transitions.png")


def plot_video_action_alignment():
    """Extract 5 evenly-spaced video frames from episode 0, annotate with actions."""
    ep_idx = 0
    _, actions = load_episode(ep_idx)
    n_frames = len(actions)
    sample_indices = np.linspace(0, n_frames - 1, 5, dtype=int)

    vid_path = DATASET_ROOT / f"videos/chunk-000/observation.images.wrist/episode_{ep_idx:06d}.mp4"

    frames = []
    sample_set = set(sample_indices)
    try:
        import av
        container = av.open(str(vid_path))
        for fi, frame in enumerate(container.decode(video=0)):
            if fi in sample_set:
                frames.append((fi, frame.to_ndarray(format="rgb24")))
            if fi > max(sample_indices):
                break
        container.close()
        # Sort by original order and extract just the arrays
        frames.sort(key=lambda x: x[0])
        frames = [f[1] for f in frames]
    except ImportError:
        try:
            import cv2
            cap = cv2.VideoCapture(str(vid_path))
            for fi in range(max(sample_indices) + 1):
                ret, frame = cap.read()
                if not ret:
                    break
                if fi in sample_set:
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
        except Exception as e:
            print(f"  WARNING: Could not read video for alignment: {e}")
            return

    if not frames:
        print("  WARNING: No frames extracted for alignment visualization")
        return

    fig, axes = plt.subplots(2, 5, figsize=(20, 8),
                             gridspec_kw={"height_ratios": [3, 1]})

    for i, (frame_idx, frame) in enumerate(zip(sample_indices, frames)):
        # Image
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f"Frame {frame_idx}", fontsize=10)
        axes[0, i].axis("off")

        # Action annotation
        ax = axes[1, i]
        act = actions[frame_idx]
        labels = ACTION_DIM_NAMES
        y_pos = np.arange(len(labels))
        colors = ["steelblue" if abs(v) < 0.01 else "coral" for v in act]
        ax.barh(y_pos, act, color=colors, height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("Value", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-0.02, 1.1)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(f"Episode {ep_idx}: Video-Action Alignment", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "03_video_action_alignment_ep0.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 03_video_action_alignment_ep0.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Script 3: Trajectories, Gripper & Video-Action Alignment")
    print("=" * 60)

    # 3D trajectories for sample episodes
    print("\n[1/4] Plotting 3D trajectories for sample episodes ...")
    for ep in SAMPLE_EPISODES:
        plot_3d_trajectory(ep)

    # Action timeseries for sample episodes
    print("\n[2/4] Plotting action timeseries for sample episodes ...")
    for ep in SAMPLE_EPISODES:
        plot_action_timeseries(ep)

    # Gripper heatmap (all episodes)
    print("\n[3/4] Computing gripper heatmap (all 250 episodes) ...")
    transition_times = plot_gripper_heatmap()
    plot_gripper_transitions(transition_times)

    # Video-action alignment
    print("\n[4/4] Plotting video-action alignment (episode 0) ...")
    plot_video_action_alignment()

    print("\n" + "=" * 60)
    print("PASS: Trajectory analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
