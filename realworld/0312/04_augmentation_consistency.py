#!/usr/bin/env python3
"""Script 4: Cross-Augmentation Validation

For 5 sample sessions (each with 5 augmented episodes), compares episode
lengths, overlays 3D trajectories, position-over-time, and gripper timeseries
to verify augmentation consistency.
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


def load_session_mapping():
    """Load episode-session mapping."""
    with open(DATASET_ROOT / "episode_session_mapping.json") as f:
        data = json.load(f)
    return data


def load_episode(ep_idx):
    """Load a single episode's data."""
    pq_path = DATASET_ROOT / f"data/chunk-000/episode_{ep_idx:06d}.parquet"
    df = pd.read_parquet(pq_path)
    states = np.stack(df["observation.state"].values)
    actions = np.stack(df["action"].values)
    return states, actions


def get_sample_sessions(mapping_data, n=5):
    """Get n evenly-spaced sessions from the mapping."""
    session_names = sorted(mapping_data["session_to_episodes"].keys())
    indices = np.linspace(0, len(session_names) - 1, n, dtype=int)
    return [(session_names[i], mapping_data["session_to_episodes"][session_names[i]]) for i in indices]


def plot_augment_trajectories(session_name, ep_indices, ep_mapping):
    """Plot 5 overlaid 3D trajectories for one session's augments."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = plt.cm.tab10(np.linspace(0, 1, len(ep_indices)))
    for i, ep_idx in enumerate(ep_indices):
        states, _ = load_episode(ep_idx)
        pos = states[:, 0:3]
        version = ep_mapping[ep_idx]["version"]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=colors[i],
                linewidth=1.2, alpha=0.8, label=f"ep{ep_idx} (v{version})")
        ax.scatter(*pos[0], color=colors[i], s=40, marker="o")
        ax.scatter(*pos[-1], color=colors[i], s=40, marker="x")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(f"Session {session_name}: 5 Augmented Trajectories")
    ax.legend(fontsize=8)
    fig.tight_layout()

    # Use session index for filename
    sess_short = session_name.replace("session_", "")
    fig.savefig(OUTPUT_DIR / f"04_augment_trajectories_{sess_short}.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 04_augment_trajectories_{sess_short}.png")


def plot_position_overlay(session_name, ep_indices, ep_mapping):
    """Plot x/y/z position over time overlaid for 5 augments of one session."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    dim_labels = ["X", "Y", "Z"]
    colors = plt.cm.tab10(np.linspace(0, 1, len(ep_indices)))

    for i, ep_idx in enumerate(ep_indices):
        states, _ = load_episode(ep_idx)
        pos = states[:, 0:3]
        version = ep_mapping[ep_idx]["version"]
        t = np.arange(len(pos))
        for d in range(3):
            axes[d].plot(t, pos[:, d], color=colors[i], linewidth=0.8,
                         alpha=0.8, label=f"v{version}" if d == 0 else None)

    for d in range(3):
        axes[d].set_ylabel(dim_labels[d])
        axes[d].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=5)
    axes[2].set_xlabel("Frame")
    fig.suptitle(f"Session {session_name}: Position Overlay", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_augment_position_overlay.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 04_augment_position_overlay.png")


def plot_length_comparison(mapping_data, ep_lengths):
    """Bar chart of lengths by session."""
    session_to_eps = mapping_data["session_to_episodes"]
    session_names = sorted(session_to_eps.keys())
    augment_versions = mapping_data["augment_versions"]
    ep_mapping = {e["episode_index"]: e for e in mapping_data["episode_mapping"]}

    fig, ax = plt.subplots(figsize=(18, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, len(augment_versions)))
    width = 0.15

    for sess_idx, sess_name in enumerate(session_names):
        ep_indices = session_to_eps[sess_name]
        for j, ep_idx in enumerate(ep_indices):
            version = ep_mapping[ep_idx]["version"]
            v_idx = augment_versions.index(version)
            length = ep_lengths.get(ep_idx, 0)
            ax.bar(sess_idx + (v_idx - 2) * width, length, width=width * 0.9,
                   color=colors[v_idx], edgecolor="black", linewidth=0.2)

    from matplotlib.patches import Patch
    legend_patches = [Patch(facecolor=colors[i], label=f"v{v}") for i, v in enumerate(augment_versions)]
    ax.legend(handles=legend_patches, fontsize=8)
    ax.set_xlabel("Session Index")
    ax.set_ylabel("Episode Length")
    ax.set_title("Episode Lengths by Session and Augment Version")
    ax.set_xticks(range(0, len(session_names), 5))
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_augment_length_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 04_augment_length_comparison.png")


def plot_gripper_comparison(sample_sessions, ep_mapping):
    """Plot gripper timeseries overlaid for selected sessions."""
    n_sessions = len(sample_sessions)
    fig, axes = plt.subplots(n_sessions, 1, figsize=(14, 3 * n_sessions), sharex=False)
    if n_sessions == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for s_idx, (sess_name, ep_indices) in enumerate(sample_sessions):
        ax = axes[s_idx]
        for i, ep_idx in enumerate(ep_indices):
            states, _ = load_episode(ep_idx)
            gripper = states[:, 9]
            version = ep_mapping[ep_idx]["version"]
            t = np.arange(len(gripper))
            ax.plot(t, gripper, color=colors[i], linewidth=1.0, alpha=0.8,
                    label=f"ep{ep_idx} (v{version})")
        ax.set_ylabel("Gripper")
        ax.set_title(f"Session {sess_name}", fontsize=10)
        ax.legend(fontsize=7, ncol=5)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.05, 0.6)

    axes[-1].set_xlabel("Frame")
    fig.suptitle("Gripper Overlay Across Augmentations", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_augment_gripper_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 04_augment_gripper_comparison.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Script 4: Augmentation Consistency")
    print("=" * 60)

    mapping_data = load_session_mapping()
    ep_mapping = {e["episode_index"]: e for e in mapping_data["episode_mapping"]}
    sample_sessions = get_sample_sessions(mapping_data, n=5)

    # Load episode lengths
    ep_lengths = {}
    with open(DATASET_ROOT / "meta/episodes.jsonl") as f:
        for line in f:
            rec = json.loads(line)
            ep_lengths[rec["episode_index"]] = rec["length"]

    # Check length consistency within sessions
    print("\n[1/4] Checking episode length consistency within sessions ...")
    issues = []
    session_to_eps = mapping_data["session_to_episodes"]
    for sess_name, ep_indices in sorted(session_to_eps.items()):
        lengths = [ep_lengths[ep] for ep in ep_indices]
        spread = max(lengths) - min(lengths)
        if spread > 1:
            issues.append(f"  {sess_name}: lengths={lengths}, spread={spread}")
    if issues:
        print(f"  {len(issues)} sessions have length spread > 1:")
        for iss in issues[:10]:
            print(f"    {iss}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    else:
        print(f"  PASS: all sessions have length spread <= 1")

    # Plot augmented trajectories
    print("\n[2/4] Plotting augmented 3D trajectories for sample sessions ...")
    for sess_name, ep_indices in sample_sessions:
        plot_augment_trajectories(sess_name, ep_indices, ep_mapping)

    # Position overlay (first sample session)
    print("\n[3/4] Plotting position overlay for first sample session ...")
    sess_name, ep_indices = sample_sessions[0]
    plot_position_overlay(sess_name, ep_indices, ep_mapping)

    # Length comparison
    print("\n[4/4] Plotting length comparison and gripper comparison ...")
    plot_length_comparison(mapping_data, ep_lengths)
    plot_gripper_comparison(sample_sessions, ep_mapping)

    print("\n" + "=" * 60)
    print("PASS: Augmentation consistency analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
