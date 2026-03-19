"""
Visualize FastUMI dataset to diagnose "staying still during grasp" issue.

Core question: Does the robot spend too many frames near-stationary while the gripper
is closed, creating an imbalanced action distribution that biases the model toward
predicting zero/still actions after grasping?

Produces 5 figures:
  1. Per-episode timeline: position action norm + gripper state
  2. Action norm distribution conditioned on gripper state (open vs closed)
  3. Time-aligned action patterns around gripper-close events
  4. Fraction of "still" frames (|action_pos| < threshold) per episode
  5. Action dimension breakdown around grasp events

Usage:
  python analysis/visualize_grasp_stillness.py \
      --data_dir playground/Datasets/FastUMI/pickandplace-ur5-0314-v2 \
      --output_dir analysis/outputs
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 11})


def load_all_episodes(data_dir: str):
    """Load all parquet episodes, return list of per-episode DataFrames."""
    data_path = Path(data_dir)
    chunk_dirs = sorted(data_path.glob("data/chunk-*"))

    episodes = {}
    for chunk_dir in chunk_dirs:
        for pq in sorted(chunk_dir.glob("episode_*.parquet")):
            df = pd.read_parquet(pq)
            ep_idx = int(df["episode_index"].iloc[0])

            actions = np.stack(df["action"].values)           # (T, 10)
            states = np.stack(df["observation.state"].values)  # (T, 10)
            timestamps = df["timestamp"].values                # (T,)

            episodes[ep_idx] = {
                "action_pos": actions[:, 0:3],      # relative xyz
                "action_rot": actions[:, 3:9],      # relative rot6d
                "action_gripper": actions[:, 9],     # gripper
                "state_pos": states[:, 0:3],         # absolute xyz
                "state_gripper": states[:, 9],       # gripper state
                "timestamps": timestamps,
                "n_frames": len(df),
            }

    print(f"Loaded {len(episodes)} episodes")
    return episodes


def compute_grasp_events(gripper_state: np.ndarray, threshold=0.5):
    """Find frames where gripper transitions from open to closed.

    Returns list of frame indices where the gripper closes.
    """
    closed = (gripper_state > threshold).astype(int)
    diff = np.diff(closed)
    # +1 = open->closed transition
    close_indices = np.where(diff == 1)[0] + 1
    # -1 = closed->open transition
    open_indices = np.where(diff == -1)[0] + 1
    return close_indices, open_indices


# ── Figure 1: Sample episode timelines ──────────────────────────────────────
def plot_episode_timelines(episodes, output_dir, n_samples=6):
    """Plot action norm + gripper for sample episodes."""
    ep_ids = sorted(episodes.keys())
    sample_ids = ep_ids[:: max(1, len(ep_ids) // n_samples)][:n_samples]

    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 3 * n_samples), sharex=False)
    if n_samples == 1:
        axes = [axes]

    for ax, ep_id in zip(axes, sample_ids):
        ep = episodes[ep_id]
        pos_norm = np.linalg.norm(ep["action_pos"], axis=1)
        t = np.arange(len(pos_norm)) / 20.0  # 20 Hz

        ax.plot(t, pos_norm, color="steelblue", linewidth=1.0, label="|action_pos|")
        ax.set_ylabel("pos norm (m)")
        ax.set_title(f"Episode {ep_id}  ({ep['n_frames']} frames)")

        # Shade regions where gripper is closed
        gripper = ep["state_gripper"]
        closed = gripper > 0.5
        ax_twin = ax.twinx()
        ax_twin.fill_between(t, 0, closed.astype(float), alpha=0.2, color="red", label="gripper closed")
        ax_twin.set_ylim(0, 1.5)
        ax_twin.set_ylabel("gripper")

        # Mark grasp events
        close_idx, open_idx = compute_grasp_events(gripper)
        for ci in close_idx:
            ax.axvline(ci / 20.0, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
        for oi in open_idx:
            ax.axvline(oi / 20.0, color="green", linestyle="--", alpha=0.6, linewidth=0.8)

        ax.set_xlabel("time (s)")
        if ep_id == sample_ids[0]:
            ax.legend(loc="upper left")
            ax_twin.legend(loc="upper right")

    fig.suptitle("Episode Timelines: Action Position Norm + Gripper State", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "1_episode_timelines.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 1_episode_timelines.png")


# ── Figure 2: Action norm distribution by gripper state ─────────────────────
def plot_action_distribution_by_gripper(episodes, output_dir, still_threshold=0.001):
    """Histogram of position action norm, conditioned on gripper open/closed."""
    all_pos_norms = []
    all_gripper_states = []

    for ep in episodes.values():
        pos_norm = np.linalg.norm(ep["action_pos"], axis=1)
        all_pos_norms.append(pos_norm)
        all_gripper_states.append(ep["state_gripper"])

    pos_norms = np.concatenate(all_pos_norms)
    grippers = np.concatenate(all_gripper_states)
    closed_mask = grippers > 0.5
    open_mask = ~closed_mask

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: overlapped histogram
    bins = np.linspace(0, np.percentile(pos_norms, 99), 80)
    axes[0].hist(pos_norms[open_mask], bins=bins, alpha=0.6, color="steelblue",
                 label=f"gripper open (n={open_mask.sum()})", density=True)
    axes[0].hist(pos_norms[closed_mask], bins=bins, alpha=0.6, color="red",
                 label=f"gripper closed (n={closed_mask.sum()})", density=True)
    axes[0].axvline(still_threshold, color="black", linestyle="--", label=f"still threshold={still_threshold}")
    axes[0].set_xlabel("|action_pos| (m)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Action Norm Distribution")
    axes[0].legend()

    # Middle: fraction of still frames
    still_open = (pos_norms[open_mask] < still_threshold).mean() * 100
    still_closed = (pos_norms[closed_mask] < still_threshold).mean() * 100
    still_all = (pos_norms < still_threshold).mean() * 100
    bars = axes[1].bar(["Open", "Closed", "All"],
                       [still_open, still_closed, still_all],
                       color=["steelblue", "red", "gray"])
    for bar, val in zip(bars, [still_open, still_closed, still_all]):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", fontsize=12)
    axes[1].set_ylabel("% of frames")
    axes[1].set_title(f"Fraction of Still Frames (|pos| < {still_threshold}m)")

    # Right: box plot by gripper state
    axes[2].boxplot([pos_norms[open_mask], pos_norms[closed_mask]],
                    labels=["Open", "Closed"], showfliers=False)
    axes[2].set_ylabel("|action_pos| (m)")
    axes[2].set_title("Action Norm by Gripper State (no outliers)")

    fig.suptitle("Action Distribution Analysis by Gripper State", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "2_action_dist_by_gripper.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 2_action_dist_by_gripper.png")

    # Print stats
    print(f"\n{'='*60}")
    print(f"ACTION DISTRIBUTION STATS:")
    print(f"  Total frames: {len(pos_norms)}")
    print(f"  Gripper open frames:   {open_mask.sum()} ({open_mask.mean()*100:.1f}%)")
    print(f"  Gripper closed frames: {closed_mask.sum()} ({closed_mask.mean()*100:.1f}%)")
    print(f"  Still frames (|pos| < {still_threshold}m):")
    print(f"    When open:   {still_open:.1f}%")
    print(f"    When closed: {still_closed:.1f}%")
    print(f"    Overall:     {still_all:.1f}%")
    print(f"  Median action norm:")
    print(f"    When open:   {np.median(pos_norms[open_mask]):.6f} m")
    print(f"    When closed: {np.median(pos_norms[closed_mask]):.6f} m")
    print(f"{'='*60}\n")


# ── Figure 3: Action patterns around grasp events ──────────────────────────
def plot_grasp_aligned_actions(episodes, output_dir, window=40):
    """Time-align actions around gripper-close events and plot average pattern."""
    aligned_pos_norms = []
    aligned_xyz = []  # (N, 2*window, 3)
    aligned_grippers = []

    for ep in episodes.values():
        close_idx, _ = compute_grasp_events(ep["state_gripper"])
        for ci in close_idx:
            start = ci - window
            end = ci + window
            if start < 0 or end > len(ep["action_pos"]):
                continue

            pos = ep["action_pos"][start:end]  # (2*window, 3)
            pos_norm = np.linalg.norm(pos, axis=1)
            gripper = ep["state_gripper"][start:end]

            aligned_pos_norms.append(pos_norm)
            aligned_xyz.append(pos)
            aligned_grippers.append(gripper)

    if not aligned_pos_norms:
        print("No grasp events found with sufficient context window!")
        return

    aligned_pos_norms = np.array(aligned_pos_norms)   # (N, 2*window)
    aligned_xyz = np.array(aligned_xyz)                # (N, 2*window, 3)
    aligned_grippers = np.array(aligned_grippers)      # (N, 2*window)

    t = (np.arange(2 * window) - window) / 20.0  # seconds relative to grasp

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig)

    # Top-left: mean + std of position norm
    ax1 = fig.add_subplot(gs[0, 0])
    mean_norm = aligned_pos_norms.mean(axis=0)
    std_norm = aligned_pos_norms.std(axis=0)
    ax1.plot(t, mean_norm, color="steelblue", linewidth=2)
    ax1.fill_between(t, mean_norm - std_norm, mean_norm + std_norm, alpha=0.2, color="steelblue")
    ax1.axvline(0, color="red", linestyle="--", label="grasp event")
    ax1.set_xlabel("time relative to grasp (s)")
    ax1.set_ylabel("|action_pos| (m)")
    ax1.set_title(f"Action Norm Around Grasp (n={len(aligned_pos_norms)} events)")
    ax1.legend()

    # Top-right: mean gripper state
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, aligned_grippers.mean(axis=0), color="red", linewidth=2)
    ax2.fill_between(t, aligned_grippers.mean(axis=0) - aligned_grippers.std(axis=0),
                     aligned_grippers.mean(axis=0) + aligned_grippers.std(axis=0),
                     alpha=0.2, color="red")
    ax2.axvline(0, color="red", linestyle="--")
    ax2.set_xlabel("time relative to grasp (s)")
    ax2.set_ylabel("gripper state")
    ax2.set_title("Gripper State Around Grasp")

    # Bottom-left: per-axis action components
    ax3 = fig.add_subplot(gs[1, 0])
    labels = ["x (forward)", "y (lateral)", "z (vertical)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for dim in range(3):
        mean_d = aligned_xyz[:, :, dim].mean(axis=0)
        std_d = aligned_xyz[:, :, dim].std(axis=0)
        ax3.plot(t, mean_d, color=colors[dim], linewidth=1.5, label=labels[dim])
        ax3.fill_between(t, mean_d - std_d, mean_d + std_d, alpha=0.12, color=colors[dim])
    ax3.axvline(0, color="red", linestyle="--")
    ax3.set_xlabel("time relative to grasp (s)")
    ax3.set_ylabel("action component (m)")
    ax3.set_title("Per-Axis Actions Around Grasp")
    ax3.legend()

    # Bottom-right: heatmap of individual events
    ax4 = fig.add_subplot(gs[1, 1])
    # Sort by post-grasp activity (events with least motion on top)
    post_grasp_activity = aligned_pos_norms[:, window:].mean(axis=1)
    sort_idx = np.argsort(post_grasp_activity)
    im = ax4.imshow(aligned_pos_norms[sort_idx], aspect="auto",
                    extent=[t[0], t[-1], len(aligned_pos_norms), 0],
                    cmap="hot_r", vmin=0, vmax=np.percentile(aligned_pos_norms, 95))
    ax4.axvline(0, color="cyan", linestyle="--", linewidth=1.5)
    ax4.set_xlabel("time relative to grasp (s)")
    ax4.set_ylabel("event (sorted by post-grasp stillness)")
    ax4.set_title("Per-Event Action Norm Heatmap")
    plt.colorbar(im, ax=ax4, label="|action_pos| (m)")

    fig.suptitle("Action Patterns Aligned to Grasp Events", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "3_grasp_aligned_actions.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 3_grasp_aligned_actions.png")

    # Print grasp-event statistics
    pre_grasp_norms = aligned_pos_norms[:, :window].mean(axis=1)
    post_grasp_norms = aligned_pos_norms[:, window:].mean(axis=1)
    print(f"\nGRASP EVENT STATS ({len(aligned_pos_norms)} events):")
    print(f"  Mean action norm BEFORE grasp: {pre_grasp_norms.mean():.6f} m")
    print(f"  Mean action norm AFTER grasp:  {post_grasp_norms.mean():.6f} m")
    print(f"  Ratio (after/before):          {post_grasp_norms.mean() / (pre_grasp_norms.mean() + 1e-9):.2f}x")

    # Count how many frames post-grasp are near-still
    still_threshold = 0.001
    post_still_frac = (aligned_pos_norms[:, window:] < still_threshold).mean() * 100
    pre_still_frac = (aligned_pos_norms[:, :window] < still_threshold).mean() * 100
    print(f"  Still frames (|pos|<{still_threshold}m) before grasp: {pre_still_frac:.1f}%")
    print(f"  Still frames (|pos|<{still_threshold}m) after grasp:  {post_still_frac:.1f}%")


# ── Figure 4: Per-episode still-frame fractions ─────────────────────────────
def plot_per_episode_stillness(episodes, output_dir, still_threshold=0.001):
    """Bar chart of still-frame fraction per episode."""
    ep_ids = sorted(episodes.keys())
    still_fracs = []
    closed_fracs = []
    still_while_closed_fracs = []

    for ep_id in ep_ids:
        ep = episodes[ep_id]
        pos_norm = np.linalg.norm(ep["action_pos"], axis=1)
        still = pos_norm < still_threshold
        closed = ep["state_gripper"] > 0.5

        still_fracs.append(still.mean() * 100)
        closed_fracs.append(closed.mean() * 100)
        if closed.sum() > 0:
            still_while_closed_fracs.append((still & closed).sum() / closed.sum() * 100)
        else:
            still_while_closed_fracs.append(0)

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

    axes[0].bar(ep_ids, still_fracs, color="gray", alpha=0.7)
    axes[0].set_ylabel("% still frames")
    axes[0].set_title(f"Still Frames per Episode (|pos| < {still_threshold}m)")
    axes[0].axhline(np.mean(still_fracs), color="red", linestyle="--",
                    label=f"mean={np.mean(still_fracs):.1f}%")
    axes[0].legend()

    axes[1].bar(ep_ids, closed_fracs, color="red", alpha=0.5)
    axes[1].set_ylabel("% gripper closed")
    axes[1].set_title("Gripper Closed Fraction per Episode")
    axes[1].axhline(np.mean(closed_fracs), color="red", linestyle="--",
                    label=f"mean={np.mean(closed_fracs):.1f}%")
    axes[1].legend()

    axes[2].bar(ep_ids, still_while_closed_fracs, color="darkred", alpha=0.6)
    axes[2].set_ylabel("% still | closed")
    axes[2].set_title("Still Frames While Gripper Closed (per Episode)")
    axes[2].set_xlabel("episode index")
    axes[2].axhline(np.mean(still_while_closed_fracs), color="red", linestyle="--",
                    label=f"mean={np.mean(still_while_closed_fracs):.1f}%")
    axes[2].legend()

    fig.suptitle("Per-Episode Stillness Analysis", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "4_per_episode_stillness.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 4_per_episode_stillness.png")


# ── Figure 5: Duration of still segments ────────────────────────────────────
def plot_still_segment_durations(episodes, output_dir, still_threshold=0.001):
    """Analyze how long consecutive still segments last, by gripper state."""
    still_durations_open = []
    still_durations_closed = []

    for ep in episodes.values():
        pos_norm = np.linalg.norm(ep["action_pos"], axis=1)
        still = pos_norm < still_threshold
        closed = ep["state_gripper"] > 0.5

        # Find consecutive still segments
        in_segment = False
        seg_start = 0
        for i in range(len(still)):
            if still[i] and not in_segment:
                in_segment = True
                seg_start = i
            elif not still[i] and in_segment:
                in_segment = False
                duration = i - seg_start
                # Classify by dominant gripper state in segment
                seg_closed = closed[seg_start:i].mean() > 0.5
                if seg_closed:
                    still_durations_closed.append(duration)
                else:
                    still_durations_open.append(duration)
        if in_segment:
            duration = len(still) - seg_start
            seg_closed = closed[seg_start:].mean() > 0.5
            if seg_closed:
                still_durations_closed.append(duration)
            else:
                still_durations_open.append(duration)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if still_durations_open:
        axes[0].hist(np.array(still_durations_open) / 20.0, bins=30, color="steelblue", alpha=0.7)
        axes[0].set_title(f"Still Segment Duration (gripper OPEN, n={len(still_durations_open)})")
        axes[0].set_xlabel("duration (s)")
        axes[0].set_ylabel("count")
        mean_dur = np.mean(still_durations_open) / 20.0
        axes[0].axvline(mean_dur, color="red", linestyle="--", label=f"mean={mean_dur:.2f}s")
        axes[0].legend()

    if still_durations_closed:
        axes[1].hist(np.array(still_durations_closed) / 20.0, bins=30, color="red", alpha=0.7)
        axes[1].set_title(f"Still Segment Duration (gripper CLOSED, n={len(still_durations_closed)})")
        axes[1].set_xlabel("duration (s)")
        axes[1].set_ylabel("count")
        mean_dur = np.mean(still_durations_closed) / 20.0
        axes[1].axvline(mean_dur, color="darkred", linestyle="--", label=f"mean={mean_dur:.2f}s")
        axes[1].legend()

    fig.suptitle(f"Consecutive Still Segment Durations (threshold={still_threshold}m)", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "5_still_segment_durations.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved 5_still_segment_durations.png")

    print(f"\nSTILL SEGMENT DURATION STATS:")
    if still_durations_open:
        d = np.array(still_durations_open) / 20.0
        print(f"  Gripper OPEN:   n={len(d)}, mean={d.mean():.2f}s, max={d.max():.2f}s, total={d.sum():.1f}s")
    if still_durations_closed:
        d = np.array(still_durations_closed) / 20.0
        print(f"  Gripper CLOSED: n={len(d)}, mean={d.mean():.2f}s, max={d.max():.2f}s, total={d.sum():.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Visualize FastUMI dataset grasp/stillness patterns")
    parser.add_argument("--data_dir", type=str,
                        default="playground/Datasets/FastUMI/pickandplace-ur5-0314-v2",
                        help="Path to LeRobot v2.1 dataset directory")
    parser.add_argument("--output_dir", type=str, default="analysis/outputs",
                        help="Directory to save figures")
    parser.add_argument("--still_threshold", type=float, default=0.001,
                        help="Position action norm threshold to consider 'still' (meters)")
    parser.add_argument("--grasp_window", type=int, default=40,
                        help="Number of frames before/after grasp event to analyze (at 20Hz)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    episodes = load_all_episodes(args.data_dir)

    plot_episode_timelines(episodes, args.output_dir)
    plot_action_distribution_by_gripper(episodes, args.output_dir, still_threshold=args.still_threshold)
    plot_grasp_aligned_actions(episodes, args.output_dir, window=args.grasp_window)
    plot_per_episode_stillness(episodes, args.output_dir, still_threshold=args.still_threshold)
    plot_still_segment_durations(episodes, args.output_dir, still_threshold=args.still_threshold)

    print(f"\nAll figures saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
