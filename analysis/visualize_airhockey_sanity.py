"""
Sanity-check visualization for the FastUMI airhockey-strike dataset.

Produces three groups of figures:
  1_action_state_dist.png    Per-dim histograms of action and state, + norm distributions
  2_eef_trajectories.png     Per-episode EEF xyz trajectories (3D + xy/xz/yz projections)
  3_video_frames_<EPID>.png  Grid of evenly sampled wrist-camera frames for sample episodes

Usage:
  python analysis/visualize_airhockey_sanity.py \
      --data_dir playground/Datasets/FastUMI/1airhockey_strike_combined \
      --output_dir analysis/outputs-airhockey
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

plt.rcParams.update({"font.size": 10})


def load_all_episodes(data_dir: Path):
    episodes = {}
    for pq in sorted((data_dir / "data").rglob("episode_*.parquet")):
        df = pd.read_parquet(pq)
        ep_idx = int(df["episode_index"].iloc[0])
        actions = np.stack(df["action"].values)
        states = np.stack(df["observation.state"].values)
        episodes[ep_idx] = {
            "action": actions,
            "state": states,
            "n_frames": len(df),
            "task_index": int(df["task_index"].iloc[0]) if "task_index" in df.columns else -1,
        }
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def load_tasks(data_dir: Path):
    tasks = {}
    tj = data_dir / "meta" / "tasks.jsonl"
    if tj.exists():
        for line in tj.read_text().splitlines():
            obj = json.loads(line)
            tasks[obj["task_index"]] = obj["task"]
    return tasks


# ── Figure 1: action / state distributions ──────────────────────────────────
def plot_action_state_dist(episodes, output_dir: Path):
    all_actions = np.concatenate([ep["action"] for ep in episodes.values()], axis=0)
    all_states = np.concatenate([ep["state"] for ep in episodes.values()], axis=0)

    dim_labels = [
        "pos_x", "pos_y", "pos_z",
        "rot6d_0", "rot6d_1", "rot6d_2", "rot6d_3", "rot6d_4", "rot6d_5",
        "gripper",
    ]

    fig, axes = plt.subplots(4, 5, figsize=(20, 14))

    # Row 0+1: action histograms (10 dims)
    for i in range(10):
        ax = axes[i // 5, i % 5]
        vals = all_actions[:, i]
        ax.hist(vals, bins=80, color="steelblue", alpha=0.85)
        ax.set_title(f"action[{i}] = {dim_labels[i]}", fontsize=10)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.text(
            0.02, 0.95,
            f"min={vals.min():.3f}\nmax={vals.max():.3f}\nμ={vals.mean():.3f}\nσ={vals.std():.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7),
        )

    # Row 2+3: state histograms (10 dims)
    for i in range(10):
        ax = axes[2 + i // 5, i % 5]
        vals = all_states[:, i]
        ax.hist(vals, bins=80, color="coral", alpha=0.85)
        ax.set_title(f"state[{i}] = {dim_labels[i]}", fontsize=10)
        ax.axvline(0, color="k", lw=0.5, alpha=0.5)
        ax.text(
            0.02, 0.95,
            f"min={vals.min():.3f}\nmax={vals.max():.3f}\nμ={vals.mean():.3f}\nσ={vals.std():.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7),
        )

    fig.suptitle(
        f"Action (top, relative) & State (bottom, absolute) — {sum(ep['n_frames'] for ep in episodes.values())} frames across {len(episodes)} episodes",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / "1_action_state_dist.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")

    # Companion: per-episode norm time series + episode-length histogram
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    # Action position norm time series (overlay)
    ax = axes[0, 0]
    for ep in episodes.values():
        pn = np.linalg.norm(ep["action"][:, 0:3], axis=1)
        ax.plot(pn, color="steelblue", alpha=0.15, lw=0.7)
    ax.set_title("|action_pos| per frame, all episodes overlaid")
    ax.set_xlabel("frame")
    ax.set_ylabel("‖Δxyz‖")

    # Action gripper time series
    ax = axes[0, 1]
    for ep in episodes.values():
        ax.plot(ep["action"][:, 9], color="green", alpha=0.15, lw=0.7)
    ax.set_title("action[gripper] per frame, all episodes overlaid")
    ax.set_xlabel("frame")

    # Episode length histogram
    ax = axes[1, 0]
    lens = [ep["n_frames"] for ep in episodes.values()]
    ax.hist(lens, bins=30, color="purple", alpha=0.8)
    ax.set_title(f"Episode lengths (n={len(lens)}, min={min(lens)}, max={max(lens)}, μ={np.mean(lens):.1f})")
    ax.set_xlabel("frames")

    # Task balance
    ax = axes[1, 1]
    tasks_count = {}
    for ep in episodes.values():
        tasks_count[ep["task_index"]] = tasks_count.get(ep["task_index"], 0) + 1
    ax.bar([str(k) for k in sorted(tasks_count.keys())], [tasks_count[k] for k in sorted(tasks_count.keys())], color="teal")
    ax.set_title("Episodes per task_index")
    ax.set_xlabel("task_index")
    ax.set_ylabel("count")

    fig.tight_layout()
    out = output_dir / "1b_timeseries_and_lengths.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 2: EEF trajectories ──────────────────────────────────────────────
def plot_eef_trajectories(episodes, output_dir: Path):
    """
    State[0:3] is absolute EEF xyz. Plot each episode as a polyline,
    with a marker at the start (green) and end (red).
    """
    fig = plt.figure(figsize=(18, 12))
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xz = fig.add_subplot(2, 2, 3)
    ax_yz = fig.add_subplot(2, 2, 4)

    cmap = plt.cm.viridis
    n = len(episodes)
    for i, (ep_id, ep) in enumerate(sorted(episodes.items())):
        xyz = ep["state"][:, 0:3]
        color = cmap(i / max(1, n - 1))
        ax3d.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, alpha=0.5, lw=0.8)
        ax_xy.plot(xyz[:, 0], xyz[:, 1], color=color, alpha=0.5, lw=0.8)
        ax_xz.plot(xyz[:, 0], xyz[:, 2], color=color, alpha=0.5, lw=0.8)
        ax_yz.plot(xyz[:, 1], xyz[:, 2], color=color, alpha=0.5, lw=0.8)

        ax_xy.scatter(xyz[0, 0], xyz[0, 1], color="green", s=8, zorder=3)
        ax_xy.scatter(xyz[-1, 0], xyz[-1, 1], color="red", s=8, zorder=3)

    ax3d.set_title("EEF position 3D")
    ax3d.set_xlabel("x"); ax3d.set_ylabel("y"); ax3d.set_zlabel("z")
    ax_xy.set_title("xy projection (green=start, red=end)")
    ax_xy.set_xlabel("x"); ax_xy.set_ylabel("y"); ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xz.set_title("xz projection")
    ax_xz.set_xlabel("x"); ax_xz.set_ylabel("z")
    ax_yz.set_title("yz projection")
    ax_yz.set_xlabel("y"); ax_yz.set_ylabel("z")

    fig.suptitle(f"EEF trajectories across {n} episodes (color = episode order)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = output_dir / "2_eef_trajectories.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


# ── Figure 3: wrist video frame grid ────────────────────────────────────────
def sample_video_frames(video_path: Path, n_frames: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return [], 0
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, total


def plot_video_grid(data_dir: Path, episodes, output_dir: Path, n_episodes=8, n_frames_per_ep=8, tasks=None):
    ep_ids = sorted(episodes.keys())
    if n_episodes >= len(ep_ids):
        sample_ids = ep_ids
    else:
        sample_ids = [ep_ids[i] for i in np.linspace(0, len(ep_ids) - 1, n_episodes).astype(int)]

    fig, axes = plt.subplots(
        len(sample_ids), n_frames_per_ep,
        figsize=(n_frames_per_ep * 1.8, len(sample_ids) * 1.9),
    )
    if len(sample_ids) == 1:
        axes = axes[None, :]

    for r, ep_id in enumerate(sample_ids):
        vp = data_dir / "videos" / "chunk-000" / "observation.images.wrist" / f"episode_{ep_id:06d}.mp4"
        frames, total = sample_video_frames(vp, n_frames_per_ep)
        task_str = ""
        if tasks is not None:
            task_str = tasks.get(episodes[ep_id]["task_index"], "")
            if len(task_str) > 28:
                task_str = task_str[:26] + "…"
        for c in range(n_frames_per_ep):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(frames):
                ax.imshow(frames[c])
            if c == 0:
                ax.set_ylabel(f"ep {ep_id}\n({total}f)\n{task_str}", fontsize=7)

    fig.suptitle("Wrist camera — evenly sampled frames", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = output_dir / "3_video_frames_grid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="playground/Datasets/FastUMI/1airhockey_strike_combined")
    p.add_argument("--output_dir", default="analysis/outputs-airhockey")
    p.add_argument("--n_video_episodes", type=int, default=8)
    p.add_argument("--n_video_frames", type=int, default=8)
    args = p.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"data_dir   : {data_dir}")
    print(f"output_dir : {output_dir}")

    episodes = load_all_episodes(data_dir)
    tasks = load_tasks(data_dir)

    print("Plot 1: action/state distributions")
    plot_action_state_dist(episodes, output_dir)
    print("Plot 2: EEF trajectories")
    plot_eef_trajectories(episodes, output_dir)
    print("Plot 3: video frame grid")
    plot_video_grid(data_dir, episodes, output_dir, args.n_video_episodes, args.n_video_frames, tasks)
    print("Done")


if __name__ == "__main__":
    main()
