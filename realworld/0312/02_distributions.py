#!/usr/bin/env python3
"""Script 2: Action/State Statistics & Rotation Validity

Loads all 250 parquets, computes per-dimension statistics,
cross-checks with stats_gr00t.json, and validates rotation
representations (rot6d unit-length + orthogonality for state,
near-identity check for action).
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
import matplotlib.gridspec as gridspec

NUM_EPISODES = 250
STATE_DIM_NAMES = ["x", "y", "z", "r1_x", "r1_y", "r1_z", "r2_x", "r2_y", "r2_z", "gripper"]
ACTION_DIM_NAMES = ["dx", "dy", "dz", "R00", "R01", "R02", "R10", "R11", "R12", "grip"]


def load_all_data():
    """Load all parquet files and stack into (N, 10) arrays."""
    all_states, all_actions = [], []
    for i in range(NUM_EPISODES):
        pq_path = DATASET_ROOT / f"data/chunk-000/episode_{i:06d}.parquet"
        df = pd.read_parquet(pq_path)
        states = np.stack(df["observation.state"].values)
        actions = np.stack(df["action"].values)
        all_states.append(states)
        all_actions.append(actions)
    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"  Loaded {all_states.shape[0]} frames: state {all_states.shape}, action {all_actions.shape}")
    return all_states, all_actions


def compute_stats(arr, name, dim_names):
    """Compute and print per-dimension statistics."""
    print(f"\n  {name} statistics ({arr.shape[0]} frames, {arr.shape[1]} dims):")
    header = f"  {'dim':>8s}  {'mean':>10s}  {'std':>10s}  {'min':>10s}  {'max':>10s}  {'q01':>10s}  {'q99':>10s}  {'outliers':>8s}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    stats = {}
    for d in range(arr.shape[1]):
        col = arr[:, d]
        mean = np.mean(col)
        std = np.std(col)
        q01 = np.percentile(col, 1)
        q99 = np.percentile(col, 99)
        cmin = np.min(col)
        cmax = np.max(col)
        n_outliers = np.sum((col < q01) | (col > q99))
        stats[d] = {"mean": mean, "std": std, "min": cmin, "max": cmax, "q01": q01, "q99": q99}
        dname = dim_names[d] if d < len(dim_names) else f"d{d}"
        print(f"  {dname:>8s}  {mean:10.6f}  {std:10.6f}  {cmin:10.6f}  {cmax:10.6f}  {q01:10.6f}  {q99:10.6f}  {n_outliers:8d}")
    return stats


def cross_check_stats(computed, stats_json_key, stats_gr00t):
    """Cross-check computed stats with stats_gr00t.json."""
    errors = []
    ref = stats_gr00t.get("abs", {}).get(stats_json_key, {})
    if not ref:
        print(f"  WARNING: no reference stats for {stats_json_key}")
        return errors

    for stat_name in ["mean", "std", "min", "max", "q01", "q99"]:
        ref_vals = ref.get(stat_name, [])
        for d in range(min(len(ref_vals), 10)):
            comp_val = computed[d][stat_name]
            ref_val = ref_vals[d]
            if abs(ref_val) > 1e-8:
                rel_err = abs(comp_val - ref_val) / abs(ref_val)
            else:
                rel_err = abs(comp_val - ref_val)
            if rel_err > 0.01:  # 1% tolerance
                errors.append(f"{stats_json_key} dim{d} {stat_name}: computed={comp_val:.6f} vs ref={ref_val:.6f} (err={rel_err:.4f})")
    if errors:
        print(f"  Cross-check FAIL ({len(errors)} mismatches):")
        for e in errors[:5]:
            print(f"    - {e}")
    else:
        print(f"  Cross-check PASS: computed stats match stats_gr00t.json")
    return errors


def check_state_rotation(states):
    """Check rot6d validity: v1=state[:,3:6], v2=state[:,6:9] should be unit + orthogonal."""
    v1 = states[:, 3:6]
    v2 = states[:, 6:9]
    v1_norm = np.linalg.norm(v1, axis=1)
    v2_norm = np.linalg.norm(v2, axis=1)
    dot = np.sum(v1 * v2, axis=1)

    print(f"\n  State rotation validity:")
    print(f"    |v1| — mean={np.mean(v1_norm):.6f}, std={np.std(v1_norm):.6f}, min={np.min(v1_norm):.6f}, max={np.max(v1_norm):.6f}")
    print(f"    |v2| — mean={np.mean(v2_norm):.6f}, std={np.std(v2_norm):.6f}, min={np.min(v2_norm):.6f}, max={np.max(v2_norm):.6f}")
    print(f"    v1·v2 — mean={np.mean(dot):.6f}, std={np.std(dot):.6f}, min={np.min(dot):.6f}, max={np.max(dot):.6f}")

    tol = 0.05
    bad_v1 = np.sum(np.abs(v1_norm - 1.0) > tol)
    bad_v2 = np.sum(np.abs(v2_norm - 1.0) > tol)
    bad_orth = np.sum(np.abs(dot) > tol)
    print(f"    |v1| off by >{tol}: {bad_v1}/{len(v1_norm)} frames")
    print(f"    |v2| off by >{tol}: {bad_v2}/{len(v2_norm)} frames")
    print(f"    |v1·v2| > {tol}: {bad_orth}/{len(dot)} frames")

    return v1_norm, v2_norm, dot


def check_action_rotation(actions):
    """Check action rotation dims: 3,7 should be ~1 (identity diagonal), others near 0."""
    print(f"\n  Action rotation check (relative rotation matrix entries):")
    rot_dims = {3: "R00 (~1)", 4: "R01 (~0)", 5: "R02 (~0)", 6: "R10 (~0)", 7: "R11 (~1)", 8: "R12 (~0)"}
    for d, desc in rot_dims.items():
        col = actions[:, d]
        print(f"    dim {d} ({desc}): mean={np.mean(col):.6f}, std={np.std(col):.6f}, min={np.min(col):.6f}, max={np.max(col):.6f}")


def plot_histograms(data, dim_names, title_prefix, filename):
    """Plot 2x5 grid of histograms."""
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for d in range(10):
        ax = axes[d]
        col = data[:, d]
        q01 = np.percentile(col, 1)
        q99 = np.percentile(col, 99)
        ax.hist(col, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
        ax.axvline(q01, color="red", linestyle="--", linewidth=1, label=f"q01={q01:.4f}")
        ax.axvline(q99, color="red", linestyle="--", linewidth=1, label=f"q99={q99:.4f}")
        ax.set_title(f"{dim_names[d]}", fontsize=10)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{title_prefix} Histograms (n={data.shape[0]})", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_boxplots(data, dim_names, title_prefix, filename):
    """Plot box plot of all 10 dims."""
    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot([data[:, d] for d in range(10)], labels=dim_names,
                    patch_artist=True, showfliers=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_title(f"{title_prefix} Box Plots (fliers hidden)")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  Saved {filename}")


def plot_rotation_validity(v1_norm, v2_norm, dot):
    """Plot histograms of rotation validity metrics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(v1_norm, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[0].axvline(1.0, color="red", linestyle="--", label="|v1|=1")
    axes[0].set_title("|v1| (state rot6d col 1)")
    axes[0].set_xlabel("Norm")
    axes[0].legend()

    axes[1].hist(v2_norm, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1].axvline(1.0, color="red", linestyle="--", label="|v2|=1")
    axes[1].set_title("|v2| (state rot6d col 2)")
    axes[1].set_xlabel("Norm")
    axes[1].legend()

    axes[2].hist(dot, bins=80, alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[2].axvline(0.0, color="red", linestyle="--", label="v1·v2=0")
    axes[2].set_title("v1 · v2 (orthogonality)")
    axes[2].set_xlabel("Dot product")
    axes[2].legend()

    fig.suptitle("State Rotation Validity (rot6d)", fontsize=14)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_rotation_validity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 02_rotation_validity.png")


def plot_action_correlation(actions):
    """Plot 10x10 correlation heatmap."""
    corr = np.corrcoef(actions.T)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10))
    ax.set_xticklabels(ACTION_DIM_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10))
    ax.set_yticklabels(ACTION_DIM_NAMES)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Action Dimension Correlation")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_action_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved 02_action_correlation.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Script 2: Distributions & Rotation Validity")
    print("=" * 60)

    # Load all data
    print("\nLoading all parquet files ...")
    states, actions = load_all_data()

    # Load reference stats
    with open(DATASET_ROOT / "meta/stats_gr00t.json") as f:
        stats_gr00t = json.load(f)

    # Per-dimension statistics
    state_stats = compute_stats(states, "State", STATE_DIM_NAMES)
    action_stats = compute_stats(actions, "Action", ACTION_DIM_NAMES)

    # Cross-check
    print("\n  Cross-checking with stats_gr00t.json ...")
    cross_check_stats(state_stats, "observation.state", stats_gr00t)
    cross_check_stats(action_stats, "action", stats_gr00t)

    # Rotation validity (state)
    v1_norm, v2_norm, dot = check_state_rotation(states)

    # Rotation check (action)
    check_action_rotation(actions)

    # Visualizations
    print("\nGenerating visualizations ...")
    plot_histograms(states, STATE_DIM_NAMES, "State", "02_state_histograms.png")
    plot_histograms(actions, ACTION_DIM_NAMES, "Action", "02_action_histograms.png")
    plot_boxplots(states, STATE_DIM_NAMES, "State", "02_state_boxplots.png")
    plot_boxplots(actions, ACTION_DIM_NAMES, "Action", "02_action_boxplots.png")
    plot_rotation_validity(v1_norm, v2_norm, dot)
    plot_action_correlation(actions)

    print("\n" + "=" * 60)
    print("PASS: Distribution analysis complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
