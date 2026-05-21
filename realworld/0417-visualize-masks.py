#!/usr/bin/env python3
"""
Standalone visualizer for the masking data logged by
0417-dd-openloop-eval-natural.py.

Loads the sibling `<stem>_masks.npz` + `<stem>_masks.meta.json` produced
by the eval script and renders:

  * mask-evolution grid — rows = RTC cycle, cols = (init + decode steps).
    Each cell is a [H, D] binary mask matrix (black = masked, white = fixed).
    A reference dashed line is drawn at y = inference_delay + execution_horizon,
    which is the early-stop boundary: early_stop fires once all rows above
    this line are unmasked. Orange line marks the prefix boundary from the
    schedule (prefix_length).

  * ending-pattern strip — 2 cells per cycle: initial mask + the final mask
    snapshot actually returned to the caller (may still contain masks when
    early_stop triggered before the full decode completed).

Usage:
    python realworld/0417-visualize-masks.py --npz .../rtc_natural_*_masks.npz
    python realworld/0417-visualize-masks.py --npz ... --max_cycles 20
    python realworld/0417-visualize-masks.py --npz ... --out_dir /tmp/plots
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_logged(npz_path: Path):
    """Return (data_dict, meta_dict). Expects sibling .meta.json."""
    d = np.load(npz_path)
    data = {k: d[k] for k in d.files}
    meta_path = npz_path.with_suffix(".meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return data, meta


def plot_mask_evolution(data, meta, out_path: Path, max_cycles: int | None = None):
    episode_meta = meta.get("episode_meta", {})
    cycles_meta = meta.get("cycles", [])
    inference_delay = int(episode_meta.get("inference_delay", 0))
    n_actions = int(episode_meta.get("n_actions", 0))
    # execution_horizon is set to n_actions in run_rtc_episode.
    execution_horizon = n_actions
    early_boundary = inference_delay + execution_horizon  # the D+N cut

    all_mats = data["mask_matrices"]
    rec_cycle = data["record_cycle"]
    rec_step = data["record_step"]

    uniq_cycles = sorted(set(int(c) for c in rec_cycle))
    if max_cycles is not None:
        uniq_cycles = uniq_cycles[:max_cycles]
    if not uniq_cycles:
        print("[WARN] no cycles to plot")
        return

    # max decode-step count across plotted cycles (for layout width)
    max_steps = 0
    for c in uniq_cycles:
        sel = rec_cycle == c
        steps = rec_step[sel]
        max_steps = max(max_steps, int((steps >= 0).sum()))
    if max_steps == 0:
        print("[WARN] no decode steps captured")
        return

    n_rows = len(uniq_cycles)
    n_cols = max_steps + 1  # init + per-step

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(0.7 * n_cols + 1.2, 0.55 * n_rows + 1.2),
        squeeze=False)
    cmap = plt.get_cmap("gray_r")

    cyc_meta_by_id = {int(c["cycle"]): c for c in cycles_meta}

    for r, cyc in enumerate(uniq_cycles):
        sel = rec_cycle == cyc
        mats = all_mats[sel]
        steps = rec_step[sel]
        cm = cyc_meta_by_id.get(cyc, {})
        stop_step = cm.get("stop_step")
        prefix_len = cm.get("prefix_length")
        cyc_type = cm.get("cycle_type", "?")

        # Order: init (step = -1) first, then decode steps ascending.
        order = np.argsort(steps)
        mats = mats[order]
        steps = steps[order]

        # Map step_id -> column index: init→0, 0→1, 1→2, …
        for col in range(n_cols):
            ax = axes[r][col]
            ax.set_xticks([])
            ax.set_yticks([])

            target_step = col - 1  # col 0 == init (step -1)
            idx = np.where(steps == target_step)[0]
            if len(idx) == 0:
                ax.set_visible(False)
                continue
            mat = mats[idx[0]]
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

            # Reference lines.
            H = mat.shape[0]
            if 0 < early_boundary < H:
                ax.axhline(early_boundary - 0.5, color="#2980b9",
                           linewidth=1.0, linestyle="--",
                           label="inference_delay + exec_h"
                                 if (r == 0 and col == 0) else None)
            if prefix_len and 0 < prefix_len < H:
                ax.axhline(prefix_len - 0.5, color="#e67e22",
                           linewidth=0.8, linestyle=":",
                           label="prefix_length"
                                 if (r == 0 and col == 0) else None)

            # Row label on the leftmost cell.
            if col == 0:
                tag = "INIT" if cyc_type == "init" else "RTC"
                step_info = ""
                if stop_step is not None and cm.get("num_steps_scheduled") is not None:
                    step_info = f" {stop_step}/{cm['num_steps_scheduled']}"
                ax.set_ylabel(
                    f"c{cyc}\n{tag}{step_info}",
                    rotation=0, labelpad=24, fontsize=7,
                    va="center", ha="right")

            # Column header.
            if r == 0:
                label = "init" if target_step == -1 else f"s{target_step + 1}"
                ax.set_title(label, fontsize=7)

            # Red outline on stop step.
            if stop_step is not None and col == stop_step:
                for spine in ax.spines.values():
                    spine.set_color("red")
                    spine.set_linewidth(1.5)

    fig.suptitle(
        f"Mask evolution — D={inference_delay}, N={execution_horizon}, "
        f"D+N early-stop boundary at row {early_boundary} (blue dashed).\n"
        f"Orange dotted = prefix_length. Red border = stop step. "
        f"Black = masked, white = fixed.",
        fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out_path}")


def plot_ending_strip(data, meta, out_path: Path, max_cycles: int | None = None):
    """Two tiles per cycle (init, final) with the early-stop line."""
    episode_meta = meta.get("episode_meta", {})
    cycles_meta = meta.get("cycles", [])
    inference_delay = int(episode_meta.get("inference_delay", 0))
    execution_horizon = int(episode_meta.get("n_actions", 0))
    early_boundary = inference_delay + execution_horizon

    all_mats = data["mask_matrices"]
    rec_cycle = data["record_cycle"]
    rec_step = data["record_step"]

    uniq_cycles = sorted(set(int(c) for c in rec_cycle))
    if max_cycles is not None:
        uniq_cycles = uniq_cycles[:max_cycles]
    if not uniq_cycles:
        return

    cyc_meta_by_id = {int(c["cycle"]): c for c in cycles_meta}

    fig, axes = plt.subplots(
        len(uniq_cycles), 2,
        figsize=(2.8, max(1.2, 0.48 * len(uniq_cycles))),
        squeeze=False)
    cmap = plt.get_cmap("gray_r")

    for r, cyc in enumerate(uniq_cycles):
        sel = rec_cycle == cyc
        mats = all_mats[sel]
        steps = rec_step[sel]
        cm = cyc_meta_by_id.get(cyc, {})
        prefix_len = cm.get("prefix_length")
        cyc_type = cm.get("cycle_type", "?")
        stop_step = cm.get("stop_step")
        sched = cm.get("num_steps_scheduled")

        order = np.argsort(steps)
        mats = mats[order]
        steps = steps[order]

        init_idx = np.where(steps == -1)[0]
        init_mat = mats[init_idx[0]] if len(init_idx) else None
        final_step_entries = steps[steps >= 0]
        final_mat = None
        if len(final_step_entries):
            last_step = int(final_step_entries.max())
            final_idx = np.where(steps == last_step)[0][0]
            final_mat = mats[final_idx]

        for col, (mat, name) in enumerate([
                (init_mat, "init"), (final_mat, "final")]):
            ax = axes[r][col]
            ax.set_xticks([])
            ax.set_yticks([])
            if mat is None:
                ax.set_visible(False)
                continue
            ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            H = mat.shape[0]
            if 0 < early_boundary < H:
                ax.axhline(early_boundary - 0.5, color="#2980b9",
                           linewidth=1.0, linestyle="--")
            if prefix_len and 0 < prefix_len < H:
                ax.axhline(prefix_len - 0.5, color="#e67e22",
                           linewidth=0.8, linestyle=":")
            if r == 0:
                ax.set_title(name, fontsize=8)

        tag = "INIT" if cyc_type == "init" else "RTC"
        info = ""
        if stop_step is not None and sched is not None:
            info = f" {stop_step}/{sched}"
        axes[r][0].set_ylabel(
            f"c{cyc}\n{tag}{info}",
            rotation=0, labelpad=22, fontsize=7,
            va="center", ha="right")

    fig.suptitle(
        f"Per-cycle init vs final — blue dashed = "
        f"D+N={early_boundary} early-stop boundary; orange dotted = prefix_length.",
        fontsize=10, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out_path}")


def plot_single_cycle_grid(data, meta, cycle_idx: int, out_path: Path,
                            n_rows: int = 3, n_cols: int = 3):
    """Render ONE cycle as a grid (default 3x3) of mask snapshots.

    The grid is laid out row-major: cell 0 = initial mask, cells 1..N = decode
    steps 1..N. Blank cells if fewer snapshots than grid capacity.
    """
    episode_meta = meta.get("episode_meta", {})
    cycles_meta = meta.get("cycles", [])
    inference_delay = int(episode_meta.get("inference_delay", 0))
    execution_horizon = int(episode_meta.get("n_actions", 0))
    early_boundary = inference_delay + execution_horizon

    all_mats = data["mask_matrices"]
    rec_cycle = data["record_cycle"]
    rec_step = data["record_step"]

    sel = rec_cycle == cycle_idx
    if not sel.any():
        raise ValueError(f"cycle {cycle_idx} not found in npz")
    mats = all_mats[sel]
    steps = rec_step[sel]
    order = np.argsort(steps)  # -1 (init) first, then 0..N
    mats = mats[order]
    steps = steps[order]

    cm = next((c for c in cycles_meta if int(c["cycle"]) == cycle_idx), {})
    stop_step = cm.get("stop_step")
    prefix_len = cm.get("prefix_length")
    sched = cm.get("num_steps_scheduled")
    cyc_type = cm.get("cycle_type", "?")

    total = n_rows * n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(1.6 * n_cols + 0.8, 1.9 * n_rows + 0.8),
                             squeeze=False)
    cmap = plt.get_cmap("gray_r")

    for i in range(total):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        ax.set_xticks([])
        ax.set_yticks([])
        if i >= len(mats):
            ax.set_visible(False)
            continue
        mat = mats[i]
        step_id = int(steps[i])
        ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")

        H = mat.shape[0]
        if 0 < early_boundary < H:
            ax.axhline(early_boundary - 0.5, color="#2980b9",
                       linewidth=1.0, linestyle="--")
        if prefix_len and 0 < prefix_len < H:
            ax.axhline(prefix_len - 0.5, color="#e67e22",
                       linewidth=0.8, linestyle=":")

        label = "init" if step_id == -1 else f"step {step_id + 1}"
        ax.set_title(label, fontsize=10)

        if stop_step is not None and step_id + 1 == stop_step:
            for spine in ax.spines.values():
                spine.set_color("red")
                spine.set_linewidth(2.0)

    tag = "INIT" if cyc_type == "init" else "RTC"
    info = f"{stop_step}/{sched}" if stop_step is not None else "?"
    fig.suptitle(
        f"Cycle {cycle_idx} ({tag})  —  stop={info}  "
        f"D={inference_delay}  N={execution_horizon}  "
        f"prefix={prefix_len}\n"
        f"blue dashed = D+N early-stop boundary ({early_boundary}); "
        f"orange dotted = prefix_length; red border = stop step",
        fontsize=11, y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved  {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize masking data logged by the episodic RTC eval")
    parser.add_argument("--npz", type=str, required=True,
                        help="Path to the *_masks.npz file")
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Where to save plots (default: alongside npz)")
    parser.add_argument("--max_cycles", type=int, default=None,
                        help="Cap the number of cycles rendered")
    parser.add_argument("--cycle", type=int, default=None,
                        help="If set, also emit a single-cycle grid plot "
                             "(default 3x3) for this cycle idx")
    parser.add_argument("--grid_rows", type=int, default=3)
    parser.add_argument("--grid_cols", type=int, default=3)
    args = parser.parse_args()

    npz_path = Path(args.npz)
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)
    out_dir = Path(args.out_dir) if args.out_dir else npz_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    data, meta = load_logged(npz_path)
    stem = npz_path.stem  # e.g. ..._masks
    em = meta.get("episode_meta", {})
    print(f"Loaded {npz_path.name}: {data['mask_matrices'].shape[0]} snapshots, "
          f"D={em.get('inference_delay')}  N={em.get('n_actions')}  "
          f"chunk_len={em.get('chunk_len')}  num_inf={em.get('num_inference_steps')}")

    plot_mask_evolution(
        data, meta,
        out_dir / f"{stem}_evo_offline.png",
        max_cycles=args.max_cycles)
    plot_ending_strip(
        data, meta,
        out_dir / f"{stem}_ending_offline.png",
        max_cycles=args.max_cycles)

    if args.cycle is not None:
        plot_single_cycle_grid(
            data, meta, cycle_idx=args.cycle,
            out_path=out_dir / f"{stem}_c{args.cycle}_grid.png",
            n_rows=args.grid_rows, n_cols=args.grid_cols)


if __name__ == "__main__":
    main()
