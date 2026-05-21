#!/usr/bin/env python3
"""
Paper-style visualizer for the RTC discrete-diffusion mask snapshots.

Reads the sibling `<stem>_masks.npz` + `<stem>_masks.meta.json` produced by
0417-dd-openloop-eval-natural.py and renders figures styled after
`discreteRTC.pdf`:

  - Time flows left-to-right (first action on the left, final on the right).
  - Rows are action dimensions.
  - Green = unmasked/committed, yellow = masked.
  - Dashed vertical boundaries separate inference-delay / intermediate /
    execution-horizon regions.

Starter command:
    python realworld/0418-rtc-figure.py --npz <...>_masks.npz --cycle 4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# Match the font stack used in the reference figure (discreteRTC.svg).
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial", "sans-serif",
]


# Palette sampled from discreteRTC.pdf
COLOR_UNMASKED = "#2e7d46"   # green
COLOR_MASKED   = "#f4cf2f"   # yellow
COLOR_BOUNDARY = "#6b6b6b"   # dashed region divider
CMAP_GY = ListedColormap([COLOR_UNMASKED, COLOR_MASKED])  # 0→green, 1→yellow


def load_logged(npz_path: Path):
    d = np.load(npz_path)
    data = {k: d[k] for k in d.files}
    with open(npz_path.with_suffix(".meta.json"), "r") as f:
        meta = json.load(f)
    return data, meta


def extract_cycle(data, cycle_idx: int):
    """Return ordered list of (step_id, mat) for the given cycle.

    step_id = -1 for the initial snapshot, 0..N-1 for each decode step.
    Each `mat` is returned TRANSPOSED so rows = action dims (D), cols = time (H),
    matching the paper figure where time flows left-to-right.
    """
    sel = data["record_cycle"] == cycle_idx
    if not sel.any():
        raise ValueError(f"cycle {cycle_idx} not found in npz")
    mats = data["mask_matrices"][sel]
    steps = data["record_step"][sel]
    order = np.argsort(steps)
    mats = mats[order]
    steps = steps[order]
    return [(int(s), m.T.copy()) for s, m in zip(steps, mats)]  # transpose → (D, H)


def draw_chunk(ax, mat, inference_delay: int, execution_horizon: int,
               title: str | None = None, show_boundaries: bool = True,
               gridline_color: str = "white", gridline_width: float = 1.0):
    """Render a single (D, H) mask matrix in paper style onto `ax`.

    - mat: (D, H) uint8, 1=masked, 0=unmasked.
    - Dashed vertical lines at the d/(H-d-s)/s region boundaries.
    """
    D, H = mat.shape
    ax.imshow(mat, cmap=CMAP_GY, vmin=0, vmax=1, aspect="equal",
              interpolation="nearest")
    # thin grid between cells
    ax.set_xticks(np.arange(-0.5, H, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, D, 1), minor=True)
    ax.grid(which="minor", color=gridline_color, linewidth=gridline_width)
    ax.tick_params(which="minor", length=0)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_boundaries:
        early_stop_col = inference_delay + execution_horizon
        if 0 < early_stop_col <= H:
            ax.add_patch(Rectangle(
                (-0.5, -0.5), early_stop_col, D,
                fill=False, edgecolor="red", linewidth=1.8, zorder=5))

    if title is not None:
        ax.set_title(title, fontsize=10, fontweight="bold")


def plot_single_cycle_grid(data, meta, cycle_idx: int, out_path: Path,
                           n_rows: int = 3, n_cols: int = 3):
    """3x3 (or custom) grid — one chunk per cell, time left→right."""
    em = meta.get("episode_meta", {})
    inference_delay = int(em.get("inference_delay", 0))
    execution_horizon = int(em.get("n_actions", 0))
    cm = next((c for c in meta.get("cycles", [])
               if int(c["cycle"]) == cycle_idx), {})
    stop_step = cm.get("stop_step")

    snaps = extract_cycle(data, cycle_idx)
    total = n_rows * n_cols

    D, H = snaps[0][1].shape if snaps else (10, 16)
    cell_inches = 0.18
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(H * cell_inches * n_cols + 0.4,
                 D * cell_inches * n_rows + 0.4),
        squeeze=False)

    for i in range(total):
        r, c = divmod(i, n_cols)
        ax = axes[r][c]
        if i >= len(snaps):
            ax.set_visible(False)
            continue
        step_id, mat = snaps[i]
        label = "Init" if step_id == -1 else f"Step {step_id + 1}"
        draw_chunk(ax, mat,
                   inference_delay=inference_delay,
                   execution_horizon=execution_horizon,
                   title=label)

    fig.tight_layout()
    out_path = Path(out_path)
    for ext in (".png", ".svg", ".pdf"):
        p = out_path.with_suffix(ext)
        fig.savefig(p, dpi=180, bbox_inches="tight")
        print(f"Saved  {p}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Paper-style RTC discrete-diffusion figure renderer.")
    p.add_argument("--npz", type=str, required=True,
                   help="Path to *_masks.npz")
    p.add_argument("--cycle", type=int, required=True,
                   help="Cycle index to visualize")
    p.add_argument("--out", type=str, default=None,
                   help="Output PNG (default: alongside npz)")
    p.add_argument("--grid_rows", type=int, default=3)
    p.add_argument("--grid_cols", type=int, default=3)
    args = p.parse_args()

    npz_path = Path(args.npz)
    data, meta = load_logged(npz_path)
    if args.out:
        out = Path(args.out)
    else:
        out = npz_path.with_name(
            f"{npz_path.stem}_c{args.cycle}_paperstyle.png")

    plot_single_cycle_grid(
        data, meta, cycle_idx=args.cycle, out_path=out,
        n_rows=args.grid_rows, n_cols=args.grid_cols)


if __name__ == "__main__":
    main()
