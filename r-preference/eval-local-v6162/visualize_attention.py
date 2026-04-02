#!/usr/bin/env python3
"""
Visualize attention maps saved by QwenOFT with STARVLA_SAVE_ATTENTION=1.

Reads .npz files from an attention output directory and produces:
  1. Aggregate bar chart: per-layer attention distribution (image / text / action)
  2. Spatial heatmap overlay: attention heatmap on observation images
  3. Attention over time: line plot of attention proportions across timesteps

Usage:
    python r-preference/eval-local-v6162/visualize_attention.py \
        --input_dir results/attention_maps/v62/
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_attention_data(input_dir: str) -> list[dict]:
    """Load all step_*.npz files sorted by step number."""
    pattern = os.path.join(input_dir, "step_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No step_*.npz files found in {input_dir}")
    print(f"Found {len(files)} attention files in {input_dir}")

    data = []
    for f in files:
        npz = np.load(f, allow_pickle=True)
        entry = {"_path": f}
        for key in npz.files:
            entry[key] = npz[key]
        data.append(entry)
    return data


def parse_aggregate(data: list[dict]) -> dict:
    """
    Parse aggregate attention data.

    Returns:
        {layer_idx: {"image": [float...], "text": [float...], "action": [float...]}}
    """
    layers = {}
    # Discover layer keys from first entry
    for key in data[0]:
        if key.startswith("agg_layer_"):
            parts = key.split("_")  # agg_layer_<N>_<type>
            layer_key = f"layer_{parts[2]}"
            if layer_key not in layers:
                layers[layer_key] = {"image": [], "text": [], "action": []}

    for entry in data:
        for layer_key in layers:
            for token_type in ("image", "text", "action"):
                npz_key = f"agg_{layer_key}_{token_type}"
                if npz_key in entry:
                    layers[layer_key][token_type].append(float(entry[npz_key]))
    return layers


def plot_aggregate_bar(layers: dict, output_path: str):
    """Stacked bar chart of average attention distribution per layer."""
    sorted_keys = sorted(layers.keys(), key=lambda k: int(k.split("_")[1]))

    img_means = [np.mean(layers[k]["image"]) for k in sorted_keys]
    txt_means = [np.mean(layers[k]["text"]) for k in sorted_keys]
    act_means = [np.mean(layers[k]["action"]) for k in sorted_keys]

    x = np.arange(len(sorted_keys))
    width = 0.6

    fig, ax = plt.subplots(figsize=(max(6, len(sorted_keys) * 1.2), 5))
    bars_img = ax.bar(x, img_means, width, label="Image", color="#4C72B0")
    bars_txt = ax.bar(x, txt_means, width, bottom=img_means, label="Text", color="#55A868")
    bars_act = ax.bar(
        x, act_means, width,
        bottom=[i + t for i, t in zip(img_means, txt_means)],
        label="Action", color="#C44E52",
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Attention Proportion")
    ax.set_title("Action Token Attention Distribution by Layer\n(averaged over all steps)")
    ax.set_xticks(x)
    ax.set_xticklabels([k.replace("layer_", "L") for k in sorted_keys])
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")

    # Add percentage labels on bars
    for bars, values in [(bars_img, img_means), (bars_txt, txt_means), (bars_act, act_means)]:
        for bar, val in zip(bars, values):
            if val > 0.05:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0%}",
                    ha="center", va="center", fontsize=9, fontweight="bold", color="white",
                )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregate bar chart → {output_path}")


def plot_attention_over_time(layers: dict, output_path: str):
    """Line plot of attention proportions across timesteps."""
    # Use the last layer
    sorted_keys = sorted(layers.keys(), key=lambda k: int(k.split("_")[1]))
    last_key = sorted_keys[-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = np.arange(len(layers[last_key]["image"]))
    ax.plot(steps, layers[last_key]["image"], "-o", label="Image", color="#4C72B0", markersize=3)
    ax.plot(steps, layers[last_key]["text"], "-s", label="Text", color="#55A868", markersize=3)
    ax.plot(steps, layers[last_key]["action"], "-^", label="Action", color="#C44E52", markersize=3)

    ax.set_xlabel("Inference Step")
    ax.set_ylabel("Attention Proportion")
    ax.set_title(f"Attention Distribution Over Time ({last_key.replace('_', ' ').title()})")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved attention-over-time plot → {output_path}")


def plot_spatial_heatmaps(data: list[dict], output_path: str, max_timesteps: int = 6):
    """Overlay spatial attention heatmaps on observation images."""
    # Pick representative timesteps (evenly spaced)
    total = len(data)
    if total <= max_timesteps:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, max_timesteps, dtype=int).tolist()

    # Discover how many images per step
    sample = data[indices[0]]
    num_images = sum(1 for k in sample if k.startswith("spatial_img"))
    if num_images == 0:
        print("No spatial attention maps found — skipping heatmap visualization")
        return

    camera_names = ["Head Camera", "Left Camera", "Right Camera"]

    fig, axes = plt.subplots(
        num_images, len(indices),
        figsize=(3.5 * len(indices), 3.5 * num_images),
        squeeze=False,
    )

    for col, step_idx in enumerate(indices):
        entry = data[step_idx]
        step_num = int(entry.get("step", step_idx))

        for row in range(num_images):
            ax = axes[row, col]
            spatial_key = f"spatial_img{row}"
            raw_key = f"raw_img{row}"

            if spatial_key not in entry:
                ax.axis("off")
                continue

            spatial_map = entry[spatial_key]  # (h, w) or (t, h, w)
            if spatial_map.ndim == 3:
                spatial_map = spatial_map[0]

            # Show raw image if available
            if raw_key in entry:
                raw_img = entry[raw_key]
                ax.imshow(raw_img)
                # Resize spatial map to image dimensions for overlay
                from PIL import Image as PILImage
                h_img, w_img = raw_img.shape[:2]
                spatial_resized = np.array(
                    PILImage.fromarray(
                        ((spatial_map - spatial_map.min()) / (spatial_map.max() - spatial_map.min() + 1e-8) * 255).astype(np.uint8)
                    ).resize((w_img, h_img), PILImage.BILINEAR)
                ).astype(float) / 255.0
                ax.imshow(spatial_resized, cmap="jet", alpha=0.45, vmin=0, vmax=1)
            else:
                im = ax.imshow(spatial_map, cmap="jet")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(im, cax=cax)

            if col == 0:
                cam_name = camera_names[row] if row < len(camera_names) else f"Image {row}"
                ax.set_ylabel(cam_name, fontsize=11)
            if row == 0:
                ax.set_title(f"Step {step_num}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Spatial Attention Heatmaps (Action → Image Tokens)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spatial heatmaps → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize starVLA attention maps")
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Directory containing step_*.npz files (e.g., results/attention_maps/v62/)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for figures (default: <input_dir>/figures/)",
    )
    parser.add_argument(
        "--max_heatmap_steps", type=int, default=6,
        help="Max timesteps to show in spatial heatmap grid (default: 6)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_attention_data(args.input_dir)
    layers = parse_aggregate(data)

    # 1. Aggregate bar chart
    plot_aggregate_bar(layers, os.path.join(output_dir, "aggregate_attention.png"))

    # 2. Attention over time
    if len(data) > 1:
        plot_attention_over_time(layers, os.path.join(output_dir, "attention_over_time.png"))
    else:
        print("Only 1 step — skipping attention-over-time plot")

    # 3. Spatial heatmaps
    plot_spatial_heatmaps(data, os.path.join(output_dir, "spatial_heatmaps.png"), max_timesteps=args.max_heatmap_steps)

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
