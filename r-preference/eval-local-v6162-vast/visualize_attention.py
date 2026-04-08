#!/usr/bin/env python3
"""
Visualize attention maps saved by QwenOFT with STARVLA_SAVE_ATTENTION=1.

Reads .npz files from an attention output directory and produces:
  1. Aggregate bar chart: per-layer attention distribution (image / text / action)
  2. Attention over time: line plot of attention proportions across timesteps
  3. Spatial heatmap overlay: attention heatmap on observation images
  --- Per-action-token analysis (requires updated QwenOFT) ---
  4. Action-token x time heatmap: per-action-token attention evolution
  5. Per-action-token profile: avg attention allocation per action position
  6. Per-action-token spatial: do immediate vs future actions look at different regions?

Usage:
    python r-preference/eval-local-v6162-vast/visualize_attention.py \
        --input_dir results/attention_maps/v62/
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# ── Data loading ─────────────────────────────────────────────────────────────


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


# ── Parsing helpers ──────────────────────────────────────────────────────────


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


def parse_per_action_aggregate(data: list[dict]) -> dict | None:
    """
    Parse per-action-token aggregate data.

    Returns:
        {layer_key: {"image": (num_steps, num_act), "text": ..., "action": ...}}
        or None if per-action-token data is not available.
    """
    layers = {}
    for key in data[0]:
        if key.startswith("per_act_agg_layer_"):
            parts = key.split("_")  # per_act_agg_layer_<N>_<type>
            layer_key = f"layer_{parts[4]}"
            if layer_key not in layers:
                layers[layer_key] = {"image": [], "text": [], "action": []}

    if not layers:
        return None

    for entry in data:
        for layer_key in layers:
            for token_type in ("image", "text", "action"):
                npz_key = f"per_act_agg_{layer_key}_{token_type}"
                if npz_key in entry:
                    layers[layer_key][token_type].append(entry[npz_key])

    # Stack: (num_steps, num_act)
    for layer_key in layers:
        for token_type in ("image", "text", "action"):
            layers[layer_key][token_type] = np.stack(layers[layer_key][token_type])
    return layers


# ── Original plots (backward compatible) ─────────────────────────────────────


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
    print(f"Saved aggregate bar chart -> {output_path}")


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
    print(f"Saved attention-over-time plot -> {output_path}")


def plot_spatial_heatmaps(data: list[dict], output_path: str, max_timesteps: int = 6):
    """Overlay spatial attention heatmaps on observation images."""
    from PIL import Image as PILImage

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
        print("No spatial attention maps found -- skipping heatmap visualization")
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

    fig.suptitle("Spatial Attention Heatmaps (Action -> Image Tokens)", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved spatial heatmaps -> {output_path}")


# ── NEW: Per-action-token plots ──────────────────────────────────────────────


def plot_action_token_time_heatmap(per_act_layers: dict, output_path: str):
    """
    2D heatmap showing how each action token's attention evolves over inference steps.

    Layout: 3 subplots (image / text / action), each is a heatmap with
      x-axis = inference step, y-axis = action token index (0 = immediate next action).

    This answers BOTH questions at once:
      - Temporal: how does attention change as the robot acts? (read along x-axis)
      - Per-token: do near-future vs far-future actions attend differently? (read along y-axis)
    """
    sorted_keys = sorted(per_act_layers.keys(), key=lambda k: int(k.split("_")[1]))
    last_key = sorted_keys[-1]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    types_cmaps = [("image", "Blues"), ("text", "Greens"), ("action", "Reds")]

    for ax, (token_type, cmap) in zip(axes, types_cmaps):
        mat = per_act_layers[last_key][token_type]  # (num_steps, num_act)
        num_act = mat.shape[1]
        im = ax.imshow(
            mat.T, aspect="auto", cmap=cmap,
            vmin=0, vmax=max(mat.max(), 0.01),
            interpolation="nearest", origin="upper",
        )
        ax.set_xlabel("Inference Step")
        if ax == axes[0]:
            ax.set_ylabel("Action Token Index\n(0 = next action, last = farthest future)")
        ax.set_title(f"Attention -> {token_type.title()}")
        ax.set_yticks(range(0, num_act, max(1, num_act // 8)))
        plt.colorbar(im, ax=ax, shrink=0.8, label="Proportion")

    layer_label = last_key.replace("_", " ").title()
    fig.suptitle(
        f"Per-Action-Token Attention Over Time ({layer_label})\n"
        "Q: Do different action positions attend differently? Does it change over time?",
        fontsize=13, y=1.04,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved action-token x time heatmap -> {output_path}")


def plot_per_action_token_profile(per_act_layers: dict, output_path: str):
    """
    Heatmap: y = action token index (0 = immediate), x = image/text/action.
    Values averaged over all inference steps.

    Quick summary: does the model differentiate between near-future and far-future
    actions in terms of what information source they attend to?
    """
    sorted_keys = sorted(per_act_layers.keys(), key=lambda k: int(k.split("_")[1]))
    last_key = sorted_keys[-1]

    img_avg = per_act_layers[last_key]["image"].mean(axis=0)  # (num_act,)
    txt_avg = per_act_layers[last_key]["text"].mean(axis=0)
    act_avg = per_act_layers[last_key]["action"].mean(axis=0)

    profile = np.stack([img_avg, txt_avg, act_avg], axis=1)  # (num_act, 3)
    num_act = profile.shape[0]

    fig, ax = plt.subplots(figsize=(4.5, max(5, num_act * 0.45)))
    im = ax.imshow(profile, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Image", "Text", "Action"])
    ax.set_ylabel("Action Token Index\n(0 = next action, last = farthest future)")
    ax.set_yticks(range(num_act))
    ax.set_yticklabels([f"act[{i}]" for i in range(num_act)])

    layer_label = last_key.replace("_", " ").title()
    ax.set_title(f"Avg Attention Profile per Action Token\n({layer_label}, averaged over all steps)")

    for i in range(num_act):
        for j in range(3):
            color = "white" if profile[i, j] > 0.5 else "black"
            ax.text(
                j, i, f"{profile[i, j]:.0%}",
                ha="center", va="center", fontsize=9, fontweight="bold", color=color,
            )

    plt.colorbar(im, ax=ax, shrink=0.6)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-action-token profile -> {output_path}")


def plot_per_action_spatial(
    data: list[dict],
    output_path: str,
    act_indices: list[int] | None = None,
    max_timesteps: int = 5,
):
    """
    Spatial attention comparison across action token positions.

    Grid layout:
      Rows: grouped by camera, sub-rows for selected action tokens
            e.g., Head Camera / act[0], Head Camera / act[15]
      Columns: representative inference steps

    This directly answers: does the immediate-next-action token look at the
    gripper/object while the far-future token looks at the target location?
    """
    from PIL import Image as PILImage

    sample = data[0]
    num_images = sum(1 for k in sample if k.startswith("per_act_spatial_img"))
    if num_images == 0:
        print("No per-action-token spatial maps -- skipping")
        return

    # Auto-detect chunk_len and pick representative action indices
    chunk_len = sample["per_act_spatial_img0"].shape[0]
    if act_indices is None:
        if chunk_len <= 4:
            act_indices = list(range(chunk_len))
        else:
            act_indices = [0, chunk_len // 2, chunk_len - 1]

    # Pick representative inference steps
    total = len(data)
    if total <= max_timesteps:
        step_indices = list(range(total))
    else:
        step_indices = np.linspace(0, total - 1, max_timesteps, dtype=int).tolist()

    camera_names = ["Head Camera", "Left Camera", "Right Camera"]
    num_act_shown = len(act_indices)
    total_rows = num_images * num_act_shown

    fig, axes = plt.subplots(
        total_rows, len(step_indices),
        figsize=(3.5 * len(step_indices), 3 * total_rows),
        squeeze=False,
    )

    for col, step_idx in enumerate(step_indices):
        entry = data[step_idx]
        step_num = int(entry.get("step", step_idx))

        for img_row in range(num_images):
            per_act_key = f"per_act_spatial_img{img_row}"
            raw_key = f"raw_img{img_row}"

            if per_act_key not in entry:
                for sub in range(num_act_shown):
                    axes[img_row * num_act_shown + sub, col].axis("off")
                continue

            per_act_spatial = entry[per_act_key]  # (num_act, h, w) or (num_act, t, h, w)
            if per_act_spatial.ndim == 4:
                per_act_spatial = per_act_spatial[:, 0]

            raw_img = entry.get(raw_key, None)

            for sub, act_idx in enumerate(act_indices):
                row = img_row * num_act_shown + sub
                ax = axes[row, col]

                if act_idx >= per_act_spatial.shape[0]:
                    ax.axis("off")
                    continue

                spatial_map = per_act_spatial[act_idx]  # (h, w)

                if raw_img is not None:
                    ax.imshow(raw_img)
                    h_img, w_img = raw_img.shape[:2]
                    smin, smax = spatial_map.min(), spatial_map.max()
                    smap_norm = (spatial_map - smin) / (smax - smin + 1e-8)
                    spatial_resized = np.array(
                        PILImage.fromarray((smap_norm * 255).astype(np.uint8))
                        .resize((w_img, h_img), PILImage.BILINEAR)
                    ).astype(float) / 255.0
                    ax.imshow(spatial_resized, cmap="jet", alpha=0.5, vmin=0, vmax=1)
                else:
                    ax.imshow(spatial_map, cmap="jet")

                if col == 0:
                    cam = camera_names[img_row] if img_row < len(camera_names) else f"Img{img_row}"
                    if act_idx == 0:
                        label = "now"
                    elif act_idx == chunk_len - 1:
                        label = "future"
                    else:
                        label = f"+{act_idx}"
                    ax.set_ylabel(f"{cam}\nact[{act_idx}] ({label})", fontsize=9)
                if row == 0:
                    ax.set_title(f"Step {step_num}", fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])

    fig.suptitle(
        "Per-Action-Token Spatial Attention\n"
        "Q: Does the immediate action look at the gripper/object while the future action looks at the target?",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-action spatial comparison -> {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────────


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
        help="Max timesteps to show in spatial heatmap grids (default: 6)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or os.path.join(args.input_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data = load_attention_data(args.input_dir)
    layers = parse_aggregate(data)
    per_act_layers = parse_per_action_aggregate(data)

    # 1. Aggregate bar chart
    plot_aggregate_bar(layers, os.path.join(output_dir, "aggregate_attention.png"))

    # 2. Attention over time
    if len(data) > 1:
        plot_attention_over_time(layers, os.path.join(output_dir, "attention_over_time.png"))
    else:
        print("Only 1 step -- skipping attention-over-time plot")

    # 3. Spatial heatmaps (averaged over action tokens)
    plot_spatial_heatmaps(
        data, os.path.join(output_dir, "spatial_heatmaps.png"),
        max_timesteps=args.max_heatmap_steps,
    )

    # --- Per-action-token analysis (requires updated QwenOFT) ---
    if per_act_layers is not None:
        # 4. Action-token x time heatmap (THE key visualization)
        if len(data) > 1:
            plot_action_token_time_heatmap(
                per_act_layers,
                os.path.join(output_dir, "action_token_time_heatmap.png"),
            )

        # 5. Per-action-token profile
        plot_per_action_token_profile(
            per_act_layers,
            os.path.join(output_dir, "per_action_token_profile.png"),
        )

        # 6. Per-action-token spatial comparison
        plot_per_action_spatial(
            data,
            os.path.join(output_dir, "per_action_spatial.png"),
            max_timesteps=args.max_heatmap_steps,
        )
    else:
        print("\nNo per-action-token data found -- skipping new visualizations.")
        print("Re-run inference with updated QwenOFT to generate per-action-token data.")

    print(f"\nAll figures saved to {output_dir}")


if __name__ == "__main__":
    main()
