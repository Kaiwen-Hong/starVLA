#!/usr/bin/env python3
"""Script 5: Training Pipeline End-to-End Check

Uses the actual starVLA data loading pipeline (make_LeRobotSingleDataset +
FastUMIDataConfig) to instantiate the dataset exactly as training does,
then validates output shapes, normalization ranges, and image format.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_ROOT = REPO_ROOT / "playground/Datasets/FastUMI"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ACTION_DIM_NAMES = ["dx", "dy", "dz", "R00", "R01", "R02", "R10", "R11", "R12", "grip"]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Script 5: Training Pipeline End-to-End Check")
    print("=" * 60)

    # Import starVLA pipeline — stub out decord so that
    # dataloader/__init__.py -> vlm_datasets -> `from decord import VideoReader`
    # doesn't crash when decord isn't installed.
    print("\nImporting starVLA data pipeline ...")
    import types
    if "decord" not in sys.modules:
        _decord = types.ModuleType("decord")
        _decord.VideoReader = None
        sys.modules["decord"] = _decord
    from starVLA.dataloader.lerobot_datasets import make_LeRobotSingleDataset

    # Create dataset the same way training does
    print("Creating LeRobotSingleDataset (video_backend=torchvision_av) ...")
    data_cfg = {"video_backend": "torchvision_av"}
    dataset = make_LeRobotSingleDataset(
        data_root_dir=DATASET_ROOT,
        data_name="pickandplace-real-0307",
        robot_type="fastumi",
        data_cfg=data_cfg,
    )
    print(f"  Dataset length: {len(dataset)}")

    # Sample 20 random items (enough to validate, keeps runtime short)
    n_samples = min(20, len(dataset))
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(len(dataset), size=n_samples, replace=False)
    sample_indices.sort()

    print(f"\nSampling {n_samples} items via __getitem__ ...")
    all_actions = []
    sample_images = []
    all_langs = []
    errors = []

    for idx_i, idx in enumerate(sample_indices):
        try:
            item = dataset[idx]
        except Exception as e:
            errors.append(f"  idx={idx}: __getitem__ failed: {e}")
            continue

        # Check action shape
        action = item.get("action")
        if action is None:
            errors.append(f"  idx={idx}: 'action' key missing")
            continue

        action_arr = np.array(action, dtype=np.float32)
        if action_arr.shape != (16, 10):
            errors.append(f"  idx={idx}: action shape {action_arr.shape}, expected (16, 10)")
        all_actions.append(action_arr)

        # Check image
        images = item.get("image")
        if images is None or len(images) == 0:
            errors.append(f"  idx={idx}: 'image' key missing or empty")
        else:
            img = images[0]  # first (wrist) image
            from PIL import Image
            if not isinstance(img, Image.Image):
                errors.append(f"  idx={idx}: image is {type(img)}, expected PIL.Image")
            else:
                if img.size != (224, 224):
                    errors.append(f"  idx={idx}: image size {img.size}, expected (224, 224)")
                if len(sample_images) < 8:
                    sample_images.append((idx, img))

        # Check lang
        lang = item.get("lang")
        if lang is None or not isinstance(lang, str) or len(lang) == 0:
            errors.append(f"  idx={idx}: 'lang' is missing or empty (got {repr(lang)})")
        else:
            all_langs.append(lang)

        if (idx_i + 1) % 20 == 0:
            print(f"    sampled {idx_i + 1}/{n_samples} ...")

    if errors:
        print(f"\n  {len(errors)} errors found:")
        for e in errors[:15]:
            print(f"    {e}")
        if len(errors) > 15:
            print(f"    ... and {len(errors) - 15} more")
    else:
        print(f"  All {n_samples} samples passed shape/type checks")

    # Language check
    if all_langs:
        unique_langs = set(all_langs)
        print(f"\n  Language strings: {len(all_langs)} non-empty, {len(unique_langs)} unique")
        for lang in list(unique_langs)[:3]:
            print(f"    \"{lang}\"")

    # Analyze normalized action ranges
    if all_actions:
        actions_np = np.stack(all_actions)  # (n_samples, 16, 10)
        # Flatten to (n_samples*16, 10) for per-dim analysis
        actions_flat = actions_np.reshape(-1, 10)

        print(f"\n  Normalized action statistics ({actions_flat.shape[0]} action steps):")
        print(f"  {'dim':>8s}  {'min':>10s}  {'max':>10s}  {'mean':>10s}  {'OOR_frac':>10s}")
        print("  " + "-" * 52)

        oor_fracs = []
        for d in range(10):
            col = actions_flat[:, d]
            cmin, cmax = np.min(col), np.max(col)
            cmean = np.mean(col)
            if d == 9:
                # Gripper: should be 0 or 1 (binary)
                oor = np.sum((col != 0) & (col != 1)) / len(col)
                range_desc = "binary"
            else:
                # Position/rot6d: should be in [-1, 1] (min_max)
                oor = np.sum((col < -1.0) | (col > 1.0)) / len(col)
                range_desc = "[-1,1]"
            oor_fracs.append(oor)
            dname = ACTION_DIM_NAMES[d]
            print(f"  {dname:>8s}  {cmin:10.4f}  {cmax:10.4f}  {cmean:10.4f}  {oor:10.4f} ({range_desc})")

        # Visualizations
        print("\nGenerating visualizations ...")

        # Histograms of normalized action dims
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        for d in range(10):
            ax = axes[d]
            col = actions_flat[:, d]
            ax.hist(col, bins=60, alpha=0.7, edgecolor="black", linewidth=0.3)
            if d < 9:
                ax.axvline(-1, color="red", linestyle="--", linewidth=1)
                ax.axvline(1, color="red", linestyle="--", linewidth=1)
            ax.set_title(f"{ACTION_DIM_NAMES[d]}", fontsize=10)
            ax.tick_params(labelsize=7)
        fig.suptitle("Normalized Action Histograms (from training pipeline)", fontsize=14)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_normalized_action_histograms.png", dpi=150)
        plt.close(fig)
        print(f"  Saved 05_normalized_action_histograms.png")

        # Min/max bar per dim
        fig, ax = plt.subplots(figsize=(12, 6))
        dim_mins = [np.min(actions_flat[:, d]) for d in range(10)]
        dim_maxs = [np.max(actions_flat[:, d]) for d in range(10)]
        x = np.arange(10)
        ax.bar(x - 0.15, dim_mins, width=0.3, label="Min", color="steelblue", alpha=0.8)
        ax.bar(x + 0.15, dim_maxs, width=0.3, label="Max", color="coral", alpha=0.8)
        ax.axhline(-1, color="red", linestyle="--", linewidth=1, alpha=0.5, label="[-1,1] bounds")
        ax.axhline(1, color="red", linestyle="--", linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(ACTION_DIM_NAMES)
        ax.set_ylabel("Value")
        ax.set_title("Normalized Action Range per Dimension")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_normalized_action_range.png", dpi=150)
        plt.close(fig)
        print(f"  Saved 05_normalized_action_range.png")

    # Sample image grid
    if sample_images:
        n_imgs = len(sample_images)
        cols = min(4, n_imgs)
        rows = (n_imgs + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for i, (idx, img) in enumerate(sample_images):
            r, c = i // cols, i % cols
            axes[r, c].imshow(img)
            axes[r, c].set_title(f"idx={idx}", fontsize=9)
            axes[r, c].axis("off")

        # Hide empty subplots
        for i in range(n_imgs, rows * cols):
            r, c = i // cols, i % cols
            axes[r, c].axis("off")

        fig.suptitle("Sample Images from Training Pipeline (224x224)", fontsize=13)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_sample_image_grid.png", dpi=150)
        plt.close(fig)
        print(f"  Saved 05_sample_image_grid.png")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} errors in pipeline validation")
    else:
        print("PASS: Training pipeline end-to-end check complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
