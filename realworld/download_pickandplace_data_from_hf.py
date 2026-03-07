#!/usr/bin/env python3
"""
Download StarVLA pickandplace dataset from HuggingFace Hub.

Quick start:
    python 0srarvla-lerobot-convert/download_starvla_pickandplace_dataset_from_hf.py

Downloads:
    kaiwen2/naturalvla-dataset (path: pickandplace-real/)
    -> 0srarvla-lerobot-convert/starvla/datasets/pickandplace_vla/
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import login, snapshot_download

REPO_ID = "kaiwen2/discreteRTC-dataset"
REPO_TYPE = "dataset"
REMOTE_PATH = "pickandplace-real-0307"

BASE_DIR = Path("/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen")
DEFAULT_OUTPUT = BASE_DIR / "starVLA/playground/Datasets/FastUMI/pickandplace-real-0307"
DEFAULT_CACHE = BASE_DIR / ".cache/huggingface/hub"


def main():
    parser = argparse.ArgumentParser(
        description="Download StarVLA pickandplace dataset from HuggingFace Hub",
    )
    parser.add_argument(
        "--output-dir", "-o", default=str(DEFAULT_OUTPUT),
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--cache-dir", default=str(DEFAULT_CACHE),
        help=f"HuggingFace cache directory (default: {DEFAULT_CACHE})",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    cache_dir_arg = Path(args.cache_dir)

    token = input("Enter your HuggingFace token (https://huggingface.co/settings/tokens): ").strip()
    if not token:
        print("[ERROR] Token is required.")
        return
    login(token=token)

    print(f"Downloading {REPO_ID}/{REMOTE_PATH}/")
    print(f"  -> {output_dir}")

    # Download to a temp cache location, then copy the subfolder
    cache_dir_arg.mkdir(parents=True, exist_ok=True)
    downloaded = snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        allow_patterns=f"{REMOTE_PATH}/**",
        token=token,
        cache_dir=str(cache_dir_arg),
    )

    # snapshot_download returns the cache dir with full repo structure
    src = Path(downloaded) / REMOTE_PATH
    if not src.exists():
        print(f"[ERROR] Expected path not found in download: {src}")
        return

    # Copy to output directory
    if output_dir.exists():
        print(f"  Removing existing: {output_dir}")
        shutil.rmtree(output_dir)
    shutil.copytree(src, output_dir)

    print(f"\nDone! Dataset saved to: {output_dir}")


if __name__ == "__main__":
    main()
