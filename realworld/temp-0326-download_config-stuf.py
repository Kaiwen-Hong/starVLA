#!/usr/bin/env python3
"""
Download config.yaml and dataset_statistics.json (everything except checkpoints)
from kaiwen2/discreteRTC (folder 0325).

Usage:
    conda activate starVLA
    python realworld/temp-0326-download_config-stuf.py
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def enable_hf_transfer():
    """Enable hf-transfer for multi-threaded downloads if available."""
    try:
        import hf_transfer  # noqa: F401
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        print("hf-transfer detected, using multi-threaded download.")
        return True
    except ImportError:
        print("hf-transfer not installed. For faster downloads: pip install hf-transfer")
        return False


def main():
    enable_hf_transfer()  # must be before importing huggingface_hub

    from huggingface_hub import snapshot_download

    token = input("Enter your HF token: ").strip()
    if not token:
        print("No token provided, aborting.")
        return

    output_dir = REPO_ROOT / "checkpoints" / "discreteRTC"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="kaiwen2/discreteRTC",
        repo_type="model",
        local_dir=str(output_dir),
        allow_patterns=[
            "0325/config.yaml",
            "0325/dataset_statistics.json",
        ],
        token=token,
        max_workers=8,
    )
    print(f"Downloaded to: {path}")
    print(f"  config.yaml and dataset_statistics.json are under: 0325/")


if __name__ == "__main__":
    main()
