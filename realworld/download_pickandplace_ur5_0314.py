#!/usr/bin/env python3
"""
Download pickandplace-ur5-0314 from kaiwen2/discreteRTC-dataset on Hugging Face.

Usage:
    conda activate starVLA
    # Optional: login for higher rate limits
    huggingface-cli login
    # Or: export HF_TOKEN=your_token

    python realworld/download_pickandplace_ur5_0314.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "playground/Datasets/FastUMI/pickandplace-ur5-0314"


def main():
    from huggingface_hub import snapshot_download

    # Download pickandplace-ur5-0314 from https://huggingface.co/datasets/kaiwen2/discreteRTC-dataset
    # Resumes if partial; max_workers=2 reduces 429 rate limits
    # Login first: huggingface-cli login (or set HF_TOKEN) for higher limits
    base = REPO_ROOT / "playground/Datasets/FastUMI/discreteRTC-dataset"
    base.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="kaiwen2/discreteRTC-dataset",
        repo_type="dataset",
        local_dir=str(base),
        allow_patterns="pickandplace-ur5-0314/*",
        max_workers=2,
    )
    # Symlink for starVLA convention (like pickandplace-real-0307)
    fastumi = REPO_ROOT / "playground/Datasets/FastUMI"
    link = fastumi / "pickandplace-ur5-0314"
    if (Path(path) / "pickandplace-ur5-0314").exists():
        if link.exists():
            link.unlink(missing_ok=True)
        link.symlink_to("discreteRTC-dataset/pickandplace-ur5-0314")
    print(f"Dataset: {path}")
    print(f"Use: playground/Datasets/FastUMI/pickandplace-ur5-0314")


if __name__ == "__main__":
    main()
