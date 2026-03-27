#!/usr/bin/env python3
"""
Download dynamic-324 from kaiwen2/discreteRTC-dataset on Hugging Face.

Usage:
    conda activate starVLA
    # Optional: login for higher rate limits
    huggingface-cli login
    # Or: export HF_TOKEN=your_token

    python realworld/download_dynamic_324.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "playground/Datasets/FastUMI/dynamic-324"


def main():
    from huggingface_hub import snapshot_download

    # Download dynamic-324 from https://huggingface.co/datasets/kaiwen2/discreteRTC-dataset
    # Resumes if partial; max_workers=2 reduces 429 rate limits
    # Login first: huggingface-cli login (or set HF_TOKEN) for higher limits
    base = REPO_ROOT / "playground/Datasets/FastUMI/discreteRTC-dataset"
    base.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="kaiwen2/discreteRTC-dataset",
        repo_type="dataset",
        local_dir=str(base),
        allow_patterns="dynamic-324/*",
        max_workers=2,
    )
    # Symlink for starVLA convention
    fastumi = REPO_ROOT / "playground/Datasets/FastUMI"
    link = fastumi / "dynamic-324"
    if (Path(path) / "dynamic-324").exists():
        if link.exists():
            link.unlink(missing_ok=True)
        link.symlink_to("discreteRTC-dataset/dynamic-324")
    print(f"Dataset: {path}")
    print(f"Use: playground/Datasets/FastUMI/dynamic-324")


if __name__ == "__main__":
    main()
