#!/usr/bin/env python3
"""
Download checkpoints from outsider86/DiscreteRTC:
  - fastumi_pickandplace_qwenPI_329v2 (steps_30000)
  - fastumi_pickandplace_qwenDiscreteDiffusion_329v2 (steps_20000)

Usage:
    conda activate starVLA

    # Fast mode (recommended): install hf-transfer first
    pip install hf-transfer
    HF_HUB_ENABLE_HF_TRANSFER=1 python realworld/0331-download_checkpoint_dynamic_0329.py

    # Normal mode:
    python realworld/0331-download_checkpoint_dynamic_0329.py
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
        repo_id="outsider86/DiscreteRTC",
        repo_type="model",
        local_dir=str(output_dir),
        allow_patterns=[
            "fastumi_pickandplace_qwenPI_329v2/config.yaml",
            "fastumi_pickandplace_qwenPI_329v2/dataset_statistics.json",
            "fastumi_pickandplace_qwenPI_329v2/summary.jsonl",
            "fastumi_pickandplace_qwenPI_329v2/**/steps_30000_pytorch_model.pt",
            "fastumi_pickandplace_qwenDiscreteDiffusion_329v2/config.yaml",
            "fastumi_pickandplace_qwenDiscreteDiffusion_329v2/dataset_statistics.json",
            "fastumi_pickandplace_qwenDiscreteDiffusion_329v2/summary.jsonl",
            "fastumi_pickandplace_qwenDiscreteDiffusion_329v2/**/steps_20000_pytorch_model.pt",
        ],
        token=token,
        max_workers=8,
    )
    print(f"Checkpoint downloaded to: {path}")
    print(f"  - fastumi_pickandplace_qwenPI_329v2 (steps_30000)")
    print(f"  - fastumi_pickandplace_qwenDiscreteDiffusion_329v2 (steps_20000)")


if __name__ == "__main__":
    main()
