#!/usr/bin/env python3
"""
Download fastumi_pickandplace_qwenPI checkpoint (steps_15000) from outsider86/DiscreteRTC.

Usage:
    conda activate starVLA
    # Optional: login for higher rate limits
    huggingface-cli login
    # Or: export HF_TOKEN=your_token

    python realworld/download_checkpoint_pickandplace_ur5_0314.py
"""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def main():
    from huggingface_hub import snapshot_download

    # Download only fastumi_pickandplace_qwenPI with steps_15000 checkpoint
    # from https://huggingface.co/outsider86/DiscreteRTC/tree/main
    # Skips steps_5000 and steps_10000 checkpoints, and the wandb folder
    output_dir = REPO_ROOT / "checkpoints/DiscreteRTC"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id="outsider86/DiscreteRTC",
        repo_type="model",
        local_dir=str(output_dir),
        allow_patterns=[
            "fastumi_pickandplace_qwenPI/config.yaml",
            "fastumi_pickandplace_qwenPI/dataset_statistics.json",
            "fastumi_pickandplace_qwenPI/summary.jsonl",
            "fastumi_pickandplace_qwenPI/checkpoints/steps_15000_pytorch_model.pt",
        ],
        max_workers=2,
    )
    print(f"Checkpoint downloaded to: {path}")
    print(f"  - fastumi_pickandplace_qwenPI/checkpoints/steps_15000_pytorch_model.pt")


if __name__ == "__main__":
    main()
