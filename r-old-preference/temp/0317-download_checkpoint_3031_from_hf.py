#!/usr/bin/env python3
"""
Download StarVLA v30 and v31 EE checkpoints from HuggingFace to Desktop.

Downloads from: kaiwen2/prefvla-models/starVLA/{run_name}/
Saves to:       <dst>/v0309_v30_ee_qwenPI_requeue/
                <dst>/v0309_v31_ee_qwenPI_requeue/

Default dst: ~/Desktop/research/starVLA/results/Checkpoints/

Usage:
  python 0317-download_checkpoint_3031_from_hf.py
  python 0317-download_checkpoint_3031_from_hf.py --steps 10000 50000
  python 0317-download_checkpoint_3031_from_hf.py --dst /path/to/starVLA/results/Checkpoints
  python 0317-download_checkpoint_3031_from_hf.py --token hf_xxxx
"""

import argparse
import getpass
import os
import shutil
import sys
from pathlib import Path

HF_REPO   = "kaiwen2/prefvla-models"
HF_PREFIX = "starVLA"

RUNS = [
    "v0309_v30_ee_qwenPI_requeue",
    "v0309_v31_ee_qwenPI_requeue",
]

DEFAULT_DST = os.path.expanduser(
    "~/Desktop/research/starVLA/results/Checkpoints"
)


def get_token(cli_token: str = None) -> str:
    if cli_token:
        return cli_token
    for path in [
        os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
            "token",
        ),
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token"),
    ]:
        if os.path.isfile(path):
            token = Path(path).read_text().strip()
            if token:
                print(f"Using stored HF token from: {path}")
                return token
    print("No stored HuggingFace token found.")
    token = getpass.getpass("Enter HuggingFace token (hf_...): ").strip()
    if not token:
        print("ERROR: Token cannot be empty.")
        sys.exit(1)
    return token


def main():
    parser = argparse.ArgumentParser(
        description="Download StarVLA v30/v31 EE checkpoints from HuggingFace"
    )
    parser.add_argument(
        "--steps", nargs="+", type=int, default=[50000],
        help="Checkpoint steps to download (default: 50000)",
    )
    parser.add_argument(
        "--dst", type=str, default=DEFAULT_DST,
        help=f"Local destination directory (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="HuggingFace token (or leave blank to use stored/prompt)",
    )
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    token = get_token(args.token)
    dst_root = Path(args.dst)
    dst_root.mkdir(parents=True, exist_ok=True)

    # Build allow_patterns: meta files + requested checkpoint steps
    meta_patterns = [
        f"{HF_PREFIX}/{{run}}/dataset_statistics.json",
        f"{HF_PREFIX}/{{run}}/config.yaml",
    ]
    step_patterns = [
        f"{HF_PREFIX}/{{run}}/checkpoints/steps_{step}_pytorch_model.pt"
        for step in args.steps
    ]

    for run_name in RUNS:
        local_run_dir = dst_root / run_name
        print(f"\n{'='*60}")
        print(f"Run:   {run_name}")
        print(f"To:    {local_run_dir}")
        print(f"Steps: {args.steps}")
        print(f"{'='*60}", flush=True)

        # Skip files that already exist
        all_patterns = (
            [p.format(run=run_name) for p in meta_patterns]
            + [p.format(run=run_name) for p in step_patterns]
        )

        patterns_to_download = []
        for pat in all_patterns:
            # pat looks like "starVLA/run_name/checkpoints/steps_50000_pytorch_model.pt"
            rel = "/".join(pat.split("/")[2:])  # strip "starVLA/run_name/"
            local_file = local_run_dir / rel
            if local_file.exists():
                size_mb = local_file.stat().st_size / (1024 ** 2)
                print(f"  SKIP (exists, {size_mb:.0f} MB): {rel}")
            else:
                patterns_to_download.append(pat)

        if not patterns_to_download:
            print(f"  All files already present, skipping download.")
            continue

        print(f"  Downloading {len(patterns_to_download)} file(s) from HF...")
        print(f"  Patterns: {patterns_to_download}", flush=True)

        # snapshot_download caches to HF cache dir, returns path to cache
        cache_dir = snapshot_download(
            repo_id=HF_REPO,
            repo_type="model",
            token=token,
            allow_patterns=patterns_to_download,
            ignore_patterns=["**/train_state/**"],
        )

        # Copy from HF cache into the destination directory
        src_dir = Path(cache_dir) / HF_PREFIX / run_name
        if not src_dir.exists():
            print(f"  ERROR: expected cache dir not found: {src_dir}")
            continue

        local_run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_dir, local_run_dir, dirs_exist_ok=True)
        print(f"  Copied to: {local_run_dir}")

        # Print sizes
        for pat in patterns_to_download:
            rel = "/".join(pat.split("/")[2:])
            f = local_run_dir / rel
            if f.exists():
                size_mb = f.stat().st_size / (1024 ** 2)
                print(f"  OK: {rel} ({size_mb:.0f} MB)")
            else:
                print(f"  WARNING: {rel} not found after copy")

        print(f"Finished: {run_name}")

    print(f"\n{'='*60}")
    print("Download complete.")
    print(f"Checkpoints at: {dst_root}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
