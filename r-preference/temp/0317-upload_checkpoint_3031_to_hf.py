#!/usr/bin/env python3
"""
Upload StarVLA v30 and v31 EE checkpoints to HuggingFace.

Runs on Kempner. Uploads to: kaiwen2/prefvla-models/starVLA/{run_name}/

Files uploaded per run (default: steps_50000 only):
  - dataset_statistics.json  (needed by ModelClient for action unnorm)
  - config.yaml              (needed by read_mode_config for chunk size)
  - checkpoints/steps_{N}_pytorch_model.pt

Usage:
  python 0317-upload_checkpoint_3031_to_hf.py
  python 0317-upload_checkpoint_3031_to_hf.py --steps 10000 50000
  python 0317-upload_checkpoint_3031_to_hf.py --token hf_xxxx
"""

import argparse
import getpass
import os
import sys
from pathlib import Path

KEMPNER_ROOT = "/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA"
CKPT_ROOT = Path(KEMPNER_ROOT) / "results" / "Checkpoints"

HF_REPO   = "kaiwen2/prefvla-models"
HF_PREFIX = "starVLA"

RUNS = [
    "v0309_v30_ee_qwenPI_requeue",
    "v0309_v31_ee_qwenPI_requeue",
]

META_FILES = [
    "dataset_statistics.json",
    "config.yaml",
]


def get_token(cli_token: str = None) -> str:
    if cli_token:
        return cli_token
    for path in [
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "token"),
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


def upload_file(api, local_path: Path, path_in_repo: str) -> None:
    size_mb = local_path.stat().st_size / (1024 ** 2)
    print(f"  Uploading {local_path.name} ({size_mb:.1f} MB) -> {path_in_repo}", flush=True)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=HF_REPO,
        repo_type="model",
    )
    print(f"  Done: {local_path.name}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Upload StarVLA v30/v31 EE checkpoints to HuggingFace")
    parser.add_argument("--steps", nargs="+", type=int, default=[50000],
                        help="Checkpoint steps to upload (default: 50000)")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or leave blank to use stored/prompt)")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    token = get_token(args.token)
    api = HfApi(token=token)

    # Verify repo access
    try:
        api.repo_info(repo_id=HF_REPO, repo_type="model")
        print(f"Repo '{HF_REPO}' accessible.")
    except Exception as e:
        print(f"ERROR: Cannot access '{HF_REPO}': {e}")
        sys.exit(1)

    for run_name in RUNS:
        run_dir = CKPT_ROOT / run_name
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        print(f"From: {run_dir}")
        print(f"Steps: {args.steps}")
        print(f"{'='*60}")

        if not run_dir.exists():
            print(f"  ERROR: {run_dir} does not exist. Skipping.")
            continue

        # Upload meta files
        for rel in META_FILES:
            local = run_dir / rel
            if not local.exists():
                print(f"  SKIP (not found): {rel}")
                continue
            upload_file(api, local, f"{HF_PREFIX}/{run_name}/{rel}")

        # Upload checkpoint steps
        for step in args.steps:
            fname = f"steps_{step}_pytorch_model.pt"
            local = run_dir / "checkpoints" / fname
            if not local.exists():
                print(f"  SKIP (not found): checkpoints/{fname}")
                continue
            upload_file(api, local, f"{HF_PREFIX}/{run_name}/checkpoints/{fname}")

        print(f"Finished: {run_name}")

    print(f"\n{'='*60}")
    print("All uploads complete.")
    print(f"View at: https://huggingface.co/{HF_REPO}/tree/main/{HF_PREFIX}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
