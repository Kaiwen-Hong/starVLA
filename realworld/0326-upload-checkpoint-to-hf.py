#!/usr/bin/env python3
"""Upload the latest dynamic_first_0324 checkpoint to HF repo kaiwen2/discreteRTC under folder 0325."""

import os
from huggingface_hub import HfApi

REPO_ID = "kaiwen2/discreteRTC"
EXPERIMENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "Checkpoints", "dynamic_first_0324",
)
CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "checkpoints")
HF_FOLDER = "0325"
EXTRA_FILES = ["config.yaml", "dataset_statistics.json"]

def get_latest_checkpoint(checkpoint_dir):
    """Find the checkpoint with the highest step number."""
    import re
    pattern = re.compile(r"steps_(\d+)_pytorch_model\.pt$")
    best = None
    best_step = -1
    for f in os.listdir(checkpoint_dir):
        m = pattern.match(f)
        if m:
            step = int(m.group(1))
            if step > best_step:
                best_step = step
                best = f
    if best is None:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return best, best_step

def main():
    token = input("Enter your HF token: ").strip()
    if not token:
        print("No token provided, aborting.")
        return

    experiment_dir = os.path.normpath(EXPERIMENT_DIR)
    checkpoint_dir = os.path.normpath(CHECKPOINT_DIR)
    filename, step = get_latest_checkpoint(checkpoint_dir)
    local_path = os.path.join(checkpoint_dir, filename)

    file_size_gb = os.path.getsize(local_path) / (1024 ** 3)
    print(f"Latest checkpoint: {filename} (step {step})")
    print(f"Local path:        {local_path}")
    print(f"File size:         {file_size_gb:.2f} GB")
    print(f"Uploading to:      {REPO_ID}/{HF_FOLDER}/{filename}")
    for ef in EXTRA_FILES:
        ef_path = os.path.join(experiment_dir, ef)
        if os.path.exists(ef_path):
            print(f"Extra file:        {ef}")
        else:
            print(f"WARNING: missing extra file: {ef_path}")
    print("This may take a while for large files...")

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload using upload_folder for progress bar support
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Symlink the checkpoint into a temp dir matching the HF folder structure
        folder_path = os.path.join(tmp_dir, HF_FOLDER)
        os.makedirs(folder_path)
        os.symlink(local_path, os.path.join(folder_path, filename))
        for ef in EXTRA_FILES:
            ef_path = os.path.join(experiment_dir, ef)
            if os.path.exists(ef_path):
                os.symlink(ef_path, os.path.join(folder_path, ef))

        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=REPO_ID,
            repo_type="model",
        )

    print(f"Done. See: https://huggingface.co/{REPO_ID}/tree/main/{HF_FOLDER}")

if __name__ == "__main__":
    main()
