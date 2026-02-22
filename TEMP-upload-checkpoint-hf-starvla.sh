#!/bin/bash
# =============================================================================
# Upload StarVLA checkpoints to HuggingFace
# Repo : kaiwen2/prefvla-models  (folder: starvla/)
#
# Uploads per run:
#   - config.yaml, dataset_statistics.json, summary.jsonl
#   - checkpoints/steps_<N>_pytorch_model.pt  (~9.2 GB each)
#   - Does NOT upload optimizer/scheduler states (steps_<N>/ directories)
#
# Default: upload only the LATEST checkpoint per run (~9.2 GB × 4 ≈ 37 GB total)
# To upload ALL checkpoints (~267 GB total), change LATEST_ONLY to false below.
# =============================================================================

set -euo pipefail

cd /net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA

# ── Config (edit here if needed) ─────────────────────────────────────────────
export HF_REPO="kaiwen2/prefvla-models"
export HF_BASE="results/Checkpoints"
export HF_LATEST_ONLY="true"   # "true" = only latest .pt per run; "false" = all
export HF_RUNS="robotwin_qwenOFT_4xH100 custom_qwenOFT_requeue custom_v0218_qwenOFT_h100-0 custom_v0218_qwenOFT_h100"
# ─────────────────────────────────────────────────────────────────────────────

echo "============================================"
echo " Upload StarVLA checkpoints → HuggingFace"
echo " Repo  : $HF_REPO"
echo " Folder: starvla/"
echo " Mode  : LATEST_ONLY=$HF_LATEST_ONLY"
echo "============================================"
echo ""
echo "Please enter your HuggingFace token (write access required, input hidden):"
read -s HF_TOKEN
echo ""
export HF_TOKEN

echo "Logging in to HuggingFace..."
huggingface-cli login --token "$HF_TOKEN" 2>&1 | tail -1
echo ""
echo "Starting upload..."
echo ""

# ── Upload (pure Python, reads all config from env vars) ─────────────────────
python3 <<'PYEOF'
import os
import sys
from huggingface_hub import HfApi

token       = os.environ["HF_TOKEN"]
repo_id     = os.environ["HF_REPO"]
base        = os.environ["HF_BASE"]
latest_only = os.environ["HF_LATEST_ONLY"].lower() == "true"
run_list    = os.environ["HF_RUNS"].split()

api = HfApi(token=token)

# Ensure repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type="model")
    print(f"Repo '{repo_id}' found.")
except Exception:
    print(f"Repo '{repo_id}' not found — creating...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False)
    print("Created.")

for run_id in run_list:
    run_dir   = f"{base}/{run_id}"
    ckpt_dir  = f"{run_dir}/checkpoints"
    hf_prefix = f"starvla/{run_id}"

    print(f"\n{'='*60}")
    print(f"Run: {run_id}")
    print(f"{'='*60}")

    if not os.path.isdir(run_dir):
        print(f"  WARNING: {run_dir} not found — skipping.")
        continue

    # -- Metadata files -------------------------------------------------------
    for fname in ["config.yaml", "dataset_statistics.json", "summary.jsonl"]:
        fpath = f"{run_dir}/{fname}"
        if os.path.exists(fpath):
            print(f"  Uploading {fname} ...")
            api.upload_file(
                path_or_fileobj=fpath,
                path_in_repo=f"{hf_prefix}/{fname}",
                repo_id=repo_id,
                repo_type="model",
            )

    # -- Model weights (.pt files only) ---------------------------------------
    if not os.path.isdir(ckpt_dir):
        print(f"  WARNING: no checkpoints/ dir — skipping weights.")
        continue

    pt_files = sorted(
        f for f in os.listdir(ckpt_dir) if f.endswith("_pytorch_model.pt")
    )

    if not pt_files:
        print(f"  WARNING: no *_pytorch_model.pt files found.")
        continue

    if latest_only:
        pt_files = [pt_files[-1]]
        print(f"  Mode: LATEST_ONLY → uploading only: {pt_files[0]}")
    else:
        print(f"  Mode: ALL → uploading {len(pt_files)} checkpoint(s)")

    for fname in pt_files:
        fpath    = f"{ckpt_dir}/{fname}"
        size_gb  = os.path.getsize(fpath) / 1e9
        print(f"  Uploading checkpoints/{fname}  ({size_gb:.1f} GB) ...")
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=f"{hf_prefix}/checkpoints/{fname}",
            repo_id=repo_id,
            repo_type="model",
        )
        print(f"    Done.")

print(f"\n{'='*60}")
print(f"All uploads complete!")
print(f"  https://huggingface.co/{repo_id}/tree/main/starvla")
print(f"{'='*60}")
PYEOF
