#!/usr/bin/env bash
# ============================================================
# Upload evaluation results to HuggingFace dataset
#
# Uploads to: kaiwen2/robotwin-prefvla / 0402-analysis/
#
# Contents uploaded:
#   - _result.txt and _episode_data.json for each eval config
#   - Eval logs (server + eval)
#   - Episode videos (.mp4)
#
# Usage:
#   conda activate starVLA  # or any env with huggingface_hub
#   bash r-preference/eval-local-v6162-vast/upload_results.sh
# ============================================================
set -euo pipefail

HF_REPO="kaiwen2/robotwin-prefvla"
HF_FOLDER="0402-analysis"
REPO_TYPE="dataset"

AR_ROOT="${AR_ROOT:-/home/user/ar-research-kempner}"
STARVLA_ROOT="${STARVLA_ROOT:-/home/user/starVLA}"
EVAL_RESULTS_ROOT="${AR_ROOT}/policy/pi05_ee/eval_logs"
EVAL_LOGS_DIR=$(ls -td "${STARVLA_ROOT}/results/eval_logs/v6162_vast_seed0_step60000_"* 2>/dev/null | head -1)

if [ -z "$EVAL_LOGS_DIR" ]; then
    echo "ERROR: No eval log directory found"
    exit 1
fi

echo "========================================"
echo "Uploading results to HuggingFace"
echo "  Repo:        ${HF_REPO}"
echo "  Folder:      ${HF_FOLDER}"
echo "  Eval logs:   ${EVAL_LOGS_DIR}"
echo "  Eval results: ${EVAL_RESULTS_ROOT}"
echo "========================================"

# Upload using huggingface_hub Python API
python3 << 'PYEOF'
import os
from huggingface_hub import HfApi

api = HfApi()

HF_REPO = os.environ.get("HF_REPO", "kaiwen2/robotwin-prefvla")
HF_FOLDER = os.environ.get("HF_FOLDER", "0402-analysis")
REPO_TYPE = "dataset"

AR_ROOT = os.environ.get("AR_ROOT", "/home/user/ar-research-kempner")
STARVLA_ROOT = os.environ.get("STARVLA_ROOT", "/home/user/starVLA")
EVAL_LOGS_DIR = os.environ.get("EVAL_LOGS_DIR", "")
EVAL_RESULTS_ROOT = f"{AR_ROOT}/policy/pi05_ee/eval_logs"

# Define what to upload
configs = {
    "v61_clean": f"{EVAL_RESULTS_ROOT}/exp-idxv61w-starvla/eval_results/place_stapler_stand/place_stapler_stand_clean1",
    "v61_wp4":   f"{EVAL_RESULTS_ROOT}/exp-idxv61w-starvla/eval_results/place_stapler_stand/place_stapler_stand_wp4",
    "v62_clean": f"{EVAL_RESULTS_ROOT}/exp-idxv62w-starvla/eval_results/place_stapler_stand/place_stapler_stand_clean1",
    "v62_wp5":   f"{EVAL_RESULTS_ROOT}/exp-idxv62w-starvla/eval_results/place_stapler_stand/place_stapler_stand_wp5",
}

uploaded = 0

# 1. Upload eval result files (_result.txt, _episode_data.json, videos)
for label, local_dir in configs.items():
    if not os.path.isdir(local_dir):
        print(f"  SKIP: {label} — directory not found: {local_dir}")
        continue

    for fname in sorted(os.listdir(local_dir)):
        local_path = os.path.join(local_dir, fname)
        if not os.path.isfile(local_path):
            continue
        remote_path = f"{HF_FOLDER}/{label}/{fname}"
        print(f"  Uploading: {remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO,
            repo_type=REPO_TYPE,
        )
        uploaded += 1

# 2. Upload eval logs (server + eval logs)
if EVAL_LOGS_DIR and os.path.isdir(EVAL_LOGS_DIR):
    log_dir_name = os.path.basename(EVAL_LOGS_DIR)
    for fname in sorted(os.listdir(EVAL_LOGS_DIR)):
        local_path = os.path.join(EVAL_LOGS_DIR, fname)
        if not os.path.isfile(local_path):
            continue
        remote_path = f"{HF_FOLDER}/logs/{fname}"
        print(f"  Uploading: {remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=HF_REPO,
            repo_type=REPO_TYPE,
        )
        uploaded += 1

print(f"\nDone! Uploaded {uploaded} files to {HF_REPO}/{HF_FOLDER}/")
PYEOF
