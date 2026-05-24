#!/usr/bin/env bash
# Upload 0522 FastUMI checkpoints to Hugging Face.
# Prerequisites:
#   1. Login: huggingface-cli login   (or set HF_TOKEN)
#   2. Optionally override HF_REPO_ORG (defaults to outsider86)

set -e

REPO_ORG="${HF_REPO_ORG:-outsider86}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_ROOT="$(cd "$SCRIPT_DIR/../results/Checkpoints" && pwd)"

upload_one() {
  local name="$1"
  local repo_id="${REPO_ORG}/${name}"
  local local_path="${CHECKPOINTS_ROOT}/${name}"
  if [[ ! -d "$local_path" ]]; then
    echo "Missing: $local_path"
    return 1
  fi
  echo "Uploading $local_path -> $repo_id"
  huggingface-cli upload "$repo_id" "$local_path" . --repo-type model \
    --include "checkpoints/steps_20000_pytorch_model.pt" \
    --include "config.yaml" \
    --include "dataset_statistics.json" \
    --include "summary.jsonl"
}

upload_one "fastumi_airhockey_bounce_qwenPI_0522_DiT-S"
upload_one "fastumi_pool_qwenPI_0522_DiT-S"

echo "Done."
echo "  https://huggingface.co/${REPO_ORG}/fastumi_airhockey_bounce_qwenPI_0522_DiT-S"
echo "  https://huggingface.co/${REPO_ORG}/fastumi_pool_qwenPI_0522_DiT-S"
