#!/usr/bin/env bash
# Upload pick-and-place checkpoints to Hugging Face.
# Prerequisites:
#   1. Login: huggingface-cli login   (or set HF_TOKEN)
#   2. Set HF_REPO_ORG to your Hugging Face username or org (e.g. StarVLA)

set -e

REPO_ORG="${HF_REPO_ORG:-}"
if [[ -z "$REPO_ORG" ]]; then
  echo "Set your Hugging Face org/username: export HF_REPO_ORG=your_username"
  echo "Example: export HF_REPO_ORG=StarVLA"
  exit 1
fi

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
  huggingface-cli upload "$repo_id" "$local_path" . --repo-type model
}

upload_one "fastumi_pickandplace_discrete_diffusion_real_0314"
upload_one "fastumi_pickandplace_qwenPI"

echo "Done. Repos: https://huggingface.co/${REPO_ORG}/fastumi_pickandplace_discrete_diffusion_real_0314 and .../fastumi_pickandplace_qwenPI"
