#!/usr/bin/env bash
# Mirror the 0522 FastUMI checkpoints into the consolidated outsider86/DiscreteRTC repo
# under per-run subfolders (in addition to the standalone per-run repos already uploaded
# by upload_0522_to_hf.sh).
#
# Layout on the Hub will be:
#   outsider86/DiscreteRTC/
#     fastumi_airhockey_bounce_qwenPI_0522_DiT-S/
#       checkpoints/steps_20000_pytorch_model.pt
#       config.yaml
#       dataset_statistics.json
#       summary.jsonl
#     fastumi_pool_qwenPI_0522_DiT-S/
#       ...
#
# Prerequisites:
#   1. huggingface-cli login   (or set HF_TOKEN)
#   2. Optionally override DST_REPO

set -e

DST_REPO="${DST_REPO:-outsider86/DiscreteRTC}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINTS_ROOT="$(cd "$SCRIPT_DIR/../results/Checkpoints" && pwd)"

upload_one() {
  local name="$1"
  local local_path="${CHECKPOINTS_ROOT}/${name}"
  if [[ ! -d "$local_path" ]]; then
    echo "Missing: $local_path"
    return 1
  fi
  echo "Uploading $local_path -> ${DST_REPO}/${name}/"
  # Upload into ${DST_REPO} with path_in_repo = ${name}/ (third positional arg).
  huggingface-cli upload "$DST_REPO" "$local_path" "$name" --repo-type model \
    --include "checkpoints/steps_20000_pytorch_model.pt" \
    --include "config.yaml" \
    --include "dataset_statistics.json" \
    --include "summary.jsonl"
}

upload_one "fastumi_airhockey_bounce_qwenPI_0522_DiT-S"
upload_one "fastumi_pool_qwenPI_0522_DiT-S"

echo "Done."
echo "  https://huggingface.co/${DST_REPO}/tree/main/fastumi_airhockey_bounce_qwenPI_0522_DiT-S"
echo "  https://huggingface.co/${DST_REPO}/tree/main/fastumi_pool_qwenPI_0522_DiT-S"
