#!/usr/bin/env bash
# Upload v41, v42, v52 final checkpoints to kaiwen2/prefvla-models
set -euo pipefail

STAR_PYTHON="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/miniforge3/envs/starVLA/bin/python"
RESULTS="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen/starVLA/results/Checkpoints"
HF_REPO="kaiwen2/prefvla-models"

declare -A MODELS=(
  ["v0320_v41_qwenOFT_finetune_v2"]="steps_60000_pytorch_model.pt"
  ["v0320_v42_qwenOFT_finetune_v2"]="steps_60000_pytorch_model.pt"
  ["v0320_v52_qwenOFT_finetune_v1"]="steps_60000_pytorch_model.pt"
)

for model_dir in "${!MODELS[@]}"; do
  ckpt_name="${MODELS[$model_dir]}"
  local_dir="${RESULTS}/${model_dir}"
  hf_prefix="starvla/${model_dir}"

  echo ""
  echo "========================================"
  echo "Uploading: ${model_dir}"
  echo "========================================"

  # Upload checkpoint
  echo "  [1/4] Uploading checkpoint (9.2GB)..."
  $STAR_PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='${local_dir}/checkpoints/${ckpt_name}',
    path_in_repo='${hf_prefix}/checkpoints/${ckpt_name}',
    repo_id='${HF_REPO}',
)
print('  checkpoint uploaded.')
"

  # Upload config.yaml
  echo "  [2/4] Uploading config.yaml..."
  $STAR_PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='${local_dir}/config.yaml',
    path_in_repo='${hf_prefix}/config.yaml',
    repo_id='${HF_REPO}',
)
print('  config.yaml uploaded.')
"

  # Upload dataset_statistics.json
  echo "  [3/4] Uploading dataset_statistics.json..."
  $STAR_PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='${local_dir}/dataset_statistics.json',
    path_in_repo='${hf_prefix}/dataset_statistics.json',
    repo_id='${HF_REPO}',
)
print('  dataset_statistics.json uploaded.')
"

  # Upload summary.jsonl
  echo "  [4/4] Uploading summary.jsonl..."
  $STAR_PYTHON -c "
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj='${local_dir}/summary.jsonl',
    path_in_repo='${hf_prefix}/summary.jsonl',
    repo_id='${HF_REPO}',
)
print('  summary.jsonl uploaded.')
"

  echo "  DONE: ${model_dir}"
done

echo ""
echo "========================================"
echo "ALL UPLOADS COMPLETE"
echo "========================================"
echo "Uploaded to: https://huggingface.co/${HF_REPO}"
echo ""
echo "On your 4090 machine, download with:"
echo "  huggingface-cli download ${HF_REPO} --include 'starvla/v0320_v4*' 'starvla/v0320_v5*' --local-dir ./hf_models"
