#!/usr/bin/env bash
# Download StarVLA/RoboTwin-Randomized-targz dataset
# All data and caches stay under lab storage — nothing in ~

set -euo pipefail

# ── paths (never use home directory) ──────────────────────────────
LAB_ROOT="/net/holy-isilon/ifs/rc_labs/ydu_lab/Lab/haonan/kaiwen"
DEST_DIR="${LAB_ROOT}/starVLA/data/RoboTwin-Randomized-targz"
HF_CACHE="${LAB_ROOT}/.cache/huggingface"

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export TMPDIR="${LAB_ROOT}/.tmp"

mkdir -p "${DEST_DIR}" "${HF_HUB_CACHE}" "${TMPDIR}"

# ── dataset info ──────────────────────────────────────────────────
REPO_ID="StarVLA/RoboTwin-Randomized-targz"

echo "============================================"
echo " Downloading: ${REPO_ID}"
echo " Destination: ${DEST_DIR}"
echo " HF cache:    ${HF_HUB_CACHE}"
echo " Temp dir:    ${TMPDIR}"
echo "============================================"

huggingface-cli download \
    --repo-type dataset \
    --local-dir "${DEST_DIR}" \
    --cache-dir "${HF_HUB_CACHE}" \
    "${REPO_ID}"

echo ""
echo "Download complete. Files saved to: ${DEST_DIR}"
ls -lh "${DEST_DIR}"
