#!/usr/bin/env bash
# Download StarVLA/RoboTwin-Randomized-targz dataset
# Uses REPO_ROOT if set; otherwise infers from script location (works on Kempner and local)

set -euo pipefail

# ── paths: REPO_ROOT (local) | LAB_ROOT/starVLA (Kempner) | script dir ───
if [ -n "${REPO_ROOT:-}" ]; then
  REPO="${REPO_ROOT}"
elif [ -n "${LAB_ROOT:-}" ]; then
  REPO="${LAB_ROOT}/starVLA"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
DEST_DIR="${REPO}/data/RoboTwin-Randomized-targz"
HF_CACHE="${REPO}/.cache/huggingface"

export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export TMPDIR="${REPO}/.tmp"

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
