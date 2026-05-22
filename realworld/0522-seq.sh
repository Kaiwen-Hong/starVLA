#!/bin/bash
# Sequential training: run bounce first, then pool-static.
# If the first job fails, the second does NOT start (set -e is inherited).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================================"
echo "[seq] Job 1/2: 3airhockey_dynamic_bounce"
echo "========================================================"
bash "$SCRIPT_DIR/0522-dynamic-air-hockey-bounce-openpi-tc.sh"

echo
echo "========================================================"
echo "[seq] Job 1/2 DONE. Starting Job 2/2: pool_strike_combined"
echo "========================================================"
bash "$SCRIPT_DIR/0522-pool-static-openpi-tc.sh"

echo
echo "========================================================"
echo "[seq] All jobs completed."
echo "========================================================"
