#!/usr/bin/env bash
# Serial pool training queue: kick (10k) -> direct (10k), all 8 H100. 2026-06-05.
# Self-contained env so it runs from a non-interactive shell (no `conda activate` needed).
# kick failure aborts the queue (won't waste GPUs training direct on a broken setup).
set -u
ENV=/mnt/localssd/kaiwenh/miniconda3/envs/starVLA
export CONDA_PREFIX="$ENV"
export PATH="$ENV/bin:$PATH"
cd /home/kaiwenh/starVLA

K=realworld/log_pool_kick_0520.log
D=realworld/log_pool_direct_0520.log

echo "[queue] $(date) START kick -> $K"
bash realworld/0520-pool-kick.sh > "$K" 2>&1
rc=$?
echo "[queue] $(date) kick exited rc=$rc"
[ $rc -ne 0 ] && { echo "[queue] kick FAILED rc=$rc -- aborting, not starting direct"; exit $rc; }

echo "[queue] $(date) START direct -> $D"
bash realworld/0520-pool-direct.sh > "$D" 2>&1
rc=$?
echo "[queue] $(date) direct exited rc=$rc"
echo "[queue] $(date) QUEUE DONE (rc=$rc)"
