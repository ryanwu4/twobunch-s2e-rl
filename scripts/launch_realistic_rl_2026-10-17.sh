#!/bin/bash
# Realistic-operation RL (directions 1 and 2) on B2-cov, 2 seeds each, 1200 iterations; ONE run per GPU at a time
# (~30 GB each; 2 per GPU next to a fine-tune ran out of memory 19:02): GPU0 noblen s0 -> bc11loop s1, GPU1
# bc11loop s0 -> noblen s1.
cd "$(dirname "$0")/.."
R=logs/2026-10-18/realistic
mkdir -p $R
N=configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml
B=configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop.yaml
( (SEED=0 bash scripts/run_queue.sh 0 $R $N); (SEED=1 bash scripts/run_queue.sh 0 $R $B) ) > $R/gpu0.log 2>&1 &
( (SEED=0 bash scripts/run_queue.sh 1 $R $B); (SEED=1 bash scripts/run_queue.sh 1 $R $N) ) > $R/gpu1.log 2>&1 &
wait
echo "[$(date +%H:%M)] realistic RL done"
