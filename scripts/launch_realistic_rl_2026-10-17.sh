#!/bin/bash
# Realistic-operation RL (directions 1 and 2) on B2-cov, 2 seeds each, 1200 iterations; two runs per GPU in parallel.
cd "$(dirname "$0")/.."
R=logs/2026-10-18/realistic
mkdir -p $R
N=configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml
B=configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop.yaml
(SEED=0 bash scripts/run_queue.sh 0 $R $N) > $R/q0.log 2>&1 &
(SEED=0 bash scripts/run_queue.sh 1 $R $B) > $R/q1.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 0 $R $B) > $R/q2.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 1 $R $N) > $R/q3.log 2>&1 &
wait
echo "[$(date +%H:%M)] realistic RL done"
