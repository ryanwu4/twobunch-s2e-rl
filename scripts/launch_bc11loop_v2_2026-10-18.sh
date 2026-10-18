#!/bin/bash
# bc11loop v2, 2 seeds (one per GPU) as soon as the s1 runs of the first round are done; then validation + cross-eval.
cd "$(dirname "$0")/.."
R=logs/2026-10-18/realistic
C=configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop_v2.yaml
until [ -e $R/real_covB2_screen_noblen_s1/final_policy.pt ] && [ -e $R/real_covB2_screen_bc11loop_s1/final_policy.pt ]; do sleep 60; done
echo "[$(date +%H:%M)] round-1 s1 done; launching bc11loop v2"
(SEED=0 bash scripts/run_queue.sh 0 $R $C) > $R/v2_gpu0.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 1 $R $C) > $R/v2_gpu1.log 2>&1 &
wait
echo "[$(date +%H:%M)] bc11loop v2 done"
