#!/bin/bash
# Exploratory direct-L1 + BLEN-obs runs on flow A (seed 0) before the B2-cov retrain: S1 cloud on GPU0, S2 v6 screen on
# GPU1, each followed by its eval (own env = flow A + DAQ injector context; compare the injflow table's "inj" column).
cd "$(dirname "$0")/.."
R=logs/2026-10-16/l1blen
mkdir -p $R
run() {  # gpu cfg
  local n; n=$(basename "$2" .yaml)
  bash scripts/run_queue.sh "$1" $R "$2"
  echo "$n | own | $(scripts/ev_summary.sh cuda:$1 --logdir $R/${n}_s0 --policy final --out $R/${n}_s0/delta_eval_final 2>&1 | tail -n 1)" >> $R/summary.txt
}
run 0 configs/rl/l1blen_2026-10-16/injflow_flowA_cloud_l1blen.yaml > $R/gpu0.log 2>&1 &
run 1 configs/rl/l1blen_2026-10-16/injflow_flowA_screen_v6_l1blen.yaml > $R/gpu1.log 2>&1 &
wait
