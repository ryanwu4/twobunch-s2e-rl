#!/bin/bash
# L2-only DAQ replay (TEST_15668, E300_15673; 64 episodes) of each S2 policy as soon as its final policy exists.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
OUT=results/daq_replay_2026-10-17
mkdir -p $OUT
RUNS=(logs/2026-10-16/injflow/injflow_flowA_screen_v6_blen_s0 logs/2026-10-16/delta_xtcav23_screen_v6_blen_s0
      logs/2026-10-16/l1blen/injflow_flowA_screen_v6_l1blen_s0
      logs/2026-10-17/covB2/injflow_covB2_screen_v6_l1blen_s0 logs/2026-10-17/covB2/injflow_covB2_screen_v6_l1blen_s1)
for L in "${RUNS[@]}"; do
  n=$(basename $L)
  until [ -e $L/final_policy.pt ]; do sleep 300; done
  for scan in TEST_15668 E300_15673; do
    [ -e $OUT/${n}_$scan/summary.json ] && continue
    line=$(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $L --scan $scan --k 64 \
           --device cuda:1 --out $OUT/${n}_$scan 2>&1 | grep "^\[" | tail -n 1)
    echo "$n | $line" | tee -a $OUT/summary.txt
  done
done
echo "[$(date +%H:%M)] DAQ replay chain done"
