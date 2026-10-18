#!/bin/bash
# Jitter robustness (2026-10-18 afternoon): (1) the current realistic policies (+ the old blen_dlog baseline) in
# continuous operation under the DAQ-calibrated jitter (nominal) and the 2024-sheet stress condition; (2) jitter-trained
# versions of both directions (2024 map, 1.5x wander; 2 seeds, 900 it), then the same evals + the DAQ replay.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
R=logs/2026-10-18/realistic
EV=results/eval_jitter_2026-10-18; mkdir -p $EV
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml; STR=configs/rl/eval/jitter_stress_2026-10-18.yaml
JN=configs/rl/realistic_2026-10-17/real_covB2_screen_noblen_jit.yaml
JB=configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop_v3_jit.yaml
log() { echo "[$(date +%H:%M)] $*"; }
ev() {  # logdir name gpu
  for c in nominal stress; do
    y=$NOM; [ $c = stress ] && y=$STR
    echo "$2 | $c | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $1 --device cuda:$3 --de-yaml $y --out $EV/$2_$c 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
}
(SEED=0 bash scripts/run_queue.sh 0 $R $JN) > $R/jit_gpu0.log 2>&1 &
(SEED=0 bash scripts/run_queue.sh 1 $R $JB) > $R/jit_gpu1.log 2>&1 &
( ev logs/2026-10-16/injflow/injflow_flowA_screen_v6_blen_s0 old_blen_dlog 0
  ev $R/real_covB2_screen_noblen_s0 noblen_s0 0; ev $R/real_covB2_screen_noblen_s1 noblen_s1 0 ) & E1=$!
( ev $R/real_covB2_screen_bc11loop_v3_s0 bc11loop_v3_s0 1; ev $R/real_covB2_screen_bc11loop_v3_s1 bc11loop_v3_s1 1 ) & E2=$!
wait $E1 $E2
log "current-policy evals done"
(SEED=1 bash scripts/run_queue.sh 0 $R $JN) > $R/jit_gpu0_s1.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 1 $R $JB) > $R/jit_gpu1_s1.log 2>&1 &
wait
log "jitter training done"
for spec in "real_covB2_screen_noblen_jit_s0:0" "real_covB2_screen_noblen_jit_s1:0" "real_covB2_screen_bc11loop_v3_jit_s0:1" "real_covB2_screen_bc11loop_v3_jit_s1:1"; do
  n=${spec%%:*}; g=${spec#*:}
  [ -e $R/$n/final_policy.pt ] || { echo "$n: no final policy" >> $EV/summary.txt; continue; }
  ev $R/$n $n $g
  for scan in TEST_15668 E300_15673; do
    echo "$n | replay $scan | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $R/$n --scan $scan --k 64 --device cuda:$g --out $EV/${n}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
done
log "jitter robustness chain done"
