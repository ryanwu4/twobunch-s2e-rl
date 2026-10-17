#!/bin/bash
# Closed-loop Bmad check (Ryan 10-16: golden + injector-varied). Waits for the injector coverage campaign to free the
# CPUs, then 4 runs in parallel (24 episodes x 24 single-thread workers each, 2e4 macroparticles):
#   S2 blen_dlog (flow A injflow s0) and S2 direct-L1 + BLEN (flow A l1blen s0), each golden + inj (DAQ-distribution
#   injector points of the coverage campaign); paired surrogate runs (--plant flow) for each.
# Then, once the B2-cov RL chain is done: the B2-cov direct-L1 + BLEN S2 s0 policy, golden + inj.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
OUT=results/closed_loop_bmad_2026-10-17
mkdir -p $OUT
until grep -q "^Done:" logs/injector_cov_2026-10-16/run.log; do
  kill -0 $(cat logs/injector_cov_2026-10-16/run.pid) 2>/dev/null || { echo "campaign gone without Done"; break; }; sleep 120; done
echo "[$(date +%H:%M)] CPUs free"
cl() {  # logdir tag mode
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir "$1" --mode "$3" --plant flow \
    --k 24 --device cuda:1 --out $OUT/$2_$3_flow > $OUT/$2_$3_flow.log 2>&1
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir "$1" --mode "$3" --plant bmad \
    --k 24 --workers 24 --num-macro 20000 --device cuda:1 --out $OUT/$2_$3_bmad > $OUT/$2_$3_bmad.log 2>&1
  echo "[$(date +%H:%M)] $2 $3: flow $(tail -n 1 $OUT/$2_$3_flow.log) | bmad $(tail -n 1 $OUT/$2_$3_bmad.log)"
}
A=logs/2026-10-16/injflow/injflow_flowA_screen_v6_blen_s0
B=logs/2026-10-16/l1blen/injflow_flowA_screen_v6_l1blen_s0
cl $A S2blen_flowA golden & cl $A S2blen_flowA inj & cl $B S2l1blen_flowA golden & cl $B S2l1blen_flowA inj &
wait
echo "[$(date +%H:%M)] flow-A closed loop done"
until grep -q "covB2 RL chain done" logs/2026-10-17/covB2_chain.log 2>/dev/null; do
  kill -0 $(cat logs/2026-10-17/covB2_chain.pid) 2>/dev/null || { echo "covB2 chain gone"; exit 1; }; sleep 300; done
C=logs/2026-10-17/covB2/injflow_covB2_screen_v6_l1blen_s0
cl $C S2l1blen_covB2 golden & cl $C S2l1blen_covB2 inj &
wait
echo "[$(date +%H:%M)] closed-loop chain done"
