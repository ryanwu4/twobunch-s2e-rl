#!/bin/bash
# Validation of the realistic-operation policies (directions 1 noblen / 2 bc11loop, 2 seeds): waits for the RL chain,
# then per policy: standard 32-step single-setpoint eval (own env: B2-cov + DAQ injector context), continuous operation
# (own config) nominal + stress (drift x2, jumps p 0.05 / 0.6 deg), DAQ replay (2 scans); then closed-loop Bmad
# (golden + inj, 24 episodes, 2e4) of the best seed per direction (by settled-within-10-um in continuous operation).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
R=logs/2026-10-18/realistic
until grep -q "realistic RL done" logs/2026-10-18/realistic_chain.log 2>/dev/null; do
  kill -0 $(cat logs/2026-10-18/realistic_chain.pid) 2>/dev/null || { echo "RL chain gone"; break; }; sleep 300; done
echo "[$(date +%H:%M)] RL done"
EV=results/eval_realistic_2026-10-18; mkdir -p $EV
STRESS=(--set v4.drift_per_step.L1PhaseSet=0.07 --set v4.drift_per_step.L2PhaseSet=0.10 --set v4.drift_ramp.L1PhaseSet=0.08
        --set v4.drift_ramp.L2PhaseSet=0.07 --set 'v4.jumps={"L2PhaseSet": [0.05, 0.6]}')
one() {  # run gpu
  local L=$R/$1 g=$2 n=$1
  [ -e $L/final_policy.pt ] || { echo "$n: no final policy" >> $EV/summary.txt; return; }
  echo "$n | standard | $(scripts/ev_summary.sh cuda:$g --logdir $L --policy final --set episode_length=32 --set v4.goal_change=null --out $EV/${n}_standard 2>&1 | tail -n 1)" >> $EV/summary.txt
  echo "$n | continuous | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g --out $EV/${n}_cont 2>&1 | grep '^\[')" >> $EV/summary.txt
  echo "$n | continuous stress | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g "${STRESS[@]}" --out $EV/${n}_cont_stress 2>&1 | grep '^\[')" >> $EV/summary.txt
  for scan in TEST_15668 E300_15673; do
    echo "$n | replay | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $L --scan $scan --k 64 --device cuda:$g --out $EV/${n}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
}
(one real_covB2_screen_noblen_s0 0; one real_covB2_screen_noblen_s1 0) &
(one real_covB2_screen_bc11loop_s0 1; one real_covB2_screen_bc11loop_s1 1) &
wait
echo "[$(date +%H:%M)] evals done"
best() {  # kind -> best seed run name
  $PY - "$1" <<'PY'
import json, sys, glob
k = sys.argv[1]
c = [(json.load(open(f))["settled_within_10um"], f.split("/")[-2][:-5]) for f in glob.glob(f"results/eval_realistic_2026-10-18/real_covB2_screen_{k}_s*_cont/summary.json")]
print(max(c)[1] if c else "")
PY
}
OUT=results/closed_loop_bmad_2026-10-18; mkdir -p $OUT
cl() {  # run mode
  local L=$R/$1
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $L --mode $2 --plant flow --k 24 --device cuda:1 --out $OUT/$1_$2_flow > $OUT/$1_$2_flow.log 2>&1
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $L --mode $2 --plant bmad --k 24 --workers 24 --num-macro 20000 --device cuda:1 --out $OUT/$1_$2_bmad > $OUT/$1_$2_bmad.log 2>&1
  echo "[$(date +%H:%M)] $1 $2: flow $(tail -n 1 $OUT/$1_$2_flow.log) | bmad $(tail -n 1 $OUT/$1_$2_bmad.log)" | tee -a $OUT/summary.txt
}
BN=$(best noblen); BB=$(best bc11loop)
echo "[$(date +%H:%M)] closed loop on $BN, $BB"
[ -n "$BN" ] && { cl $BN golden & cl $BN inj & }
[ -n "$BB" ] && { cl $BB golden & cl $BB inj & }
wait
echo "[$(date +%H:%M)] validation chain done"
