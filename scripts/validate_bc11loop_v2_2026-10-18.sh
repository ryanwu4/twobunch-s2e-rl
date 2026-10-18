#!/bin/bash
# bc11loop v2: deadline 07:30 (newest checkpoint if unfinished), then the same validation as the orchestrator
# (standard 32-step eval, continuous nominal + stress, DAQ replay) + the standard cross-eval on B2-cov + spacing head.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
R=logs/2026-10-18/realistic
EV=results/eval_realistic_2026-10-18
ZS=$(ls trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/*.ckpt)
STRESS=(--set v4.drift_per_step.L1PhaseSet=0.07 --set v4.drift_per_step.L2PhaseSet=0.10 --set v4.drift_ramp.L1PhaseSet=0.08
        --set v4.drift_ramp.L2PhaseSet=0.07 --set 'v4.jumps={"L2PhaseSet": [0.05, 0.6]}')
bash scripts/deadline_2026-10-18.sh 07:30 real_covB2_screen_bc11loop_v2_s0 real_covB2_screen_bc11loop_v2_s1
v() {
  local n=$1 g=$2 L=$R/$1
  until [ -e $L/final_policy.pt ]; do sleep 60; done
  echo "$n | standard | $(scripts/ev_summary.sh cuda:$g --logdir $L --policy final --set episode_length=32 --set v4.goal_change=null --out $EV/${n}_standard 2>&1 | tail -n 1)" >> $EV/summary.txt
  echo "$n | continuous | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g --out $EV/${n}_cont 2>&1 | grep '^\[')" >> $EV/summary.txt
  echo "$n | continuous stress | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g "${STRESS[@]}" --out $EV/${n}_cont_stress 2>&1 | grep '^\[')" >> $EV/summary.txt
  for scan in TEST_15668 E300_15673; do
    echo "$n | replay $scan | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $L --scan $scan --k 64 --device cuda:$g --out $EV/${n}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
  echo "$n | standard on B2-cov+spacing head | $(scripts/ev_summary.sh cuda:$g --logdir $L --policy final --set episode_length=32 --set v4.goal_change=null --flow-ckpt $ZS --de-yaml configs/rl/eval/injector_ctx_2026-10-16.yaml --out $EV/${n}_standard_zsflow 2>&1 | tail -n 1)" >> $EV/summary.txt
  echo "[$(date +%H:%M)] validated $n"
}
v real_covB2_screen_bc11loop_v2_s0 0 & v real_covB2_screen_bc11loop_v2_s1 1 & wait
echo "[$(date +%H:%M)] v2 validation done"
