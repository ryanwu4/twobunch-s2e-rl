#!/bin/bash
# Overnight 10-18 orchestration (replaces launch_realistic_rl / validate_realistic after the 19:45 reshuffle):
#  - now: bc11loop s0 on GPU1 (noblen s0 already training on GPU0)
#  - when the centroid fine-tunes D1 / D2 end: their scorecards (vs B2-cov), then noblen s1 (GPU0) + bc11loop s1 (GPU1)
#  - every policy as soon as its final policy exists: standard 32-step eval, continuous operation nominal + stress,
#    DAQ replay (2 scans); the s0 policies then closed-loop Bmad golden + inj (24 episodes, 2e4)
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
R=logs/2026-10-18/realistic
N=configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml
B=configs/rl/realistic_2026-10-17/real_covB2_screen_bc11loop.yaml
EV=results/eval_realistic_2026-10-18; CL=results/closed_loop_bmad_2026-10-18; mkdir -p $EV $CL
STRESS=(--set v4.drift_per_step.L1PhaseSet=0.07 --set v4.drift_per_step.L2PhaseSet=0.10 --set v4.drift_ramp.L1PhaseSet=0.08
        --set v4.drift_ramp.L2PhaseSet=0.07 --set 'v4.jumps={"L2PhaseSet": [0.05, 0.6]}')
log() { echo "[$(date +%H:%M)] $*"; }
validate() {  # run gpu closedloop(0/1)
  local n=$1 g=$2 L=$R/$1
  until [ -e $L/final_policy.pt ]; do [ -e $L ] || { sleep 60; continue; }; sleep 120; done
  log "validate $n"
  echo "$n | standard | $(scripts/ev_summary.sh cuda:$g --logdir $L --policy final --set episode_length=32 --set v4.goal_change=null --out $EV/${n}_standard 2>&1 | tail -n 1)" >> $EV/summary.txt
  echo "$n | continuous | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g --out $EV/${n}_cont 2>&1 | grep '^\[')" >> $EV/summary.txt
  echo "$n | continuous stress | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $L --device cuda:$g "${STRESS[@]}" --out $EV/${n}_cont_stress 2>&1 | grep '^\[')" >> $EV/summary.txt
  for scan in TEST_15668 E300_15673; do
    echo "$n | replay $scan | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $L --scan $scan --k 64 --device cuda:$g --out $EV/${n}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
  log "validated $n"
  if [ "$3" = 1 ]; then
    for mode in golden inj; do
      ( PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $L --mode $mode --plant flow --k 24 --device cuda:$g --out $CL/${n}_${mode}_flow > $CL/${n}_${mode}_flow.log 2>&1
        PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $L --mode $mode --plant bmad --k 24 --workers 24 --num-macro 20000 --device cuda:$g --out $CL/${n}_${mode}_bmad > $CL/${n}_${mode}_bmad.log 2>&1
        echo "[$(date +%H:%M)] $n $mode: flow $(tail -n 1 $CL/${n}_${mode}_flow.log) | bmad $(tail -n 1 $CL/${n}_${mode}_bmad.log)" >> $CL/summary.txt ) &
    done
    wait
    log "closed loop done $n"
  fi
}
(SEED=0 bash scripts/run_queue.sh 1 $R $B) > $R/gpu1_s0.log 2>&1 &
validate real_covB2_screen_noblen_s0 0 1 &
validate real_covB2_screen_bc11loop_s0 1 1 &
# fine-tunes -> scorecards -> second seeds
for n in D1 D2; do
  until [ -e trained/xtcav23e_inj_centroid${n}_2026-10-17/checkpoints/final.ckpt ]; do sleep 120; done
done
log "centroid fine-tunes done"
SC=results/surrogate_scorecard_2026-10-18; mkdir -p $SC
b() { ls trained/xtcav23e_inj_centroid$1_2026-10-17/checkpoints/best-epoch=*.ckpt | sort -t= -k3 -g | head -n 1; }
cat > $SC/models.json <<J
{"B2-cov best": "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",
 "D1 centroid (w_pair 3) best": "$(b D1)", "D2 centroid (w_pair 10) best": "$(b D2)"}
J
(SEED=1 bash scripts/run_queue.sh 0 $R $N) > $R/gpu0_s1.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 1 $R $B) > $R/gpu1_s1.log 2>&1 &
for spec in "topup_table:processed/xtcav23e_inj_topup_v2_2026-10-16.h5" "cov_table:processed/xtcav23e_inj_cov_2026-10-17.h5"; do
  t=${spec#*:}; nm=${spec%%:*}
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 $PY scripts/surrogate_scorecard.py --processed $t \
    --role-map processed/role_map_ftc_s0_2026-10-16.json --models $SC/models.json --out $SC/$nm > $SC/$nm.log 2>&1
  log "scorecard $nm done"
done
validate real_covB2_screen_noblen_s1 0 0 &
validate real_covB2_screen_bc11loop_s1 1 0 &
wait
log "orchestration done"
