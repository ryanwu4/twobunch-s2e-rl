#!/bin/bash
# Night 2026-10-21 -> 22: the two incumbent controllers (both loops off = noblen; plain + goal-conditioned) retrained on
# the 10-21 surrogate C_B3_ema (configs/rl/surrC_2026-10-21: only flow_ckpt / logdir changed vs configs/rl/nocrop_2026-10-19).
#   4 trainings at once (plain 900 it, GC 1000 it; seeds 0, 1; one of each family per GPU);
#   every run evaluated when done (nominal / far1 / local / stress, DAQ replay x 2), + far1 on the OLD surrogate (covB2zs);
#   meanwhile: the incumbents (BEST_plain / BEST_gc) evaluated on the NEW surrogate (nominal / far1) -> surrogate-matched comparison;
#   per family, the better seed (far1 settled within 10 um) -> noise-free closed-loop Bmad slew eval (same args as
#   results/3_rl_eval/slew_ideal_bmad_2026-10-19, which holds the incumbents'), 56 Bmad workers per family.
# Log logs/2026-10-21/surrC_night.log; results/3_rl_eval/eval_surrC_2026-10-21/summary.txt, results/3_rl_eval/slew_ideal_bmad_2026-10-21/.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
N=logs/2026-10-21/surrC
EV=results/3_rl_eval/eval_surrC_2026-10-21
SL=results/3_rl_eval/slew_ideal_bmad_2026-10-21
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml; STR=configs/rl/eval/jitter_stress_2026-10-18.yaml
C=configs/rl/surrC_2026-10-21
NEWF=trained/xtcav23e_inj_covB2_C_B3_2026-10-21/checkpoints/final_ema.ckpt
OLDF=trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt
mkdir -p $N $EV $SL
log() { echo "[$(date +%H:%M)] $*"; }
FAR1=(--set v4.goal_far_frac=1.0)
LOC=(--set v4.goal_far_frac=0.0)
ev1() {  # logdir name gpu tag de_yaml [args...]
  d=$1; n=$2; g=$3; t=$4; y=$5; shift 5
  echo "$n | $t | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --device cuda:$g --de-yaml $y "$@" --out $EV/${n}_$t 2>&1 | grep '^\[')" >> $EV/summary.txt
}
rp() {  # logdir name gpu
  for scan in TEST_15668 E300_15673; do
    echo "$2 | replay $scan | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $1 --scan $scan --k 64 --device cuda:$3 --out $EV/${2}_replay_$scan 2>&1 | grep '^\[T\|^\[E')" >> $EV/summary.txt
  done
}
evall() {  # logdir name gpu gc(0/1)
  local so=()
  [ "$4" = 1 ] && so=(--set v4.spacing_only_frac=1e-9)
  ev1 $1 $2 $3 nominal $NOM
  ev1 $1 $2 $3 far1 $NOM "${FAR1[@]}" "${so[@]}"
  ev1 $1 $2 $3 local $NOM "${LOC[@]}" "${so[@]}"
  ev1 $1 $2 $3 stress $STR
  rp $1 $2 $3
  ev1 $1 $2 $3 far1_on_oldsurr $NOM "${FAR1[@]}" "${so[@]}" --flow $OLDF
  touch $N/${2}.evaluated
}
run() {  # gpu seed cfg gc
  local d=$N/${3}_s$2
  if [ -e $d ]; then log "skip $3 s$2 (exists)"; else
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 \
      $PY -m twobunch_s2e_rl.rl.train_bptt --cfg $C/$3.yaml --logdir $d --seed $2 --device cuda:$1 > $d.log 2>&1
    log "$3 s$2 train rc=$?"
  fi
  [ -e $d/final_policy.pt ] && evall $d ${3}_s$2 $1 $4 || echo "${3}_s$2 | no final policy" >> $EV/summary.txt
  log "$3 s$2 evaluated"
}
slew() {  # cfg tag
  until [ -e $N/${1}_s0.evaluated ] && [ -e $N/${1}_s1.evaluated ]; do sleep 120; done
  local best
  best=$(grep -E "^${1}_s[01] \| far1 \|" $EV/summary.txt | sed -E 's/^([^ ]+) .*settled within 10 um ([0-9.]+),.*/\2 \1/' \
         | sort -rn | head -1 | cut -d' ' -f2)
  [ -n "$best" ] || { log "slew $2: no best run"; return; }
  log "slew $2: best $best"
  echo $N/$best > $N/BEST_$2
  local A=(--logdir $N/$best --mode golden --slew 3 --seg-steps 16 --ideal configs/rl/eval/ideal_machine_2026-10-19.yaml --k 48 --device cuda:0)
  OMP_NUM_THREADS=1 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant flow --out $SL/${best}_flow > $SL/${best}_flow.log 2>&1
  OMP_NUM_THREADS=1 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant bmad --workers 56 --num-macro 20000 \
    --out $SL/${best}_bmad > $SL/${best}_bmad.log 2>&1
  log "slew $2 ($best) rc=$?"
}
log "start"
run 0 0 nc_c3_screen_noblen_jit 0 & P1=$!
run 0 0 gc_nc_c3_screen_noblen_jit 1 & P2=$!
run 1 1 nc_c3_screen_noblen_jit 0 & P3=$!
run 1 1 gc_nc_c3_screen_noblen_jit 1 & P4=$!
( for spec in "$(cat logs/2026-10-19/nocrop/BEST_plain):inc_plain:0" "$(cat logs/2026-10-19/nocrop/BEST_gc):inc_gc:1"; do
    IFS=: read -r d n gcf <<< "$spec"
    so=(); [ "$gcf" = 1 ] && so=(--set v4.spacing_only_frac=1e-9)
    ev1 $d ${n}_on_newsurr 1 nominal $NOM --flow $NEWF
    ev1 $d ${n}_on_newsurr 1 far1 $NOM "${FAR1[@]}" "${so[@]}" --flow $NEWF
  done
  log "incumbent cross-evals done" ) &
slew nc_c3_screen_noblen_jit plain &
slew gc_nc_c3_screen_noblen_jit gc &
wait $P1 $P2 $P3 $P4
log "trainings + evals done"
wait
log "surrC night chain done"
