#!/bin/bash
# Night 2026-10-19 -> 20, no auto-crop (screen window = the deployed hand crop, configs/rl/nocrop_2026-10-19):
#   slots (2 per GPU): goal-conditioned s0 (1000 it) and plain s0 (900 it) per direction, then the s1 runs;
#   every run evaluated when done (nominal / far1 / local / stress, DAQ replay on the no-crop readings);
#   when both s0 runs of a family are evaluated, the better one (far1 settled within 10 um) gets the noise-free Bmad
#   slew eval (48 workers);
#   meanwhile (GPU, light): the current policies and the cropped GC runs' best checkpoints (750 it) under the new window.
# Log logs/2026-10-19/nocrop_night.log; results/3_rl_eval/eval_nocrop_2026-10-19/summary.txt,
# results/3_rl_eval/slew_ideal_bmad_2026-10-19/.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
N=logs/2026-10-19/nocrop
EV=results/3_rl_eval/eval_nocrop_2026-10-19
SL=results/3_rl_eval/slew_ideal_bmad_2026-10-19
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml; STR=configs/rl/eval/jitter_stress_2026-10-18.yaml
C=configs/rl/nocrop_2026-10-19
V7=results/4_daq_realism/online_replay_v7_noauto_2026-10-19
mkdir -p $N $EV $SL
log() { echo "[$(date +%H:%M)] $*"; }
FAR1=(--set v4.goal_far_frac=1.0)
LOC=(--set v4.goal_far_frac=0.0)
WIN=(--set 'screen.roi={"x": 0.0068625, "y": 0.004575}')
ev1() {  # logdir name gpu tag de_yaml [args...]
  d=$1; n=$2; g=$3; t=$4; y=$5; shift 5
  echo "$n | $t | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --device cuda:$g --de-yaml $y "$@" --out $EV/${n}_$t 2>&1 | grep '^\[')" >> $EV/summary.txt
}
rp() {  # logdir name gpu [args...]
  d=$1; n=$2; g=$3; shift 3
  for scan in TEST_15668 E300_15673; do
    echo "$n | replay $scan | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $d --scan $scan --k 64 --device cuda:$g "$@" --out $EV/${n}_replay_$scan 2>&1 | grep '^\[T\|^\[E')" >> $EV/summary.txt
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
  touch $N/${2}.evaluated
}
train() {  # gpu seed cfg
  local d=$N/${3}_s$2
  [ -e $d ] && { log "skip $3 s$2 (exists)"; return; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 \
    $PY -m twobunch_s2e_rl.rl.train_bptt --cfg $C/$3.yaml --logdir $d --seed $2 --device cuda:$1 > $d.log 2>&1
  log "$3 s$2 train rc=$?"
}
slot() {  # gpu cfg gc
  for s in 0 1; do
    train $1 $s $2
    [ -e $N/${2}_s$s/final_policy.pt ] && evall $N/${2}_s$s ${2}_s$s $1 $3 || echo "${2}_s$s | no final policy" >> $EV/summary.txt
  done
}
slew() {  # family-prefix tag
  until [ -e $N/${1}noblen_jit_s0.evaluated ] && [ -e $N/${1}bc11loop_jit_s0.evaluated ]; do sleep 120; done
  local best
  best=$(grep -E "^${1}(noblen|bc11loop)_jit_s0 \| far1 \|" $EV/summary.txt | sed -E 's/^([^ ]+) .*settled within 10 um ([0-9.]+),.*/\2 \1/' \
         | sort -rn | head -1 | cut -d' ' -f2)
  [ -n "$best" ] || { log "slew $2: no best run"; return; }
  log "slew $2: best $best"
  echo $N/$best > $N/BEST_$2
  local A=(--logdir $N/$best --mode golden --slew 3 --seg-steps 16 --ideal configs/rl/eval/ideal_machine_2026-10-19.yaml --k 48 --device cuda:0)
  OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant flow --out $SL/${best}_flow > $SL/${best}_flow.log 2>&1
  OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant bmad --workers 48 --num-macro 20000 \
    --out $SL/${best}_bmad > $SL/${best}_bmad.log 2>&1
  log "slew $2 ($best) rc=$?"
}
log "start"
slot 0 gc_nc_zs_screen_noblen_jit 1 & P1=$!
slot 0 nc_zs_screen_noblen_jit 0 & P2=$!
slot 1 gc_nc_zs_screen_bc11loop_jit 1 & P3=$!
slot 1 nc_zs_screen_bc11loop_jit 0 & P4=$!
# current policies (cropped training) and the cropped GC runs' best checkpoints under the new window
( for spec in "logs/2026-10-18/night/zs_screen_noblen_jit_s0:old_zs_noblen_s0" "logs/2026-10-18/night/zs_screen_bc11loop_jit_s1:old_zs_bc11loop_s1"; do
    d=${spec%%:*}; n=${spec#*:}
    ev1 $d ${n}_window 0 nominal $NOM "${WIN[@]}"
    ev1 $d ${n}_window 0 far1 $NOM "${WIN[@]}" "${FAR1[@]}"
    rp $d ${n} 0 --replay-dir $V7
  done
  for s in 0 1; do for dd in noblen bc11loop; do
    d=logs/2026-10-19/gc/gc_zs_screen_${dd}_jit_s$s
    [ -e $d/best_policy.pt ] || continue
    for t in nominal far1 local; do
      case $t in far1) X=("${FAR1[@]}" --set v4.spacing_only_frac=1e-9);; local) X=("${LOC[@]}" --set v4.spacing_only_frac=1e-9);; *) X=();; esac
      echo "cropGC750_${dd}_s$s | $t | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --policy best --device cuda:1 --de-yaml $NOM "${X[@]}" --out $EV/cropGC750_${dd}_s${s}_$t 2>&1 | grep '^\[')" >> $EV/summary.txt
    done
  done; done
  log "reference evals done" ) &
slew nc_zs_screen_ plain &
slew gc_nc_zs_screen_ gc &
wait $P1 $P2 $P3 $P4
log "trainings + evals done"
wait
log "nocrop night chain done"
