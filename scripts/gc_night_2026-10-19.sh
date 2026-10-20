#!/bin/bash
# Night 2026-10-19 -> 20: goal-conditioned training (configs/rl/gc_2026-10-19: half the setpoints anywhere in the start
# box, half spacing-only), both directions x seeds {0, 1}, 1200 it, 2 per GPU; then per run
#   nominal (own mix; free / constrained split), far1 (every setpoint anywhere, constrained), local (trained radius,
#   constrained), stress, DAQ replay x 2;
# the two plain spacing-head baselines under far1 / local; pick the best GC run (far1 settled within 10 um) ->
# logs/2026-10-19/gc/BEST, then the noise-free Bmad slew eval of it (CPU, 48 workers).
# Log logs/2026-10-19/gc_night.log; results/3_rl_eval/eval_gc_2026-10-19/summary.txt.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
G=logs/2026-10-19/gc
EV=results/3_rl_eval/eval_gc_2026-10-19
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml; STR=configs/rl/eval/jitter_stress_2026-10-18.yaml
C=configs/rl/gc_2026-10-19
mkdir -p $G $EV
log() { echo "[$(date +%H:%M)] $*"; }
FAR1=(--set v4.goal_far_frac=1.0 --set v4.spacing_only_frac=1e-9)
LOC=(--set v4.goal_far_frac=0.0 --set v4.spacing_only_frac=1e-9)
ev1() {  # logdir name gpu tag de_yaml [sets...]
  d=$1; n=$2; g=$3; t=$4; y=$5; shift 5
  echo "$n | $t | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --device cuda:$g --de-yaml $y "$@" --out $EV/${n}_$t 2>&1 | grep '^\[')" >> $EV/summary.txt
}
evall() {  # logdir name gpu
  ev1 $1 $2 $3 nominal $NOM
  ev1 $1 $2 $3 far1 $NOM "${FAR1[@]}"
  ev1 $1 $2 $3 local $NOM "${LOC[@]}"
  ev1 $1 $2 $3 stress $STR
  for scan in TEST_15668 E300_15673; do
    echo "$2 | replay $scan | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $1 --scan $scan --k 64 --device cuda:$3 --out $EV/${2}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
}
slot() {  # gpu seed cfg
  g=$1; s=$2; name=$3; d=$G/${name}_s$s
  [ -e $d ] && { log "skip $name s$s (exists)"; return; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 \
    $PY -m twobunch_s2e_rl.rl.train_bptt --cfg $C/$name.yaml --logdir $d --seed $s --device cuda:$g > $d.log 2>&1
  log "$name s$s train rc=$?"
  [ -e $d/final_policy.pt ] && evall $d ${name}_s$s $g || echo "${name}_s$s | no final policy" >> $EV/summary.txt
}
log "start"
( ev1 logs/2026-10-18/night/zs_screen_noblen_jit_s0 base_zs_noblen_s0 0 far1 $NOM --set v4.goal_far_frac=1.0
  ev1 logs/2026-10-18/night/zs_screen_noblen_jit_s0 base_zs_noblen_s0 0 local $NOM
  ev1 logs/2026-10-18/night/zs_screen_bc11loop_jit_s1 base_zs_bc11loop_s1 1 far1 $NOM --set v4.goal_far_frac=1.0
  ev1 logs/2026-10-18/night/zs_screen_bc11loop_jit_s1 base_zs_bc11loop_s1 1 local $NOM ) &
slot 0 0 gc_zs_screen_noblen_jit & P1=$!
slot 0 1 gc_zs_screen_noblen_jit & P2=$!
slot 1 0 gc_zs_screen_bc11loop_jit & P3=$!
slot 1 1 gc_zs_screen_bc11loop_jit & P4=$!
wait $P1 $P2 $P3 $P4
wait
log "GC trainings + evals done"
BEST=$(grep "^gc_.* | far1 |" $EV/summary.txt | sed -E 's/^([^ ]+) .*settled within 10 um ([0-9.]+),.*/\2 \1/' | sort -rn | head -1 | cut -d' ' -f2)
[ -n "$BEST" ] || { log "could not pick a best GC run"; exit 1; }
echo $G/$BEST > $G/BEST
log "best GC run: $BEST"
OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $G/$BEST --mode golden --slew 3 --seg-steps 16 \
  --ideal configs/rl/eval/ideal_machine_2026-10-19.yaml --plant flow --k 48 --device cuda:0 \
  --out results/3_rl_eval/slew_ideal_bmad_2026-10-19/${BEST}_flow > $G/slew_flow.log 2>&1
OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $G/$BEST --mode golden --slew 3 --seg-steps 16 \
  --ideal configs/rl/eval/ideal_machine_2026-10-19.yaml --plant bmad --k 48 --workers 48 --num-macro 20000 --device cuda:0 \
  --out results/3_rl_eval/slew_ideal_bmad_2026-10-19/${BEST}_bmad > $G/slew_bmad.log 2>&1
log "GC slew eval rc=$?"
log "gc night chain done"
