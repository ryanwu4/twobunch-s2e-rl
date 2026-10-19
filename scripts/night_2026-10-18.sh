#!/bin/bash
# Night 2026-10-18 -> 19 (after the jitter robustness chain + the old-baseline eval):
#  1. large-setpoint-move eval (goal radius x2 and the full window) of the best jitter-trained policy per direction
#  2. closed-loop Bmad (CPU) of the same two policies at 24 held-out injector points (test split), + paired flow run
#  3. 8 trainings, 900 it, 2 per GPU: {noblen, bc11loop} x {spacing head, + failure detection} x seeds {0, 1} on the
#     spacing-head flow (configs/rl/night_2026-10-18), each evaluated (nominal / stress jitter, DAQ replay) when done.
# Log logs/2026-10-18/night.log; results/eval_night_2026-10-18/summary.txt, results/closed_loop_bmad_2026-10-19/.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
R=logs/2026-10-18/realistic
N=logs/2026-10-18/night
EV=results/eval_night_2026-10-18
CL=results/closed_loop_bmad_2026-10-19
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml; STR=configs/rl/eval/jitter_stress_2026-10-18.yaml
C=configs/rl/night_2026-10-18
mkdir -p $N $EV $CL
log() { echo "[$(date +%H:%M)] $*"; }

until grep -qs "jitter robustness chain done" logs/2026-10-18/jitter_chain.log; do sleep 120; done
while kill -0 2793403 2>/dev/null; do sleep 60; done          # old-baseline eval (jitter_baseline_eval_2026-10-18.sh)
log "jitter chain + baseline eval done"

# best seed per direction by nominal settled spacing (results/eval_jitter_2026-10-18/summary.txt)
best() {
  grep "^real_covB2_screen_$1_jit_s[01] | nominal |" results/eval_jitter_2026-10-18/summary.txt \
    | sed -E 's/^([^ ]+) .*settled spacing ([0-9.]+) um.*/\2 \1/' | sort -n | head -1 | cut -d' ' -f2
}
BN=$(best noblen); BB=$(best bc11loop_v3)
log "best jitter-trained: $BN, $BB"
[ -n "$BN" ] && [ -n "$BB" ] || { log "could not pick best policies"; exit 1; }

# 1. large setpoint moves (eval only; trained goal radius 2.0 / 2.5 deg)
for spec in "$BN:0" "$BB:1"; do
  n=${spec%%:*}; g=${spec#*:}
  for rad in "x2:4.0:5.0" "full:8.0:8.0"; do
    t=${rad%%:*}; r=${rad#*:}; r1=${r%%:*}; r2=${r#*:}
    echo "$n | nominal goal_radius $t | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $R/$n --device cuda:$g \
      --de-yaml $NOM --set "v4.goal_radius={\"L1PhaseSet\": $r1, \"L2PhaseSet\": $r2}" --out $EV/${n}_bigmove_$t 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
done
log "large-move evals done"

# 2. closed-loop Bmad at held-out injector points (CPU, 24 workers each; ~3 h), + the paired surrogate run
for spec in "$BN:0" "$BB:1"; do
  n=${spec%%:*}; g=${spec#*:}
  ( OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $R/$n --mode test \
      --points-json configs/rl/eval/heldout_injectors_2026-10-18.json --plant flow --k 24 --device cuda:$g \
      --out $CL/${n}_test_flow > $CL/${n}_test_flow.log 2>&1
    OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $R/$n --mode test \
      --points-json configs/rl/eval/heldout_injectors_2026-10-18.json --plant bmad --k 24 --workers 24 \
      --num-macro 20000 --device cuda:$g --out $CL/${n}_test_bmad > $CL/${n}_test_bmad.log 2>&1
    log "closed-loop $n done" ) &
done

# 3. training slots: train, then evaluate, per config
ev() {  # logdir name gpu
  for c in nominal stress; do
    y=$NOM; [ $c = stress ] && y=$STR
    echo "$2 | $c | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $1 --device cuda:$3 --de-yaml $y --out $EV/$2_$c 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
  for scan in TEST_15668 E300_15673; do
    echo "$2 | replay $scan | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.daq_replay --logdir $1 --scan $scan --k 64 --device cuda:$3 --out $EV/${2}_replay_$scan 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
}
slot() {  # gpu seed cfg...
  g=$1; s=$2; shift 2
  for name in "$@"; do
    d=$N/${name}_s$s
    [ -e $d ] && { log "skip $name s$s (exists)"; continue; }
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 nice -n 5 \
      $PY -m twobunch_s2e_rl.rl.train_bptt --cfg $C/$name.yaml --logdir $d --seed $s --device cuda:$g > $d.log 2>&1
    log "$name s$s train rc=$?"
    [ -e $d/final_policy.pt ] && ev $d ${name}_s$s $g || echo "${name}_s$s | no final policy" >> $EV/summary.txt
  done
}
slot 0 0 zs_screen_noblen_jit zs_screen_noblen_jit_fd & P1=$!
slot 0 1 zs_screen_noblen_jit_fd zs_screen_noblen_jit & P2=$!
slot 1 0 zs_screen_bc11loop_jit zs_screen_bc11loop_jit_fd & P3=$!
slot 1 1 zs_screen_bc11loop_jit_fd zs_screen_bc11loop_jit & P4=$!
wait $P1 $P2 $P3 $P4
log "night trainings + evals done"
wait
log "night chain done"
