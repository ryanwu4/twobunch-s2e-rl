#!/bin/bash
# Standard eval set (2026-10-21 -> 22) for the controllers retrained on C_B3_ema (logs/2026-10-21/surrC/BEST_{plain,gc},
# picked by scripts/surrC_night_2026-10-21.sh) AND the incumbents (logs/2026-10-19/nocrop/BEST_{plain,gc}), which
# never had these on the no-crop runs -> matched comparison. (The chain already does nominal / far1 / local / stress,
# DAQ replay x 2, cross-surrogate far1 and the noise-free Bmad slew.)
#   A. closed-loop Bmad, flow + Bmad plant, k = 24: --mode test (24 held-out injector points,
#      configs/rl/eval/heldout_injectors_2026-10-18.json), golden, inj  [as results/3_rl_eval/closed_loop_bmad_2026-10-1{8,9}]
#      incumbents start now (2 lanes x 24 workers, alongside the trainings); new policies after the chain (2 lanes x 48).
#   B. after the chain (GPU): large setpoint moves (goal radius x2 / full window) and the validity-band eval (k 512,
#      nominal; errors.npz per goal), for all four policies.
# Log logs/2026-10-21/surrC_standard.log; results/3_rl_eval/{closed_loop_bmad,eval_std}_2026-10-22/.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
CL=results/3_rl_eval/closed_loop_bmad_2026-10-22
EV=results/3_rl_eval/eval_std_2026-10-22
NOM=configs/rl/eval/jitter_nominal_2026-10-18.yaml
PTS=configs/rl/eval/heldout_injectors_2026-10-18.json
mkdir -p $CL $EV
log() { echo "[$(date +%H:%M)] $*"; }
cl() {  # logdir tag workers gpu mode
  local d=$1 t=$2 w=$3 g=$4 m=$5 X=()
  [ $m = test ] && X=(--points-json $PTS)
  OMP_NUM_THREADS=1 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $d --mode $m "${X[@]}" --plant flow --k 24 --device cuda:$g \
    --out $CL/${t}_${m}_flow > $CL/${t}_${m}_flow.log 2>&1
  OMP_NUM_THREADS=1 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir $d --mode $m "${X[@]}" --plant bmad --k 24 \
    --workers $w --num-macro 20000 --device cuda:$g --out $CL/${t}_${m}_bmad > $CL/${t}_${m}_bmad.log 2>&1
  echo "$t | $m | flow $(grep -h "^\[$m/flow\]" $CL/${t}_${m}_flow.log | tail -1) | bmad $(grep -h "^\[$m/bmad\]" $CL/${t}_${m}_bmad.log | tail -1)" >> $CL/summary.txt
  log "closed-loop $t $m done"
}
lane() { for m in test golden inj; do cl $1 $2 $3 $4 $m; done; }
gpu_evals() {  # logdir tag gpu gc
  local d=$1 t=$2 g=$3 so=()
  [ "$4" = 1 ] && so=(--set v4.spacing_only_frac=1e-9)
  for rad in "x2:4.0:5.0" "full:8.0:8.0"; do
    local k=${rad%%:*} r=${rad#*:}; local r1=${r%%:*} r2=${r#*:}
    echo "$t | nominal goal_radius $k | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --device cuda:$g --de-yaml $NOM \
      --set "v4.goal_radius={\"L1PhaseSet\": $r1, \"L2PhaseSet\": $r2}" "${so[@]}" --out $EV/${t}_bigmove_$k 2>&1 | grep '^\[')" >> $EV/summary.txt
  done
  echo "$t | validity band k512 nominal | $(OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --device cuda:$g --k 512 \
    --de-yaml $NOM --out $EV/${t}_validity 2>&1 | grep '^\[')" >> $EV/summary.txt
  log "gpu evals $t done"
}
IP=$(cat logs/2026-10-19/nocrop/BEST_plain); IG=$(cat logs/2026-10-19/nocrop/BEST_gc)
log "start: incumbents $IP, $IG"
lane $IP inc_plain 24 0 & L1=$!
lane $IG inc_gc 24 1 & L2=$!
until grep -qs "surrC night chain done" logs/2026-10-21/surrC_night.log; do sleep 120; done
NP=$(cat logs/2026-10-21/surrC/BEST_plain 2>/dev/null); NG=$(cat logs/2026-10-21/surrC/BEST_gc 2>/dev/null)
log "chain done: new $NP, $NG"
( gpu_evals $IP inc_plain 0 0; gpu_evals $NP new_plain 0 0 ) &
( gpu_evals $IG inc_gc 1 1; gpu_evals $NG new_gc 1 1 ) &
wait $L1 $L2
log "incumbent lanes done"
[ -n "$NP" ] && lane $NP new_plain 56 0 &
[ -n "$NG" ] && lane $NG new_gc 56 1 &
wait
log "standard evals done"
