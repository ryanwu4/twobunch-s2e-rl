#!/bin/bash
# Closed-loop Bmad of bc11loop v3 s0 (golden + inj, 24 episodes, 2e4), paired surrogate runs first.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
for mode in golden inj; do
  ( PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir logs/2026-10-18/realistic/real_covB2_screen_bc11loop_v3_s0 --mode $mode --plant flow --k 24 --device cuda:$([ $mode = golden ] && echo 0 || echo 1) --out results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_flow > results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_flow.log 2>&1
    PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir logs/2026-10-18/realistic/real_covB2_screen_bc11loop_v3_s0 --mode $mode --plant bmad --k 24 --workers 24 --num-macro 20000 --device cuda:$([ $mode = golden ] && echo 0 || echo 1) --out results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_bmad > results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_bmad.log 2>&1
    echo "[$(date +%H:%M)] bc11loop_v3_s0 $mode: flow $(tail -n 1 results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_flow.log) | bmad $(tail -n 1 results/closed_loop_bmad_2026-10-18/real_covB2_screen_bc11loop_v3_s0_${mode}_bmad.log)" >> results/closed_loop_bmad_2026-10-18/summary.txt ) &
done
wait
