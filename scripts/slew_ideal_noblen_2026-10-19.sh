#!/bin/bash
# Noise-free (ideal machine) Bmad slew eval of zs_screen_noblen_jit_s0: 48 golden-injector episodes x 3 setpoints
# anywhere in the start box x 16 steps; paired surrogate run first. results/3_rl_eval/slew_ideal_bmad_2026-10-19/.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
export PYTHONPATH=$PWD/src
L=logs/2026-10-18/night/zs_screen_noblen_jit_s0
O=results/3_rl_eval/slew_ideal_bmad_2026-10-19
A=(--logdir $L --mode golden --slew 3 --seg-steps 16 --ideal configs/rl/eval/ideal_machine_2026-10-19.yaml --k 48 --device cuda:1)
OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant flow --out $O/zs_screen_noblen_jit_s0_flow
OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.rl.closed_loop_bmad "${A[@]}" --plant bmad --workers 48 --num-macro 20000 \
  --out $O/zs_screen_noblen_jit_s0_bmad
echo "[$(date +%H:%M)] done rc=$?"
