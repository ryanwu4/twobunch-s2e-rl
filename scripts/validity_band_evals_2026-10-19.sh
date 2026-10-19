#!/bin/bash
# Validity-band check (2026-10-19): continuous evals (nominal jitter, own flow) saving per-segment goal spacing.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
O=results/2_surrogate_screen/validity_band_2026-10-19/eval
run() { PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $1 --device cuda:$2 --k 512 \
  --de-yaml configs/rl/eval/jitter_nominal_2026-10-18.yaml --out $O/$(basename $1) 2>&1 | grep '^\['; }
( run logs/2026-10-18/night/zs_screen_noblen_jit_s0 0; run logs/2026-10-18/realistic/real_covB2_screen_noblen_jit_s0 0 ) &
( run logs/2026-10-18/night/zs_screen_bc11loop_jit_s1 1; run logs/2026-10-18/realistic/real_covB2_screen_bc11loop_v3_jit_s0 1 ) &
wait
echo done
