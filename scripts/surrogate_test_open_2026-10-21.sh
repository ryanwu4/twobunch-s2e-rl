#!/bin/bash
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
for t in C_B3 C_B2; do
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=16 $PY scripts/flow_sharpness_eval_2026-10-20.py --tag ${t}_ema \
    --ckpt trained/xtcav23e_inj_covB2_${t}_2026-10-21/checkpoints/final_ema.ckpt > logs/2026-10-21_sharp_${t}.log 2>&1 &
done
PYTHONPATH=$PWD/src OMP_NUM_THREADS=32 $PY scripts/surrogate_test_open_2026-10-21.py > logs/2026-10-21_test_open.log 2>&1
wait
echo "[$(date +%H:%M)] test open done"
