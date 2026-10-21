#!/bin/bash
# When the B arms finish: validation-protocol scores of each arm's final + best checkpoint (CPU, 2 at a time).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
until grep -q "B done" logs/2026-10-21_surrogate_B.log && grep -q "B done" logs/2026-10-21_surrogate_B23.log; do sleep 120; done
for arm in B0_rs1024 B1_rs1024_sw B2_rs1024_lat B3_rs1024_sw_lat; do
  d=trained/xtcav23e_inj_covB2_${arm}_2026-10-21/checkpoints
  best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
  ( PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/surrogate_val_select_2026-10-21.py --tag ${arm}_final --ckpt $d/final.ckpt > results/2_surrogate_screen/val_select_2026-10-21/${arm}_final.log 2>&1
    PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/surrogate_val_select_2026-10-21.py --tag ${arm}_best --ckpt "$best" > results/2_surrogate_screen/val_select_2026-10-21/${arm}_best.log 2>&1 ) &
  [ $arm = B1_rs1024_sw ] && wait
done
wait
echo "[$(date +%H:%M)] B val done"
