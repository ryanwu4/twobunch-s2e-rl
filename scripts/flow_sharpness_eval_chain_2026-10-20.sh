#!/bin/bash
# Evaluate each flow-sharpness arm when its training finishes: best (val_paired_delta_rms) and final checkpoints.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
for arm in curve rqs32 wide; do
  ( until grep -q "^\[.*\] $arm rc=" logs/2026-10-20_flow_sharpness.log; do sleep 120; done
    d=trained/xtcav23e_inj_covB2_${arm}_2026-10-20/checkpoints
    best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
    for spec in "best:$best" "final:$d/final.ckpt"; do
      t=${spec%%:*}; c=${spec#*:}
      [ -e "$c" ] || continue
      PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/flow_sharpness_eval_2026-10-20.py --tag ${arm}_$t --ckpt "$c" --screen \
        > results/2_surrogate_screen/flow_sharpness_2026-10-20/${arm}_$t.log 2>&1
    done
    echo "[$(date +%H:%M)] eval $arm done" ) &
done
wait
echo "[$(date +%H:%M)] evals done"
