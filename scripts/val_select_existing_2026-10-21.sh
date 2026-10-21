#!/bin/bash
# Validation-protocol scores of the 10-20/21 candidates (scripts/surrogate_val_select_2026-10-21.py), sequential, CPU.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
for spec in "lat8_P4096_final:trained/xtcav23e_inj_covB2_lat8_P4096_2026-10-21/checkpoints/final.ckpt" \
            "curve_final:trained/xtcav23e_inj_covB2_curve_2026-10-20/checkpoints/final.ckpt" \
            "lat8_P1024_final:trained/xtcav23e_inj_covB2_lat8_P1024_2026-10-21/checkpoints/final.ckpt" \
            "bb32x_final:trained/xtcav23e_inj_covB2_bb32x_P1024_2026-10-21/checkpoints/final.ckpt" \
            "spacing_head:trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt"; do
  t=${spec%%:*}; c=${spec#*:}
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=16 $PY scripts/surrogate_val_select_2026-10-21.py --tag $t --ckpt "$c" > results/2_surrogate_screen/val_select_2026-10-21/$t.log 2>&1
done
echo "[$(date +%H:%M)] existing done"
