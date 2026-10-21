#!/bin/bash
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
T=processed/xtcav23e_inj_cov_sw_P4096_2026-10-21.h5
for spec in "B2-cov_sw:trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt" \
            "B3_final_sw:trained/xtcav23e_inj_covB2_B3_rs1024_sw_lat_2026-10-21/checkpoints/final.ckpt" \
            "B2_final_sw:trained/xtcav23e_inj_covB2_B2_rs1024_lat_2026-10-21/checkpoints/final.ckpt" \
            "lat8_P4096_final_sw:trained/xtcav23e_inj_covB2_lat8_P4096_2026-10-21/checkpoints/final.ckpt"; do
  t=${spec%%:*}; c=${spec#*:}
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/surrogate_val_select_2026-10-21.py --tag $t --ckpt "$c" --sweeps $T > results/2_surrogate_screen/val_select_2026-10-21/$t.log 2>&1
done
echo "[$(date +%H:%M)] sweep baselines done"
