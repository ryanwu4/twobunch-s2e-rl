#!/bin/bash
# Backbone-residual flow (2026-10-21, ~04:30): each beam sheared by its own binned backbone (32 equal-charge pz bins, z and x,
# PCA 16), knob-conditioned Gaussian prior over the PCA coefficients for sampling. From B2-cov, 1024 table, 30 epochs, GPU1.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
arm=bb32_P1024; out=trained/xtcav23e_inj_covB2_${arm}_2026-10-21
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train \
  --processed processed/xtcav23e_inj_cov_2026-10-17.h5 --batch-size 64 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json \
  --select-metric val_paired_delta_rms_um --patience 0 --init-ckpt "$B2" --epochs 30 --lr-schedule cosine --num-workers 4 --seed 0 \
  --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0 --lr 5e-5 \
  --backbone --backbone-bins 32 --backbone-pca 16 --backbone-dims 2 0 --w-backbone-prior 1.0 --backbone-lr 1e-3 \
  --out $out > logs/surrogate/xtcav23e_inj_covB2_${arm}_2026-10-21.log 2>&1
echo "[$(date +%H:%M)] $arm rc=$?"
d=$out/checkpoints
best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
for s in "final:$d/final.ckpt" "best:$best"; do
  t=${s%%:*}; c=${s#*:}
  [ -e "$c" ] || continue
  PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/flow_sharpness_eval_2026-10-20.py --tag ${arm}_$t --ckpt "$c" --screen \
    > results/2_surrogate_screen/flow_sharpness_2026-10-20/${arm}_$t.log 2>&1
done
echo "[$(date +%H:%M)] eval $arm done"
