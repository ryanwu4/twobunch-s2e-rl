#!/bin/bash
# Control A (2026-10-21, reviewer plan): lat8_P4096's config WITHOUT the beam latent (4096 table, batch 16, 40 epochs,
# lr 5e-5, from B2-cov) -- attributes lat8_P4096's gains between the latent, the particle count and the step count.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
arm=ctrlA_P4096; out=trained/xtcav23e_inj_covB2_${arm}_2026-10-21
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train \
  --processed processed/xtcav23e_inj_cov_P4096_2026-10-20.h5 --batch-size 16 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json \
  --select-metric val_paired_delta_rms_um --patience 0 --init-ckpt "$B2" --epochs 40 --lr-schedule cosine --num-workers 4 --seed 0 \
  --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0 --lr 5e-5 \
  --out $out > logs/surrogate/xtcav23e_inj_covB2_${arm}_2026-10-21.log 2>&1
echo "[$(date +%H:%M)] $arm rc=$?"
