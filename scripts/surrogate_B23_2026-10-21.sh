#!/bin/bash
# B arms (2026-10-21, reviewer plan): from B2-cov, 4096 table resampled to 1024 per visit, full-particle truths, batch 64,
# 40 epochs, lr 5e-5, both on GPU0. B2 = B0 + beam latent (8, beta 1); B3 = B1 + beam latent (the latent x SW 2x2, GPU0; replaces control A).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
COMMON=(--processed processed/xtcav23e_inj_cov_P4096_2026-10-20.h5 --truth-h5 processed/xtcav23e_inj_cov_P4096_2026-10-20_truth.h5
        --resample-p 1024 --batch-size 64 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0 --init-ckpt "$B2" --epochs 40 --lr-schedule cosine --num-workers 4
        --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0 --lr 5e-5)
run() {  # arm args...
  arm=$1; shift; out=trained/xtcav23e_inj_covB2_${arm}_2026-10-21
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" "$@" \
    --out $out > logs/surrogate/xtcav23e_inj_covB2_${arm}_2026-10-21.log 2>&1
  echo "[$(date +%H:%M)] $arm rc=$?"
}
run B2_rs1024_lat --beam-latent-dim 8 --beam-latent-beta 1.0 --beam-latent-lr 1e-3 &
run B3_rs1024_sw_lat --w-sw 2.0 --w-logsz 0.5 --beam-latent-dim 8 --beam-latent-beta 1.0 --beam-latent-lr 1e-3 &
wait
echo "[$(date +%H:%M)] B done"
