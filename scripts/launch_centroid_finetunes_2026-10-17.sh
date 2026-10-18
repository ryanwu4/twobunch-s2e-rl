#!/bin/bash
# Deterministic-centroid-head fine-tunes from B2-cov best on the coverage table (B2 recipe otherwise): D1 = w_pair 3,
# D2 = w_pair 10 (the centroid is decoupled from the shape, so a higher pair weight should not cost sigma_z).
# Centroid heads at their own lr 1e-3 (the first launch at the global 3e-5 learned too slowly: aborted 19:12).
cd "$(dirname "$0")/.."
COMMON=(--processed processed/xtcav23e_inj_cov_2026-10-17.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0
        --init-ckpt "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
        --lr 3e-5 --epochs 40 --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3
        --t-two-part --centroid-head --w-centroid 1.0 --centroid-prefit-steps 20000 --centroid-head-lr 1e-3)
mkdir -p logs/surrogate
for spec in "D1:3.0:0" "D2:10.0:1"; do
  IFS=: read n wp gpu <<< "$spec"
  OUT=trained/xtcav23e_inj_centroid${n}_2026-10-17
  [ -e "$OUT" ] && { echo "$OUT exists"; continue; }
  CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 setsid nohup nice -n 5 \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" --w-pair "$wp" --out "$OUT" \
    > logs/surrogate/xtcav23e_inj_centroid${n}_2026-10-17.log 2>&1 < /dev/null &
  echo "$n launched $! on GPU$gpu"
done
