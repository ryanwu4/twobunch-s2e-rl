#!/bin/bash
# Phase C (2026-10-21, ~16:35): the latent B arms continued on the merged table (cov + compression sweeps, 4096 resampled to
# 1024, full-particle truths), lr 2e-5 cosine, 20 epochs, EMA 0.999, one arm per GPU. Then val-protocol + val-sweep scores.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
T=processed/xtcav23e_inj_cov_sw_P4096_2026-10-21.h5
COMMON=(--processed $T --truth-h5 ${T%.h5}_truth.h5 --resample-p 1024 --batch-size 64 --split group
        --role-map processed/role_map_ftc_s0_2026-10-16.json --select-metric val_paired_delta_rms_um --patience 0 --epochs 20
        --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0
        --lr 2e-5 --ema-decay 0.999 --beam-latent-dim 8 --beam-latent-beta 1.0 --beam-latent-lr 3e-4)
run() {  # gpu arm init args...
  g=$1; arm=$2; init=$3; shift 3; out=trained/xtcav23e_inj_covB2_${arm}_2026-10-21
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" \
    --init-ckpt $init "$@" --out $out > logs/surrogate/xtcav23e_inj_covB2_${arm}_2026-10-21.log 2>&1
  echo "[$(date +%H:%M)] $arm rc=$?"
  for c in final final_ema; do
    PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/surrogate_val_select_2026-10-21.py --tag ${arm}_$c --ckpt $out/checkpoints/$c.ckpt \
      --sweeps $T > results/2_surrogate_screen/val_select_2026-10-21/${arm}_$c.log 2>&1
  done
  echo "[$(date +%H:%M)] $arm scored"
}
run 0 C_B3 trained/xtcav23e_inj_covB2_B3_rs1024_sw_lat_2026-10-21/checkpoints/final.ckpt --w-sw 2.0 --w-logsz 0.5 &
run 1 C_B2 trained/xtcav23e_inj_covB2_B2_rs1024_lat_2026-10-21/checkpoints/final.ckpt &
wait
echo "[$(date +%H:%M)] C done"
