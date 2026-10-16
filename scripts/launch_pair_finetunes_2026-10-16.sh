#!/bin/bash
# Pair-loss fine-tunes A (+ two-part T head) and A2 (pair loss only) on GPU0, warm from ftc_s0 (overnight 2026-10-16).
cd "$(dirname "$0")/.."
COMMON=(--processed processed/xtcav23e_inj_v2_2026-10-16.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0
        --init-ckpt "trained/xtcav23e_inj_ftc_s0_2026-10-15/checkpoints/best-epoch=046-val_paired_delta_rms_um=7.5721.ckpt"
        --lr 3e-5 --epochs ${EPOCHS:-80} --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --w-pair 1.0 --save-top-k 3)
for n in A A2; do
  OUT=trained/xtcav23e_inj_pair${n}${SUFFIX:-}_2026-10-16
  [ -e "$OUT" ] && { echo "$OUT exists"; continue; }
  EXTRA=(); [ "$n" = A ] && EXTRA=(--t-two-part)
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 setsid nohup nice -n 5 \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" "${EXTRA[@]}" --out "$OUT" \
    > logs/surrogate/xtcav23e_inj_pair${n}${SUFFIX:-}_2026-10-16.log 2>&1 < /dev/null &
  echo "$n launched $!"
done
