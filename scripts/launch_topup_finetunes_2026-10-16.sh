#!/bin/bash
# Fine-tunes on the merged top-up table (overnight 2026-10-16), warm from A best (pair + two-part T), frozen role map:
# B = A's recipe (w_pair 1), B2 = w_pair 3 (push the attenuated pair slope). GPU0, 40 epochs cosine.
cd "$(dirname "$0")/.."
COMMON=(--processed processed/xtcav23e_inj_topup_2026-10-16.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0
        --init-ckpt "trained/xtcav23e_inj_pairA40_2026-10-16/checkpoints/best-epoch=034-val_paired_delta_rms_um=6.8576.ckpt"
        --lr 3e-5 --epochs 40 --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3
        --t-two-part)
for spec in "B:1.0" "B2:3.0"; do
  n=${spec%%:*}; wp=${spec#*:}
  OUT=trained/xtcav23e_inj_topup${n}_2026-10-16
  [ -e "$OUT" ] && { echo "$OUT exists"; continue; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 setsid nohup nice -n 5 \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" --w-pair "$wp" --out "$OUT" \
    > logs/surrogate/xtcav23e_inj_topup${n}_2026-10-16.log 2>&1 < /dev/null &
  echo "$n launched $!"
done
