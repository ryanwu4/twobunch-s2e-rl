#!/bin/bash
# Pair-weight push from B2 best (overnight 2026-10-16 06:35): C = w_pair 6, C2 = w_pair 10; top-up v2 table, frozen
# role map, 25 epochs cosine (time-boxed before 10:00). GPU0.
cd "$(dirname "$0")/.."
COMMON=(--processed processed/xtcav23e_inj_topup_v2_2026-10-16.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0
        --init-ckpt "trained/xtcav23e_inj_topupB2v2_2026-10-16/checkpoints/best-epoch=027-val_paired_delta_rms_um=5.9350.ckpt"
        --lr 3e-5 --epochs 25 --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part)
for spec in "C:6.0" "C2:10.0"; do
  n=${spec%%:*}; wp=${spec#*:}
  OUT=trained/xtcav23e_inj_pairw${n}_2026-10-16
  [ -e "$OUT" ] && { echo "$OUT exists"; continue; }
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 setsid nohup nice -n 5 \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" --w-pair "$wp" --out "$OUT" \
    > logs/surrogate/xtcav23e_inj_pairw${n}_2026-10-16.log 2>&1 < /dev/null &
  echo "$n launched $!"
done
