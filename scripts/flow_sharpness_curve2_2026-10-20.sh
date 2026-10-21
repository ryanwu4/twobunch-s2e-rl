#!/bin/bash
# curve2 (2026-10-20): the curve head regressed onto per-row least-squares backbones (w_curve 10, order 12) from B2-cov;
# the unsupervised curve arm left the head at 2-7% of the z spread. Then the sharpness eval (best + final).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
out=trained/xtcav23e_inj_covB2_curve2_2026-10-20
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train \
  --processed processed/xtcav23e_inj_cov_2026-10-17.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json \
  --select-metric val_paired_delta_rms_um --patience 0 --init-ckpt "$B2" --epochs 40 --lr-schedule cosine \
  --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0 \
  --lr 5e-5 --curve-head --curve-dims 2 0 --curve-order 12 --curve-head-lr 1e-3 --w-curve 10 \
  --out $out > logs/surrogate/xtcav23e_inj_covB2_curve2_2026-10-20.log 2>&1
echo "[$(date +%H:%M)] curve2 rc=$?"
d=$out/checkpoints
best=$(ls $d/best-*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
for spec in "best:$best" "final:$d/final.ckpt"; do
  t=${spec%%:*}; c=${spec#*:}
  [ -e "$c" ] && PYTHONPATH=$PWD/src OMP_NUM_THREADS=12 $PY scripts/flow_sharpness_eval_2026-10-20.py --tag curve2_$t --ckpt "$c" --screen \
    > results/2_surrogate_screen/flow_sharpness_2026-10-20/curve2_$t.log 2>&1
done
echo "[$(date +%H:%M)] eval curve2 done"
