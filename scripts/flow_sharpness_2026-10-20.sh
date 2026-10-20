#!/bin/bash
# Flow sharpness arms (2026-10-20), all from B2-cov, 40 epochs cosine, same losses as B2-cov:
#   curve : chirp-curve head (z, x sheared by Chebyshev g(pz), order 8), full warm start, head lr 1e-3
#   rqs32 : tail bound 4, 32 bins (spline output layers fresh), lr 1e-4
#   wide  : rqs32 + hidden 256 (partial warm start ~ fresh), lr 1e-4
# Logs logs/surrogate/xtcav23e_inj_covB2_<arm>_2026-10-20.log; outputs trained/xtcav23e_inj_covB2_<arm>_2026-10-20.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
B2="trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
COMMON=(--processed processed/xtcav23e_inj_cov_2026-10-17.h5 --split group --role-map processed/role_map_ftc_s0_2026-10-16.json
        --select-metric val_paired_delta_rms_um --patience 0 --init-ckpt "$B2" --epochs 40 --lr-schedule cosine
        --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 --t-two-part --w-pair 3.0)
run() {  # gpu arm args...
  g=$1; arm=$2; shift 2
  out=trained/xtcav23e_inj_covB2_${arm}_2026-10-20
  [ -e $out ] && { echo "skip $arm"; return; }
  CUDA_VISIBLE_DEVICES=$g PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train "${COMMON[@]}" "$@" \
    --out $out > logs/surrogate/xtcav23e_inj_covB2_${arm}_2026-10-20.log 2>&1
  echo "[$(date +%H:%M)] $arm rc=$?"
}
run 1 curve --lr 3e-5 --curve-head --curve-dims 2 0 --curve-order 8 --curve-head-lr 1e-3 &
run 0 rqs32 --lr 1e-4 --tail-bound 4 --n-bins 32 --init-partial &
run 0 wide --lr 1e-4 --tail-bound 4 --n-bins 32 --hidden-dim 256 --init-partial &
wait
echo "[$(date +%H:%M)] all done"
