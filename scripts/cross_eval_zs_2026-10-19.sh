#!/bin/bash
# Common-plant comparison (2026-10-19 night): every jitter-trained policy (B2-cov flow and spacing-head flow) evaluated
# under nominal jitter on the SPACING-HEAD flow (the more accurate surrogate). Args: run dirs. Appends to
# results/3_rl_eval/eval_night_2026-10-18/cross_zs.txt.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
ZS=trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt
OUT=results/3_rl_eval/eval_night_2026-10-18/cross_zs; mkdir -p $OUT
g=${GPU:-0}
for d in "$@"; do
  n=$(basename $d)
  echo "$n | nominal on zs flow | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir $d --flow $ZS --device cuda:$g --de-yaml configs/rl/eval/jitter_nominal_2026-10-18.yaml --out $OUT/$n 2>&1 | grep '^\[')" >> results/3_rl_eval/eval_night_2026-10-18/cross_zs.txt
done
