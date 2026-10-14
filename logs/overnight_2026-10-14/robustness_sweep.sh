#!/bin/bash
# Robustness sweep of the overnight delta controllers (eval only, conditions outside training).
cd /home/rwu4/photoinjector-rl/twobunch-s2e-rl
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHONPATH=$PWD/src
PY="nice -n 10 /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.rl.eval_delta --n-episodes 256"
L=logs/overnight_2026-10-14
$PY --logdir $L/delta_xtcav23_cloud_s0  --set gain_halfwidth=0.4 --tag gain0p4
$PY --logdir $L/delta_xtcav23_screen_s0 --set gain_halfwidth=0.4 --tag gain0p4
$PY --logdir $L/delta_xtcav23_screen_s0 --set measurement.scale_error_halfwidth=0.3 --tag scale0p3
$PY --logdir $L/delta_xtcav23_screen_s0 --set measurement.shots_per_step=10 --tag shots10
$PY --logdir $L/delta_xtcav23_screen_s0 --set "measurement.feature_offset_sigma={drive_charge_fraction: 0.035}" --tag injoffset
echo SWEEP_DONE
