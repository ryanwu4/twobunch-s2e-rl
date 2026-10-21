#!/bin/bash
# 4096-particle training table (2026-10-20): the 10-17 cov table's command with --P 4096 (new file; the 1024 table stays).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot,injector_full,injector_topup,injector_cov_2026-10-16 --sweep-set xtcav23e_l0b2_inj12 \
  --treaty-point XTCAVF --transmission per_sample --P 4096 --out processed/xtcav23e_inj_cov_P4096_2026-10-20.h5 \
  --allow-fixed-mismatch num_macro_particles \
  --campaign-json xtcav23_full=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_full.json \
  --campaign-json xtcav23_pilot=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_pilot.json
echo "[$(date +%H:%M)] preprocess P4096 rc=$?"
