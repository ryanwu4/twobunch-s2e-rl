#!/bin/bash
# After the compression sweeps (2026-10-21): merged 4096-particle table (the cov table's campaigns + compress_sweeps) and its
# full-particle truth; also the sweeps alone (for the val-sweep evaluation). New files only.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
until grep -q "^Done:" logs/compress_sweeps_2026-10-21/run.log; do sleep 120; done
echo "[$(date +%H:%M)] sweeps: $(tail -n 1 logs/compress_sweeps_2026-10-21/run.log)"
COMMON=(--sweep-set xtcav23e_l0b2_inj12 --treaty-point XTCAVF --transmission per_sample --P 4096 --allow-fixed-mismatch num_macro_particles
  --campaign-json xtcav23_full=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_full.json
  --campaign-json xtcav23_pilot=results/1_datagen/injector_pilot_2026-10-14/campaign_backfill/xtcav23_pilot.json)
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nice -n 10 $PY -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot,injector_full,injector_topup,injector_cov_2026-10-16,compress_sweeps_2026-10-21 \
  "${COMMON[@]}" --out processed/xtcav23e_inj_cov_sw_P4096_2026-10-21.h5 > logs/surrogate/preprocess_cov_sw_P4096_2026-10-21.log 2>&1
echo "[$(date +%H:%M)] merged table rc=$?"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 $PY scripts/build_truth_table_2026-10-21.py --table processed/xtcav23e_inj_cov_sw_P4096_2026-10-21.h5 --workers 48
echo "[$(date +%H:%M)] merged truth done"
