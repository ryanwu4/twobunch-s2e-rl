#!/usr/bin/env bash
# Tier 1: run the three matched L2 phase scans concurrently with bounded threads.
#
# 3 datasets x BMAD_OMP_THREADS=20 = 60 OpenMP threads total (cap 64). Bmad's
# OpenMP scaling saturates well below 32 threads, so per-run slowdown vs 32
# threads is small; expected wall time ~25-30 min for all three (vs ~65-70 min
# serial at 32 threads).
#
# Usage (from twobunch-s2e-rl root):
#   bash scripts/run_tier1_scans.sh [L2REF] [QUADMAP]
# Defaults: L2REF=-38, QUADMAP=value_match. Logs + outputs land under
# XTCAV_analysis/sim2real/results/tier1_sim_scans/{DATASET}/.

set -u
cd "$(dirname "$0")/.."

L2REF="${1:--38}"
QUADMAP="${2:-value_match}"
THREADS_PER_RUN=20
PY=/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python
OUT_ROOT=/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results/tier1_sim_scans
DATASETS=(E300_15671 TEST_15668 E300_15673)

declare -A PIDS
for DS in "${DATASETS[@]}"; do
    mkdir -p "$OUT_ROOT/$DS"
    LOG="$OUT_ROOT/$DS/run_l2ref${L2REF}_${QUADMAP}.log"
    echo "launching $DS (threads=$THREADS_PER_RUN, l2ref=$L2REF, quad-map=$QUADMAP) -> $LOG"
    BMAD_OMP_THREADS=$THREADS_PER_RUN PYTHONPATH=$PWD/src MPLBACKEND=Agg \
        "$PY" -u scripts/xtcav_sim_scan.py "$DS" \
        --l2ref "$L2REF" --quad-map "$QUADMAP" \
        > "$LOG" 2>&1 &
    PIDS[$DS]=$!
done

FAIL=0
for DS in "${DATASETS[@]}"; do
    if wait "${PIDS[$DS]}"; then
        echo "$DS: DONE  ($(grep -o 'done in [0-9]* s' "$OUT_ROOT/$DS/run_l2ref${L2REF}_${QUADMAP}.log" | tail -1))"
    else
        echo "$DS: FAILED (see $OUT_ROOT/$DS/run_l2ref${L2REF}_${QUADMAP}.log)"
        FAIL=1
    fi
done

echo
for DS in "${DATASETS[@]}"; do
    grep -E "cal_scaled_sim|step [0-9]+: track" \
        "$OUT_ROOT/$DS/run_l2ref${L2REF}_${QUADMAP}.log" | tail -3 | sed "s/^/[$DS] /"
done
exit $FAIL
