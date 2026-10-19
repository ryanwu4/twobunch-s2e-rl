#!/bin/bash
# Gradient check (2026-10-18 night): Bmad finite differences (96 single-thread workers, 50k macroparticles), then the
# surrogate autograd / FD comparison. Log logs/2026-10-18/gradient_check.log.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
log() { echo "[$(date +%H:%M)] $*"; }
log "bmad phase start"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 $PY scripts/gradient_check_2026-10-18.py bmad --workers 96 --num-macro 50000 || { log "bmad phase FAILED"; exit 1; }
log "surr phase start"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=16 $PY scripts/gradient_check_2026-10-18.py surr --seeds 8 --n 20000 || { log "surr phase FAILED"; exit 1; }
log "gradient check done"
