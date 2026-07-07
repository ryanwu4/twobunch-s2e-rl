#!/usr/bin/env bash
# One-shot goal-conditioned sigma_z pipeline: train -> transfer setpoints -> validate in Bmad.
# Minimal chaining of the three existing entry points; nothing new in src/. Steps 1-2 run in the
# `slac-rl` env (GPU, surrogate/RL), step 3 in `bmad-qpad-dev` (CPU, Bmad/Tao tracking) -- so we
# switch conda envs between step 2 and step 3 via `conda run`.
#
# Usage (long -- run in tmux):  bash scripts/run_sigmaz_pipeline.sh
# Override any knob inline, e.g.:  NUM_MACRO=20000 TARGETS=150,250 bash scripts/run_sigmaz_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."                       # repo root (twobunch-s2e-rl)

# ---- tunables ----
CONFIG=${CONFIG:-configs/rl/bptt_gc_sigmaz.yaml}
FLOW_CKPT=${FLOW_CKPT:-trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt}
LOGDIR=${LOGDIR:-logs/bptt_gc_sigmaz}         # must match `logdir:` in $CONFIG
RES=${RES:-results/rl/bptt_gc_sigmaz}
TARGETS=${TARGETS:-100,150,200,250,300}       # comma-separated target spacings (um)
NUM_MACRO=${NUM_MACRO:-100000}                # Bmad macroparticles (20000 for a fast look)
GPU=${GPU:-1}                                 # physical GPU for train + transfer (config device is cuda:0)

export PYTHONPATH="$PWD/src"
export MPLBACKEND=Agg
mkdir -p "$LOGDIR" "$RES/setpoints" "$RES/bmad_validation"

echo "==================================================================="
echo "[1/3] TRAIN (slac-rl, GPU $GPU)   cfg=$CONFIG"
echo "==================================================================="
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n slac-rl \
  python -u -m twobunch_s2e_rl.rl.train_bptt --cfg "$CONFIG" --flow-ckpt "$FLOW_CKPT" \
  2>&1 | tee "$LOGDIR/train.log"

echo "==================================================================="
echo "[2/3] TRANSFER SETPOINTS (slac-rl, GPU $GPU)   targets=$TARGETS"
echo "==================================================================="
CUDA_VISIBLE_DEVICES=$GPU conda run --no-capture-output -n slac-rl \
  python scripts/transfer_setpoints.py --logdir "$LOGDIR" --flow-ckpt "$FLOW_CKPT" \
  --targets-um "$TARGETS" --out "$RES/setpoints" \
  2>&1 | tee "$RES/setpoints/transfer.log"

echo "==================================================================="
echo "[3/3] BMAD VALIDATION (bmad-qpad-dev, CPU)   n=$NUM_MACRO"
echo "==================================================================="
conda run --no-capture-output -n bmad-qpad-dev \
  env PYTHONPATH="$PWD/src" \
  python scripts/validate_bmad.py --setpoints-dir "$RES/setpoints" --out "$RES/bmad_validation" \
  --norm-json processed/twobunch_combined_norm.json --num-macro-particles "$NUM_MACRO" \
  2>&1 | tee "$RES/bmad_validation/validate.log"

echo "==================================================================="
echo "DONE.  setpoints:   $RES/setpoints/setpoints_goal*um.json"
echo "       validation:  $RES/bmad_validation/validation_summary.csv"
echo "==================================================================="
