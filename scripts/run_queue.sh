#!/bin/bash
# Sequential BPTT queue on one GPU: run_queue.sh <gpu> <logroot> <cfg1> [<cfg2> ...]; logdir = <logroot>/<cfg stem>_s$seed
gpu=$1; root=$2; shift 2; seed=${SEED:-0}
cd "$(dirname "$0")/.."
mkdir -p "$root"
for c in "$@"; do
  n=$(basename "$c" .yaml)
  [ -e "$root/${n}_s$seed" ] && { echo "skip $n (exists)"; continue; }
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=2 PYTHONPATH=$PWD/src nice -n 5 \
    /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.rl.train_bptt --cfg "$c" \
    --logdir "$root/${n}_s$seed" --seed $seed --device cuda:$gpu > "$root/${n}_s$seed.log" 2>&1
  echo "$n rc=$?"
done
