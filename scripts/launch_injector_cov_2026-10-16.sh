#!/usr/bin/env bash
# Launch the injector coverage campaign (configs/datagen/injector_cov_2026-10-16.yaml) from a worktree
# pinned to the current HEAD (later edits in the main checkout can't change the code mid-run).
# Resumable: rerun this script (finished rows / cached beams are skipped).
set -euo pipefail
REPO=/home/rwu4/photoinjector-rl/twobunch-s2e-rl
PY=/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python
CFG=configs/datagen/injector_cov_2026-10-16.yaml
SHA=$(git -C "$REPO" rev-parse HEAD)
WT=$REPO/.worktrees/injector_cov_${SHA:0:10}
LOG=$REPO/logs/injector_cov_2026-10-16
mkdir -p "$LOG"
[ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" "$SHA"
# output_dir must be absolute (the run is from the worktree)
sed "s#^output_dir: data/#output_dir: $REPO/data/#" "$WT/$CFG" > "$LOG/launch_config.yaml"
cd "$WT"
PYTHONPATH=$WT/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup nice -n 10 \
  "$PY" -u -m twobunch_s2e_rl.datagen.injector_campaign "$LOG/launch_config.yaml" >> "$LOG/run.log" 2>&1 &
echo $! > "$LOG/run.pid"
echo "launched pid $(cat "$LOG/run.pid") from $WT ($SHA)"
