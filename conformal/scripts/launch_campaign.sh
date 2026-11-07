#!/usr/bin/env bash
# Launch (or resume) a calibration campaign with the generator pinned at COMMIT, from a detached worktree, so
# later edits to the branch can't change the code mid-run. Logs go to the campaign's output_dir (run.log, run.pid).
#   conformal/scripts/launch_campaign.sh conformal/datasets/<name> <commit> [extra injector_campaign args]
set -euo pipefail
DS=$(realpath "$1"); SHA=$2; shift 2
REPO=/home/rwu4/photoinjector-rl/twobunch-s2e-rl
PY=/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python
SHA=$(git -C "$REPO" rev-parse "$SHA")
WT=$REPO/.worktrees/conformal_run_${SHA:0:10}
[ -d "$WT" ] || git -C "$REPO" worktree add --detach "$WT" "$SHA"
OUT=$("$PY" -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$DS/campaign.yaml")
mkdir -p "$OUT"
cd "$WT"
PYTHONPATH=$WT/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 setsid nohup nice -n 10 \
  "$PY" -u -m twobunch_s2e_rl.datagen.injector_campaign "$DS/campaign.yaml" "$@" >> "$OUT/run.log" 2>&1 &
echo $! > "$OUT/run.pid"
echo "launched pid $(cat "$OUT/run.pid") from $WT"
