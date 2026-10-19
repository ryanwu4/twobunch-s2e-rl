#!/bin/bash
# The old blen_dlog baseline under the jitter conditions (its config predates continuous operation, so the same
# realistic overrides as its 10-17 baseline run are set); after the jitter training frees GPU memory.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
EV=results/3_rl_eval/eval_jitter_2026-10-18
until grep -qs "jitter training done" logs/2026-10-18/jitter_chain.log; do sleep 120; done
SETS=(--set 'v4.goal_change=[16,48]' --set 'v4.drift_ramp.L2PhaseSet=0.04' --set 'v4.jumps={"L2PhaseSet": [0.02, 0.45]}'
      --set 'v4.window={"L1PhaseSet": [-26.5, -18.5], "L2PhaseSet": [-41.0, -33.0]}')
for c in nominal stress; do
  echo "old_blen_dlog | $c | $(PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 $PY -m twobunch_s2e_rl.rl.eval_continuous --logdir logs/2026-10-16/injflow/injflow_flowA_screen_v6_blen_s0 --device cuda:1 --de-yaml configs/rl/eval/jitter_${c}_2026-10-18.yaml "${SETS[@]}" --out $EV/old_blen_dlog_$c 2>&1 | grep '^\[')" >> $EV/summary.txt
done
