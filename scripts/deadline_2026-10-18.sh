#!/bin/bash
# Deadline for the realistic RL runs: at the given clock time, a run without final_policy.pt is stopped and its newest
# periodic checkpoint (DeltaBPTTpolicy_iter*.pt, same [actor, obs_rms] pickle) becomes final_policy.pt (TRUNCATED.txt
# records the iteration). Usage: deadline_2026-10-18.sh HH:MM run_name [run_name ...]
cd "$(dirname "$0")/.."
R=logs/2026-10-18/realistic
t=$1; shift
until [ "$(date +%H:%M)" = "$t" ]; do sleep 30; done
for n in "$@"; do
  L=$R/$n
  [ -e $L/final_policy.pt ] && continue
  pid=$(pgrep -f -- "--logdir $L --seed" | head -n 1)
  [ -n "$pid" ] && kill $pid && sleep 5
  last=$(ls $L/DeltaBPTTpolicy_iter*_reward*.pt 2>/dev/null | sed 's/.*iter\([0-9]*\)_.*/\1 &/' | sort -n | tail -n 1 | cut -d' ' -f2)
  [ -n "$last" ] || { echo "[$t] $n: no checkpoint" ; continue; }
  cp "$last" $L/final_policy.pt
  echo "stopped at the $t deadline; final_policy.pt = $(basename $last)" > $L/TRUNCATED.txt
  echo "[$(date +%H:%M)] $n truncated -> $(basename $last)"
done
