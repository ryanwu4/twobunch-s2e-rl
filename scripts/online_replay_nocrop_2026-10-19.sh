#!/bin/bash
# Real-data check of "no crop" (2026-10-19): the deployed online analysis (replay v6 settings: step-median cut, crop-edge
# check) on the three 04-15 scans with (a) the auto-crop off, hand crop kept, (b) both crops off (full camera frame).
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python
for v in "v7_noauto:--no-auto-crop" "v7_fullframe:--no-auto-crop --no-hand-crop"; do
  tag=${v%%:*}; flags=${v#*:}
  O=results/4_daq_realism/online_replay_${tag}_2026-10-19; mkdir -p $O
  for d in E300_15671 E300_15673 TEST_15668; do
    ( PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nice -n 10 $PY scripts/xtcav_online_replay.py $d --out $O $flags --qa-steps 3,12,24 > $O/$d.log 2>&1
      echo "[$(date +%H:%M)] $tag $d rc=$?" ) &
  done
done
wait
echo "[$(date +%H:%M)] all done"
