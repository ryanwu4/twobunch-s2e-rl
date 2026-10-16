#!/bin/bash
# After the injector top-up (overnight 2026-10-16): preprocess old + injector_full + injector_topup into a NEW table,
# check the frozen role map (top-up rows inherit their partner's role; no split group spans roles), then the
# golden-injector open-loop Bmad check of 4 policies on the freed CPUs.
set -e
cd "$(dirname "$0")/.."
LOG=logs/injector_topup
until grep -q "^Done:" $LOG/run.log; do kill -0 $(cat $LOG/run.pid) 2>/dev/null || { echo "top-up driver gone without Done"; exit 1; }; sleep 60; done
echo "[$(date +%H:%M)] top-up: $(tail -1 $LOG/run.log)"
OUT=processed/xtcav23e_inj_topup_2026-10-16.h5
[ -e $OUT ] || PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot,injector_full,injector_topup --sweep-set xtcav23e_l0b2_inj12 --treaty-point XTCAVF \
  --transmission per_sample --P 1024 --out $OUT --allow-fixed-mismatch num_macro_particles \
  --campaign-json xtcav23_full=results/injector_pilot_2026-10-14/campaign_backfill/xtcav23_full.json \
  --campaign-json xtcav23_pilot=results/injector_pilot_2026-10-14/campaign_backfill/xtcav23_pilot.json > logs/surrogate/preprocess_topup_2026-10-16.log 2>&1
echo "[$(date +%H:%M)] preprocess: $(tail -2 logs/surrogate/preprocess_topup_2026-10-16.log | head -1)"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 /home/rwu4/miniconda3/envs/slac-rl/bin/python - <<'PY'
import json, numpy as np
from collections import Counter
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
roles = json.load(open("processed/role_map_ftc_s0_2026-10-16.json"))["roles"]
dm = TwoBunchFlowDataModule(processed_h5="processed/xtcav23e_inj_topup_2026-10-16.h5", split="group", orig_split=roles)
dm.setup()                                                    # raises if a split group spans two roles
k = dm.full.row_keys
got = {k[int(i)]: n for n, s in (("train", dm.train), ("val", dm.val), ("test", dm.test)) for i in s.indices}
same = all(got.get(kk) == r for kk, r in roles.items())
top = Counter(r for kk, r in got.items() if kk.startswith("injector_topup/"))
print("old roles reproduced:", same, "| top-up rows by role:", dict(top), "| total", len(got), "of", len(k))
assert same
PY
for L in delta_xtcav23_cloud_v4r_blen_s0 delta_xtcav23_cloud_v4r_adapter_s0 delta_xtcav23_screen_v5_blen_s0 screen_v5_long600_s0; do
  case $L in screen_v5_long600_s0) D=logs/2026-10-15/ablation/$L;; *) D=logs/2026-10-15/$L;; esac
  echo "[$(date +%H:%M)] bmad track $L"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src nice -n 10 /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    -m twobunch_s2e_rl.rl.validate_delta track --logdir $D --num-macro 100000 --workers 24 --tag 2026-10-16 > logs/injector_topup/bmad_track_$L.log 2>&1
  PYTHONPATH=$PWD/src /home/rwu4/miniconda3/envs/slac-rl/bin/python -m twobunch_s2e_rl.rl.validate_delta compare --logdir $D --tag 2026-10-16 > logs/injector_topup/bmad_compare_$L.log 2>&1 || true
done
echo "[$(date +%H:%M)] after-topup chain done"
