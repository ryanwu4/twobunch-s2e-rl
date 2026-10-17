#!/bin/bash
# After the injector coverage campaign (2026-10-16): preprocess old + injector_full + injector_topup + injector_cov
# into a NEW table, check the split (frozen old roles reproduced; new rows split by injector point within their
# campaign; pre-registered holdout -> test; new pairs share a group with their nested row), then fine-tune with
# B2's recipe on the expanded table (Ryan 10-16: "just do B2, trained on the expanded dataset"), then scorecards.
# B2 recipe = warm from A best, w_pair 3, 40 epochs cosine, lr 3e-5, centroid + two-part T; only the table differs.
set -e
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
LOG=logs/injector_cov_2026-10-16
until grep -q "^Done:" $LOG/run.log; do kill -0 $(cat $LOG/run.pid) 2>/dev/null || { echo "campaign driver gone without Done"; exit 1; }; sleep 120; done
echo "[$(date +%H:%M)] campaign: $(tail -n 1 $LOG/run.log)"
TABLE=processed/xtcav23e_inj_cov_2026-10-17.h5
[ -e $TABLE ] || PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 nice -n 10 $PY -m twobunch_s2e_rl.surrogate.preprocess \
  --subdir xtcav23_full,xtcav23_pilot,injector_full,injector_topup,injector_cov_2026-10-16 --sweep-set xtcav23e_l0b2_inj12 \
  --treaty-point XTCAVF --transmission per_sample --P 1024 --out $TABLE --allow-fixed-mismatch num_macro_particles \
  --campaign-json xtcav23_full=results/injector_pilot_2026-10-14/campaign_backfill/xtcav23_full.json \
  --campaign-json xtcav23_pilot=results/injector_pilot_2026-10-14/campaign_backfill/xtcav23_pilot.json > logs/surrogate/preprocess_cov_2026-10-17.log 2>&1
echo "[$(date +%H:%M)] preprocess: $(tail -n 2 logs/surrogate/preprocess_cov_2026-10-17.log | head -n 1)"
PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 $PY - <<'PY'
import json, h5py, numpy as np
from collections import Counter
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
T = "processed/xtcav23e_inj_cov_2026-10-17.h5"
roles = json.load(open("processed/role_map_ftc_s0_2026-10-16.json"))["roles"]
dm = TwoBunchFlowDataModule(processed_h5=T, split="group", orig_split=roles)
dm.setup()                                                    # raises if a split group spans two roles
k = dm.full.row_keys
got = {k[int(i)]: n for n, s in (("train", dm.train), ("val", dm.val), ("test", dm.test)) for i in s.indices}
assert all(got.get(kk) == r for kk, r in roles.items()), "old roles not reproduced"
new = Counter(r for kk, r in got.items() if kk.startswith("injector_cov_2026-10-16/"))
with h5py.File(T, "r") as f:
    camps = f.attrs["campaigns"].split(","); c = f["campaign"][:]; g = f["group"][:]
    codes = f.attrs["block_codes"].split(","); blk = np.array([codes[b] for b in f["block"][:]])
    hold = f["holdout"][:]; lid = f["linac_id"][:]
ci = camps.index("injector_cov_2026-10-16"); m = c == ci
nested_groups = set(g[m & (blk == "nested")]); nested_lid = set(lid[m & (blk == "nested")])
pg = np.flatnonzero(m & (blk == "paired_golden"))
orphans = [i for i in pg if g[i] not in nested_groups]
bad = [i for i in orphans if lid[i] in nested_lid]          # partner present but not grouped: a real leak
assert not bad, f"{len(bad)} new paired_golden rows not grouped with their present nested partner"
if orphans:
    print(f"note: {len(orphans)} paired_golden rows whose nested partner is missing (failed row)")
assert all(got.get(kk) == "test" for kk, h in zip(k, hold) if h), "holdout row outside test"
print(f"old roles reproduced; new campaign rows {int(m.sum())} by role {dict(new)}; "
      f"new holdout {int((m & hold).sum())}; total {len(got)} of {len(k)}")
PY
OUT=trained/xtcav23e_inj_covB2_2026-10-17
if [ ! -e $OUT ]; then
  echo "[$(date +%H:%M)] fine-tune B2-cov (GPU0)"
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=2 nice -n 5 $PY -m twobunch_s2e_rl.surrogate.train \
    --processed $TABLE --split group --role-map processed/role_map_ftc_s0_2026-10-16.json \
    --select-metric val_paired_delta_rms_um --patience 0 \
    --init-ckpt "trained/xtcav23e_inj_pairA40_2026-10-16/checkpoints/best-epoch=034-val_paired_delta_rms_um=6.8576.ckpt" \
    --lr 3e-5 --epochs 40 --lr-schedule cosine --num-workers 4 --seed 0 --w-mean 1.0 --w-spacing 1.0 --save-top-k 3 \
    --t-two-part --w-pair 3.0 --out $OUT > logs/surrogate/xtcav23e_inj_covB2_2026-10-17.log 2>&1 < /dev/null
fi
BEST=$(ls $OUT/checkpoints/best-epoch=*.ckpt | sort -t= -k3 -g | head -n 1)
echo "[$(date +%H:%M)] B2-cov best: $BEST"
SC=results/surrogate_scorecard_2026-10-17
mkdir -p $SC
cat > $SC/models_cov.json <<EOF
{"B2 best": "trained/xtcav23e_inj_topupB2v2_2026-10-16/checkpoints/best-epoch=027-val_paired_delta_rms_um=5.9350.ckpt",
 "B2-cov best": "$BEST",
 "B2-cov final": "$OUT/checkpoints/final.ckpt"}
EOF
# (1) the 10-16 table: same val / test rows as B2's scorecard (apples to apples); (2) the expanded table (more pairs)
for spec in "topup_table:processed/xtcav23e_inj_topup_v2_2026-10-16.h5" "cov_table:$TABLE"; do
  n=${spec%%:*}; t=${spec#*:}
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src OMP_NUM_THREADS=4 $PY scripts/surrogate_scorecard.py --processed $t \
    --role-map processed/role_map_ftc_s0_2026-10-16.json --models $SC/models_cov.json --out $SC/$n > $SC/$n.log 2>&1
  echo "[$(date +%H:%M)] scorecard $n done"
done
echo "[$(date +%H:%M)] after-cov chain done"
