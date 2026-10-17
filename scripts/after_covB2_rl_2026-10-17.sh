#!/bin/bash
# RL retrain on B2-cov (Ryan 10-16: direct-L1 + BLEN obs only): waits for the after-cov chain (B2-cov fine-tune +
# scorecards), writes the injector-flow configs for B2-cov best, adds the BLEN obs (direct-L1 + BC11 / BC14), trains
# S2 v6 and S1 x 2 seeds (GPU0: S2 s0 -> S1 s0, GPU1: S2 s1 -> S1 s1), then evals each on its own env (B2-cov +
# DAQ injector context) and at the golden injector; plus the old best S2 / S1 blen_dlog injflow policies on B2-cov
# for reference.
cd "$(dirname "$0")/.."
PY=/home/rwu4/miniconda3/envs/slac-rl/bin/python
C=logs/injector_cov_2026-10-16
until grep -q "after-cov chain done" $C/after_chain.log 2>/dev/null; do
  kill -0 $(cat $C/after_chain.pid) 2>/dev/null || { echo "after-cov chain gone without done"; exit 1; }; sleep 120; done
OUT=trained/xtcav23e_inj_covB2_2026-10-17
BEST=$(ls $OUT/checkpoints/best-epoch=*.ckpt | sort -t= -k3 -g | head -n 1)
echo "[$(date +%H:%M)] B2-cov best $BEST"
$PY scripts/make_injflow_rl_configs.py --flow "$BEST" --tag covB2 || exit 1
$PY scripts/make_l1blen_rl_configs.py --src configs/rl/injflow_2026-10-16/injflow_covB2_screen_v6_l1direct_v2.yaml \
  configs/rl/injflow_2026-10-16/injflow_covB2_cloud_l1direct_v2.yaml || exit 1
R=logs/2026-10-17/covB2
mkdir -p $R
S2=configs/rl/l1blen_2026-10-16/injflow_covB2_screen_v6_l1blen.yaml
S1=configs/rl/l1blen_2026-10-16/injflow_covB2_cloud_l1blen.yaml
(SEED=0 bash scripts/run_queue.sh 0 $R $S2 $S1) > $R/gpu0.log 2>&1 &
(SEED=1 bash scripts/run_queue.sh 1 $R $S2 $S1) > $R/gpu1.log 2>&1 &
wait
echo "[$(date +%H:%M)] training done"
EV=results/cross_eval_2026-10-17/covB2; mkdir -p $EV
for L in $R/injflow_covB2_screen_v6_l1blen_s0 $R/injflow_covB2_screen_v6_l1blen_s1 $R/injflow_covB2_cloud_l1blen_s0 \
         $R/injflow_covB2_cloud_l1blen_s1 logs/2026-10-16/injflow/injflow_flowA_screen_v6_blen_s0 \
         logs/2026-10-16/injflow/injflow_flowA_cloud_blen_s1 logs/2026-10-16/l1blen/injflow_flowA_screen_v6_l1blen_s0 \
         logs/2026-10-16/l1blen/injflow_flowA_cloud_l1blen_s0; do
  [ -e "$L/final_policy.pt" ] || { echo "$(basename $L): no final policy" >> $EV/summary.txt; continue; }
  for mode in inj golden; do
    case $mode in
      inj) y=configs/rl/eval/injector_ctx_2026-10-16.yaml;;
      golden) y=configs/rl/eval/injector_golden_2026-10-16.yaml;;
    esac
    line=$(scripts/ev_summary.sh cuda:0 --logdir "$L" --policy final --flow-ckpt "$BEST" --de-yaml $y --out "$EV/$(basename $L)_$mode" 2>&1 | tail -n 1)
    echo "$(basename $L) | covB2 $mode | $line" | tee -a $EV/summary.txt
  done
done
echo "[$(date +%H:%M)] covB2 RL chain done"
