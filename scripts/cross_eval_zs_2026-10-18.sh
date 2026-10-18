#!/bin/bash
# Cross-eval of the realistic-operation policies on B2-cov + spacing head (standard 32-step eval, DAQ injector context),
# each after its own validation (orchestrator) has run, so it never competes with training for GPU memory.
cd "$(dirname "$0")/.."
ZS=trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt
EV=results/eval_realistic_2026-10-18
for spec in "real_covB2_screen_noblen_s0:0" "real_covB2_screen_bc11loop_s0:1" "real_covB2_screen_noblen_s1:0" "real_covB2_screen_bc11loop_s1:1"; do
  n=${spec%%:*}; g=${spec#*:}
  until grep -q "validated $n" logs/2026-10-18/orchestrate.log 2>/dev/null; do sleep 300; done
  echo "$n | standard on B2-cov+spacing head | $(scripts/ev_summary.sh cuda:$g --logdir logs/2026-10-18/realistic/$n --policy final --set episode_length=32 --set v4.goal_change=null --flow-ckpt $ZS --de-yaml configs/rl/eval/injector_ctx_2026-10-16.yaml --out $EV/${n}_standard_zsflow 2>&1 | tail -n 1)" >> $EV/summary.txt
done
