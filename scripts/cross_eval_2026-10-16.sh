#!/bin/bash
# Cross-flow eval of the existing delta-controller policies (overnight 2026-10-16): each policy (final weights, fixed
# eval seed 12345, 512 episodes incl. its own off-context goals) on (own) its own env, (golden) the eval flow with the
# injector at golden, (inj) the eval flow with the DAQ-derived injector context. Reward-independent summary lines.
#   bash scripts/cross_eval_2026-10-16.sh <eval_flow_ckpt> <flow_tag> <gpu>
cd "$(dirname "$0")/.."
FLOW=$1; TAG=$2; GPU=${3:-0}
OUT=results/cross_eval_2026-10-16/$TAG; mkdir -p $OUT
R=logs/2026-10-15; S=$R/seeds; A=$R/ablation; N=logs/2026-10-16
POL=(
 "S1 blen s0|$R/delta_xtcav23_cloud_v4r_blen_s0" "S1 blen s1|$S/delta_xtcav23_cloud_v4_blen_s1" "S1 blen s2|$S/delta_xtcav23_cloud_v4_blen_s2"
 "S1 blen s3|$S/delta_xtcav23_cloud_v4_blen_s3" "S1 adapter s0|$R/delta_xtcav23_cloud_v4r_adapter_s0" "S1 adapter s1|$S/delta_xtcav23_cloud_v4_adapter_s1"
 "S1 l1direct v1 (leaky readback) s0|$N/delta_xtcav23_cloud_v4_l1direct_s0" "S1 l1direct v2 s0|$N/delta_xtcav23_cloud_v4_l1direct_v2_s0"
 "S2 v5 blen 300it s0|$R/delta_xtcav23_screen_v5_blen_s0" "S2 v5 blen 600it s0|$A/screen_v5_long600_s0" "S2 v5 blen 600it s1|$A/screen_v5_long600_s1"
 "S2 v5 blen 600it s2|$A/screen_v5_long600_s2" "S2 v5 adapter s0|$R/delta_xtcav23_screen_v5_adapter_s0"
 "S2 v6 blen 600it s0|$N/delta_xtcav23_screen_v6_blen_s0" "S2 v6 l1direct v2 s0|$N/delta_xtcav23_screen_v6_l1direct_v2_s0")
for p in "${POL[@]}"; do
  name=${p%%|*}; L=${p#*|}
  [ -e "$L/final_policy.pt" ] || { echo "$name: (no final policy yet)" >> $OUT/summary.txt; continue; }
  for mode in own golden inj; do
    case $mode in
      own) extra=();;
      golden) extra=(--flow-ckpt "$FLOW" --de-yaml configs/rl/eval/injector_golden_2026-10-16.yaml);;
      inj) extra=(--flow-ckpt "$FLOW" --de-yaml configs/rl/eval/injector_ctx_2026-10-16.yaml);;
    esac
    line=$(scripts/ev_summary.sh cuda:$GPU --logdir "$L" --policy final "${extra[@]}" --out "$OUT/$(basename $L)_$mode" 2>&1 | tail -1)
    echo "$name | $mode | $line" | tee -a $OUT/summary.txt
  done
done
