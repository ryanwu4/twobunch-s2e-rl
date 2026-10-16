# Overnight 2026-10-15 → 16: surrogate pair loss + top-up data, RL on the new flow, direct-L1 variant

Ground rules as before: `dev/2026-10-13`, commit per step, never push; new outputs only; FACET2-S2E untouched.

## Plan (agreed with Ryan, 21:45)
- Surrogate: paired-Δ loss + two-part T head; fine-tune from ftc_s0; gate (pairs R² ≥ 0.8, slope 0.85–1.15, p90 ≤ 8 µm; old-val no regression; T calibration ≤ 0.02).
- Data: ~1000 Bmad-only top-up rows (≈ 500 pairs on existing nested rows + 250 new nested draws each with a golden pair), cached injector beams, then preprocess + second fine-tune.
- RL: S1 + S2 v6 (600 iters) blen_dlog on the best flow by ~2 am, 2 seeds each; direct-L1 variant (BC11 loop off) on S1 + S2.
- CPU after the top-up: closed-loop Bmad validation of the S1 / S2 v5 policies.
- Light: BLEN signal model from the saved monitor beams; DAQ / PV review of direct L1 control.
- Stopped the scratch control (21:47).

## Revised plan after the review agent (22:35; review findings in this log's problems section as they are fixed)
1. Freeze ftc_s0's exact row-role map (campaign/idx -> train/val/test) and load it as the split for every new table; top-up rows inherit their partner's role; assert no group spans two roles and that the step-1 table reproduces ftc_s0's split.
2. Pair source: TRAIN-only, non-holdout pairs; crossed (golden-linac) pairs capped / weighted by unique linac point; log counts.
3. Fine-tunes on GPU0 in parallel: A = centroid + pair loss + two-part T (T head own param group lr 1e-3, bias init logit P(T=1)); A2 = centroid + pair loss only. Profile 20 batches first. Select on VAL; scorecard with bootstrap CIs; test quoted once (fixed original 50 test pairs).
4. Fix consumers for multi-campaign tables (campaign-aware pairing and sample-json lookup) before scoring anything with the top-up.
5. After the top-up (~01:45): determinism probe (re-run ~5 existing paired_golden rows, require identical outputs); preprocess old + injector_full + topup with the frozen role map; fine-tune B ~02:00. injector_evalset stays eval-only.
6. RL (Ryan): injector context = DAQ-derived shift-to-shift (gun +-0.5% / +-0.5 deg, others at readback scale, drive fraction over 0.66-0.80 observed with small error). First a cross-eval of ALL existing policies on one fixed eval flow with fixed seeds and reward-independent metrics; then exploratory S1 / S2 v6 training on the val-selected new flow (2 seeds), new config stems / logroot, eval scheduled after each run.
7. Bmad step re-scoped: golden-injector open-loop validate_delta of the S1 / S2 v5 policies (injector-randomized path not implemented).
8. Light: BLEN signal model; DAQ/PV review of direct L1.

## Decisions taken in Ryan's absence
1. Direct-L1 actuator `direct_rb`: step 0.25°, gain U(0.7, 1.3), FAST_PACT readback noise 0.02°, |cum| ≤ 2.5°, L1 drift 0.05°/step (ASSUMED = L2; no BC11 loop to compensate). → v2 after the L1 review (see the progress log).
2. Injector context (Ryan: DAQ-derived shift-to-shift), widths: gun amplitude σ 0.25% / phase σ 0.25° (DAQ run-to-run 0.06–0.47% / 0.18–0.56°); laser timing and d2 delay σ 0.1 ps (hidden); drive fraction uniform 0.75 ± 0.06 clipped to ≤ 0.80 (observed via the measured charge fraction); charge σ 0.3%; passive knobs at readback scale. In `configs/rl/eval/injector_ctx_2026-10-16.yaml`.
3. Fine-tunes at 40 epochs instead of 80 (time; ftc_s0 peaked by epoch 46).

## Problems encountered
1. **zsh launch loop again (22:15):** `$COMMON` in a zsh `for` loop is not word-split, so both pair fine-tunes exited on argparse. Logs moved to the scratchpad; relaunched with `scripts/launch_pair_finetunes_2026-10-16.sh` (bash arrays). Rule from now on: multi-argument launches go through a bash script.

## Progress log
- 22:00 top-up campaign launched (`data/injector_topup`, 1000 rows, 0 Impact-T; worktree `.worktrees/injector_topup_65346c751f`).
- 22:10 review agent; revised plan (above).
- 22:20 role map exported (`processed/role_map_ftc_s0_2026-10-16.json`: 7403 / 826 / 325); preprocess cross-campaign links + block/linac_id datasets; `processed/xtcav23e_inj_v2_2026-10-16.h5` reproduces the old table bit-for-bit and ftc_s0's split exactly; pair source 391 train pairs (188 nested + 203 crossed, crossed ≤ 30% of draws). 290 tests pass.
- 22:30 fine-tunes A (centroid + pair + two-part T) and A2 (centroid + pair) running on GPU0.
- 21:47 scratch control stopped; `direct_rb` actuator + test; direct-L1 configs (S1 cloud, S2 v6).
- 22:40 A / A2 restarted at 40 epochs (`trained/xtcav23e_inj_pair{A,A2}40_2026-10-16`; 80 epochs was ~5 min/epoch with two runs on GPU0, ETA 04:50). The partial 80-epoch dirs are marked ABORTED.
- 22:45 L1 review (agent; `results/l1_direct_review_2026-10-16/REPORT.md`): no L1 multiknob in any DAQ run (all scan L2_PHASE.MKB); the BC11 energy loop appears to act on KLYS_LI11_21 amplitude; FAST_PACT is the klystron phase, so it must not reveal the beam-side gain; loop-off drift is a ramp invisible to PACT; recommends a BC11 BLEN observation. The v1 `direct_rb` leaked the hidden gain: S2 v6 l1direct v1 stopped (ABORTED); the S1 v1 run is kept for reference only. v2 actuator (klystron-side readback gain U(0.95, 1.05), RW 0.025 + ramp U(±0.05)°/step) queued on GPU1 after S2 v6 blen_dlog.
- Cross-flow eval tooling (`eval_delta --flow-ckpt --de-yaml`); injector context fragment `configs/rl/eval/injector_ctx_2026-10-16.yaml` (decision 2). Smoke: the S1 blen_dlog s0 old-flow policy on ftc_s0 + injector context gets 5.6 µm (own env 5.0).
- Scorecard made campaign-aware, with bootstrap CIs; ftc_s0 test-pair R² 0.76, CI [0.57, 0.86] (n = 48).
