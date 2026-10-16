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
1. Direct-L1 actuator `direct_rb`: step 0.25°, gain U(0.7, 1.3), FAST_PACT readback noise 0.02°, |cum| ≤ 2.5°, L1 drift 0.05°/step (ASSUMED = L2; no BC11 loop to compensate).

## Problems encountered
1. **zsh launch loop again (22:15):** `$COMMON` in a zsh `for` loop is not word-split, so both pair fine-tunes exited on argparse. Logs moved to the scratchpad; relaunched with `scripts/launch_pair_finetunes_2026-10-16.sh` (bash arrays). Rule from now on: multi-argument launches go through a bash script.

## Progress log
- 22:00 top-up campaign launched (`data/injector_topup`, 1000 rows, 0 Impact-T; worktree `.worktrees/injector_topup_65346c751f`).
- 22:10 review agent; revised plan (above).
- 22:20 role map exported (`processed/role_map_ftc_s0_2026-10-16.json`: 7403 / 826 / 325); preprocess cross-campaign links + block/linac_id datasets; `processed/xtcav23e_inj_v2_2026-10-16.h5` reproduces the old table bit-for-bit and ftc_s0's split exactly; pair source 391 train pairs (188 nested + 203 crossed, crossed ≤ 30% of draws). 290 tests pass.
- 22:30 fine-tunes A (centroid + pair + two-part T) and A2 (centroid + pair) running on GPU0.
- 21:47 scratch control stopped; `direct_rb` actuator + test; direct-L1 configs (S1 cloud, S2 v6).
