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
4. **Flow selection (01:40): A best (epoch 34)** = centroid + pair loss + two-part T head. It ties A2 on val pairs (R² 0.73) and wins on T: T calibration 0.022 → 0.0033 (gate passes), witness T MAE 0.048 → 0.029. It is the eval flow for the cross-eval and the warm start for B / B2 (top-up table; w_pair 1 and 3).
3. Fine-tunes at 40 epochs instead of 80 (time; ftc_s0 peaked by epoch 46).

## Problems encountered
2. **Top-up link fields were lost (01:45).** The campaign driver writes a fixed field set per row, so `pair_campaign` / `point_campaign` (manifest-only) never reached the sample jsons. The first merged table had no cross-campaign pairs, and its top-up rows were split by their own random group split: possible val leak via new nested rows of val injector points. The role-map check only verified the old rows, so it passed. B / B2 were stopped after ~10 min (ABORTED marker); the table was renamed `*.BROKEN_no_links.h5`. Fix: preprocess fills the link fields from the campaign's `manifest.json` (test). The v2 table check now also asserts every top-up row shares its group and role with an injector_full row (0 violations). Pairs: train 188 → 794 (+ 203 crossed), val 31 → 94, test 48 → 121.
3. The warm-start reset the trained T head's bias to the data prior (B/B2 first launch). Fixed: only a fresh head gets the prior bias.
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
- 23:00 BLEN model v1 (agent; `results/blen_model_2026-10-16/REPORT.md`):
  - The BC11 BLEN is a **witness-length** monitor above ~0.3 THz.
  - DAQ-consistent detector band ≲ 1 THz; there g ≈ 3.7–4.6 deg/ln at the operating point and 2.8–3.7 at −22 to −24° (upper end of the env's U(1.5, 4), state-dependent ~1.7×).
  - Drive fraction ±0.07 → ∓1° L1 through a closed BC11 loop.
  - BC14 log-ratio worth having as a realised-L2 readback (with a hidden L2 gain: spacing R² 0.69 → 0.89).
  - Env integration (solved BC11 loop, BC14 obs) left for Ryan's review (not tonight).
- 23:00 golden-injector Bmad check: settings exported for S1 blen_dlog s0, S1 adapter s0, S2 v5 blen s0, S2 v5 600-it s0 (24 each; `results/rl/<run>/bmad_check_2026-10-16/`); tracking starts after the top-up frees the CPUs.
- 23:05 **determinism probe passed**: 5 injector_full rows (3 paired_golden, 2 nested) re-run in `data/determinism_probe_2026-10-16` reproduce their seeds, specs and XTCAVF beams bit-for-bit, so the top-up's cross-campaign pairs are exact.
- 23:05 cloud plant reports the drive charge fraction when observed; injector-flow RL config generator (`scripts/make_injflow_rl_configs.py`; S1 obs 57, S2 obs 58, 37 knobs); cross-eval driver (`scripts/cross_eval_2026-10-16.sh`).
- 23:55 top-up 514/1000 (ETA ~01:50). Fine-tunes at epoch 16/40 (ETA ~01:40). Val paired-Δ rms: A 7.62 → 7.13, A2 7.52 → 7.02 µm; A's witness T MAE 0.051 → 0.021 (two-part head). `scripts/after_topup_2026-10-16.sh` is waiting: preprocess of the merged table, frozen-split check, then the golden-injector Bmad check (4 policies × 24, 1e5, 24 workers).
- 00:28 GPU1 (old flow), true median spacing error / within 10 µm, final policies incl. off-context goals:
  - S2 v6 blen_dlog 600 it s0: 7.0 µm / 63% (v5 600 it: 7.2–7.6 / 61–63%), so the −V energy axis is neutral;
  - S1 direct-L1 v1 s0: 5.1 µm / 77% (leaky readback, reference only; blen_dlog 4.6–5.0).
  - Direct-L1 v2 S1 running, then S2 v6 v2.
- 01:40 scorecard (`results/surrogate_scorecard_2026-10-16/A/scorecard.md`, v2 table, frozen split), ftc_s0 → A best:
  - val pairs (selection, n = 31) R² 0.68 → 0.73, slope 0.61 → 0.64;
  - test pairs (n = 48, quoted once) R² 0.76 [0.56, 0.87] → 0.84 [0.70, 0.90], slope 0.68 → 0.72, p90 12.7 → 10.3 µm;
  - train pairs R² 0.74 → 0.85, slope 0.72 → 0.81;
  - golden-linac effect R² 0.98 → 0.99, slope 1.05 → 1.01;
  - old-val spacing MAE 4.38 → 4.32 µm;
  - T calibration 0.022 → 0.0033.

  Gates: old-val ✓, T ✓, pairs ✗ (slope < 0.85, p90 > 8 µm). A2 (pair loss only) matches A on everything but T.
- 01:30 top-up done (1000 ok, 0 failed, 3.7 h). Bmad tracking of the golden-injector check started 01:34.
- 01:55 B / B2 relaunched on `processed/xtcav23e_inj_topup_v2_2026-10-16.h5`; cross-eval of all policies on flow A started (GPU0).
- 02:25 direct-L1 v2 (old flow, seed 0, true median / within 10 µm): S1 6.0 µm / 73% (blen_dlog 4.6–5.0 / 80–83%); S2 v6 8.1 µm / 58% (blen_dlog 7.0 / 63%). About 1 µm worse than blen_dlog: the beam-side gain is hidden from the readback and the hidden drift ramps (no BC11 loop). The BC11 BLEN observation the L1 review recommends (now modellable, `results/blen_model_2026-10-16`) is the natural next step for this variant. GPU1 → exploratory injector-flow S1 / S2 on flow A.
- 03:05 **golden-injector Bmad check** (open loop, 24 final settings per policy, 1e5 particles, CPU; `results/rl/<run>/bmad_check_2026-10-16/compare.json`). Bmad |spacing − goal| median / drive σz rel error vs goal; surrogate-vs-Bmad spacing rel error:

  | policy | spacing vs goal | σz vs goal | surrogate vs Bmad |
  |---|---|---|---|
  | S1 blen_dlog s0 | 7.8 µm | 4.3% | 2.3% |
  | S1 adapter s0 | 7.2 µm | 4.0% | 2.1% |
  | S2 v5 blen s0 | 7.1 µm | 15% | 1.9% |
  | S2 v5 600-it s0 | 7.9 µm | 14% | 2.4% |

  All 24/24 tracked. Consistent with the v3 Bmad result (6–7 µm): the v4/v5 policies transfer to Bmad at the golden injector.
- 04:05 **exploratory RL on flow A** (injector context, drive charge fraction observed, T hinge 0.98; seed 0; true median / within 10 µm):
  - S1 blen_dlog: 5.2 µm / 80% (injector at golden 5.3);
  - S2 v6 blen_dlog: 8.1 / 57% (golden 8.1).

  No gain over the old-flow policies evaluated on flow A (S1 4.9–5.9). Note: flow A's two-part T head predicts T ≈ 1.000 across the controller box, so the T hinge is effectively inactive there (it no longer reports the small mean losses the MSE head did). Seed 1 of both is running on GPU1.
- 04:30 **cross-eval complete** (`results/cross_eval_2026-10-16/flowA/TABLE.md`): S1 policies are robust to DAQ-level injector variation on flow A (≤ +0.5 µm); S2 loses 1–2 µm (600-it 7.2–7.4 → 7.9–9.2 µm; v6 7.3 → 8.7; direct-L1 v2 S2 8.0 → 11.2). The flow-A-trained exploratory S2 (8.1 µm) is in the same range: no gain yet from training on the injector-aware flow (seed 1 running).
- 05:40 exploratory injflow seed 1: S1 4.7 µm / 81%, S2 v6 8.5 / 57%. Two seeds: S1 4.7–5.2 µm, S2 8.1–8.5 µm, in line with the old-flow policies evaluated on flow A + injector context (S1 4.9–5.9, S2 7.9–9.2). GPU1 idle now; B / B2 on GPU0 until ~07:15.
- 06:30 **B / B2 scorecard** (`results/surrogate_scorecard_2026-10-16/B/scorecard.md`; top-up v2 table, frozen split; val 93 pairs = selection, test 117, original-48 subset):

  | pairs | ftc_s0 | A | B (w_pair 1) | **B2 (w_pair 3)** | gate |
  |---|---|---|---|---|---|
  | val R² [90% CI] | 0.74 | 0.77 | 0.80 | **0.81 [0.73, 0.87]** | ≥ 0.8 ✓ |
  | val slope | 0.71 | 0.74 | 0.77 | **0.79** | 0.85–1.15 ✗ |
  | val p90 | 12.3 µm | 11.9 | 11.4 | **8.8** | ≤ 8 ✗ |
  | test R² / slope / p90 (once) | 0.75 / 0.69 / 12.0 | 0.82 / 0.75 / 9.3 | 0.86 / 0.78 / 9.7 | **0.88 / 0.82 / 8.1** | |
  | train R² / slope | 0.71 / 0.70 | 0.78 / 0.76 | 0.84 / 0.81 | 0.88 / 0.85 | |

  B2: old-val spacing 4.33 µm ✓, σz 0.109 ≤ 0.112 ✓, T calibration 0.0035 ✓, other observables unchanged. Both the top-up data (A → B) and the pair weight (B → B2) help.
- 06:35 decision 5: time-boxed pair-weight push from B2 (C = w_pair 6, C2 = w_pair 10; 25 epochs; GPU0; ETA ~09:45) to try the slope gate. B2 is the current best flow regardless.
- 07:10 direct-L1 v2 on flow A + injector context (seed 0): S1 5.4 µm / 78% (blen_dlog 4.7–5.2); **S2 v6 12.8 / 43%** (blen_dlog 8.1–8.5). Training completed (600 it); the loss was ~5% above blen_dlog's. In screen mode the direct-L1 variant has no beam-side L1 signal (the beam gain is hidden from FAST_PACT and the drift ramp is invisible), and injector variation adds L1-like shifts; the BC11 BLEN observation is the missing piece.
