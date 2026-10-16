# Overnight 2026-10-15 → 16: surrogate pair loss + top-up data, RL on the new flow, direct-L1 variant

Ground rules as before: `dev/2026-10-13`, commit per step, never push; new outputs only; FACET2-S2E untouched.

## Plan (agreed with Ryan, 21:45)
- Surrogate: paired-Δ loss + two-part T head; fine-tune from ftc_s0; gate (pairs R² ≥ 0.8, slope 0.85–1.15, p90 ≤ 8 µm; old-val no regression; T calibration ≤ 0.02).
- Data: ~1000 Bmad-only top-up rows (≈ 500 pairs on existing nested rows + 250 new nested draws each with a golden pair), cached injector beams, then preprocess + second fine-tune.
- RL: S1 + S2 v6 (600 iters) blen_dlog on the best flow by ~2 am, 2 seeds each; direct-L1 variant (BC11 loop off) on S1 + S2.
- CPU after the top-up: closed-loop Bmad validation of the S1 / S2 v5 policies.
- Light: BLEN signal model from the saved monitor beams; DAQ / PV review of direct L1 control.
- Stopped the scratch control (21:47).

## Decisions taken in Ryan's absence
1. Direct-L1 actuator `direct_rb`: step 0.25°, gain U(0.7, 1.3), FAST_PACT readback noise 0.02°, |cum| ≤ 2.5°, L1 drift 0.05°/step (ASSUMED = L2; no BC11 loop to compensate).

## Problems encountered

## Progress log
- 21:47 scratch control stopped; `direct_rb` actuator + test; direct-L1 configs (S1 cloud, S2 v6).
