# Direct-L1 delta actuator: PV / DAQ evidence review (2026-10-16)

Written by a review agent (read-only scan of all 17 DAQ runs: TEST_15668, E300_15671, E300_15673 on 2026-04-15; E331_16019, 16021–16025, 16027–16030, 16032–16035 on 2026-05-01, via `results/harness_audit_2026-10-14/scripts/common.py::load_mat`). [data] = computed / read here, [prior] = earlier audit, [inf] = inference.

## 1. Which knob a direct-L1 action would write
- **No L1 multiknob anywhere in the evidence.** All 17 runs scan `L2_PHASE.MKB` only; Yiheng: the MKB is L2 only (`ryan_yiheng_convo.txt`); ops question 7 (L1 multiknob?) is open. **No dataset varies L1 directly**: L1 moves only as BC11-loop tracking (0.5–1.6° per run).
- L1 = `KLYS_LI11_11`, `KLYS_LI11_21`; setpoints `KLYS_LI11_{11,21}_SSSB_PDES` (nonBSA, −18.03° in all runs); readbacks `KLYS_LI11_{11,21}_FB_FAST_PACT` / `_FAST_AACT` (BSA) [data]. `LI11_SBST_1_PDES` feeds L2 stations in LI11: not an L1 knob [inf].
- The BC11 loop writes an **unrecorded** PV; PACT − SSSB_PDES spans −0.07° … −7.9° (loop trim) [data/prior].
- **Verdict:** a direct-L1 delta needs (a) a new `L1_PHASE.MKB` from ops (preferred), or (b) a common-mode write of the loop's trim PV / SSSB_PDES with the loop off; ops must name and approve the PV [inf].

## 2. Feedback interactions
- **BC11 BLEN loop must be OFF** (otherwise it cancels the move). At episode end set its setpoint to the measured `BLEN_LI11_359_BRAW` mean before re-closing [inf].
- **BC11 energy loop appears to actuate `KLYS_LI11_21` amplitude** [data + inf; contradicts ops MAJOR 4]:
  - AACT anti-correlates with the common-mode PACT (corr ≤ −0.70 in 10/17 runs, slope −0.49 … −0.88 %/°, close to the pure energy-compensation prediction −0.57 … −0.85 %/°);
  - `KLYS_LI11_11` AACT is flat;
  - `BPMS_LI11_333_X` is nearly independent of PACT;
  - this matches the 2026-06-28 beamtime note.

  So the sim's ideal BC11-energy hold is the right dressing; the open item is the loop's time constant.
- With the loop off, `BLEN_LI11_359_BRAW` is a free beam-side observable: per-shot relative rms 0.038–0.059; d ln S/dφ ≈ 0.29–0.38 /°, so a 0.25° step ≈ 7–10% (~2σ per shot) [data/inf]. It is the only measure of the beam-effective L1 gain and drift.

## 3. Actuator numbers (loop ON in all data)
| quantity | E300 (150 shots/step) | E331 (~50 shots/step) |
|---|---|---|
| per-shot rms of PACT about the step mean | 0.135–0.140° | 0.091–0.114° |
| step-mean scatter (detrended Δ/√2) | 0.11–0.13° | 0.055–0.088° |
| PACT linear drift | −0.043 … −0.023 °/step | −0.015 … +0.021 °/step |
| 11−21 per-shot rms | 0.23° | 0.18–0.23° |

- Loop-off readback noise / gain / drift: **no data**. White estimate ≈ 0.02–0.025° per 30-shot step; the loop-on step scatter (0.055–0.13°) is an upper bound.
- FAST_PACT has dropouts reading exactly 0: drop them before averaging.
- The PACT ramps are the loop cancelling **beam-side** drift, so with the loop off the drift is of that size, directional (ramp-like), and invisible to PACT [inf].

## 4. Recommended `direct_rb` parameters (vs assumed)
| param | assumed (v1) | recommended |
|---|---|---|
| step | 0.25° | 0.25° |
| hidden gain | U(0.7, 1.3), revealed by the readback (**bug**: `d_obs = d_true + noise`) | beam-side U(0.7, 1.3); readback = klystron phase: a·step·g_rf + noise, g_rf ~ U(0.95, 1.05) |
| readback noise | 0.02° | 0.02° (DR up to 0.05°) |
| drift | hidden RW 0.05°/step | hidden RW 0.02–0.03°/step + per-episode ramp U(±0.05)°/step; stress ×2 |
| cum limit | ±2.5° | ±2.5° until ops give one |
| obs | PACT Δ | + BC11 BLEN log-ratio (shot rms ~5%) |

## 5. Questions for ops / Yiheng
1. Is there or can there be an `L1_PHASE.MKB`? Otherwise which PV does the BC11 BLEN loop write, and may we write it with the loop off?
2. Does the BC11 energy feedback actuate `KLYS_LI11_21` amplitude? Rate, gain, time constant, range for ±2.5° of L1?
3. Allowed L1 travel per step and in total, interlocks, how long BC11 BLEN may stay off; bumpless re-close OK?
4. Is SSSB_PDES the setpoint frame of FAST_PACT? Known PACT dropouts / 11-1 vs 11-2 offsets?
5. Can the BC11 loop state, setpoint and the L1 write PV be added to the DAQ?
6. Shift test: 10 min of L1 hold with BC11 BLEN off (drift), then ±0.25° / ±0.5° direct steps (beam gain via BLEN and XTCAV spacing; energy-loop transient on BPMS_LI11_333_X / KLYS_LI11_21 AACT).

## References
`results/harness_audit_2026-10-14/REPORT.md` §§3–4; `results/plan_review_2026-10-14/round2/{ops_review,physics_review}.md`; `/home/rwu4/photoinjector-rl/ryan_yiheng_convo.txt`; `lab-notebook/projects/slac-rl/meetings/2026-06-28_facet_beamtime.md`; DAQ `/home/rwu4/photoinjector-rl/XTCAV_analysis/{TEST_15668,E300_15671,E300_15673,E331_160xx}/*.mat`.
