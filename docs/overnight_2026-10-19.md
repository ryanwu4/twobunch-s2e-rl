# Overnight 2026-10-18 → 19

Plan agreed with Ryan 18:40 (back ~08:00). Chains: `scripts/gradient_check_2026-10-18.sh` (log `logs/2026-10-18/gradient_check.log`), `scripts/night_2026-10-18.sh` (log `logs/2026-10-18/night.log`; waits for the jitter chain).

## Decisions (Ryan)
- Train BOTH directions (both loops off = noblen; BC11 loop on = bc11loop v3) on the spacing-head flow, ± failure detection, 2 seeds, 900 it → 8 runs, 2 per GPU (~10 h). No closed-loop Bmad on these tonight.
- Gradient check = research, not a deployment gate.
- Goal conditioning: σz "mixed + mask" (spacing-only setpoints with a mask bit), setpoints "mix local + random". Build + smoke tonight, train tomorrow.

## A. Screen failure detection (done)
`scripts/screen_failure_detect_2026-10-18.py` → `results/2_surrogate_screen/screen_failure_detect_2026-10-18/summary.json`. 688 test rows, failure = |screen − own cloud spacing| > 30 µm or misassign > 0.15, among deployed-valid readings.

| source | valid | fail rate | AUC cut_sens | AUC charge-frac dev | AUC combined |
|---|---|---|---|---|---|
| Bmad | 356 | 0.37 | 0.78 | 0.88 | 0.91 |
| spacing head | 344 | 0.47 | 0.84 | 0.93 | 0.95 |

- P(fail | cut_sens) is similar in both: 0–2 µm 24% / 24%, 2–5 µm 62% / 80%.
- **Decision (mine):** the surrogate's screen already fails at least as often as Bmad's with the same diagnostic signature, so no synthetic failure injection; the failure-detection arm only *observes* `screen_cut_sens` (± 2 px cut shift, 1.5 µm/shot noise; fixed z-score mean 2 / std 3 µm). The measured drive charge fraction (the stronger flag) was already observed.
- Real DAQ steps carry `cut_sens_um`; valid-step distributions match the sim (TEST_15668 median 1.0 µm, E300_15673 4.3 µm vs Bmad 1.1 µm). `daq_replay` now feeds it to the policy.

## B. Night trainings (queued)
Configs `configs/rl/night_2026-10-18/zs_screen_{noblen,bc11loop}_jit[_fd].yaml` = the `*_jit` configs with the spacing-head flow (`covB2zs`), `_fd` + `screen_cut_sens`. Slots: GPU0 {noblen s0 → noblen_fd s0}, {noblen_fd s1 → noblen s1}; GPU1 same for bc11loop. Each run evaluated when done (nominal / stress jitter, DAQ replay TEST_15668 + E300_15673) → `results/3_rl_eval/eval_night_2026-10-18/summary.txt`.

## C. Gradient check (running)
`scripts/gradient_check_2026-10-18.py`: 3 injector states (golden, held-out lhsloc102, lhs123) × 4 (L1, L2) points; Bmad central differences at h = 0.3 / 0.6° (108 runs, 50k macroparticles, same beam + Tao seed); surrogate autograd + FD (8 seeds, 20k particles) for B2-cov and the spacing head. Features: beam spacing / σz and screen spacing / σz.
- Smoke (1 point): spacing slopes 54 (L1) / 39 (L2) µm/deg in Bmad; surrogate ratio 0.95–1.02. Screen σz vs L1: surrogate ratio ~0.2 — check on the full set.

## D. Closed-loop Bmad at held-out injectors (queued, after the jitter chain)
New `closed_loop_bmad --mode test --points-json configs/rl/eval/heldout_injectors_2026-10-18.json`: 12 DAQ-like + 11 full-box test-split injector points + golden (none seen in training). Best jitter-trained seed per direction (by nominal settled spacing), + the paired surrogate run.

## E. Goal conditioning (built, untrained)
- `DeltaControlEnvV4(goal_far_frac=…, spacing_only_frac=…, free_goal_keys=("drive_sigma_z",))`, off by default (RNG stream unchanged). Far setpoints: uniform over the start box. Spacing-only: σz target achieved := goal in the reward view (no penalty, no gradient), σz goal / error obs zeroed, mask bit appended; mask redrawn with every setpoint. `eval_continuous` reports free / constrained segments separately, σz metrics on constrained only. Tests: `test_far_goals_leave_the_goal_radius`, `test_spacing_only_goals_mask_sigma_z`.
- Configs `configs/rl/gc_2026-10-19/gc_zs_screen_{noblen,bc11loop}_jit.yaml` (0.5 / 0.5; rebased 09:00 on the plain arm, the `_fd` versions were smoke-trained and removed).
- Large-move eval of the current policy (pre-check, noblen s0, nominal jitter, full-box setpoints): 13.2 µm settled / 36% within 10 µm / 78% reached vs 11.0 / 45% / 86% for local setpoints. The night chain repeats this for the best jitter-trained policies (radius ×2 and full box).

## Results

### C. Gradient check (done 19:52) — `results/2_surrogate_screen/gradient_check_2026-10-18/summary.json`
Ratio = surrogate / Bmad slope (Bmad FD at h = 0.3°; FD 0.3 vs 0.6° agree to 0.1% for beam spacing, 3% beam σz, 21% screen σz). Median [p10, p90] over 12 points × 2 controls; spacing head (B2-cov the same except where noted).

| feature | Bmad median \|slope\| L1 / L2 | ratio (autograd) | per control L1 / L2 | sign agree | corr |
|---|---|---|---|---|---|
| beam spacing | 49 / 41 µm/deg | 0.99 [0.90, 1.01] | 0.97 / 0.99 | 1.00 | 1.000 |
| beam σz (d ln σz) | 0.07 / 0.23 /deg | 0.96 [0.27, 1.11] | 0.88 / 0.96 | 0.96 | 0.95 |
| screen spacing | 45 / 39 µm/deg | 1.00 [0.43, 4.2] | 1.00 / 1.00 | 0.92 | 0.82 |
| screen σz | 0.21 / 0.19 /deg | 0.59 [−0.4, 4.4] | **0.23** / 1.08 | 0.79 | 0.65 |

- **Beam level: the surrogate's gradients are right.** Spacing slopes within 1–3%, every sign right; σz within ~10%. The spacing head changes the spacing ratio from 0.98 / 1.01 (B2-cov) to 0.97 / 0.99 and screen spacing from 0.94 / 1.01 to 1.00 / 1.00.
- **Screen spacing:** right at the median, but with a tail (p90 ratio 4) at a few points near the split / validity edge.
- **Screen σz vs L1 is the gap.** Bmad's *measured* σz responds to L1 3–6× more than the beam's true σz (e.g. 0.29 vs 0.04 /deg at L1 −20, L2 −35, where the screen also reads 39 µm for a 25 µm beam). The surrogate's screen follows the true beam instead (0.10). At the compressed ~70 µm-spacing points (σz ~20 µm) the surrogate's screen σz slope has the wrong sign for both L1 and L2.
- Reading: the extra L1 dependence is a measurement effect in Bmad's screen (the split / resolution term picks up L1 through the energy structure) that the smoothed surrogate beam does not reproduce. Consequence: the controller underestimates how much L1 moves the *measured* σz; plausible contributor to the flat σz errors. Not a spacing problem.
- Figure: TODO (per-point slopes, Bmad vs surrogate, beam vs screen).

### Jitter-trained policies (chain done 20:53) — `results/3_rl_eval/eval_jitter_2026-10-18/summary.txt`
Settled spacing median µm / settled within 10 µm (continuous operation, 128 steps × 256 envs).

| policy | nominal jitter | stress jitter | replay TEST_15668 (in10) | replay E300_15673 (in10) |
|---|---|---|---|---|
| noblen s0 / s1 (no jitter training) | 11.0 / 0.45 · 11.4 / 0.43 | 15.6 / 0.24 · 16.9 / 0.25 | — | — |
| **noblen_jit** s0 / s1 | 11.5 / 0.44 · 12.4 / 0.40 | 15.9 / 0.25 · 16.6 / 0.23 | 0.73 · 0.73 | 0.33 · 0.31 |
| bc11loop v3 s0 / s1 (no jitter training) | 12.3 / 0.38 · 12.6 / 0.39 | 18.2 / 0.21 · 17.7 / 0.23 | — | — |
| **bc11loop_v3_jit** s0 / s1 | 12.6 / 0.39 · 14.0 / 0.36 | 16.6 / 0.26 · 17.3 / 0.22 | 0.73 · 0.58 | 0.20 · 0.22 |
| old blen_dlog baseline | 12.3 / 0.39 | 17.6 / 0.21 | — | — |

- **Training with the jitter model did not help**: nominal unchanged-to-slightly-worse, stress 0–1.6 µm better (BC11 loop only). The ~7 µm / 85% perfect-controller floor is far away, so the gap is not jitter-limited.
- Both loops off stays ≥ 1 µm better than BC11 loop on under nominal jitter.

### Large setpoint moves (best jitter-trained seed, nominal jitter; trained radius 2.0 / 2.5°)

| policy | local (trained) | radius ×2 (4 / 5°) | full start box |
|---|---|---|---|
| noblen_jit s0 | 11.5 / 0.44 / reached 0.86 | 12.3 / 0.39 / 0.81 | 13.7 / 0.37 / 0.75 |
| bc11loop_v3_jit s0 | 12.6 / 0.39 / 0.83 | 12.9 / 0.39 / 0.79 | 14.7 / 0.34 / 0.74 |

- Large moves degrade gracefully in spacing (+2 µm, −7 pts within 10 µm) but σz settles much worse (0.15 → 0.24–0.27 relative): target for the goal-conditioned training.

### Closed-loop at held-out injectors (surrogate run; Bmad running)
24 held-out injector points, 1 episode each, final true error. noblen_jit s0: 14.5 µm median, 33% within 10 µm, σz 0.21. bc11loop_v3_jit s0: 15.4 µm, 38%, σz 0.13. (For comparison, 10-18 closed-loop Bmad of noblen s0 at the golden injector: 9.7–10.3 µm / 46–54%. Single episodes per point, so ±~10 pts noise.)

### Closed-loop Bmad at held-out injectors (done 00:24) — `results/3_rl_eval/closed_loop_bmad_2026-10-19/`
24 envs (12 DAQ-like, 11 full-box test-split injector points, 1 golden), one 32-step episode each, 20k macroparticles; final true spacing error (median µm / within 10 µm).

| policy | plant | all 24 | DAQ-like (12) | full-box (11) | σz rel |
|---|---|---|---|---|---|
| noblen_jit s0 | surrogate | 14.5 / 0.33 | 12.9 / 0.42 | 18.2 / 0.27 | 0.21 |
| noblen_jit s0 | **Bmad** | **13.7 / 0.33** | 13.2 / 0.33 | 13.5 / 0.36 | 0.22 |
| bc11loop_v3_jit s0 | surrogate | 15.4 / 0.38 | 15.7 / 0.42 | 13.3 / 0.36 | 0.13 |
| bc11loop_v3_jit s0 | **Bmad** | **21.5 / 0.29** | 27.0 / 0.17 | 14.0 / 0.45 | 0.21 |

- Both loops off transfers from surrogate to Bmad at unseen injectors (14.5 → 13.7 µm); BC11 loop on loses 6 µm on Bmad, mostly at the DAQ-like points.
- Held-out injectors are harder than golden (10-18 closed-loop Bmad at golden: 9.7–10.3 µm / 46–54%), consistent with the surrogate's own held-out run — a policy problem (injector generalization), not a surrogate-transfer one, for the noblen direction.
- The single golden env failed in both bc11loop runs (~265 µm: goal on the other side of the split); 1 env, not interpreted.
- 12 envs per group: ±~14 pts on the fractions.

### Spacing-head trainings, wave 1 (trained 02:00; evals in `results/3_rl_eval/eval_night_2026-10-18/summary.txt`)
Common plant: all policies evaluated under nominal jitter on the **spacing-head flow** (`eval_continuous --flow`, `results/3_rl_eval/eval_night_2026-10-18/cross_zs.txt`). Settled spacing µm / within 10 µm; replay = DAQ replay within 10 µm.

| policy | flow trained on | nominal on zs flow | replay TEST_15668 | replay E300_15673 |
|---|---|---|---|---|
| noblen s0 / s1 | B2-cov, no jitter | 11.0 / 0.45 · 11.9 / 0.43 | — | — |
| noblen_jit s0 / s1 | B2-cov | 11.1 / 0.45 · 12.2 / 0.41 | 0.73 · 0.73 | 0.33 · 0.31 |
| **zs noblen_jit s0** | spacing head | 11.7 / 0.42 | **0.84** | 0.34 |
| **zs noblen_jit_fd s1** | spacing head | 12.4 / 0.38 | 0.69 | 0.25 |
| bc11loop v3 s0 / s1 | B2-cov, no jitter | 12.6 / 0.38 · 12.9 / 0.38 | — | — |
| bc11loop_v3_jit s0 / s1 | B2-cov | 11.7 / 0.42 · 14.7 / 0.33 | 0.73 · 0.58 | 0.20 · 0.22 |
| **zs bc11loop_jit s0** | spacing head | 14.2 / 0.32 | **0.88** | 0.34 |
| **zs bc11loop_jit_fd s1** | spacing head | 14.4 / 0.35 | **0.88** | 0.14 |

- On the common simulated plant, training on the spacing-head flow does **not** improve the policies (same or ~1–2 µm worse; seed spread is ~1–3 µm).
- On **real data** (TEST_15668 replay) 3 of 4 spacing-head runs are better (0.84–0.88 vs 0.58–0.73). E300_15673 stays poor for all (0.14–0.34).
- Failure detection: no visible effect yet on either plant (one seed per arm; wave 2 adds the other seed).

### All 8 spacing-head trainings (done 08:56)
Nominal / stress = continuous eval on the spacing-head flow (= the common plant above). Settled spacing µm / within 10 µm; replay = within 10 µm. Two seeds per arm (s0 · s1).

| arm | nominal | stress | replay TEST_15668 | replay E300_15673 |
|---|---|---|---|---|
| noblen | 12.2 / 0.39 · 12.9 / 0.38 | 16.5 · 16.1 | 0.84 · 0.77 | 0.34 · 0.25 |
| noblen + failure detection | 12.0 / 0.42 · 12.9 / 0.39 | 16.8 · 16.5 | 0.62 · 0.69 | 0.19 · 0.25 |
| bc11loop | 13.7 / 0.34 · 13.0 / 0.37 | 17.8 · 17.0 | **0.88 · 0.88** | **0.34 · 0.42** |
| bc11loop + failure detection | 12.4 / 0.38 · 14.2 / 0.33 | 17.5 · 17.5 | 0.78 · 0.88 | 0.20 · 0.14 |
| (ref) B2-cov noblen_jit, common plant | 11.1 / 0.45 · 12.2 / 0.41 | — | 0.73 · 0.73 | 0.33 · 0.31 |
| (ref) B2-cov bc11loop_v3_jit, common plant | 11.7 / 0.42 · 14.7 / 0.33 | — | 0.73 · 0.58 | 0.20 · 0.22 |

- **Failure detection (observing `screen_cut_sens`) hurts on real data** (TEST −0.1 to −0.2, E300 −0.1 to −0.2 for 3 of 4 seeds) and is neutral in simulation. Likely a sim→real shift of the diagnostic itself: real E300 cut_sens median 4.3 µm vs 1.1 µm in Bmad, so the policy sees "untrustworthy" readings it was trained to treat differently. Not pursued further; the GC configs use the plain arm.
- **Spacing-head flow, BC11 loop on is the best policy on real data so far** (TEST 0.88 both seeds, E300 0.34 / 0.42 — the first runs above ~0.33 on E300), though ~1–2 µm worse in simulation than both-loops-off. The replay freezes L1 (the scans are L2-only, BC11 loop on), so it scores L2 control specifically.
- Simulated scores: no gain from the spacing-head flow (all arms within seed spread, 12–14 µm).

## Summary for the morning
1. Gradients: the surrogate's beam-level L1/L2 gradients match Bmad (spacing 1–3%, σz ~10%); the measured σz's L1 response does not (surrogate 0.23× Bmad) — a screen-measurement effect missing from the smooth surrogate beam.
2. Jitter training: no gain. Spacing-head flow: no simulated gain, better TEST replay (0.84–0.88 vs ≤ 0.73).
3. Failure-detection observation: hurts on real replays → dropped. Detection itself works (AUC 0.91 on Bmad), so a rule-based hold on high cut_sens / charge-fraction deviation remains an option, calibrated on real data.
4. Held-out injectors in closed-loop Bmad: both loops off transfers (13.7 µm / 33%), BC11 loop on does not (21.5 / 29%); both are worse than at golden (~10 µm / ~50%) → injector generalization is the main sim-side gap.
5. Goal conditioning: built + tested; large moves cost ~2 µm spacing and a lot of σz (0.15 → 0.25). Configs ready (plain arm, both directions).

## Day 10-19 additions

### Validity band (`results/2_surrogate_screen/validity_band_2026-10-19/summary.json`)
Bands = |true spacing|. Valid rate = deployed screen validity.

| band [µm] | <60 | 60–100 | 100–150 | 150–250 | >250 |
|---|---|---|---|---|---|
| test rows valid, Bmad / surrogate (4096 parts) | 0.07 / 0.06 | 0.65 / 0.53 | 0.76 / 0.73 | 0.85 / 0.86 | 0.26 / 0.30 |
| training goal candidates valid (1024 parts) | 0.01 | 0.61–0.63 | 0.95 | 0.91 | 0.28–0.29 |
| share of accepted training goals | 1% | 10% | 23% | 44% | 22% |
| continuous eval, within 10 µm (4 policies) | 0.08–0.25 | 0.40–0.49 | 0.46–0.50 | 0.41–0.46 | 0.25–0.32 |
| E300_15673 replay, within 10 µm (zs plain arms) | — | **0.00–0.13** | 0.08–0.54 | 0.43–0.57 | — |
| TEST_15668 replay, within 10 µm (zs plain arms) | — | 0.62–0.88 | 0.78–1.00 | 0.84–0.94 | — |

- In simulation the compressed band is not a problem: band-level validity at training N matches Bmad (0.61 vs 0.65), the band gets ~10% of goals, and its error is like the 100–250 µm bands. The two compressed points of the particle-count study (σz ≈ 20 µm) are the extreme end of the band.
- The weak simulated band is **> 250 µm** (22% of goals; screen valid only ~28% — crop-edge clipping): settled 14–20 µm, 25–32% within 10 µm.
- On **real data** the compressed band is where E300_15673 fails: its 60–100 µm goals (23% of the replay goals) are within 10 µm in 0–13% of episodes vs 43–57% at 150–250 µm. TEST_15668's 60–100 µm goals are fine (62–88%). n = 15 goals per band per policy.

### Gradient parity figure
`results/2_surrogate_screen/gradient_check_2026-10-18/gradient_parity.png` (`scripts/gradient_check_figs_2026-10-19.py`). Breakdowns sit at the compressed linac point (~70 µm, σz ≈ 20 µm: beam σz d/dL1 0.14–0.18 Bmad vs −0.01–0.04 surrogate; screen spacing d/dL1 45 vs 20 µm/deg) and at the ~360 µm point (Bmad screen invalid, slope meaningless).

### Other
- σz vs particle count: `results/2_surrogate_screen/sigmaz_convergence_2026-10-19/` — no sampling bias; surrogate σz offsets are model error; validity at the compressed points falls with N in the surrogate.
- DAQ settling: `XTCAV_analysis/sim2real/results/settling_2026-10-19/` + `scripts/xtcav_{settling_check,settling_null,tcav_flip_lag}_2026-10-19.py` — image lags the TCAV phase readback by ~1 s after a ±V flip; no resolvable beam transient after the 6.4–7.5 s L2-step gap except the BC11 loop's L1 readback (~3 s).
- Slew GIFs: `results/5_reports/slew_gif_2026-10-19/slew_{good,median}.gif`.
- `results/` sorted into `1_datagen/ 2_surrogate_screen/ 3_rl_eval/ 4_daq_realism/ 5_reports/` (`scripts/organize_results_2026-10-19.py`, index + old→new table in `results/README.md`); `results/jitter_mc_2026-10-18` left as a symlink for saved run configs.

## Night 10-19 → 20 (plan agreed 18:45; Ryan back ~12:00)
- GC training: `scripts/gc_night_2026-10-19.sh` (log `logs/2026-10-19/gc_night.log`): `gc_zs_screen_{noblen,bc11loop}_jit` × s0/s1, 1200 it, then nominal / far1 / local / stress / replay evals (`results/3_rl_eval/eval_gc_2026-10-19/summary.txt`), baselines zs noblen s0 + zs bc11loop s1 under far1 / local; best GC (far1 settled within 10 µm) → noise-free Bmad slew eval.
- Noise-free Bmad slew eval: new `closed_loop_bmad --slew N --seg-steps S --ideal <yaml>` (ideal machine = `_idealize`: no measurement noise / outliers / calibration error / drift / wander / jumps / jitter map; nominal gains and readbacks; golden optics; nominal TCAV). 48 golden-injector episodes × 3 setpoints anywhere in the start box × 16 steps. zs noblen s0 launched 19:05 (`scripts/slew_ideal_noblen_2026-10-19.sh`, ~5 h). Surrogate pair: setpoints reached 90–96%, final 3.7–5.4 µm median, 69–77% within 10 µm (an earlier identical smoke gave 5.7–7.2 µm: closed_loop_bmad does not seed the flow's sampling).

### Crop window (`results/2_surrogate_screen/crop_window_2026-10-19/`, `scripts/crop_window_check_2026-10-19.py`)
The env crop (±4.6 × ±2.44 mm) is the deployed auto-crop (150 px); the DAQ camera window is ~±7.8 × ±3.9 mm (≈ ×1.7). 688 test rows, Bmad vs surrogate (spacing head), noise-free.

| crop | valid Bmad / surr | > 250 µm valid | > 250 µm Bmad screen within 10 µm of beam | 100–150 µm valid | surr vs Bmad reading within 10 µm |
|---|---|---|---|---|---|
| ×1 (deployed) | 0.52 / 0.51 | 0.26 / 0.31 | 0.55 (median 9 µm) | 0.76 / 0.76 | 0.72 |
| ×1.5 | 0.67 / 0.64 | 0.77 / 0.76 | 0.36 (12 µm) | 0.93 / 0.91 | 0.71 |
| ×2 | 0.69 / 0.67 | 0.91 / 0.95 | 0.31 (13 µm) | 0.93 / 0.91 | 0.71 |
| none | 0.70 / 0.67 | 0.92 / 0.95 | 0.30 (13 µm) | 0.92 / 0.91 | 0.71 |

- Widening the crop recovers most of the large-spacing (and 100–150 µm) readings: ×1.5 gets nearly all of the gain, ×2 ≈ no crop.
- The surrogate keeps tracking Bmad equally well (validity agreement 0.91 → 0.93; > 250 µm reading agreement 0.79 → 0.86): crop size does not hurt surrogate fidelity.
- ~~Newly admitted readings less accurate~~ — that was an artifact of the absolute 10 µm threshold at 250+ µm. In **relative** error (|reading − true beam| / true beam, Bmad, valid readings) the > 250 µm band stays at a 3.4% median for every crop, and within 10% rises 0.72 (×1, n = 29) → 0.93 (×1.5, n = 85) → 0.89 (×2, n = 101); surrogate-vs-Bmad within 10% 0.79 → 0.92 at ×1.5. **×1.5 is the sweet spot** (≈ the DAQ camera frame, so an analysis-only change).
- Relative error is worst at small spacing: 60–100 µm 13.7% (×1) → 17.6% (×1.5) median, 100–150 µm 8.8–9.5%, vs 3.3–3.4% above 150 µm; widening slightly hurts the 60–100 µm band (newly admitted poorer readings).

### Real-data check of "no crop" (`results/4_daq_realism/online_replay_v7_{noauto,fullframe}_2026-10-19/`, `scripts/online_replay_nocrop_2026-10-19.sh`)
Deployed online analysis (replay v6 settings) on TEST_15668 / E300_15671 / E300_15673 with the auto-crop off (hand crop kept) and with both crops off. No truth on real data: judged by validity, smoothness of spacing vs scan, scatter, cut_sens.

| scan | analysis | valid steps | roughness (median abs 2nd diff of spacing vs scan, µm) | single-image sigma | drive charge frac |
|---|---|---|---|---|---|
| TEST_15668 | deployed / no auto-crop / full frame | 17 / 17 / 18 | 4.4 / 4.4 / 4.8 | 15.4 / 15.5 / 15.3 | 0.713 / 0.719 / 0.718 |
| E300_15673 | deployed / no auto-crop / full frame | 18 / 24 / 24 | 7.0 / 5.6 / 4.7 | 15.8 / 15.9 / 15.2 | 0.682 / 0.718 / 0.720 |

- Holds up: E300_15673's six large-spacing steps (deployed: invalid, edge_frac 0.015–0.031) become valid with 5–50 µm larger spacing (the witness was being clipped); the spacing curve gets smoother (the deployed step-11 outlier 170 µm → 188 µm, in line with its neighbours); single-image scatter unchanged.
- Systematic shift: with the auto-crop off, unclipped steps read +3–18 µm higher (60–150 µm region on both scans) and the drive charge fraction rises (0.68 → 0.72 on E300_15673). In Bmad the same change moves readings by −0.1 µm, so this is a real-image effect the sim does not have (the auto-crop cuts real image tails / halo).
- Full frame (hand crop off too) is unstable where the beam is unreadable (E300_15671 steps 10–25 and TEST_15668 steps 26–30 flip sign or jump 10–20 µm: stray light enters). Keep the hand crop: "no crop" = auto-crop off. The hand crop (450 × 300 px) is ≈ ×1.5 of the auto-crop in x, inside the sim's ×1.5–none plateau.

### 23:15 re-plan: no auto-crop (Ryan)
- Cropped GC chain stopped at 750 it (all four runs; `logs/2026-10-19/gc/`, kept as a crop reference: their best checkpoints are evaluated under far1 / local). Measured speed with 4 runs on 2 GPUs: 750 it / 4.3 h.
- Sim window = the deployed hand crop (450 × 300 px → half-widths 6.86 × 4.58 mm), crop-edge check kept (Ryan's choice). Approximation: the sim window is centred on the beam, the real hand crop is fixed.
- `daq_replay` now reads the real readings matching the policy's window: `online_replay_v7_noauto` for the new window, v6 (auto-crop) for older policies; `--replay-dir` overrides.
- `scripts/nocrop_night_2026-10-19.sh` (log `logs/2026-10-19/nocrop_night.log`): one mixed wave, then seeds 1. GC s0 (1000 it, ~05:15) + plain s0 (900 it, ~04:40) per direction; s1 runs follow in each slot (~10:00 plain, ~11:00 GC). Best s0 per family (far1 within 10 µm) → noise-free Bmad slew eval (plain ~05:15 → ~10:20, GC ~05:45 → ~10:50). Plus: current policies and the cropped GC best checkpoints under the new window (eval-only).
- The cropped zs noblen s0 noise-free slew eval keeps running (crop baseline, ~00:20).
- Cropped baselines on the far1 / local conditions (before the re-plan): zs noblen s0 13.4 µm / 0.36 within 10 µm / reached 0.79 (far1) vs 11.6 / 0.43 / 0.85 (local); zs bc11loop s1 13.5 / 0.35 / 0.80 vs 13.0 / 0.37 / 0.85; σz settles much worse on far setpoints (0.24–0.28 vs 0.15).

### Noise-free Bmad slew eval, crop baseline zs noblen s0 (done 23:30; `results/3_rl_eval/slew_ideal_bmad_2026-10-19/zs_screen_noblen_jit_s0_{flow,bmad}`)
48 golden-injector episodes, 3 setpoints anywhere in the start box (median jump 72 µm), 16 steps each, ideal machine. Final true error vs true goal per setpoint.

| setpoint | Bmad: final median / within 10 µm / reached | surrogate (same episodes) | σz final rel (Bmad) |
|---|---|---|---|
| 1 | 3.4 µm / 0.90 / 0.98 | 5.4 µm / 0.69 / 0.92 | 0.27 |
| 2 | 4.6 µm / 0.81 / 0.98 | 3.7 µm / 0.77 / 0.90 | 0.18 |
| 3 | 6.2 µm / 0.71 / 0.98 | 3.8 µm / 0.75 / 0.96 | 0.29 |

- On an ideal Bmad machine the controller reaches 98% of slew targets and settles at 3–6 µm median spacing error; as good as or better than on its own surrogate. σz is the weak target (17–29% final error).

### No-crop seed-0 results (06:15; `results/3_rl_eval/eval_nocrop_2026-10-19/summary.txt`)
Timing: 4 concurrent runs took 4.6 h (plain noblen, 900 it), 5.3 h (GC noblen, 1000), 6.1 h (plain bc11loop), 6.9 h (GC bc11loop): the BC11-loop env is slower. Plain slew eval started 06:10 (~11:15); GC slew after the GC bc11loop evals (~07:00 → ~12:00). s1 runs overrun noon.
Settled spacing µm / within 10 µm (nominal jitter, own flow = spacing head); replay = within 10 µm.

| policy | window | nominal | far1 | local | stress | replay TEST / E300 |
|---|---|---|---|---|---|---|
| zs noblen s0 (old, crop-trained) | eval with new window | 11.7 / 0.42 | 14.3 / 0.34 | — | — | **0.92 / 0.42** (v7 readings) |
| zs bc11loop s1 (old) | eval with new window | 13.7 / 0.35 | 15.5 / 0.31 | — | — | **0.91 / 0.50** |
| nc plain noblen s0 | trained no-crop | 11.6 / 0.42 | 14.4 / 0.33 | 12.2 / 0.39 | 17.0 / 0.21 | 0.84 / 0.27 |
| nc plain bc11loop s0 | trained no-crop | 15.2 / 0.29 | 18.7 / 0.23 | 15.5 / 0.28 | 18.5 / 0.18 | 0.81 / 0.34 |
| nc GC noblen s0 | trained no-crop | 14.8 / 0.31 (mixed) | 15.0 / 0.33 | 13.3 / 0.36 | 20.0 / 0.19 | 0.55 / 0.31 |
| crop GC 750 (best ckpt), 4 runs | crop | 14.4–15.4 / 0.32–0.34 | 14.9–17.5 / 0.27–0.34 | 12.9–15.5 / 0.31–0.37 | — | — |

- **The old crop-trained policies do better on real data when fed no-crop readings** (TEST 0.84–0.88 → 0.91–0.92, E300 0.34–0.42 → 0.42–0.50), and are unchanged in simulation.
- Retraining with the new window: noblen unchanged in sim, worse on the replays (one seed); bc11loop worse in sim (one seed).
- **Goal conditioning (far setpoints + spacing-only) gives no gain** on far setpoints (far1 15.0 / 0.33 vs plain 14.4 / 0.33; σz 0.23 vs 0.23) and costs on local setpoints and on the replays. The far-setpoint penalty is mostly short segments (12–24 steps: 15.6 µm vs 13.3–13.6 µm for longer ones); given 36+ steps, far ≈ local (0.31–0.37 vs 0.34–0.39 within 10 µm). So the limit is the per-step slew budget plus the noise floor, not the goal distribution.
- Seed-1 runs pending; the conclusions above are one seed per arm.
- (Cosmetic: the old-policy far1 / replay lines are labelled `*_window_window` in summary.txt — a shell variable clash; the evals themselves are correct.)
- nc GC bc11loop s0: nominal 17.6 / 0.26, far1 18.1 / 0.24, local 16.5 / 0.28, stress 21.1 / 0.17, replay TEST 0.55 / E300 0.19 (worst of the night). GC slew eval → gc_nc_zs_screen_noblen_jit_s0 (07:00 → ~12:00); plain slew (nc noblen s0) 10/54 calls at 07:00 (~10:40).

### Morning 10-20 (12:40)
**Noise-free Bmad slew, no-crop plain noblen s0** (`results/3_rl_eval/slew_ideal_bmad_2026-10-19/nc_zs_screen_noblen_jit_s0_*`, figures `slew_traces_*.png`, `slew_summary.png`; done 12:36 — two Bmad evals sharing the CPUs ran ~2 h slower than planned).

| setpoint | no-crop: final median / within 10 µm / reached | crop baseline | σz final rel, no-crop vs crop |
|---|---|---|---|
| 1 | 3.4 µm / 0.90 / 1.00 | 3.4 / 0.90 / 0.98 | 0.16 vs 0.27 |
| 2 | 6.7 µm / 0.67 / 0.96 | 4.6 / 0.81 / 0.98 | 0.18 vs 0.18 |
| 3 | 3.9 µm / 0.85 / 0.98 | 6.2 / 0.71 / 0.98 | 0.11 vs 0.29 |

By goal spacing (all setpoints): < 100 µm 0.20 within 10 µm (n = 5; crop 0.46, n = 13), 100–150 0.74 (0.86), 150–250 0.88 (0.88), **250–400 0.82 (n = 49; crop 0.70, n = 20)**. The wider window admits more large-spacing goals and tracks them better; < 100 µm remains the failure region in both.

**Seed 1** (nominal / far1 / replay TEST / E300): plain noblen s1 11.6 / 0.41, 13.3 / 0.36, 0.84 / 0.38; GC noblen s1 13.1 / 0.36, 14.0 / 0.34, 0.77 / 0.30; plain bc11loop s1 12.5 / 0.37, 14.0 / 0.32, 0.84 / 0.44 (s0 was 15.2 / 0.29: seed spread ~3 µm). GC bc11loop s1 still training; GC noblen s0 Bmad slew at 44 / 54 calls (~13:30).

**Conclusions**
1. On an ideal Bmad machine the controller slews to targets: 96–100% of setpoints reached, final median 3–7 µm, 67–90% within 10 µm; no-crop training also roughly halves the σz error at the end of slews.
2. Goal conditioning (far setpoints + spacing-only) does not beat the plain arm on far setpoints in either seed (far1 0.33–0.34 vs 0.33–0.36 within 10 µm) and costs on local setpoints and replays. Far-move performance is limited by the per-step slew budget and the noise floor.
3. No-crop retraining ≈ crop training in simulation (within seed spread); on the real replays the crop-trained policies fed no-crop readings remain best (TEST 0.91–0.92, E300 0.42–0.50 vs 0.81–0.84 / 0.27–0.44 retrained).
4. Setpoints below ~100 µm fail even noise-free (surrogate's weak L1 response near compression).
- 13:23 chain done. GC bc11loop s1: nominal 13.8 / 0.34, far1 14.6 / 0.31, local 12.6 / 0.39, replay TEST 0.67 / E300 0.23. Noise-free Bmad slew of GC noblen s0: setpoints reached 0.98–1.00, final median 6.3–7.7 µm, within 10 µm 0.65–0.73, σz 0.15–0.27 (vs plain no-crop 3.4–6.7 µm / 0.67–0.90): goal conditioning is worse on the ideal machine too.
- Replay gap crop-trained vs no-crop-retrained (TEST 0.89–0.94 vs 0.81–0.84, 4 runs each; E300 overlapping): same L2 convergence, spread over all goal bands; training goal mix shifted (> 250 µm 22% → 34%, TEST range 70–200 µm 55% → 47%). Likely cause, untested.
