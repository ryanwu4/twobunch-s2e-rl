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
