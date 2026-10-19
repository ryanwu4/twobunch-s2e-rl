# Overnight 2026-10-18 → 19

Plan agreed with Ryan 18:40 (back ~08:00). Chains: `scripts/gradient_check_2026-10-18.sh` (log `logs/2026-10-18/gradient_check.log`), `scripts/night_2026-10-18.sh` (log `logs/2026-10-18/night.log`; waits for the jitter chain).

## Decisions (Ryan)
- Train BOTH directions (both loops off = noblen; BC11 loop on = bc11loop v3) on the spacing-head flow, ± failure detection, 2 seeds, 900 it → 8 runs, 2 per GPU (~10 h). No closed-loop Bmad on these tonight.
- Gradient check = research, not a deployment gate.
- Goal conditioning: σz "mixed + mask" (spacing-only setpoints with a mask bit), setpoints "mix local + random". Build + smoke tonight, train tomorrow.

## A. Screen failure detection (done)
`scripts/screen_failure_detect_2026-10-18.py` → `results/screen_failure_detect_2026-10-18/summary.json`. 688 test rows, failure = |screen − own cloud spacing| > 30 µm or misassign > 0.15, among deployed-valid readings.

| source | valid | fail rate | AUC cut_sens | AUC charge-frac dev | AUC combined |
|---|---|---|---|---|---|
| Bmad | 356 | 0.37 | 0.78 | 0.88 | 0.91 |
| spacing head | 344 | 0.47 | 0.84 | 0.93 | 0.95 |

- P(fail | cut_sens) is similar in both: 0–2 µm 24% / 24%, 2–5 µm 62% / 80%.
- **Decision (mine):** the surrogate's screen already fails at least as often as Bmad's with the same diagnostic signature, so no synthetic failure injection; the failure-detection arm only *observes* `screen_cut_sens` (± 2 px cut shift, 1.5 µm/shot noise; fixed z-score mean 2 / std 3 µm). The measured drive charge fraction (the stronger flag) was already observed.
- Real DAQ steps carry `cut_sens_um`; valid-step distributions match the sim (TEST_15668 median 1.0 µm, E300_15673 4.3 µm vs Bmad 1.1 µm). `daq_replay` now feeds it to the policy.

## B. Night trainings (queued)
Configs `configs/rl/night_2026-10-18/zs_screen_{noblen,bc11loop}_jit[_fd].yaml` = the `*_jit` configs with the spacing-head flow (`covB2zs`), `_fd` + `screen_cut_sens`. Slots: GPU0 {noblen s0 → noblen_fd s0}, {noblen_fd s1 → noblen s1}; GPU1 same for bc11loop. Each run evaluated when done (nominal / stress jitter, DAQ replay TEST_15668 + E300_15673) → `results/eval_night_2026-10-18/summary.txt`.

## C. Gradient check (running)
`scripts/gradient_check_2026-10-18.py`: 3 injector states (golden, held-out lhsloc102, lhs123) × 4 (L1, L2) points; Bmad central differences at h = 0.3 / 0.6° (108 runs, 50k macroparticles, same beam + Tao seed); surrogate autograd + FD (8 seeds, 20k particles) for B2-cov and the spacing head. Features: beam spacing / σz and screen spacing / σz.
- Smoke (1 point): spacing slopes 54 (L1) / 39 (L2) µm/deg in Bmad; surrogate ratio 0.95–1.02. Screen σz vs L1: surrogate ratio ~0.2 — check on the full set.

## D. Closed-loop Bmad at held-out injectors (queued, after the jitter chain)
New `closed_loop_bmad --mode test --points-json configs/rl/eval/heldout_injectors_2026-10-18.json`: 12 DAQ-like + 11 full-box test-split injector points + golden (none seen in training). Best jitter-trained seed per direction (by nominal settled spacing), + the paired surrogate run.

## E. Goal conditioning (built, untrained)
- `DeltaControlEnvV4(goal_far_frac=…, spacing_only_frac=…, free_goal_keys=("drive_sigma_z",))`, off by default (RNG stream unchanged). Far setpoints: uniform over the start box. Spacing-only: σz target achieved := goal in the reward view (no penalty, no gradient), σz goal / error obs zeroed, mask bit appended; mask redrawn with every setpoint. `eval_continuous` reports free / constrained segments separately, σz metrics on constrained only. Tests: `test_far_goals_leave_the_goal_radius`, `test_spacing_only_goals_mask_sigma_z`.
- Configs `configs/rl/gc_2026-10-19/gc_zs_screen_{noblen,bc11loop}_jit_fd.yaml` (0.5 / 0.5), smoke-trained. Rebase on the winning arm before training.
- Large-move eval of the current policy (pre-check, noblen s0, nominal jitter, full-box setpoints): 13.2 µm settled / 36% within 10 µm / 78% reached vs 11.0 / 45% / 86% for local setpoints. The night chain repeats this for the best jitter-trained policies (radius ×2 and full box).

## Results

### C. Gradient check (done 19:52) — `results/gradient_check_2026-10-18/summary.json`
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
