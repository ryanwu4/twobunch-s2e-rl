# Overnight 2026-10-16 → 17

Ryan's scope (19:20): (1+4) direct-L1 + BLEN obs and the retrain on B2-cov, direct-L1 + BLEN **only**; (3) closed-loop Bmad, golden + injector-varied; (5) L2-only DAQ replay. Not now: jump filter, σz diagnosis, machine interface. Back ~20:00 on 10-17.

## Status

| Item | State |
|---|---|
| Injector coverage campaign | running, ETA ~00:45 (`logs/injector_cov_2026-10-16`) |
| B2-cov fine-tune + scorecards | waiting on the campaign (`scripts/after_injector_cov_2026-10-16.sh`), ETA ~06:30 |
| Direct-L1 + BLEN obs (env) | done: `v4.blen_obs` (BC11 + BC14 log-ratios), tests |
| Flow-A direct-L1 + BLEN S1 / S2 (exploratory) | training, ETA ~20:45 (`logs/2026-10-16/l1blen`) |
| B2-cov direct-L1 + BLEN S1 / S2 × 2 seeds + evals | waiting on B2-cov (`scripts/after_covB2_rl_2026-10-17.sh`) |
| Closed-loop Bmad | code done, smoke running; full runs wait for free CPUs (`scripts/closed_loop_bmad_2026-10-17.sh`) |
| DAQ L2 replay | code done; first results below |

## Log

- 19:30 **BLEN obs** in the v4 env (`blen_obs: {monitor: {centers, poly, slope_scale, noise}}`): per monitor the log-ratio to the episode-start reading, s·(P(true now) − P(true start)) + noise, hidden per-episode slope scale s. BC11: the BLEN-model DAQ-band cubic in L1 (s ~ U(0.8, 1.3), noise 0.025 ln); refused with a blen_dlog / adapter L1 (the loop servoes it). BC14: lo-band quadratic in (L2, L1) fitted on the controller-box rows of the BLEN cache (R² 0.95, residual 0.10 ln incl. varied injectors; slope 0.20/° vs the DAQ 0.10–0.42 per multiknob degree), s ~ U(0.7, 1.5), noise 0.035 ln. Off = byte-identical env. Configs: `scripts/make_l1blen_rl_configs.py` (adds the block to a direct-L1 v2 injflow config).
- 19:35 **bug (mine), fixed:** `build_delta_env_fn` builds the v4 kwargs by hand and did not forward `blen_obs`; the first two flow-A runs trained without the BLEN columns (caught by building the env from the config: 57 vs 59 obs). Aborted (`logs/2026-10-16/l1blen/ABORTED_noblenobs_*`), fixed, test that every v4 option is forwarded, relaunched.
- 19:50 **closed-loop Bmad**: plants take a swappable beam source (`observe.set_cloud_source`); `rl/_bmad_xtcav_server.py` (bmad-qpad-dev pool, campaign physics, XTCAVF) + `rl/closed_loop_bmad.py` (the policy's own env and screen pipeline on Bmad beams; golden or the coverage campaign's DAQ-distribution injector points with their cached Impact-T beams, each downsampled once, seeded; Tao seed fixed per episode; goals = Bmad readings at the goal points; `--plant flow` = the same episodes on the surrogate). Flow path smoke: S2 blen_dlog flow A, inj, 24 episodes: 9.5 µm / 54% within 10 µm (512-episode eval 8.1 / 57%).
- 19:55 **DAQ L2 replay** (`rl/daq_replay.py`): the policy's L2 on a real multiknob scan (L1 frozen, the BC11 loop held it), readings = the deployed online features of the scan step, spacing re-drawn per visit from the step's own images; BLEN-obs policies get the real BC14 BLEN of the scan. Metric: |Δmkb| to the goal's L2 setting (no truth on real data) and the measured spacing error. S2 blen_dlog (flow A injflow s0), 64 episodes:
  - TEST_15668 (valid 0–4°): |Δmkb| 0.88 → 0.17° (p90 0.45; 64% ≤ 0.25°); measured spacing error 3.8 µm, 81% within 10 µm.
  - E300_15673 (valid 1.5–5.75°, 140–300 µm, noisier): 0.88 → 0.33° (p90 2.25); 11.8 µm, 45%.
  - Real BC14 BLEN: E300_15673 rises 0.23 ln/° (env model 0.20); TEST_15668 is non-monotonic (+0.23 then below 0 by 2.5°).
- 20:05 closed-loop Bmad smoke (2 episodes, 1 step, inj): 3 Bmad calls of 6.0–6.5 min, 0 failures. Paired with the surrogate at identical start / goal points: goal truth spacing 82.2 / 101.0 µm (Bmad) vs 80.4 / 105.8 (flow), goal σz 31.6 / 24.6 vs 33.2 / 26.9 µm. Full chain launched (`scripts/closed_loop_bmad_2026-10-17.sh`, log `logs/2026-10-17/closed_loop_chain.log`): waits for the campaign, then S2 blen_dlog + S2 direct-L1 + BLEN (flow A) × golden / inj, 24 episodes each, 2e4 macroparticles (~3.7 h); then the B2-cov direct-L1 + BLEN S2 s0 once its chain is done. Out `results/closed_loop_bmad_2026-10-17/`.
- 20:15 DAQ replay, old-flow S2 v6 blen s0: TEST_15668 |Δmkb| 0.76 → 0.13° (80% ≤ 0.25°), measured spacing 3.0 µm / 89% within 10 µm; E300_15673 0.76 → 0.36° (p90 3.99°), 10.7 µm / 45%. Same pattern as the flow-A policy: TEST fine, E300_15673 has a tail of episodes that run away from the goal (p90 2–4°). To look at: whether those are the high-spacing (140–300 µm), noisy-image steps.

## Results (all chains done by 08:29; reviewed 17:20)

- Coverage campaign: 2751 / 2751 ok, 11.8 h. Table `processed/xtcav23e_inj_cov_2026-10-17.h5` (12305 rows; old roles reproduced; new rows 2222 / 254 / 275).
- **B2-cov** (`trained/xtcav23e_inj_covB2_2026-10-17`, best epoch 34), same val rows as B2: pair R² 0.81 → 0.83, slope 0.79 → 0.80, p90 8.8 → 8.6 µm; σz 0.109 → 0.108; T 0.0027. Still misses slope (≥ 0.85) and p90 (≤ 8). Train pairs did not improve (0.88 / 0.85), and on the new table train vs held-out is 0.86 vs 0.85: the gap is closed, so the remaining error is bias (pair weight / model), not data.
- RL on B2-cov + DAQ injector context (512 episodes, true median / within 10 µm): S2 direct-L1 + BLEN s0 9.5 / 52%, s1 11.7 / 45% (blen_dlog reference 8.8 / 55%; direct-L1 without BLEN on flow A 12.8 / 43%); S1 direct-L1 + BLEN 4.8–5.4 / 80% (blen_dlog 4.7 / 80%).
- **Closed-loop Bmad** (24 episodes, 2e4; Bmad vs paired surrogate): S2 blen_dlog flow A golden 5.3 / 71% (7.3 / 62%), inj 8.1 / 67% (6.4 / 71%); S2 direct-L1 + BLEN flow A golden 5.6 / 79% (7.3 / 58%), inj 5.4 / 67% (8.7 / 58%); B2-cov s0 golden 7.8 / 58% (9.2 / 54%), inj 6.3 / 75% (8.5 / 54%). Closed loop on Bmad is as good as on the surrogate (±20 points at n = 24); σz rel 0.12–0.18.
- **DAQ replay**, direct-L1 + BLEN policies: with the real BC14 BLEN TEST_15668 47–73% / E300_15673 27–34% within 10 µm; with the env's BC14 model (`--blen-source model`) 72–77% / 33–42%; blen_dlog policies 81–89% / 45%. The real BC14 signal (non-monotonic in TEST) costs 5–25 points; the rest of the gap vs blen_dlog may be L1 being frozen in the replay.
