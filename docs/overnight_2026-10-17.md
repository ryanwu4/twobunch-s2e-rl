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
