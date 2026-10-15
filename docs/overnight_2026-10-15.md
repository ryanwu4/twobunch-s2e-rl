# Overnight 2026-10-14 → 15: injector campaign + training-side work

Ryan away until ~11:00 PDT 2026-10-15. Same ground rules as the first overnight
(`docs/overnight_2026-10-14.md`): work on `dev/2026-10-13`, commit per step, never push; never modify or delete
existing data/, processed/, trained/, logs/, results/ content (new outputs in new dirs); FACET2-S2E untouched;
CPUs belong to the campaign (64 workers, nice 10) — other CPU work kept light; GPUs free for training.

## Status at a glance (updated through the night)

| Item | State |
|---|---|
| Injector campaign `injector_full` (3027 rows, 5e4 / grid 64, commit d20918d, worktree `.worktrees/injector_full_d20918df54`) | launched 18:50 |
| env v4 (`rl/delta_env_v4.py`) + S1 cloud trainings (blen_dlog vs adapter, seed 0, GPUs 0/1) | started ~19:28 |
| Screen v4 (optics/voltage randomization, ROI, arrival jitter, charge orientation, deployed validity) + S2 config | built, smoke-tested |
| Post-campaign chain (`scripts/after_injector_campaign.py`): evalset launch, preprocess, widen, 3 fine-tunes + scratch | waiting on the campaign |
| Surrogate gates script (`scripts/surrogate_gates.py`) | ready |

## Decisions taken in Ryan's absence (numbered; each with the reason)

8. **GPU priority: S2 v5 over adapter seeds 2–3** (23:10). With corrected eval, blen_dlog ≈ adapter on S1
   (5.1 vs 4.9 µm true median, one seed each), so Ryan's choice (blen_dlog) gets 4 seeds (s0–s3) and adapter 2
   (s0, s1); the adapter s2/s3 logdirs hold a SKIPPED placeholder. GPU0 then trains S2 v5.
9. **S2 v5 goal sampling: goals must be readable by the deployed analysis** (`goal_valid_retries 4`), as an
   operator would not set a goal the screen cannot measure. Off-context goals are kept at 20%.

1. **env v4 is a subclass** (`rl/delta_env_v4.py`, `diff_env.v4:` block), so v1–v3 envs and their policies
   stay byte-identical.
2. **Both L1 actuator models are trained, not just Ryan's choice.** Ryan chose "dln(BC11 BLEN setpoint) as the
   action, wide hidden gain, FAST_PACT dL1 observed" (`blen_dlog`) before the round-2 ML review argued BPTT
   can't learn active gain identification and recommended a structured adapter (policy asks dL1 in deg;
   forgetting-LS gain estimate turns it into dln). Training both on identical settings answers that question
   with data; nothing about Ryan's choice is changed.
3. **S1 actuator / noise numbers** (`configs/rl/delta_xtcav23_cloud_v4_{blen,adapter}.yaml`):
   L1 gain U(1.5, 4) deg/ln (ops round 2: DAQ run medians 2.2–3.4, local 1.9–3.7), dln step 0.1,
   |cum dln| ≤ 0.75 (≈ ±2.5°); FAST_PACT readback noise 0.02°/step + loop drift compensation 0.04°/step
   (DAQ: 0.03–0.06°/step at fixed setpoint); L2 MKB gain U(0.35, 1.1) (physics round 2 M5), step 0.3°,
   |cum| ≤ 3.5°; hidden drift L1 0.02°/step (residual with the loop on), **L2 0.05°/step = ASSUMED** (not
   measurable from the DAQ; system-ID item); M = 30 valid ±V images (ops: 25–30); shot noise 15 µm spacing /
   6 µm drive σz; per-step floor 6 / 2.5 µm; 1% outlier steps ×5; streak scale U(0.8, 1.2).
4. **Explicit reward tolerances** (D6): new objective `scale` → 1 penalty unit = 10 µm spacing = 10% drive σz
   (0.0414 in log10) = 1% transmission loss. Old configs unchanged (no `scale` → campaign std).
5. **Transmission hinge at 0.95 (not 0.98) in S1**: S1 runs on the current xtcav23 flow, whose 'max'-normalized
   T labels saturate at ~0.986 with no loss (assumptions scan B1). The per_sample relabel fixes this for the
   injector flow; the hinge goes back to ~0.98 there.
7. **Goal radius L1 2.0° / L2 2.5°** (see problem 4): goals the capped actuators can reach in 32 steps for most
   gains; the 20% off-context goals remain as the harder deployment case.
6. **S1 start box** L1 [−26, −19], L2 [−40.5, −33.5] (inside the plant box; covers the 04-15 operating points
   per physics round 2 M3). Action cost 0.02·a², 20% off-context goals, 8-step history, reward on truth.

## Problems encountered

7. **S2 screen readings at large spacing are clipped by the auto-crop and still called valid** (23:00). With
   corrected eval, S2 v4r reaches 11.6 µm true median spacing error (p90 54 µm). Even noise-free, the measured
   goal differs from the true goal by 23.5 µm median (p90 146 µm). The worst case is true spacing ≥ 300 µm:
   |meas − true| p90 650 µm, because the 4.6 mm half-width crop (= the deployed auto-crop, 150 px) cuts the
   witness at streak 15–20× (22–30 MV). This is a real limitation, not a sim artifact: the real 300 µm steps of
   E300_15673 were at streak 14.4 and fit inside. The deployed rule (two peaks, |spacing| > 60 µm) cannot see it.
   Fix (S2 v5, `configs/rl/delta_xtcav23_screen_v5_blen.yaml`): a truth-free crop-edge check (`edge_frac_max`
   0.01: > 1% of in-window charge within 0.3 mm of the crop edge → invalid); truth validity also counts clipped
   charge; `goal_valid_retries 4` redraws goals the deployed analysis cannot read. Goal-point bias p90 270 → 75 µm,
   deployed-valid goals 60 → 91%. The remaining bias is energy-gate misassignment at 60–200 µm (problem 3).
   The online image features need the same edge check (todo).

6. **My `while pgrep -f <pattern>` waiters never exit**: the waiting shell's own command line contains the
   pattern. That delayed ablation queue B by ~30 min. The waiters were stopped and queue B started directly.
   The post-campaign chain is not affected: its `pgrep -f train_bptt` runs from Python, whose command line
   does not contain the pattern.

5. **env v4 policies are far worse than v3** (21:00). With reachable goals the v4 controllers stall at 18–22 µm
   true spacing error (31–37% within 10 µm; ×2 episode length does not help), against 2.6 µm / 97% for the v3 cloud
   controller (`delta_xtcav23_cloud_s0`). blen_dlog and adapter fail alike, so it is not L1 gain identification.
   The S2 screen v4r is similar: 13 µm true median, p90 ~100 µm. Diagnosis: the policies are bang-bang (50% of
   actions saturated, mean |a| 1.04) and leave ~0.3° median / 1.3° p90 per-control residuals, about the
   scale of the L2 hidden drift (0.05·√32 ≈ 0.28°). The learning curve plateaus by ~50k steps. Ablation
   trainings (120 iters each, `configs/rl/ablation_2026-10-15/`, logs `logs/2026-10-15/ablation/`): base,
   nodrift, nonoise (measurement model off), noactcost, all3off. More seeds and S2 variants are held until
   this is understood.
   **Result (on-context eval):** base 21.7 µm / 29% within 10 µm; **nonoise 4.5 µm / 82%**; all3off 4.0 µm / 86%.
   The measurement model is the cause; sub-ablations noise_only / scale_only are running.
   **RETRACTED (22:30): the regression was an EVAL BUG, not the policies.** scale_only trained 20 µm, but the
   same policy evaluated at scale error 0 got 4.8 µm, and an all3off policy evaluated WITH the scale error got
   21 µm. A direct trajectory check showed ~5 µm true error in every scale bin. Cause: `eval_delta` read
   `env._scale` / `env._gain` after the rollout. The rollout runs the full episode, so the env auto-resets on the
   last step and those values belong to the next episode, which broke `goal_true = goal / scale`. Also, with v4
   `reward_on: true`, the "measured" column compared the true feature against the measured goal. Both read
   ≈ (1 − s) × 180 µm ≈ 20 µm for any policy under the ±20% streak-scale error. Fixed (`<key>_meas` in
   achieved, true goal from the goal cloud, hidden draws read before the rollout); all v4 runs re-evaluated
   (`fx_*` tags). The earlier v4 numbers in this log (problem 4, the 20:00 S1 entry, the v4r S1/S2 numbers)
   used the buggy eval. The v3 numbers are unaffected (no scale error in cloud runs; screen runs use the
   cloud branch).

4. **First S1 runs: goals mostly unreachable** (my env-design error). v4 added actuator caps (L1 |dln| ≤ 0.75,
   L2 ≤ 3.5°) and low L2 gains (≥ 0.35) but kept goals anywhere in a 7°-wide start box: only 37–43% were reachable.
   Final TRUE spacing error 25 µm median / 26% within 10 µm overall, but on reachable goals 7.4 µm (blen_dlog) /
   6.6 µm (adapter), 67–70% within 10 µm (vs unreachable 16–19 µm). Fixed: `goal_radius` (L1 2.0°, L2 2.5° around
   the start, inside the start box; a39691b). The S2 run started with the flaw was stopped after ~5 min
   (`logs/2026-10-15/delta_xtcav23_screen_v4_blen_s0`, partial, ignore); reruns are the `*_v4r_*` logdirs.

3. **The controller-region widening to L1 −26.5° (round-2 physics M3, adopted for the campaign) rests on raw
   machine phase readbacks; the sim2real BC11-servo result maps the real 04-15 L1 −26° to sim ≈ −22.35° at
   matched compression.** In the sim, L1 below ~−23° is over-compressed: Bmad pilot rows at L1 −26.8 / L2 −32.0
   have ΔE 0.44% (vs 1.85% at golden) and spacing ~57 µm, and NO bunch split works there (energy gate: best
   possible single cut still misassigns 11–12%; valley 60%; z-lobe 56–66%). Consequences:
   - the running campaign is left as configured (Ryan's decision; the data are valid, only less useful):
     ~5/8 of the 1000 controller-region rows (L1 −23.5…−18.5) are in the relevant band;
   - RL start boxes use L1 [−23.5, −19] (S2 onward). S1 (cloud, already running) uses [−26, −19]; harmless
     in cloud mode (ground truth) but its evaluation is split by region.
   - the deployed validity rule (two peaks + |spacing| > 60 µm) calls ~91% of start-box readings valid while
     only ~60% are truly < 10% misassigned: the online analysis cannot detect misassignment from the image
     alone. Worth a real-data check of how often the real energy gate fails near full compression.

2. **Training launch loop typo** (≈19:40): a zsh `for`/`set --` loop did not word-split, so both S1 runs
   failed to start (bad config path). Removed the 4 junk log/pid files it created and relaunched explicitly.
   No other effect.
1. **Auto-launcher stopped at `git add`** (18:48): `results/` is gitignored, so adding `decision.json` failed;
   nothing had been launched or written. Fixed (decision summary now in the launch config header, 5c7425c) and
   relaunched 18:50.

## Progress log

- 18:50 campaign launched (5e4 / grid 64 chosen by the pass rules: noise −0.06/−0.15 µm spacing;
  g64 vs g100 +0.05/−0.23 µm, −1.1%/+0.07% drive σz; 5e4 vs 1e5 ≤ 0.11 µm, all effects ≤ 0.1 µm).
- 19:0x surrogate training-side fixes (all tested; 269 tests pass):
  - `--orig-split processed/xtcav23_combined.h5=xtcav23_full,xtcav23_pilot`: reproduces the old flow's
    random split exactly (4975 train / 553 val, verified against the DataModule) so its val rows stay in val
    and its train rows in train (round-2 ML MAJOR 2); group split stratified by campaign; holdout rows
    excluded (`dm.test`).
  - selection metrics (`surrogate/selection.py`): `val_paired_delta_rms_um` (CRN predicted vs true injector
    effect on spacing over val pairs) and `val_old_spacing_mae_um`; `--select-metric`, `--patience 0`
    (fixed-epoch cosine fine-tune). preprocess writes `pair_key` / `pair_ref`.
  - smoke on real data (pilot + valgate in the 37-knob frame; widened xtcav23 ckpt; 2-epoch warm start on
    GPU): all weights loaded, val_old_spacing_mae 7.4 µm (= the known flow error).
  - screen plant: injected charges follow drive_fraction / total charge under per_sample tables
    (q_in = Q f 1.001, Q (1−f) 0.999; matches the pilot's recorded charges to 1.3e-4) — round-2 ML BLOCKER 1.
- 19:30 env v4 built (subclass; 5 new tests) + explicit reward tolerances; S1 cloud configs; launched both S1 runs
  (blen_dlog / adapter, seed 0, 32 actors, 300 iters ≈ 30 min each).
- 19:35 screen v4 pieces (features: `orientation: charge`, `roi`, `screen_valid_deployed`; plant: `optics_choices`,
  `tdc_voltage_range` with per-env calibrated streak, `arrival_jitter_s`); S2 config; smoke OK (obs 58, grads flow).
- 19:40 found the deep-compression box issue (problem 3); S2 start box L1 [−23.5, −19]: truth-valid 60%,
  deployed-valid 91%.
- 19:45 post-campaign chain started (waits); surrogate gate script written + smoke-run.
- Impact-T at 5e4 / grid 64, 64 concurrent: ~2080 s per beam (the grid solve dominates; 1e5 was 2450 s) →
  ~2.3 h of Impact-T, then the Bmad rows; campaign ETA ~08:00.
- 20:00 S1 (first) evaluated: see problem 4. Actuator comparison on reachable goals: adapter 6.6 µm / 70% vs
  blen_dlog 7.4 µm / 67% (one seed each; not significant). Relaunched with reachable goals: S2 screen blen_dlog
  (GPU 0) and S1r blen → adapter (GPU 1, sequential).
- 20:10 online features: per-image `band_stats` + per-step `step_features` (median per sign, then the ±V
  half-difference; on the machine +V and −V images are ~11 s apart, not the same beam). Test: reduces exactly
  to the pair function. `scripts/xtcav_online_replay.py` replays it on the real E300 L2 scans with the deployed
  preprocessing + Tier-0 calibration → `results/online_replay_2026-10-15/` (running, CPU 1 thread nice 10).
  Smoke (30 shots/sign/step, E300_15671): step 1 spacing 99.8 µm with +V/−V single-sign estimates 98 / 101 µm
  (consistent); single-image robust σ 3–6 µm on clean steps, ~55–80 µm on the near-full-compression steps
  where the energy gate often fails (10–21 of 30 images).
- 21:20 real-DAQ replay done (`results/online_replay_v3_2026-10-15/REPORT.md`). Fix found: the energy cut must
  be one per step from the +V images and shared with −V (D2). A per-image −V valley locks onto another dip because
  the TCAV's energy kick flips with V, which wrecked E300_15673 steps 9–18 (fixed: smooth 240 → 143 µm). Noise vs
  env: per-image robust σ 6–15 µm (env 15), per-step ~1–2.6 µm at M = 30 (env floor 6 µm: conservative).
  Unsolved: valley-jump misassignment at TEST_15668 steps 25/26/29 that the deployed rule calls valid (drive
  fraction jumps; candidate truth-free check).
- 23:40 crop-edge check added to the online features too and replayed on the real data
  (`results/online_replay_v4_2026-10-15/REPORT.md`): it flags only E300_15673 steps 1–6 (277–304 µm; the real
  images touch the auto-crop edge there too) and nothing else. The sim's and the real data's large-spacing
  clipping agree.
