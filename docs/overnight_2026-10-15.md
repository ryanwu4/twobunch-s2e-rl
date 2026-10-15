# Overnight 2026-10-14 → 15: injector campaign + training-side work

Ryan away until ~11:00 PDT 2026-10-15. Same ground rules as the first overnight
(`docs/overnight_2026-10-14.md`): work on `dev/2026-10-13`, commit per step, never push; never modify or delete
existing data/, processed/, trained/, logs/, results/ content (new outputs in new dirs); FACET2-S2E untouched;
CPUs belong to the campaign (64 workers, nice 10) — other CPU work kept light; GPUs free for training.

## Status at a glance (updated through the night)

| Item | State |
|---|---|
| Injector campaign `injector_full` (3027 rows, 5e4 / grid 64, commit d20918d, worktree `.worktrees/injector_full_d20918df54`) | launched 18:50 |
| env v4 (`rl/delta_env_v4.py`) + S1 cloud trainings (blen_dlog vs adapter, seed 0, GPUs 0/1) | started ~19:45 |

## Decisions taken in Ryan's absence (numbered; each with the reason)

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
6. **S1 start box** L1 [−26, −19], L2 [−40.5, −33.5] (inside the plant box; covers the 04-15 operating points
   per physics round 2 M3). Action cost 0.02·a², 20% off-context goals, 8-step history, reward on truth.

## Problems encountered

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
