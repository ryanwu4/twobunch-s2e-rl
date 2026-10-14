# Overnight run log — 2026-10-14

Autonomous run on branch `dev/2026-10-13`, following `docs/architecture_plan.md`.
Started ~01:00 PDT; Ryan checks back ~09:00 PDT.

## Ground rules (agreed before starting)

- Work only on `dev/2026-10-13` (and a new branch in xtcav-gpsr-tuning); commit per phase, never push.
- Never modify or delete existing `data/`, `processed/`, `trained/`, `logs/`, `results/` content;
  new outputs go to new dated directories.
- FACET2-S2E untouched (Impact-T wrapper copies inputs into per-run workdirs).
- ≤ ~96 CPU workers at `nice 10`; both GPUs; ≤ 500 GB new disk.
- Cheetah 0.8.4 installed into `slac-rl` (approved).

## Assumptions made in Ryan's absence

(Each entry: what, why, how to undo.)

1. **Policy → flow ckpt map** (`tests/compat_artifacts.py::POLICY_FLOW`): 8-knob runs (bptt, bptt_dr,
   shac, shac_dr, bptt_gc) → `twobunch_flow_v4`; 26-knob gc runs (combined, matched, tightbox,
   sigmaz) → `twobunch_combined_ft`. Evidence: config `campaign_h5` / "Run with:" comments, the
   presentation_figures README, and the setpoint json provenance blocks. Not yet cross-checked by
   reproducing `eval_best_goal*.json` (planned).
2. **v1 `twobunch_flow` ckpt is already unloadable** with the current model code (has `whiten_head`,
   code expects per-bunch `whiten_heads.{0,1}`) — pre-existing, not caused tonight. Listed as
   `KNOWN_BROKEN_CKPTS`, not guarded. No policy depends on it. Fixing needs a legacy model path —
   left for you to decide.
3. **Validation-gate campaign design** (`scripts/make_valgate_manifest.py`): 120 "controller-region"
   samples (L1 = golden ± 2°, L2 = golden ± 2.5° uniform — my choice of controller sub-box; all
   other xtcav23 knobs + Q2/Q1/Q0FF Gaussian at readback-error σ: magnets 1%, movers 0.1 mm,
   kickers 5e-4 kG·m, energy offsets 0.1% of section gain) + 40 wide-LHS samples over the full
   xtcav23 box with Q2/Q1/Q0FF ±5% of golden + 2 baseline repeats. 100k particles, CSR + wakes,
   xtcav23 fixed_overrides, beams saved at XTCAVF and PENT. 56 workers, nice 10.
4. **Kicker readback σ** = 5e-4 kG·m (~10% of golden; no readback spec available) — assumed.
5. **Delta controller is a separate env class** (`rl/delta_env.py::DeltaControlEnv`) rather than new
   options on `TwoBunchFlowEnv` (plan said extend diff_env). Reason: keeps the existing env — and
   every policy trained on it — byte-identical, and keeps the delta logic readable in one place.
   Same DiffRL API, so the vendored BPTT trains it unchanged.
6. **The ±1° readback offset is invisible to a delta controller by construction.** With obs built
   only from deltas and beam measurements, a constant offset between readback and truth cannot
   change the dynamics or the obs (the true start point is randomized anyway). It is kept as the
   readback offset (`cmd0 = start − offset`) for honest logs; the **gain error** U(0.8, 1.2) is
   the misspecification the policy actually has to tolerate. Worth confirming this matches your
   intent.
7. **Default readback σ for phases = 0.3°** (other knobs per assumption 3) in `delta_env.readback_sigma`.
8. **Delta-controller settings (my choices, `configs/rl/delta_xtcav23_cloud.yaml`)**: action step
   0.3°/unit action/step; episode 32 steps; start box L1 [−22.5, −18.95] (asymmetric: golden −20.29
   is only 2.2° below the xtcav23 box top −18.5, kept 0.45° margin), L2 golden ± 2.5° =
   [−38.04, −33.04]; reward weights spacing 1.0, drive σ_z 0.5, survival hinges 2.0 at T = 0.98,
   boundary 0.5 (controls only); ΔE observed only; witness σ_z not used; 32 actors (64 OOMs the
   80 GB A100 with BPTT's retained graph), 300 epochs, lr 5e-4.
9. **Cloud mode = pure ground truth** (no shot noise, no scale error) — the clean baseline. Shot
   jitter and streak-scale error are applied in screen mode only.
10. **Goal ranges in the objectives block are only for obs normalization** under achievable-goal
    sampling (goals come from the plant itself), so they are wide placeholders.
11. **Screen features are a two-image (+V / −V) measurement** (`xtcav_segment/features.py`):
    centroid half-difference cancels each bunch's TCAV-off x offset; the mean of the two image
    variances keeps the TCAV-off spot as resolution (≈ 30 µm z-equivalent at 22 MV — "blur
    included" per your answer; I removed my extra 17 µm term to avoid double counting). A single
    +V image gave spacing with the wrong sign (transverse offsets read as time).
12. **Bunch split = z-lobe, not energy gate, for screen training.** In sim the bunches overlap in
    energy at DTOTR: the energy gate misassigns ~13 % of charge everywhere in the start box (0 %
    valid). The z-lobe split (gate on each image's streak axis) gives 67 % valid over the start box,
    spacing error 1.0 % (median, valid), charge fraction 4 %, drive σ_z +12 % (resolution).
    Both splits are implemented (`screen.split: energy | zlobe`). **Conflict to resolve:** your
    2026-07-23 sim2real notes name the energy-gated split as the reference; the GPSR plan
    (`screen.charge_split`) says the z-lobe split is the robust real observable.
13. **ΔE from the screen is ~27 % off** (y-axis TCAV-off offsets aren't cancelled). ΔE is
    observe-only, so it is left as is and flagged.
14. **B5D36 at DTOTR kept at fixed field (machine BDES)**, as in the original port. Bmad instead
    scales its field with the linac energy setpoints → up to ~1 % energy-axis scale difference at
    DTOTR (no effect at PENT). Your call which to follow.
15. **TCAV 22 MV, phase 0** (the valtracks / Stage-A setting); segment quads Q2/Q1/Q0FF held at the
    spec's golden values in the screen plant (no readback jitter there in v1).
16. **Screen measurement realism defaults**: streak-scale error U(0.8, 1.2) on spacing and drive
    σ_z; shot noise per shot 15 µm spacing (DAQ: 10–15 µm at fixed step, an upper bound), 2 µm drive
    σ_z (assumed), 0.03 drive charge fraction (DAQ), averaged over M = 50.
17. **Stopped the cloud seed-1 run at iteration 102** to free GPU 1 for the screen run (one seed of
    cloud mode continues). Screen run uses 16 actors (memory: ~37 GB) vs 32 for cloud — not an
    exact like-for-like comparison.

## Progress

| Time (PDT) | Phase | What happened |
|---|---|---|
| 00:58 | plan | Plan committed (`f9ff67a`). |
| 01:05 | 0 | Compatibility guards + goldens (`b1d2459`): 153 pass + 1 strict xfail (L2 drift bug). |
| 01:07 | 2 | Validation-gate Bmad campaign launched (`data/xtcav23_valgate_2026-10-14/`, log `logs/overnight_2026-10-14/valgate_sweep.log`). |
| 01:07 | 3a, 2 | Two parallel agents: injector wrapper + Impact-T sensitivity study; xtcav_segment port + Cheetah 0.8.4 + Stage-A re-validation. |
| 01:15–01:21 | 1a, 1b | KnobSpace, artifacts, L2 frame fix, glob → resolve_ckpt (`c73ac37`); loud norms cache (`cdf5fc0`); generic goals (`15d4f73`). |
| 01:24–01:37 | 3 | DeltaControlEnv (`662506e`); training wiring + cloud config (`dbf7870`); rollout + eval_delta (`7e46f8c`). |
| 01:32 | 3 | Cloud-mode training seed 0 on GPU 0 (seed 1 on GPU 1 at 01:35). First attempt OOM'd (2 runs × 64 actors on one GPU) → 1 run/GPU, 32 actors. |
| 01:37 | 2 | xtcav_segment port (`51d05c0`): Stage A on 0.8.4 bit-identical to the clone; Cheetah vs Bmad at PENT ≤ 2.3e-7 rel. on 8 valgate samples. |
| 01:45 | 3 | Cloud seed 1 stopped (iter 102); screen-mode training seed 0 on GPU 1 (`delta_xtcav23_screen_s0`). |
| 01:51 | 3 | Screen plant + features + fallback (`36768b5`). |
| 01:53–01:56 | 2 | sim/bmad.py, run_sweep on it (`ab33c02`); validate_delta export/track/compare (`53c7c31`), smoke-tested end to end at 5k particles. |

| 02:03 | 3 | Cloud seed 0 finished (300 iters, 30 min); screen seed 1 started on GPU 0. |
| 02:05 | 2 | **PENT gate passes**: Cheetah XTCAVF→PENT vs Bmad on all 56 finished valgate samples, every observable ≤ 8e-7 relative (`results/segment_validation_084/pent_valgate.json`). |
| 02:10 | 4 | Surrogate YAML config + knob_keys/treaty_point hparams (`ae68b63`); campaign.json writer for new campaigns (`6ece2ae`). |
| 02:19 | 3 | **Cloud delta controller Bmad check**: 24/24 setpoints tracked (20k particles), see Results. |

| 02:23 | 3a | Injector sensitivity study finished (81 runs, 43 min); see Results. |
| 03:13 | 3 | Screen seed 0 finished; eval above. 100k Bmad re-check of the cloud policy done. Valgate campaign complete (162/162). |
| 03:15 | 3 | Started: screen + injector stand-in run on GPU 1 (`delta_xtcav23_screen_inj_s0`); Bmad check of screen seed 0 at 100k (24 workers); final PENT gate over all 162 valgate samples. |

| 03:34 | 2 | Final PENT gate over all 162 samples: passes (worst 4.5e-6 rel.). |

(Note: an earlier version of this table had wrong, estimated clock times; corrected from git at 01:58.)

## Results so far

**Cloud-mode delta controller** (`logs/overnight_2026-10-14/delta_xtcav23_cloud_s0`, L1/L2 phase deltas
only, hidden gain error U(0.8,1.2) + readback-scale jitter on the other 21 knobs + random starts):

- Surrogate eval, 512 episodes (`results/rl/delta_xtcav23_cloud_s0/delta_eval/`): spacing error
  median 52 → **3 µm** (p90 7 µm), 97 % within 10 µm, settles in 4 steps; drive σ_z error 29 % →
  **4.8 %** (p90 16 %), 78 % within 10 %, settles in ~26 steps (slow — σ_z is the harder, weaker-
  weighted target). Survival T_d 0.99 / T_w 0.97.
- **Bmad open-loop check**, 24 setpoints, 20k particles (`results/rl/delta_xtcav23_cloud_s0/bmad_check/compare.json`):
  Bmad vs goal spacing **6.8 µm** median (4.9 %), drive σ_z **2.7 %**; surrogate vs Bmad spacing
  2.9 % median (p90 11 %), σ_z 5.0 % (p90 16 %), ΔE 4.2 %.
- **Same check at 100k particles** (campaign fidelity, `bmad_check_N100000/`): Bmad vs goal spacing
  **6.75 µm**, drive σ_z **3.2 %**; surrogate vs Bmad 3.1 % / 4.9 % — the 20k result holds.

**Screen-mode delta controller** (`delta_xtcav23_screen_s0`; XTCAV ±V features, z-lobe split,
streak-scale error U(0.8,1.2), shot noise M=50, invalid readings fall back to the cloud twin in the
reward), surrogate eval 256 episodes: spacing error 53 → **6 µm measured / 5 µm true** (p90 17 µm),
72 % within 10 µm, settles in ~23 steps (vs 4 in cloud mode: noisy, sometimes-invalid
measurements); drive σ_z 23 % → **4.5 % measured / 8.1 % true**, 78 % within 10 %. Screen reading
valid 66–71 % of the time. Survival unchanged (0.99 / 0.97).

**Injector sensitivity (Impact-T → Bmad → XTCAVF, 2e4 particles, grid 32; `data/injector_sensitivity_2026-10-14/summary.md`)**,
at the approved per-episode magnitudes, "absolute" arrival mode (the injector ⟨t⟩ shift kept —
physically complete): laser timing ±0.2 ps → spacing ±7.6–7.9 µm, drive σ_z ∓1.1 µm (≈ 0.31° L1 /
0.16° L2 equivalent); gun amplitude ±0.25 % → ∓4 µm; bunch delay ±0.1 ps → ±4.6 µm; drive
fraction ±0.02 → ±2.4 µm and the charge ratio itself ±0.035; gun phase ±0.25° → ±1.3 µm;
solenoid / spot / MTE / σ_t / L0A phase / charge ≤ 0.8 µm. Controller authority: 47 µm/° (L1),
−44 µm/° (L2). Noise floor 0.02 µm (Hammersley burnin pair), so all effects are resolved.
- **Implication for the delta controller:** these act mainly as ≤ 0.3° common shifts of the
  effective L1/L2 phases, which random true starts already cover (3× below the readback offset).
  The part a feedback controller can't absorb this way is the direct charge-ratio change →
  added `measurement.feature_offset_sigma` (hidden per-episode offset on observed features, e.g.
  `drive_charge_fraction: 0.035`) for future runs; tonight's screen runs don't use it.
- **Pipeline finding:** in the legacy "centred" mode (FACET2-S2E time-centres the input beam) the
  timing effect is ~6× smaller (spacing ±1.3 µm): every existing campaign erases injector arrival-
  time jitter. The injector campaign should keep absolute timing.
- Baseline reproduction (2e4 / grid 32 vs the golden 5e5 / grid 100 beam): charge, charge ratio,
  energies, bunch separation reproduce; transverse sizes +4–7 %, emittance +6–19 % (fidelity).

**PENT gate (Cheetah segment vs Bmad)**: passes on all 162 valgate samples (120 controller-region +
42 wide-box incl. Q2/Q1/Q0FF ±5 %): worst relative difference over every observable 4.5e-6
(witness BMAG_y); spacing / σ_z / emittances ≤ 1e-6.

## Needs Ryan's review in the morning

- Energy-gate vs z-lobe split (assumption 12); B5D36 fixed-field vs Bmad energy-scaled (14).
- Whether the readback offset being irrelevant to the delta controller matches your intent (6).
- v1 `twobunch_flow` ckpt unloadable (2): fix with a legacy model path, or leave?
- xtcav-gpsr-tuning branch `reexport-xtcav-segment` (commit e47ac1b, not pushed; its tests pass).

