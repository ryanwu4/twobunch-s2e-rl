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

## Progress

| Time (PDT) | Phase | What happened |
|---|---|---|
| 00:55 | plan | Plan committed (`f9ff67a`). |
| 01:05 | 0 | Compatibility guards + goldens committed (`b1d2459`): 153 pass + 1 strict xfail (L2 drift bug). |
| 01:07 | 2 | Validation-gate Bmad campaign launched (`data/xtcav23_valgate_2026-10-14/`, log `logs/overnight_2026-10-14/valgate_sweep.log`). |
| 01:07 | 3a, 2 | Two parallel agents started: injector wrapper + Impact-T sensitivity study; xtcav_segment port + Cheetah 0.8.4 + Stage-A re-validation. |

## Needs Ryan's review in the morning

