# Surrogate roadmap (v3 conditional NF)

Plan for `twobunch_s2e_rl.surrogate` — not yet implemented. The authoritative design note
is the lab notebook: `lab-notebook/claude/analyses/2026-06-09_nf-two-bunch-surrogate.md`
(§2 = v3). This file maps that design onto module boundaries and lists code to reuse.

## Goal

Given the 8 sweep knobs (+ domain-randomization vars), **generate** the 6D drive+witness
beam at PENT — the full particle distribution, for matching against downstream diagnostics
— plus per-bunch transmission. Trains on the campaign output in `data/<name>/`.

## What the data looks like

- `data/<name>/sample_{idx:05d}.json` — `knobs` (8-D input) + per-treaty-point scalar
  specs. Knobs are the conditioning vector. (Particles are NOT in the json.)
- `data/<name>/sample_{idx:05d}_PENT.h5` — openPMD `ParticleGroup`, 100k macroparticles,
  drive/witness tagged by two unique per-particle `weight` values (±0.1%).
- Join json↔h5 by `idx`.

## Components (planned modules)

| module | responsibility |
|---|---|
| `surrogate/preprocess.py` | walk `data/<name>/`, load each `*_PENT.h5` via `pmd_beamphysics.ParticleGroup`, split drive/witness by `weight`, stack 6D coords, subsample to fixed P, join the 8 knobs from the json; emit `processed/*.h5` (settings, per-bunch particles, transmission) + `*_norm.json` (StandardScaler stats, knob bounds) |
| `surrogate/dataset.py` | load preprocessed h5; normalize knobs→[0,1], standardize particles; per-bunch indexed samples |
| `surrogate/model.py` | whitening MLP `(c, ε, s, k)→(μ_k, Σ_k, T_k)` + conditional NF (RealNVP / RQ-spline coupling) on the standardized residual; per-bunch (drive/witness) mixture base |
| `surrogate/train.py` | NLL (+ light moment terms: per-bunch emittance/energy, Δz spacing, ΔE, transmission); Lightning, checkpoints to `trained/` |
| `surrogate/eval.py` | phase-space overlays, per-bunch moment parity, transmission calibration |

## Reuse (don't reinvent)

- **openPMD → tensor + split:** templates in
  `prev_work/photoinjector-rl-clean/src/photoinjector_rl/flow_surrogate/{preprocess,dataset,model}.py`
  (also `slac-photoinjector-rl/src/photoinjector_rl/surrogates/flow/`). Use
  `pmd_beamphysics.ParticleGroup` (not raw h5py).
- **Drive/witness split:** `FACET2_S2E.getDriverAndWitness(P)` (splits by the two unique
  `weight` values) — or replicate inline to avoid the heavy import in the torch env.
- **NF skeleton:** `prev_work/accelerator_flow_model/train_norm_flow_conditional_rqs.py`
  (normflows, 16 coupling layers, RQ-spline).

## Key deltas from the legacy photoinjector flow

1. **Condition on the 8 knobs** (+ DR vars + per-bunch index k), NOT on 45-D initial
   moments. The legacy flows condition on init-moments only; this surrogate is knob→beam.
2. **Add a transmission / acceptance head.** ~31% of the sampled knob space loses the
   witness bunch entirely before PENT (see the campaign analysis), and the drive also
   scrapes. A normalized density cannot represent loss — model pre-cut density + an
   explicit per-bunch transmission `T_k` (and treat witness-destroyed samples as a
   distinct, learnable outcome, not dropped data).
3. **Two-component (per-bunch) base** so the drive/witness gap is not bridged by one
   continuous map.

## Then: RL (`twobunch_s2e_rl.rl`)

Mirror `photoinjector-rl-clean`: a differentiable surrogate env + `SurrogateVecEnv`, with
PPO/SHAC/BPTT. Reward from witness emittance/BMAG, bunch spacing, and collinearity
(transverse offset + angular misalignment), gated by feasibility (witness survival).
