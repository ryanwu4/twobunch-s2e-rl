"""Conditional normalizing-flow surrogate over the two-bunch PENT beam.

Generates the drive + witness 6D phase-space at PENT conditioned on the 8 sweep knobs,
plus per-bunch feasibility (witness viability + per-bunch transmission). One shared
RealNVP flow selected by a per-bunch index k, with a whitening MLP that places each bunch
(mu_k, Sigma_k) so the flow only models the standardized residual. Sampling is
reparameterized so observables are differentiable w.r.t. the knobs (for later MBRL).

Design: lab-notebook 2026-06-09_nf-two-bunch-surrogate.md (v3). Flow/coupling math reused
from slac-photoinjector-rl surrogates/flow. Runs in the `slac-rl` (torch) env.
"""

# 6D phase-space column order; matches compute_emittance_torch and the ParticleGroup read.
COORD_KEYS = ("x", "y", "z", "px", "py", "pz")
LATENT_DIM = 6
# Electron rest energy [eV]: geometric 4D emittance / mc^2^2 = norm_emit_4d.
ELECTRON_MC2_EV = 0.51099895e6
# Fixed macroparticles kept per bunch (subsample down; bunches with fewer live particles
# are excluded from density training but still used for the feasibility heads).
DEFAULT_P = 1024
BUNCHES = ("drive", "witness")  # index k: 0=drive, 1=witness

__all__ = ["COORD_KEYS", "LATENT_DIM", "ELECTRON_MC2_EV", "DEFAULT_P", "BUNCHES"]
