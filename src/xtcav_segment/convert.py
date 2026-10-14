"""openPMD ParticleGroup <-> Cheetah ParticleBeam conversion.

Ported unchanged (apart from dropping the Cheetah-clone ``sys.path`` hook) from
xtcav-gpsr-tuning @ c1f68ce ``src/xtcav_gpsr_tuning/convert.py``; runs on
cheetah-accelerator 0.8.4.

Coordinate contract (plan gotcha #6 — read before touching):

* The per-particle **absolute time `t` is authoritative** for the longitudinal
  coordinate. Beams saved by ``FACET2_S2E.getBeamAtElement`` carry an
  overwritten ``z = -c*(t - <t>_w)`` (relative to the BEAM's charge-weighted
  mean arrival, NOT the lattice reference particle) plus the untouched
  absolute ``t``. We ignore the stored z and rebuild from t.
* Cheetah's longitudinal coordinate is ``tau = +c*(t - t_ref)`` (positive =
  LATE = tail); Bmad's is ``z = -beta*c*(t - t_ref)`` (positive = EARLY =
  head). Cheetah's Bmad-X maps convert internally via ``z = -beta*tau``.
* ``t_ref``: use the **charge-weighted beam mean arrival** (the
  ``ref_time_s=None`` default). Measured 2026-07-30 on the baseline reference
  beams: Bmad's effective crab-cavity phase reference is the bunch's own z
  frame — regressing the per-particle Bmad kick (plus−off at XTCAVF exit)
  gives slope +0.4788/m = V*k_rf/p0c to 0.02% with a phase offset of
  −0.001 deg, i.e. the beam mean IS the zero crossing in Bmad's frame. The
  lattice ``ele_head ref_time`` is on a DIFFERENT clock (bunch mean arrival
  is −13.8 ns ≈ −4.13 m off it for the 2024-10 input beam) and must NOT be
  used as the RF phase reference (11.424 GHz: 1 ps = 4.1 deg; using it
  produced a 14 mm spurious centroid kick and a sign-flipped streak).
  ``ref_time_s`` stays available for explicit control.
* ``ref_energy_eV`` should be the lattice reference energy at the element
  (spec meta ``e_tot``), not the beam median: Bmad normalizes px, py, pz by
  the lattice p0c. Cheetah delta = (E - E_ref)/p0c, matching Bmad pz to
  O((mc^2/E)^2) ~ 3e-9 at 10 GeV.
"""

from __future__ import annotations

import math

import numpy as np
import torch


C_LIGHT = 299792458.0


def _species(name: str, factory_kwargs: dict):
    from cheetah.particles import Species

    return Species(name, **factory_kwargs)


def particlegroup_to_beam(
    P,
    ref_energy_eV: float,
    ref_time_s: float | None = None,
    device=None,
    dtype=torch.float64,
):
    """Convert a pmd_beamphysics ParticleGroup to a Cheetah ParticleBeam.

    Uses absolute t (authoritative), referenced to ``ref_time_s`` (lattice
    reference time at the element; defaults to the charge-weighted mean
    arrival). Returns (beam, info_dict).
    """
    import cheetah

    fk = {"device": device, "dtype": dtype}
    species = _species(str(P.species), fk)
    mc2 = float(species.mass_eV)
    p0c = math.sqrt(ref_energy_eV**2 - mc2**2)

    t = np.asarray(P.t, dtype=np.float64)
    w = np.asarray(P.weight, dtype=np.float64)
    if ref_time_s is None:
        ref_time_s = float(np.average(t, weights=w))

    x = torch.as_tensor(P.x, **fk)
    y = torch.as_tensor(P.y, **fk)
    px = torch.as_tensor(P.px, **fk) / p0c
    py = torch.as_tensor(P.py, **fk) / p0c
    tau = torch.as_tensor(C_LIGHT * (t - ref_time_s), **fk)
    delta = (torch.as_tensor(P.energy, **fk) - ref_energy_eV) / p0c

    particles = torch.stack([x, px, y, py, tau, delta, torch.ones_like(x)], dim=-1)
    beam = cheetah.ParticleBeam(
        particles=particles,
        energy=torch.tensor(float(ref_energy_eV), **fk),
        particle_charges=torch.as_tensor(w, **fk),
        survival_probabilities=torch.ones_like(x),
        species=species,
        device=device,
        dtype=dtype,
    )
    info = {
        "ref_energy_eV": float(ref_energy_eV),
        "ref_time_s": float(ref_time_s),
        "p0c_eV": float(p0c),
        "n_particles": int(len(t)),
    }
    return beam, info


def cloud_to_beam(
    cloud: torch.Tensor,
    ref_energy_eV: float,
    particle_charges: torch.Tensor,
    device=None,
    dtype=torch.float64,
):
    """Flow-sampled cloud (..., N, 6) in (x, y, z, px, py, pz) [m, eV/c] ->
    Cheetah ParticleBeam. Fully torch-differentiable (no ParticleGroup detour) —
    knob-δ gradients flow through to the segment/screen.

    Conventions (critic finding M2 — verified against the campaign save path):
    * The training clouds' z is EXACTLY -c*(t - <t>_w) (``getBeamAtElement`` writes
      it with full c, not beta*c), and Bmad's TDC zero crossing is the charge-
      weighted beam mean (Stage-A measurement) -> **tau = -z exactly**, no beta
      correction. No re-centering is applied here: the flow's frame IS the
      training frame (per-sample beam mean = RF reference); per-sample residual
      phase offsets are measured in Stage C, not assumed away.
    * The flow emits momenta, not energy: delta = (sqrt(px^2+py^2+pz^2+mc2^2)
      - E_ref)/p0c, with E_ref the lattice reference energy at the handoff
      (spec meta ``ref_energy_eV``), matching ``particlegroup_to_beam``.
    * ``particle_charges``: per-particle charge in C, shape broadcastable to
      cloud[..., 0]. For a flow bunch, pass T_k * Q_k_full / N per particle so
      the downstream KDE charge weighting matches the plan.
    """
    import cheetah

    fk = {"device": device, "dtype": dtype}
    species = _species("electron", fk)
    mc2 = float(species.mass_eV)
    p0c = math.sqrt(ref_energy_eV**2 - mc2**2)

    c = cloud.to(**{k: v for k, v in fk.items() if v is not None})
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    px, py, pz = c[..., 3], c[..., 4], c[..., 5]
    tau = -z
    energy = (px**2 + py**2 + pz**2 + mc2**2).sqrt()
    delta = (energy - ref_energy_eV) / p0c

    particles = torch.stack(
        [x, px / p0c, y, py / p0c, tau, delta, torch.ones_like(x)], dim=-1
    )
    charges = torch.as_tensor(particle_charges, **fk).broadcast_to(x.shape)
    return cheetah.ParticleBeam(
        particles=particles,
        energy=torch.tensor(float(ref_energy_eV), **fk),
        particle_charges=charges,
        survival_probabilities=torch.ones_like(x),
        species=species,
        device=device,
        dtype=dtype,
    )


def beam_to_cloud(beam) -> torch.Tensor:
    """Inverse of ``cloud_to_beam``: Cheetah ParticleBeam -> (..., N, 6) cloud in
    (x, y, z, px, py, pz) [m, eV/c], differentiable.

    z = -tau (same frame contract as ``cloud_to_beam``: tau is referenced to the
    beam's frame at the handoff, so z is the openPMD-style -c*(t - t_ref)).
    Momenta are rebuilt from delta at the beam's reference energy:
    E = E_ref + delta*p0c, |p| = sqrt(E^2 - mc2^2), pz = sqrt(|p|^2 - px^2 - py^2).
    Particle charges / survival are NOT returned; take them from the beam.
    """
    p = beam.particles
    mc2 = beam.species.mass_eV
    e_ref = beam.energy.unsqueeze(-1) if beam.energy.dim() else beam.energy
    p0c = (e_ref**2 - mc2**2).sqrt()
    energy = e_ref + p[..., 5] * p0c
    ptot = (energy**2 - mc2**2).sqrt()
    px = p[..., 1] * p0c
    py = p[..., 3] * p0c
    pz = (ptot**2 - px**2 - py**2).sqrt()
    return torch.stack([p[..., 0], p[..., 2], -p[..., 4], px, py, pz], dim=-1)


def weighted_stats(values: torch.Tensor, weights: torch.Tensor) -> dict:
    """Charge-weighted mean/std of a 1-D coordinate tensor."""
    w = weights / weights.sum()
    mean = (w * values).sum()
    std = ((w * (values - mean) ** 2).sum()).sqrt()
    return {"mean": float(mean), "std": float(std)}


COORDS = ("x", "px", "y", "py", "tau", "delta")


def compare_beams(beam_a, beam_b, center: tuple = ("tau",)) -> dict:
    """Index-aligned per-particle differences between two ParticleBeams.

    Coordinates listed in ``center`` are compared after removing each beam's
    charge-weighted mean (separates shape agreement from reference-time /
    reference-orbit bookkeeping); their mean offsets are reported separately.
    Returns {coord: {rms, max, mean_offset}}.
    """
    assert beam_a.particles.shape == beam_b.particles.shape, "particle count mismatch"
    w = beam_a.particle_charges
    w = w / w.sum()
    out = {}
    for i, name in enumerate(COORDS):
        a = beam_a.particles[..., i]
        b = beam_b.particles[..., i]
        mean_a = float((w * a).sum())
        mean_b = float((w * b).sum())
        if name in center:
            a = a - mean_a
            b = b - mean_b
        d = a - b
        out[name] = {
            "rms": float((w * d**2).sum() ** 0.5),
            "max": float(d.abs().max()),
            "mean_offset": mean_a - mean_b,
            "std_a": weighted_stats(beam_a.particles[..., i], beam_a.particle_charges)["std"],
            "std_b": weighted_stats(beam_b.particles[..., i], beam_b.particle_charges)["std"],
        }
    return out
