"""Differentiable beam observables on (B, N, 6) clouds in (x,y,z,px,py,pz).

Positions [m], momenta [eV/c]. All functions return (B,) and are differentiable, so
the surrogate's sampled clouds give gradients w.r.t. the knobs for MBRL. The emittance
kernel is reused verbatim from the photoinjector flow (slac-photoinjector-rl) and matches
ParticleGroup.norm_emit_x/4d to ~1e-9.

Two conventions to note vs the campaign's getBeamSpecs quantities:
- Centroids/offsets here use the MEAN (differentiable); getBeamSpecs uses the MEDIAN.
  On real data the transverse offset / angular misalignment differ by ~8-12% (median) and
  up to ~55% (tail, where halo drags the mean). Means are used deliberately for MBRL
  gradients; treat campaign-median parity as approximate.
- The PENT openPMD frame stores z = -(t - <t>)*c, so larger z = earlier arrival. bunch
  spacing below is signed to match the campaign (drive - witness in z == witness - drive
  in arrival time), i.e. positive when the witness trails the drive.
"""
from __future__ import annotations

import torch

from . import ELECTRON_MC2_EV

MC2 = ELECTRON_MC2_EV


def compute_emittance_torch(particles: torch.Tensor) -> dict[str, torch.Tensor]:
    """Geometric 2D/4D/6D emittances (sqrt det Sigma) for (B,N,6) in (x,y,z,px,py,pz)."""
    means = particles.mean(dim=1, keepdim=True)
    centered = particles - means
    cov = torch.bmm(centered.transpose(1, 2), centered) / (particles.shape[1] - 1)
    out: dict[str, torch.Tensor] = {}
    cov_x = cov[:, [0, 3]][:, :, [0, 3]]
    out["x_xp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_x)) + 1e-16)
    cov_y = cov[:, [1, 4]][:, :, [1, 4]]
    out["y_yp"] = torch.sqrt(torch.abs(torch.linalg.det(cov_y)) + 1e-16)
    cov_z = cov[:, [2, 5]][:, :, [2, 5]]
    out["z_delta"] = torch.sqrt(torch.abs(torch.linalg.det(cov_z)) + 1e-16)
    cov4 = cov[:, [0, 3, 1, 4]][:, :, [0, 3, 1, 4]]
    out["fourd"] = torch.sqrt(torch.abs(torch.linalg.det(cov4)) + 1e-16)
    cov6 = cov[:, [0, 3, 1, 4, 2, 5]][:, :, [0, 3, 1, 4, 2, 5]]
    out["sixd"] = torch.sqrt(torch.abs(torch.linalg.det(cov6)) + 1e-16)
    return out


def norm_emits(parts: torch.Tensor) -> dict[str, torch.Tensor]:
    """Normalized emittances (m-rad): geo_2d/mc2, geo_4d/mc2^2 (= ParticleGroup.norm_emit_*)."""
    g = compute_emittance_torch(parts)
    return {
        "norm_emit_x": g["x_xp"] / MC2,
        "norm_emit_y": g["y_yp"] / MC2,
        "norm_emit_4d": g["fourd"] / (MC2 ** 2),
    }


def _energy(parts: torch.Tensor) -> torch.Tensor:  # (B,N) eV
    px, py, pz = parts[..., 3], parts[..., 4], parts[..., 5]
    return torch.sqrt(px * px + py * py + pz * pz + MC2 * MC2)


def per_bunch(parts: torch.Tensor) -> dict[str, torch.Tensor]:
    """Per-bunch scalar observables (all (B,))."""
    ne = norm_emits(parts)
    e = _energy(parts)
    return {
        **ne,
        "sigma_x": parts[..., 0].std(dim=1, unbiased=False),
        "sigma_y": parts[..., 1].std(dim=1, unbiased=False),
        "sigma_z": parts[..., 2].std(dim=1, unbiased=False),
        "centroid_x": parts[..., 0].mean(dim=1),
        "centroid_y": parts[..., 1].mean(dim=1),
        "centroid_z": parts[..., 2].mean(dim=1),
        "mean_energy": e.mean(dim=1),
        "energy_spread": e.std(dim=1, unbiased=False),
    }


def _slopes(parts: torch.Tensor):  # xp, yp = px/pz, py/pz centroids (B,)
    pz = parts[..., 5].clamp_min(1.0)
    return (parts[..., 3] / pz).mean(dim=1), (parts[..., 4] / pz).mean(dim=1)


def inter_bunch(drive: torch.Tensor, witness: torch.Tensor) -> dict[str, torch.Tensor]:
    """Drive<->witness relative observables (the campaign 'quality' quantities)."""
    cz_d, cz_w = drive[..., 2].mean(1), witness[..., 2].mean(1)
    cx_d, cx_w = drive[..., 0].mean(1), witness[..., 0].mean(1)
    cy_d, cy_w = drive[..., 1].mean(1), witness[..., 1].mean(1)
    e_d, e_w = _energy(drive).mean(1), _energy(witness).mean(1)
    xpd, ypd = _slopes(drive)
    xpw, ypw = _slopes(witness)
    return {
        # stored z = -(t-<t>)*c, so (drive_z - witness_z) matches the campaign's signed
        # bunchSpacing (witness_zCentroid - drive_zCentroid in arrival-time*c): + = witness trails.
        "bunch_spacing": cz_d - cz_w,
        "energy_difference": e_d - e_w,
        "transverse_offset": torch.sqrt((cx_d - cx_w) ** 2 + (cy_d - cy_w) ** 2 + 1e-30),
        "angular_misalignment": torch.sqrt((xpd - xpw) ** 2 + (ypd - ypw) ** 2 + 1e-30),
    }
