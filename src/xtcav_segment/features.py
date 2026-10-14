"""XTCAV screen features in particle space (training) and their ground-truth cloud twins.

Screen mode emulates what the deployed XTCAV analysis reads off the DTOTR image, but computed
directly from the tracked particles (rendering the 300x160 KDE image inside BPTT is far too
expensive): x is the streak axis, y the energy (B5D36 dispersion) axis.

- two images, TCAV at +V and -V; S = streak dx/dz (fixed calibration constant)
- bunch membership              `split="energy"`: soft gate on the energy axis y (the 2026-07-23
                                sim2real reference); `split="zlobe"`: soft gate on each image's
                                streak axis (the GPSR-plan "robust real observable"). One pixel wide,
                                threshold midway between the two bunch centroids.
- spacing                       [(<x+>_d - <x->_d) - (<x+>_w - <x->_w)] / 2S  (sign as properties)
- drive sigma_z                 sqrt(mean(var x+, var x-)) / |S|   (transverse blur included)
- energy difference / spread    from y through the dispersion eta_y (energy axis)
- drive charge fraction         gated drive charge / total charge
- screen_valid                  1 if separation > floor AND the gate misassigns < 10 % of charge

The cloud twins compute the same names exactly from the XTCAVF bunches (ground truth): what the
screen is trying to measure. Same names in both modes, so objectives and obs don't change when the
observable source is toggled.
"""
from __future__ import annotations

import torch

FEATURE_KEYS = ("bunch_spacing", "drive_sigma_z", "energy_difference", "drive_energy_spread",
                "drive_charge_fraction")
PIXEL_M = 30.5e-6
SEPARATION_FLOOR_M = 50e-6          # the deployed analysis distrusts separations below ~50 um
MISASSIGN_MAX = 0.10


def _wmean(v, w):
    return (v * w).sum(-1) / w.sum(-1).clamp_min(1e-30)


def _wvar(v, w):
    m = _wmean(v, w).unsqueeze(-1)
    return ((v - m) ** 2 * w).sum(-1) / w.sum(-1).clamp_min(1e-30)


def _gate(v, v_d, v_w, width):
    """Soft witness membership: sigmoid across the midpoint between the two bunch centroids of
    coordinate v, oriented witness-side. Threshold/orientation detached (the deployed analysis
    finds them from the histogram valley); one pixel wide."""
    with torch.no_grad():
        md, mw = v_d.mean(-1), v_w.mean(-1)
        thr, sign = 0.5 * (md + mw), torch.sign(mw - md)
    return torch.sigmoid(sign.unsqueeze(-1) * (v - thr.unsqueeze(-1)) / width)


def screen_features(xp_d, xm_d, y_d, xp_w, xm_w, y_w, q_d, q_w, *, streak: float, eta_y: float,
                    ref_energy_eV: float, resolution_m: float = 0.0,
                    split: str = "energy") -> dict[str, torch.Tensor]:
    """Two-image (+V / -V) XTCAV measurement, as done on the machine.

    xp_*, xm_*: (B, N) DTOTR streak-axis positions with the TCAV at +V and -V; y_*: (B, N) energy
    axis; q_*: (B,) bunch charge. The half-difference of the two images' centroids cancels each
    bunch's TCAV-off x offset; averaging their variances cancels the x-z cross term, leaving
    S^2 sigma_z^2 + sigma_x,off^2 — the transverse spot is the instrument resolution and stays in
    (blur included, as deployed). `resolution_m` adds any further blur in quadrature (default 0).
    Returns (B,) features + screen_valid / screen_misassign."""
    xp = torch.cat([xp_d, xp_w], -1)
    xm = torch.cat([xm_d, xm_w], -1)
    y = torch.cat([y_d, y_w], -1)
    nd, nw = xp_d.shape[-1], xp_w.shape[-1]
    q = torch.cat([(q_d / nd).unsqueeze(-1).expand(-1, nd), (q_w / nw).unsqueeze(-1).expand(-1, nw)], -1)
    truth_w = torch.cat([torch.zeros_like(xp_d), torch.ones_like(xp_w)], -1)

    # bunch membership per image. "energy": gate on the energy axis y (same for both images);
    # "zlobe": gate on each image's streak axis separately (the bunches are separated in time).
    if split == "energy":
        wp = wm = _gate(y, y_d, y_w, PIXEL_M)
    elif split == "zlobe":
        wp, wm = _gate(xp, xp_d, xp_w, PIXEL_M), _gate(xm, xm_d, xm_w, PIXEL_M)
    else:
        raise ValueError(f"split must be 'energy' or 'zlobe', got {split!r}")
    wdp, wwp, wdm, wwm = q * (1 - wp), q * wp, q * (1 - wm), q * wm
    wd, ww = 0.5 * (wdp + wdm), 0.5 * (wwp + wwm)          # for the energy-axis quantities

    # centroids: half-difference of the +V and -V images; widths: mean of the two variances
    zc = lambda w_p, w_m: (_wmean(xp, w_p) - _wmean(xm, w_m)) / (2 * streak)
    spacing = zc(wdp, wdm) - zc(wwp, wwm)
    var_d = 0.5 * (_wvar(xp, wdp) + _wvar(xm, wdm)) / streak ** 2
    sig_z = (var_d + resolution_m ** 2).clamp_min(1e-30).sqrt()
    de = (_wmean(y, wd) - _wmean(y, ww)) / eta_y * ref_energy_eV
    espread = _wvar(y, wd).clamp_min(1e-30).sqrt() / abs(eta_y) * ref_energy_eV
    frac_d = wd.sum(-1) / q.sum(-1)

    with torch.no_grad():
        misassign = 0.5 * ((q * (wp - truth_w).abs()).sum(-1) + (q * (wm - truth_w).abs()).sum(-1)) / q.sum(-1)
        valid = ((spacing.abs() > SEPARATION_FLOOR_M) & (misassign < MISASSIGN_MAX)).float()
    return {"bunch_spacing": spacing, "drive_sigma_z": sig_z, "energy_difference": de,
            "drive_energy_spread": espread, "drive_charge_fraction": frac_d,
            "screen_valid": valid, "screen_misassign": misassign}


def cloud_features(obs: dict[str, torch.Tensor], q_d, q_w) -> dict[str, torch.Tensor]:
    """Ground-truth twins from TwoBunchFlow.observables() at XTCAVF (+ bunch charges)."""
    return {"bunch_spacing": obs["bunch_spacing"], "drive_sigma_z": obs["drive_sigma_z"],
            "energy_difference": obs["energy_difference"],
            "drive_energy_spread": obs["drive_energy_spread"],
            "drive_charge_fraction": q_d / (q_d + q_w)}


def calibrate_streak(x: torch.Tensor, z: torch.Tensor) -> float:
    """Streak constant dx/dz from a reference tracking (least squares over all particles), the
    sim analogue of the deployed phase-scan calibration. Fixed for a run."""
    xc, zc = x - x.mean(-1, keepdim=True), z - z.mean(-1, keepdim=True)
    return float((xc * zc).sum() / (zc * zc).sum())
