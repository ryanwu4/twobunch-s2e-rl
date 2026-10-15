"""XTCAV screen features in particle space (training) and their ground-truth cloud twins.

Screen mode emulates what the deployed XTCAV analysis reads off the DTOTR image, but computed
directly from the tracked particles (rendering the 300x160 KDE image inside BPTT is far too
expensive): x is the streak axis, y the energy (B5D36 dispersion) axis.

- two images, TCAV at +V and -V; S = streak dx/dz (fixed calibration constant)
- off-shot row correction        (`row_correct=True`) subtract, per screen row (energy pixel), the
                                charge-weighted mean TCAV-off x — what the deployed per-step off-shot
                                row-distortion correction does. Removes the residual-dispersion
                                x(delta) shear, which otherwise reads as ~30 um of extra drive length.
                                The TCAV-off image is (x+ + x-)/2 (exact to first order).
- bunch membership              `split="energy"`: soft gate on the energy axis y (the 2026-07-23
                                sim2real reference); `split="zlobe"`: soft gate on each image's
                                streak axis. One pixel wide. `threshold="valley"`: at the minimum of
                                the smoothed pixel projection between the most prominent peak and
                                the nearest peak on the charge-bulk side (no truth used; see
                                `_valley` for why not the deployed "two most prominent");
                                `threshold="midpoint"`: halfway between the two TRUE bunch centroids
                                (v1; misassigns ~12 % on the energy axis because the drive's energy
                                spread is ~5x the witness's).
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

import numpy as np
import torch
from scipy import ndimage, signal

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


@torch.no_grad()
def _valley(v, q, smooth_px: float = 2.0, min_rel_prominence: float = 0.05):
    """(B,) split threshold on v (B, N) from the charge-weighted pixel projection (smoothed by
    `smooth_px`), and (B,) ok. Truth-free: the most prominent peak is taken as the witness (narrow,
    tall), the drive side is where the bulk of the charge lies, and the cut is the projection
    minimum between the witness peak and the NEAREST drive-side peak of at least
    `min_rel_prominence` of the top prominence. (The deployed auto rule — minimum between the two
    MOST prominent peaks — cuts inside a lumpy drive flat-top, which the flow surrogate produces;
    the nearest-peak rule is what an operator picking idx_drive_split by eye does.)"""
    v_np, q_np = v.detach().double().cpu().numpy(), q.detach().double().cpu().numpy()
    thr, ok = np.zeros(len(v_np)), np.zeros(len(v_np), bool)
    for b, (vb, qb) in enumerate(zip(v_np, q_np)):
        thr[b] = np.average(vb, weights=qb)
        edges = np.arange(vb.min() - 5 * PIXEL_M, vb.max() + 6 * PIXEL_M, PIXEL_M)
        centres = 0.5 * (edges[1:] + edges[:-1])
        proj = ndimage.gaussian_filter1d(np.histogram(vb, edges, weights=qb)[0], smooth_px)
        pks, _ = signal.find_peaks(proj)
        if len(pks) < 2:
            continue
        prom = signal.peak_prominences(proj, pks)[0]
        top = pks[np.argmax(prom)]
        side = np.sign(thr[b] - centres[top])                 # towards the bulk of the charge
        cand = pks[(np.sign(pks - top) == side) & (prom >= min_rel_prominence * prom.max())]
        if len(cand) == 0:
            continue
        near = cand[np.argmin(np.abs(cand - top))]
        lo, hi = sorted((top, near))
        thr[b], ok[b] = centres[lo + int(np.argmin(proj[lo:hi + 1]))], True
    return torch.as_tensor(thr, dtype=v.dtype, device=v.device), torch.as_tensor(ok, device=v.device)


def _row_mean(x, y, q):
    """Per-particle charge-weighted mean of x over its screen row (pixel of y); rows detached."""
    with torch.no_grad():
        rows = ((y - y.min(-1, keepdim=True).values) / PIXEL_M).floor().long()
        n_rows = int(rows.max()) + 1
    num = torch.zeros(x.shape[0], n_rows, dtype=x.dtype, device=x.device).scatter_add(-1, rows, q * x)
    den = torch.zeros_like(num).scatter_add(-1, rows, q)
    return (num / den.clamp_min(1e-30)).gather(-1, rows)


def screen_features(xp_d, xm_d, y_d, xp_w, xm_w, y_w, q_d, q_w, *, streak: float, eta_y: float,
                    ref_energy_eV: float, resolution_m: float = 0.0,
                    split: str = "energy", threshold: str = "midpoint",
                    row_correct: bool = False, orientation: str = "truth", roi=None,
                    deployed_floor_m: float = 60e-6, edge_band_m: float = 0.3e-3,
                    edge_frac_max: float | None = None) -> dict[str, torch.Tensor]:
    """Two-image (+V / -V) XTCAV measurement, as done on the machine.

    xp_*, xm_*: (B, N) DTOTR streak-axis positions with the TCAV at +V and -V; y_*: (B, N) energy
    axis; q_*: (B,) bunch charge. The half-difference of the two images' centroids cancels each
    bunch's TCAV-off x offset; averaging their variances cancels the x-z cross term, leaving
    S^2 sigma_z^2 + sigma_x,off^2 — the transverse spot is the instrument resolution and stays in
    (blur included, as deployed). `resolution_m` adds any further blur in quadrature (default 0).
    `threshold` / `row_correct`: see the module docstring (defaults = v1 behaviour, kept so runs
    trained with it reproduce). `streak` may be a (B,) tensor (per-env optics / voltage).
    v4 options (defaults = v1-v3 behaviour):
      orientation "charge": the witness is the LOWER-charge side of the cut (truth-free; the deployed
                  analysis knows drive = majority charge) instead of the true witness centroid.
      roi {x: half-width, y: half-width} [m]: camera window centred on each image's charge centroid
                  (detached, as the auto-crop): particles outside carry no charge in that image.
      screen_valid_deployed: what the online analysis can check -- two peaks found and the measured
                  |spacing| above `deployed_floor_m` (round 2: real invalid region ~50-80 um).
      edge_frac_max (with roi; None = off): the crop-edge check -- also deployed-invalid when more than
                  this fraction of an image's in-window charge lies within `edge_band_m` of the window's
                  x edge (a band cut by the auto-crop). With roi, the TRUTH validity also requires the
                  charge clipped out of either image to stay below MISASSIGN_MAX.
    Returns (B,) features + screen_valid / screen_valid_deployed / screen_misassign."""
    xp = torch.cat([xp_d, xp_w], -1)
    xm = torch.cat([xm_d, xm_w], -1)
    y = torch.cat([y_d, y_w], -1)
    nd, nw = xp_d.shape[-1], xp_w.shape[-1]
    q = torch.cat([(q_d / nd).unsqueeze(-1).expand(-1, nd), (q_w / nw).unsqueeze(-1).expand(-1, nw)], -1)
    truth_w = torch.cat([torch.zeros_like(xp_d), torch.ones_like(xp_w)], -1)
    qp = qm = q
    if roi is not None:
        with torch.no_grad():
            hx, hy = float(roi["x"]), float(roi["y"])
            cy = _wmean(y, q).unsqueeze(-1)
            inside = lambda x: ((x - _wmean(x, q).unsqueeze(-1)).abs() < hx) & ((y - cy).abs() < hy)
            mp, mm = inside(xp).to(q.dtype), inside(xm).to(q.dtype)
            edge = lambda x, msk: (q * msk * ((x - _wmean(x, q).unsqueeze(-1)).abs() > hx - edge_band_m)).sum(-1) \
                / (q * msk).sum(-1).clamp_min(1e-30)
            edge_frac = torch.maximum(edge(xp, mp), edge(xm, mm))
            clipped = 1 - torch.minimum((q * mp).sum(-1), (q * mm).sum(-1)) / q.sum(-1).clamp_min(1e-30)
        qp, qm = q * mp, q * mm
    if orientation not in ("truth", "charge"):
        raise ValueError(f"orientation must be 'truth' or 'charge', got {orientation!r}")
    if row_correct:
        r = _row_mean(0.5 * (xp + xm), y, q)
        xp, xm = xp - r, xm - r
    if threshold not in ("midpoint", "valley"):
        raise ValueError(f"threshold must be 'midpoint' or 'valley', got {threshold!r}")

    # bunch membership per image. "energy": gate on the energy axis y (same for both images);
    # "zlobe": gate on each image's streak axis separately (the bunches are separated in time).
    peaks_ok = torch.ones(xp.shape[0], dtype=torch.bool, device=xp.device)
    if split == "energy":
        axes = [y]
    elif split == "zlobe":
        axes = [xp, xm]
    else:
        raise ValueError(f"split must be 'energy' or 'zlobe', got {split!r}")
    gates = []
    for v, qv in zip(axes, [qp] if split == "energy" else [qp, qm]):
        if threshold == "midpoint":
            gates.append(_gate(v, v[:, :nd], v[:, nd:], PIXEL_M))
        else:
            thr, ok = _valley(v, qv)
            peaks_ok &= ok
            with torch.no_grad():   # orientation only
                if orientation == "truth":
                    sign = torch.sign(v[:, nd:].mean(-1) - thr)
                else:               # witness = the lower-charge side of the cut
                    above = (qv * (v > thr.unsqueeze(-1))).sum(-1)
                    sign = torch.where(above < 0.5 * qv.sum(-1), 1.0, -1.0).to(v.dtype)
            gates.append(torch.sigmoid(sign.unsqueeze(-1) * (v - thr.unsqueeze(-1)) / PIXEL_M))
    wp, wm = (gates[0], gates[0]) if split == "energy" else (gates[0], gates[1])
    wdp, wwp, wdm, wwm = qp * (1 - wp), qp * wp, qm * (1 - wm), qm * wm
    wd, ww = 0.5 * (wdp + wdm), 0.5 * (wwp + wwm)          # for the energy-axis quantities

    # centroids: half-difference of the +V and -V images; widths: mean of the two variances
    zc = lambda w_p, w_m: (_wmean(xp, w_p) - _wmean(xm, w_m)) / (2 * streak)
    spacing = zc(wdp, wdm) - zc(wwp, wwm)
    var_d = 0.5 * (_wvar(xp, wdp) + _wvar(xm, wdm)) / streak ** 2
    sig_z = (var_d + resolution_m ** 2).clamp_min(1e-30).sqrt()
    de = (_wmean(y, wd) - _wmean(y, ww)) / eta_y * ref_energy_eV
    espread = _wvar(y, wd).clamp_min(1e-30).sqrt() / abs(eta_y) * ref_energy_eV
    frac_d = wd.sum(-1) / (0.5 * (qp + qm)).sum(-1).clamp_min(1e-30)

    with torch.no_grad():
        misassign = 0.5 * ((q * (wp - truth_w).abs()).sum(-1) + (q * (wm - truth_w).abs()).sum(-1)) / q.sum(-1)
        valid = (spacing.abs() > SEPARATION_FLOOR_M) & (misassign < MISASSIGN_MAX) & peaks_ok
        valid_dep = (spacing.abs() > deployed_floor_m) & peaks_ok
        if roi is not None and edge_frac_max is not None:
            valid &= clipped < MISASSIGN_MAX
            valid_dep &= edge_frac < edge_frac_max
    out = {"bunch_spacing": spacing, "drive_sigma_z": sig_z, "energy_difference": de,
           "drive_energy_spread": espread, "drive_charge_fraction": frac_d,
           "screen_valid": valid.float(), "screen_valid_deployed": valid_dep.float(), "screen_misassign": misassign}
    if roi is not None:
        out["screen_edge_frac"], out["screen_clipped"] = edge_frac, clipped
    return out


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
