"""Online XTCAV features from DTOTR2 images: the same measurement the RL env emulates in particle space
(xtcav_segment/features.py, v4 options), applied to a +V / -V image pair.

Inputs are preprocessed images (background-subtracted, cropped, off-shot row-corrected -- the deployed
xtcav.py chain does that), rows = energy axis y, columns = streak axis x, on a common pixel grid.
Same rules as training (decision D2, 2026-10-14):
  - energy gate on the y projection of the +V image: minimum of the 2-px-smoothed projection between
    the most prominent peak and the nearest peak (>= 5% of the top prominence) on the charge-bulk side
    (xtcav_segment.features._valley); witness = the lower-charge side of the cut
  - spacing = [(<x+>_d - <x->_d) - (<x+>_w - <x->_w)] / 2S with the SIGNED streak S (screen um per um
    of z; decision D3: signed, golden calibration, fixed per shift)
  - drive sigma_z = sqrt(mean(var x+_d, var x-_d)) / |S| (instrument blur included, as deployed)
  - drive charge fraction = drive-band intensity / total, averaged over the two images
  - valid = two peaks found and |spacing| > floor (the deployed rule; 60 um default)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage, signal

PIXEL_M = 30.5e-6


def valley_threshold(proj: np.ndarray, smooth_px: float = 2.0, min_rel_prominence: float = 0.05):
    """(index of the cut between bins, ok) on a 1-D intensity projection (the training rule)."""
    p = ndimage.gaussian_filter1d(np.asarray(proj, float), smooth_px)
    pks, _ = signal.find_peaks(p)
    if len(pks) < 2:
        return None, False
    prom = signal.peak_prominences(p, pks)[0]
    top = pks[np.argmax(prom)]
    centre = np.average(np.arange(len(p)), weights=np.clip(p, 0, None) + 1e-30)
    side = np.sign(centre - top)
    cand = pks[(np.sign(pks - top) == side) & (prom >= min_rel_prominence * prom.max())]
    if len(cand) == 0:
        return None, False
    near = cand[np.argmin(np.abs(cand - top))]
    lo, hi = sorted((top, near))
    return lo + int(np.argmin(p[lo:hi + 1])), True


def band_stats(img: np.ndarray, x_m: np.ndarray, cut_row: int | None = None) -> dict | None:
    """Per-image band statistics: the energy cut (valley rule, or a given cut_row), witness = the
    lower-charge band; drive / witness x centroid, drive x variance, drive charge fraction."""
    im = np.clip(np.asarray(img, float), 0, None)
    if cut_row is None:
        cut_row, ok = valley_threshold(im.sum(axis=1))
        if not ok:
            return None
    below = np.arange(im.shape[0]) <= cut_row
    wit = below if im[below].sum() < im[~below].sum() else ~below
    pd_, pw = im[~wit].sum(axis=0), im[wit].sum(axis=0)
    qd, qw = pd_.sum(), pw.sum()
    if qd <= 0 or qw <= 0:
        return None
    md = np.average(x_m, weights=pd_)
    return {"cut_row": int(cut_row), "x_d": float(md), "x_w": float(np.average(x_m, weights=pw)),
            "var_d": float(np.average((x_m - md) ** 2, weights=pd_)), "frac_d": float(qd / (qd + qw))}


def step_features(plus: list[dict], minus: list[dict], streak: float, floor_m: float = 60e-6,
                  agg=np.median) -> dict:
    """Step-level features from the band stats of the step's +V images and -V images (on the machine
    they are ~11 s apart, not the same beam): robust (median) aggregation per sign, then the
    half-difference of the per-image band offsets. Valid if both signs have images and |spacing| > floor."""
    plus, minus = [p for p in plus if p], [m for m in minus if m]
    if not plus or not minus:
        return {"valid": 0.0, "n_plus": len(plus), "n_minus": len(minus)}
    a = lambda L, k: float(agg([d[k] for d in L]))
    # aggregate the per-image band offset x_d - x_w (crop- and image-position-independent; equal to
    # differencing the aggregated centroids for a mean, robust to band-swap outliers for a median)
    off = lambda L: float(agg([d["x_d"] - d["x_w"] for d in L]))
    spacing = (off(plus) - off(minus)) / (2 * streak)
    sig_z = np.sqrt(0.5 * (a(plus, "var_d") + a(minus, "var_d"))) / abs(streak)
    frac = 0.5 * (a(plus, "frac_d") + a(minus, "frac_d"))
    return {"bunch_spacing": spacing, "drive_sigma_z": float(sig_z), "drive_charge_fraction": frac,
            "valid": float(abs(spacing) > floor_m), "n_plus": len(plus), "n_minus": len(minus)}


def image_features(img_plus: np.ndarray, img_minus: np.ndarray, x_m: np.ndarray, y_m: np.ndarray,
                   streak: float, floor_m: float = 60e-6) -> dict:
    """Features of one +V / -V image pair. img_*: (ny, nx) nonnegative intensities; x_m, y_m: pixel
    centre coordinates [m]; streak: signed dx/dz (screen per bunch length)."""
    ip, im = np.clip(np.asarray(img_plus, float), 0, None), np.clip(np.asarray(img_minus, float), 0, None)
    k, ok = valley_threshold(ip.sum(axis=1))
    out = {"valid": 0.0, "peaks_ok": bool(ok)}
    if not ok:
        return out
    rows = np.arange(ip.shape[0])
    below = rows <= k                               # rows on the low-y side of the cut (incl. the cut)
    # witness = the lower-charge band (drive = majority charge)
    w_low = ip[below].sum() < ip[~below].sum()
    wit = below if w_low else ~below

    def band(img, sel):
        prof = img[sel].sum(axis=0)
        q = prof.sum()
        if q <= 0:
            return np.nan, np.nan, 0.0
        m = np.average(x_m, weights=prof)
        v = np.average((x_m - m) ** 2, weights=prof)
        return m, v, q

    mpd, vpd, qpd = band(ip, ~wit)
    mpw, _, qpw = band(ip, wit)
    mmd, vmd, qmd = band(im, ~wit)
    mmw, _, qmw = band(im, wit)
    spacing = ((mpd - mmd) - (mpw - mmw)) / (2 * streak)
    sig_z = np.sqrt(0.5 * (vpd + vmd)) / abs(streak)
    frac = 0.5 * (qpd / max(qpd + qpw, 1e-30) + qmd / max(qmd + qmw, 1e-30))
    out.update(bunch_spacing=float(spacing), drive_sigma_z=float(sig_z), drive_charge_fraction=float(frac),
               cut_row=int(k), valid=float(abs(spacing) > floor_m))
    return out


def render(x: np.ndarray, y: np.ndarray, q: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray) -> np.ndarray:
    """Charge-binned screen image (rows = y) on the given pixel edges (for tests / sim replay)."""
    img, _, _ = np.histogram2d(y, x, [y_edges, x_edges], weights=q)
    return img
