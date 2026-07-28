"""Two-bunch metrics on longitudinal current profiles I(z).

Vendored from Yiheng Yang's deployed E300 XTCAV analysis
(XTCAV_analysis/yiheng/xtcav.py, itself porting C. Emma's MATLAB live script),
keeping only the dependency-free profile-metric functions so the same
algorithms the control room runs on real DTOTR2 images can be applied to
simulated screens. Function signatures, conventions, and defaults are kept
identical to the deployed code:

  - find_bunch_separation : two most prominent peaks, relative-prominence cut
    (default 5%), never raises (ok=False on failure).
  - fit_bigaussian        : auto-windowed two-Gaussian fit seeded by the peaks;
    falls back to single Gaussian; R^2 on the fitted window.
  - split_drive_witness_charge : integrate I dt on either side of the current
    minimum between the peaks; drive = smaller z ("z increases tail-to-head").

Deployed conventions worth remembering when comparing to simulation truth:
drive = larger-amplitude peak; z-axis is x/cal_scaled with the smoothed
projection peak shifted to z=0; current normalized so integral of I dt = Q(toroid).
"""

import numpy as np
from scipy import constants, integrate, ndimage, signal

C_LIGHT = constants.physical_constants["speed of light in vacuum"][0]
F_TCAV = 11.424e9                                # Hz, X-band
UM_PER_DEG_RF = C_LIGHT / F_TCAV / 360.0 * 1e6   # ~72.90 um/deg


def _weighted_median(values, weights):
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0:
        return float("nan")
    return float(np.interp(0.5 * cum[-1], cum, v))


def energy_gated_image_features(img, x_mm, y_mm, cal_scaled, total_charge_nC,
                                smooth_sigma_px=2.0):
    """Energy-projection gating on a screen image array (sim or real).

    Splits the image along the energy (y) axis at the minimum of the smoothed
    y-projection between its two most prominent peaks (auto version of the
    deployed hand-picked idx_drive_split), then computes per-band charges and
    per-band z statistics from the x-projections. z = x/cal_scaled; both
    charge-weighted mean and weighted-median centroids are returned (the sim
    audit showed mean-vs-median is a ~6% systematic on bunch separation).

    Parameters
    ----------
    img : 2-D ndarray (rows = y, cols = x), any nonnegative intensity units
    x_mm, y_mm : 1-D center coordinates of columns/rows [mm]
    cal_scaled : float — streak factor [um on screen per um of bunch length]
    total_charge_nC : float — charge to distribute over the image intensity
    smooth_sigma_px : float — Gaussian smoothing of the y-projection

    Returns
    -------
    dict: ok, split_row, y_split_mm, band_low_y/band_high_y (Q_nC,
    z_mean_um, z_median_um), separation_mean_um, separation_median_um.
    """
    img = np.asarray(img, float)
    yproj = ndimage.gaussian_filter(img.sum(axis=1), sigma=smooth_sigma_px)
    pks, _ = signal.find_peaks(yproj)
    if len(pks) < 2:
        return {"ok": False}
    prom = signal.peak_prominences(yproj, pks)[0]
    top2 = np.sort(pks[np.argsort(prom)[::-1][:2]])
    between = np.arange(top2[0], top2[1] + 1)
    split_row = int(between[np.argmin(yproj[between])])

    z_um = 1e3 * np.asarray(x_mm, float) / cal_scaled
    total = img.sum()
    if total <= 0:
        return {"ok": False}
    out = {"ok": True, "split_row": split_row,
           "y_split_mm": float(y_mm[split_row])}
    for tag, sl in [("band_low_y", slice(0, split_row)),
                    ("band_high_y", slice(split_row, None))]:
        sub = img[sl, :].sum(axis=0)
        s = sub.sum()
        out[tag] = {
            "Q_nC": float(total_charge_nC * s / total),
            "z_mean_um": float(np.average(z_um, weights=sub)) if s > 0 else float("nan"),
            "z_median_um": _weighted_median(z_um, sub) if s > 0 else float("nan"),
        }
    out["separation_mean_um"] = float(abs(out["band_high_y"]["z_mean_um"]
                                          - out["band_low_y"]["z_mean_um"]))
    out["separation_median_um"] = float(abs(out["band_high_y"]["z_median_um"]
                                            - out["band_low_y"]["z_median_um"]))
    return out


def find_bunch_separation(z_um, current_kA, prominence_ratio=0.05,
                          min_prominence=None):
    """Find drive/witness peak separation in a longitudinal current profile.

    See module docstring; verbatim port of the deployed function.
    """
    z = np.asarray(z_um, float)
    I = np.asarray(current_kA, float)
    if z.shape != I.shape or z.ndim != 1:
        raise ValueError("z_um and current_kA must be 1-D arrays of the same length")

    fail = {
        "separation_um": float("nan"),
        "peak_positions_um": np.array([]),
        "peak_currents_kA": np.array([]),
        "peak_prominences": np.array([]),
        "n_peaks_found": 0,
        "ok": False,
    }

    if not np.any(np.isfinite(I)) or np.nanmax(I) <= 0:
        return fail

    locs, _ = signal.find_peaks(I, prominence=min_prominence)
    if len(locs) < 1:
        return fail

    prom = signal.peak_prominences(I, locs)[0]
    if len(locs) == 1:
        out = dict(fail)
        out["peak_positions_um"] = np.array([z[locs[0]]])
        out["peak_currents_kA"] = np.array([I[locs[0]]])
        out["peak_prominences"] = np.array([prom[0]])
        out["n_peaks_found"] = 1
        return out

    order = np.argsort(prom)[::-1]
    top2_idx = order[:2]
    top2_locs = locs[top2_idx]
    top2_prom = prom[top2_idx]

    p_big, p_small = float(top2_prom[0]), float(top2_prom[1])
    if p_big <= 0 or p_small / p_big < prominence_ratio:
        out = dict(fail)
        out["peak_positions_um"] = np.array([z[top2_locs[0]]])
        out["peak_currents_kA"] = np.array([I[top2_locs[0]]])
        out["peak_prominences"] = np.array([p_big])
        out["n_peaks_found"] = 1
        return out

    z_pair = z[top2_locs]
    sort_idx = np.argsort(z_pair)
    z_pair = z_pair[sort_idx]
    I_pair = I[top2_locs][sort_idx]
    p_pair = top2_prom[sort_idx]

    return {
        "separation_um": float(abs(z_pair[1] - z_pair[0])),
        "peak_positions_um": z_pair,
        "peak_currents_kA": I_pair,
        "peak_prominences": p_pair,
        "n_peaks_found": 2,
        "ok": True,
    }


def _bigaussian(z, A1, mu1, sigma1, A2, mu2, sigma2):
    return (A1 * np.exp(-0.5 * ((z - mu1) / sigma1) ** 2)
            + A2 * np.exp(-0.5 * ((z - mu2) / sigma2) ** 2))


def _fwhm_sigma_guess(z, I, peak_idx):
    half = 0.5 * I[peak_idx]
    i = peak_idx
    while i > 0 and I[i] > half:
        i -= 1
    z_left = z[i]
    j = peak_idx
    while j < len(I) - 1 and I[j] > half:
        j += 1
    z_right = z[j]
    fwhm = abs(z_right - z_left)
    sigma = fwhm / 2.355 if fwhm > 0 else (z[-1] - z[0]) / 20.0
    return max(sigma, 1.0)


def fit_bigaussian(z_um, current_kA, peaks=None, z_window=None,
                   padding_factor=4.0):
    """Fit a sum of two Gaussians to a current profile. Deployed-code port."""
    from scipy import optimize

    z = np.asarray(z_um, float)
    I = np.asarray(current_kA, float)
    if z.shape != I.shape or z.ndim != 1:
        raise ValueError("z_um and current_kA must be 1-D arrays of the same length")

    if peaks is None:
        peaks = find_bunch_separation(z, I)

    fail = {
        "A1": float("nan"), "mu1_um": float("nan"), "sigma1_um": float("nan"),
        "A2": float("nan"), "mu2_um": float("nan"), "sigma2_um": float("nan"),
        "separation_um": float("nan"),
        "R_squared": float("nan"),
        "fit_curve": np.array([]),
        "z_fit": np.array([]),
        "z_window_used": (float("nan"), float("nan")),
        "two_bunch": False,
        "ok": False,
    }

    if z_window is not None:
        z_lo, z_hi = float(z_window[0]), float(z_window[1])
    elif peaks["ok"]:
        mus = peaks["peak_positions_um"]
        sigmas = []
        for mu in mus:
            idx = int(np.argmin(np.abs(z - mu)))
            sigmas.append(_fwhm_sigma_guess(z, I, idx))
        sg = max(sigmas)
        z_lo = float(np.min(mus) - padding_factor * sg)
        z_hi = float(np.max(mus) + padding_factor * sg)
    elif peaks["n_peaks_found"] == 1:
        mu = float(peaks["peak_positions_um"][0])
        idx = int(np.argmin(np.abs(z - mu)))
        sg = _fwhm_sigma_guess(z, I, idx)
        z_lo = mu - padding_factor * sg
        z_hi = mu + padding_factor * sg
    else:
        z_lo = float(z[0])
        z_hi = float(z[-1])

    z_lo = max(z_lo, float(z[0]))
    z_hi = min(z_hi, float(z[-1]))
    if z_hi <= z_lo:
        return fail

    mask = (z >= z_lo) & (z <= z_hi)
    zw = z[mask]
    Iw = I[mask]
    if zw.size < 6:
        return fail

    if peaks["ok"] and peaks["n_peaks_found"] == 2:
        mus = peaks["peak_positions_um"]
        Is_pk = peaks["peak_currents_kA"]
        sigmas = []
        for mu in mus:
            idx = int(np.argmin(np.abs(z - mu)))
            sigmas.append(_fwhm_sigma_guess(z, I, idx))

        p0 = [float(Is_pk[0]), float(mus[0]), float(sigmas[0]),
              float(Is_pk[1]), float(mus[1]), float(sigmas[1])]

        try:
            popt, _ = optimize.curve_fit(_bigaussian, zw, Iw, p0=p0, maxfev=5000)
        except Exception:
            return fail

        A1, mu1, s1, A2, mu2, s2 = popt
        s1, s2 = abs(float(s1)), abs(float(s2))

        amp_ratio = abs(A2) / max(abs(A1), 1e-30)
        two_bunch = amp_ratio > 0.02 and np.isfinite(A2) and np.isfinite(mu2)

        if two_bunch and mu1 > mu2:
            A1, mu1, s1, A2, mu2, s2 = A2, mu2, s2, A1, mu1, s1

        fit_curve = _bigaussian(zw, A1, mu1, s1, A2, mu2, s2)
        ss_res = float(np.sum((Iw - fit_curve) ** 2))
        ss_tot = float(np.sum((Iw - np.mean(Iw)) ** 2))
        R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        return {
            "A1": float(A1), "mu1_um": float(mu1), "sigma1_um": float(s1),
            "A2": float(A2) if two_bunch else float("nan"),
            "mu2_um": float(mu2) if two_bunch else float("nan"),
            "sigma2_um": float(s2) if two_bunch else float("nan"),
            "separation_um": float(abs(mu2 - mu1)) if two_bunch else float("nan"),
            "R_squared": R2,
            "fit_curve": fit_curve,
            "z_fit": zw,
            "z_window_used": (z_lo, z_hi),
            "two_bunch": bool(two_bunch),
            "ok": True,
        }

    def _gauss(z, A, mu, sigma):
        return A * np.exp(-0.5 * ((z - mu) / sigma) ** 2)

    if peaks["n_peaks_found"] >= 1:
        mu0 = float(peaks["peak_positions_um"][0])
        A0 = float(peaks["peak_currents_kA"][0])
        idx = int(np.argmin(np.abs(z - mu0)))
        s0 = _fwhm_sigma_guess(z, I, idx)
    else:
        idx = int(np.argmax(Iw))
        mu0 = float(zw[idx])
        A0 = float(Iw[idx])
        s0 = (z_hi - z_lo) / 6.0

    from scipy import optimize
    try:
        popt, _ = optimize.curve_fit(_gauss, zw, Iw, p0=[A0, mu0, s0], maxfev=3000)
    except Exception:
        return fail

    A1, mu1, s1 = popt
    s1 = abs(float(s1))
    fit_curve = _gauss(zw, A1, mu1, s1)
    ss_res = float(np.sum((Iw - fit_curve) ** 2))
    ss_tot = float(np.sum((Iw - np.mean(Iw)) ** 2))
    R2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "A1": float(A1), "mu1_um": float(mu1), "sigma1_um": float(s1),
        "A2": float("nan"), "mu2_um": float("nan"), "sigma2_um": float("nan"),
        "separation_um": float("nan"),
        "R_squared": R2,
        "fit_curve": fit_curve,
        "z_fit": zw,
        "z_window_used": (z_lo, z_hi),
        "two_bunch": False,
        "ok": True,
    }


def split_drive_witness_charge(z_um, current_kA, z_split_um=None, peaks=None):
    """Integrate I(z) on either side of a split point. Deployed-code port."""
    z = np.asarray(z_um, float)
    I = np.asarray(current_kA, float) * 1e3      # kA -> A
    if z.shape != I.shape or z.ndim != 1:
        raise ValueError("z_um and current_kA must be 1-D arrays of the same length")

    t_s = z * 1e-6 / C_LIGHT

    if z_split_um is not None:
        z_split = float(z_split_um)
        method = "user"
        ok = True
    else:
        if peaks is None:
            peaks = find_bunch_separation(z, np.asarray(current_kA, float))
        if peaks["ok"] and peaks["n_peaks_found"] == 2:
            mus = peaks["peak_positions_um"]
            z_lo, z_hi = float(np.min(mus)), float(np.max(mus))
            mask = (z >= z_lo) & (z <= z_hi)
            if np.any(mask):
                between_z = z[mask]
                between_I = I[mask]
                z_split = float(between_z[int(np.argmin(between_I))])
                method = "minimum"
                ok = True
            else:
                z_split = float("nan")
                method = "fallback"
                ok = False
        else:
            z_split = float("nan")
            method = "fallback"
            ok = False

    if not ok:
        Q_total_C = float(integrate.trapezoid(I, t_s))
        return {
            "Q_drive_nC": Q_total_C * 1e9,
            "Q_witness_nC": 0.0,
            "z_split_um_used": z_split,
            "split_method": method,
            "ok": False,
        }

    drive_mask = z <= z_split
    wit_mask = z > z_split
    Q_drive_C = float(integrate.trapezoid(I[drive_mask], t_s[drive_mask])) \
                if np.any(drive_mask) else 0.0
    Q_wit_C = float(integrate.trapezoid(I[wit_mask], t_s[wit_mask])) \
              if np.any(wit_mask) else 0.0

    return {
        "Q_drive_nC": Q_drive_C * 1e9,
        "Q_witness_nC": Q_wit_C * 1e9,
        "z_split_um_used": z_split,
        "split_method": method,
        "ok": True,
    }
