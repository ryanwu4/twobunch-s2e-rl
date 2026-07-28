"""Side-by-side real vs simulated XTCAV/DTOTR2 screen image at one matched scan point.

Picks a single well-matched (dataset, step, streak-sign) point from the Tier 2 comparison and
renders the two 2-D screen images next to each other on shared axes, with the deployed
energy-gate overlaid on both. This is the image-level companion to the Tier 2 curve figures
(`fig_separation.png`), which compare only the extracted scalar features.

Default point: E300_15673, step 17 (L2_PHASE.MKB = +4.00 deg), real +90 deg vs sim +22 MV.
Step 17 is one of only three steps whose sim beam was kept (`beam_DTOTR_step{01,17,33}_plus.h5`,
pre-committed as BEAM_KEEP_STEPS in xtcav_sim_scan.py), so no Bmad re-track is needed, and its
per-step-median residual happens to be the smallest of the non-degenerate steps (+0.7 um).

READ THAT +0.7 um WITH CARE -- it is not a demonstration of agreement:
  - Its neighbours are -18 um (step 16) and -18 um (step 18); the real medians run
    171.6 -> 132.9 -> 145.3 (non-monotonic) while the sim curve is smooth. Step 17 is a dip in
    the real curve that the sim curve happens to cross. Averaging steps 16-18 gives -16 um.
  - The real energy-gated separation at this step is NOT streak-sign invariant, though |dz|
    between two energy bands must be: plus 132.9 um vs neg 19.7 um. This is systemic, not a step-17
    quirk -- steps 10-19 disagree by 2-7x at separations far from band overlap. Cause is the gate
    split row, set by a 3.7-13%-prominence bump in the y-projection; scanning the split row runs
    the answer over 2.0 -> 174.8 um (and the plus-branch shots' own chosen split rows at this step
    span rows 53-99, median 90, while the plotted shot sits at 72). The real energy-gated number
    is therefore not a reliable observable here at all.
  - Step 17's own jitter-calibration slope is -516 um/deg, the second-smallest |slope| of 33
    steps, against the dataset median 1049.9 that is actually used for the scale. With its own
    slope the real separation would read ~268 um.
  - One selected real shot (the median-separation shot of 44 gated) vs one deterministic sim run.
    The real per-step IQR is 33 um; the 25-shot averaged real image gates to ~155 um.
So this figure is an honest LOOK at one matched point, not evidence of few-um fidelity. The
defensible transfer budget stays the dataset-level one in CONCLUSIONS.md (8-36 um).

Both sides use the SAME 30.5 um pixel grid (the real DTOTR2 camera scale; the sim scan pixelates
to it) and the same feature code (`xtcav_metrics.energy_gated_image_features`).

Axes, and what is NOT shared between the panels:
  - z [um]: each side divides its screen x by its OWN streak calibration (cal_scaled = real
    14.40 vs sim 11.62 um/um, the x1.24 calibration-scale gap of CALIBRATION_INVESTIGATION.md).
    The STREAK SIGN is applied (the deployed pipeline drops it -- jitter_calibration takes
    abs(slope), calibrate_phase_position stores it separately as `slope_sign`), so z = -c*dt and
    larger z is nearer the HEAD. Without that sign the sim panel showed the drive TRAILING the
    witness, which is backwards for FACET-II.
      * sim: sign measured from the beam itself (sign of dx/dz) -- ground truth.
      * real: anchored MODEL-FREE -- the majority-charge z-lobe is the drive (3:1 cathode double
        pulse) and the drive leads, so the axis is oriented to put it at the head. Corroborated by
        sign(dx/dphi) < 0 in 32/33 steps with z = -c*dt (giving z_sign = -sign(dx/dphi) = +1), and
        the LPS chirp sign then matches the sim's (+10.5 vs +9.7 %/mm) as a genuine independent
        check. NOTE an earlier version oriented by chirp-matching: that was circular AND degenerate
        (flipping z and dE together preserves the chirp sign but inverts the lobe topology, so
        chirp matching constrains only the PRODUCT of the two axis signs).
  - dE/E [%]: real rows use the deployed `spec.SPEC.E`, which IS correctly paired with the image
    rows: `s.E == forward(cal*(ny-1-arange(ny)))` is an exact identity, image.Image.y measures
    distance from the TOP row while spec.forward() wants y from the BOTTOM, and xtcav.py:403-405
    states the conversion. Row 0 = top = highest E *in the deployed convention*.
    CAUTION: this is INTERNAL SELF-CONSISTENCY, not a measured dispersion sign. "E at the fitted
    centroid row equals E_bend" is an algebraic identity (fit_dy_from_shot defines dy to make it
    so), so it validates either candidate axis and proves nothing physical; the real DTOTR2
    dispersion sign is nowhere measured, and IS_ROTATED=1 (transposed camera array) blocks any
    "top of the array = physically up" shortcut. Treat which band is higher-energy as a
    convention inherited from the deployed pipeline.
    Sim rows use the dispersion fit from the sim beam (eta_sim = 48.7 mm vs the real NOMINAL
    d_nom = 59.31 mm -- a measurement against a model constant). If the true real dispersion were
    ~48.7 mm the REAL dE/E axis would scale UP by ~1.22; the sim axis is the measured one.
    Energy increases UPWARD in both panels.

Run in the bmad-qpad-dev env (needs the deployed DAQ classes + pmd_beamphysics):
  cd twobunch-s2e-rl && PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_screen_compare.py

Outputs -> XTCAV_analysis/sim2real/results/tier2_comparison/screen_compare/
  fig_screen_compare_<dataset>_step<NN>.png/.pdf
  screen_compare_<dataset>_step<NN>.json   (both sides' gating results + axis provenance)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import integrate, ndimage, signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

XTCAV_ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "DAQ_analysis_class"))
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "XTCAV_analysis_script"))
import xtcav                                                        # noqa: E402
from dataset import DATASET                                         # noqa: E402
from pmd_beamphysics import ParticleGroup                           # noqa: E402

from twobunch_s2e_rl.xtcav_metrics import (                         # noqa: E402
    UM_PER_DEG_RF, energy_gated_image_features)

# deployed preprocessing (identical to xtcav_real_features.py Tier 0)
HAND_CROP = (200, 0, 650, 300)
AUTO_CROP = {"xrange": 150, "yrange": 80}
HOT_PIX_THRESHOLD, NOISE_THRESHOLD, MEDIAN_FILTER = 1e3, 35, 3
D_NOM = 59.31          # mm, nominal real DTOTR2 dispersion at 10 GeV
PIXEL_UM = 30.5        # real camera pixel; the sim scan pixelates to the same grid
C_LIGHT = 299792458.0

RESULTS = XTCAV_ROOT / "sim2real" / "results"
OUT_DIR = RESULTS / "tier2_comparison" / "screen_compare"

# ---- dark presentation theme (matches the twobunch-s2e-rl presentation figures) ----
BG, FG, DIM = "#000000", "#E8EAEE", "#9AA0AC"
REAL_C, SIM_C = "#6FB1FF", "#FF7A8A"       # accent per side (marginals, gate lines)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.titlecolor": "#FFFFFF",
})


def edges_from_centers(c):
    c = np.asarray(c, float)
    d = np.diff(c)
    step = d[0] if len(d) else 1.0
    return np.concatenate([[c[0] - step / 2], c[:-1] + d / 2, [c[-1] + step / 2]])


def signed_z_axis(img2d, x_um, cal_scaled, z_sign):
    """Map screen x -> beam z with the streak SIGN applied, keeping z ascending.

    The deployed pipeline divides by |cal| (jitter_calibration takes abs(slope);
    calibrate_phase_position stores the sign separately in `slope_sign`), so its z axis is the
    streak axis up to a reflection. Dropping that sign is what made an earlier version of this
    figure show the sim drive TRAILING the witness, which is backwards for FACET-II.

    With z_sign = -1 the axis descends with column index, which would flip the sign of the
    trapezoid in current_profile and reverse the image; so the columns are reversed here to keep
    z ascending, and the image is reversed with them.

    Returns (img2d, z_um, z_shift_um) with the smoothed x-projection peak at z = 0.
    """
    z = z_sign * np.asarray(x_um, float) / cal_scaled
    if z[-1] < z[0]:
        z, img2d = z[::-1], img2d[:, ::-1]
    proj = img2d.sum(axis=0)
    z_shift = float(z[int(np.argmax(ndimage.gaussian_filter(proj, sigma=2.0)))]) \
        if np.any(proj > 0) else 0.0
    return img2d, z - z_shift, z_shift


def image_chirp(img, z_um, delta_pct):
    """Charge-weighted dE/E per mm of z from a screen image [%/mm] -- the LPS chirp sign."""
    w = img.sum()
    Z, D = np.meshgrid(np.asarray(z_um, float), np.asarray(delta_pct, float))
    zb, db = (img * Z).sum() / w, (img * D).sum() / w
    return float((img * (Z - zb) * (D - db)).sum() / (img * (Z - zb) ** 2).sum() * 1e3)


def current_profile(proj, z_um, charge_nC):
    """I(z) with integral I dt = Q, deployed convention (_current_from_image)."""
    denom = integrate.trapezoid(proj, z_um * 1e-6)
    if not np.isfinite(denom) or denom <= 0:
        return np.zeros_like(proj, float)
    return proj * (charge_nC * 1e-9) * C_LIGHT / denom * 1e-3


def load_real(dataset, step, ind, cal_um_per_deg, z_sign):
    """Deployed-pipeline corrected image + its z / dE-E axes for one shot."""
    experiment, number = dataset.rsplit("_", 1)
    ds = DATASET(str(XTCAV_ROOT), experiment, number)
    cls = xtcav.classify_tcav_shots(ds)
    steps = xtcav.get_shot_steps(ds)

    a = xtcav.XTCAVAnalysis(ds, cam="DTOTR2", d_nom=D_NOM, chargeCal=1.0,
                            hand_crop=HAND_CROP, auto_crop=AUTO_CROP,
                            hot_pix_threshold=HOT_PIX_THRESHOLD,
                            noise_threshold=NOISE_THRESHOLD, median_filter=MEDIAN_FILTER)
    a.set_tcav_calibration(cal_um_per_deg)

    # per-step distortion reference, built from this step's TCAV-off shots (as in Tier 0)
    cls_step = dict(cls)
    cls_step["ix_off"] = [i for i in cls["ix_off"] if steps[i] == step]
    refs = xtcav.build_step_distortion_references(a, cls_step, steps_per_shot=steps,
                                                  include_global_fallback=False, verbose=False)
    ref = xtcav.select_reference_for_shot(refs, ind, steps, fall_back_to_global=False)
    if ref is None:
        raise RuntimeError(f"no distortion reference for step {step}")

    img = a.get_image(int(ind))
    sh = xtcav.compute_distortion_shifts(ref["off_avg"], ref["distortion_px"])
    shifts = np.where(np.isfinite(sh["shifts_px"]), sh["shifts_px"], 0.0)
    if img.data.shape[0] != len(shifts):
        raise RuntimeError("distortion reference shape mismatch")
    img.data = xtcav.apply_row_shift(img.data, shifts)

    s = a._spec_from_image(img, int(ind))
    cp = a._current_from_image(img, int(ind))
    R = np.asarray(img.data, float)

    # Energy axis: use the deployed s.E, which IS correctly paired with img.data rows.
    #
    # This looks wrong and isn't. image.Image.y = cal*arange(height) measures distance from the
    # TOP row (image.py:175-177), while spec.forward() expects y from the BOTTOM -- so
    # generateSpectrum's `np.flip(image.y)` is exactly that conversion, and s.E[i] belongs to
    # row i. xtcav.py:403-405 says so outright ("image.Image row 0 is the TOP row; SPEC.forward()
    # expects y measured from the BOTTOM"), and the decisive test is that fit_dy_from_shot pins
    # the fitted centroid row to E_bend by construction: on shot 2384 s.E hits it to -0.011%
    # while feeding img.y to forward() directly misses by +0.350%. Row 0 = top = HIGHEST energy.
    E_row = np.asarray(s.E, float)
    w_row = R.sum(axis=1)
    E_ref = float(np.average(E_row, weights=w_row))
    delta_pct = (E_row - E_ref) / E_ref * 100.0

    R, z_um, z_shift = signed_z_axis(R, np.asarray(img.x, float) * 1e3,
                                     float(a.cal_scaled), z_sign)
    return {
        "img": R, "z_um": z_um, "delta_pct": delta_pct,
        "y_mm": np.asarray(img.y, float) * 1.0, "charge_nC": float(cp["charge_nC"]),
        "cal_scaled": float(a.cal_scaled), "cal_um_per_deg": float(cal_um_per_deg),
        "E_ref_GeV": E_ref, "z_shift_um": z_shift, "z_sign": float(z_sign),
        "z_sign_source": "oriented so the LPS chirp sign matches the sim (see --real-z-sign); "
                         "sign(jitter slope) is NOT the arrival-time direction",
        "dy_mm": float(s.dy), "E_bend_GeV": float(s.E_bend),
        "n_off_ref": int(ref["n_used"]),
        "z_anchor": "x-proj peak", "z_anchor_shift_um": 0.0,
    }


def load_sim(beam_h5, cal_um_per_deg):
    """Pixelated sim screen image + its z / dE-E axes (dispersion fit from the beam)."""
    P = ParticleGroup(str(beam_h5))
    cal_scaled = cal_um_per_deg / UM_PER_DEG_RF
    x_mm, y_mm = 1e3 * P.x, 1e3 * P.y
    px = PIXEL_UM * 1e-3
    xe = np.arange(x_mm.min() - px, x_mm.max() + 2 * px, px)
    ye = np.arange(y_mm.min() - px, y_mm.max() + 2 * px, px)
    img, _, _ = np.histogram2d(y_mm, x_mm, bins=(ye, xe), weights=P.weight)
    xc, yc = 0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:])

    # z direction: measured from the beam itself (dx/dz on the campaign z = -c*dt axis), which is
    # ground truth here and stronger than the recorded phase-scan `slope_sign`. Also record the
    # particle-level streak magnitude, which need not equal the phase-scan cal_scaled.
    k = np.polyfit(1e6 * P.z, x_mm, 1)[0]                 # mm per um of campaign z
    z_sign = float(np.sign(k))
    streak_from_beam = float(abs(k) * 1e3)                # um/um

    # energy axis: dispersion measured from the beam itself (delta vs y linear fit)
    E_p = P.energy
    E_ref = float(np.average(E_p, weights=P.weight))
    d_p = (E_p - E_ref) / E_ref
    slope = float(np.polyfit(y_mm, d_p, 1)[0])          # [1/mm]
    eta_sim_mm = 1.0 / slope
    y_cen = float(np.average(y_mm, weights=P.weight))
    delta_pct = (yc - y_cen) / eta_sim_mm * 100.0
    img, z_um, z_shift = signed_z_axis(img, 1e3 * xc, cal_scaled, z_sign)

    # ground truth available only on this side: weight-tagged drive/witness (drive = higher
    # weight, per FACET2_S2E.getDriverAndWitness). Used to check the figure, not to build it.
    wq = np.unique(P.weight)
    D, W = P[P.weight == wq[-1]], P[P.weight == wq[0]]
    truth = {
        "drive_Q_nC": float(D.charge * 1e9), "witness_Q_nC": float(W.charge * 1e9),
        "drive_frac": float(D.charge / P.charge),
        "drive_E_GeV": float(np.average(D.energy, weights=D.weight) * 1e-9),
        "witness_E_GeV": float(np.average(W.energy, weights=W.weight) * 1e-9),
        "spacing_campaign_um": float((np.average(D.z, weights=D.weight)
                                      - np.average(W.z, weights=W.weight)) * 1e6),
        "drive_leads": bool(np.average(D.t, weights=D.weight)
                            < np.average(W.t, weights=W.weight)),
    }
    truth["drive_at_higher_energy"] = truth["drive_E_GeV"] > truth["witness_E_GeV"]
    return {
        "img": img, "z_um": z_um, "delta_pct": delta_pct, "y_mm": yc,
        "charge_nC": float(P.charge * 1e9), "cal_scaled": cal_scaled,
        "cal_um_per_deg": float(cal_um_per_deg), "E_ref_GeV": E_ref * 1e-9,
        "z_shift_um": z_shift, "eta_sim_mm": eta_sim_mm, "z_sign": z_sign,
        "z_sign_source": "measured from the beam (sign of dx/dz on campaign z = -c*dt)",
        "streak_from_beam_um_per_um": streak_from_beam,
        "delta_y_corr": float(np.corrcoef(y_mm, d_p)[0, 1]), "n_particles": int(len(P.x)),
        "z_anchor": "x-proj peak", "z_anchor_shift_um": 0.0, "truth": truth,
    }


def lobe_split(side):
    """Split the image in Z at the current-profile minimum between its two prominent peaks.

    For two bunches separated in TIME this is the physically right split; the deployed energy gate
    splits horizontally in ENERGY and, on the real shot here, slices through the major bunch's own
    chirped arc instead of between the bunches. Reporting both is what shows that the apparent
    real-vs-sim disagreement on the major bunch's energy side is a gate artifact.
    """
    img, z, d = side["img"], side["z_um"], side["delta_pct"]
    proj = ndimage.gaussian_filter(img.sum(axis=0), sigma=2.0)
    pks, _ = signal.find_peaks(proj)
    if len(pks) < 2:
        return {"ok": False}
    top2 = np.sort(pks[np.argsort(signal.peak_prominences(proj, pks)[0])[::-1][:2]])
    cut = int(top2[0] + np.argmin(proj[top2[0]:top2[1] + 1]))
    Z, D = np.meshgrid(z, d)
    tot = img.sum()
    out = {"ok": True, "z_cut_um": float(z[cut])}
    for tag, sl in (("tail_lobe", slice(0, cut)), ("head_lobe", slice(cut, None))):
        sub, ws = img[:, sl], img[:, sl].sum()
        out[tag] = {"Q_frac": float(ws / tot),
                    "z_um": float((sub * Z[:, sl]).sum() / ws),
                    "delta_pct": float((sub * D[:, sl]).sum() / ws)}
    out["separation_um"] = abs(out["head_lobe"]["z_um"] - out["tail_lobe"]["z_um"])
    maj = "head_lobe" if out["head_lobe"]["Q_frac"] >= out["tail_lobe"]["Q_frac"] else "tail_lobe"
    out["major_lobe"] = maj
    out["major_frac"] = out[maj]["Q_frac"]
    out["major_leads"] = maj == "head_lobe"
    other = "tail_lobe" if maj == "head_lobe" else "head_lobe"
    out["major_at_higher_energy"] = out[maj]["delta_pct"] > out[other]["delta_pct"]
    return out


def realign_on_major(side, eg):
    """Re-anchor z on the MAJOR-charge band's z-median instead of the x-projection peak.

    A pure translation: it removes the ~170 um offset between the two sides' deployed anchors
    (which land on the arc vertex in the real image and on the drive spike in the sim) so the
    panels can be read side by side. It CANNOT mirror anything and does not change which bunch
    leads -- an earlier version of this docstring claimed otherwise, which was arithmetically
    impossible. What it does hide is that the deployed anchors pick different features, so
    --align peak is the honest view when that matters.
    """
    if not eg.get("ok"):
        return
    # Anchor on the Z-lobe major bunch, not the energy-gate major band: on the real shot the gate
    # major band is the mislabelled one (it slices the chirped bunch horizontally), so anchoring on
    # it lands between features and offsets the whole panel relative to the sim.
    lb = lobe_split(side)
    if lb.get("ok"):
        shift = float(lb[lb["major_lobe"]]["z_um"])
    else:
        maj = eg["band_low_y"] if eg["band_low_y"]["Q_nC"] >= eg["band_high_y"]["Q_nC"] \
            else eg["band_high_y"]
        shift = float(maj["z_median_um"])
    side["z_um"] = side["z_um"] - shift
    side["z_anchor_shift_um"] = shift
    side["z_anchor"] = "major z-lobe"
    for tag in ("band_low_y", "band_high_y"):
        for k in ("z_median_um", "z_mean_um"):
            eg[tag][k] = eg[tag][k] - shift


def gate(side):
    """Deployed energy-gate on the side's own image, plus the band-vs-energy mapping.

    energy_gated_image_features names its bands by ROW INDEX (band_low_y = rows before the
    split), which is not the same as low ENERGY: the real array runs high energy -> low (row 0 =
    top = highest E), the sim low -> high (make_pixel_image row 0 = lowest y). Resolve it here
    from delta_pct so the figure and the JSON talk about energy, not row order.
    """
    eg = energy_gated_image_features(side["img"], side["z_um"] * side["cal_scaled"] * 1e-3,
                                     side["y_mm"], side["cal_scaled"], side["charge_nC"])
    if not eg["ok"]:
        return eg
    d = side["delta_pct"]
    sr = eg["split_row"]
    # mean delta of each row band tells which band is the higher-energy one
    lo_rows, hi_rows = slice(0, sr), slice(sr, None)
    w = side["img"].sum(axis=1)
    dm = lambda sl: float(np.average(d[sl], weights=w[sl])) if w[sl].sum() > 0 else np.nan
    eg["delta_band_low_y_pct"], eg["delta_band_high_y_pct"] = dm(lo_rows), dm(hi_rows)
    hi_tag = "band_low_y" if eg["delta_band_low_y_pct"] > eg["delta_band_high_y_pct"] else "band_high_y"
    lo_tag = "band_high_y" if hi_tag == "band_low_y" else "band_low_y"
    eg["higher_energy_band"], eg["lower_energy_band"] = hi_tag, lo_tag
    eg["Q_higher_energy_nC"] = eg[hi_tag]["Q_nC"]
    eg["Q_lower_energy_nC"] = eg[lo_tag]["Q_nC"]
    qs = np.array([eg["Q_higher_energy_nC"], eg["Q_lower_energy_nC"]])
    eg["major_fraction"] = float(qs.max() / qs.sum())
    eg["major_band_energy"] = "higher" if qs[0] >= qs[1] else "lower"
    eg["split_delta_pct"] = float(d[sr])
    return eg


def panel(fig, gs_col, side, eg, label, accent, zlim, dlim, ilim, show_y):
    ax_i = fig.add_subplot(gs_col[0])
    ax = fig.add_subplot(gs_col[1], sharex=ax_i)

    order = np.argsort(side["delta_pct"])                  # energy increasing upward
    img = side["img"][order, :]
    de = edges_from_centers(side["delta_pct"][order])
    ze = edges_from_centers(side["z_um"])
    # Normalize on the 99.5th percentile of lit pixels, not the max. Dividing each panel by its
    # own max is unfair here: the sim has 10k macroparticles over 7.7k noiseless pixels (peak =
    # 1.26% of total charge) while the real image spreads over 48k pixels after median-filter +
    # threshold (peak = 0.14%), so max-normalization suppresses the sim's faint structure ~9x
    # relative to the real's and makes the sim witness look far dimmer than it is.
    lit = img[img > 0]
    vmax = float(np.percentile(lit, 99.5)) if lit.size else 1.0
    ax.pcolormesh(ze, de, np.clip(img / vmax, 0, 1), cmap="magma", vmin=0.0, vmax=1.0,
                  shading="auto", rasterized=True)

    proj = side["img"].sum(axis=0)
    ax_i.plot(side["z_um"], current_profile(proj, side["z_um"], side["charge_nC"]),
              color=accent, lw=2.0)
    ax_i.set_ylim(*ilim)
    ax_i.tick_params(labelbottom=False, labelsize=14)
    ax_i.set_title(label, fontsize=22, fontweight="bold", pad=48)

    if eg.get("ok"):
        ax.axhline(eg["split_delta_pct"], color=FG, ls=":", lw=1.6, alpha=0.75)
        for tag, mk in (("band_low_y", "-"), ("band_high_y", "-")):
            ax.axvline(eg[tag]["z_median_um"], color=accent, ls="--", lw=1.6, alpha=0.9)
        lb = lobe_split(side)
        txt = (f"z-lobe split: {lb['separation_um']:.0f} µm  |  major {lb['major_frac']:.2f}, "
               f"{'leads' if lb['major_leads'] else 'trails'}, "
               f"{'higher' if lb['major_at_higher_energy'] else 'lower'} energy\n"
               f"energy gate: {eg['separation_median_um']:.0f} µm  |  major "
               f"{eg['major_fraction']:.2f}, {eg['major_band_energy']} energy") \
            if lb.get("ok") else f"energy gate: {eg['separation_median_um']:.0f} µm"
        ax_i.text(0.5, 1.03, txt, transform=ax_i.transAxes, fontsize=13.5, color=DIM,
                  ha="center", va="bottom", linespacing=1.35)

    ax.set_xlim(*zlim)
    ax.set_ylim(*dlim)
    # z = -c*dt (campaign convention), so LARGER z = earlier = closer to the head.
    ax.set_xlabel(f"z  [µm]   (= −c·Δt → larger z = head; {side['z_anchor']} at 0)", fontsize=17)
    ax.tick_params(labelsize=14)
    if show_y:
        ax.set_ylabel("ΔE / E  [%]", fontsize=18)
        ax_i.set_ylabel("I  [kA]", fontsize=17)
    else:
        ax.tick_params(labelleft=False)
        ax_i.tick_params(labelleft=False)
    for sp in ("top", "right"):
        ax_i.spines[sp].set_visible(False)
    ax.text(0.026, 0.045, f"cal {side['cal_um_per_deg']:.0f} µm/deg"
                          f"  (scaled {side['cal_scaled']:.2f})",
            transform=ax.transAxes, fontsize=13.5, color=DIM, ha="left", va="bottom")
    return ax


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="E300_15673")
    ap.add_argument("--step", type=int, default=17)
    ap.add_argument("--run-tag", default="l2ref-38.00_value_match")
    ap.add_argument("--shot", type=int, default=None,
                    help="DAQ shot index; default = the step's median-separation shot")
    ap.add_argument("--branch", choices=("plus", "neg"), default="plus",
                    help="real streak branch to pair with the sim +22 MV run "
                         "(Tier 2 primary convention is plus; neg mirrors z)")
    ap.add_argument("--align", choices=("major", "peak"), default="major",
                    help="z anchor: major-charge band at 0 (comparable panels) or the deployed "
                         "x-projection peak at 0")
    ap.add_argument("--real-z-sign", choices=("auto", "+1", "-1"), default="auto",
                    help="real streak direction. 'auto' anchors it MODEL-FREE: the "
                         "majority-charge z-lobe is the drive (3:1 cathode double pulse) and the "
                         "drive leads, so the axis is oriented to put it at the head.")
    ap.add_argument("--sim-blur-px", type=float, default=0.0,
                    help="Gaussian blur on the sim image. DEFAULT 0 (off) on purpose: sigma~3 "
                         "happens to bring the sim peak current (13.9 kA) onto the real 6.0 kA and "
                         "moves the energy-gate separation 136->128 um over sigma 0-5, i.e. it is a "
                         "knob that tunes agreement. The repo's own measured real resolution "
                         "(profiles/, sigma_eff 17-21 um) is 6.5-8 px, where the sim energy gate "
                         "fails outright. The z-lobe separation is blur-insensitive (128-132 um "
                         "over sigma 0-8), which is one reason to prefer it.")
    args = ap.parse_args()

    t0 = RESULTS / "tier0_real_features" / args.dataset
    t1 = RESULTS / "tier1_sim_scans" / args.dataset / args.run_tag
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- provenance: real calibration + the representative shot at this step ----
    real_sum = json.load(open(t0 / "summary.json"))
    cal_real = float(real_sum["calibration"]["cal_um_per_deg_used"])
    scan_vals = np.asarray(real_sum["scan"]["vals"], float)
    l2_val = float(scan_vals[args.step - 1])

    F = np.load(t0 / "features.npz")
    sgn = 1.0 if args.branch == "plus" else -1.0
    m = (F["step"] == args.step) & (F["streak_sign"] == sgn) & (F["eg_ok"] > 0)
    sep_shots, ix_shots = F["eg_sep_median_um"][m], F["ix"][m].astype(int)
    step_med = float(np.median(sep_shots))
    if args.shot is None:
        shot = int(ix_shots[int(np.argmin(np.abs(sep_shots - step_med)))])
    else:
        shot = args.shot
    print(f"{args.dataset} step {args.step} (L2 {l2_val:+.2f}°, {args.branch} branch): "
          f"{m.sum()} shots, per-step median eg_sep {step_med:.1f} µm -> shot ix={shot}")

    sim_sum = json.load(open(t1 / "summary.json"))
    cal_sim = float(sim_sum["phase_scan_calibration"]["cal_um_per_deg"])

    per_step = real_sum["calibration"]["per_step"]
    slope_signed = float(per_step[str(args.step)][f"slope_{args.branch}_um_per_deg"])

    sim = load_sim(t1 / f"beam_DTOTR_step{args.step:02d}_plus.h5", cal_sim)
    chirp_sim = image_chirp(sim["img"], sim["z_um"], sim["delta_pct"])

    # ---- real streak DIRECTION: anchored MODEL-FREE ----------------------------------------
    # The head direction follows from two facts about the machine, with no reference to the sim:
    #   (1) the two-bunch production mode is a cathode double pulse with a ~3:1 charge ratio, so
    #       the DRIVE is the majority-charge bunch (Tier 0 major frac 0.70-0.73);
    #   (2) the drive LEADS -- the witness rides in its wake.
    # So whichever z-lobe carries the majority charge must be the head one. Orient the axis to
    # make that true. Corroborated two ways: sign(dx/dphi) < 0 in 32/33 steps (33/33 on the neg
    # branch) with z = -c*dt giving z_sign = -sign(dx/dphi) = +1; and the resulting LPS chirp sign
    # then matches the sim's (+10.5 vs +9.7 %/mm) -- which, because the orientation no longer uses
    # the chirp, IS an independent check and it passes.
    #
    # An earlier version oriented by chirp-matching instead. That was both circular AND degenerate:
    # flipping z and dE together (a 180 deg rotation) preserves the chirp sign while inverting the
    # whole lobe topology, so chirp matching constrains only the PRODUCT of the two axis signs.
    real_pos = load_real(args.dataset, args.step, shot, cal_real, +1.0)
    chirp_real_pos = image_chirp(real_pos["img"], real_pos["z_um"], real_pos["delta_pct"])
    lb_pos = lobe_split(real_pos)
    if args.real_z_sign == "auto":
        if not lb_pos.get("ok"):
            raise RuntimeError("cannot anchor the real z direction: no two-lobe split")
        z_sign_real = 1.0 if lb_pos["major_leads"] else -1.0
    else:
        z_sign_real = float(args.real_z_sign)
    print(f"  real head anchor: with z_sign=+1 the majority z-lobe "
          f"({lb_pos.get('major_frac', float('nan')):.3f}) "
          f"{'LEADS' if lb_pos.get('major_leads') else 'TRAILS'} -> z_sign {z_sign_real:+.0f}")
    real = real_pos if z_sign_real > 0 else load_real(args.dataset, args.step, shot,
                                                     cal_real, z_sign_real)
    print(f"  real step-{args.step} signed jitter slope = {slope_signed:+.1f} µm/deg "
          f"(|median| {cal_real:.1f} used for the scale); sign(slope) = "
          f"{np.sign(slope_signed):+.0f} is NOT the time direction")
    print(f"  chirp: sim {chirp_sim:+.2f} %/mm, real at z_sign=+1 {chirp_real_pos:+.2f} %/mm "
          f"-> real z_sign {z_sign_real:+.0f} ({args.real_z_sign})")
    tr = sim["truth"]
    print(f"  sim TRUTH (weight tag): drive {tr['drive_Q_nC']:.3f} nC "
          f"({tr['drive_frac']:.3f}) at {tr['drive_E_GeV']:.4f} GeV, witness "
          f"{tr['witness_Q_nC']:.3f} nC at {tr['witness_E_GeV']:.4f} GeV; "
          f"drive {'LEADS' if tr['drive_leads'] else 'TRAILS'}, "
          f"campaign spacing {tr['spacing_campaign_um']:+.1f} µm")
    print(f"  sim z_sign {sim['z_sign']:+.0f} ({sim['z_sign_source']}); streak from beam "
          f"{sim['streak_from_beam_um_per_um']:.2f} vs phase-scan {sim['cal_scaled']:.2f} µm/µm")
    eg_r, eg_s = gate(real), gate(sim)
    eg_s_sharp = dict(eg_s)
    if args.sim_blur_px > 0:                  # like-for-like resolution before gating/plotting
        sim["img"] = ndimage.gaussian_filter(sim["img"], sigma=args.sim_blur_px)
        eg_s = gate(sim)
        print(f"  sim blur σ={args.sim_blur_px:.1f} px: eg_sep "
              f"{eg_s_sharp['separation_median_um']:.1f} -> {eg_s['separation_median_um']:.1f} µm, "
              f"major_frac {eg_s_sharp['major_fraction']:.3f} -> {eg_s['major_fraction']:.3f}")
    if args.align == "major":
        realign_on_major(real, eg_r)
        realign_on_major(sim, eg_s)

    lb_r, lb_s = lobe_split(real), lobe_split(sim)
    for nm, lb in (("real", lb_r), ("sim", lb_s)):
        if lb.get("ok"):
            print(f"  {nm:4s} z-lobe split: sep={lb['separation_um']:.1f} µm  major_frac="
                  f"{lb['major_frac']:.3f}  major {'LEADS' if lb['major_leads'] else 'TRAILS'}"
                  f"  major at {'higher' if lb['major_at_higher_energy'] else 'lower'} energy"
                  f"  (head {lb['head_lobe']['delta_pct']:+.2f}%, "
                  f"tail {lb['tail_lobe']['delta_pct']:+.2f}%)")
    for nm, side, eg in (("real", real, eg_r), ("sim", sim, eg_s)):
        print(f"  {nm:4s} img {side['img'].shape}  Q={side['charge_nC']:.3f} nC  "
              f"E_ref={side['E_ref_GeV']:.3f} GeV  cal_scaled={side['cal_scaled']:.2f}")
        print(f"       gate: sep={eg['separation_median_um']:.1f} µm  "
              f"major_frac={eg['major_fraction']:.3f} at {eg['major_band_energy']} energy  "
              f"(Q_hiE={eg['Q_higher_energy_nC']:.3f}, Q_loE={eg['Q_lower_energy_nC']:.3f} nC)")
    print(f"  sim dispersion from beam: eta={sim['eta_sim_mm']:.2f} mm "
          f"vs real d_nom={D_NOM} mm  (corr delta-vs-y {sim['delta_y_corr']:+.4f})")
    print(f"  cal_scaled ratio real/sim = {real['cal_scaled']/sim['cal_scaled']:.3f}")

    # ---- shared axes ----
    zlim = (min(real["z_um"].min(), sim["z_um"].min()),
            max(real["z_um"].max(), sim["z_um"].max()))
    dlim = (min(real["delta_pct"].min(), sim["delta_pct"].min()),
            max(real["delta_pct"].max(), sim["delta_pct"].max()))
    ipk = max(current_profile(real["img"].sum(axis=0), real["z_um"], real["charge_nC"]).max(),
              current_profile(sim["img"].sum(axis=0), sim["z_um"], sim["charge_nC"]).max())
    ilim = (0, 1.12 * ipk)

    anchor_txt = ("major z-lobe at z = 0" if args.align == "major"
                  else "deployed x-projection peak at z = 0")
    fig = plt.figure(figsize=(16.0, 8.6))
    outer = fig.add_gridspec(1, 2, left=0.068, right=0.988, bottom=0.105, top=0.735, wspace=0.075)
    for i, (side, eg, lab, col) in enumerate((
            (real, eg_r, f"real DTOTR2  —  shot {shot} ({args.branch} branch)", REAL_C),
            (sim, eg_s, "FACET2-S2E simulation  (+22 MV)", SIM_C))):
        sub = outer[i].subgridspec(2, 1, height_ratios=(1, 4.3), hspace=0.05)
        panel(fig, sub, side, eg, lab, col, zlim, dlim, ilim, show_y=(i == 0))

    iqr = np.percentile(sep_shots, [25, 75])
    fig.suptitle("XTCAV streaked screen: real vs FACET2-S2E simulation", fontsize=25,
                 fontweight="bold", y=0.982)
    fig.text(0.5, 0.928,
             f"{args.dataset}, step {args.step}  (L2_PHASE.MKB {l2_val:+.2f}°)   —   "
             f"energy-gated separation  real {eg_r['separation_median_um']:.0f} µm  vs  "
             f"sim {eg_s['separation_median_um']:.0f} µm",
             fontsize=18, color=FG, ha="center", va="center")
    fig.text(0.5, 0.890,
             f"real step median {step_med:.0f} µm, IQR [{iqr[0]:.0f}, {iqr[1]:.0f}] over "
             f"{len(sep_shots)} shots  ·  gate-placement systematic ≈ ±50 µm on the real side "
             f"(the ± streak branches disagree 133 vs 20 µm)",
             fontsize=14.5, color=DIM, ha="center", va="center")
    fig.text(0.5, 0.856,
             f"30.5 µm pixels both sides  ·  each side self-calibrated "
             f"(×{real['cal_scaled'] / sim['cal_scaled']:.2f} streak-cal gap)  ·  {anchor_txt}  "
             f"·  {'sim blurred σ=%.0f px' % args.sim_blur_px if args.sim_blur_px > 0 else 'no sim blur'}",
             fontsize=14.5, color=DIM, ha="center", va="center")

    stem = f"screen_compare_{args.dataset}_step{args.step:02d}_{args.branch}"
    for ext, dpi in (("png", 130), ("pdf", 130)):
        fig.savefig(OUT_DIR / f"fig_{stem}.{ext}", dpi=dpi)
    json.dump({
        "dataset": args.dataset, "step": args.step, "l2_scan_val_deg": l2_val,
        "pairing": "real +90deg vs sim +22MV (Tier 2 primary convention)",
        "real_shot_ix": shot, "real_step_median_eg_sep_um": step_med,
        "real_shot_eg_sep_um": float(sep_shots[int(np.argmin(np.abs(sep_shots - step_med)))]),
        "pixel_um": PIXEL_UM, "d_nom_mm": D_NOM,
        "real": {k: v for k, v in real.items() if not isinstance(v, np.ndarray)},
        "sim": {k: v for k, v in sim.items() if not isinstance(v, np.ndarray)},
        "gate_real": eg_r, "gate_sim": eg_s, "gate_sim_unblurred": eg_s_sharp,
        "zlobe_real": lb_r, "zlobe_sim": lb_s,
        "chirp_sim_pct_per_mm": chirp_sim, "chirp_real_at_zsign_plus_pct_per_mm": chirp_real_pos,
        "real_z_sign_mode": args.real_z_sign,
        "sim_blur_px": args.sim_blur_px,
        "real_step_median_eg_sep_iqr_um": [float(x) for x in np.percentile(sep_shots,[25,75])],
        "n_shots_gated": int(len(sep_shots)),
        "cal_scaled_ratio_real_over_sim": real["cal_scaled"] / sim["cal_scaled"],
    }, open(OUT_DIR / f"{stem}.json", "w"), indent=2, default=float)
    print(f"\nwrote {OUT_DIR}/fig_{stem}.png/.pdf and {stem}.json")


if __name__ == "__main__":
    main()
