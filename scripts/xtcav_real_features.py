"""Tier 0 sim2real: extract per-shot XTCAV features from local E300 DAQ datasets.

Runs the deployed E300 analysis (yiheng's xtcav.py + DAQ_analysis_class,
imported verbatim) over the local L2-phase-scan datasets and writes per-shot
feature tables + per-step aggregates + QA figures. These are the real-data
targets that a matched FACET2-S2E scan (Tier 1) will be compared against.

Key choices (see plan / session notes):
  - Streak calibration is data-driven per dataset: these scans carry no TCAV
    phase ramp, but each step interleaves +90/-90 deg shots, and the centroid
    offset between the signs is twice the deflection amplitude A, so the
    zero-crossing calibration is cal_um_per_deg = (|dx|/2) * (pi/180).
    Centroids for this are fit on hand-crop-free, non-auto-cropped frames
    (auto-crop centers each shot on its own COM and would erase dx), exactly
    like the deployed calibrate_phase_position. Cross-checked against the
    notebook constant 1200 um/deg; both recorded.
  - Per-step off-shot row-distortion references (deployed correction) are
    applied to every analyzed shot.
  - Energy-gated features use the shared helper in
    twobunch_s2e_rl.xtcav_metrics (identical code path as the sim side),
    with both mean and weighted-median band centroids.

Run (bmad-qpad-dev env, from twobunch-s2e-rl root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    scripts/xtcav_real_features.py E300_15671 [--limit 100] [--qa]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

XTCAV_ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "DAQ_analysis_class"))
sys.path.insert(0, str(XTCAV_ROOT / "yiheng" / "XTCAV_analysis_script"))

import xtcav  # noqa: E402  (deployed analysis, imported verbatim)
import analysis as an  # noqa: E402
from dataset import DATASET  # noqa: E402

from twobunch_s2e_rl.xtcav_metrics import energy_gated_image_features  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Notebook-tuned preprocessing (identical across all three dataset notebooks)
HAND_CROP = (200, 0, 650, 300)
HOT_PIX_THRESHOLD = 1e3
NOISE_THRESHOLD = 35
MEDIAN_FILTER = 3
AUTO_CROP = {"xrange": 150, "yrange": 80}
D_NOM = 59.31          # mm, nominal dispersion at DTOTR2
NOTEBOOK_CAL = 1200.0  # um/deg, from the dedicated calibration dataset

COLLIMATOR_PVS = [
    "COLL_LI20_2069_MOTR_RBV", "COLL_LI20_2072_MOTR_RBV",
    "COLL_LI20_2073_MOTR_RBV", "COLL_LI20_2085_MOTR_RBV",
    "COLL_LI20_2086_MOTR_RBV",
]
L1_KLYS_PVS = ["KLYS_LI11_11_FB_FAST_PACT", "KLYS_LI11_21_FB_FAST_PACT"]


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def raw_centroid_um(ds, ind, bkgd):
    """Full-frame x-projection Gaussian centroid [um], deployed-calibration style."""
    img = ds.getImage("DTOTR2", ind)
    if bkgd is not None:
        img.subtract_background(bkgd)
    img.median_filter(MEDIAN_FILTER)
    proj = np.sum(img.data, axis=0)
    nx = len(proj)
    p0 = (float(np.argmax(proj)), float(np.max(proj)), nx / 10.0,
          float(np.median(proj)))
    _, popt = an.fit_gaussian(np.arange(nx), proj, p0=p0)
    x0 = float(popt[0])
    if not (0 < x0 < nx):
        raise RuntimeError("centroid out of bounds")
    return x0 * img.cal * 1e3


def jitter_calibration(ds, cls, steps_per_shot, limit_per_group=40):
    """Data-driven streak calibration from natural TCAV phase jitter.

    At a zero crossing, d<x>/dphase = deflection amplitude = cal_um_per_deg
    (the same slope the dedicated phase-scan measures). The ±90 clusters have
    ~0.34 deg rms phase jitter -> ~400 um of centroid motion at nominal cal,
    well above the spot size, so a per-(step, sign) linear regression of
    full-frame centroid vs phase readback recovers the calibration without a
    dedicated scan. The ± centroid *difference* is also recorded: it measures
    the streaked RF-pivot offset of the bunch (2·S·z_pivot), a diagnostic,
    NOT a calibration.
    """
    bkgd = ds.getImageBackground("DTOTR2")
    slopes, per_step = [], {}
    for step in np.unique(steps_per_shot):
        entry = {}
        for tag, ix_all in [("plus", cls["ix_plus"]), ("neg", cls["ix_neg"])]:
            ix = [i for i in ix_all if steps_per_shot[i] == step][:limit_per_group]
            phases, cents = [], []
            for i in ix:
                try:
                    cents.append(raw_centroid_um(ds, int(i), bkgd))
                    phases.append(float(cls["phase"][i]))
                except Exception:
                    continue
            if len(cents) >= 10 and np.std(phases) > 0.05:
                try:
                    fit = xtcav.fit_linear(np.array(phases), np.array(cents))
                except Exception:
                    continue
                entry[f"slope_{tag}_um_per_deg"] = float(fit["slope"])
                entry[f"slope_{tag}_err"] = float(fit["slope_err"])
                entry[f"centroid_{tag}_um"] = float(np.median(cents))
                entry[f"n_{tag}"] = len(cents)
                slopes.append(abs(fit["slope"]))
        if "centroid_plus_um" in entry and "centroid_neg_um" in entry:
            entry["pm_centroid_diff_um"] = (entry["centroid_plus_um"]
                                            - entry["centroid_neg_um"])
        if entry:
            per_step[int(step)] = entry
    slopes = np.array(slopes)
    out = {
        "per_step": per_step,
        "notebook_cal_um_per_deg": NOTEBOOK_CAL,
        "n_regressions": int(len(slopes)),
    }
    if len(slopes):
        out["cal_um_per_deg_jitter_median"] = float(np.median(slopes))
        out["cal_um_per_deg_jitter_iqr"] = [float(np.percentile(slopes, 25)),
                                            float(np.percentile(slopes, 75))]
    return out


def corrected_shot(a, ind, ref):
    """get_shot_corrected equivalent that also returns the corrected image.

    Mirrors xtcav.get_shot_corrected (preprocess -> row-shift by the step's
    off-shot distortion reference -> spec + current profile), but keeps the
    corrected image.IMAGE so the energy-gated features can run on it.
    """
    img = a.get_image(int(ind))
    corrected = False
    if ref is not None:
        sh = xtcav.compute_distortion_shifts(ref["off_avg"], ref["distortion_px"])
        shifts = np.where(np.isfinite(sh["shifts_px"]), sh["shifts_px"], 0.0)
        if img.data.shape[0] == len(shifts):
            img.data = xtcav.apply_row_shift(img.data, shifts)
            corrected = True
    s = a._spec_from_image(img, int(ind))
    cp = a._current_from_image(img, int(ind))
    shot = {
        "spec": s, "z_um": cp["z_um"], "z_shift_um": cp["z_shift_um"],
        "current_kA": cp["current_kA"], "proj": cp["proj"],
        "charge_nC": cp["charge_nC"], "phase_deg": a.get_phase(int(ind)),
        "E_bend": a.get_E_bend(int(ind)), "dy": s.dy, "ind": int(ind),
    }
    return shot, img, corrected


def spectrum_features(s):
    """Peak/mean/rms of the dQ/dE spectrum from a spec.SPEC."""
    E, spec = np.asarray(s.E, float), np.asarray(s.spectrum, float)
    ok = np.isfinite(E) & np.isfinite(spec) & (spec > 0)
    if not np.any(ok):
        return dict(E_peak_GeV=np.nan, E_mean_GeV=np.nan, E_rms_GeV=np.nan)
    E, spec = E[ok], spec[ok]
    mean = np.average(E, weights=spec)
    rms = np.sqrt(np.average((E - mean) ** 2, weights=spec))
    return dict(E_peak_GeV=float(E[np.argmax(spec)]), E_mean_GeV=float(mean),
                E_rms_GeV=float(rms))


def off_spot_sigma_um(a, ix_off, steps_per_shot):
    """Per-step TCAV-off streak-axis spot sigma [um] (resolution numerator)."""
    out = {}
    for step in np.unique(steps_per_shot[np.asarray(ix_off, int)]):
        sigmas = []
        for i in [j for j in ix_off if steps_per_shot[j] == step]:
            try:
                img = a.get_image(int(i))
                proj = np.sum(img.data, axis=0)
                nx = len(proj)
                p0 = (float(np.argmax(proj)), float(np.max(proj)), nx / 10.0,
                      float(np.median(proj)))
                _, popt = an.fit_gaussian(np.arange(nx), proj, p0=p0)
                sigmas.append(abs(float(popt[2])) * img.cal * 1e3)
            except Exception:
                continue
        if sigmas:
            out[int(step)] = float(np.median(sigmas))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="e.g. E300_15671, TEST_15668, E300_15673")
    ap.add_argument("--limit", type=int, default=None,
                    help="max on-shots to analyze (smoke runs)")
    ap.add_argument("--qa", action="store_true", help="extra QA example figures")
    args = ap.parse_args()

    experiment, number = args.dataset.rsplit("_", 1)
    out_dir = XTCAV_ROOT / "sim2real" / "results" / "tier0_real_features" / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ds = DATASET(str(XTCAV_ROOT), experiment, number)
    scan_vals = np.asarray(ds.params["scanVals"], float)
    print(f"{args.dataset}: N={ds.N} matched shots, "
          f"{ds.params['totalSteps']} steps, comment={ds.params['comment']!r}")

    cls = xtcav.classify_tcav_shots(ds)
    steps_per_shot = xtcav.get_shot_steps(ds)
    print(f"off/plus/neg: {len(cls['ix_off'])}/{len(cls['ix_plus'])}/"
          f"{len(cls['ix_neg'])}, mean on-amp {cls['mean_on_amp']:.2f}")

    # ---- calibration: phase-jitter regression, notebook value as anchor ---
    cal = jitter_calibration(ds, cls, steps_per_shot)
    jit = cal.get("cal_um_per_deg_jitter_median")
    if jit is not None and 500.0 < jit < 3000.0:
        cal_use, cal_source = jit, "jitter_regression"
    else:
        cal_use, cal_source = NOTEBOOK_CAL, "notebook_constant"
    cal["cal_um_per_deg_used"] = cal_use
    cal["cal_source"] = cal_source
    print(f"calibration: jitter={jit} um/deg "
          f"(IQR {cal.get('cal_um_per_deg_jitter_iqr')}), "
          f"notebook={NOTEBOOK_CAL} -> using {cal_use:.1f} ({cal_source})")

    # ---- analysis object + per-step distortion references ----------------
    a = xtcav.XTCAVAnalysis(
        ds, cam="DTOTR2", d_nom=D_NOM, chargeCal=1.0,
        hand_crop=HAND_CROP, auto_crop=AUTO_CROP,
        hot_pix_threshold=HOT_PIX_THRESHOLD, noise_threshold=NOISE_THRESHOLD,
        median_filter=MEDIAN_FILTER,
    )
    a.set_tcav_calibration(cal_use)
    refs = xtcav.build_step_distortion_references(a, cls,
                                                  steps_per_shot=steps_per_shot)

    # ---- covariate arrays (aligned to matched-shot index) ----------------
    covar = {}
    for pv in L1_KLYS_PVS:
        arr, _, _ = xtcav._read_scalar(ds, pv)
        covar[pv] = arr if arr is not None else np.full(ds.N, np.nan)
    coll = {}
    for pv in COLLIMATOR_PVS:
        try:
            arr = np.asarray(ds.getScalar("nonBSA_List_S20", pv), float)
        except Exception:
            continue
        coll[pv] = {int(s): float(np.nanmedian(arr[steps_per_shot == s]))
                    for s in np.unique(steps_per_shot)}

    # ---- main loop over on-shots -----------------------------------------
    ix_on = np.concatenate([cls["ix_plus"], cls["ix_neg"]]).astype(int)
    ix_on.sort()
    if args.limit:
        ix_on = ix_on[: args.limit]
    N = len(ix_on)

    scalar_keys = [
        "step", "scan_val", "streak_sign", "tcav_amp", "tcav_phase",
        "charge_nC", "E_bend_GeV", "dy_mm", "z_shift_um",
        "sep_peak_um", "sep_bigauss_um", "bigauss_R2", "bigauss_two_bunch",
        "sigma1_um", "sigma2_um", "A1_kA", "A2_kA",
        "Q_minsplit_lo_nC", "Q_minsplit_hi_nC",
        "eg_ok", "eg_Q_low_nC", "eg_Q_high_nC",
        "eg_sep_mean_um", "eg_sep_median_um", "eg_split_row",
        "E_peak_GeV", "E_mean_GeV", "E_rms_GeV",
        "l1_klys11_pact", "l1_klys21_pact", "corrected",
    ]
    F = {k: np.full(N, np.nan) for k in scalar_keys}
    z_grid, currents = None, None
    n_ok = 0
    t_loop = time.perf_counter()

    for k, ind in enumerate(ix_on):
        if k and k % 250 == 0:
            rate = k / (time.perf_counter() - t_loop)
            print(f"  {k}/{N} ({n_ok} ok, {rate:.1f} shots/s)")
        step = int(steps_per_shot[ind])
        F["step"][k] = step
        F["scan_val"][k] = scan_vals[step - 1] if 1 <= step <= len(scan_vals) else np.nan
        F["streak_sign"][k] = 1.0 if ind in cls["ix_plus"] else -1.0
        F["tcav_amp"][k] = cls["amp"][ind]
        F["tcav_phase"][k] = cls["phase"][ind]
        F["l1_klys11_pact"][k] = covar[L1_KLYS_PVS[0]][ind]
        F["l1_klys21_pact"][k] = covar[L1_KLYS_PVS[1]][ind]
        try:
            ref = xtcav.select_reference_for_shot(refs, int(ind), steps_per_shot)
            shot, img, corrected = corrected_shot(a, int(ind), ref)
            r = a.get_two_bunch_metrics(int(ind), precomputed_shot=shot)
        except Exception:
            continue
        n_ok += 1
        F["corrected"][k] = float(corrected)
        F["charge_nC"][k] = shot["charge_nC"]
        F["E_bend_GeV"][k] = shot["E_bend"]
        F["dy_mm"][k] = shot["dy"]
        F["z_shift_um"][k] = shot["z_shift_um"]

        if z_grid is None:
            z_grid = np.asarray(shot["z_um"], float).copy()
            currents = np.full((N, z_grid.size), np.nan)
        currents[k, :] = np.interp(z_grid, shot["z_um"], shot["current_kA"],
                                   left=0.0, right=0.0)

        sep = r["bunch_separation"]
        if sep["ok"]:
            F["sep_peak_um"][k] = sep["separation_um"]
        fit = r["bigaussian_fit"]
        if fit["ok"]:
            F["bigauss_R2"][k] = fit["R_squared"]
            F["bigauss_two_bunch"][k] = float(fit["two_bunch"])
            F["sigma1_um"][k] = fit["sigma1_um"]
            F["A1_kA"][k] = fit["A1"]
            if fit["two_bunch"]:
                F["sep_bigauss_um"][k] = fit["separation_um"]
                F["sigma2_um"][k] = fit["sigma2_um"]
                F["A2_kA"][k] = fit["A2"]
        sp = r["drive_witness_split"]
        if sp["ok"]:
            F["Q_minsplit_lo_nC"][k] = sp["Q_drive_nC"]
            F["Q_minsplit_hi_nC"][k] = sp["Q_witness_nC"]

        eg = energy_gated_image_features(
            img.data, np.asarray(img.x, float), np.asarray(img.y, float),
            a.cal_scaled, shot["charge_nC"])
        F["eg_ok"][k] = float(eg["ok"])
        if eg["ok"]:
            F["eg_Q_low_nC"][k] = eg["band_low_y"]["Q_nC"]
            F["eg_Q_high_nC"][k] = eg["band_high_y"]["Q_nC"]
            F["eg_sep_mean_um"][k] = eg["separation_mean_um"]
            F["eg_sep_median_um"][k] = eg["separation_median_um"]
            F["eg_split_row"][k] = eg["split_row"]

        sf = spectrum_features(shot["spec"])
        F["E_peak_GeV"][k] = sf["E_peak_GeV"]
        F["E_mean_GeV"][k] = sf["E_mean_GeV"]
        F["E_rms_GeV"][k] = sf["E_rms_GeV"]

    print(f"loop done: {n_ok}/{N} shots ok in "
          f"{time.perf_counter() - t_loop:.0f} s")

    off_spots = off_spot_sigma_um(a, cls["ix_off"], steps_per_shot)

    # ---- save -------------------------------------------------------------
    np.savez_compressed(
        out_dir / "features.npz", ix=ix_on,
        z_grid_um=z_grid if z_grid is not None else np.array([]),
        current_kA=currents if currents is not None else np.zeros((N, 0)),
        **F,
    )

    # per-(step, sign) aggregates
    agg = {}
    for step in sorted(set(F["step"][np.isfinite(F["step"])].astype(int))):
        for sign, tag in [(1.0, "plus"), (-1.0, "neg")]:
            m = (F["step"] == step) & (F["streak_sign"] == sign)
            if not np.any(m):
                continue
            entry = {"n": int(m.sum())}
            for key in ["sep_peak_um", "sep_bigauss_um", "sigma1_um", "sigma2_um",
                        "eg_sep_median_um", "eg_sep_mean_um", "eg_Q_low_nC",
                        "eg_Q_high_nC", "charge_nC", "E_mean_GeV", "E_rms_GeV",
                        "l1_klys11_pact"]:
                v = F[key][m]
                v = v[np.isfinite(v)]
                if len(v):
                    entry[key] = {"median": float(np.median(v)),
                                  "iqr": [float(np.percentile(v, 25)),
                                          float(np.percentile(v, 75))]}
            agg[f"step{step:02d}_{tag}"] = entry

    summary = {
        "dataset": args.dataset,
        "comment": str(ds.params["comment"]),
        "n_matched_shots": int(ds.N),
        "n_analyzed": int(N), "n_ok": int(n_ok),
        "scan": {"pv": str(ds.params["scanPVs"]),
                 "vals": scan_vals.tolist()},
        "preprocessing": {"hand_crop": HAND_CROP, "auto_crop": AUTO_CROP,
                          "hot_pix_threshold": HOT_PIX_THRESHOLD,
                          "noise_threshold": NOISE_THRESHOLD,
                          "median_filter": MEDIAN_FILTER, "d_nom_mm": D_NOM},
        "calibration": cal | {"cal_um_per_deg_used": cal_use,
                              "cal_scaled_used": a.cal_scaled},
        "collimators_per_step": coll,
        "off_spot_sigma_um_per_step": off_spots,
        "aggregates": agg,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=to_jsonable)

    # ---- QA figures ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=130)
    axc = axes[0, 0]
    stps = sorted(cal["per_step"])
    for tag, mk in [("plus", "o"), ("neg", "s")]:
        vals = [(s, abs(cal["per_step"][s][f"slope_{tag}_um_per_deg"]))
                for s in stps if f"slope_{tag}_um_per_deg" in cal["per_step"][s]]
        if vals:
            axc.plot(*zip(*vals), mk + "-", label=f"|slope| {tag}", ms=4)
    axc.axhline(NOTEBOOK_CAL, color="gray", ls="--", label="notebook 1200")
    axc.set_xlabel("step"); axc.set_ylabel("cal [um/deg]"); axc.legend(fontsize=8)
    axc.set_title("phase-jitter streak calibration per step")

    axw = axes[0, 1]
    if currents is not None:
        med_by_step = []
        for step in range(1, len(scan_vals) + 1):
            m = (F["step"] == step) & (F["streak_sign"] == 1.0)
            med_by_step.append(np.nanmedian(currents[m], axis=0)
                               if np.any(m) else np.full(z_grid.size, np.nan))
        axw.imshow(np.asarray(med_by_step), aspect="auto", origin="lower",
                   extent=[z_grid[0], z_grid[-1], 1, len(scan_vals)],
                   cmap="viridis")
        axw.set_xlabel("z [um]"); axw.set_ylabel("step")
        axw.set_title("median I(z) waterfall (+90 shots)")

    axs = axes[1, 0]
    for key, lbl in [("sep_peak_um", "peak sep"),
                     ("eg_sep_median_um", "energy-gated sep (median)")]:
        med = [np.nanmedian(F[key][(F["step"] == s) & (F["streak_sign"] == 1.0)])
               for s in range(1, len(scan_vals) + 1)]
        axs.plot(scan_vals, med, "o-", label=lbl)
    axs.set_xlabel("L2 phase scan value [deg]"); axs.set_ylabel("[um]")
    axs.legend(); axs.set_title("separation vs L2 phase")

    axq = axes[1, 1]
    for key, lbl in [("eg_Q_high_nC", "band_high_y Q"),
                     ("eg_Q_low_nC", "band_low_y Q"),
                     ("charge_nC", "toroid Q")]:
        med = [np.nanmedian(F[key][(F["step"] == s) & (F["streak_sign"] == 1.0)])
               for s in range(1, len(scan_vals) + 1)]
        axq.plot(scan_vals, med, "o-", label=lbl)
    axq.set_xlabel("L2 phase scan value [deg]"); axq.set_ylabel("[nC]")
    axq.legend(); axq.set_title("charge split vs L2 phase")
    fig.tight_layout()
    fig.savefig(out_dir / "qa_overview.png")
    plt.close(fig)

    if args.qa:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=130)
        for ax, (tag, ix_list) in zip(
                axes, [("plus", cls["ix_plus"]), ("neg", cls["ix_neg"]),
                       ("off", cls["ix_off"])]):
            if not len(ix_list):
                continue
            img = a.get_image(int(ix_list[len(ix_list) // 2]))
            ax.imshow(img.data, origin="upper", aspect="auto", cmap="viridis",
                      vmax=np.percentile(img.data, 99.8),
                      extent=[img.x[0], img.x[-1], img.y[-1], img.y[0]])
            ax.set_title(f"{tag} (preprocessed)", fontsize=9)
            ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]")
        fig.tight_layout()
        fig.savefig(out_dir / "qa_examples.png")
        plt.close(fig)

    print(f"\ninit {time.perf_counter() - t0:.0f} s total. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
