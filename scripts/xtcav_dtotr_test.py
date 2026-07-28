"""XTCAV test run on the two-bunch baseline: simulated DTOTR readout.

Tracks the known-working two-bunch baseline (setLattice_configs/2024-10-14_twoBunch_baseline.yml)
once with XTCAV off, checkpoints the beam at ENDBC20, then retracks ENDBC20 -> DTOTR for each
XTCAV case (off / +V / optionally -V). The DTOTR readout is the (x, y) particle distribution:
x is the XTCAVF streak axis (horizontal deflector), y is the B5D36 energy-dispersion axis
(vertical bend), i.e. the screen image IS the longitudinal phase space analog
(Storey et al., PRAB 27, 051302, Fig. 6).

XTCAV recipe follows the latest working ARCHIVE notebook
("2025-04-18 TCAV optics re-revisit - multiparticle tracking.ipynb"):
VOLTAGE = 20 MV, PHI0 = 0 (deflection zero-crossing = max streak), set via raw tao commands.

ASSUMPTIONS (also recorded in results.json):
 1. Baseline two-bunch config, CSR on, transverse wakes OFF (matches datagen/phase0
    run_baseline.py; the full sweep campaign used wakes ON).
 2. Default 10k macroparticles (campaign standard is 100k) -> image shot noise; override
    with --nmacro.
 3. XTCAV at 20 MV / PHI0=0 per archive practice. No RF jitter or phase error modeled.
 4. FF/spectrometer quads: golden baseline optics. The streak-optimized optics from the
    ARCHIVE notebooks were derived for the ONE-BUNCH config and are not reused here;
    --optics configurator applies the 2024-08-16 "S20 configurator TCAV optics" set instead.
 5. B5D36 spectrometer dipole left at the config value (10 GeV/c).
 6. DTOTR is an ideal screen: physical coordinates, no pixel size / PSF / saturation /
    background / camera calibration.
 7. Checkpoint-and-retrack at ENDBC20 is exact because XTCAVF and every element changed
    between cases are downstream of ENDBC20.
 8. plasmaSIM off: the beam drifts through the (empty) plasma IP region. This mimics real
    operations, where the (invasive) XTCAV is used without plasma.
 9. Streak calibration is computed from the per-particle x(DTOTR) vs z(XTCAVF) correlation
    (a simulation-only trick); the real diagnostic self-calibrates via an RF phase scan.

Run with the bmad-qpad-dev env, from the twobunch-s2e-rl repo root:
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    scripts/xtcav_dtotr_test.py [--nmacro 1e4] [--voltage 20e6] [--cases off,plus] \
                                [--optics baseline|configurator]
"""

import os

# Single serial Bmad track -> use multiple OpenMP threads. Set before pytao import.
_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import argparse
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs

from twobunch_s2e_rl.datagen.paths import facet2_root, repo_root
from twobunch_s2e_rl import xtcav_metrics as xm

S2E_ROOT = facet2_root()

CONFIG_FILE = "setLattice_configs/2024-10-14_twoBunch_baseline.yml"
CHECKPOINT = "ENDBC20"  # everything XTCAV-dependent is downstream of here
C_LIGHT = 299792458.0
F_RF = 11.424e9  # XTCAVF rf_frequency [Hz] (LI20.bmad)
K_RF = 2 * np.pi * F_RF / C_LIGHT  # [rad/m]

# Real DTOTR2 camera pixel size: RESOLUTION=30.5 um/px in the E300_15671 DAQ
# metadata (Mako G-125B). Used to pixelate the synthetic screen.
PIXEL_UM = 30.5
# Phase offsets for the emulated RF phase-scan calibration [deg]
PHASE_SCAN_DEG = (-4.0, -2.0, 0.0, 2.0, 4.0)

# "S20 configurator TCAV optics" (2024-08-16 GUI values, quoted in the 2025-02-03 and
# 2025-04-18 ARCHIVE notebooks). Derived for one-bunch operation -- comparison case only.
CONFIGURATOR_TCAV_OPTICS = {
    "Q5FFkG": -192.1, "Q4FFkG": -162.2, "Q3FFkG": 105.6,
    "Q2FFkG": 120.5, "Q1FFkG": -209.4, "Q0FFkG": 120.5,
    "Q0DkG": -139.3, "Q1DkG": 225.4, "Q2DkG": -139.3,
}


def to_jsonable(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def set_xtcav(tao, voltage, phase_deg=0.0):
    """Raw-command XTCAV setting, as in the ARCHIVE notebooks."""
    tao.cmd(f"set ele XTCAVF VOLTAGE = {voltage}")
    tao.cmd(f"set ele XTCAVF PHI0 = {phase_deg / 360.0}")


def raw_beam(tao, ele):
    """Unfiltered ParticleGroup (keeps lost particles so indices align across elements)."""
    return ParticleGroup(data=tao.bunch_data(ele))


def bunch_stats(P):
    """Per-bunch DTOTR/XTCAVF readout statistics."""
    out = {}
    PDrive, PWitness = qs.getDriverAndWitness(P)
    for tag, PB in [("drive", PDrive), ("witness", PWitness)]:
        out[f"{tag}_n_live"] = len(PB.x)
        out[f"{tag}_charge_nC"] = PB.charge * 1e9
        out[f"{tag}_median_x_mm"] = 1e3 * np.median(PB.x)
        out[f"{tag}_median_y_mm"] = 1e3 * np.median(PB.y)
        out[f"{tag}_sigmaSI90_x_um"] = 1e6 * qs.smallestIntervalImpliedSigma(PB.x)
        out[f"{tag}_sigmaSI90_y_um"] = 1e6 * qs.smallestIntervalImpliedSigma(PB.y)
        out[f"{tag}_median_energy_GeV"] = 1e-9 * np.median(PB.energy)
        out[f"{tag}_sigmaSI90_z_um"] = 1e6 * qs.smallestIntervalImpliedSigma(
            PB.t * C_LIGHT
        )
    return out


def dtotr_figure(P_dtotr, title, png_path, bins=300):
    """Simulated DTOTR image (x = streak axis, y = dispersion axis) + per-bunch x profiles."""
    PDrive, PWitness = qs.getDriverAndWitness(P_dtotr)
    fig, (ax_img, ax_prof) = plt.subplots(
        2, 1, figsize=(8, 8), sharex=True, height_ratios=[3, 1]
    )
    x_mm, y_mm = 1e3 * P_dtotr.x, 1e3 * P_dtotr.y
    h = ax_img.hist2d(x_mm, y_mm, bins=bins, weights=P_dtotr.weight, cmap="viridis")
    fig.colorbar(h[3], ax=ax_img, label="charge [C/bin]")
    ax_img.set_ylabel("y [mm] (B5D36 dispersion axis)")
    ax_img.set_title(title)

    bin_edges = np.histogram_bin_edges(x_mm, bins=bins)
    for tag, PB, color in [("drive", PDrive, "C0"), ("witness", PWitness, "C1")]:
        ax_prof.hist(
            1e3 * PB.x, bins=bin_edges, weights=1e9 * PB.weight,
            histtype="step", color=color, label=tag,
        )
    ax_prof.set_xlabel("x [mm] (XTCAVF streak axis)")
    ax_prof.set_ylabel("charge [nC/bin]")
    ax_prof.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def phase_scan_calibration(tao, config, voltage, phases_deg=PHASE_SCAN_DEG):
    """Emulate the operational TCAV RF phase-scan calibration.

    This is the real diagnostic's self-calibration path (deployed as
    xtcav.calibrate_phase_position): scan the deflector phase at fixed
    amplitude, measure the beam x-centroid on the screen at each phase, and
    fit centroid [um] vs phase [deg]. cal_scaled = cal_um_per_deg / (c/f/360)
    is the dimensionless streak factor used to map screen x -> bunch z.

    Simulation-side difference from deployed: the centroid here is the
    charge-weighted particle mean instead of a Gaussian fit to the projection
    (which is ill-defined on a bimodal two-bunch profile); each phase point is
    a retrack from the ENDBC20 checkpoint (~0.5 s each).
    """
    cent_um, phases = [], []
    for ph in phases_deg:
        set_xtcav(tao, voltage, ph)
        qs.trackBeam(tao, str(S2E_ROOT), trackStart=CHECKPOINT, trackEnd="DTOTR",
                     **config)
        P = qs.getBeamAtElement(tao, "DTOTR")
        cent_um.append(1e6 * np.average(P.x, weights=P.weight))
        phases.append(ph)
    slope, intercept = np.polyfit(phases, cent_um, 1)
    cal_um_per_deg = abs(float(slope))
    return {
        "phases_deg": list(phases),
        "centroids_um": [float(c) for c in cent_um],
        "cal_um_per_deg": cal_um_per_deg,
        "cal_scaled": cal_um_per_deg / xm.UM_PER_DEG_RF,
        "slope_sign": float(np.sign(slope)),
    }


def make_pixel_image(P, pixel_um=PIXEL_UM):
    """Bin the screen beam into DTOTR2-sized pixels (charge-weighted).

    Returns (image[ny, nx], x_edges_mm, y_edges_mm); row 0 is the lowest y,
    axes in physical mm on the screen. No camera noise/PSF/saturation.
    """
    x_mm, y_mm = 1e3 * P.x, 1e3 * P.y
    px_mm = pixel_um * 1e-3
    x_edges = np.arange(x_mm.min() - px_mm, x_mm.max() + 2 * px_mm, px_mm)
    y_edges = np.arange(y_mm.min() - px_mm, y_mm.max() + 2 * px_mm, px_mm)
    img, _, _ = np.histogram2d(y_mm, x_mm, bins=(y_edges, x_edges),
                               weights=P.weight)
    return img, x_edges, y_edges


def deployed_current_profile(P, cal_scaled, pixel_um=PIXEL_UM):
    """Current profile I(z) following the deployed pipeline conventions.

    x-projection of the pixelated image -> z = x_um / cal_scaled -> shift the
    (lightly smoothed) peak to z = 0 -> normalize so integral I dt = Q. Q is the
    total live charge at the screen (the sim stand-in for the toroid TMIT).
    """
    from scipy import integrate, ndimage

    img, x_edges, _ = make_pixel_image(P, pixel_um)
    proj = img.sum(axis=0)
    x_um = 1e3 * 0.5 * (x_edges[:-1] + x_edges[1:])
    z_um_raw = x_um / cal_scaled

    proj_smooth = ndimage.gaussian_filter(proj, sigma=2.0)
    z_shift = float(z_um_raw[int(np.argmax(proj_smooth))])
    z_um = z_um_raw - z_shift
    # Keep z ascending for the metric functions (cal sign can mirror the axis)
    order = np.argsort(z_um)
    z_um, proj = z_um[order], proj[order]

    Q_nC = P.charge * 1e9
    denom = integrate.trapezoid(proj, z_um * 1e-6)
    current_kA = (proj * (Q_nC * 1e-9) * C_LIGHT / denom * 1e-3
                  if denom > 0 else np.zeros_like(proj))
    return {"z_um": z_um, "current_kA": current_kA, "charge_nC": float(Q_nC)}


def energy_gated_analysis(P, cal_scaled, pixel_um=PIXEL_UM):
    """Emulate the deployed energy-projection cross-check.

    Port of the concept behind charge_from_energy_projection (Cemma's MATLAB /
    yiheng's xtcav.py): the spectrometer y-axis separates drive and witness by
    energy, independently of streak quality. Deployed code splits at a
    hand-picked row; here the split row is auto-picked at the minimum of the
    smoothed y-projection between its two most prominent peaks. Adds per-band
    z centroids (an energy-gated bunch separation the deployed code doesn't
    compute, but a natural extension).
    """
    from scipy import ndimage, signal as ssig

    img, x_edges, y_edges = make_pixel_image(P, pixel_um)
    yproj = ndimage.gaussian_filter(img.sum(axis=1), sigma=2.0)
    pks, _ = ssig.find_peaks(yproj)
    if len(pks) < 2:
        return {"ok": False}
    prom = ssig.peak_prominences(yproj, pks)[0]
    top2 = np.sort(pks[np.argsort(prom)[::-1][:2]])
    between = np.arange(top2[0], top2[1] + 1)
    split_row = int(between[np.argmin(yproj[between])])

    x_um = 1e3 * 0.5 * (x_edges[:-1] + x_edges[1:])
    z_um = x_um / cal_scaled
    Q_nC = P.charge * 1e9
    total = img.sum()
    out = {"ok": True, "split_row": split_row,
           "y_split_mm": float(0.5 * (y_edges[split_row] + y_edges[split_row + 1]))}
    for tag, sl in [("band_low_y", slice(0, split_row)),
                    ("band_high_y", slice(split_row, None))]:
        sub = img[sl, :].sum(axis=0)
        s = sub.sum()
        out[tag] = {
            "Q_nC": float(Q_nC * s / total) if total > 0 else float("nan"),
            "z_centroid_um": float(np.average(z_um, weights=sub)) if s > 0 else float("nan"),
        }
    out["separation_um"] = float(abs(out["band_high_y"]["z_centroid_um"]
                                     - out["band_low_y"]["z_centroid_um"]))
    return out


def deployed_metrics_figure(prof, sep, fit, img, x_edges, y_edges, title, png_path):
    """Pixelated DTOTR2-like image + deployed-style I(z) with bi-Gaussian fit."""
    fig, (ax_img, ax_i) = plt.subplots(2, 1, figsize=(8, 8),
                                       height_ratios=[2, 1])
    ax_img.imshow(img, origin="lower", aspect="auto", cmap="viridis",
                  extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]])
    ax_img.set_xlabel("x [mm]")
    ax_img.set_ylabel("y [mm]")
    ax_img.set_title(title)

    ax_i.plot(prof["z_um"], prof["current_kA"], label="I(z)")
    if fit["ok"] and len(fit["z_fit"]):
        ax_i.plot(fit["z_fit"], fit["fit_curve"], "--",
                  label=f"bi-Gaussian (R$^2$={fit['R_squared']:.3f})")
    if sep["ok"]:
        for zp in sep["peak_positions_um"]:
            ax_i.axvline(zp, color="gray", lw=0.5)
    ax_i.set_xlabel("z [um] (peak at 0; deployed convention)")
    ax_i.set_ylabel("I [kA]")
    ax_i.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def streak_calibration(tao, voltage):
    """Streak factor S = dx(DTOTR)/dz(XTCAVF) [m/m].

    Empirical: per-particle least squares on matched live particles.
    Analytic:  S = R12(XTCAVF->DTOTR) * k_rf * eV0/pc (thin-kick estimate; XTCAVF is
               ~1 m long so treat as approximate).
    """
    P1, P2 = raw_beam(tao, "XTCAVF"), raw_beam(tao, "DTOTR")
    mask = (P1.status == 1) & (P2.status == 1)
    z1 = -C_LIGHT * (P1.t[mask] - np.mean(P1.t[mask]))
    x2 = P2.x[mask]
    S_fit = np.polyfit(z1, x2, 1)[0]

    R = qs.getMatrix(tao, "XTCAVF", "DTOTR")
    pc = np.median(P2.energy)  # ultrarelativistic
    S_analytic = R[0][1] * K_RF * voltage / pc
    return S_fit, S_analytic, R


def reconstruction_figure(tao, S_fit, png_path, bins=200):
    """True z-profile at XTCAVF vs profile reconstructed from the streaked DTOTR x."""
    P1, P2 = raw_beam(tao, "XTCAVF"), raw_beam(tao, "DTOTR")
    mask = (P1.status == 1) & (P2.status == 1)
    z_true = -C_LIGHT * (P1.t[mask] - np.mean(P1.t[mask]))
    z_rec = (P2.x[mask] - np.mean(P2.x[mask])) / S_fit
    w = P2.weight[mask]

    fig, ax = plt.subplots(figsize=(8, 4))
    edges = np.histogram_bin_edges(z_true, bins=bins)
    ax.hist(1e6 * z_true, bins=1e6 * edges, weights=1e9 * w, histtype="step",
            label="true z @ XTCAVF")
    ax.hist(1e6 * z_rec, bins=1e6 * edges, weights=1e9 * w, histtype="step",
            label="reconstructed from DTOTR x / S")
    ax.set_xlabel("z [um] (head at +z)")
    ax.set_ylabel("charge [nC/bin]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmacro", type=float, default=1e4)
    ap.add_argument("--voltage", type=float, default=20e6)
    ap.add_argument("--cases", type=str, default="off,plus",
                    help="comma list from {off,plus,minus}")
    ap.add_argument("--optics", choices=["baseline", "configurator"], default="baseline")
    ap.add_argument("--outdir", type=str, default=None)
    args = ap.parse_args()

    out_dir = (
        repo_root() / "data" / f"xtcav_test_{args.optics}"
        if args.outdir is None
        else __import__("pathlib").Path(args.outdir)
    )
    scratch = out_dir / "scratch"
    out_dir.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)

    voltages = {"off": 0.0, "plus": args.voltage, "minus": -args.voltage}
    cases = [c.strip() for c in args.cases.split(",")]

    config = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))
    if args.optics == "configurator":
        config = {**config, **CONFIGURATOR_TCAV_OPTICS}

    timing = {}
    t0 = time.perf_counter()
    tao = qs.initializeTao(
        filePath=str(S2E_ROOT),
        inputBeamFilePathSuffix=config["inputBeamFilePathSuffix"],
        csrTF=True,
        numMacroParticles=args.nmacro,
        scratchPath=str(scratch),
        randomizeFileNames=True,
    )
    timing["initializeTao_s"] = time.perf_counter() - t0

    qs.setLattice(tao, **config)  # baseline defaults have XTCAVvoltage = 0

    # Full track, XTCAV off, to establish the checkpoint + unstreaked reference
    t0 = time.perf_counter()
    qs.trackBeam(tao, str(S2E_ROOT), trackEnd="DTOTR", **config)
    timing["trackBeam_full_s"] = time.perf_counter() - t0
    print(f"\n=== Full track (XTCAV off) done in {timing['trackBeam_full_s']:.1f} s ===\n")

    P_ckpt = qs.getBeamAtElement(tao, CHECKPOINT)
    qs.makeBeamActiveBeamFile(P_ckpt, tao=tao)

    results = {
        "config_file": CONFIG_FILE,
        "optics": args.optics,
        "num_macro_particles": int(args.nmacro),
        "csr": True,
        "transverse_wakes": False,
        "xtcav_voltage_V": args.voltage,
        "xtcav_phase_deg": 0.0,
        "checkpoint": CHECKPOINT,
        "cases": {},
    }

    for case in cases:
        V = voltages[case]
        set_xtcav(tao, V, 0.0)
        t0 = time.perf_counter()
        qs.trackBeam(tao, str(S2E_ROOT), trackStart=CHECKPOINT, trackEnd="DTOTR", **config)
        timing[f"trackBeam_{case}_s"] = time.perf_counter() - t0

        case_res = {"voltage_V": V}
        for ele in ["XTCAVF", "DTOTR"]:
            P = qs.getBeamAtElement(tao, ele)
            P.write(str(out_dir / f"beam_{ele}_{case}.h5"))
            case_res[ele] = bunch_stats(P)
            case_res[ele]["n_live_total"] = len(P.x)

        P_dtotr = qs.getBeamAtElement(tao, "DTOTR")
        dtotr_figure(
            P_dtotr,
            f"Simulated DTOTR, two-bunch baseline, XTCAV {case} ({V / 1e6:+.0f} MV)",
            out_dir / f"dtotr_image_{case}.png",
        )

        if V != 0.0:
            S_fit, S_analytic, R = streak_calibration(tao, V)
            case_res["streak_S_fit"] = S_fit
            case_res["streak_S_analytic_thin_kick"] = S_analytic
            case_res["R12_XTCAVF_DTOTR_m"] = R[0][1]
            case_res["R16_XTCAVF_DTOTR"] = R[0][5]
            case_res["R36_XTCAVF_DTOTR"] = R[2][5]
            reconstruction_figure(tao, S_fit, out_dir / f"z_reconstruction_{case}.png")

        results["cases"][case] = case_res
        print(f"=== case {case}: track {timing[f'trackBeam_{case}_s']:.1f} s ===")

    # Time resolution estimate: unstreaked spot / streak factor
    if "off" in results["cases"] and any(v != 0 for v in
                                         [results["cases"][c]["voltage_V"] for c in cases]):
        on_case = next(c for c in cases if results["cases"][c]["voltage_V"] != 0)
        off = results["cases"]["off"]["DTOTR"]
        S = abs(results["cases"][on_case]["streak_S_fit"])
        for tag in ["drive", "witness"]:
            sig_off = off[f"{tag}_sigmaSI90_x_um"]
            results[f"resolution_{tag}_um"] = sig_off / S if S > 0 else None
        results["streak_used_for_resolution"] = on_case

    # --- Deployed-pipeline emulation (Yiheng's E300 DTOTR2 analysis) --------
    on_cases = [c for c in cases if voltages[c] != 0.0]
    if on_cases:
        on_case = on_cases[0]
        t0 = time.perf_counter()
        cal = phase_scan_calibration(tao, config, voltages[on_case])
        timing["phase_scan_s"] = time.perf_counter() - t0
        results["phase_scan_calibration"] = cal

        P_on = ParticleGroup(str(out_dir / f"beam_DTOTR_{on_case}.h5"))
        P_on = P_on[P_on.status == 1]
        prof = deployed_current_profile(P_on, cal["cal_scaled"])
        sep = xm.find_bunch_separation(prof["z_um"], prof["current_kA"])
        fit = xm.fit_bigaussian(prof["z_um"], prof["current_kA"], peaks=sep)
        split = xm.split_drive_witness_charge(prof["z_um"], prof["current_kA"],
                                              peaks=sep)

        img, x_edges, y_edges = make_pixel_image(P_on)
        deployed_metrics_figure(
            prof, sep, fit, img, x_edges, y_edges,
            f"Pixelated DTOTR2-like screen ({PIXEL_UM} um/px), XTCAV {on_case}",
            out_dir / f"dtotr2_deployed_{on_case}.png",
        )

        # Ground truth from macroparticle weight tags at XTCAVF
        P_x = ParticleGroup(str(out_dir / f"beam_XTCAVF_{on_case}.h5"))
        P_x = P_x[P_x.status == 1]
        PD, PW = qs.getDriverAndWitness(P_x)
        truth = {
            "separation_um": 1e6 * abs(np.median(PW.t) - np.median(PD.t)) * C_LIGHT,
            "sigma_drive_um": 1e6 * np.std(PD.t * C_LIGHT),
            "sigma_witness_um": 1e6 * np.std(PW.t * C_LIGHT),
            "Q_drive_nC": PD.charge * 1e9,
            "Q_witness_nC": PW.charge * 1e9,
        }

        # Deployed convention: drive = larger-amplitude fit component. The z
        # axis has a mirror ambiguity (cal is unsigned), so the charge split
        # sides are reported as-is plus amplitude-matched.
        meas = {
            "cal_um_per_deg": cal["cal_um_per_deg"],
            "cal_scaled": cal["cal_scaled"],
            "peak_separation_um": sep["separation_um"] if sep["ok"] else None,
            "bigauss_separation_um": fit["separation_um"] if fit["ok"] else None,
            "bigauss_R2": fit["R_squared"] if fit["ok"] else None,
            "Q_small_z_side_nC": split["Q_drive_nC"] if split["ok"] else None,
            "Q_large_z_side_nC": split["Q_witness_nC"] if split["ok"] else None,
        }
        if fit["ok"] and fit["two_bunch"]:
            if fit["A1"] >= fit["A2"]:
                meas["sigma_drive_um"] = fit["sigma1_um"]
                meas["sigma_witness_um"] = fit["sigma2_um"]
            else:
                meas["sigma_drive_um"] = fit["sigma2_um"]
                meas["sigma_witness_um"] = fit["sigma1_um"]
        egate = energy_gated_analysis(P_on, cal["cal_scaled"])
        results["deployed_analysis"] = {
            "case": on_case,
            "pixel_um": PIXEL_UM,
            "measured": meas,
            "energy_projection_crosscheck": egate,
            "truth_from_weights": truth,
        }

        print(f"\n=== Deployed-style analysis (case {on_case}) ===")
        print(f"phase-scan cal: {cal['cal_um_per_deg']:.1f} um/deg "
              f"(cal_scaled = {cal['cal_scaled']:.2f} um/um)")
        print(f"separation: measured peak={meas['peak_separation_um']}, "
              f"bigauss={meas['bigauss_separation_um']}, "
              f"truth={truth['separation_um']:.1f} um")
        if "sigma_drive_um" in meas:
            print(f"sigma drive:   measured={meas['sigma_drive_um']:.1f}, "
                  f"truth={truth['sigma_drive_um']:.1f} um")
            print(f"sigma witness: measured={meas['sigma_witness_um']:.1f}, "
                  f"truth={truth['sigma_witness_um']:.1f} um")
        print(f"Q split (small-z/large-z): {meas['Q_small_z_side_nC']:.2f}/"
              f"{meas['Q_large_z_side_nC']:.2f} nC; "
              f"truth drive/witness: {truth['Q_drive_nC']:.2f}/"
              f"{truth['Q_witness_nC']:.2f} nC")
        if egate["ok"]:
            print(f"energy-projection crosscheck: "
                  f"Q(bands)={egate['band_high_y']['Q_nC']:.2f}/"
                  f"{egate['band_low_y']['Q_nC']:.2f} nC, "
                  f"gated separation={egate['separation_um']:.1f} um "
                  f"(truth {truth['separation_um']:.1f})")

    results["timing"] = timing
    results["assumptions"] = [
        "Baseline 2024-10-14_twoBunch_baseline.yml; CSR on; transverse wakes OFF (phase0 convention, campaign used wakes ON)",
        f"{int(args.nmacro)} macroparticles (campaign standard 100k); shot noise scales accordingly",
        "XTCAV 20 MV, PHI0=0 zero-crossing (2025-04-18 ARCHIVE recipe); no RF jitter/phase error",
        f"FF/spectrometer optics: {args.optics} (streak-optimized optics only exist for one-bunch config)",
        "B5D36 at config value 10 GeV/c",
        "Ideal screen: no pixel size/PSF/saturation/background",
        "Checkpoint at ENDBC20; only ENDBC20->DTOTR retracked per case (XTCAVF is downstream)",
        "No plasma (plasmaSIM off) -- mimics invasive XTCAV use without plasma",
        "Streak calibration from per-particle correlation (sim-only); real device uses RF phase scan",
        f"Deployed-emulation: pixelated at DTOTR2's {PIXEL_UM} um/px (E300_15671 metadata) but NO camera noise, PSF, saturation, or the median-filter/hot-pixel/threshold preprocessing chain",
        "Phase-scan calibration emulated with charge-weighted centroid (deployed fits a Gaussian to the x-projection); phases +/-4 deg at fixed amplitude",
        "Toroid charge stand-in = total live macro charge at the screen (real TORO:LI20:2040:TMIT is upstream of the IP)",
        "Deployed metric conventions: drive = larger-amplitude bi-Gaussian component; unsigned cal leaves a z mirror ambiguity in the charge split sides",
    ]

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=to_jsonable)

    # Console summary
    print(f"\n=== XTCAV/DTOTR summary (optics = {args.optics}) ===")
    for case in cases:
        cr = results["cases"][case]
        d = cr["DTOTR"]
        print(f"\ncase {case} ({cr['voltage_V'] / 1e6:+.0f} MV):")
        for tag in ["drive", "witness"]:
            print(
                f"  {tag}: <x>={d[f'{tag}_median_x_mm']:+.3f} mm, "
                f"sig_x={d[f'{tag}_sigmaSI90_x_um']:.1f} um, "
                f"<y>={d[f'{tag}_median_y_mm']:+.3f} mm, "
                f"sig_y={d[f'{tag}_sigmaSI90_y_um']:.1f} um, "
                f"E={d[f'{tag}_median_energy_GeV']:.3f} GeV, "
                f"n={d[f'{tag}_n_live']}"
            )
        if "streak_S_fit" in cr:
            print(
                f"  streak S: fit={cr['streak_S_fit']:.3f}, "
                f"thin-kick={cr['streak_S_analytic_thin_kick']:.3f} "
                f"(R12={cr['R12_XTCAVF_DTOTR_m']:.2f} m)"
            )
    for tag in ["drive", "witness"]:
        if f"resolution_{tag}_um" in results:
            print(f"resolution ({tag}): {results[f'resolution_{tag}_um']:.1f} um")
    print(f"\nOutputs in {out_dir}")


if __name__ == "__main__":
    main()
