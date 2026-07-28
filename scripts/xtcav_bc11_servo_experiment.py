"""Small self-contained experiment: BC11 bunch-length servo in Bmad.

Emulates the machine's BC11 BLEN feedback (which held bunch length, not phase,
during the real scans) and measures the effect of the BC11 compression state on
the E300_15671 anomalies: the drive over-compression fold-over horn, the
witness-drive separation, and profile agreement with the real data.

Method: fix the E300_15671 step-17 operating point (L2 = -38 + 4 deg, value_match
optics, XTCAV +22 MV). Reference = L1 phase anchored at the recorded -26 deg;
measure sigma_z at BL11356 (the BLEN device element, post-BC11). Then for a
sweep of sigma_z setpoints around the reference, SOLVE the L1 phase that
produces that BC11 bunch length (secant servo on an ENDDL10 checkpoint, like
the real quasi-static loop), track to DTOTR, and record:
  - horn charge: drive-tagged charge > +80 um from the drive core (weight tags)
  - witness-drive median separation (truth, from t)
  - streaked-profile shape L2 vs the real step-17 median (established
    conventions: mirror, xcorr align, 13.4 um effective-resolution smear)
  - the solved L1 phase (the BLEN<->phase<->sigma_z map)

Runtime ~8 min (1 full track + 5 setpoints x (4 servo iterations x ~8 s + 1
ENDDL10->DTOTR track)).

Run (bmad-qpad-dev, from twobunch-s2e-rl root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
    scripts/xtcav_bc11_servo_experiment.py
"""

import os

_THREADS = os.environ.get("BMAD_OMP_THREADS", "32")
os.environ["OMP_NUM_THREADS"] = _THREADS
os.environ["OPENBLAS_NUM_THREADS"] = _THREADS
os.environ["MKL_NUM_THREADS"] = _THREADS

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

import FACET2_S2E as qs

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from xtcav_dtotr_test import S2E_ROOT, CONFIG_FILE, set_xtcav, make_pixel_image  # noqa: E402

C = 299792458.0
SIM2REAL = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results")
OUT = SIM2REAL / "tier2_comparison" / "bc11_servo_experiment"
OUT.mkdir(parents=True, exist_ok=True)

L2REF = -38.0
L1_GAIN_EV = 0.335e9 - 0.125e9      # setLinacsHelper's L1 energy gain

# (streak-branch sign, mirror, effective-resolution smear um) per dataset,
# from the audited profile comparison conventions
CONVENTIONS = {
    "E300_15671": (+1, True, 13.4),
    "TEST_15668": (-1, False, 14.8),
    "E300_15673": (-1, False, 18.3),
}
# set in main() from CLI:
DS, STEP = "E300_15671", 17
SETPOINT_FACTORS = (0.75, 0.9, 1.0, 1.15, 1.4)
V_TCAV, MIRROR, SIM_SMEAR_UM = 22e6, True, 13.4


def sigma_z_at_blen(tao):
    P = qs.getBeamAtElement(tao, "BL11356")
    return float(np.std(P.t) * C * 1e6), len(P.x)


def set_l1(tao, phase_deg):
    qs.setLinacPhase(tao, "L1", phase_deg)
    qs.setLinacGradientAuto(tao, "L1", L1_GAIN_EV)


def servo_l1(tao, cfg, target_um, phi0, tol_rel=0.02, max_iter=8, verbose=True):
    """Secant solve for L1 phase giving sigma_z(BL11356) = target, from ENDDL10."""
    history = []

    def measure(phi):
        set_l1(tao, phi)
        qs.trackBeam(tao, str(S2E_ROOT), trackStart="ENDDL10",
                     trackEnd="BL11356", **cfg)
        s, n = sigma_z_at_blen(tao)
        history.append((phi, s))
        if verbose:
            print(f"    servo: L1 {phi:+.3f} deg -> sigma_z {s:.1f} um (n={n})")
        return s

    s0 = measure(phi0)
    if abs(s0 - target_um) <= tol_rel * target_um:
        return phi0, s0, history
    phi1 = phi0 + (1.0 if s0 < target_um else -1.0)   # less negative = less compression
    s1 = measure(phi1)
    for _ in range(max_iter):
        if abs(s1 - target_um) <= tol_rel * target_um:
            return phi1, s1, history
        if s1 == s0:
            break
        phi2 = phi1 + (target_um - s1) * (phi1 - phi0) / (s1 - s0)
        phi2 = float(np.clip(phi2, -45.0, -5.0))
        phi0, s0, phi1 = phi1, s1, phi2
        s1 = measure(phi1)
    return phi1, s1, history


def profile_and_metrics(tao, cal_scaled):
    """Streaked DTOTR profile + weight-tagged horn/separation metrics."""
    P = qs.getBeamAtElement(tao, "DTOTR")
    img, x_edges, _ = make_pixel_image(P)
    proj = img.sum(axis=0)
    x_um = 1e3 * 0.5 * (x_edges[:-1] + x_edges[1:])
    z = x_um / cal_scaled
    z = z - z[int(np.argmax(ndimage.gaussian_filter(proj, 2.0)))]
    order = np.argsort(z)
    z, proj = z[order], proj[order]
    from scipy import integrate
    Q_nC = P.charge * 1e9
    denom = integrate.trapezoid(proj, z * 1e-6)
    I_kA = proj * Q_nC * 1e-9 * C / denom * 1e-3 if denom > 0 else 0 * proj

    w = np.sort(np.unique(P.weight))
    PW, PD = P[P.weight == w[0]], P[P.weight == w[1]]
    zD = -(PD.t - np.median(PD.t)) * C * 1e6
    z_rel_D = -(P.t - np.median(PD.t)) * C * 1e6
    horn_nC = float(PD.weight[(-(PD.t - np.median(PD.t)) * C * 1e6) > 80].sum() * 1e9)
    sep_um = float(abs(np.median(PW.t) - np.median(PD.t)) * C * 1e6)
    return {
        "z_um": z, "I_kA": I_kA,
        "horn_drive_nC_beyond80": horn_nC,
        "witness_drive_sep_um": sep_um,
        "drive_rms_um": float(np.std(zD)),
        "n_live": int(len(P.x)),
    }


def real_reference():
    R = np.load(SIM2REAL / "tier0_real_features" / DS / "features.npz")
    zR, IR = R["z_grid_um"], R["current_kA"]
    m = (R["step"] == STEP) & (R["streak_sign"] > 0) & np.isfinite(IR).all(axis=1)
    med = np.nanmedian(IR[m], axis=0)
    I = np.clip(med, 0, None)
    cum = np.cumsum(I)
    zc = zR - np.interp(0.5 * cum[-1], cum, zR)
    z_grid = np.arange(-420.0, 420.0, 4.0)
    return z_grid, np.interp(z_grid, zc, med, left=0, right=0)


def shape_l2(z, a, b):
    a, b = np.clip(a, 0, None), np.clip(b, 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    a = a / np.trapezoid(a, z)
    b = b / np.trapezoid(b, z)
    return float(np.sqrt(np.trapezoid((a - b) ** 2, z) /
                         np.trapezoid((a ** 2 + b ** 2) / 2, z)))


def compare_to_real(z_s, I_s, z_grid, real_med):
    dz = float(np.median(np.diff(z_s)))
    if SIM_SMEAR_UM > dz:
        I_s = ndimage.gaussian_filter1d(I_s, SIM_SMEAR_UM / dz)
    I = np.clip(I_s, 0, None)
    cum = np.cumsum(I)
    zc = z_s - np.interp(0.5 * cum[-1], cum, z_s)
    sim = np.interp(z_grid, zc, I_s, left=0, right=0)
    if MIRROR:
        sim = sim[::-1]
    # bounded xcorr shift
    n = int(60.0 / 4.0)
    best_s, best_c = 0, -np.inf
    for s in range(-n, n + 1):
        c = float(np.dot(np.roll(sim, s), real_med))
        if c > best_c:
            best_s, best_c = s, c
    sim = np.roll(sim, best_s)
    return sim, shape_l2(z_grid, real_med, sim)


def main():
    import argparse
    global DS, STEP, SETPOINT_FACTORS, V_TCAV, MIRROR, SIM_SMEAR_UM, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="E300_15671", choices=list(CONVENTIONS))
    ap.add_argument("--step", type=int, default=17)
    ap.add_argument("--factors", type=str, default="0.75,0.9,1.0,1.15,1.4")
    args = ap.parse_args()
    DS, STEP = args.dataset, args.step
    SETPOINT_FACTORS = tuple(float(f) for f in args.factors.split(","))
    sgn, MIRROR, SIM_SMEAR_UM = CONVENTIONS[DS]
    V_TCAV = sgn * 22e6
    OUT = SIM2REAL / "tier2_comparison" / "bc11_servo_experiment" / DS
    OUT.mkdir(parents=True, exist_ok=True)

    mc = json.load(open(SIM2REAL / "tier1_sim_scans" / DS / "machine_config.json"))
    ss = json.load(open(SIM2REAL / "tier1_sim_scans" / DS /
                        "l2ref-38.00_value_match" / "summary.json"))
    cal_scaled = ss["phase_scan_calibration"]["cal_scaled"]
    phi_anchor = mc["L1_anchor_deg"] + mc["L1_delta_per_step_deg"][str(STEP)]

    base = qs.loadConfig(f"/{CONFIG_FILE}", str(S2E_ROOT))
    ov = {e: v["kG"] for e, v in mc["quad_mappings"]["value_match"].items()}
    cfg = {**base, **ov, "B5D36GeVc": mc["B5D36GeVc"], "Q19851kG": mc["Q19851kG"],
           "Q19871kG": mc["Q19871kG"], "L1PhaseSet": phi_anchor,
           "L2PhaseSet": L2REF + mc["L2_scan_vals_deg"][STEP - 1]}

    out_scratch = OUT / "scratch"
    out_scratch.mkdir(exist_ok=True)
    tao = qs.initializeTao(filePath=str(S2E_ROOT),
                           inputBeamFilePathSuffix=base["inputBeamFilePathSuffix"],
                           csrTF=True, numMacroParticles=1e4,
                           scratchPath=str(out_scratch), randomizeFileNames=True)
    try:
        tao.cmd("set beam add_saved_at = BL11356")
    except Exception as e:
        print(f"add_saved_at BL11356: {e}")
    qs.setLattice(tao, **cfg)
    set_xtcav(tao, V_TCAV, 0.0)

    # Reference full track at the phase anchor
    t0 = time.perf_counter()
    qs.trackBeam(tao, str(S2E_ROOT), trackEnd="DTOTR", **cfg)
    print(f"reference track {time.perf_counter() - t0:.0f} s")
    sig_ref, _ = sigma_z_at_blen(tao)
    print(f"phase-anchored reference: L1 {phi_anchor:+.2f} deg -> "
          f"sigma_z(BL11356) = {sig_ref:.1f} um")
    P_dl10 = qs.getBeamAtElement(tao, "ENDDL10")
    qs.makeBeamActiveBeamFile(P_dl10, tao=tao)

    z_grid, real_med = real_reference()

    results = {"dataset": DS, "step": STEP, "phi_anchor_deg": phi_anchor,
               "sigma_ref_um": sig_ref, "cal_scaled": cal_scaled,
               "setpoints": []}
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.plot(z_grid, real_med, color="#2563EB", lw=2.2, label="real step-17 median")
    colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(SETPOINT_FACTORS)))

    for fac, col in zip(SETPOINT_FACTORS, colors):
        target = fac * sig_ref
        print(f"--- setpoint {fac:.2f} x ref = {target:.1f} um")
        phi, achieved, hist = servo_l1(tao, cfg, target, phi_anchor)
        t0 = time.perf_counter()
        qs.trackBeam(tao, str(S2E_ROOT), trackStart="ENDDL10", trackEnd="DTOTR",
                     **cfg)
        mets = profile_and_metrics(tao, cal_scaled)
        sim_aligned, l2 = compare_to_real(mets.pop("z_um"), mets.pop("I_kA"),
                                          z_grid, real_med)
        row = {"factor": fac, "target_um": target, "achieved_um": achieved,
               "L1_phase_deg": phi, "servo_iters": len(hist),
               "shape_l2_vs_real": l2, **mets}
        results["setpoints"].append(row)
        print(f"    L1={phi:+.2f} deg, horn={mets['horn_drive_nC_beyond80']:.3f} nC, "
              f"sep={mets['witness_drive_sep_um']:.1f} um, L2={l2:.2f} "
              f"({time.perf_counter() - t0:.0f} s track)")
        ax.plot(z_grid, sim_aligned, color=col, lw=1.3,
                label=f"sim σ_z={achieved:.0f} µm (L1 {phi:+.1f}°), L2={l2:.2f}")

    ax.set_xlim(-300, 300)
    ax.set_xlabel("z [µm]")
    ax.set_ylabel("I [kA]")
    ax.set_title(f"BC11 bunch-length servo: streaked profile vs σ_z setpoint "
                 f"({DS} step {STEP})", fontsize=11, loc="left")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(OUT / "servo_profiles.png")
    plt.close(fig)

    # summary panel
    sp = results["setpoints"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
    x = [r["achieved_um"] for r in sp]
    for ax, key, lbl in [(axes[0], "horn_drive_nC_beyond80", "drive horn charge [nC]"),
                         (axes[1], "witness_drive_sep_um", "witness–drive sep [µm]"),
                         (axes[2], "shape_l2_vs_real", "shape L2 vs real")]:
        ax.plot(x, [r[key] for r in sp], "o-", color="#EA8600")
        ax.axvline(sig_ref, color="#6B7689", ls=":", lw=1)
        ax.text(sig_ref, ax.get_ylim()[1], " phase-anchored ref", fontsize=7,
                color="#6B7689", va="top")
        ax.set_xlabel("σ_z(BL11356) achieved [µm]")
        ax.set_title(lbl, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "servo_metrics.png")
    plt.close(fig)

    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
