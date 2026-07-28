"""Next-phase sim2real benchmark: current-profile I(z) comparison.

Compares per-step longitudinal current profiles between the real scans (Tier 0:
per-shot I(z) on a common z grid, toroid-normalized, deployed conventions) and
the matched sim scans (Tier 1: per-step I(z), same conventions, all steps
persisted). Both z-axes are self-calibrated (real: jitter cal; sim: phase-scan
cal), so shapes and — up to the real-cal estimator systematic (~8-12%) —
amplitudes are directly comparable.

Fairness steps per the design discussion:
  - Sim profiles are convolved with the quadrature spot deficit
    sqrt(max(sig_real^2 - sig_sim^2, 0)) / cal_scaled per step (real screens
    carry more spot/PSF smearing than the ideal sim screen).
  - Alignment: charge-median centering + mirror orientation chosen per step by
    maximizing correlation with the real median profile (the unsigned cal
    leaves a mirror ambiguity on both sides).
  - Real reference: per-step median profile over +90-deg shots, with a 16-84%
    per-z band; real amplitude rescaled by TORO transmission to the screen is
    NOT applied (documented ~5% bias).

Metrics per step: peak current (real med vs sim), FWHM, shape distance
(L2 norm of charge-normalized profiles after alignment), horn count.
Outputs -> results/tier2_comparison/profiles/ (figures + profile_metrics.json).

Run (seconds, after the Tier 1 rerun):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_profile_compare.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, signal

SIM2REAL = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results")
OUT = SIM2REAL / "tier2_comparison" / "profiles"
OUT.mkdir(parents=True, exist_ok=True)

DATASETS = ["E300_15671", "TEST_15668", "E300_15673"]
LABELS = {"E300_15671": "E300_15671 (L1 −26°)", "TEST_15668": "TEST_15668 (L1 −23.7°)",
          "E300_15673": "E300_15673 (L1 −20°)"}
RUN_TAG = "l2ref-38.00_value_match"
SHOW_STEPS = (1, 9, 17, 25, 33)   # overlay figure columns
BLUE, ORANGE, MUT = "#2563EB", "#EA8600", "#6B7689"

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.color": "#DCE2EC",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def center_median(z, I):
    """Shift z so the charge-weighted median sits at 0."""
    I = np.clip(I, 0, None)
    if I.sum() <= 0:
        return z
    cum = np.cumsum(I)
    zm = np.interp(0.5 * cum[-1], cum, z)
    return z - zm


def resample(z_src, I_src, z_dst):
    return np.interp(z_dst, z_src, I_src, left=0.0, right=0.0)


def xcorr_shift(z, I_sim, I_ref, max_shift_um=60.0):
    """Refine alignment: shift (on-grid) maximizing overlap with the reference.

    Charge-median centering is ambiguous for asymmetric spike+tail profiles
    (audit finding A3b); a bounded correlation shift absorbs that.
    """
    dz = float(np.median(np.diff(z)))
    n = int(max_shift_um / dz)
    best_s, best_c = 0, -np.inf
    for s in range(-n, n + 1):
        c = float(np.dot(np.roll(I_sim, s), I_ref))
        if c > best_c:
            best_s, best_c = s, c
    out = np.roll(I_sim, best_s)
    if best_s > 0:
        out[:best_s] = 0.0
    elif best_s < 0:
        out[best_s:] = 0.0
    return out, best_s * dz


def fwhm(z, I):
    if not np.any(I > 0):
        return float("nan")
    half = 0.5 * np.max(I)
    above = np.where(I >= half)[0]
    return float(z[above[-1]] - z[above[0]]) if len(above) else float("nan")


def shape_l2(z, I_a, I_b):
    """L2 distance between charge-normalized profiles (dimensionless, 0=identical)."""
    a = np.clip(I_a, 0, None)
    b = np.clip(I_b, 0, None)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("nan")
    a = a / np.trapezoid(a, z)
    b = b / np.trapezoid(b, z)
    return float(np.sqrt(np.trapezoid((a - b) ** 2, z) /
                         np.trapezoid((a ** 2 + b ** 2) / 2, z)))


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=[],
                    help="per-dataset run-tag override, e.g. E300_15671=l1sigma370_full")
    ap.add_argument("--out-base", type=str,
                    default=str(SIM2REAL / "tier2_comparison"),
                    help="base dir: reads comparison.json here, writes profiles/ under it")
    args = ap.parse_args()
    out_base = Path(args.out_base)
    OUT = out_base / "profiles"
    OUT.mkdir(parents=True, exist_ok=True)
    run_tags = {ds: RUN_TAG for ds in DATASETS}
    for t in args.tag:
        ds_k, tag = t.split("=", 1)
        run_tags[ds_k] = tag
    comp = json.load(open(out_base / "comparison.json"))
    metrics = {}
    for ds in DATASETS:
        # ---- real per-step median profiles --------------------------------
        R = np.load(SIM2REAL / "tier0_real_features" / ds / "features.npz")
        rs = json.load(open(SIM2REAL / "tier0_real_features" / ds / "summary.json"))
        zR = R["z_grid_um"]
        IR = R["current_kA"]
        steps, signs = R["step"], R["streak_sign"]

        # ---- sim per-step profiles ------------------------------------------
        run = SIM2REAL / "tier1_sim_scans" / ds / run_tags[ds]
        ss = json.load(open(run / "summary.json"))
        cal_sim = ss["phase_scan_calibration"]["cal_scaled"]

        # Audit fix A3: pick ONE sim streak branch per dataset (the one the
        # separation comparison fits better) and forbid per-step mirroring —
        # chirp orientation cannot flip step to step.
        pair = comp[ds]["pairings"]
        branch = ("plus" if pair["realplus_simplus"]["fit_cost"]
                  <= pair["realplus_simneg"]["fit_cost"] else "neg")

        # Audit fix A4: the off-spot quadrature deficit is a no-op (~0 um) while
        # the real resolution-saturated peak ceiling implies an EFFECTIVE drive
        # resolution ~16-21 um. Calibrate sigma_eff empirically from mid-scan
        # real peaks (drive charge over resolution-limited spike) and smear the
        # sim to match. Also suppresses the 10k-macro shot noise (audit A6).
        mid = (steps >= 8) & (steps <= 26) & (signs > 0)
        pk_real_mid = np.nanmedian(np.nanmax(IR[mid], axis=1))
        Q_drive = 1.2e-9
        sig_eff = Q_drive * 299792458.0 / (np.sqrt(2 * np.pi) * pk_real_mid * 1e3) * 1e6
        sig_sim_res = np.nanmedian([ss["steps"][k]["off_sigma_x_um"]
                                    for k in ss["steps"]]) / cal_sim
        extra_um = float(np.sqrt(max(sig_eff ** 2 - sig_sim_res ** 2, 0.0)))

        z_grid = np.arange(-420.0, 420.0, 4.0)
        rows = []
        n_steps = len(rs["scan"]["vals"])

        # Fixed per-dataset orientation: majority vote of per-step preference
        votes = []
        cache = {}
        for st in range(1, n_steps + 1):
            pf = run / f"profile_step{st:02d}_{branch}.npz"
            m = (steps == st) & (signs > 0) & np.isfinite(IR).all(axis=1)
            if not pf.exists() or m.sum() < 5:
                continue
            S = np.load(pf)
            I_s = S["current_kA"]
            dz = float(np.median(np.diff(S["z_um"])))
            if extra_um > dz:
                I_s = ndimage.gaussian_filter1d(I_s, extra_um / dz)
            z_sc = center_median(S["z_um"], I_s)
            sim_i = resample(z_sc, I_s, z_grid)
            real_med = resample(center_median(zR, np.nanmedian(IR[m], axis=0)),
                                np.nanmedian(IR[m], axis=0), z_grid)
            cache[st] = (m, sim_i, real_med)
            a, _ = xcorr_shift(z_grid, sim_i, real_med)
            b, _ = xcorr_shift(z_grid, sim_i[::-1], real_med)
            votes.append(float(np.dot(a, real_med)) >= float(np.dot(b, real_med)))
        mirror_all = (np.mean(votes) < 0.5) if votes else False

        for st in sorted(cache):
            m, sim_i, _ = cache[st]
            # real: median + band, median-centered on common grid
            zc = center_median(zR, np.nanmedian(IR[m], axis=0))
            real_med = resample(zc, np.nanmedian(IR[m], axis=0), z_grid)
            real_lo = resample(zc, np.nanpercentile(IR[m], 16, axis=0), z_grid)
            real_hi = resample(zc, np.nanpercentile(IR[m], 84, axis=0), z_grid)
            if mirror_all:
                sim_i = sim_i[::-1]
            sim_i, shift_um = xcorr_shift(z_grid, sim_i, real_med)
            mirrored = bool(mirror_all)

            pk_r, pk_s = float(np.max(real_med)), float(np.max(sim_i))
            n_horn_r = len(signal.find_peaks(real_med,
                                             prominence=0.15 * pk_r)[0])
            n_horn_s = len(signal.find_peaks(sim_i, prominence=0.15 * pk_s)[0])
            rows.append({
                "step": st, "scan_val": rs["scan"]["vals"][st - 1],
                "branch": branch, "mirrored": mirrored,
                "align_shift_um": shift_um, "n_real_shots": int(m.sum()),
                "peak_real_kA": pk_r, "peak_sim_kA": pk_s,
                "peak_real_iqr_kA": [float(np.max(real_lo)), float(np.max(real_hi))],
                "fwhm_real_um": fwhm(z_grid, real_med),
                "fwhm_sim_um": fwhm(z_grid, sim_i),
                "shape_l2": shape_l2(z_grid, real_med, sim_i),
                "n_peaks_real": n_horn_r, "n_peaks_sim": n_horn_s,
                "sigma_eff_real_um": float(sig_eff),
                "sim_smear_applied_um": float(extra_um),
                "_real_med": real_med, "_real_lo": real_lo, "_real_hi": real_hi,
                "_sim": sim_i,
            })
        print(f"{ds}: branch={branch}, mirror_all={mirror_all}, "
              f"sigma_eff={sig_eff:.1f} um, sim smear={extra_um:.1f} um")
        metrics[ds] = rows

        # ---- overlay figure -------------------------------------------------
        fig, axes = plt.subplots(1, len(SHOW_STEPS), figsize=(16, 3.4), dpi=150,
                                 sharex=True)
        for ax, st in zip(axes, SHOW_STEPS):
            r = next((r for r in rows if r["step"] == st), None)
            if r is None:
                ax.set_axis_off()
                continue
            ax.fill_between(z_grid, r["_real_lo"], r["_real_hi"], color=BLUE,
                            alpha=0.18, lw=0)
            ax.plot(z_grid, r["_real_med"], color=BLUE, lw=1.4, label="real median")
            ax.plot(z_grid, r["_sim"], color=ORANGE, lw=1.4,
                    label="sim (spot-conv.)")
            ax.set_title(f"step {st} (+{r['scan_val']:.2f}°)  "
                         f"L2 shape dist {r['shape_l2']:.2f}", fontsize=9)
            ax.set_xlabel("z [µm]")
            ax.set_xlim(-300, 300)
        axes[0].set_ylabel("I [kA]")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"I(z) real vs matched sim — {LABELS[ds]}", fontsize=12,
                     x=0.02, ha="left")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(OUT / f"profiles_{ds}.png")
        plt.close(fig)

    # ---- summary figure: peak current + shape distance vs scan --------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=150, sharex="col")
    for col, ds in enumerate(DATASETS):
        rows = metrics[ds]
        sv = [r["scan_val"] for r in rows]
        ax = axes[0, col]
        ax.fill_between(sv, [r["peak_real_iqr_kA"][0] for r in rows],
                        [r["peak_real_iqr_kA"][1] for r in rows],
                        color=BLUE, alpha=0.18, lw=0)
        ax.plot(sv, [r["peak_real_kA"] for r in rows], "o-", color=BLUE, ms=3,
                label="real")
        ax.plot(sv, [r["peak_sim_kA"] for r in rows], "s-", color=ORANGE, ms=3,
                label="sim")
        ax.set_title(LABELS[ds], fontsize=11, loc="left")
        ax.set_ylabel("peak I [kA]")
        if col == 0:
            ax.legend(fontsize=8)
        ax = axes[1, col]
        ax.plot(sv, [r["shape_l2"] for r in rows], "o-", color=MUT, ms=3)
        ax.set_ylabel("shape L2 distance")
        ax.set_xlabel("L2 phase scan value [deg]")
        ax.set_ylim(0, 1.5)
    fig.suptitle("Current-profile benchmark: peak current (top) and normalized "
                 "shape distance (bottom)", fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "profile_summary.png")
    plt.close(fig)

    # strip private arrays before saving json
    for ds in DATASETS:
        for r in metrics[ds]:
            for k in list(r):
                if k.startswith("_"):
                    del r[k]
    with open(OUT / "profile_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    for ds in DATASETS:
        rows = metrics[ds]
        med_shape = np.nanmedian([r["shape_l2"] for r in rows])
        pk_ratio = np.nanmedian([r["peak_sim_kA"] / r["peak_real_kA"]
                                 for r in rows])
        print(f"{ds}: {len(rows)} steps, median shape L2 {med_shape:.2f}, "
              f"median peak ratio sim/real {pk_ratio:.2f}")
    print(f"outputs -> {OUT}")


if __name__ == "__main__":
    main()
