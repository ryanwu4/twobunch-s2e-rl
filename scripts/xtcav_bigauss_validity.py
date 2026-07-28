"""Where does the deployed bi-Gaussian fit work on the real E300 scans?

Yiheng's operational analysis uses the bi-Gaussian current-profile fit as the
primary two-bunch metric, with the understanding that it fails at extreme
phases where the longitudinal phase space is too distorted. This script maps
that validity domain quantitatively, using the energy-gated separation (which
tags bunches on the spectrometer axis, independent of the current profile) as
the per-shot reference.

Per (dataset, step): fraction of on-shots where the fit converged / returned a
two-bunch solution / AGREES with the energy gate (|sep_bg - sep_eg| within
max(20 um, 25%)); medians of both separations, R^2, and the witness/drive
amplitude ratio. Outputs figures + bigauss_validity.json under
results/tier2_comparison/.

Run:  PYTHONPATH=$PWD/src MPLBACKEND=Agg \
  /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_bigauss_validity.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM2REAL = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results")
OUT = SIM2REAL / "tier2_comparison"
DATASETS = ["E300_15671", "TEST_15668", "E300_15673"]
LABELS = {"E300_15671": "E300_15671 (L1 −26°)", "TEST_15668": "TEST_15668 (L1 −23.7°)",
          "E300_15673": "E300_15673 (L1 −20°)"}
BLUE, ORANGE, GREEN, MUT = "#2563EB", "#EA8600", "#1A7F4B", "#6B7689"

plt.rcParams.update({
    "font.size": 10.5, "axes.grid": True, "grid.color": "#DCE2EC",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def agree(sep_bg, sep_eg):
    tol = np.maximum(20.0, 0.25 * sep_eg)
    return np.abs(sep_bg - sep_eg) < tol


def main():
    report = {}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), dpi=150, sharex="col")
    figs, axs = plt.subplots(1, 3, figsize=(14, 4.3), dpi=150)

    for col, ds in enumerate(DATASETS):
        F = np.load(SIM2REAL / "tier0_real_features" / ds / "features.npz")
        rs = json.load(open(SIM2REAL / "tier0_real_features" / ds / "summary.json"))
        sv_all = np.asarray(rs["scan"]["vals"], float)
        steps = F["step"]
        on = np.isfinite(steps)

        two = F["bigauss_two_bunch"] == 1
        fit_ok = np.isfinite(F["bigauss_R2"])
        eg_ok = (F["eg_ok"] == 1) & np.isfinite(F["eg_sep_median_um"])
        both = two & eg_ok
        ag = np.zeros_like(two, dtype=bool)
        ag[both] = agree(F["sep_bigauss_um"][both], F["eg_sep_median_um"][both])
        # witness/drive amplitude ratio (amplitude-ordered)
        amp_ratio = np.minimum(np.abs(F["A1_kA"]), np.abs(F["A2_kA"])) / \
            np.maximum(np.abs(F["A1_kA"]), np.abs(F["A2_kA"]))

        per_step = []
        for s in range(1, len(sv_all) + 1):
            m = on & (steps == s)
            n = int(m.sum())
            if not n:
                continue
            entry = {
                "step": s, "scan_val": float(sv_all[s - 1]), "n": n,
                "frac_fit_converged": float(np.mean(fit_ok[m])),
                "frac_two_bunch": float(np.mean(two[m])),
                "frac_agree_eg": float(np.mean(ag[m])),
                "sep_bg_med": float(np.nanmedian(F["sep_bigauss_um"][m & two]))
                    if np.any(m & two) else None,
                "sep_eg_med": float(np.nanmedian(F["eg_sep_median_um"][m & eg_ok]))
                    if np.any(m & eg_ok) else None,
                "R2_med": float(np.nanmedian(F["bigauss_R2"][m & fit_ok]))
                    if np.any(m & fit_ok) else None,
                "amp_ratio_med": float(np.nanmedian(amp_ratio[m & two]))
                    if np.any(m & two) else None,
            }
            per_step.append(entry)
        report[ds] = per_step

        sv = np.array([e["scan_val"] for e in per_step])
        sep_bg = np.array([e["sep_bg_med"] or np.nan for e in per_step], float)
        sep_eg = np.array([e["sep_eg_med"] or np.nan for e in per_step], float)
        f_two = np.array([e["frac_two_bunch"] for e in per_step])
        f_ag = np.array([e["frac_agree_eg"] for e in per_step])
        r2 = np.array([e["R2_med"] or np.nan for e in per_step], float)

        ax = axes[0, col]
        ax.plot(sv, sep_eg, "o-", color=BLUE, ms=3.5, label="energy-gated (ref)")
        ax.plot(sv, sep_bg, "s-", color=ORANGE, ms=3.5, label="bi-Gaussian")
        ax.set_title(LABELS[ds], fontsize=11, loc="left")
        ax.set_ylabel("separation [µm]")
        if col == 0:
            ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.plot(sv, f_two, "o-", color=MUT, ms=3, label="frac two-bunch solution")
        ax.plot(sv, f_ag, "o-", color=GREEN, ms=3, label="frac agrees with gate")
        ax.plot(sv, r2, "--", color=ORANGE, lw=1.2, label="median R²")
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("L2 phase scan value [deg]")
        ax.set_ylabel("fraction")
        if col == 0:
            ax.legend(fontsize=8)

        # per-shot validity scatter: agreement vs gated separation
        axsc = axs[col]
        x = F["eg_sep_median_um"][both]
        y = F["sep_bigauss_um"][both]
        c = np.where(ag[both], GREEN, "#C2413B")
        axsc.scatter(x, y, s=4, c=c, alpha=0.35, lw=0)
        lim = np.nanpercentile(x, 99.5)
        axsc.plot([0, lim], [0, lim], color=MUT, lw=1, ls=":")
        axsc.set_xlim(0, lim)
        axsc.set_ylim(0, lim)
        axsc.set_title(LABELS[ds], fontsize=11, loc="left")
        axsc.set_xlabel("energy-gated separation [µm]")
        if col == 0:
            axsc.set_ylabel("bi-Gaussian separation [µm]")

    # global failure-predictor summary: agreement vs gated separation bins
    edges = np.array([0, 30, 60, 90, 120, 160, 200, 260, 340, 450])
    print(f"{'sep_eg bin [µm]':>16} " + " ".join(f"{d:>12}" for d in DATASETS))
    bin_stats = {}
    for ds in DATASETS:
        F = np.load(SIM2REAL / "tier0_real_features" / ds / "features.npz")
        two = F["bigauss_two_bunch"] == 1
        eg_ok = (F["eg_ok"] == 1) & np.isfinite(F["eg_sep_median_um"])
        both = two & eg_ok
        a = agree(F["sep_bigauss_um"][both], F["eg_sep_median_um"][both])
        x = F["eg_sep_median_um"][both]
        bin_stats[ds] = [
            (float(np.mean(a[(x >= lo) & (x < hi)])) if np.any((x >= lo) & (x < hi))
             else None)
            for lo, hi in zip(edges[:-1], edges[1:])]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        row = " ".join(
            f"{bin_stats[ds][i]:>12.2f}" if bin_stats[ds][i] is not None else f"{'—':>12}"
            for ds in DATASETS)
        print(f"{f'{lo:.0f}–{hi:.0f}':>16} {row}")
    report["_agreement_vs_sep_eg_bins"] = {"edges_um": edges.tolist(),
                                           "frac_agree": bin_stats}

    fig.suptitle("Bi-Gaussian fit validity on the real scans — separations (top), "
                 "solution/agreement fractions and R² (bottom)",
                 fontsize=12, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig_bigauss_validity.png")
    figs.suptitle("Per-shot bi-Gaussian vs energy-gated separation "
                  "(green = agree within max(20 µm, 25%))", fontsize=12, x=0.02,
                  ha="left")
    figs.tight_layout(rect=[0, 0, 1, 0.93])
    figs.savefig(OUT / "fig_bigauss_scatter.png")

    with open(OUT / "bigauss_validity.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\noutputs -> {OUT}/fig_bigauss_validity.png, fig_bigauss_scatter.png, "
          f"bigauss_validity.json")


if __name__ == "__main__":
    main()
