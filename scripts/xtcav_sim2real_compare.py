"""Tier 2 sim2real comparison: Tier 1 matched sim scans vs Tier 0 real features.

Per dataset: aligns the sim L2 scan to the real one by fitting the single free
parameter (L2_ref offset delta, TIER1_PLAN §3) on the energy-gated median
separation curve, then computes gap metrics:
  - separation curve: offset at overlap center, slope ratio (the Jacobian the
    RL controller cares about), RMS residual normalized to the real per-step IQR;
  - charge split: major-bunch fraction real vs sim;
  - calibration scale: sim phase-scan vs real jitter cal.

Outputs -> XTCAV_analysis/sim2real/results/tier2_comparison/
  (comparison.json, fig_separation.png, fig_chargefrac.png, CONCLUSIONS.md seed)

Run (fast, seconds):
  cd twobunch-s2e-rl && PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_sim2real_compare.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SIM2REAL = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis/sim2real/results")
OUT = SIM2REAL / "tier2_comparison"
OUT.mkdir(exist_ok=True)

DATASETS = ["E300_15671", "TEST_15668", "E300_15673"]
LABELS = {"E300_15671": "E300_15671 (L1 −26°)", "TEST_15668": "TEST_15668 (L1 −23.7°)",
          "E300_15673": "E300_15673 (L1 −20°)"}
RUN_TAG = "l2ref-38.00_value_match"
L2REF_SIM = -38.0
BLUE, ORANGE, MUT = "#2563EB", "#EA8600", "#6B7689"

plt.rcParams.update({
    "font.size": 11, "axes.grid": True, "grid.color": "#DCE2EC",
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def per_step(vals, steps, signs, key_steps, sign=+1.0, q=(25, 50, 75)):
    """Median and IQR of vals per step, one streak sign."""
    out = {p: np.full(len(key_steps), np.nan) for p in q}
    for i, s in enumerate(key_steps):
        v = vals[(steps == s) & (signs == sign)]
        v = v[np.isfinite(v)]
        if len(v):
            for p in q:
                out[p][i] = np.percentile(v, p)
    return out[50], out[25], out[75]


def major_fraction(qhi, qlo):
    qmaj = np.maximum(qhi, qlo)
    qmin = np.minimum(qhi, qlo)
    with np.errstate(invalid="ignore"):
        return qmaj / (qmaj + qmin)


def fit_shift(sv, real_med, sim_sv, sim_med, fit_mask=None, bracket=4.0,
              ds_grid=0.05):
    """delta minimizing sum (real(s) - sim(s + delta))^2 over the overlap.

    Convention: real(s) is compared to sim evaluated at s + delta, so the
    real absolute reference is L2REF_SIM + delta. fit_mask restricts the cost
    to trusted (pre-minimum) real steps. NaN sim points are dropped before
    interpolation (np.interp NaN-poisons adjacent intervals otherwise).
    """
    ok_s = np.isfinite(sim_med)
    sim_sv, sim_med = sim_sv[ok_s], sim_med[ok_s]
    okr = np.isfinite(real_med)
    if fit_mask is not None:
        okr = okr & fit_mask
    deltas = np.arange(-bracket, bracket + 1e-9, ds_grid)
    best, best_cost = 0.0, np.inf
    for d in deltas:
        sim_i = np.interp(sv, sim_sv - d, sim_med, left=np.nan, right=np.nan)
        m = okr & np.isfinite(sim_i)
        if m.sum() < 6:
            continue
        cost = float(np.mean((real_med[m] - sim_i[m]) ** 2))
        if cost < best_cost:
            best, best_cost = float(d), cost
    at_edge = bool(abs(abs(best) - bracket) < ds_grid)
    return best, best_cost, at_edge


def main():
    import argparse
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", action="append", default=[],
                    help="per-dataset run-tag override, e.g. E300_15671=l1sigma370_full")
    ap.add_argument("--out", type=str, default=str(OUT),
                    help="output dir (use a fresh dir to avoid clobbering baselines)")
    args = ap.parse_args()
    OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    run_tags = {ds: RUN_TAG for ds in DATASETS}
    for t in args.tag:
        ds_k, tag = t.split("=", 1)
        run_tags[ds_k] = tag

    summary = {}
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4.3), dpi=150, sharey=False)
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.3), dpi=150, sharey=True)

    for ax1, ax2, ds in zip(axes1, axes2, DATASETS):
        # ---- real (Tier 0) -------------------------------------------------
        R = np.load(SIM2REAL / "tier0_real_features" / ds / "features.npz")
        rs = json.load(open(SIM2REAL / "tier0_real_features" / ds / "summary.json"))
        sv = np.asarray(rs["scan"]["vals"], float)
        key_steps = np.arange(1, len(sv) + 1)
        r_med, r_lo, r_hi = per_step(R["eg_sep_median_um"], R["step"],
                                     R["streak_sign"], key_steps)
        r_frac_med, r_frac_lo, r_frac_hi = per_step(
            major_fraction(R["eg_Q_high_nC"], R["eg_Q_low_nC"]),
            R["step"], R["streak_sign"], key_steps)

        # ---- sim (Tier 1), both streak signs ---------------------------------
        S = np.load(SIM2REAL / "tier1_sim_scans" / ds / run_tags[ds] / "features.npz")
        ss = json.load(open(SIM2REAL / "tier1_sim_scans" / ds / run_tags[ds] / "summary.json"))
        sim = {}
        for sgn, sname in [(+1, "plus"), (-1, "neg")]:
            sel = S["streak_sign"] == sgn
            o = np.argsort(S["scan_val"][sel])
            sim[sname] = {
                "sv": S["scan_val"][sel][o],
                "sep": S["eg_sep_median_um"][sel][o],
                "frac": major_fraction(S["eg_Q_high_nC"], S["eg_Q_low_nC"])[sel][o],
            }

        # Pre-minimum (trusted) region of the real curve: beyond full
        # compression the gated separation is artifact-dominated on both sides.
        i_min = int(np.nanargmin(r_med))
        pre = np.zeros_like(sv, dtype=bool)
        pre[: i_min + 1] = True

        # ---- L2_ref alignment fit on the trusted region, both pairings -------
        pairings = {}
        for sname in ["plus", "neg"]:
            d, c, edge = fit_shift(sv, r_med, sim[sname]["sv"], sim[sname]["sep"],
                                   fit_mask=pre)
            son = np.interp(sv, sim[sname]["sv"] - d, sim[sname]["sep"],
                            left=np.nan, right=np.nan)
            m = pre & np.isfinite(r_med) & np.isfinite(son)
            resid = son[m] - r_med[m]
            iqr = np.maximum(r_hi[m] - r_lo[m], 1e-9)
            pairings[f"realplus_sim{sname}"] = {
                "delta_deg": d, "fit_cost": c, "fit_at_bracket_edge": edge,
                "l2ref_fit_deg": L2REF_SIM + d,   # real(s) ~ sim(s + delta)
                "n_overlap_steps": int(m.sum()),
                "sep_rms_residual_um": float(np.sqrt(np.mean(resid ** 2))),
                "sep_rms_residual_over_realIQR": float(
                    np.sqrt(np.mean((resid / iqr) ** 2))),
                "slope_real_um_per_deg": float(np.polyfit(sv[m], r_med[m], 1)[0]),
                "slope_sim_um_per_deg": float(np.polyfit(sv[m], son[m], 1)[0]),
            }
            pairings[f"realplus_sim{sname}"]["slope_ratio_sim_over_real"] = float(
                pairings[f"realplus_sim{sname}"]["slope_sim_um_per_deg"]
                / pairings[f"realplus_sim{sname}"]["slope_real_um_per_deg"])

        prim = pairings["realplus_simplus"]
        delta = prim["delta_deg"]
        sim_on_real = np.interp(sv, sim["plus"]["sv"] - delta, sim["plus"]["sep"],
                                left=np.nan, right=np.nan)
        s_frac = sim["plus"]["frac"]

        # charge fraction on separable steps only (real median sep > 60 um)
        separable = pre & np.isfinite(r_med) & (r_med > 60)
        sim_frac_on_real = np.interp(sv, sim["plus"]["sv"] - delta, s_frac,
                                     left=np.nan, right=np.nan)
        cal = rs["calibration"]
        cal_sim = ss["phase_scan_calibration"]["cal_um_per_deg"]
        summary[ds] = {
            "run_tag": run_tags[ds],
            "primary_pairing": "realplus_simplus",
            **prim,
            "pairings": pairings,
            "frac_real_separable_median": float(np.nanmedian(r_frac_med[separable])),
            "frac_sim_separable_median": float(np.nanmedian(sim_frac_on_real[separable])),
            "cal_real_jitter_um_per_deg": cal["cal_um_per_deg_used"],
            "cal_real_deployed_um_per_deg": 1200.0,
            "cal_sim_um_per_deg": cal_sim,
            "cal_ratio_jitter_over_sim": float(cal["cal_um_per_deg_used"] / cal_sim),
            "cal_ratio_deployed_over_sim": float(1200.0 / cal_sim),
        }
        m = pre & np.isfinite(r_med) & np.isfinite(sim_on_real)

        # ---- figures ---------------------------------------------------------
        for ax in (ax1, ax2):
            if i_min < len(sv) - 1:
                ax.axvspan(sv[i_min], sv[-1], color="#8A93A6", alpha=0.10, lw=0)
        ax1.text(0.98, 0.96, "bands merged →\ndiagnostic unreliable",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=7.5, color=MUT)
        ax1.fill_between(sv, r_lo, r_hi, color=BLUE, alpha=0.18, lw=0)
        ax1.plot(sv, r_med, "o-", color=BLUE, ms=3.5, label="real (median, IQR)")
        ax1.plot(sv, sim_on_real, "s-", color=ORANGE, ms=3.5,
                 label=f"sim +V (aligned, Δ={delta:+.2f}°)")
        ax1.set_title(LABELS[ds], fontsize=11, loc="left")
        ax1.set_xlabel("L2 phase scan value [deg]")
        ax1.set_ylabel("energy-gated separation [µm]")

        ax2.fill_between(sv, r_frac_lo, r_frac_hi, color=BLUE, alpha=0.18, lw=0)
        ax2.plot(sv, r_frac_med, "o-", color=BLUE, ms=3.5, label="real")
        ax2.plot(sv, sim_frac_on_real, "s-", color=ORANGE, ms=3.5,
                 label="sim +V (aligned)")
        ax2.axhline(0.75, color=MUT, ls=":", lw=1)
        ax2.text(0.2, 0.755, "laser 3:1", color=MUT, fontsize=8)
        ax2.set_title(LABELS[ds], fontsize=11, loc="left")
        ax2.set_xlabel("L2 phase scan value [deg]")
        ax2.set_ylabel("major-bunch charge fraction")
        ax2.set_ylim(0.45, 1.0)

    for ax in list(axes1) + list(axes2):
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig1.suptitle("Bunch separation vs L2 phase — real (Tier 0) vs matched sim (Tier 1)",
                  fontsize=12, x=0.02, ha="left")
    fig2.suptitle("Drive charge fraction vs L2 phase — real vs matched sim",
                  fontsize=12, x=0.02, ha="left")
    for f, n in [(fig1, "fig_separation.png"), (fig2, "fig_chargefrac.png")]:
        f.tight_layout(rect=[0, 0, 1, 0.94])
        f.savefig(OUT / n)
        plt.close(f)

    with open(OUT / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"{'dataset':<13} {'L2ref_fit':>9} {'sepRMS':>8} {'RMS/IQR':>8} "
          f"{'slope s/r':>9} {'frac r/s (sep)':>14} {'cal j/d':>11}")
    for ds, v in summary.items():
        n = v["pairings"]["realplus_simneg"]
        print(f"{ds:<13} {v['l2ref_fit_deg']:>8.2f}° {v['sep_rms_residual_um']:>7.1f}µ "
              f"{v['sep_rms_residual_over_realIQR']:>8.2f} "
              f"{v['slope_ratio_sim_over_real']:>9.2f} "
              f"{v['frac_real_separable_median']:>6.2f}/{v['frac_sim_separable_median']:<6.2f} "
              f"{v['cal_ratio_jitter_over_sim']:>5.2f}/{v['cal_ratio_deployed_over_sim']:<5.2f}")
        print(f"{'':>13}  [neg branch: RMS/IQR {n['sep_rms_residual_over_realIQR']:.2f}, "
              f"slope {n['slope_ratio_sim_over_real']:.2f}, "
              f"delta {n['delta_deg']:+.2f}"
              f"{' (bracket edge)' if n['fit_at_bracket_edge'] else ''}]")
    print(f"\noutputs -> {OUT}")


if __name__ == "__main__":
    main()
