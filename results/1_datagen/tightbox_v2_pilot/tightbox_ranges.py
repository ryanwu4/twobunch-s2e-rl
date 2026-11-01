"""Observed-range figures for the tightened-box pilots.

Answers the descriptive question: given the tightened LHS box, what *range* of each
objective did the campaign actually produce, and how did tightening move it? Compares the
tightbox pilots (v1 = FF + kicker tightened only; v2 = + BC20 sextupole strengths + movers)
against the golden two-bunch working point (baseline-repeat median).

Two honesty points baked in:
  - The scorer's absolute cuts (offset < 10 um, BMAG < 1.5) are *stricter than golden itself*
    (golden offset ~47 um, golden projected BMAG_max ~2.0 -- the chirp inflates projected
    BMAG). So the decision metric here is "fraction at-or-better-than-golden", not the
    absolute cuts.
  - Projected BMAG vs beta*=0.5 is reported as-is (it is what it is), but read it relative to
    the golden line, not relative to 1.0.

Reads data/<subdir>/sample_*.json via achievable_targets.{load,derived}. No new tracking.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/tightbox_v2_pilot/tightbox_ranges.py [v2_subdir] [v1_subdir]
    (defaults: tightbox_v2_pilot tightbox_pilot)
Outputs: written beside this script in results/tightbox_v2_pilot/
         (range_comparison.png, fraction_vs_golden.png, <v2_subdir>_range_comparison.csv)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from twobunch_s2e_rl.analysis_io import load, derived

GREY, BLUE, GREEN = "#9aa4b0", "#4c72b0", "#55a868"

# (key, label, unit, log?, direction)  direction: "min" lower-is-better, "abs" |.| lower-better,
#                                        "trans" higher-is-better (transmission)
OBJECTIVES = [
    ("transmission",   "transmission",             "frac",  False, "trans"),
    ("BMAG_max",       "proj. BMAG_max (worst/4)", "-",     True,  "min"),
    ("offset",         "transverse offset",        "um",    True,  "min"),
    ("angle",          "angular misalignment",     "urad",  True,  "min"),
    ("spacing",        "bunch spacing",            "um",    False, "free"),
    ("dE",             "energy diff (D-W)",        "MeV",   False, "free"),
    ("drive_emit_x",   "drive emit x",             "um-rad",True,  "min"),
    ("drive_emit_y",   "drive emit y",             "um-rad",True,  "min"),
    ("witness_emit_x", "witness emit x",           "um-rad",True,  "min"),
    ("witness_emit_y", "witness emit y",           "um-rad",True,  "min"),
    ("drive_sigz",     "drive sigma_z",            "um",    True,  "min"),
    ("witness_sigz",   "witness sigma_z",          "um",    True,  "min"),
]


def _viable_derived(subdir):
    """Return (derived-dict on witness-viable LHS subset, golden-dict from baseline repeats)."""
    df = load(subdir)
    lhs = df[(df["success"] == True) & (df["is_baseline_repeat"] == False)].copy()
    base = df[df["is_baseline_repeat"] == True].copy()
    q, qb = derived(lhs), derived(base)
    viable = np.isfinite(q["witness_BMAG_x"])
    qv = {k: v[viable] for k, v in q.items()}
    golden = {k: np.nanmedian(qb[k]) for k in qb}
    return qv, golden, int(viable.sum())


def _frac_at_least_golden(v, g, direction):
    """Fraction of draws at least as good as golden for this objective."""
    v = v[np.isfinite(v)]
    if not v.size or not np.isfinite(g):
        return np.nan
    if direction == "trans":
        return np.mean(v >= g)
    if direction in ("min", "abs"):
        return np.mean(np.abs(v) <= abs(g))
    return np.nan  # "free" objectives have no better/worse


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("v2", nargs="?", default="tightbox_v2_pilot")
    ap.add_argument("v1", nargs="?", default="tightbox_pilot")
    args = ap.parse_args()

    q2, golden, n2 = _viable_derived(args.v2)
    q1, _, n1 = _viable_derived(args.v1)
    figdir = Path(__file__).resolve().parent

    labels = {"v1": f"v1 FF+kick (n={n1})", "v2": f"v2 +sext+movers (n={n2})"}

    # ---- summary table -----------------------------------------------------------------
    rows = []
    for key, label, unit, _, direction in OBJECTIVES:
        d = dict(objective=label, unit=unit, golden=golden.get(key, np.nan))
        for tag, q, n in (("v1", q1, n1), ("v2", q2, n2)):
            v = q[key][np.isfinite(q[key])]
            if v.size:
                p5, p50, p95 = np.percentile(v, [5, 50, 95])
                d[f"{tag}_p5"], d[f"{tag}_p50"], d[f"{tag}_p95"] = p5, p50, p95
            d[f"{tag}_frac_ge_golden"] = _frac_at_least_golden(q[key], golden.get(key, np.nan), direction)
        rows.append(d)
    tbl = pd.DataFrame(rows)
    csv = figdir / f"{args.v2}_range_comparison.csv"
    tbl.to_csv(csv, index=False)
    print(f"\n=== observed-range comparison: {args.v1} (v1) vs {args.v2} (v2) ===\n")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nwrote {csv}")

    # ---- FIG 1: per-objective observed ranges (box: p5-p95 whisker, box p25-p75, median) --
    ncol = 4
    nrow = int(np.ceil(len(OBJECTIVES) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow))
    for ax, (key, label, unit, log, direction) in zip(axes.ravel(), OBJECTIVES):
        data, pos, colors = [], [], []
        for i, (tag, q) in enumerate((("v1", q1), ("v2", q2))):
            v = q[key][np.isfinite(q[key])]
            if log:
                v = v[v > 0]
            if v.size:
                data.append(v); pos.append(i); colors.append(GREY if tag == "v1" else BLUE)
        if data:
            bp = ax.boxplot(data, positions=pos, widths=0.6, whis=(5, 95), showfliers=False,
                            patch_artist=True, medianprops=dict(color="k", lw=1.6))
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c); patch.set_alpha(0.75)
        g = golden.get(key, np.nan)
        if np.isfinite(g) and not (log and g <= 0):
            ax.axhline(g, color=GREEN, ls="--", lw=1.6, label=f"golden {g:.3g}")
            ax.legend(fontsize=8, loc="best")
        if log:
            ax.set_yscale("log")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["v1", "v2"], fontsize=9)
        ax.set_title(label, fontsize=10); ax.set_ylabel(unit, fontsize=9)
    for ax in axes.ravel()[len(OBJECTIVES):]:
        ax.axis("off")
    fig.suptitle(f"Observed objective ranges at PENT (box=p25-75, whisker=p5-95, line=median)\n"
                 f"{labels['v1']}  vs  {labels['v2']}   [golden = baseline-repeat median]",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figdir / "range_comparison.png", dpi=130); plt.close(fig)

    # ---- FIG 2: fraction at-or-better-than-golden --------------------------------------
    minobj = [(k, lab, d) for k, lab, u, lg, d in OBJECTIVES if d in ("min", "abs", "trans")]
    names = [lab for _, lab, _ in minobj]
    f1 = [_frac_at_least_golden(q1[k], golden[k], d) for k, _, d in minobj]
    f2 = [_frac_at_least_golden(q2[k], golden[k], d) for k, _, d in minobj]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(names) + 1.5))
    ax.barh(y - 0.2, f1, height=0.38, color=GREY, label=labels["v1"])
    ax.barh(y + 0.2, f2, height=0.38, color=BLUE, label=labels["v2"])
    for yi, (a, b) in enumerate(zip(f1, f2)):
        if np.isfinite(a):
            ax.text(a + 0.01, yi - 0.2, f"{a:.0%}", va="center", fontsize=8)
        if np.isfinite(b):
            ax.text(b + 0.01, yi + 0.2, f"{b:.0%}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, 1.05); ax.set_xlabel("fraction of viable draws at-or-better-than golden")
    ax.set_title("How often does the tight box match/beat the golden working point?", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(figdir / "fraction_vs_golden.png", dpi=130); plt.close(fig)

    print(f"wrote 2 figures to {figdir}")


if __name__ == "__main__":
    main()
