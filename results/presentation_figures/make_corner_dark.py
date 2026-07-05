#!/usr/bin/env python
"""Dark-theme version of the baseline-vs-optimized 6D corner plot.

Reuses cached Bmad clouds (no re-run):
  baseline  -> results/rl/openloop/clouds_baseline.npz   (bmad_drive / bmad_witness)
  optimized -> results/rl/openloop/clouds_bptt_gc_goal200um.npz (bmad_drive_0 / bmad_witness_0)

Layout mirrors src/twobunch_s2e_rl/rl/_eval_plots.py::_corner_figure (lower-triangle scatter,
diagonal 1-D step histograms, shared per-coord limits), restyled black-background with large
presentation fonts.

Color scheme -- two orthogonal contrasts kept legible on black:
  baseline = COOL (driver azure-blue, witness bright cyan)
  optimized = WARM (driver coral-red, witness amber)
so bunch role (blue/red drivers vs cyan/amber witnesses) AND set (cool baseline vs warm
optimized) are both separable.

Output: presentation_figures/corner_baseline_vs_opt_dark.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CLOUD_DIR = ROOT / "results/rl/openloop"
OUT = Path(__file__).resolve().parent / "corner_baseline_vs_opt_dark.png"

# (col, label, unit, display-scale) on (x,y,z,px,py,pz)
COORDS = [(0, "x", "mm", 1e3), (1, "y", "mm", 1e3), (2, "z", "mm", 1e3),
          (3, "px", "MeV/c", 1e-6), (4, "py", "MeV/c", 1e-6), (5, "pz", "GeV/c", 1e-9)]

# ---- presentation theme (matches nf_surrogate_architecture.tex / LPS gif) -----
BG, FG = "#000000", "#E8EAEE"
BL_DRIVE, BL_WIT = "#5B8FF9", "#2EE6D6"   # baseline: azure / bright cyan  (cool)
OP_DRIVE, OP_WIT = "#FF5D62", "#FFB13C"   # optimized: coral-red / amber   (warm)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG,
})


def load_series():
    bl = np.load(CLOUD_DIR / "clouds_baseline.npz")
    op = np.load(CLOUD_DIR / "clouds_bptt_gc_goal200um.npz")
    return [
        ("baseline driver", BL_DRIVE, bl["bmad_drive"]),
        ("baseline witness", BL_WIT, bl["bmad_witness"]),
        ("MBRL driver", OP_DRIVE, op["bmad_drive_0"]),
        ("MBRL witness", OP_WIT, op["bmad_witness_0"]),
    ]


def shared_lims(series):
    lims = {}
    for ci, _, _, sc in COORDS:
        vals = np.concatenate([a[:, ci] * sc for _, _, a in series])
        lo, hi = np.percentile(vals, [0.5, 99.5])
        pad = 0.05 * (hi - lo + 1e-12)
        lims[ci] = (lo - pad, hi + pad)
    return lims


def main():
    series = load_series()
    lims = shared_lims(series)
    nc = len(COORDS)
    fig, axes = plt.subplots(nc, nc, figsize=(16.5, 16.5))

    for r in range(nc):
        ci_r, lab_r, u_r, sc_r = COORDS[r]
        for c in range(nc):
            ci_c, lab_c, u_c, sc_c = COORDS[c]
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if c == r:                                   # diagonal: 1-D step density
                bins = np.linspace(*lims[ci_r], 46)
                for _, col, a in series:
                    ax.hist(a[:, ci_r] * sc_r, bins=bins, color=col, histtype="step",
                            lw=2.2, density=True)
                ax.set_xlim(*lims[ci_r])
                ax.set_yticks([])
            else:                                        # lower triangle: col_c (x) vs col_r (y)
                for _, col, a in series:
                    ax.scatter(a[:, ci_c] * sc_c, a[:, ci_r] * sc_r, s=3, alpha=0.35,
                               color=col, edgecolors="none")
                ax.set_xlim(*lims[ci_c])
                ax.set_ylim(*lims[ci_r])
                ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.xaxis.set_major_locator(MaxNLocator(4))
            for sp in ax.spines.values():
                sp.set_color(FG)
            ax.tick_params(labelsize=15, length=4)
            if r == nc - 1:
                ax.set_xlabel(f"{lab_c} [{u_c}]", fontsize=22, labelpad=8)
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(f"{lab_r} [{u_r}]", fontsize=22, labelpad=8)
            elif c != 0:
                ax.set_yticklabels([])

    handles = [plt.Line2D([], [], marker="o", ls="", ms=15, color=col, label=nm)
               for nm, col, _ in series]
    leg = fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.93),
                     fontsize=24, framealpha=0.0, handletextpad=0.4, borderpad=0.6)
    for t in leg.get_texts():
        t.set_color(FG)

    fig.suptitle("Full 6D phase space (both bunches, Bmad)\n"
                 "hand-tuned baseline  vs  MBRL 200 um set-point",
                 fontsize=27, fontweight="bold", color="#FFFFFF", y=0.985)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.92, bottom=0.055,
                        wspace=0.08, hspace=0.08)
    fig.savefig(OUT, dpi=100)
    print(f"wrote {OUT}  ({int(16.5*100)}x{int(16.5*100)})")


if __name__ == "__main__":
    main()
