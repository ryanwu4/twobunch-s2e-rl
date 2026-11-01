"""Drive<->witness relative ('collision quality') quantities for the two-bunch sweep.

Reads results/tables/dataset.pkl. All four are defined only where the witness is viable
(bunchCount==2). Definitions match UTILITY_quickstart.getBeamSpecs:
  bunch spacing        = PWitness_zCentroid - PDrive_zCentroid           (signed, m)
  transverse offset    = |median (x,y)_drive - median (x,y)_witness|     (>=0, m)
  energy difference    = PDrive_median_energy - PWitness_median_energy   (signed, eV)
  angular misalignment = |median (xp,yp)_drive - median (xp,yp)_witness| (>=0, rad)

Produces, beside this script, fig5_twobunch_quality_PENT.png and fig6_..._evolution.png,
and prints per-point summary stats.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/dataset_overview/twobunch_quality_plots.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from twobunch_s2e_rl.datagen.paths import tables_dir

ART = tables_dir()                          # dataset.pkl lives here
FIG = Path(__file__).resolve().parent       # write figures beside this script
POINTS = ["BEGBC20", "MFFF", "PENT"]
df = pd.read_pickle(ART / "dataset.pkl")


def derive(pt):
    """Return dict of the 4 relative quantities (display units) for treaty point pt.
    NaN where the witness is not viable."""
    g = lambda s: df[f"{pt}__{s}"].to_numpy(dtype=float)
    spacing = g("bunchSpacing") * 1e6  # um
    offset = g("transverseCentroidOffset") * 1e6  # um
    de = (g("PDrive_median_energy") - g("PWitness_median_energy")) * 1e-6  # MeV
    ang = np.sqrt((g("PDrive_median_xp") - g("PWitness_median_xp"))**2 +
                  (g("PDrive_median_yp") - g("PWitness_median_yp"))**2) * 1e6  # urad
    return {"spacing": spacing, "offset": offset, "de": de, "ang": ang}


# (key, title, unit, signed?  -> signed=False uses log axis)
QDEF = [
    ("spacing", "bunch spacing  (witness − drive)", "µm", True),
    ("de",      "energy difference  (drive − witness)", "MeV", True),
    ("offset",  "transverse centroid offset", "µm", False),
    ("ang",     "angular misalignment", "µrad", False),
]


def fig_pent():
    q = derive("PENT")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (k, title, unit, signed) in zip(axes.ravel(), QDEF):
        v = q[k]
        v = v[np.isfinite(v)]
        if signed:
            ax.hist(v, bins=50, color="#4c72b0", edgecolor="white", lw=0.3)
            ax.axvline(0, color="k", lw=1, ls=":")
        else:
            vp = v[v > 0]
            bins = np.logspace(np.log10(vp.min()), np.log10(vp.max()), 50)
            ax.hist(vp, bins=bins, color="#4c72b0", edgecolor="white", lw=0.3)
            ax.set_xscale("log")
        med = np.median(v)
        ax.axvline(med, color="#dd8452", lw=2, label=f"median = {med:.3g} {unit}")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel(unit, fontsize=10)
        ax.set_ylabel("count", fontsize=10)
        ax.legend(fontsize=9)
    n = int(np.isfinite(q["spacing"]).sum())
    fig.suptitle(f"Two-bunch relative quantities at PENT — witness-viable subset (n={n})",
                 fontsize=13)
    fig.tight_layout()
    p = FIG / "fig5_twobunch_quality_PENT.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def fig_evolution():
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    xpos = np.arange(len(POINTS))
    for ax, (k, title, unit, signed) in zip(axes, QDEF):
        meds, q1, q3 = [], [], []
        for pt in POINTS:
            v = derive(pt)[k]
            v = v[np.isfinite(v)]
            if not signed:
                v = v[v > 0]
            meds.append(np.median(v))
            q1.append(np.percentile(v, 16))
            q3.append(np.percentile(v, 84))
        ax.plot(xpos, meds, "o-", color="#4c72b0")
        ax.fill_between(xpos, q1, q3, color="#4c72b0", alpha=0.2)
        if not signed:
            ax.set_yscale("log")
        else:
            ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.set_xticks(xpos)
        ax.set_xticklabels(POINTS)
        ax.set_title(f"{title}\n[{unit}]", fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Two-bunch relative quantities along the line — median (band = 16–84%), witness-viable subset",
                 fontsize=13)
    fig.tight_layout()
    p = FIG / "fig6_twobunch_quality_evolution.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


def stats():
    print("\n=== TWO-BUNCH RELATIVE QUANTITIES (witness-viable subset) ===")
    print(f"{'quantity':32s}{'point':9s}{'min':>11s}{'16%':>11s}{'median':>11s}{'84%':>11s}{'max':>11s}")
    for k, title, unit, _ in QDEF:
        for pt in POINTS:
            v = derive(pt)[k]
            v = v[np.isfinite(v)]
            print(f"{(title.split('  ')[0]+' ['+unit+']'):32s}{pt:9s}"
                  f"{v.min():11.3g}{np.percentile(v,16):11.3g}{np.median(v):11.3g}"
                  f"{np.percentile(v,84):11.3g}{v.max():11.3g}")


if __name__ == "__main__":
    stats()
    fig_pent()
    fig_evolution()
