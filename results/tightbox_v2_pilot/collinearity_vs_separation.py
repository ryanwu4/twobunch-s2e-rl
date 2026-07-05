"""Is collinearity (offset + angle) separation-dependent, or flat across bunch spacing?

Answers: does the tightened box deliver better transverse offset / angular misalignment near
the golden bunch separation, or uniformly across the swept separation range? And if the floor
is flat in spacing, which knob actually sets it?

Bins the witness-viable draws by bunch spacing and plots offset/angle vs spacing (flatness
test), then vs L3EnergyOffset (the dominant dispersion lever) to show the real dependence.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/tightbox_v2_pilot/collinearity_vs_separation.py [subdir]
    (default: tightbox_v2_pilot)
Outputs: collinearity_vs_separation.png beside this script (results/tightbox_v2_pilot/)
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from twobunch_s2e_rl.analysis_io import load, derived

BLUE, ORANGE, GREEN = "#4c72b0", "#dd8452", "#55a868"


def _binned_median(x, y, nb=8):
    edges = np.quantile(x, np.linspace(0, 1, nb + 1))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    cx, cy = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 3:
            cx.append(np.median(x[m])); cy.append(np.median(y[m]))
    return np.array(cx), np.array(cy)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subdir", nargs="?", default="tightbox_v2_pilot")
    args = ap.parse_args()

    df = load(args.subdir)
    lhs = df[(df["success"] == True) & (df["is_baseline_repeat"] == False)].copy()
    base = df[df["is_baseline_repeat"] == True].copy()
    q, qb = derived(lhs), derived(base)
    v = np.isfinite(q["witness_BMAG_x"])
    sp = np.abs(q["spacing"][v]); off = q["offset"][v]; ang = q["angle"][v]
    l3 = lhs["L3EnergyOffset"].to_numpy(float)[v] * 1e-6  # MeV-ish scale (raw eV -> MeV)
    g_off = np.nanmedian(qb["offset"]); g_ang = np.nanmedian(qb["angle"])
    g_sp = np.nanmedian(np.abs(qb["spacing"]))
    m = np.isfinite(sp) & np.isfinite(off) & np.isfinite(ang) & np.isfinite(l3)
    sp, off, ang, l3 = sp[m], off[m], ang[m], l3[m]

    figdir = Path(__file__).resolve().parent
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    panels = [
        (ax[0, 0], sp, off, "bunch spacing [um]", "transverse offset [um]", g_sp, g_off, True),
        (ax[0, 1], sp, ang, "bunch spacing [um]", "angular misalignment [urad]", g_sp, g_ang, True),
        (ax[1, 0], l3, off, "L3EnergyOffset [MeV]", "transverse offset [um]", None, g_off, True),
        (ax[1, 1], l3, ang, "L3EnergyOffset [MeV]", "angular misalignment [urad]", None, g_ang, True),
    ]
    for a, x, y, xl, yl, gx, gy, logy in panels:
        a.scatter(x, y, s=10, alpha=0.35, color=BLUE)
        cx, cy = _binned_median(x, y)
        a.plot(cx, cy, "-o", color=ORANGE, lw=2, ms=5, label="binned median")
        r = np.corrcoef(x, y)[0, 1]
        if gy is not None:
            a.axhline(gy, color=GREEN, ls="--", lw=1.4, label=f"golden {gy:.0f}")
        if gx is not None:
            a.axvline(gx, color=GREEN, ls=":", lw=1.2, label=f"golden sep {gx:.0f}")
        if logy:
            a.set_yscale("log")
        a.set_xlabel(xl); a.set_ylabel(yl)
        a.set_title(f"{yl.split(' [')[0]} vs {xl.split(' [')[0]}   (corr={r:+.2f})", fontsize=10)
        a.legend(fontsize=8, loc="best")

    fig.suptitle(f"{args.subdir}: collinearity is flat in separation, set by the dispersion "
                 f"lever (L3EnergyOffset)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figdir / "collinearity_vs_separation.png", dpi=130)
    plt.close(fig)

    print(f"corr(spacing, offset)={np.corrcoef(sp,off)[0,1]:+.3f}  "
          f"corr(spacing, angle)={np.corrcoef(sp,ang)[0,1]:+.3f}")
    print(f"corr(L3Eoff, offset)={np.corrcoef(l3,off)[0,1]:+.3f}  "
          f"corr(L3Eoff, angle)={np.corrcoef(l3,ang)[0,1]:+.3f}")
    print(f"wrote {figdir/'collinearity_vs_separation.png'}")


if __name__ == "__main__":
    main()
