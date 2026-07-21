#!/usr/bin/env python
"""Transverse-constraint panel for the DL4SCI poster (light theme) -- results section.

Companion to make_lps_goals_poster.py: while the policy sweeps the commanded bunch
separation (100-300 um), the transverse beam quality stays pinned. Three small multiples
of Bmad-tracked values vs the ACHIEVED (Bmad) separation:
  1. witness norm. emittance ex/ey, with the manually-optimized-baseline beam's values as
     dotted refs (computed from data/tightbox_v2_full/sample_06000_PENT.h5 with the same
     90%-core pmd standard as getBeamSpecs / validate_bmad.py)
  2. transmission, drive + witness
  3. drive-witness transverse centroid offset, with the manual-baseline 26.3 um reference
     (Ryan's updated slide value; the older in-repo plot_validation.py reference was 47 um,
     the difference is a simulation-settings mismatch)

Data: results/rl/bptt_gc_sigmaz/bmad_validation/validate_goal*um.json (Bmad values only --
this figure is about the physics constraint, not surrogate fidelity).

Poster styling matches make_lps_goals_poster.py: Lato (Light body / Regular titles),
Cardinal red is the highlight (these are Bmad numbers, matching red = Bmad in the LPS
figure); drive is de-emphasized in Stanford coolgray, identity carried by direct labels.

Usage (repo root):
  PYTHONPATH=$PWD/src MPLBACKEND=Agg conda run -n slac-rl \
    python results/presentation_figures/make_transverse_goals_poster.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup

HERE = Path(__file__).resolve().parent                  # presentation_figures/
ROOT = HERE.parents[1]                                  # twobunch-s2e-rl/
VAL = ROOT / "results/rl/bptt_gc_sigmaz/bmad_validation"
BASELINE_H5 = ROOT / "data/tightbox_v2_full/sample_06000_PENT.h5"
GOALS = [100, 150, 200, 250, 300]
BASELINE_OFFSET_UM = 26.3                               # manual-baseline ref (updated slide value)

DRIVE, WITNESS = "#4D4F53", "#B01818"                   # coolgray (context) / cardinal (highlight)
INK, MUTED = "#1a1a1a", "#666666"

from matplotlib import font_manager
for _f in ("Lato-Light.ttf", "Lato-Regular.ttf", "Lato-Bold.ttf"):
    font_manager.fontManager.addfont(f"/usr/share/fonts/truetype/lato/{_f}")

plt.rcParams.update({
    "font.family": "Lato",
    "font.weight": "light",                             # poster body = Lato Light
    "axes.labelweight": "light",
    "figure.facecolor": "white", "savefig.facecolor": "white", "axes.facecolor": "white",
    "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": MUTED,
    "xtick.color": INK, "ytick.color": INK,
})

FS_TITLE, FS_LABEL, FS_TICK, FS_ANN = 23, 22, 19, 19


def baseline_emit():
    """Manual-baseline witness ex/ey [um-rad], 90%-core pmd standard (= getBeamSpecs)."""
    P = ParticleGroup(str(BASELINE_H5))
    w = np.unique(P.weight)
    wit = P[P.weight == w[0]]
    return (wit.twiss(plane="x", fraction=0.9)["norm_emit_x"] * 1e6,
            wit.twiss(plane="y", fraction=0.9)["norm_emit_y"] * 1e6)


def main():
    recs = [json.load(open(VAL / f"validate_goal{g}um.json")) for g in GOALS]
    bm = lambda k: np.array([r["comparison"][k]["bmad"] for r in recs])
    bx, by = baseline_emit()
    x = bm("spacing_um")                                # achieved (Bmad) separation

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.9))
    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.21, top=0.87, wspace=0.30)

    def line(ax, y, color, ls, label):
        ax.plot(x, y, ls, color=color, lw=2.5, marker="o", ms=8, clip_on=False)
        ax.annotate(label, (x[-1], y[-1]), xytext=(10, 0), textcoords="offset points",
                    fontsize=FS_ANN, color=color, va="center")

    for ax in axes:
        ax.set_xlim(145, 315)
        ax.set_xticks([150, 200, 250, 300])
        ax.set_xlabel("achieved separation [µm]", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ---- 1: witness emittance vs golden baseline ---------------------------------
    ax = axes[0]
    line(ax, bm("witness_emit_x_um_rad"), WITNESS, "-", "$\\epsilon_x$")
    line(ax, bm("witness_emit_y_um_rad"), WITNESS, "--", "$\\epsilon_y$")
    for b in (bx, by):
        ax.axhline(b, color=MUTED, ls=":", lw=1.8)
    ax.annotate("manual baseline", (147, max(bx, by)), xytext=(0, 6), textcoords="offset points",
                fontsize=FS_ANN, color=MUTED, va="bottom")
    ax.set_ylim(0, 13.5)
    ax.set_title("witness emittance [µm·rad]", fontsize=FS_TITLE,
                 fontweight="regular", pad=12)

    # ---- 2: transmission (lines nearly coincide: 0.995 vs 0.991 -> dash + stagger) --
    ax = axes[1]
    for key, col, ls, dy in (("T_drive", DRIVE, "-", -12), ("T_witness", WITNESS, "--", -34)):
        y = bm(key)
        ax.plot(x, y, ls, color=col, lw=2.5, marker="o", ms=8, clip_on=False)
        ax.annotate(f"{key[2:]} {y[-1]:.3f}", (float(np.median(x)), np.median(y)),
                    xytext=(0, dy), textcoords="offset points", fontsize=FS_ANN,
                    color=col, ha="center", va="top")
    ax.set_ylim(0, 1.05)
    ax.set_title("transmission", fontsize=FS_TITLE, fontweight="regular", pad=12)

    # ---- 3: drive-witness transverse offset ----------------------------------------
    ax = axes[2]
    ax.plot(x, bm("transverse_offset_um"), "-", color=WITNESS, lw=2.5, marker="o", ms=8,
            clip_on=False)
    ax.axhline(BASELINE_OFFSET_UM, color=MUTED, ls=":", lw=1.8)
    ax.annotate(f"manual baseline {BASELINE_OFFSET_UM:.1f} µm", (147, BASELINE_OFFSET_UM),
                xytext=(0, -8), textcoords="offset points",
                fontsize=FS_ANN, color=MUTED, va="top")
    ax.set_ylim(0, 30)
    ax.set_title("drive–witness offset [µm]", fontsize=FS_TITLE,
                 fontweight="regular", pad=12)

    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"transverse_goals_poster.{ext}", dpi=220)
    print(f"wrote transverse_goals_poster.png/.pdf to {HERE}")


if __name__ == "__main__":
    main()
