#!/usr/bin/env python
"""Dark, presentation-scale re-render of the goal-conditioned knob set-points figure.

Data: logs/bptt_gc/eval_best_goal{100..300}um.json  ("knob_median", normalized [0,1],
in PARAM_KEYS order). Physical set-point = lo + knob * (hi - lo); energy offsets eV->MeV.
Sweep bounds + 2024-10-14 baseline are the SWEEP_PARAMS table (datagen/sweep_params.py).

Run from the repo root:  python results/presentation_figures/make_gc_knob_setpoints.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]          # twobunch-s2e-rl/
OUT = Path(__file__).resolve().parent               # .../presentation_figures
GOALS = [100, 150, 200, 250, 300]
STAR_GOAL = 200

# key -> (lo, hi, baseline) in raw units (datagen/sweep_params.SWEEP_PARAMS)
SWEEP = {
    "L1PhaseSet":     (-22.8,          -17.8,          -20.2889213421),
    "L2PhaseSet":     (-38.0,          -34.0,          -35.5447801603),
    "L1EnergyOffset": (-2.1e6,          2.1e6,          0.0),
    "L2EnergyOffset": (-48.5e6,         34.9e6,        -6.817565553821569e6),
    "L3EnergyOffset": (-43.3e6,         66.7e6,         1.1703527144314773e7),
    "S1ELkG":         (0.0,             2590.0,         2089.4846449653),
    "S2ELkG":         (-5931.561690604, -1977.187230201, -3954.374460403),
    "S3ELkG":         (-2625.0,         0.0,           -1087.9568814486),
}
KEYS = list(SWEEP)
# per-knob display scale + unit
UNITS = {
    "L1PhaseSet": (1.0, "deg"), "L2PhaseSet": (1.0, "deg"),
    "L1EnergyOffset": (1e-6, "MeV"), "L2EnergyOffset": (1e-6, "MeV"),
    "L3EnergyOffset": (1e-6, "MeV"),
    "S1ELkG": (1.0, "kG"), "S2ELkG": (1.0, "kG"), "S3ELkG": (1.0, "kG"),
}

# ---- load median commanded knobs per goal -----------------------------------
knob = {}  # knob[g] = list of 8 normalized medians (KEYS order)
for g in GOALS:
    d = json.loads((ROOT / f"logs/bptt_gc/eval_best_goal{g}um.json").read_text())
    knob[g] = d["knob_median"]


def setpoint(key, g):
    lo, hi, _ = SWEEP[key]
    i = KEYS.index(key)
    return lo + knob[g][i] * (hi - lo)


# ---- dark presentation style -------------------------------------------------
BG = "#000000"
FG = "#EAECEF"          # near-white text / ticks
SPINE = "#5A5E66"       # muted frame
BLUE = "#5BA8FF"        # policy median
BASE = "#AEB4BE"        # baseline dashed
BAND = "#23272E"        # LHS sweep-range panel
STAR = "#FF5D5D"        # 200 um set-point

# prefer Lato (a clean presentation face) if available; fall back to DejaVu Sans
family = "DejaVu Sans"
for f in ("/usr/share/fonts/truetype/lato/Lato-Regular.ttf",
          "/usr/share/fonts/truetype/lato/Lato-Bold.ttf"):
    if Path(f).exists():
        font_manager.fontManager.addfont(f)
        family = "Lato"

plt.rcParams.update({
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "font.family": family,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": SPINE,
    "xtick.color": FG, "ytick.color": FG,
    "axes.titlesize": 21, "axes.labelsize": 17,
    "xtick.labelsize": 14, "ytick.labelsize": 14,
    "axes.linewidth": 1.2, "axes.grid": True,
    "grid.color": FG, "grid.alpha": 0.12, "grid.linewidth": 0.8,
})

fig, axes = plt.subplots(2, 4, figsize=(22, 10.6))
for ax, key in zip(axes.flat, KEYS):
    scale, unit = UNITS[key]
    lo, hi, base = (v * scale for v in SWEEP[key])
    y = [setpoint(key, g) * scale for g in GOALS]

    ax.set_axisbelow(True)
    ax.axhspan(lo, hi, facecolor=BAND, edgecolor="none", zorder=0)      # sweep range
    ax.axhline(base, ls="--", color=BASE, lw=2.0, zorder=1)             # baseline
    ax.plot(GOALS, y, "-o", color=BLUE, lw=3.2, ms=10,
            mfc=BLUE, mec=BG, mew=1.0, zorder=3)                        # policy median
    yi = y[GOALS.index(STAR_GOAL)]
    ax.plot(STAR_GOAL, yi, marker="*", ms=30, color=STAR,
            mec="white", mew=1.0, ls="none", zorder=5)                  # 200 um star

    span = hi - lo
    ax.set_ylim(lo - 0.09 * span, hi + 0.09 * span)
    ax.set_xticks(GOALS)
    ax.set_title(key, color=FG, fontweight="bold", pad=10)
    ax.set_ylabel(unit)
    for s in ax.spines.values():
        s.set_color(SPINE)

# x-labels on the bottom row only (less clutter, big and clean)
for ax in axes[-1]:
    ax.set_xlabel("target spacing  [µm]")

# ---- title + single figure legend -------------------------------------------
fig.suptitle("Goal-conditioned knob set-points across the spacing scan",
             color=FG, fontsize=27, fontweight="bold", y=0.985)

handles = [
    Line2D([0], [0], color=BLUE, lw=3.2, marker="o", ms=10, mec=BG, label="policy median  (256 rollouts)"),
    Line2D([0], [0], color=BASE, lw=2.0, ls="--", label="baseline  (2024-10-14)"),
    Patch(facecolor=BAND, edgecolor=SPINE, label="LHS sweep range"),
    Line2D([0], [0], color=STAR, marker="*", ms=20, mec="white", ls="none", label="200 µm set-point"),
]
leg = fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
                 bbox_to_anchor=(0.5, 0.935), fontsize=16, handletextpad=0.5,
                 columnspacing=2.2)
for t in leg.get_texts():
    t.set_color(FG)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.subplots_adjust(hspace=0.32, wspace=0.28)

png = OUT / "gc_knob_setpoints.png"
pdf = OUT / "gc_knob_setpoints.pdf"
fig.savefig(png, dpi=300)
fig.savefig(pdf)
print("wrote", png, "and", pdf)
