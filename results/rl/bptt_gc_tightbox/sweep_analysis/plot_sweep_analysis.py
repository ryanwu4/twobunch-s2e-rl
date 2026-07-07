"""Full-sweep (100-300 um) analysis for the tightbox goal-conditioned controller:
  - agreement_vs_goal.png : surrogate vs Bmad for spacing / offset / angle vs target
  - spacing_tracking.png  : achieved spacing (surrogate & Bmad) vs commanded, with y=x + golden band
  - knob_ranges.png       : per-knob policy median (denormalized) vs target for ALL 26 knobs, with the
                            tightbox clamp band, the full sweep range, the baseline (2024-10-14), and
                            the 200 um set-point star  (cf. results/presentation_figures/gc_knob_setpoints.png)

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/rl/bptt_gc_tightbox/sweep_analysis/plot_sweep_analysis.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from twobunch_s2e_rl.datagen.paths import repo_root
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "bmad_validation"
GOALS = [100, 150, 200, 250, 300]
BLUE, ORANGE, GREEN, GREY, RED = "#4c72b0", "#dd8452", "#55a868", "#8c8c8c", "#e24a4a"


def _unit_scale(k):
    if k.endswith("PhaseSet"):     return "deg", 1.0
    if k.endswith("EnergyOffset"): return "MeV", 1e-6
    if k.endswith("Offset"):       return "mm", 1e3      # movers: m -> mm
    return "kG", 1.0                                     # sextupole strengths, FF quads, kickers


def _val(g, metric):
    c = json.load(open(VAL / f"validate_goal{g}um.json"))["comparison"][metric]
    return c["surrogate"], c["bmad"]


def main():
    keys, ulo, uhi, _ = resolve_sweep_set("tightbox+expanded")
    ulo, uhi = np.array(ulo), np.array(uhi)
    _, tlo, thi, base = resolve_sweep_set("tightbox")           # clamp box + golden baseline
    tlo, thi = np.array(tlo), np.array(thi)
    kmed = np.array([json.load(open(repo_root() / f"logs/bptt_gc_tightbox/eval_best_goal{g}um.json"))
                     ["knob_median"] for g in GOALS])            # (nG, 26) normalized
    kmed_phys = ulo + kmed * (uhi - ulo)                         # -> physical
    sp200 = json.load(open(VAL.parent / "setpoints" / "setpoints_goal200um.json"))["knob_setpoints_physical"]

    # ---- FIG 1: surrogate vs Bmad agreement vs goal --------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for a, (m, lab, unit) in zip(ax, [("spacing_um", "bunch spacing", "µm"),
                                      ("transverse_offset_um", "transverse offset", "µm"),
                                      ("angular_misalignment_urad", "angular misalign", "µrad")]):
        s = [_val(g, m)[0] for g in GOALS]; b = [_val(g, m)[1] for g in GOALS]
        a.plot(GOALS, s, "-o", color=BLUE, label="surrogate")
        a.plot(GOALS, b, "-o", color=ORANGE, label="Bmad")
        if m == "spacing_um":
            a.plot(GOALS, GOALS, "--", color=GREEN, label="target (y=x)")
        if m == "transverse_offset_um":
            a.axhline(47, color=GREY, ls=":", label="golden 47 µm")
        a.set_xlabel("target spacing [µm]"); a.set_ylabel(f"{lab} [{unit}]")
        a.set_title(lab, fontsize=11); a.legend(fontsize=8)
    fig.suptitle("Surrogate vs Bmad across the spacing sweep (tightbox)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(HERE / "agreement_vs_goal.png", dpi=130); plt.close(fig)

    # ---- FIG 2: achieved vs commanded spacing --------------------------------------------------
    fig, a = plt.subplots(figsize=(6.5, 6))
    s = [_val(g, "spacing_um")[0] for g in GOALS]; b = [_val(g, "spacing_um")[1] for g in GOALS]
    a.plot([90, 320], [90, 320], "--", color=GREEN, label="commanded = achieved")
    a.plot(GOALS, s, "-o", color=BLUE, label="surrogate")
    a.plot(GOALS, b, "-o", color=ORANGE, label="Bmad")
    a.fill_between([90, 320], [90 - 30, 320 - 30], [90 + 30, 320 + 30], color=GREY, alpha=0.12,
                   label="±30 µm")
    a.set_xlabel("commanded spacing [µm]"); a.set_ylabel("achieved spacing [µm]")
    a.set_title("Spacing tracking (tightbox): Bmad ~+30 µm systematic over-shoot", fontsize=11)
    a.legend(fontsize=9); a.set_aspect("equal")
    fig.tight_layout(); fig.savefig(HERE / "spacing_tracking.png", dpi=130); plt.close(fig)

    # ---- FIG 3: knob ranges across the sweep (all 26) ------------------------------------------
    ncol = 5; nrow = int(np.ceil(len(keys) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 2.7 * nrow))
    for idx, (k, ax) in enumerate(zip(keys, axes.ravel())):
        unit, sc = _unit_scale(k)
        ax.axhspan(ulo[idx] * sc, uhi[idx] * sc, color=GREY, alpha=0.10)          # full sweep range
        ax.axhspan(tlo[idx] * sc, thi[idx] * sc, color=BLUE, alpha=0.15)          # tightbox clamp
        ax.plot(GOALS, kmed_phys[:, idx] * sc, "-o", color=BLUE, ms=4, lw=1.6)    # policy median
        ax.axhline(base[k] * sc, color=GREY, ls="--", lw=1.3)                     # baseline
        ax.plot(200, sp200[k] * sc, "*", color=RED, ms=15, mec="k", mew=0.5, zorder=6)  # 200um setpoint
        ax.set_title(k, fontsize=9.5); ax.set_ylabel(unit, fontsize=8)
        ax.set_ylim(ulo[idx] * sc, uhi[idx] * sc); ax.tick_params(labelsize=7)
        if idx >= len(keys) - ncol:
            ax.set_xlabel("target spacing [µm]", fontsize=8)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", label="policy median (256 rollouts)"),
               plt.Line2D([], [], color=GREY, ls="--", label="baseline (2024-10-14)"),
               plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.15, label="tightbox clamp"),
               plt.Rectangle((0, 0), 1, 1, color=GREY, alpha=0.10, label="full sweep range"),
               plt.Line2D([], [], color=RED, marker="*", ls="", label="200 µm set-point")]
    fig.suptitle("Goal-conditioned knob set-points across the spacing sweep (tightbox, all 26 knobs)",
                 fontsize=13, y=0.998)
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10, bbox_to_anchor=(0.5, 0.965))
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(HERE / "knob_ranges.png", dpi=130); plt.close(fig)
    print(f"wrote agreement_vs_goal.png, spacing_tracking.png, knob_ranges.png to {HERE}")


if __name__ == "__main__":
    main()
