#!/usr/bin/env python
"""Three-goal LPS panel for the DL4SCI poster (light theme) -- results section.

Side-by-side z-pz phase space at three commanded bunch separations (100/200/300 um),
Bmad tracking vs surrogate prediction overlaid, top z-density marginals, shared axes.
Data = the bptt_gc_sigmaz Bmad validation run (same clouds as lps_compare_sweep.gif):
  - Bmad:      results/rl/bptt_gc_sigmaz/bmad_validation/sample_*_PENT.h5 (split by weight)
  - surrogate: re-sampled from the combined-ft flow at the controller setpoints
  - spacing annotations: validate_goal*um.json (comparison.spacing_um)

Palette matches the poster (Stanford beamer theme) and is validated for CVD + contrast
on white (dataviz six checks): Bmad = cardinal #B01818, surrogate = skyblue #0098DB.
Font matches the poster body: Lato Light, with Lato Regular as "bold" (the gemini theme
maps BoldFont=Lato-Regular).

Usage (repo root):
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD/src MPLBACKEND=Agg conda run -n slac-rl \
    python results/presentation_figures/make_lps_goals_poster.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup

from twobunch_s2e_rl.surrogate import COORD_KEYS
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow

HERE = Path(__file__).resolve().parent                  # presentation_figures/
ROOT = HERE.parents[1]                                  # twobunch-s2e-rl/
VAL = ROOT / "results/rl/bptt_gc_sigmaz/bmad_validation"
SETP = ROOT / "results/rl/bptt_gc_sigmaz/setpoints"
CKPT = ROOT / "trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt"
GOALS = [100, 200, 300]
N_SURR = 2500                                           # surrogate particles per bunch

SCALE = np.array([1e3, 1e3, 1e3, 1e-6, 1e-6, 1e-9])    # -> mm / MeV/c / GeV/c
BMAD, SURR = "#B01818", "#0098DB"                       # cardinal / skyblue (Stanford theme)
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


def split_h5(path):
    P = ParticleGroup(str(path))
    w = np.unique(P.weight)
    st = lambda pg: np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float64) * SCALE
    return st(P[P.weight == w[-1]]), st(P[P.weight == w[0]])


@torch.no_grad()
def surr_clouds(flow, knorm, dev):
    kt = torch.tensor(knorm, dtype=torch.float32, device=dev).unsqueeze(0)
    return (flow.sample_bunch(kt, 0, N_SURR)[0].cpu().numpy() * SCALE,
            flow.sample_bunch(kt, 1, N_SURR)[0].cpu().numpy() * SCALE)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    flow = TwoBunchFlow.load_from_checkpoint(str(CKPT), map_location=dev).eval().to(dev)
    rng = np.random.default_rng(0)

    frames = []
    for g in GOALS:
        bd, bw = split_h5(VAL / f"sample_{g:05d}_PENT.h5")
        sp = json.load(open(SETP / f"setpoints_goal{g}um.json"))
        kn = np.array(list(sp["knob_setpoints_normalized"].values()), np.float32)
        sd, sw = surr_clouds(flow, kn, dev)
        spacing = json.load(open(VAL / f"validate_goal{g}um.json"))["comparison"]["spacing_um"]["bmad"]
        frames.append({"goal": g, "bmad": np.vstack([bd, bw]), "surr": np.vstack([sd, sw]),
                       "cen": (bd[:, 2].mean(), bw[:, 2].mean()), "spacing": spacing})

    # shared limits (1-99 pct over everything, small pad) so panels are directly comparable
    allc = np.vstack([fr[k] for fr in frames for k in ("bmad", "surr")])
    lims = []
    for col in (2, 5):
        lo, hi = np.percentile(allc[:, col], [1, 99])
        pad = 0.06 * (hi - lo)
        lims.append((lo - pad, hi + pad))
    zlim, pzlim = lims

    fig = plt.figure(figsize=(14.2, 6.1))
    gs = fig.add_gridspec(2, 3, height_ratios=(1, 4.2), left=0.082, right=0.985,
                          bottom=0.12, top=0.82, wspace=0.10, hspace=0.05)
    zbins = np.linspace(*zlim, 70)

    for i, fr in enumerate(frames):
        ax = fig.add_subplot(gs[1, i])
        axt = fig.add_subplot(gs[0, i], sharex=ax)

        for key, col, lab in (("bmad", BMAD, "Bmad (truth)"), ("surr", SURR, "surrogate")):
            c = fr[key]
            c = c[rng.choice(len(c), min(len(c), 5000), replace=False)]
            ax.scatter(c[:, 2], c[:, 5], s=3, alpha=0.25, color=col, edgecolors="none",
                       label=lab, rasterized=True)
            axt.hist(fr[key][:, 2], bins=zbins, density=True, histtype="step", lw=2.0, color=col)

        for cz in fr["cen"]:                                # Bmad per-bunch z-centroids
            ax.axvline(cz, color=MUTED, ls="--", lw=1.4, alpha=0.9, zorder=1)

        ax.set_xlim(*zlim); ax.set_ylim(*pzlim)
        ax.set_xlabel("z [mm]", fontsize=24)
        ax.tick_params(labelsize=20)
        axt.set_yticks([]); axt.tick_params(labelbottom=False)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False); axt.spines[sp_].set_visible(False)
        axt.spines["left"].set_visible(False)

        axt.set_title(f"goal {fr['goal']} µm", fontsize=28, fontweight="regular", pad=36)
        axt.text(0.5, 1.14, f"Bmad spacing {fr['spacing']:.0f} µm", transform=axt.transAxes,
                 fontsize=21, color=MUTED, ha="center", va="bottom")
        if i == 0:
            ax.set_ylabel("pz [GeV/c]", fontsize=24)
            leg = ax.legend(fontsize=20, markerscale=4.5, loc="upper left",
                            framealpha=0.9, edgecolor="none", handletextpad=0.3, borderaxespad=0.2)
            for t, c in zip(leg.get_texts(), (BMAD, SURR)):
                t.set_color(c)                              # direct-labeled legend (relief for WARN)
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)
        else:
            ax.tick_params(labelleft=False)

    for ext, dpi in (("png", 220), ("pdf", 220)):
        fig.savefig(HERE / f"lps_gc_goals_poster.{ext}", dpi=dpi)
    print(f"wrote lps_gc_goals_poster.png/.pdf to {HERE}")


if __name__ == "__main__":
    main()
