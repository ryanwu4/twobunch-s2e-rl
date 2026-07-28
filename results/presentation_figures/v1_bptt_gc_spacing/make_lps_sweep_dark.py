#!/usr/bin/env python
"""Goal-sweep LPS GIF in the dark presentation theme -- CONTROLLER v1 (spacing-only).

v1 = the `bptt_gc` run: 8-knob action space over the twobunch_flow_v4 surrogate, reward targets
bunch spacing with no bunch-length term. The v2 counterpart (26-knob `bptt_gc_sigmaz`, spacing +
sigma_z matched to golden) is ../v2_bptt_gc_sigmaz/make_lps_sweep_dark.py -- same layout and
palette, so the two GIFs can be shown back to back.

Reuses the CACHED Bmad clouds (results/rl/bptt_gc/openloop/clouds_goal*um.npz, the exact
bmad_drive_0/bmad_witness_0 plotted in the original lps_bptt_gc_sweep.gif) and the title values
(bunch spacing, T_w) from eval_bmad_bptt_gc_openloop.json -- so nothing needs re-running.

Layout mirrors src/twobunch_s2e_rl/rl/_eval_plots.py::_lps_figure (top z-density marginal + z-pz
scatter, dashed per-bunch z-centroids, shared axes across the sweep), restyled black-background
with the presentation palette (drive = blue, witness = rose-red).

Output: lps_sweep_dark_v1.gif + per-goal lps_goal<g>um_dark_v1.png (beside this script).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---- paths -------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent             # .../presentation_figures/v1_bptt_gc_spacing
# repo root by marker, not by index -- survives moving this script between figure subfolders
ROOT = next(p for p in OUT_DIR.parents if (p / "pyproject.toml").exists())   # twobunch-s2e-rl/
CLOUD_DIR = ROOT / "results/rl/bptt_gc/openloop"
EVAL_JSON = CLOUD_DIR / "eval_bmad.json"
GOALS = [100, 150, 200, 250, 300]
FPS = 1.5

# ---- presentation theme (matches nf_surrogate_architecture.tex palette) ------
BG = "#000000"
FG = "#E8EAEE"        # labels / ticks / spines
DIM = "#9AA0AC"       # secondary
DRIVE = "#6FB1FF"     # blue  (figure mlpc/basec family)
WITNESS = "#FF7A8A"   # rose-red (harmonizes with figure rose, reads as witness)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "axes.titlecolor": "#FFFFFF",
})


def load_frames():
    meta = json.load(open(EVAL_JSON))["by_goal"]
    frames = []
    for g in GOALS:
        d = np.load(CLOUD_DIR / f"clouds_goal{g}um.npz")
        gap = meta[f"goal{g}um"]["gap"]
        sp_um = gap["bunch_spacing"]["bmad_med"] * 1e6
        tw = gap["T_witness"]["bmad_med"]
        frames.append({
            "title": f"goal {g} um   |   Bmad spacing {sp_um:.0f} um   |   T_w {tw:.2f}",
            "drive": d["bmad_drive_0"], "witness": d["bmad_witness_0"],
        })
    return frames


def shared_lims(frames, col, scale):
    """[0.5, 99.5] pct over both bunches across all frames, 5% pad -- matches _coord_lims so the
    witness visibly slides relative to the drive as the goal sweeps."""
    vals = np.concatenate([fr[b][:, col] * scale for fr in frames for b in ("drive", "witness")])
    lo, hi = np.percentile(vals, [0.5, 99.5])
    pad = 0.05 * (hi - lo + 1e-12)
    return lo - pad, hi + pad


def render(fr, zlim, pzlim):
    fig, (axh, ax) = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True,
                                  gridspec_kw={"height_ratios": (1, 5), "hspace": 0.06})
    zbins = np.linspace(zlim[0], zlim[1], 80)
    for nm, col in (("drive", DRIVE), ("witness", WITNESS)):
        a = fr[nm]
        z = a[:, 2] * 1e3
        ax.scatter(z, a[:, 5] * 1e-9, s=5, alpha=0.5, color=col, edgecolors="none",
                   label=f"{nm} (n={len(a)})")
        ax.axvline(float(np.mean(z)), color=col, ls="--", lw=1.8, alpha=0.95)
        axh.hist(z, bins=zbins, color=col, alpha=0.6, density=True)

    axh.set_ylabel("density", fontsize=19)
    axh.set_yticks([])
    axh.tick_params(labelbottom=False)
    axh.set_title(fr["title"], fontsize=25, fontweight="bold", pad=14)
    for sp in ("top", "right"):
        axh.spines[sp].set_visible(False)
        ax.spines[sp].set_visible(False)

    ax.set_xlabel("z [mm]", fontsize=22)
    ax.set_ylabel("pz [GeV/c]", fontsize=22)
    ax.set_xlim(*zlim)
    ax.set_ylim(*pzlim)
    ax.tick_params(labelsize=17)
    leg = ax.legend(fontsize=19, markerscale=2.8, loc="upper left", framealpha=0.0)
    for t in leg.get_texts():
        t.set_color(FG)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.88, bottom=0.12)
    return fig


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def main():
    frames = load_frames()
    zlim = shared_lims(frames, 2, 1e3)
    pzlim = shared_lims(frames, 5, 1e-9)
    rgb = []
    for g, fr in zip(GOALS, frames):
        fig = render(fr, zlim, pzlim)
        rgb.append(fig_to_rgb(fig))
        fig.savefig(OUT_DIR / f"lps_goal{g}um_dark_v1.png", dpi=100)
        plt.close(fig)
    imgs = [Image.fromarray(f) for f in rgb]
    gif = OUT_DIR / "lps_sweep_dark_v1.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000.0 / FPS), loop=0)
    print(f"wrote {gif}  ({len(imgs)} frames, {imgs[0].size[0]}x{imgs[0].size[1]})")


if __name__ == "__main__":
    main()
