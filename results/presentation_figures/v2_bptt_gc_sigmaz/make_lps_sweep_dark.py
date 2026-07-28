#!/usr/bin/env python
"""Goal-sweep LPS GIF in the dark presentation theme -- CONTROLLER v2 (spacing + sigma_z).

v2 = the `bptt_gc_sigmaz` run: 26-knob action space over the twobunch_combined_ft surrogate,
knobs clamped to the tightbox box, reward = spacing target PLUS a bunch-length (sigma_z) target on
both bunches matched near their golden PENT values. v1 (`bptt_gc`, 8-knob twobunch_flow_v4,
spacing-only, no length control) is ../v1_bptt_gc_spacing/make_lps_sweep_dark.py -- same layout and
palette so the two GIFs can be shown back to back; the visible difference is that the v2 witness
stays short across the whole sweep instead of stretching as spacing shrinks.

Data = the bptt_gc_sigmaz Bmad validation run (the same clouds behind lps_goals_poster_v2.png),
so nothing needs re-running:
  - clouds : results/rl/bptt_gc_sigmaz/bmad_validation/sample_<g>_PENT.h5, 100k-macroparticle
             Bmad/Tao tracks (CSR + transverse wakes on), split into drive/witness by the two
             unique per-particle weights and subsampled to --n-particles per bunch.
  - titles : validate_goal<g>um.json (comparison.*.bmad -- spacing, T_witness, per-bunch sigma_z)
  - sigma_z reward targets in the subtitle are read from configs/rl/bptt_gc_sigmaz.yaml so the
    annotation cannot drift from what the controller actually optimized.

Layout mirrors src/twobunch_s2e_rl/rl/_eval_plots.py::_lps_figure (top z-density marginal + z-pz
scatter, dashed per-bunch z-centroids, shared axes across the sweep), restyled black-background
with the presentation palette (drive = blue, witness = rose-red).

Run (no GPU needed -- Bmad clouds only):
  cd twobunch-s2e-rl && MPLBACKEND=Agg \
    python results/presentation_figures/v2_bptt_gc_sigmaz/make_lps_sweep_dark.py

Output: lps_sweep_dark_v2.gif + per-goal lps_goal<g>um_dark_v2.png (beside this script).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pmd_beamphysics import ParticleGroup

# ---- paths -------------------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent             # .../presentation_figures/v2_bptt_gc_sigmaz
# repo root by marker, not by index -- survives moving this script between figure subfolders
ROOT = next(p for p in OUT_DIR.parents if (p / "pyproject.toml").exists())   # twobunch-s2e-rl/
VAL_DIR = ROOT / "results/rl/bptt_gc_sigmaz/bmad_validation"
CFG = ROOT / "configs/rl/bptt_gc_sigmaz.yaml"
GOALS = [100, 150, 200, 250, 300]
FPS = 1.5

# ---- presentation theme (identical to the v1 script, so the GIFs are interchangeable) --------
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


def sigma_z_targets():
    """(drive, witness) sigma_z reward targets [um] from the controller config."""
    objs = yaml.safe_load(open(CFG))["params"]["diff_env"]["objectives"]
    # float(): PyYAML leaves dot-less exponentials like `2e-5` as strings (YAML 1.1 float rule)
    tgt = {o["key"]: float(o["target"]) for o in objs if o["key"].endswith("sigma_z")}
    return tgt["drive_sigma_z"] * 1e6, tgt["witness_sigma_z"] * 1e6


def split_h5(path, n, rng):
    """PENT beam -> (drive, witness) (<=n, 6) clouds in (x,y,z,px,py,pz), pos [m] / mom [eV/c].

    drive = higher-weight subset, witness = lower (the campaign's +/-0.1% weight tag, matching
    FACET2_S2E.getDriverAndWitness and surrogate.preprocess).
    """
    P = ParticleGroup(str(path))
    w = np.unique(P.weight)
    def stack(pg):
        a = np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], axis=1).astype(np.float64)
        return a if len(a) <= n else a[rng.choice(len(a), n, replace=False)]
    return stack(P[P.weight == w[-1]]), stack(P[P.weight == w[0]])


def load_frames(n_particles, seed):
    rng = np.random.default_rng(seed)
    tgt_d, tgt_w = sigma_z_targets()
    frames = []
    for g in GOALS:
        cmp_ = json.load(open(VAL_DIR / f"validate_goal{g}um.json"))["comparison"]
        b = lambda k: cmp_[k]["bmad"]
        drive, witness = split_h5(VAL_DIR / f"sample_{g:05d}_PENT.h5", n_particles, rng)
        frames.append({
            "title": f"goal {g} um   |   Bmad spacing {b('spacing_um'):.0f} um"
                     f"   |   T_w {b('T_witness'):.2f}",
            "subtitle": f"sigma_z   drive {b('drive_sigma_z_um'):.0f} um (target {tgt_d:.0f})"
                        f"    witness {b('witness_sigma_z_um'):.0f} um (target {tgt_w:.0f})",
            "drive": drive, "witness": witness,
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
    axh.set_title(fr["title"], fontsize=25, fontweight="bold", pad=40)
    axh.text(0.5, 1.10, fr["subtitle"], transform=axh.transAxes, fontsize=18, color=DIM,
             ha="center", va="bottom")
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
    fig.subplots_adjust(left=0.10, right=0.975, top=0.855, bottom=0.12)
    return fig


def fig_to_rgb(fig):
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    return np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(h, w, 4)[..., :3].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-particles", type=int, default=2048,
                    help="particles per bunch drawn from the 100k Bmad track (2048 matches v1)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    frames = load_frames(args.n_particles, args.seed)
    zlim = shared_lims(frames, 2, 1e3)
    pzlim = shared_lims(frames, 5, 1e-9)
    rgb = []
    for g, fr in zip(GOALS, frames):
        fig = render(fr, zlim, pzlim)
        rgb.append(fig_to_rgb(fig))
        fig.savefig(OUT_DIR / f"lps_goal{g}um_dark_v2.png", dpi=100)
        plt.close(fig)
        print(f"  {fr['title']}   [{fr['subtitle']}]")
    imgs = [Image.fromarray(f) for f in rgb]
    gif = OUT_DIR / "lps_sweep_dark_v2.gif"
    imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                 duration=int(1000.0 / FPS), loop=0)
    print(f"wrote {gif}  ({len(imgs)} frames, {imgs[0].size[0]}x{imgs[0].size[1]})")


if __name__ == "__main__":
    main()
