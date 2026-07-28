"""Clean presentation version of the real-vs-sim XTCAV screen comparison.

Same data, same orientation logic and same axes as `xtcav_screen_compare.py` (imported from it, so
the two figures cannot drift apart), stripped to just the four elements: two current-profile plots
and two screen images. No gate lines, no metric annotations, no calibration text -- the numbers and
all the caveats live in the analysis figure and in
results/tier2_comparison/screen_compare/README.md, which you should read before showing this.

The orientation logic is NOT cosmetic and is inherited deliberately:
  - z = -c*dt with the streak sign applied, so larger z is nearer the HEAD. The sim sign comes from
    the beam itself; the REAL sign is anchored model-free: the majority-charge z-lobe is
    the drive (3:1 cathode double pulse) and the drive leads, so the axis is oriented to put it
    at the head.
  - dE/E from the deployed spectrometer mapping (real) and the beam's own dispersion fit (sim).
  - z anchored on each side's major z-lobe; both panels normalized on the 99.5th percentile of
    lit pixels. NO sim blur by default -- sigma~3 px would bring the sim's 13.9 kA peak current onto
    the real 6.0 kA and manufacture agreement in the I(z) panels. So the sim looks sharper and
    grainier than the real image, which is honest: it has no camera PSF, noise or saturation.

Run in the bmad-qpad-dev env:
  cd twobunch-s2e-rl && PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python scripts/xtcav_screen_compare_clean.py

Output -> XTCAV_analysis/sim2real/results/tier2_comparison/screen_compare/
  fig_screen_compare_<dataset>_step<NN>_<branch>_clean.png/.pdf
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_SRC = Path(__file__).resolve().parent / "xtcav_screen_compare.py"
_spec = importlib.util.spec_from_file_location("xtcav_screen_compare", _SRC)
X = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(X)                      # reuse loaders/orientation/diagnostics verbatim

BG, FG, DIM = X.BG, X.FG, X.DIM
REAL_C, SIM_C = X.REAL_C, X.SIM_C
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Nimbus Sans", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": BG, "savefig.facecolor": BG, "axes.facecolor": BG,
    "text.color": FG, "axes.labelcolor": FG, "axes.edgecolor": FG,
    "xtick.color": FG, "ytick.color": FG, "axes.titlecolor": "#FFFFFF",
})


def build_sides(dataset, step, branch, shot, run_tag, blur_px, real_z_sign):
    """Load both sides with the analysis script's orientation decisions applied."""
    t0 = X.RESULTS / "tier0_real_features" / dataset
    t1 = X.RESULTS / "tier1_sim_scans" / dataset / run_tag

    real_sum = json.load(open(t0 / "summary.json"))
    cal_real = float(real_sum["calibration"]["cal_um_per_deg_used"])
    l2_val = float(np.asarray(real_sum["scan"]["vals"], float)[step - 1])

    F = np.load(t0 / "features.npz")
    m = (F["step"] == step) & (F["streak_sign"] == (1.0 if branch == "plus" else -1.0)) \
        & (F["eg_ok"] > 0)
    sep_shots, ix_shots = F["eg_sep_median_um"][m], F["ix"][m].astype(int)
    if shot is None:
        shot = int(ix_shots[int(np.argmin(np.abs(sep_shots - float(np.median(sep_shots)))))])

    cal_sim = float(json.load(open(t1 / "summary.json"))
                    ["phase_scan_calibration"]["cal_um_per_deg"])
    sim = X.load_sim(t1 / f"beam_DTOTR_step{step:02d}_plus.h5", cal_sim)

    # real head direction, model-free: majority z-lobe = drive (3:1 cathode double pulse), and the
    # drive leads. Same rule as xtcav_screen_compare.py -- keep the two in step.
    real_pos = X.load_real(dataset, step, shot, cal_real, +1.0)
    if real_z_sign == "auto":
        zs = 1.0 if X.lobe_split(real_pos)["major_leads"] else -1.0
    else:
        zs = float(real_z_sign)
    real = real_pos if zs > 0 else X.load_real(dataset, step, shot, cal_real, zs)

    if blur_px > 0:
        sim["img"] = ndimage.gaussian_filter(sim["img"], sigma=blur_px)

    for side in (real, sim):                     # anchor each side's major z-lobe at z = 0
        X.realign_on_major(side, X.gate(side))
    return real, sim, shot, l2_val


def panel(fig, gs_col, side, label, accent, zlim, dlim, ilim, show_y):
    ax_i = fig.add_subplot(gs_col[0])
    ax = fig.add_subplot(gs_col[1], sharex=ax_i)

    order = np.argsort(side["delta_pct"])                   # energy increasing upward
    img = side["img"][order, :]
    lit = img[img > 0]
    vmax = float(np.percentile(lit, 99.5)) if lit.size else 1.0
    ax.pcolormesh(X.edges_from_centers(side["z_um"]),
                  X.edges_from_centers(side["delta_pct"][order]),
                  np.clip(img / vmax, 0, 1), cmap="magma", vmin=0.0, vmax=1.0,
                  shading="auto", rasterized=True)

    ax_i.plot(side["z_um"],
              X.current_profile(side["img"].sum(axis=0), side["z_um"], side["charge_nC"]),
              color=accent, lw=2.2)
    ax_i.set_ylim(*ilim)
    ax_i.tick_params(labelbottom=False, labelsize=15)
    ax_i.set_title(label, fontsize=24, fontweight="bold", pad=16)
    for sp in ("top", "right"):
        ax_i.spines[sp].set_visible(False)

    ax.set_xlim(*zlim)
    ax.set_ylim(*dlim)
    ax.set_xlabel("z  [µm]        head →", fontsize=20, labelpad=6)
    ax.tick_params(labelsize=15)
    if show_y:
        ax.set_ylabel("ΔE / E  [%]", fontsize=20)
        ax_i.set_ylabel("I  [kA]", fontsize=18)
    else:
        ax.tick_params(labelleft=False)
        ax_i.tick_params(labelleft=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="E300_15673")
    ap.add_argument("--step", type=int, default=17)
    ap.add_argument("--branch", choices=("plus", "neg"), default="plus")
    ap.add_argument("--run-tag", default="l2ref-38.00_value_match")
    ap.add_argument("--shot", type=int, default=None)
    ap.add_argument("--real-z-sign", choices=("auto", "+1", "-1"), default="auto")
    ap.add_argument("--sim-blur-px", type=float, default=0.0,
                    help="off by default: a sigma~3 blur tunes the sim peak current "
                         "onto the real one, manufacturing agreement in the I(z) panels")
    ap.add_argument("--no-title", action="store_true",
                    help="omit the one-line title (for slides that supply their own)")
    args = ap.parse_args()

    real, sim, shot, l2_val = build_sides(args.dataset, args.step, args.branch, args.shot,
                                          args.run_tag, args.sim_blur_px, args.real_z_sign)

    zlim = (min(real["z_um"].min(), sim["z_um"].min()),
            max(real["z_um"].max(), sim["z_um"].max()))
    dlim = (min(real["delta_pct"].min(), sim["delta_pct"].min()),
            max(real["delta_pct"].max(), sim["delta_pct"].max()))
    ipk = max(X.current_profile(real["img"].sum(axis=0), real["z_um"], real["charge_nC"]).max(),
              X.current_profile(sim["img"].sum(axis=0), sim["z_um"], sim["charge_nC"]).max())
    ilim = (0, 1.12 * ipk)

    top = 0.90 if args.no_title else 0.845
    fig = plt.figure(figsize=(15.0, 7.6))
    outer = fig.add_gridspec(1, 2, left=0.072, right=0.988, bottom=0.115, top=top, wspace=0.06)
    for i, (side, lab, col) in enumerate(((real, "measured", REAL_C),
                                          (sim, "FACET2-S2E simulation", SIM_C))):
        sub = outer[i].subgridspec(2, 1, height_ratios=(1, 4.3), hspace=0.05)
        panel(fig, sub, side, lab, col, zlim, dlim, ilim, show_y=(i == 0))

    if not args.no_title:
        fig.suptitle(f"XTCAV streaked phase space at DTOTR2   —   {args.dataset}, "
                     f"L2 phase {l2_val:+.2f}°", fontsize=23, fontweight="bold", y=0.965)

    stem = (f"fig_screen_compare_{args.dataset}_step{args.step:02d}_{args.branch}_clean")
    X.OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(X.OUT_DIR / f"{stem}.{ext}", dpi=200)
    print(f"real shot {shot}, L2 {l2_val:+.2f}°, real z_sign {real['z_sign']:+.0f}")
    print(f"wrote {X.OUT_DIR}/{stem}.png/.pdf")


if __name__ == "__main__":
    main()
