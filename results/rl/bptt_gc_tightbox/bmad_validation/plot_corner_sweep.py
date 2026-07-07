"""Bmad-vs-surrogate corner/LPS across the 100-300 um sweep: per-goal overlay PNGs, a corner GIF,
a dedicated LPS (z-pz) GIF, and a 200 um-vs-baseline corner (both Bmad).

Both clouds are in the flow's frame COORD_KEYS=(x,y,z,px,py,pz) [m, eV/c], shown in mm / MeV/c / GeV/c.
GIF frames use fixed global axis limits so the animation doesn't jump.

Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/rl/bptt_gc_tightbox/bmad_validation/plot_corner_sweep.py \
      --flow-ckpt trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt
"""
import argparse
import glob
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
from twobunch_s2e_rl.rl._eval_plots import _save_gif, _fig_to_rgba

HERE = Path(__file__).resolve().parent
SETP = HERE.parent / "setpoints"
BASELINE_H5 = "data/tightbox_v2_full/sample_06000_PENT.h5"     # a golden baseline-repeat PENT beam
LABELS = ["x [mm]", "y [mm]", "z [mm]", "px [MeV/c]", "py [MeV/c]", "pz [GeV/c]"]
SCALE = np.array([1e3, 1e3, 1e3, 1e-6, 1e-6, 1e-9])
A_C, B_C = "#4c72b0", "#dd8452"


def _split_h5(path):
    P = ParticleGroup(str(path)); w = np.unique(P.weight)
    st = lambda pg: np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float64) * SCALE
    return st(P[P.weight == w[-1]]), (st(P[P.weight == w[0]]) if len(w) >= 2 else None)


@torch.no_grad()
def _surr(flow, knorm, n, dev):
    kt = torch.tensor(knorm, dtype=torch.float32, device=dev).unsqueeze(0)
    return (flow.sample_bunch(kt, 0, n)[0].cpu().numpy() * SCALE,
            flow.sample_bunch(kt, 1, n)[0].cpu().numpy() * SCALE)


def _sub(a, n, rng):
    return a if (a is None or len(a) <= n) else a[rng.choice(len(a), n, replace=False)]


def _lims(all_clouds):
    C = np.vstack([c for c in all_clouds if c is not None])
    return [np.percentile(C[:, i], [1, 99]) for i in range(6)]


def corner_fig(A, B, cA, cB, labA, labB, title, lims, rng, n=2500):
    (Ad, Aw), (Bd, Bw) = A, B
    fig, axg = plt.subplots(6, 6, figsize=(15, 15))
    for src, col, lab in ((np.vstack([x for x in A if x is not None]), cA, labA),
                          (np.vstack([x for x in B if x is not None]), cB, labB)):
        c = _sub(src, n, rng)
        for i in range(6):
            for j in range(6):
                ax = axg[i][j]
                if i == j:
                    ax.hist(c[:, i], bins=np.linspace(*lims[i], 40), density=True,
                            histtype="step", lw=1.4, color=col)
                elif i > j:
                    ax.scatter(c[:, j], c[:, i], s=3, alpha=0.22, color=col, edgecolors="none")
    for (d, w), col in ((A, cA), (B, cB)):
        for bunch, mk in ((d, "*"), (w, "P")):
            if bunch is None:
                continue
            cen = bunch.mean(0)
            for i in range(6):
                for j in range(6):
                    if i > j:
                        axg[i][j].scatter(cen[j], cen[i], s=90, marker=mk, color=col,
                                          edgecolors="k", lw=0.7, zorder=6)
    for i in range(6):
        for j in range(6):
            ax = axg[i][j]
            if i < j:
                ax.axis("off"); continue
            ax.set_xlim(*lims[j])
            if i != j:
                ax.set_ylim(*lims[i])
            if j == 0 and i > 0:
                ax.set_ylabel(LABELS[i], fontsize=8)
            if i == 5:
                ax.set_xlabel(LABELS[j], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.legend(handles=[plt.Line2D([], [], color=cA, marker="o", ls="", label=labA),
                        plt.Line2D([], [], color=cB, marker="o", ls="", label=labB),
                        plt.Line2D([], [], color="grey", marker="*", ls="", label="drive centroid"),
                        plt.Line2D([], [], color="grey", marker="P", ls="", label="witness centroid")],
               loc="upper right", fontsize=12)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def lps_fig(A, B, cA, cB, labA, labB, title, lims):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for (d, w), col, lab in ((A, cA, labA), (B, cB, labB)):
        c = np.vstack([x for x in (d, w) if x is not None])
        ax.scatter(c[:, 2], c[:, 5], s=4, alpha=0.25, color=col, edgecolors="none", label=lab)
    ax.set_xlim(*lims[2]); ax.set_ylim(*lims[5])
    ax.set_xlabel("z [mm]"); ax.set_ylabel("pz [GeV/c]"); ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--n", type=int, default=2500)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = sorted(glob.glob(args.flow_ckpt))[-1] if "*" in args.flow_ckpt else args.flow_ckpt
    flow = TwoBunchFlow.load_from_checkpoint(ck, map_location=dev).eval().to(dev)
    rng = np.random.default_rng(0)

    goals = sorted(int(Path(f).stem.split("_")[1]) for f in glob.glob(str(HERE / "sample_*_PENT.h5")))
    data = {}
    for g in goals:
        bd, bw = _split_h5(HERE / f"sample_{g:05d}_PENT.h5")
        sp = json.load(open(SETP / f"setpoints_goal{g}um.json"))
        kn = np.array([sp["knob_setpoints_normalized"][k] for k in sp["knob_setpoints_normalized"]], np.float32)
        sd, sw = _surr(flow, kn, args.n, dev)
        data[g] = ((bd, bw), (sd, sw))
    lims = _lims([c for g in goals for src in data[g] for c in src])

    corner_frames, lps_frames = [], []
    for g in goals:
        A, B = data[g]
        t = f"{g} µm setpoint — Bmad vs surrogate"
        cf = corner_fig(A, B, A_C, B_C, "Bmad (truth)", "surrogate", t, lims, rng, args.n)
        cf.savefig(HERE / f"corner_compare_goal{g}um.png", dpi=110); corner_frames.append(_fig_to_rgba(cf)); plt.close(cf)
        lf = lps_fig(A, B, A_C, B_C, "Bmad", "surrogate", f"{g} µm — LPS (z-pz)", lims)
        lps_frames.append(_fig_to_rgba(lf)); plt.close(lf)
    _save_gif(corner_frames, HERE / "corner_compare_sweep.gif", fps=1.2)
    _save_gif(lps_frames, HERE / "lps_compare_sweep.gif", fps=1.2)

    # ---- 200 um setpoint vs baseline (both Bmad) ----
    bl_d, bl_w = _split_h5(Path(BASELINE_H5))
    A = data[200][0]                                    # Bmad 200 um
    l2 = _lims([*A, bl_d, bl_w])
    cf = corner_fig(A, (bl_d, bl_w), "#c44e52", "#8c8c8c", "200 µm setpoint (Bmad)",
                    "baseline 2024-10-14 (Bmad)", "200 µm setpoint vs baseline — PENT phase space",
                    l2, rng, args.n)
    cf.savefig(HERE / "corner_200um_vs_baseline.png", dpi=110); plt.close(cf)
    print(f"wrote corner_compare_goal*um.png ({len(goals)}), corner_compare_sweep.gif, "
          f"lps_compare_sweep.gif, corner_200um_vs_baseline.png to {HERE}")


if __name__ == "__main__":
    main()
