"""Overlay Bmad-tracked vs surrogate-predicted 6D phase space (corner plots) at each setpoint.

Both clouds are put in the SAME frame the flow was trained on -- COORD_KEYS = (x,y,z,px,py,pz),
positions [m], momenta [eV/c] (surrogate/preprocess._stack) -- then displayed in mm / MeV/c / GeV/c.
Per-bunch centroids are marked so the inter-bunch transverse OFFSET (the metric the surrogate got
badly wrong) is explicit: Bmad's drive/witness centroids are separated, the surrogate's coincide.

Reads results/rl/bptt_gc_combined/bmad_validation/sample_<g>_PENT.h5 (Bmad) + results/rl/bptt_gc_combined/setpoints/setpoints_goal<g>um.json
(the normalized knobs) + the flow checkpoint.

Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/rl/bptt_gc_combined/bmad_validation/plot_corner_compare.py \
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

HERE = Path(__file__).resolve().parent
SETP = HERE.parent / "setpoints"
BMAD_C, SURR_C = "#4c72b0", "#dd8452"           # Bmad = blue, surrogate = orange
LABELS = ["x [mm]", "y [mm]", "z [mm]", "px [MeV/c]", "py [MeV/c]", "pz [GeV/c]"]
SCALE = np.array([1e3, 1e3, 1e3, 1e-6, 1e-6, 1e-9])   # (m,eV/c) -> (mm, MeV/c, GeV/c), COORD_KEYS order


def _stack(pg):
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float64) * SCALE


def _split(P):
    """drive = higher-weight subset, witness = lower (matches preprocess/getDriverAndWitness)."""
    w = np.unique(P.weight)
    drive = _stack(P[P.weight == w[-1]])
    witness = _stack(P[P.weight == w[0]]) if len(w) >= 2 else None
    return drive, witness


def _sub(a, n, rng):
    if a is None or len(a) <= n:
        return a
    return a[rng.choice(len(a), n, replace=False)]


def corner(ax_grid, cloud, color, marker, alpha, label, rng, n=2500):
    c = _sub(cloud, n, rng)
    d = LABELS.__len__()
    for i in range(d):
        for j in range(d):
            ax = ax_grid[i][j]
            if i == j:
                v = c[:, i]
                ax.hist(v, bins=40, color=color, histtype="step", lw=1.4, density=True)
            elif i > j:
                ax.scatter(c[:, j], c[:, i], s=3, alpha=alpha, color=color, edgecolors="none",
                           label=label if (i == 1 and j == 0) else None)


def centroids(ax_grid, drive, witness, color, edge):
    d = LABELS.__len__()
    for bunch, mk in ((drive, "*"), (witness, "P")):
        if bunch is None:
            continue
        cen = bunch.mean(0)
        for i in range(d):
            for j in range(d):
                if i > j:
                    ax_grid[i][j].scatter(cen[j], cen[i], s=90, marker=mk, color=color,
                                          edgecolors=edge, linewidths=0.8, zorder=6)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--n-cloud", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = sorted(glob.glob(args.flow_ckpt))[-1] if "*" in args.flow_ckpt else args.flow_ckpt
    flow = TwoBunchFlow.load_from_checkpoint(ck, map_location=device).eval().to(device)
    rng = np.random.default_rng(args.seed)

    for h5 in sorted(glob.glob(str(HERE / "sample_*_PENT.h5"))):
        g = int(Path(h5).stem.split("_")[1])                 # sample_00150_PENT -> 150
        sp = json.load(open(SETP / f"setpoints_goal{g}um.json"))
        knorm = np.array([sp["knob_setpoints_normalized"][k] for k in
                          sp["knob_setpoints_normalized"]], dtype=np.float32)
        kt = torch.tensor(knorm, device=device).unsqueeze(0)
        with torch.no_grad():
            s_d = flow.sample_bunch(kt, 0, args.n_cloud)[0].cpu().numpy() * SCALE
            s_w = flow.sample_bunch(kt, 1, args.n_cloud)[0].cpu().numpy() * SCALE
        b_d, b_w = _split(ParticleGroup(h5))

        d = len(LABELS)
        fig, axg = plt.subplots(d, d, figsize=(15, 15))
        for cloud, col, mk, al, lab in ((np.vstack([x for x in (b_d, b_w) if x is not None]), BMAD_C, "o", 0.25, "Bmad (truth)"),
                                        (np.vstack([s_d, s_w]), SURR_C, "x", 0.22, "surrogate")):
            corner(axg, cloud, col, mk, al, lab, rng, n=args.n_cloud)
        centroids(axg, b_d, b_w, BMAD_C, "k")               # * = drive centroid, P = witness centroid
        centroids(axg, s_d, s_w, SURR_C, "k")
        for i in range(d):
            for j in range(d):
                ax = axg[i][j]
                if i < j:
                    ax.axis("off"); continue
                if j == 0 and i > 0:
                    ax.set_ylabel(LABELS[i], fontsize=8)
                if i == d - 1:
                    ax.set_xlabel(LABELS[j], fontsize=8)
                ax.tick_params(labelsize=6)
        off_b = sp["surrogate_metrics"]  # for annotation reference
        val = json.load(open(HERE / f"validate_goal{g}um.json"))["comparison"]["transverse_offset_um"]
        handles = [plt.Line2D([], [], color=BMAD_C, marker="o", ls="", label="Bmad (truth)"),
                   plt.Line2D([], [], color=SURR_C, marker="x", ls="", label="surrogate"),
                   plt.Line2D([], [], color="grey", marker="*", ls="", label="drive centroid"),
                   plt.Line2D([], [], color="grey", marker="P", ls="", label="witness centroid")]
        fig.legend(handles=handles, loc="upper right", fontsize=11, framealpha=0.9)
        fig.suptitle(f"PENT phase space @ {g} µm setpoint — Bmad vs surrogate\n"
                     f"transverse offset: surrogate {val['surrogate']:.1f} µm vs Bmad {val['bmad']:.0f} µm "
                     f"(the drive/witness centroid split)", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out = HERE / f"corner_compare_goal{g}um.png"
        fig.savefig(out, dpi=110); plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
