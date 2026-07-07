"""Investigate the witness leading-edge density spike the surrogate predicts but Bmad doesn't show.

Diagnostics (witness bunch, z = longitudinal position):
  1. Reproduce: witness z-current, Bmad vs surrogate (full sample).
  2. Origin in the flow: surrogate FULL (mu + L * coupling(z)) vs CORE (mu + L * z, coupling skipped).
     The whitening (mu + L*.) is affine -> it cannot manufacture a spike from a smooth input; so if
     CORE is smooth and FULL is spiky, the spike is an RQS/affine COUPLING-transform caustic
     (a near-flat spline segment piling up probability mass).
  3. Extrapolation test: surrogate witness z-current at the (off-distribution, edge-pinned) SETPOINT
     vs at the IN-distribution golden knobs. If the spike is only off-distribution -> extrapolation.
  4. Reality check: z-current of real training witness beams (processed h5) nearest the setpoint --
     do real Bmad-tracked witnesses have a leading-edge current spike at all?

Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/rl/bptt_gc_combined/spike_study/investigate_spike.py \
      --flow-ckpt trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup

from twobunch_s2e_rl.surrogate import COORD_KEYS
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set

HERE = Path(__file__).resolve().parent
VAL = HERE.parent / "bmad_validation"
SETP = HERE.parent / "setpoints"
ZI = COORD_KEYS.index("z")          # z column
MM = 1e3                            # m -> mm
BMAD_C, FULL_C, CORE_C, GOLD_C = "#4c72b0", "#dd8452", "#55a868", "#8172b3"


def witness_from_h5(path):
    P = ParticleGroup(path)
    w = np.unique(P.weight)
    pg = P[P.weight == w[0]]         # witness = lower-weight subset
    return np.stack([getattr(pg, k) for k in COORD_KEYS], axis=1).astype(np.float64)


@torch.no_grad()
def core_and_full(flow, kt, k, n, device):
    """Return (full, core) physical witness clouds. core skips the coupling transform."""
    h = flow._encode(kt, k)
    h_flat = h.unsqueeze(1).expand(1, n, -1).reshape(n, -1)
    z = torch.randn(n, flow.latent_dim, device=device)
    w_full, _ = flow._forward_enc(z, h_flat)
    mu, Lm = flow._whiten(h, k)
    mean, std = flow._scaler(k)
    def place(w):
        x_std = mu.unsqueeze(1) + torch.einsum("bij,bnj->bni", Lm, w.reshape(1, n, -1))
        return (x_std * std + mean)[0].cpu().numpy()
    return place(w_full), place(z)


def zcur(ax, clouds, colors, labels, title, bins=80):
    allz = np.concatenate([c[:, ZI] * MM for c in clouds])
    edges = np.linspace(np.percentile(allz, 0.5), np.percentile(allz, 99.5), bins)
    for c, col, lab in zip(clouds, colors, labels):
        ax.hist(c[:, ZI] * MM, bins=edges, density=True, histtype="step", lw=1.8, color=col, label=lab)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("witness current [arb]"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--processed", default="processed/twobunch_combined.h5")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = sorted(glob.glob(args.flow_ckpt))[-1] if "*" in args.flow_ckpt else args.flow_ckpt
    flow = TwoBunchFlow.load_from_checkpoint(ck, map_location=device).eval().to(device)

    keys, lo, hi, base = resolve_sweep_set("tightbox+expanded")
    lo, hi = np.array(lo), np.array(hi)
    golden_norm = np.clip((np.array([base[k] for k in keys]) - lo) / (hi - lo), 0, 1)
    gt = torch.tensor(golden_norm, dtype=torch.float32, device=device).unsqueeze(0)

    setfiles = sorted(glob.glob(str(SETP / "setpoints_goal*um.json")))
    fig, axes = plt.subplots(len(setfiles), 3, figsize=(17, 5.0 * len(setfiles)))
    if len(setfiles) == 1:
        axes = axes[None, :]
    for r, sf in enumerate(setfiles):
        sp = json.load(open(sf))
        g = int(round(sp["target_um"]))
        knorm = np.array([sp["knob_setpoints_normalized"][k] for k in sp["knob_setpoints_normalized"]],
                         dtype=np.float32)
        kt = torch.tensor(knorm, device=device).unsqueeze(0)
        full, core = core_and_full(flow, kt, 1, args.n, device)
        gold_full, _ = core_and_full(flow, gt, 1, args.n, device)
        bmad = witness_from_h5(str(VAL / f"sample_{g:05d}_PENT.h5"))

        zcur(axes[r, 0], [bmad, full], [BMAD_C, FULL_C], ["Bmad (truth)", "surrogate (full)"],
             f"{g} µm: reproduce the spike")
        zcur(axes[r, 1], [full, core], [FULL_C, CORE_C],
             ["surrogate FULL (with coupling)", "surrogate CORE (coupling skipped)"],
             f"{g} µm: is it a coupling-transform caustic?")
        zcur(axes[r, 2], [full, gold_full], [FULL_C, GOLD_C],
             [f"setpoint (off-distribution)", "golden knobs (in-distribution)"],
             f"{g} µm: extrapolation test")

    fig.suptitle("Witness leading-edge density spike — origin diagnostics (z = longitudinal)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "spike_origin.png", dpi=130); plt.close(fig)

    # ---- reality check: real training witness z-profiles nearest the first setpoint -----------
    sp0 = json.load(open(setfiles[0]))
    k0 = np.array([sp0["knob_setpoints_normalized"][k] for k in sp0["knob_setpoints_normalized"]])
    with h5py.File(args.processed) as hf:
        wd = hf["witness_density"][:].astype(bool)
        knobs_raw = hf["knobs"][:][wd]
        wparts = hf["witness_parts"][:][wd]
    knobs_norm = np.clip((knobs_raw - lo) / (hi - lo), 0, 1)
    nearest = np.argsort(np.linalg.norm(knobs_norm - k0, axis=1))[:20]
    fig2, ax = plt.subplots(figsize=(9, 5.5))
    for i in nearest:
        z = wparts[i][:, ZI] * MM
        ax.hist(z - np.median(z), bins=60, density=True, histtype="step", lw=1.0, alpha=0.55, color="#4c72b0")
    ax.plot([], [], color="#4c72b0", label="real training witnesses (20 nearest, z-centered)")
    full0, _ = core_and_full(flow, torch.tensor(k0, dtype=torch.float32, device=device).unsqueeze(0), 1, args.n, device)
    z0 = full0[:, ZI] * MM
    ax.hist(z0 - np.median(z0), bins=60, density=True, histtype="step", lw=2.4, color="#dd8452",
            label="surrogate at setpoint (z-centered)")
    ax.set_xlabel("z - median [mm]"); ax.set_ylabel("witness current [arb]")
    ax.set_title("Reality check: do real training witnesses have a leading-edge current spike?", fontsize=11)
    ax.legend(fontsize=9)
    fig2.tight_layout(); fig2.savefig(HERE / "spike_training_reality.png", dpi=130); plt.close(fig2)

    print(f"wrote spike_origin.png + spike_training_reality.png to {HERE}")


if __name__ == "__main__":
    main()
