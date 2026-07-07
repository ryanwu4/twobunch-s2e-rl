"""Knob-attribution test: is the witness leading-edge spike caused by the edge-PINNED knobs?

At the 150 um setpoint the pinned (railed) knobs are the 3 BC20 sextupole strengths (S1/S2/S3ELkG)
+ YC1FFkG + S1EL_xOffset. Starting from GOLDEN (in-distribution, smooth), we test:
  - SUFFICIENCY: golden with only the pinned knobs moved to their setpoint (edge) values -> spike?
  - NECESSITY : full setpoint with only the pinned knobs restored to golden -> spike gone?
  - isolate the sextupoles alone (the longitudinally-relevant pinned knobs).
Plus a continuous scan of a spike metric (peak current density) vs golden->setpoint interpolation,
for (a) sextupole strengths only, (b) all pinned, (c) all NON-pinned knobs (control). If the spike
tracks (a)/(b) and the control (c) stays flat, the pinned sextupoles are the cause.

Usage: CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/rl/bptt_gc_combined/spike_study/knob_attribution.py \
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

from twobunch_s2e_rl.surrogate import COORD_KEYS
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set

HERE = Path(__file__).resolve().parent
SETP = HERE.parent / "setpoints"
ZI = COORD_KEYS.index("z")
MM = 1e3
BINW = 0.002   # mm, fixed physical bin width for the peak-density (spike) metric


@torch.no_grad()
def witness_z(flow, knorm, device, n=15000):
    kt = torch.tensor(knorm, dtype=torch.float32, device=device).unsqueeze(0)
    return flow.sample_bunch(kt, 1, n)[0].cpu().numpy()[:, ZI] * MM


def peak_density(z):
    """Max normalized current density [per mm] -- location-invariant spike-height metric."""
    lo, hi = np.percentile(z, [0.5, 99.5])
    bins = np.arange(lo, hi + BINW, BINW)
    d, _ = np.histogram(z, bins=bins, density=True)
    return float(d.max())


def profile(ax, z, color, label):
    lo, hi = np.percentile(z, [0.5, 99.5])
    ax.hist(z, bins=np.arange(lo, hi + BINW, BINW), density=True, histtype="step", lw=1.9,
            color=color, label=f"{label}  (peak {peak_density(z):.0f})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--n", type=int, default=15000)
    ap.add_argument("--target", type=int, default=150)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = sorted(glob.glob(args.flow_ckpt))[-1] if "*" in args.flow_ckpt else args.flow_ckpt
    flow = TwoBunchFlow.load_from_checkpoint(ck, map_location=device).eval().to(device)

    keys, lo, hi, base = resolve_sweep_set("tightbox+expanded")
    lo, hi = np.array(lo), np.array(hi)
    gold = np.clip((np.array([base[k] for k in keys]) - lo) / (hi - lo), 0, 1)
    sp = json.load(open(SETP / f"setpoints_goal{args.target}um.json"))
    setn = np.array([sp["knob_setpoints_normalized"][k] for k in sp["knob_setpoints_normalized"]])

    pinned = [i for i, k in enumerate(keys) if setn[i] <= 0.02 or setn[i] >= 0.98]
    sext = [i for i, k in enumerate(keys) if k in ("S1ELkG", "S2ELkG", "S3ELkG")]
    nonpin = [i for i in range(len(keys)) if i not in pinned]
    print("pinned:", [keys[i] for i in pinned], "| sextupoles:", [keys[i] for i in sext])

    def mix(basev, group, src):   # basev with `group` indices replaced by src
        v = basev.copy(); v[group] = src[group]; return v

    # ---- FIG 1: necessity / sufficiency profiles ---------------------------------------
    configs = [
        (gold, "#8172b3", "golden (in-distribution)"),
        (setn, "#dd8452", "setpoint (full)"),
        (mix(gold, pinned, setn), "#c44e52", "golden + PINNED→edge (sufficiency)"),
        (mix(setn, pinned, gold), "#55a868", "setpoint + PINNED→golden (necessity)"),
        (mix(gold, sext, setn), "#4c72b0", "golden + SEXTUPOLES→edge only"),
    ]
    fig, ax = plt.subplots(figsize=(11, 6))
    for knorm, col, lab in configs:
        profile(ax, witness_z(flow, knorm, device, args.n), col, lab)
    ax.set_xlabel("z [mm]"); ax.set_ylabel("witness current [arb]")
    ax.set_title(f"Knob attribution @ {args.target} µm: does moving only the pinned knobs make the spike?",
                 fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "knob_attribution_profiles.png", dpi=130); plt.close(fig)

    # ---- FIG 2: spike metric vs golden->setpoint interpolation --------------------------
    alphas = np.linspace(0, 1, 9)
    groups = [(sext, "sextupole strengths (S1/S2/S3ELkG)", "#4c72b0"),
              (pinned, "all pinned knobs", "#c44e52"),
              (nonpin, "all NON-pinned knobs (control)", "#8c8c8c")]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for gidx, lab, col in groups:
        pk = []
        for a in alphas:
            v = gold.copy(); v[gidx] = gold[gidx] + a * (setn[gidx] - gold[gidx])
            pk.append(peak_density(witness_z(flow, v, device, args.n)))
        ax.plot(alphas, pk, "-o", color=col, lw=2, label=lab)
    ax.axhline(peak_density(witness_z(flow, gold, device, args.n)), color="#8172b3", ls=":",
               label="golden")
    ax.axhline(peak_density(witness_z(flow, setn, device, args.n)), color="#dd8452", ls=":",
               label="full setpoint")
    ax.set_xlabel("interpolation golden → setpoint (only the named group moves)")
    ax.set_ylabel("peak witness current density [per mm]  (spike sharpness)")
    ax.set_title(f"Spike onset vs knob group @ {args.target} µm", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / "knob_attribution_scan.png", dpi=130); plt.close(fig)
    print(f"wrote knob_attribution_profiles.png + knob_attribution_scan.png to {HERE}")


if __name__ == "__main__":
    main()
