"""Did manifold anchoring deliver near-matched two-bunch beams *for the real tracked beam*?

The achievable_targets funnel scores BMAG vs golden beta=0.5 m, which is the WRONG metric for
anchored draws: each anchor sample targets its own beta* (down to 7.6 cm), so it reads high
BMAG vs 0.5 m by construction. This script scores each anchored PENT cloud against ITS OWN
target beta* (from the manifest) using the slice-beta tools, and checks:
  - does the achieved witness slice-beta track the commanded beta*?  (anchoring fidelity)
  - is the witness matched to its target (slice BMAG vs beta* ~ 1)?
  - is the offset controlled (anchor vs wide tail)?

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.anchored_validation
"""
import glob
import json
import os
import re

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from ..datagen.paths import repo_root
from ..surrogate.properties import twiss_bmag, slice_twiss_bmag, _bmag

SUB = "expanded_anchored_pilot"
COORD = ["x", "y", "z", "px", "py", "pz"]


def _to_torch(pg):
    return torch.tensor(np.stack([getattr(pg, k) for k in COORD], axis=1),
                        dtype=torch.float64).unsqueeze(0)  # (1, N, 6)


def main():
    ddir = repo_root() / "data" / SUB
    man = {m["idx"]: m for m in json.load(open(ddir / "manifest.json"))}
    rows = []
    files = sorted(glob.glob(str(ddir / "sample_*_PENT.h5")))
    for i, f in enumerate(files):
        idx = int(re.search(r"sample_(\d+)_PENT", f).group(1))
        m = man.get(idx, {})
        if m.get("is_baseline_repeat"):
            continue
        block = m.get("block", "?")
        beta_t = m.get("ff_target_beta_m") or 0.5
        P = ParticleGroup(h5=f)
        res = qs.getDriverAndWitness(P)            # None if witness scraped (single weight)
        PD, PW = res if res is not None else (None, None)
        rec = {"idx": idx, "block": block, "beta_t": float(beta_t),
               "viable": PW is not None and len(PW) > 50}
        if rec["viable"]:
            w = _to_torch(PW)
            d = _to_torch(PD)
            sl = slice_twiss_bmag(w, n_slices=5, beta0=beta_t, alpha0=0.0)
            sld = slice_twiss_bmag(d, n_slices=5, beta0=beta_t, alpha0=0.0)
            pr = twiss_bmag(w, beta0=beta_t, alpha0=0.0)
            off = float(torch.sqrt((d[..., 0].mean() - w[..., 0].mean())**2 +
                                   (d[..., 1].mean() - w[..., 1].mean())**2) * 1e6)  # um
            rec.update(slice_beta_y=float(sl["slice_beta_y_core"]),
                       slice_beta_x=float(sl["slice_beta_x_core"]),
                       drive_slice_beta_y=float(sld["slice_beta_y_core"]),
                       drive_slice_bmag_vs_t=float(sld["slice_bmag_max"]),
                       slice_bmag_vs_t=float(sl["slice_bmag_max"]),
                       proj_bmag_vs_t=float(torch.maximum(pr["bmag_x"], pr["bmag_y"])),
                       offset_um=off)
        rows.append(rec)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")

    anc = [r for r in rows if r["block"] == "anchor" and r["viable"]]
    tail = [r for r in rows if r["block"] == "tail" and r["viable"]]
    n_anc = sum(r["block"] == "anchor" for r in rows)
    n_tail = sum(r["block"] == "tail" for r in rows)
    print(f"\nanchor: {len(anc)}/{n_anc} witness-viable; tail: {len(tail)}/{n_tail} viable")

    def pct(rs, k):
        v = np.array([r[k] for r in rs])
        return np.percentile(v, [5, 50, 95])

    print("\n-- scored against each sample's OWN target beta* --")
    for label, rs in [("anchor", anc), ("tail", tail)]:
        if not rs:
            continue
        p = pct(rs, "slice_bmag_vs_t")
        po = pct(rs, "offset_um")
        print(f"  {label:7s} witness slice-BMAG vs beta* : p5/p50/p95 = "
              f"{p[0]:.2f}/{p[1]:.2f}/{p[2]:.2f}   "
              f"| <1.5: {np.mean([r['slice_bmag_vs_t']<1.5 for r in rs]):.0%}  "
              f"<2: {np.mean([r['slice_bmag_vs_t']<2 for r in rs]):.0%}")
        print(f"  {label:7s} offset [um]                 : p5/p50/p95 = "
              f"{po[0]:.0f}/{po[1]:.0f}/{po[2]:.0f}  | <10um: {np.mean([r['offset_um']<10 for r in rs]):.0%}")

    # anchoring fidelity: achieved slice-beta vs commanded beta*, drive vs witness
    if anc:
        bt = np.array([r["beta_t"] for r in anc]) * 100
        bay = np.array([r["slice_beta_y"] for r in anc]) * 100
        bdy = np.array([r["drive_slice_beta_y"] for r in anc]) * 100
        rho = np.corrcoef(np.log(bt), np.log(np.clip(bay, 1e-3, None)))[0, 1]
        rho_d = np.corrcoef(np.log(bt), np.log(np.clip(bdy, 1e-3, None)))[0, 1]
        print(f"\nanchoring fidelity corr(log beta*, log achieved slice-beta_y):")
        print(f"   witness rho = {rho:+.2f}   (achieved slice-beta_y median {np.median(bay):.1f} cm)")
        print(f"   drive   rho = {rho_d:+.2f}   (achieved slice-beta_y median {np.median(bdy):.1f} cm)")
        dp = np.percentile([r["drive_slice_bmag_vs_t"] for r in anc], [5, 50, 95])
        print(f"   drive slice-BMAG vs beta*: p5/p50/p95 = {dp[0]:.2f}/{dp[1]:.2f}/{dp[2]:.2f}")

    figdir = repo_root() / "artifacts" / "figures" / SUB
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    if anc:
        ax[0].scatter(bt, bay, s=12, alpha=0.5)
        lim = [min(bt.min(), bay.min()), max(bt.max(), bay.max())]
        ax[0].plot(lim, lim, "k--", lw=1)
        ax[0].set_xlabel("target beta* [cm]"); ax[0].set_ylabel("achieved witness slice-beta_y [cm]")
        ax[0].set_title(f"anchoring fidelity (rho={rho:.2f})"); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    for rs, c, lab in [(anc, "#4c72b0", "anchor"), (tail, "#dd8452", "tail")]:
        if rs:
            v = np.clip([r["slice_bmag_vs_t"] for r in rs], 0.5, 50)
            ax[1].hist(v, bins=np.logspace(0, 1.7, 30), alpha=0.6, color=c, label=lab)
            o = np.clip([r["offset_um"] for r in rs], 1, 5000)
            ax[2].hist(o, bins=np.logspace(0, 3.7, 30), alpha=0.6, color=c, label=lab)
    ax[1].axvline(1.5, color="g", ls="--"); ax[1].set_xscale("log")
    ax[1].set_xlabel("witness slice-BMAG vs target beta*"); ax[1].set_title("matching to own target"); ax[1].legend()
    ax[2].axvline(10, color="g", ls="--"); ax[2].set_xscale("log")
    ax[2].set_xlabel("driver-witness offset [um]"); ax[2].set_title("collinearity"); ax[2].legend()
    fig.suptitle(f"{SUB}: anchored beams scored against their own target beta*")
    fig.tight_layout()
    p = figdir / "anchored_validation.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
