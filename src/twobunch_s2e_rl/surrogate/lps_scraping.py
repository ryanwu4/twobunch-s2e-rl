"""Compare the surrogate's longitudinal-phase-space (z-pz) error for witness bunches that
were scraped vs intact.

Groups witness-density beams by surviving fraction (intact >= 0.9, scraped < 0.9) and, per
beam, measures how well the flow reproduces the witness z-pz: filament thinness, z-pz
correlation, longitudinal emittance (sqrt det cov(z,pz)), and energy spread sigma_pz.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.surrogate.lps_scraping \
      --ckpt "trained/twobunch_flow_v2/checkpoints/best-*.ckpt" --processed processed/twobunch_flow_v2.h5

Writes <--out, default results/surrogate/v2/diagnostics>/lps_scraping_{box,trend,examples}.png + a summary.
"""
from __future__ import annotations

import argparse
import glob
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..datagen.paths import repo_root
from .diagnostics import Bundle

INTACT = 0.9  # surviving-fraction threshold: >= intact, < scraped
C_T, C_F = "#1f77b4", "#d62728"


def _metrics(c):
    """z-pz LPS metrics for a cloud (P,6): thinness, corr, long-emit, sigma_pz."""
    z, pz = c[:, 2].astype(float), c[:, 5].astype(float)
    cov = np.cov(np.stack([z, pz]))
    ez = float(np.sqrt(max(np.linalg.det(cov), 0.0)))
    spz = float(np.sqrt(cov[1, 1]))
    corr = float(cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + 1e-30))
    Z = (z - z.mean()) / (z.std() + 1e-30)
    P = (pz - pz.mean()) / (pz.std() + 1e-30)
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([Z, P]))))
    thin = float(np.sqrt(ev[0] / max(ev[1], 1e-30)))
    return thin, corr, ez, spz


@torch.no_grad()
def _flow_batched(b, rows, n=2048, chunk=256):
    out = []
    for i in range(0, len(rows), chunk):
        kn = b.ds.knobs[rows[i:i + chunk]].to(b.dev)
        out.append(b.m.sample_bunch(kn, 1, n).cpu().numpy())
    return np.concatenate(out, 0)


def analyze(ckpt, processed, out_dir, n=2048):
    b = Bundle(ckpt, processed)
    ds = b.ds
    wm, ws = np.array(ds.norm["witness_mean"]), np.array(ds.norm["witness_std"])
    rows = np.where(ds.witness_density.numpy().astype(bool))[0]
    frac = ds.witness_frac.numpy()[rows]
    true_parts = ds.witness[rows].numpy() * ws + wm                 # (R,P,6) physical
    flow_parts = _flow_batched(b, rows, n=n)                        # (R,n,6) physical

    mt = np.array([_metrics(c) for c in true_parts])                # (R,4)
    mf = np.array([_metrics(c) for c in flow_parts])
    thin_t, corr_t, ez_t, spz_t = mt.T
    thin_f, corr_f, ez_f, spz_f = mf.T
    dcorr = np.abs(corr_f - corr_t)
    ez_ratio = ez_f / np.maximum(ez_t, 1e-30)
    spz_ratio = spz_f / np.maximum(spz_t, 1e-30)

    intact = frac >= INTACT
    scraped = ~intact
    g = {"intact": intact, "scraped": scraped}

    def med(x, m):
        return float(np.median(x[m]))

    summary = {grp: {
        "n": int(mask.sum()),
        "thin_true_med": med(thin_t, mask), "thin_flow_med": med(thin_f, mask),
        "dcorr_med": med(dcorr, mask), "ez_ratio_med": med(ez_ratio, mask),
        "spz_ratio_med": med(spz_ratio, mask),
    } for grp, mask in g.items()}

    out = repo_root() / out_dir
    out.mkdir(parents=True, exist_ok=True)

    # ---- Fig 1: boxplots intact vs scraped ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    ax[0, 0].boxplot([thin_t[intact], thin_f[intact], thin_t[scraped], thin_f[scraped]],
                     tick_labels=["intact\ntrue", "intact\nflow", "scraped\ntrue", "scraped\nflow"],
                     showfliers=False)
    ax[0, 0].set_title("z-pz thinness (lower = thinner filament)")
    ax[0, 0].set_ylabel("minor/major axis ratio")
    ax[0, 1].boxplot([dcorr[intact], dcorr[scraped]], tick_labels=["intact", "scraped"], showfliers=False)
    ax[0, 1].set_title("|Δ z-pz correlation| (flow − true)")
    for a, dat, ttl in [(ax[1, 0], ez_ratio, "longitudinal emittance ratio flow/true"),
                        (ax[1, 1], spz_ratio, "energy spread σ_pz ratio flow/true")]:
        a.boxplot([dat[intact], dat[scraped]], tick_labels=["intact", "scraped"], showfliers=False)
        a.axhline(1.0, color="#dd8452", ls="--", lw=1)
        a.set_title(ttl)
    fig.suptitle(f"Witness LPS error: scraped vs intact (intact n={intact.sum()}, scraped n={scraped.sum()})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "lps_scraping_box.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 2: error vs surviving fraction ----
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].scatter(frac, thin_t, s=5, alpha=0.25, color=C_T, label="true")
    ax[0].scatter(frac, thin_f, s=5, alpha=0.25, color=C_F, label="flow")
    ax[0].set_ylabel("z-pz thinness"); ax[0].legend(markerscale=2, fontsize=9)
    ax[1].scatter(frac, dcorr, s=5, alpha=0.25, color="#555")
    ax[1].set_ylabel("|Δ z-pz correlation|")
    ax[2].scatter(frac, ez_ratio, s=5, alpha=0.25, color="#555")
    ax[2].axhline(1.0, color="#dd8452", ls="--", lw=1); ax[2].set_ylabel("long-emit ratio flow/true")
    for a in ax:
        a.axvline(INTACT, color="k", ls=":", lw=1); a.set_xlabel("witness surviving fraction")
    fig.suptitle("Witness LPS error vs surviving fraction (dotted = intact/scraped split)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "lps_scraping_trend.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Fig 3: example z-pz overlays (2 intact, 2 scraped) ----
    pick = {"intact": rows[intact][np.argsort(-frac[intact])[:2]],
            "scraped": rows[scraped][np.argsort(frac[scraped])[:2]]}  # most-scraped
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for col, grp in enumerate(("intact", "scraped")):
        for r2, row in enumerate(pick[grp]):
            a = axes[r2, col]
            t = ds.witness[row].numpy() * ws + wm
            f = b.flow(row, 1, 4096)
            pzref = t[:, 5].mean()
            a.scatter(t[:, 2] * 1e6, (t[:, 5] - pzref) * 1e-6, s=3, alpha=0.3, color=C_T, label="true")
            a.scatter(f[:, 2] * 1e6, (f[:, 5] - pzref) * 1e-6, s=3, alpha=0.3, color=C_F, label="flow")
            a.set_xlabel("z [µm]"); a.set_ylabel("Δpz [MeV/c]")
            a.set_title(f"{grp}: row {row}, frac {ds.witness_frac[row]:.2f}")
            if r2 == 0 and col == 0:
                a.legend(markerscale=3, fontsize=9)
    fig.suptitle("Witness z-pz: true vs flow — intact (left) vs scraped (right)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "lps_scraping_examples.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(out / "lps_scraping_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))
    print("wrote", out / "lps_scraping_{box,trend,examples}.png")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--processed", default=str(repo_root() / "processed" / "twobunch_flow_v2.h5"))
    ap.add_argument("--out", default="results/surrogate/v2/diagnostics")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    analyze(ckpt, args.processed, args.out)


if __name__ == "__main__":
    main()
