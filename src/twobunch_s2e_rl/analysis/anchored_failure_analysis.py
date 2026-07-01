"""Why did FF manifold-anchoring fail to transfer to the real two-bunch beam?

Loads the anchored-pilot PENT clouds once, scores each against its OWN target beta*, and tests
three candidate causes with discriminating figures + correlations:

  H_B (joint manifold / upstream swamp): the FF curve assumes a fixed upstream beam, but the 8
      longitudinal knobs (phases, energies, BC20 sextupole STRENGTHS = chromatic correction) are
      box-sampled full-range. Prediction: match quality degrades with upstream excursion; the
      near-golden-upstream subset should recover fidelity (achieved beta tracks beta*).
  H_C (chromatic witness): the ~181 MeV driver-witness energy difference defocuses the witness
      through the chromatic FF. Prediction: witness slice-beta grows with |dE|.
  (H_A design-vs-beam incoming Twiss is the residual once H_B/H_C are accounted for.)

Writes a per-sample CSV (artifacts/) and figures to lab-notebook/images/.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.anchored_failure_analysis
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
from ..datagen.sweep_params import resolve_sweep_set, PARAM_KEYS
from ..datagen.ff_manifold import FF_KEYS
from ..surrogate.properties import twiss_bmag, slice_twiss_bmag, _energy

SUB = "expanded_anchored_pilot"
COORD = ["x", "y", "z", "px", "py", "pz"]
FIGDIR = "/home/rwu4/photoinjector-rl/lab-notebook/images"
MC2 = 0.51099895e6  # eV, electron


def _to_torch(pg):
    return torch.tensor(np.stack([getattr(pg, k) for k in COORD], axis=1),
                        dtype=torch.float64).unsqueeze(0)


def _excursion(knobs, keys, base, half):
    """RMS normalized deviation of a knob subset from golden (per-knob half-range units)."""
    return float(np.sqrt(np.mean([((knobs[k] - base[k]) / half[k]) ** 2 for k in keys])))


def build_table():
    ddir = repo_root() / "data" / SUB
    man = {m["idx"]: m for m in json.load(open(ddir / "manifest.json"))}
    keys, low, high, base = resolve_sweep_set("expanded_anchored")
    half = {k: (high[i] - low[i]) / 2 for i, k in enumerate(keys)}
    base = {k: base[k] for k in keys}

    rows = []
    files = sorted(glob.glob(str(ddir / "sample_*_PENT.h5")))
    for i, f in enumerate(files):
        idx = int(re.search(r"sample_(\d+)_PENT", f).group(1))
        m = man.get(idx, {})
        if m.get("is_baseline_repeat") or m.get("block") != "anchor":
            continue                                  # focus on the anchor block
        kn = m["knobs"]
        beta_t = float(m.get("ff_target_beta_m") or 0.5)
        res = qs.getDriverAndWitness(ParticleGroup(h5=f))
        if res is None or res[1] is None or len(res[1]) < 50:
            continue
        d, w = _to_torch(res[0]), _to_torch(res[1])
        sd = slice_twiss_bmag(d, beta0=beta_t)
        sw = slice_twiss_bmag(w, beta0=beta_t)
        dE = float((_energy(d).mean() - _energy(w).mean()) * 1e-6)        # MeV
        rows.append(dict(
            idx=idx, beta_t=beta_t,
            up_exc=_excursion(kn, PARAM_KEYS, base, half),               # longitudinal/upstream
            ff_exc=_excursion(kn, FF_KEYS, base, half),                  # FF-curve distance
            sext_exc=_excursion(kn, ["S1ELkG", "S2ELkG", "S3ELkG"], base, half),
            drive_beta_y=float(sd["slice_beta_y_core"]),
            witness_beta_y=float(sw["slice_beta_y_core"]),
            drive_bmag=float(sd["slice_bmag_max"]),
            witness_bmag=float(sw["slice_bmag_max"]),
            dE=dE,
        ))
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}")
    return rows


def main():
    cache = repo_root() / "artifacts" / f"{SUB}_failure.csv"
    if cache.exists():
        import csv
        rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(cache))]
        print(f"loaded {len(rows)} rows from {cache}")
    else:
        rows = build_table()
        import csv
        with open(cache, "w", newline="") as fh:
            wcsv = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wcsv.writeheader(); wcsv.writerows(rows)
        print(f"wrote {cache} ({len(rows)} anchor rows)")

    bt = np.array([r["beta_t"] for r in rows]) * 100        # cm
    up = np.array([r["up_exc"] for r in rows])
    ff = np.array([r["ff_exc"] for r in rows])
    dby = np.array([r["drive_beta_y"] for r in rows]) * 100
    wby = np.array([r["witness_beta_y"] for r in rows]) * 100
    wbm = np.array([r["witness_bmag"] for r in rows])
    dbm = np.array([r["drive_bmag"] for r in rows])
    dE = np.abs(np.array([r["dE"] for r in rows]))

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1])

    print("\n=== discriminating correlations (anchor block, n=%d) ===" % len(rows))
    print(f"H_B upstream-swamp:  corr(up_exc, log drive_bmag)  = {corr(up, np.log(dbm)):+.2f}")
    print(f"                     corr(up_exc, log witness_bmag)= {corr(up, np.log(wbm)):+.2f}")
    print(f"                     corr(ff_exc, log witness_bmag)= {corr(ff, np.log(wbm)):+.2f}  (FF should matter less)")
    print(f"H_C chromatic:       corr(|dE|,   log witness_beta)= {corr(dE, np.log(wby)):+.2f}")
    print(f"                     corr(|dE|,   log drive_beta)  = {corr(dE, np.log(dby)):+.2f}  (drive ~ on-energy)")
    # fidelity overall vs near-golden-upstream subset (H_B's key prediction)
    lo = up <= np.percentile(up, 25)
    print(f"\nfidelity corr(log beta*, log achieved drive_beta_y):")
    print(f"   all anchor          rho = {corr(np.log(bt), np.log(dby)):+.2f}")
    print(f"   near-golden upstream rho = {corr(np.log(bt[lo]), np.log(dby[lo])):+.2f}  "
          f"(up_exc <= {np.percentile(up,25):.2f}, n={lo.sum()})")
    print(f"   drive slice-beta_y [cm]: all median {np.median(dby):.0f}, "
          f"near-golden median {np.median(dby[lo]):.0f}")

    # ---- figures -----------------------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))
    # (a) fidelity colored by upstream excursion
    sc = ax[0, 0].scatter(bt, dby, c=up, s=16, cmap="viridis_r")
    lim = [min(bt.min(), dby.min()), max(bt.max(), dby.max())]
    ax[0, 0].plot(lim, lim, "k--", lw=1, label="achieved = target")
    ax[0, 0].set_xscale("log"); ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("target beta* [cm]"); ax[0, 0].set_ylabel("achieved DRIVE slice-beta_y [cm]")
    ax[0, 0].set_title("(a) fidelity vs target, coloured by upstream excursion"); ax[0, 0].legend(fontsize=8)
    plt.colorbar(sc, ax=ax[0, 0], label="upstream excursion (RMS)")
    # (b) the discriminator: witness BMAG vs upstream vs FF excursion
    ax[0, 1].scatter(up, wbm, s=16, alpha=0.6, color="#4c72b0", label=f"vs upstream (rho={corr(up,np.log(wbm)):+.2f})")
    ax[0, 1].scatter(ff, wbm, s=16, alpha=0.6, color="#dd8452", marker="^",
                     label=f"vs FF (rho={corr(ff,np.log(wbm)):+.2f})")
    ax[0, 1].set_yscale("log"); ax[0, 1].axhline(1.5, color="g", ls="--", lw=1)
    ax[0, 1].set_xlabel("excursion (RMS, half-range units)")
    ax[0, 1].set_ylabel("witness slice-BMAG vs own beta*")
    ax[0, 1].set_title("(b) match degrades with UPSTREAM, not FF, excursion"); ax[0, 1].legend(fontsize=8)
    # (c) chromatic: witness slice-beta vs |dE|, drive overlaid
    ax[1, 0].scatter(dE, wby, s=16, alpha=0.6, color="#c44e52", label=f"witness (rho={corr(dE,np.log(wby)):+.2f})")
    ax[1, 0].scatter(dE, dby, s=16, alpha=0.6, color="#4c72b0", label=f"drive (rho={corr(dE,np.log(dby)):+.2f})")
    ax[1, 0].set_yscale("log")
    ax[1, 0].set_xlabel("|driver-witness energy difference| [MeV]")
    ax[1, 0].set_ylabel("achieved slice-beta_y [cm]")
    ax[1, 0].set_title("(c) chromatic: witness defocus grows with dE"); ax[1, 0].legend(fontsize=8)
    # (d) near-golden-upstream subset fidelity
    ax[1, 1].scatter(bt[~lo], dby[~lo], s=14, alpha=0.3, color="0.6", label="rest")
    ax[1, 1].scatter(bt[lo], dby[lo], s=24, color="#4c72b0",
                     label=f"near-golden upstream (rho={corr(np.log(bt[lo]),np.log(dby[lo])):+.2f})")
    ax[1, 1].plot(lim, lim, "k--", lw=1)
    ax[1, 1].set_xscale("log"); ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("target beta* [cm]"); ax[1, 1].set_ylabel("achieved DRIVE slice-beta_y [cm]")
    ax[1, 1].set_title("(d) does fidelity recover near golden upstream?"); ax[1, 1].legend(fontsize=8)
    fig.suptitle("Why FF anchoring did not transfer: the matched manifold is JOINT over all knobs",
                 fontsize=13)
    fig.tight_layout()
    p = f"{FIGDIR}/2026-06-21_anchored_failure.png"
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
