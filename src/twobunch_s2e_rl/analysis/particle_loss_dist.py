"""Particle-loss distribution in the two-bunch LHS campaign + a suggested zero-loss config.

Reads the processed flow campaign h5 (default processed/twobunch_flow_v4.h5; fields
drive_frac/witness_frac are the surviving fractions T, *_density mark rows with a real cloud)
and answers: how much particle loss does the dataset actually contain, and where in knob space
is loss minimized? This grounds the MBRL survival target -- the campaign supports both-bunch
T>=0.99 (~14% of samples) but T>=0.995 in 0% of samples, so the reward should not chase >0.99.

Writes artifacts/figures/particle_loss.png:
  (a) loss = 1 - T histograms (drive vs witness), full range + a 0-10% zoom
  (b) "both bunches survive at T >= threshold" fraction-vs-threshold curve (the 0.99->0.995 cliff)
  (c) per-knob clean-vs-dirty separation (std units) -- which knob drives clean joint survival

and prints the threshold table + the suggested "aim for zero loss" config (clean-subset physical
medians) to stdout.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python -m twobunch_s2e_rl.analysis.particle_loss_dist
       [--h5 processed/twobunch_flow_v4.h5] [--clean-thr 0.99]
"""
from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datagen.paths import repo_root
from ..datagen.sweep_params import PARAM_KEYS, SWEEP_PARAMS

DRIVE_C = "#1f77b4"
WIT_C = "#d62728"
THRESHOLDS = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)


def _load(h5_path: str):
    with h5py.File(h5_path, "r") as h:
        d = dict(
            df=h["drive_frac"][...].astype(np.float64),
            wf=h["witness_frac"][...].astype(np.float64),
            dd=h["drive_density"][...].astype(bool),
            wd=h["witness_density"][...].astype(bool),
            knobs=h["knobs"][...].astype(np.float64),   # raw physical
        )
    return d


def _norm_knobs(knobs: np.ndarray) -> np.ndarray:
    lo = np.array([SWEEP_PARAMS[k][0] for k in PARAM_KEYS])
    hi = np.array([SWEEP_PARAMS[k][1] for k in PARAM_KEYS])
    return (knobs - lo) / (hi - lo)


def _pctl_line(a: np.ndarray, name: str) -> str:
    a = a[np.isfinite(a)]
    qs = [5, 25, 50, 75, 90, 95, 99]
    body = "  ".join(f"p{q}={np.percentile(a, q):.4f}" for q in qs)
    return f"  {name:14s} n={a.size:5d} mean={a.mean():.4f}  {body}"


def analyze(h5_path: str, clean_thr: float = 0.99):
    d = _load(h5_path)
    df, wf, dd, wd = d["df"], d["wf"], d["dd"], d["wd"]
    both = dd & wd
    loss_d = 1.0 - df[dd]
    loss_w = 1.0 - wf[wd]

    print(f"\n=== particle loss in {h5_path} (N={len(df)}) ===")
    print(f"  drive density-valid {dd.sum()}   witness density-valid {wd.sum()}   both {both.sum()}")
    print("\n  surviving fraction T (density-valid rows):")
    print(_pctl_line(df[dd], "T_drive"))
    print(_pctl_line(wf[wd], "T_witness"))
    print("\n  particle LOSS = 1 - T (density-valid rows):")
    print(_pctl_line(loss_d, "loss_drive"))
    print(_pctl_line(loss_w, "loss_witness"))

    print("\n  fraction of samples surviving at T >= threshold:")
    print(f"  {'thr':>7s}  {'drive':>8s}  {'witness':>8s}  {'BOTH':>8s}  {'#both':>6s}")
    for thr in THRESHOLDS:
        fd = (df[dd] >= thr).mean()
        fw = (wf[wd] >= thr).mean()
        jb = ((df >= thr) & (wf >= thr) & both).sum()
        print(f"  {thr:7.3f}  {fd*100:7.1f}%  {fw*100:7.1f}%  {jb/both.sum()*100:7.1f}%  {jb:6d}")

    # clean basin = both bunches survive at >= clean_thr
    clean = both & (df >= clean_thr) & (wf >= clean_thr)
    dirty = both & ~clean
    kn = _norm_knobs(d["knobs"])
    print(f"\n=== suggested 'aim for zero loss' config (medians of both-T>={clean_thr} subset, "
          f"n={clean.sum()}) ===")
    print(f"  {'knob':16s} {'baseline':>14s} {'suggested':>14s} {'norm[p10,p90]':>22s} {'sep(std)':>9s}")
    sep = {}
    for j, k in enumerate(PARAM_KEYS):
        base = SWEEP_PARAMS[k][2]
        cmed = np.median(d["knobs"][clean, j])
        n10, n90 = np.percentile(kn[clean, j], [10, 90])
        s = (kn[clean, j].mean() - kn[dirty, j].mean()) / (kn[both, j].std() + 1e-9)
        sep[k] = s
        print(f"  {k:16s} {base:14.4g} {cmed:14.4g}   [{n10:6.3f},{n90:6.3f}]  {s:+9.3f}")
    lead = max(sep, key=lambda k: abs(sep[k]))
    print(f"  -> dominant lever: {lead} (separation {sep[lead]:+.3f} std); larger |sep| = more decisive.")

    _plot(h5_path, loss_d, loss_w, df, wf, dd, wd, both, kn, clean, dirty, sep, clean_thr)
    return clean, sep


def _plot(h5_path, loss_d, loss_w, df, wf, dd, wd, both, kn, clean, dirty, sep, clean_thr):
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) full-range loss histograms (log-y)
    bins = np.linspace(0, 1, 51)
    ax[0, 0].hist(loss_d, bins=bins, color=DRIVE_C, alpha=0.6, label=f"drive (n={loss_d.size})")
    ax[0, 0].hist(loss_w, bins=bins, color=WIT_C, alpha=0.6, label=f"witness (n={loss_w.size})")
    ax[0, 0].set_yscale("log")
    ax[0, 0].set_xlabel("particle loss  1 - T"); ax[0, 0].set_ylabel("samples (log)")
    ax[0, 0].set_title("(a) particle-loss distribution (density-valid rows)")
    ax[0, 0].legend()

    # (b) zoom on the clean region [0, 0.1]
    zb = np.linspace(0, 0.1, 41)
    ax[0, 1].hist(np.clip(loss_d, 0, 0.1), bins=zb, color=DRIVE_C, alpha=0.6, label="drive")
    ax[0, 1].hist(np.clip(loss_w, 0, 0.1), bins=zb, color=WIT_C, alpha=0.6, label="witness")
    ax[0, 1].axvline(1 - clean_thr, color="k", ls="--", lw=1,
                     label=f"clean thr (loss={1-clean_thr:.2f})")
    ax[0, 1].set_xlabel("particle loss  1 - T  (zoom 0-10%)"); ax[0, 1].set_ylabel("samples")
    ax[0, 1].set_title("(b) clean-region zoom"); ax[0, 1].legend()

    # (c) survival fraction vs threshold (drive / witness / both) -- the cliff
    thr = np.linspace(0.90, 1.0, 101)
    fd = np.array([(df[dd] >= t).mean() for t in thr]) * 100
    fw = np.array([(wf[wd] >= t).mean() for t in thr]) * 100
    fj = np.array([(((df >= t) & (wf >= t) & both).sum() / both.sum()) for t in thr]) * 100
    ax[1, 0].plot(thr, fd, color=DRIVE_C, label="drive")
    ax[1, 0].plot(thr, fw, color=WIT_C, label="witness")
    ax[1, 0].plot(thr, fj, color="k", lw=2, label="BOTH")
    ax[1, 0].axvline(clean_thr, color="gray", ls=":", lw=1)
    ax[1, 0].set_xlabel("survival threshold T*"); ax[1, 0].set_ylabel("% of (both-valid) samples with T >= T*")
    ax[1, 0].set_title("(c) survival-fraction vs threshold (note the 0.99->0.995 cliff)")
    ax[1, 0].legend()

    # (d) per-knob clean-vs-dirty separation (std units) -- which knob drives clean joint survival
    keys = PARAM_KEYS
    vals = [sep[k] for k in keys]
    cols = ["#2ca02c" if abs(v) == max(abs(x) for x in vals) else "#888888" for v in vals]
    yp = np.arange(len(keys))
    ax[1, 1].barh(yp, vals, color=cols)
    ax[1, 1].set_yticks(yp); ax[1, 1].set_yticklabels(keys, fontsize=8)
    ax[1, 1].axvline(0, color="k", lw=0.8)
    ax[1, 1].set_xlabel("clean - dirty knob mean (std units)")
    ax[1, 1].set_title(f"(d) knob lever for both-T>={clean_thr} (green = dominant)")
    ax[1, 1].invert_yaxis()

    fig.suptitle(f"Two-bunch particle loss -- {os.path.basename(h5_path)}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = repo_root() / "artifacts" / "figures"
    os.makedirs(out, exist_ok=True)
    p = out / "particle_loss.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  wrote {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", default=str(repo_root() / "processed" / "twobunch_flow_v4.h5"))
    ap.add_argument("--clean-thr", type=float, default=0.99,
                    help="both-bunch T threshold defining the clean (zero-loss) subset")
    args = ap.parse_args()
    analyze(args.h5, clean_thr=args.clean_thr)


if __name__ == "__main__":
    main()
