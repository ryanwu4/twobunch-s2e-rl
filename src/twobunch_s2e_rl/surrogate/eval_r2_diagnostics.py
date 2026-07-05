"""Diagnose WHY per-observable R² is low: metric artifact (log-space tails / near-scraped
witnesses) vs genuine model error.

For each parity observable, recompute agreement several ways and attribute the log-space
residual to the value tails and to witness particle count:
  - r2_log     : the eval.py metric (R² in log10 space)  -- tail-sensitive
  - r2_linear  : R² in linear space
  - spearman   : rank correlation (monotonic, scale/tail-insensitive)
  - r2_log_trim: r2_log after dropping samples outside [p2,p98] of the TRUE value
  - ss_res share from the smallest-true decile and largest-true decile (which tail hurts)
  - r2_log restricted to well-resolved witnesses (n_witness >= median)  -- near-scraped test
  - Spearman(|Δlog10|, n_witness): do errors concentrate on low-count witnesses?

Makes a diagnostic figure for the weak observables: log parity colored by n_witness + residual
vs true, and a bar chart of the R² variants.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python -m twobunch_s2e_rl.surrogate.eval_r2_diagnostics --ckpt '<glob>' --processed <h5> [--out <dir>]
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from ..datagen.paths import repo_root
from .model import TwoBunchFlow
from .dataset import TwoBunchFlowDataModule
from .properties import per_bunch, inter_bunch
from .eval import PARITY_KEYS, _destd

WEAK_FOCUS = ["witness_norm_emit_x", "witness_norm_emit_y", "witness_slice_bmag_max",
              "witness_bmag_y", "witness_bmag_x"]


def _r2(t, p):
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - t.mean()) ** 2) + 1e-30
    return 1 - ss_res / ss_tot


def _variants(t_raw, p_raw, logp, nwit):
    """Return dict of R² variants + tail attribution for one observable (raw linear arrays)."""
    m = np.isfinite(t_raw) & np.isfinite(p_raw)
    t_raw, p_raw = t_raw[m], p_raw[m]
    nwit = nwit[m] if nwit is not None else None
    out = {"n": int(m.sum())}
    # linear
    out["r2_linear"] = float(_r2(t_raw, p_raw))
    out["spearman"] = float(spearmanr(t_raw, p_raw).statistic)
    if logp:
        tt, pp = np.log10(np.abs(t_raw) + 1e-30), np.log10(np.abs(p_raw) + 1e-30)
    else:
        tt, pp = t_raw, p_raw
    out["r2_log"] = float(_r2(tt, pp))
    # trim to [p2,p98] of TRUE
    lo, hi = np.percentile(tt, [2, 98])
    k = (tt >= lo) & (tt <= hi)
    out["r2_log_trim"] = float(_r2(tt[k], pp[k])) if k.sum() > 10 else np.nan
    # decile ss_res attribution (in the space R² is computed)
    resid2 = (tt - pp) ** 2
    order = np.argsort(tt)
    d = len(tt) // 10 or 1
    tot = resid2.sum() + 1e-30
    out["ssres_low_decile"] = float(resid2[order[:d]].sum() / tot)
    out["ssres_high_decile"] = float(resid2[order[-d:]].sum() / tot)
    out["med_abs_logerr"] = float(np.median(np.abs(tt - pp))) if logp else np.nan
    # near-scraped test (witness only)
    if nwit is not None:
        out["spearman_err_vs_nwit"] = float(spearmanr(np.abs(tt - pp), nwit).statistic)
        thr = np.median(nwit)
        good = nwit >= thr
        out["r2_log_wellresolved"] = float(_r2(tt[good], pp[good])) if good.sum() > 10 else np.nan
        out["nwit_median"] = float(thr)
    return out, (tt, pp, nwit)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--out", default="results/surrogate/default/r2")
    ap.add_argument("--n", type=int, default=2048)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt

    model = TwoBunchFlow.load_from_checkpoint(ckpt, map_location=device).eval().to(device)
    dm = TwoBunchFlowDataModule(processed_h5=args.processed, batch_size=256, num_workers=0)
    dm.setup()
    val = dm.val
    val_idx = np.asarray(val.indices)  # into the full h5, for n_witness alignment
    import h5py
    with h5py.File(args.processed) as h:
        nwit_all = h["n_witness"][:] if "n_witness" in h else None
    nwit_val = nwit_all[val_idx] if nwit_all is not None else None

    knobs = torch.stack([val[i]["knobs"] for i in range(len(val))]).to(device)
    flags = {k: torch.stack([val[i][k] for i in range(len(val))]).cpu().numpy().astype(bool)
             for k in ("drive_density", "witness_density")}
    td = _destd(torch.stack([val[i]["drive"] for i in range(len(val))]).to(device),
               model.drive_mean, model.drive_std)
    tw = _destd(torch.stack([val[i]["witness"] for i in range(len(val))]).to(device),
               model.witness_mean, model.witness_std)
    pred = model.observables(knobs, n=args.n)
    true = {}
    for nm, parts in (("drive", td), ("witness", tw)):
        for kk, vv in per_bunch(parts).items():
            true[f"{nm}_{kk}"] = vv
    true.update(inter_bunch(td, tw))

    dd, wd = flags["drive_density"], flags["witness_density"]
    both = dd & wd
    results, raw = {}, {}
    for key, label, logp in PARITY_KEYS:
        bunch = key.split("_")[0]
        mask = wd if bunch == "witness" else dd
        if any(k in key for k in ("spacing", "difference", "offset")):
            mask = both
        t = true[key].cpu().numpy()[mask]
        p = pred[key].cpu().numpy()[mask]
        nw = nwit_val[mask] if (nwit_val is not None and bunch == "witness") else None
        v, arrs = _variants(t, p, logp, nw)
        v["label"] = label
        results[key] = v
        raw[key] = arrs

    os.makedirs(repo_root() / args.out, exist_ok=True)
    with open(repo_root() / args.out / "r2_variants.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- print table ----
    print(f"\n=== R² variants ({ckpt.split('/')[-1]}) ===")
    hdr = f"{'observable':26s} {'r2_log':>7} {'r2_lin':>7} {'spear':>6} {'r2_trim':>7} " \
          f"{'lowdec':>6} {'hidec':>6} {'r2_wellres':>10} {'err~nwit':>8}"
    print(hdr)
    for key, v in results.items():
        print(f"{key:26s} {v['r2_log']:+7.3f} {v['r2_linear']:+7.3f} {v['spearman']:+6.3f} "
              f"{v.get('r2_log_trim', float('nan')):+7.3f} {v['ssres_low_decile']:6.2f} "
              f"{v['ssres_high_decile']:6.2f} {v.get('r2_log_wellresolved', float('nan')):+10.3f} "
              f"{v.get('spearman_err_vs_nwit', float('nan')):+8.3f}")

    # ---- diagnostic figure for the weak observables ----
    focus = [k for k in WEAK_FOCUS if k in raw]
    fig, axes = plt.subplots(2, len(focus), figsize=(4.2 * len(focus), 8))
    for j, key in enumerate(focus):
        tt, pp, nw = raw[key]
        v = results[key]
        # top: log parity colored by n_witness
        ax = axes[0, j]
        c = nw if nw is not None else "steelblue"
        sc = ax.scatter(tt, pp, c=c, s=8, alpha=0.5, cmap="viridis")
        lo, hi = np.percentile(np.concatenate([tt, pp]), [1, 99])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        if nw is not None:
            plt.colorbar(sc, ax=ax, label="n_witness", fraction=0.046)
        ax.set_title(f"{v['label']}\nr2_log={v['r2_log']:.2f} lin={v['r2_linear']:.2f} "
                     f"spear={v['spearman']:.2f}", fontsize=9)
        ax.set_xlabel("log10 true"); ax.set_ylabel("log10 surrogate")
        # bottom: residual vs true
        ax = axes[1, j]
        ax.scatter(tt, pp - tt, c=c, s=8, alpha=0.5, cmap="viridis")
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_xlabel("log10 true"); ax.set_ylabel("log10 resid (pred-true)")
        ax.set_title(f"trim r2={v.get('r2_log_trim', float('nan')):.2f} | "
                     f"wellres r2={v.get('r2_log_wellresolved', float('nan')):.2f}", fontsize=9)
    fig.suptitle("Weak-observable R² diagnostics: is it log-tails / near-scraped, or real?",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(repo_root() / args.out / "r2_diagnostics.png", dpi=130)
    plt.close(fig)
    print(f"\nwrote {args.out}/r2_variants.json + r2_diagnostics.png")


if __name__ == "__main__":
    main()
