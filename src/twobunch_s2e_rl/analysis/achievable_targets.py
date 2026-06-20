"""Achievable-target diagnostics for an expanded (26-D) sweep.

Answers: across the swept box, what objective values are actually *reachable*, and how often?
This is the pilot's decision tool -- in particular the FF box-LHS verdict (do enough draws
land near-matched, or should the full run switch FF to a manifold-anchored sampler?).

Reads the per-sample JSONs in data/<subdir> (default expanded_pilot) via
build_dataset.flatten_sample, then computes/plots over the LHS subset:
  - viability/survival (transmission, witness-resolved fraction)
  - matching: PENT BMAG (projected + sliced) for drive & witness; near-matched fractions
  - per-bunch emittance, bunch length, spacing, energy difference (achievable ranges)
  - collinearity: transverse centroid offset & angular misalignment (sub-10 um target)
  - an achievable-target "funnel" (cumulative fraction passing successively stricter cuts)
  - BMAG vs FF-quad off-manifold distance (the box-LHS diagnostic)
Golden working point shown from the baseline-repeat samples.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python -m twobunch_s2e_rl.analysis.achievable_targets [subdir]
Outputs: artifacts/figures/<subdir>/*.png and artifacts/<subdir>_achievable_summary.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datagen.paths import repo_root
from ..datagen.sweep_params import SWEEP_PARAMS_EXPANDED_EXTRA
from .build_dataset import flatten_sample

FF_QUADS = ["Q5FFkG", "Q4FFkG", "Q3FFkG", "Q2FFkG", "Q1FFkG", "Q0FFkG"]
BLUE, ORANGE, GREEN, RED = "#4c72b0", "#dd8452", "#55a868", "#c44e52"


def load(subdir):
    files = sorted(glob.glob(str(repo_root() / "data" / subdir / "sample_*.json")))
    if not files:
        raise SystemExit(f"no sample_*.json under data/{subdir}")
    df = pd.DataFrame(flatten_sample(json.load(open(f))) for f in files)
    return df.sort_values("idx").reset_index(drop=True)


def P(df, metric):
    """PENT column as float array (NaN where absent, e.g. witness not resolved)."""
    col = f"PENT__{metric}"
    return df[col].to_numpy(dtype=float) if col in df else np.full(len(df), np.nan)


def derived(df):
    """Display-unit target quantities at PENT (per row; NaN where undefined)."""
    ang = np.sqrt((P(df, "PDrive_median_xp") - P(df, "PWitness_median_xp"))**2 +
                  (P(df, "PDrive_median_yp") - P(df, "PWitness_median_yp"))**2) * 1e6  # urad
    bmag_max = np.nanmax(np.vstack([P(df, f"P{b}_BMAG_{a}")
                                    for b in ("Drive", "Witness") for a in ("x", "y")]), axis=0)
    return {
        "transmission":      P(df, "transmission_total"),                       # fraction
        "spacing":           P(df, "bunchSpacing") * 1e6,                        # um (signed)
        "offset":            P(df, "transverseCentroidOffset") * 1e6,           # um
        "angle":             ang,                                                # urad
        "dE":               (P(df, "PDrive_median_energy") - P(df, "PWitness_median_energy")) * 1e-6,  # MeV
        "drive_emit_x":      P(df, "PDrive_norm_emit_x") * 1e6,                  # um-rad
        "drive_emit_y":      P(df, "PDrive_norm_emit_y") * 1e6,
        "witness_emit_x":    P(df, "PWitness_norm_emit_x") * 1e6,
        "witness_emit_y":    P(df, "PWitness_norm_emit_y") * 1e6,
        "drive_BMAG_x":      P(df, "PDrive_BMAG_x"),
        "drive_BMAG_y":      P(df, "PDrive_BMAG_y"),
        "witness_BMAG_x":    P(df, "PWitness_BMAG_x"),
        "witness_BMAG_y":    P(df, "PWitness_BMAG_y"),
        "BMAG_max":          bmag_max,                                           # worst of the 4
        "drive_sigz":        P(df, "PDrive_sigmaSI90_z") * 1e6,                  # um
        "witness_sigz":      P(df, "PWitness_sigmaSI90_z") * 1e6,
    }


def ff_offmanifold(df):
    """Per-row L2 distance of the 6 FF quads from golden, in units of each half-range."""
    z = np.zeros(len(df))
    for q in FF_QUADS:
        lo, hi, base = SWEEP_PARAMS_EXPANDED_EXTRA[q]
        half = (hi - lo) / 2.0
        z = z + ((df[q].to_numpy(dtype=float) - base) / half) ** 2
    return np.sqrt(z)


# (key, label, unit, log?)
TARGETS = [
    ("spacing", "bunch spacing", "um", False),
    ("offset", "transverse offset", "um", True),
    ("angle", "angular misalignment", "urad", True),
    ("dE", "energy difference (drive-witness)", "MeV", False),
    ("drive_emit_x", "drive norm.emit x", "um-rad", True),
    ("drive_emit_y", "drive norm.emit y", "um-rad", True),
    ("witness_emit_x", "witness norm.emit x", "um-rad", True),
    ("witness_emit_y", "witness norm.emit y", "um-rad", True),
    ("drive_sigz", "drive bunch length sigma_z", "um", False),
    ("witness_sigz", "witness bunch length sigma_z", "um", False),
]


def hist(ax, v, unit, log, golden=None):
    v = v[np.isfinite(v)]
    if log:
        v = v[v > 0]
        bins = np.logspace(np.log10(v.min()), np.log10(v.max()), 45) if v.size else 45
        ax.set_xscale("log")
    else:
        bins = 45
    ax.hist(v, bins=bins, color=BLUE, edgecolor="white", lw=0.3)
    med = np.median(v)
    ax.axvline(med, color=ORANGE, lw=2, label=f"median {med:.3g}")
    if golden is not None and np.isfinite(golden):
        ax.axvline(golden, color=GREEN, lw=2, ls="--", label=f"golden {golden:.3g}")
    ax.set_xlabel(unit, fontsize=9)
    ax.legend(fontsize=8)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("subdir", nargs="?", default="expanded_pilot")
    args = ap.parse_args()

    df = load(args.subdir)
    figdir = repo_root() / "artifacts" / "figures" / args.subdir
    os.makedirs(figdir, exist_ok=True)

    lhs = df[(df["success"] == True) & (df["is_baseline_repeat"] == False)].copy()
    base = df[df["is_baseline_repeat"] == True].copy()
    n = len(lhs)
    q_lhs, q_base = derived(lhs), derived(base)
    golden = {k: np.nanmedian(q_base[k]) for k in q_lhs}  # golden ref = baseline-repeat median

    witness_ok = np.isfinite(q_lhs["witness_BMAG_x"])      # witness resolved as a 2nd bunch
    n_wit = int(witness_ok.sum())
    print(f"\n=== {args.subdir}: {len(df)} samples ({n} LHS + {len(base)} baseline repeats) ===")
    print(f"all tracked successfully; witness resolved in {n_wit}/{n} = {n_wit/n:.0%} of LHS draws")

    # ---- feasibility fractions over the LHS box ----------------------------------------
    tm = q_lhs["transmission"]
    bm = q_lhs["BMAG_max"]
    off = q_lhs["offset"]
    cuts = [
        ("tracked", np.ones(n, bool)),
        ("witness resolved", witness_ok),
        ("transmission > 0.90", tm > 0.90),
        ("transmission > 0.99", tm > 0.99),
        ("BMAG_max < 2", bm < 2),
        ("BMAG_max < 1.5", bm < 1.5),
        ("offset < 10 um", off < 10),
    ]
    print("\n-- feasibility fractions (LHS box) --")
    for name, m in cuts:
        print(f"   {name:24s} {int(np.nansum(m)):4d}/{n}  = {np.nansum(m)/n:6.1%}")
    # joint near-deliverable target
    joint = witness_ok & (tm > 0.99) & (bm < 1.5) & (off < 10)
    print(f"   {'JOINT (all above)':24s} {int(joint.sum()):4d}/{n}  = {joint.sum()/n:6.1%}")

    # ---- achievable ranges over the witness-viable subset ------------------------------
    print("\n-- achievable ranges (witness-resolved subset, p5 / p50 / p95) --")
    rows = []
    for k, label, unit, _ in [("transmission", "transmission", "frac", False)] + TARGETS \
            + [("BMAG_max", "BMAG_max (worst of 4)", "-", False)]:
        v = q_lhs[k][witness_ok]
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        g = golden.get(k, np.nan)
        print(f"   {label:30s}[{unit:7s}] {p5:11.3g} {p50:11.3g} {p95:11.3g}   golden {g:.3g}")
        rows.append(dict(target=label, unit=unit, p5=p5, p50=p50, p95=p95,
                         min=v.min(), max=v.max(), golden=g))
    summ = repo_root() / "artifacts" / f"{args.subdir}_achievable_summary.csv"
    pd.DataFrame(rows).to_csv(summ, index=False)
    print(f"\nwrote {summ}")

    # ---- FIG 1: viability ---------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].hist(tm[np.isfinite(tm)], bins=40, color=BLUE, edgecolor="white", lw=0.3)
    ax[0].axvline(np.nanmedian(tm), color=ORANGE, lw=2, label=f"median {np.nanmedian(tm):.2f}")
    ax[0].axvline(np.nanmedian(q_base["transmission"]), color=GREEN, ls="--", lw=2,
                  label=f"golden {np.nanmedian(q_base['transmission']):.2f}")
    ax[0].set_title("total transmission"); ax[0].set_xlabel("fraction"); ax[0].legend(fontsize=8)
    wfrac = P(lhs, "n_live_witness") / np.where(P(lhs, "n_live_total") > 0, P(lhs, "n_live_total"), np.nan)
    wfrac = wfrac[np.isfinite(wfrac)]
    ax[1].hist(wfrac, bins=40, color=BLUE, edgecolor="white", lw=0.3)
    ax[1].set_title("witness / total live charge"); ax[1].set_xlabel("fraction")
    names = [c[0] for c in cuts] + ["JOINT"]
    fracs = [np.nansum(m) / n for _, m in cuts] + [joint.sum() / n]
    ax[2].barh(range(len(names)), fracs, color=BLUE)
    ax[2].set_yticks(range(len(names))); ax[2].set_yticklabels(names, fontsize=8)
    ax[2].invert_yaxis(); ax[2].set_xlim(0, 1); ax[2].set_xlabel("fraction of LHS box")
    ax[2].set_title("achievable-target funnel")
    for i, f in enumerate(fracs):
        ax[2].text(min(f + 0.01, 0.9), i, f"{f:.0%}", va="center", fontsize=8)
    fig.suptitle(f"{args.subdir}: viability  (n={n} LHS draws)", fontsize=13)
    fig.tight_layout(); fig.savefig(figdir / "viability.png", dpi=130); plt.close(fig)

    # ---- FIG 2: matching (BMAG) ---------------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    bmag_panels = [("drive_BMAG_x", "drive BMAG_x"), ("drive_BMAG_y", "drive BMAG_y"),
                   ("BMAG_max", "BMAG max (worst of 4)"),
                   ("witness_BMAG_x", "witness BMAG_x"), ("witness_BMAG_y", "witness BMAG_y")]
    for a, (k, t) in zip(ax.ravel(), bmag_panels):
        v = q_lhs[k][witness_ok] if "witness" in k or k == "BMAG_max" else q_lhs[k]
        v = v[np.isfinite(v)]
        vp = v[v > 0]
        bins = np.logspace(np.log10(max(vp.min(), 0.5)), np.log10(vp.max()), 45)
        a.hist(vp, bins=bins, color=BLUE, edgecolor="white", lw=0.3); a.set_xscale("log")
        a.axvline(1.0, color="k", ls=":", lw=1)
        a.axvline(np.median(vp), color=ORANGE, lw=2, label=f"median {np.median(vp):.2f}")
        a.set_title(t); a.set_xlabel("BMAG"); a.legend(fontsize=8)
    # near-matched fraction text
    nm15 = np.nanmean(q_lhs["BMAG_max"][witness_ok] < 1.5)
    nm2 = np.nanmean(q_lhs["BMAG_max"][witness_ok] < 2.0)
    ax.ravel()[5].axis("off")
    ax.ravel()[5].text(0.05, 0.6, f"witness-viable: {n_wit}/{n} = {n_wit/n:.0%}\n"
                       f"BMAG_max < 1.5: {nm15:.0%} of viable\n"
                       f"BMAG_max < 2.0: {nm2:.0%} of viable",
                       fontsize=12, va="center")
    fig.suptitle(f"{args.subdir}: PENT matching (BMAG=1 is matched)", fontsize=13)
    fig.tight_layout(); fig.savefig(figdir / "matching_bmag.png", dpi=130); plt.close(fig)

    # ---- FIG 3: achievable target distributions ----------------------------------------
    fig, ax = plt.subplots(2, 5, figsize=(22, 8.5))
    for a, (k, label, unit, log) in zip(ax.ravel(), TARGETS):
        hist(a, q_lhs[k][witness_ok], unit, log, golden=golden.get(k))
        a.set_title(label, fontsize=10)
    fig.suptitle(f"{args.subdir}: achievable target ranges (witness-viable subset, n={n_wit})",
                 fontsize=13)
    fig.tight_layout(); fig.savefig(figdir / "achievable_targets.png", dpi=130); plt.close(fig)

    # ---- FIG 4: box-LHS diagnostic -- BMAG vs FF off-manifold distance ------------------
    z = ff_offmanifold(lhs)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(z[witness_ok], q_lhs["BMAG_max"][witness_ok], s=10, alpha=0.5, color=BLUE)
    ax[0].scatter(z[~witness_ok], np.full((~witness_ok).sum(), 0.6), s=12, marker="x",
                  color=RED, label="witness lost")
    ax[0].axhline(1.5, color=GREEN, ls="--", lw=1, label="BMAG=1.5")
    ax[0].set_yscale("log"); ax[0].set_xlabel("FF off-manifold distance  (|FF-golden| / half-range)")
    ax[0].set_ylabel("BMAG max"); ax[0].set_title("matching vs FF-quad excursion"); ax[0].legend(fontsize=8)
    ax[1].scatter(z, tm, s=10, alpha=0.5, color=BLUE)
    ax[1].set_xlabel("FF off-manifold distance"); ax[1].set_ylabel("transmission")
    ax[1].set_title("survival vs FF-quad excursion")
    fig.suptitle(f"{args.subdir}: independent FF box-LHS -> off-manifold mismatch", fontsize=13)
    fig.tight_layout(); fig.savefig(figdir / "ff_box_lhs_diagnostic.png", dpi=130); plt.close(fig)

    print(f"wrote 4 figures to {figdir}")


if __name__ == "__main__":
    main()
