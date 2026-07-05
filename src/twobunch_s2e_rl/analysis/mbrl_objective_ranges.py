"""Achievable ranges of each downstream MBRL reward objective, over the combined dataset.

For every term the two-bunch RL reward optimizes (survival, collinearity, per-bunch emittance,
matching, bunch length / peak current, spacing goal), show the range the dataset actually spans
-- i.e. what the agent can be asked to hit -- split into the dense good box and the broad wide
box, with the golden working point marked and the reward direction annotated.

Objectives (reward semantics):
  survival        T_drive/T_witness -> transmission        maximize
  collinearity    transverse_offset, angular_misalignment  minimize
  matching        drive/witness projected BMAG             -> 1 (target); golden ~2 (chirp)
  emittance       drive/witness norm.emit x/y              minimize (floor at golden)
  bunch length    drive/witness sigma_z (90%)              shorter -> higher peak current
  peak current    ~ c*Q / (sqrt(2pi) sigma_z)              maximize (derived)
  spacing         bunch_spacing                            goal-conditioned target

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python -m twobunch_s2e_rl.analysis.mbrl_objective_ranges [good_subdir] [wide_subdir]
Outputs: artifacts/figures/combined_dataset/{mbrl_objective_ranges,mbrl_reachability}.png
         artifacts/combined_mbrl_ranges.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..datagen.paths import repo_root
from .achievable_targets import load, derived, P

BLUE, ORANGE, GREEN, GREY = "#4c72b0", "#dd8452", "#55a868", "#8c8c8c"
C_LIGHT = 2.99792458e8
SQRT_2PI = np.sqrt(2 * np.pi)


def _peak_current_kA(charge_nC, sigz_um):
    """Approx peak current from charge and sigma_z(90%): I = c*Q/(sqrt(2pi) sigma_z)."""
    q = np.asarray(charge_nC, float) * 1e-9
    sz = np.asarray(sigz_um, float) * 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        return (C_LIGHT * q / (SQRT_2PI * sz)) / 1e3


def _objectives(subdir):
    df = load(subdir)
    df = df[df["is_baseline_repeat"] == False].copy()
    q = derived(df)
    wok = np.isfinite(q["witness_BMAG_x"])
    q["drive_Ipk"] = _peak_current_kA(P(df, "PDrive_charge_nC"), q["drive_sigz"])
    q["witness_Ipk"] = _peak_current_kA(P(df, "PWitness_charge_nC"), q["witness_sigz"])
    return q, wok


# (key, group, label, unit, direction, log, witness_gated)
OBJ = [
    ("transmission",       "survival",      "transmission",          "frac",   "max",    False, False),
    ("offset",             "collinearity",  "transverse offset",     "um",     "min",    True,  True),
    ("angle",              "collinearity",  "angular misalignment",  "urad",   "min",    True,  True),
    ("drive_BMAG_x",       "matching",      "drive proj. BMAG_x",    "-",      "target", True,  False),
    ("witness_BMAG_x",     "matching",      "witness proj. BMAG_x",  "-",      "target", True,  True),
    ("witness_BMAG_y",     "matching",      "witness proj. BMAG_y",  "-",      "target", True,  True),
    ("drive_emit_x",       "emittance",     "drive norm.emit x",     "um-rad", "min",    True,  False),
    ("drive_emit_y",       "emittance",     "drive norm.emit y",     "um-rad", "min",    True,  False),
    ("witness_emit_x",     "emittance",     "witness norm.emit x",   "um-rad", "min",    True,  True),
    ("witness_emit_y",     "emittance",     "witness norm.emit y",   "um-rad", "min",    True,  True),
    ("drive_sigz",         "bunch length",  "drive sigma_z (90%)",   "um",     "min",    True,  False),
    ("witness_sigz",       "bunch length",  "witness sigma_z (90%)", "um",     "min",    True,  True),
    ("drive_Ipk",          "peak current",  "drive peak current",    "kA",     "max",    False, False),
    ("witness_Ipk",        "peak current",  "witness peak current",  "kA",     "max",    False, True),
    ("spacing",            "spacing(goal)", "bunch spacing",         "um",     "goal",   False, True),
]
GLYPH = {"min": "min ↓", "max": "max ↑", "target": "target ◎", "goal": "goal (free)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("good", nargs="?", default="tightbox_v2_full")
    ap.add_argument("wide", nargs="?", default="expanded_full")
    args = ap.parse_args()
    figdir = repo_root() / "artifacts" / "figures" / "combined_dataset"
    os.makedirs(figdir, exist_ok=True)

    qg, wg = _objectives(args.good)
    qw, ww = _objectives(args.wide)
    base = load(args.good); base = base[base["is_baseline_repeat"] == True]
    gq = derived(base)
    gq_extra = {
        "drive_Ipk": _peak_current_kA(P(base, "PDrive_charge_nC"), gq["drive_sigz"]),
        "witness_Ipk": _peak_current_kA(P(base, "PWitness_charge_nC"), gq["witness_sigz"]),
    }

    def gval(k):
        v = gq_extra[k] if k in gq_extra else gq[k]
        v = np.abs(v) if k in ("offset", "spacing") else v
        return float(np.nanmedian(v))

    def vals(q, ok, k, gated):
        v = q[k].copy()
        if k in ("offset", "spacing"):
            v = np.abs(v)
        v = v[ok] if gated else v
        return v[np.isfinite(v)]

    # ---- summary table + reachability numbers -----------------------------------------
    rows = []
    for key, grp, label, unit, direction, log, gated in OBJ:
        vc = np.concatenate([vals(qg, wg, key, gated), vals(qw, ww, key, gated)])
        vc = vc[vc > 0] if log else vc
        p5, p50, p95 = np.percentile(vc, [5, 50, 95])
        best = vc.min() if direction in ("min",) else (vc.max() if direction == "max" else np.nan)
        rows.append(dict(group=grp, objective=label, unit=unit, direction=direction,
                         p5=p5, p50=p50, p95=p95, best=best, golden=gval(key)))
    tbl = pd.DataFrame(rows)
    csv = repo_root() / "artifacts" / "combined_mbrl_ranges.csv"
    tbl.to_csv(csv, index=False)
    print(f"\n=== achievable MBRL objective ranges (combined {args.good}+{args.wide}) ===\n")
    with pd.option_context("display.width", 220):
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3g}"))
    print(f"\nwrote {csv}")

    # ---- FIG 1: per-objective distributions (good vs wide), golden marked ---------------
    ncol = 4
    nrow = int(np.ceil(len(OBJ) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.3 * nrow))
    for ax, (key, grp, label, unit, direction, log, gated) in zip(axes.ravel(), OBJ):
        vg = vals(qg, wg, key, gated)
        vw = vals(qw, ww, key, gated)
        allv = np.concatenate([vg, vw])
        if log:
            vg, vw, allv = vg[vg > 0], vw[vw > 0], allv[allv > 0]
            bins = np.logspace(np.log10(allv.min()), np.log10(allv.max()), 40)
            ax.set_xscale("log")
        else:
            bins = np.linspace(allv.min(), allv.max(), 40)
        ax.hist(vw, bins=bins, color=ORANGE, alpha=0.55, label="wide", edgecolor="none")
        ax.hist(vg, bins=bins, color=BLUE, alpha=0.65, label="good", edgecolor="none")
        g = gval(key)
        if np.isfinite(g):
            ax.axvline(g, color=GREEN, ls="--", lw=1.8, label=f"golden {g:.3g}")
        if direction == "target" and (not log or 1.0 > 0):
            ax.axvline(1.0, color="k", ls=":", lw=1.2, label="matched 1.0")
        ax.set_title(f"{label}   [{GLYPH[direction]}]", fontsize=9.5)
        ax.set_xlabel(unit, fontsize=8.5); ax.legend(fontsize=7, loc="best")
        ax.tick_params(labelsize=8)
    for ax in axes.ravel()[len(OBJ):]:
        ax.axis("off")
    fig.suptitle(f"Achievable ranges of each downstream MBRL objective  "
                 f"(good/tight blue, wide/failing orange, golden dashed)   "
                 f"good n={int(wg.sum())} wit-viable / wide n={int(ww.sum())}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(figdir / "mbrl_objective_ranges.png", dpi=130); plt.close(fig)

    # ---- FIG 2: reachability summary -- range bars vs golden, per objective -------------
    # normalize each objective's [p5,p95] and golden to log or linear; plot as ratio-to-golden
    order = [r for r in rows]
    names = [f"{r['group']}: {r['objective']}" for r in order]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11, 0.5 * len(names) + 1.6))
    for i, r in enumerate(order):
        g = r["golden"]
        if not np.isfinite(g) or g == 0:
            continue
        lo, med, hi = r["p5"] / g, r["p50"] / g, r["p95"] / g
        ax.plot([lo, hi], [i, i], color=BLUE, lw=3, solid_capstyle="round", zorder=2)
        ax.plot(med, i, "o", color=BLUE, ms=6, zorder=3)
        if np.isfinite(r["best"]):
            ax.plot(r["best"] / g, i, "|", color=ORANGE, ms=13, mew=2.2, zorder=4)
        ax.annotate(GLYPH[r["direction"]].split()[0], (hi, i), fontsize=7, color=GREY,
                    xytext=(4, 0), textcoords="offset points", va="center")
    ax.axvline(1.0, color=GREEN, ls="--", lw=1.8, label="golden (=1)")
    ax.plot([], [], color=BLUE, lw=3, marker="o", label="p5-p95 range (median dot)")
    ax.plot([], [], "|", color=ORANGE, ms=13, mew=2.2, label="best achieved (min/max dir.)")
    ax.set_xscale("log"); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel("value / golden  (1.0 = golden working point)")
    ax.set_title("MBRL objective reachability, relative to golden", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(figdir / "mbrl_reachability.png", dpi=130); plt.close(fig)

    print(f"wrote 2 figures to {figdir}")


if __name__ == "__main__":
    main()
