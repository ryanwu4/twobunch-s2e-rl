"""Coverage + success-rate diagnostics for the combined good+failing training dataset.

The training set is two campaigns: a dense "good" tight box (tightbox_v2_full) and a broad
"failing/boundary" wide box (expanded_full). This scores the dataset the surrogate will see:
  - success/feasibility funnel per box + combined (tracked -> drive -> witness -> transmission
    -> matched), the "success rates" view
  - objective-space coverage: where each box lands in (offset, BMAG, spacing, emittance),
    showing the good cluster embedded in the failing spread
  - knob-space coverage: each knob's span per box within the union normalization frame
  - feasibility boundary: witness-resolution + transmission vs excursion from golden -- the
    survival gradient the viability head learns from the wide box

Reads data/<good>/ and data/<wide>/ via achievable_targets.{load,derived}.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg \
    python results/combined_dataset/dataset_coverage.py [good_subdir] [wide_subdir]
    (defaults: tightbox_v2_full expanded_full)
Outputs: written beside this script in results/combined_dataset/ (*.png + combined_dataset_summary.csv)
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

from twobunch_s2e_rl.datagen.sweep_params import resolve_sweep_set
from twobunch_s2e_rl.analysis_io import load, derived, P

BLUE, ORANGE, GREEN, RED, GREY = "#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8c8c8c"
GOOD_SET, WIDE_SET, UNION_SET = "tightbox", "expanded", "tightbox+expanded"


def _prep(subdir):
    df = load(subdir)
    df = df[df["is_baseline_repeat"] == False].copy()
    q = derived(df)
    q["witness_ok"] = np.isfinite(q["witness_BMAG_x"])
    q["drive_ok"] = np.isfinite(P(df, "PDrive_norm_emit_x"))
    return df, q


def _funnel(q, golden_off, golden_bmag):
    n = len(q["transmission"])
    tm, wok = q["transmission"], q["witness_ok"]
    bm, off = q["BMAG_max"], q["offset"]
    stages = [
        ("tracked", np.ones(n, bool)),
        ("drive resolved", q["drive_ok"]),
        ("witness resolved", wok),
        ("T > 0.90", tm > 0.90),
        ("T > 0.99", tm > 0.99),
        (f"BMAG_max <= golden ({golden_bmag:.1f})", wok & (bm <= golden_bmag)),
        (f"offset <= golden ({golden_off:.0f}um)", wok & (np.abs(off) <= golden_off)),
    ]
    return [(name, float(np.nansum(m)) / n) for name, m in stages], n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("good", nargs="?", default="tightbox_v2_full")
    ap.add_argument("wide", nargs="?", default="expanded_full")
    args = ap.parse_args()

    figdir = Path(__file__).resolve().parent

    dg, qg = _prep(args.good)
    dw, qw = _prep(args.wide)
    # golden reference from the good box's baseline repeats
    base = load(args.good)
    base = base[base["is_baseline_repeat"] == True]
    gq = derived(base)
    g_off = float(np.nanmedian(gq["offset"])); g_bmag = float(np.nanmedian(gq["BMAG_max"]))
    g_sp = float(np.nanmedian(np.abs(gq["spacing"])))
    ng, nw = len(qg["transmission"]), len(qw["transmission"])
    lab_g, lab_w = f"good/tight (n={ng})", f"wide/failing (n={nw})"

    # ---- success funnel table ----------------------------------------------------------
    fg, _ = _funnel(qg, g_off, g_bmag)
    fw, _ = _funnel(qw, g_off, g_bmag)
    names = [s[0] for s in fg]
    rows = [dict(stage=nm, good_frac=a[1], wide_frac=b[1],
                 good_n=int(round(a[1]*ng)), wide_n=int(round(b[1]*nw)),
                 combined_frac=(a[1]*ng + b[1]*nw)/(ng+nw)) for nm, a, b in zip(names, fg, fw)]
    tbl = pd.DataFrame(rows)
    csv = figdir / "combined_dataset_summary.csv"
    tbl.to_csv(csv, index=False)
    print(f"\n=== combined dataset: {args.good} (good) + {args.wide} (wide) ===")
    print(f"golden: offset {g_off:.0f}um  BMAG_max {g_bmag:.2f}  spacing {g_sp:.0f}um\n")
    with pd.option_context("display.width", 200):
        print(tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {csv}")

    # ---- FIG 1: success/feasibility funnel ---------------------------------------------
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 0.62 * len(names) + 1.6))
    ax.barh(y - 0.2, [s[1] for s in fg], height=0.38, color=BLUE, label=lab_g)
    ax.barh(y + 0.2, [s[1] for s in fw], height=0.38, color=ORANGE, label=lab_w)
    for yi, (a, b) in enumerate(zip(fg, fw)):
        ax.text(a[1] + 0.01, yi - 0.2, f"{a[1]:.0%}", va="center", fontsize=8)
        ax.text(b[1] + 0.01, yi + 0.2, f"{b[1]:.0%}", va="center", fontsize=8)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=9); ax.invert_yaxis()
    ax.set_xlim(0, 1.08); ax.set_xlabel("fraction of box")
    ax.set_title(f"Success / feasibility funnel  ({args.good} vs {args.wide})", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout(); fig.savefig(figdir / "success_funnel.png", dpi=130); plt.close(fig)

    # ---- FIG 2: objective-space coverage (good embedded in failing spread) --------------
    def scat(ax, xg, yg, xw, yw, xl, yl, gx, gy, logx, logy):
        ax.scatter(xw, yw, s=7, alpha=0.25, color=ORANGE, label=lab_w, rasterized=True)
        ax.scatter(xg, yg, s=7, alpha=0.30, color=BLUE, label=lab_g, rasterized=True)
        if gx is not None and gy is not None:
            ax.scatter([gx], [gy], marker="*", s=260, color=GREEN, edgecolor="k",
                       zorder=5, label="golden")
        if logx: ax.set_xscale("log")
        if logy: ax.set_yscale("log")
        ax.set_xlabel(xl, fontsize=9); ax.set_ylabel(yl, fontsize=9)
        ax.legend(fontsize=7, loc="best", framealpha=0.85)

    def col(q, k, wok=True):
        v = q[k].copy()
        return v[q["witness_ok"]] if wok else v

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    scat(ax[0, 0], np.abs(col(qg, "offset")), col(qg, "BMAG_max"),
         np.abs(col(qw, "offset")), col(qw, "BMAG_max"),
         "transverse offset [um]", "BMAG_max (worst of 4)", g_off, g_bmag, True, True)
    ax[0, 0].set_title("matching vs collinearity", fontsize=10)
    scat(ax[0, 1], np.abs(col(qg, "spacing")), col(qg, "BMAG_max"),
         np.abs(col(qw, "spacing")), col(qw, "BMAG_max"),
         "bunch spacing [um]", "BMAG_max", g_sp, g_bmag, False, True)
    ax[0, 1].set_title("matching vs separation", fontsize=10)
    scat(ax[1, 0], col(qg, "witness_emit_x"), col(qg, "witness_emit_y"),
         col(qw, "witness_emit_x"), col(qw, "witness_emit_y"),
         "witness norm.emit x [um-rad]", "witness norm.emit y [um-rad]",
         float(np.nanmedian(gq["witness_emit_x"])), float(np.nanmedian(gq["witness_emit_y"])),
         True, True)
    ax[1, 0].set_title("witness emittance plane", fontsize=10)
    # transmission vs BMAG over ALL resolved (failing tail visible); use full (not witness-gated) tm
    scat(ax[1, 1], col(qg, "BMAG_max"), qg["transmission"][qg["witness_ok"]],
         col(qw, "BMAG_max"), qw["transmission"][qw["witness_ok"]],
         "BMAG_max", "transmission", g_bmag, 1.0, True, False)
    ax[1, 1].set_title("survival vs matching", fontsize=10)
    fig.suptitle("Objective-space coverage: dense good cluster embedded in the failing spread",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(figdir / "objective_coverage.png", dpi=130); plt.close(fig)

    # ---- FIG 3: knob-space coverage (span of each box in the union frame) ---------------
    keys, ulo, uhi = resolve_sweep_set(UNION_SET)[:3]
    _, glo, ghi = resolve_sweep_set(GOOD_SET)[:3]
    _, wlo, whi = resolve_sweep_set(WIDE_SET)[:3]
    _, _, _, gbase = resolve_sweep_set(GOOD_SET)
    def nrm(v, i): return (np.asarray(v) - ulo[i]) / (uhi[i] - ulo[i])
    y = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(11, 0.36 * len(keys) + 1.6))
    for i, k in enumerate(keys):
        ax.plot([nrm(wlo[i], i), nrm(whi[i], i)], [i + 0.16, i + 0.16], color=ORANGE, lw=6,
                solid_capstyle="butt", alpha=0.9)
        ax.plot([nrm(glo[i], i), nrm(ghi[i], i)], [i - 0.16, i - 0.16], color=BLUE, lw=6,
                solid_capstyle="butt", alpha=0.9)
        ax.plot(nrm(gbase[k], i), i, "o", color=GREEN, ms=5, zorder=5)
    ax.set_yticks(y); ax.set_yticklabels(keys, fontsize=8); ax.invert_yaxis()
    ax.set_xlim(-0.03, 1.03); ax.set_xlabel("position within union range  (0=union low, 1=union high)")
    ax.set_title("Knob-space coverage: good (blue) nested in wide (orange); golden = green dot",
                 fontsize=12)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=BLUE, label=f"good/tight span ({GOOD_SET})"),
                       Patch(color=ORANGE, label=f"wide span ({WIDE_SET})"),
                       plt.Line2D([], [], marker="o", color=GREEN, ls="", label="golden")],
              fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(figdir / "knob_coverage.png", dpi=130); plt.close(fig)

    # ---- FIG 4: feasibility boundary vs excursion from golden ---------------------------
    def excursion(df):
        z = np.zeros(len(df))
        for i, k in enumerate(keys):
            half = (uhi[i] - ulo[i]) / 2.0
            z = z + ((df[k].to_numpy(float) - gbase[k]) / half) ** 2
        return np.sqrt(z / len(keys))  # RMS normalized excursion
    zg, zw = excursion(dg), excursion(dw)
    z = np.concatenate([zg, zw])
    wok = np.concatenate([qg["witness_ok"], qw["witness_ok"]])
    tm = np.concatenate([qg["transmission"], qw["transmission"]])
    order = np.argsort(z)
    edges = np.quantile(z, np.linspace(0, 1, 11))
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(z, edges) - 1, 0, 9)
    cx, res_rate, tm_med = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 5:
            cx.append(np.median(z[m])); res_rate.append(np.mean(wok[m]))
            tm_med.append(np.nanmedian(tm[m]))
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].scatter(zg, qg["witness_ok"] + np.random.default_rng(0).uniform(-0.03, 0.03, len(zg)),
                  s=5, alpha=0.15, color=BLUE, label=lab_g, rasterized=True)
    ax[0].scatter(zw, qw["witness_ok"] + np.random.default_rng(1).uniform(-0.03, 0.03, len(zw)),
                  s=5, alpha=0.15, color=ORANGE, label=lab_w, rasterized=True)
    ax[0].plot(cx, res_rate, "-o", color=RED, lw=2.2, label="witness-resolution rate")
    ax[0].set_xlabel("RMS knob excursion from golden  (union half-range units)")
    ax[0].set_ylabel("witness resolved (0/1) + rate"); ax[0].set_ylim(-0.1, 1.1)
    ax[0].set_title("feasibility boundary: witness survival vs excursion"); ax[0].legend(fontsize=8)
    ax[1].scatter(zg, qg["transmission"], s=5, alpha=0.15, color=BLUE, rasterized=True)
    ax[1].scatter(zw, qw["transmission"], s=5, alpha=0.15, color=ORANGE, rasterized=True)
    ax[1].plot(cx, tm_med, "-o", color=RED, lw=2.2, label="median transmission")
    ax[1].set_xlabel("RMS knob excursion from golden"); ax[1].set_ylabel("transmission")
    ax[1].set_title("transmission vs excursion"); ax[1].legend(fontsize=8)
    fig.suptitle("Survival gradient the viability head learns (good box low-excursion, "
                 "wide box spans the boundary)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figdir / "feasibility_boundary.png", dpi=130); plt.close(fig)

    print(f"wrote 4 figures to {figdir}")


if __name__ == "__main__":
    main()
