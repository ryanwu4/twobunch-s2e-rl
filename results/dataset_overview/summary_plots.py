"""Summary plots for the two-bunch LHS sweep.

Reads results/tables/dataset.pkl (built by analysis_tools/build_dataset.py). Produces, beside this script:
  fig1_inputs.png        - 8 sweep-knob input distributions (bounds + baseline)
  fig2_feasibility.png   - witness-survival taxonomy per treaty point + transmission
  fig3_outputs_PENT.png  - key output distributions at PENT (drive vs witness)
  fig4_emit_evolution.png- emittance / BMAG growth BEGBC20 -> MFFF -> PENT

Also prints a feasibility table and output summary stats to stdout.

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/dataset_overview/summary_plots.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pathlib import Path

from twobunch_s2e_rl.datagen.paths import tables_dir
from twobunch_s2e_rl.datagen.sweep_params import SWEEP_PARAMS, PARAM_KEYS

ART = tables_dir()                          # dataset.pkl lives here
FIG = Path(__file__).resolve().parent       # write figures beside this script
POINTS = ["BEGBC20", "MFFF", "PENT"]
DRIVE_C = "#1f77b4"
WIT_C = "#d62728"

df = pd.read_pickle(ART / "dataset.pkl")
N = len(df)

# ---- knob metadata (label, unit, scale-to-display) ----
KNOB_META = {
    "L1PhaseSet":     ("L1 phase", "deg", 1.0),
    "L2PhaseSet":     ("L2 phase", "deg", 1.0),
    "L1EnergyOffset": ("L1 energy offset", "MeV", 1e-6),
    "L2EnergyOffset": ("L2 energy offset", "MeV", 1e-6),
    "L3EnergyOffset": ("L3 energy offset", "MeV", 1e-6),
    "S1ELkG":         ("S1E sextupole", "kG", 1.0),
    "S2ELkG":         ("S2E sextupole", "kG", 1.0),
    "S3ELkG":         ("S3E sextupole", "kG", 1.0),
}


# ====================================================================
# Feasibility taxonomy
# ====================================================================
def witness_viable(pt):
    return df[f"{pt}__PWitness_norm_emit_x"].notna()


def drive_viable(pt):
    return df[f"{pt}__PDrive_norm_emit_x"].notna()


def specs_error(pt):
    # 5-key dict => getBeamSpecs threw; PDrive moments absent
    return df[f"{pt}__PDrive_median_x"].isna()


def feasibility_table():
    print("\n=== WITNESS / DRIVE SURVIVAL PER TREATY POINT (N=%d) ===" % N)
    print(f"{'point':10s} {'wit_viable':>11s} {'wit_destroyed':>14s} "
          f"{'(specs_err)':>12s} {'drive_lost':>11s}")
    rows = {}
    for pt in POINTS:
        wv = int(witness_viable(pt).sum())
        se = int(specs_error(pt).sum())
        dl = int((~drive_viable(pt)).sum())
        wd = N - wv
        rows[pt] = dict(wit_viable=wv, wit_destroyed=wd, specs_err=se, drive_lost=dl)
        print(f"{pt:10s} {wv:5d} ({100*wv/N:4.1f}%) {wd:6d} ({100*wd/N:4.1f}%) "
              f"{se:6d}      {dl:6d}")
    return rows


# ====================================================================
# Fig 1 - input knob distributions
# ====================================================================
def fig_inputs():
    fig, axes = plt.subplots(2, 4, figsize=(17, 8))
    for ax, k in zip(axes.ravel(), PARAM_KEYS):
        lo, hi, base = SWEEP_PARAMS[k]
        label, unit, s = KNOB_META[k]
        v = df[k].to_numpy() * s
        ax.hist(v, bins=40, color="#4c72b0", alpha=0.85, edgecolor="white", lw=0.3)
        ax.axvline(lo * s, color="k", ls="--", lw=1.2)
        ax.axvline(hi * s, color="k", ls="--", lw=1.2)
        ax.axvline(base * s, color="#dd8452", ls="-", lw=2.0)
        ax.set_title(f"{label}", fontsize=11)
        ax.set_xlabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("count")
    axes[1, 0].set_ylabel("count")
    handles = [Line2D([0], [0], color="k", ls="--", label="sweep bounds"),
               Line2D([0], [0], color="#dd8452", lw=2, label="baseline (2024-10-14)")]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"Input sweep knobs — 8-D Latin hypercube, N={N}", y=1.06, fontsize=13)
    fig.tight_layout()
    p = FIG / "fig1_inputs.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ====================================================================
# Fig 2 - feasibility taxonomy
# ====================================================================
def fig_feasibility(rows):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))

    # 2a stacked survival bars
    cats = ["witness viable", "witness scraped (no specs_err)", "specs_error"]
    colors = ["#55a868", "#c44e52", "#8172b3"]
    viable = [rows[p]["wit_viable"] for p in POINTS]
    serr = [rows[p]["specs_err"] for p in POINTS]
    scraped = [rows[p]["wit_destroyed"] - rows[p]["specs_err"] for p in POINTS]
    x = np.arange(len(POINTS))
    ax1.bar(x, viable, color=colors[0], label=cats[0])
    ax1.bar(x, scraped, bottom=viable, color=colors[1], label=cats[1])
    ax1.bar(x, serr, bottom=np.array(viable) + np.array(scraped), color=colors[2], label=cats[2])
    for xi, p in zip(x, POINTS):
        ax1.text(xi, rows[p]["wit_viable"] / 2, f"{100*rows[p]['wit_viable']/N:.0f}%",
                 ha="center", va="center", color="white", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(POINTS)
    ax1.set_ylabel("samples")
    ax1.set_title("Witness-bunch survival by treaty point")
    ax1.legend(fontsize=9, loc="lower left")

    # 2b transmission distribution at PENT, split by witness viability
    wv = witness_viable("PENT")
    t = df["PENT__transmission_total"].to_numpy()
    bins = np.linspace(0.4, 1.001, 50)
    ax2.hist(t[wv.to_numpy()], bins=bins, color="#55a868", alpha=0.8, label="witness viable")
    ax2.hist(t[~wv.to_numpy()], bins=bins, color="#c44e52", alpha=0.8, label="witness destroyed")
    ax2.set_xlabel("total transmission at PENT")
    ax2.set_ylabel("count")
    ax2.set_title("Transmission to PENT (drive+witness)")
    ax2.legend(fontsize=9)

    fig.suptitle("Feasibility: ~31% of the sampled knob space loses the witness bunch before the IP",
                 fontsize=12)
    fig.tight_layout()
    p = FIG / "fig2_feasibility.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ====================================================================
# Fig 3 - output distributions at PENT
# ====================================================================
# (column-suffix, label, unit, display-scale, log?)
OUT_METRICS = [
    ("norm_emit_x",        "norm. emittance x",   "µm",   1e6, True),
    ("norm_emit_y",        "norm. emittance y",   "µm",   1e6, True),
    ("BMAG_x",             "BMAG x",              "",     1.0, True),
    ("BMAG_y",             "BMAG y",              "",     1.0, True),
    ("sigmaSI90_x",        "spot size x (90%)",   "µm",   1e6, True),
    ("sigmaSI90_y",        "spot size y (90%)",   "µm",   1e6, True),
    ("sigmaSI90_z",        "bunch length z (90%)", "µm",  1e6, True),
    ("sigmaSI90_energy",   "energy spread (90%)", "MeV",  1e-6, True),
]


def _logbins(vals, n=45):
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 30
    return np.logspace(np.log10(vals.min()), np.log10(vals.max()), n)


def fig_outputs(pt="PENT"):
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    wv = witness_viable(pt).to_numpy()
    for ax, (suf, label, unit, s, logx) in zip(axes.ravel(), OUT_METRICS):
        d = df[f"{pt}__PDrive_{suf}"].to_numpy() * s
        w = df[f"{pt}__PWitness_{suf}"].to_numpy() * s
        d = d[np.isfinite(d) & (drive_viable(pt).to_numpy())]
        w = w[np.isfinite(w) & wv]
        allv = np.concatenate([d, w])
        allv = allv[np.isfinite(allv) & (allv > 0)] if logx else allv[np.isfinite(allv)]
        bins = _logbins(allv) if logx else np.linspace(np.nanmin(allv), np.nanmax(allv), 45)
        ax.hist(d, bins=bins, color=DRIVE_C, alpha=0.6, label="drive")
        ax.hist(w, bins=bins, color=WIT_C, alpha=0.6, label="witness")
        if logx:
            ax.set_xscale("log")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(unit, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Output distributions at {pt} — witness-viable subset "
                 f"(drive n={int(drive_viable(pt).sum())}, witness n={int(wv.sum())})",
                 fontsize=13)
    fig.tight_layout()
    p = FIG / f"fig3_outputs_{pt}.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ====================================================================
# Fig 4 - emittance & BMAG evolution across treaty points
# ====================================================================
def fig_evolution():
    metrics = [("norm_emit_x", "norm. emittance x [µm]", 1e6, True),
               ("norm_emit_y", "norm. emittance y [µm]", 1e6, True),
               ("BMAG_x", "BMAG x", 1.0, True),
               ("BMAG_y", "BMAG y", 1.0, True)]
    fig, axes = plt.subplots(1, 4, figsize=(19, 5))
    xpos = np.arange(len(POINTS))
    for ax, (suf, label, s, logy) in zip(axes, metrics):
        for bunch, col in [("PDrive", DRIVE_C), ("PWitness", WIT_C)]:
            meds, q1, q3 = [], [], []
            for pt in POINTS:
                v = df[f"{pt}__{bunch}_{suf}"].to_numpy() * s
                v = v[np.isfinite(v) & (v > 0)]
                meds.append(np.median(v))
                q1.append(np.percentile(v, 16))
                q3.append(np.percentile(v, 84))
            ax.plot(xpos, meds, "o-", color=col, label=bunch.replace("P", ""))
            ax.fill_between(xpos, q1, q3, color=col, alpha=0.18)
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(xpos)
        ax.set_xticklabels(POINTS)
        ax.set_title(label, fontsize=11)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=10)
    fig.suptitle("Median (band = 16–84%) evolution along the line — witness-viable subset", fontsize=13)
    fig.tight_layout()
    p = FIG / "fig4_emit_evolution.png"
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote", p)


# ====================================================================
def output_stats():
    pt = "PENT"
    wv = witness_viable(pt).to_numpy()
    dv = drive_viable(pt).to_numpy()
    print(f"\n=== OUTPUT SUMMARY @ {pt} (witness-viable subset) ===")
    print(f"{'metric':28s}{'bunch':8s}{'min':>11s}{'median':>11s}{'max':>11s}")
    for suf, label, unit, s, _ in OUT_METRICS:
        for bunch, mask in [("drive", dv), ("witness", wv)]:
            col = f"{pt}__P{'Drive' if bunch=='drive' else 'Witness'}_{suf}"
            v = df[col].to_numpy()[mask] * s
            v = v[np.isfinite(v)]
            print(f"{label+' ['+unit+']':28s}{bunch:8s}{v.min():11.3g}{np.median(v):11.3g}{v.max():11.3g}")
    for col, label, s in [("PENT__bunchSpacing", "bunchSpacing [µm]", 1e6),
                          ("PENT__transverseCentroidOffset", "transvCentroidOff [µm]", 1e6),
                          ("PENT__transmission_total", "transmission [-]", 1.0)]:
        v = df[col].to_numpy()
        v = v[np.isfinite(v)]
        print(f"{label:28s}{'beam':8s}{v.min()*s:11.3g}{np.median(v)*s:11.3g}{v.max()*s:11.3g}")


if __name__ == "__main__":
    rows = feasibility_table()
    output_stats()
    fig_inputs()
    fig_feasibility(rows)
    fig_outputs("PENT")
    fig_evolution()
    print("\nAll figures in", FIG)
