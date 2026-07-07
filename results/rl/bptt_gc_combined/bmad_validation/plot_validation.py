"""Plot surrogate-vs-Bmad validation of the controller setpoints.

Reads results/rl/bptt_gc_combined/bmad_validation/validate_goal*um.json (from scripts/validate_bmad.py) and makes:
  - validation_bars.png   : per-metric grouped bars (surrogate vs Bmad) for each target, with
                            target/golden reference lines and %-diff annotations
  - validation_parity.png : surrogate (x) vs Bmad (y) log-log parity, y=x = perfect agreement;
                            points far off the diagonal are where the surrogate misled the policy

Usage: PYTHONPATH=$PWD/src MPLBACKEND=Agg python results/rl/bptt_gc_combined/bmad_validation/plot_validation.py
"""
import glob
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BLUE, ORANGE, GREEN, GREY = "#4c72b0", "#dd8452", "#55a868", "#8c8c8c"

# (key, label, log?, reference)   reference: "target" (spacing) | 47.0 (golden offset) | None
METRICS = [
    ("spacing_um", "bunch spacing [µm]", False, "target"),
    ("T_drive", "T drive", False, None),
    ("T_witness", "T witness", False, None),
    ("drive_emit_x_um_rad", "drive εx [µm·rad]", False, None),
    ("drive_emit_y_um_rad", "drive εy [µm·rad]", False, None),
    ("witness_emit_x_um_rad", "witness εx [µm·rad]", False, None),
    ("witness_emit_y_um_rad", "witness εy [µm·rad]", False, None),
    ("transverse_offset_um", "transverse offset [µm]", True, 47.0),
    ("angular_misalignment_urad", "angular misalign [µrad]", True, None),
]


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(str(HERE / "validate_goal*um.json")))]
    if not recs:
        raise SystemExit(f"no validate_goal*um.json in {HERE}")
    targets = [r["target_um"] for r in recs]
    tlabels = [f"{int(round(t))} µm" for t in targets]
    x = np.arange(len(targets))

    # ---- FIG 1: per-metric grouped bars ------------------------------------------------
    ncol = 3
    nrow = int(np.ceil(len(METRICS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.0 * ncol, 3.4 * nrow))
    for ax, (key, label, log, ref) in zip(axes.ravel(), METRICS):
        surr = np.array([r["comparison"][key]["surrogate"] for r in recs], float)
        bmad = np.array([r["comparison"][key]["bmad"] for r in recs], float)
        ax.bar(x - 0.2, surr, 0.38, color=BLUE, label="surrogate")
        ax.bar(x + 0.2, bmad, 0.38, color=ORANGE, label="Bmad")
        if log:
            ax.set_yscale("log")
        # %-diff annotation over the Bmad bar
        for xi, (s, b) in enumerate(zip(surr, bmad)):
            if s != 0 and np.isfinite(b):
                ax.annotate(f"{100*(b-s)/s:+.0f}%", (xi + 0.2, b), ha="center",
                            va="bottom", fontsize=7.5, color=GREY)
        if ref == "target":                               # spacing: mark the commanded target
            for xi, t in enumerate(targets):
                ax.plot([xi - 0.4, xi + 0.4], [t, t], color=GREEN, lw=1.8, ls="--")
            ax.plot([], [], color=GREEN, lw=1.8, ls="--", label="target")
        elif isinstance(ref, (int, float)):               # golden reference line
            ax.axhline(ref, color=GREEN, lw=1.6, ls="--", label=f"golden {ref:g}")
        ax.set_xticks(x); ax.set_xticklabels(tlabels, fontsize=9)
        ax.set_title(label, fontsize=10); ax.legend(fontsize=7.5, loc="best")
        ax.margins(y=0.22)
    for ax in axes.ravel()[len(METRICS):]:
        ax.axis("off")
    fig.suptitle("Setpoint validation: surrogate prediction vs Bmad tracking (per target)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "validation_bars.png", dpi=130); plt.close(fig)

    # ---- FIG 2: surrogate-vs-Bmad parity (log-log) -------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7.5))
    cmap = plt.get_cmap("tab10")
    for mi, (key, label, _, _) in enumerate(METRICS):
        for ti, r in enumerate(recs):
            s = r["comparison"][key]["surrogate"]; b = r["comparison"][key]["bmad"]
            if s > 0 and np.isfinite(b) and b > 0:
                ax.scatter(s, b, s=70, color=cmap(mi % 10), marker="o" if ti == 0 else "^",
                           edgecolor="k", lw=0.4, zorder=3,
                           label=label if ti == 0 else None)
                ax.annotate(f"{label.split(' [')[0]}", (s, b), fontsize=6.5, color=GREY,
                            xytext=(4, 3), textcoords="offset points")
    lo, hi = 1e-1, 1e3
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect (y=x)")
    for f, lab in ((10, "10× off"), (0.1, None)):
        ax.plot([lo, hi], [lo * f, hi * f], color=GREY, ls=":", lw=0.8, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("surrogate prediction"); ax.set_ylabel("Bmad (truth)")
    ax.set_title("Surrogate vs Bmad — points far above y=x = optimistic surrogate\n"
                 "(○ = 150 µm target, △ = 250 µm target)", fontsize=11)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.set_aspect("equal")
    fig.tight_layout(); fig.savefig(HERE / "validation_parity.png", dpi=130); plt.close(fig)
    print(f"wrote validation_bars.png + validation_parity.png to {HERE}")


if __name__ == "__main__":
    main()
