"""Figures for the overnight 2026-10-16 notebook update: surrogate progression (pair metrics with 90% CIs, sigma_z
gate) and the cross-flow RL eval. Writes lab-notebook/images/2026-10-16_*.png."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False})


def surrogate():
    cb = json.loads(Path("results/surrogate_scorecard_2026-10-16/B/scorecard.json").read_text())["models"]
    cc = json.loads(Path("results/surrogate_scorecard_2026-10-16/C/scorecard.json").read_text())["models"]
    seq = [("ftc_s0\n(centroid)", cb["ftc_s0 best"]), ("A\n(+pair, T2)", cb["A best"]), ("B\n(+top-up)", cb["B (top-up, w_pair 1) best"]),
           ("B2\n(w_pair 3)", cb["B2 (top-up, w_pair 3) best"]), ("C\n(w_pair 6)", cc["C (w_pair 6) best"]),
           ("C2\n(w_pair 10)", cc["C2 (w_pair 10) best"])]
    x = np.arange(len(seq))
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
    for ax, (k, lab, gate) in zip(axes[:3], (("R2", "val pair R²", (0.8, None)), ("slope", "val pair slope", (0.85, 1.15)),
                                             ("p90_um", "val pair p90 [µm]", (None, 8.0)))):
        v = np.array([m["paired_val"][k] for _, m in seq])
        lo = np.array([m["paired_val"]["ci90"][k][0] for _, m in seq])
        hi = np.array([m["paired_val"]["ci90"][k][1] for _, m in seq])
        ax.errorbar(x, v, yerr=[v - lo, hi - v], fmt="o-", color=BLUE, capsize=3, lw=1.5, ms=5)
        t = np.array([m["paired_test"][k] for _, m in seq])
        ax.plot(x, t, "s--", color=INK2, ms=4, lw=1, label="test (quoted once)")
        for g in gate:
            if g is not None:
                ax.axhline(g, color=ORANGE, ls=":", lw=1.2)
        ax.set_xticks(x, [s for s, _ in seq], fontsize=7.5)
        ax.set_title(lab + " (90% CI; orange = gate)", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].legend(fontsize=7, frameon=False)
    ax = axes[3]
    sz = [m["gate_oldval"]["sigma_z_rel"] for _, m in seq]
    ax.plot(x, sz, "o-", color=AQUA, lw=1.5)
    ax.axhline(0.107 + 0.005, color=ORANGE, ls=":", lw=1.2)
    ax.set_xticks(x, [s for s, _ in seq], fontsize=7.5)
    ax.set_title("old-val drive σz rel error (gate ≤ base + 0.005)", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("Injector-aware flow: pair loss + top-up data move the injector × linac effect toward the gate; "
                 "a higher pair weight costs σz", fontsize=10, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-16_surrogate_progression.png", dpi=140)
    plt.close(fig)


def cross_eval():
    rows = []
    for line in Path("results/cross_eval_2026-10-16/flowA/TABLE.md").read_text().splitlines():
        m = re.match(r"\| (S[12] [^|]+?) \| ([\d.]+) / \d+%[^|]*\| (?:([\d.]+) / \d+%)? *\| (?:([\d.]+) / \d+%)? *\|", line)
        if m:
            rows.append((m.group(1), float(m.group(2)), float(m.group(3)) if m.group(3) else np.nan,
                         float(m.group(4)) if m.group(4) else np.nan))
    rows = [r for r in rows if "leaky" not in r[0]]
    # injflow runs: their own env IS flow A + injector variation
    rows = [(n, np.nan, g, o) if "injflow" in n else (n, o, g, i) for n, o, g, i in rows]
    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    ax.scatter([r[1] for r in rows], y, color=INK2, s=28, label="own env (old flow)", zorder=3)
    ax.scatter([r[2] for r in rows], y, color=BLUE, s=28, marker="s", label="flow A, injector golden", zorder=3)
    ax.scatter([r[3] for r in rows], y, color=ORANGE, s=34, marker="D", label="flow A, injector variation", zorder=3)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7.5)
    ax.axvline(10, color=GRID)
    ax.set(xlabel="true median spacing error [µm]", xlim=(3, 13.5))
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title("Cross-flow eval: S1 robust to injector variation; S2 loses 1–2 µm (direct-L1 S2 more)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-16_cross_eval.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    surrogate()
    cross_eval()
