"""Morning-report figures for overnight 2026-10-17 -> 18 (results/report_2026-10-18/*.png). Skips missing inputs.

  /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/report_figs_2026-10-18.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/report_2026-10-18")
OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, RED, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#9a9892"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
EV = Path("results/eval_realistic_2026-10-18")
EC = Path("results/eval_continuous_2026-10-18")
# one fixed colour per controller design
POL = [("old blen_dlog (flow A)", EC / "base_S2blen_flowA", GREY),
       ("old direct-L1 + BLEN obs", EC / "base_S2l1blen_covB2", "#c9c6bd"),
       ("no-BLEN s0", EV / "real_covB2_screen_noblen_s0", BLUE), ("no-BLEN s1", EV / "real_covB2_screen_noblen_s1", BLUE),
       ("BC11 loop v1 s0", EV / "real_covB2_screen_bc11loop_s0", RED), ("BC11 loop v1 s1", EV / "real_covB2_screen_bc11loop_s1", RED),
       ("BC11 loop v3 s0", EV / "real_covB2_screen_bc11loop_v3_s0", ORANGE),
       ("BC11 loop v3 s1", EV / "real_covB2_screen_bc11loop_v3_s1", ORANGE)]


def jload(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, json.JSONDecodeError):
        return None


def surrogate():
    rows = []
    a = jload("results/surrogate_scorecard_2026-10-18/cov_table/scorecard.json")
    b = jload("results/surrogate_scorecard_2026-10-18/zs_cov_table/scorecard.json")
    if not (a and b):
        return
    for name, src, key in (("B2-cov", b, "B2-cov best"), ("6-D head, fine-tuned (w_pair 3)", a, "D1 centroid (w_pair 3) best"),
                           ("6-D head, fine-tuned (w_pair 10)", a, "D2 centroid (w_pair 10) best"),
                           ("z head per bunch", b, "B2-cov + z head (per bunch)"), ("spacing head", b, "B2-cov + spacing head")):
        rows.append((name, src["models"][key]))
    cols = [GREY, RED, RED, ORANGE, BLUE]
    panels = [("val pair R²", lambda m: m["paired_val"]["R2"], lambda m: m["paired_val"]["ci90"]["R2"], (0.8, None)),
              ("val pair slope", lambda m: m["paired_val"]["slope"], lambda m: m["paired_val"]["ci90"]["slope"], (0.85, 1.15)),
              ("held-out pair p90 [µm]", lambda m: m["pairs_heldout"]["p90_um"], lambda m: m["pairs_heldout"]["ci90"]["p90_um"], (None, 8.0)),
              ("witness emittance R² (old val)", lambda m: m["orig_eval_oldval"]["witness_norm_emit_x"], None, None),
              ("drive σz rel error", lambda m: m["gate_oldval"]["sigma_z_rel"], None, (None, 0.112)),
              ("|spacing| MAE old val [µm]", lambda m: m["gate_oldval"]["spacing_mae_um"], None, None)]
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.6))
    for ax, (title, f, ci, gate) in zip(axes, panels):
        for i, (name, m) in enumerate(rows):
            v = f(m)
            if ci is not None:
                lo, hi = ci(m)
                ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt="o", color=cols[i], capsize=3)
            else:
                ax.plot(i, v, "o", color=cols[i])
        if gate:
            lo, hi = gate
            ax.axhspan(lo if lo is not None else ax.get_ylim()[0], hi if hi is not None else ax.get_ylim()[1],
                       color=ORANGE, alpha=0.08, lw=0)
        ax.set_xticks(range(len(rows)), [r[0] for r in rows], rotation=60, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.6)
    fig.suptitle("Surrogate: deterministic centroid variants on B2-cov (cov table; shaded = gate pass region)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "surrogate_centroid_variants.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def continuous():
    have = [(n, jload(p.parent / (p.name + "_cont") / "summary.json") if p.parent == EV else jload(p / "summary.json"),
             jload(p.parent / (p.name + "_cont_stress") / "summary.json") if p.parent == EV else None, c)
            for n, p, c in POL]
    have = [h for h in have if h[1]]
    if not have:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for j, (key, title, sc) in enumerate((("settled_spacing_median_um", "settled spacing error [µm] (median)", 1),
                                          ("settled_within_10um", "setpoints settled within 10 µm", 1),
                                          ("frac_segments_reached", "setpoints reached (within 10 µm at any time)", 1))):
        ax = axes[j]
        for i, (n, s, st, c) in enumerate(have):
            ax.barh(i, s[key] * sc, color=c, height=0.38, align="edge")
            if st:
                ax.barh(i - 0.38, st[key] * sc, color=c, alpha=0.4, height=0.38, align="edge")
            ax.text(s[key] * sc, i + 0.19, f" {s[key] * sc:.2g}", va="center", fontsize=7, color=INK2)
        ax.set_yticks(range(len(have)), [h[0] for h in have] if j == 0 else [])
        ax.set_title(title, fontsize=9)
        ax.grid(axis="x", color=GRID, lw=0.6)
    axes[0].invert_yaxis(); axes[1].invert_yaxis(); axes[2].invert_yaxis()
    fig.suptitle("Continuous operation (128-step episodes, new setpoint every 16-48 steps, DAQ drift/jumps; solid = nominal, faint = 2x drift + jumps)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "continuous_operation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def trace():
    runs = [(n, p, c) for n, p, c in POL if p.parent == EV and (p.parent / (p.name + "_cont") / "errors.npz").exists()
            and n.endswith("s0")]
    if not runs:
        return
    fig, axes = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 3.4), squeeze=False)
    for ax, (n, p, c) in zip(axes[0], runs):
        d = np.load(p.parent / (p.name + "_cont") / "errors.npz")
        e, chg = d["e_sp"] * 1e6, d["chg"]
        for b in range(4):
            ax.plot(e[:, b], color=c, lw=0.9, alpha=0.85)
            for t in np.flatnonzero(chg[1:, b]) + 1:
                ax.axvline(t, color=c, lw=0.5, alpha=0.3)
        ax.plot(np.median(e, 1), color=INK, lw=1.5, label="median over 256")
        ax.axhline(10, color=INK2, ls="--", lw=0.8)
        ax.set(yscale="log", ylim=(0.3, 400), xlabel="step (~20 s)", title=n)
        ax.grid(color=GRID, lw=0.5)
    axes[0][0].set_ylabel("true |spacing − goal| [µm]")
    axes[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Continuous operation traces (4 episodes; vertical lines = setpoint changes)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "continuous_traces.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def closed_loop():
    CL = Path("results/closed_loop_bmad_2026-10-18")
    tags = sorted({p.name.rsplit("_", 2)[0] for p in CL.glob("*_bmad") if (p / "summary.json").exists()})
    if not tags:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    xt = []
    for i, (t, mode) in enumerate([(t, m) for t in tags for m in ("golden", "inj")]):
        for j, plant in enumerate(("bmad", "flow")):
            s = jload(CL / f"{t}_{mode}_{plant}" / "summary.json")
            if not s:
                continue
            sp = s["targets"]["bunch_spacing"]
            e = np.asarray(sp["per_episode_final_true"]) * 1e6
            x = i + (j - 0.5) * 0.3
            c = BLUE if "noblen" in t else ORANGE
            axes[0].scatter(x + np.random.default_rng(0).uniform(-0.05, 0.05, len(e)), e, s=6, color=c, alpha=0.35)
            axes[0].plot(x, np.median(e), "o" if plant == "bmad" else "D", ms=8, mfc=c if plant == "bmad" else SURF, mec=c)
            axes[1].plot(x, sp["frac_within_tol_final_true"], "o" if plant == "bmad" else "D", ms=8,
                         mfc=c if plant == "bmad" else SURF, mec=c)
        xt.append(f"{t.replace('real_covB2_screen_', '')}\n{mode}")
    for ax in axes:
        ax.set_xticks(range(len(xt)), xt, fontsize=7)
        ax.grid(axis="y", color=GRID, lw=0.6)
    axes[0].set(yscale="log", ylabel="final true |spacing − goal| [µm]", title="per episode + median (filled = Bmad, open = surrogate)")
    axes[0].axhline(10, color=INK2, ls="--", lw=0.8)
    axes[1].set(ylim=(0, 1), ylabel="fraction within 10 µm", title="within 10 µm (24 episodes)")
    fig.suptitle("Closed-loop Bmad (32 steps, 2e4 macroparticles) vs the paired surrogate run", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "closed_loop_bmad.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def replay():
    old = Path("results/daq_replay_2026-10-17")
    pols = [("old blen_dlog (flow A)", old / "injflow_flowA_screen_v6_blen_s0", GREY),
            ("old direct-L1 + BLEN (B2-cov s0)", old / "injflow_covB2_screen_v6_l1blen_s0", "#c9c6bd")]
    pols += [(n, p, c) for n, p, c in POL if p.parent == EV]
    rows = []
    for n, p, c in pols:
        vals = []
        for scan in ("TEST_15668", "E300_15673"):
            s = jload(f"{p}_{scan}/summary.json") if p.parent == old else jload(p.parent / f"{p.name}_replay_{scan}" / "summary.json")
            vals.append(s["frac_spacing_within_10um_meas"] if s else np.nan)
        if not all(np.isnan(vals)):
            rows.append((n, vals, c))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * len(rows) + 1.2))
    for i, (n, (a, b), c) in enumerate(rows):
        ax.barh(i, a, color=c, height=0.38, align="edge")
        ax.barh(i - 0.38, b, color=c, alpha=0.45, height=0.38, align="edge")
        ax.text(a, i + 0.19, f" {a:.2f}", va="center", fontsize=7, color=INK2)
        ax.text(b, i - 0.19, f" {b:.2f}", va="center", fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set(xlim=(0, 1), xlabel="fraction of 64 episodes ending within 10 µm (measured)",
           title="Real-scan L2 replay: solid = TEST_15668, faint = E300_15673")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "daq_replay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (surrogate, continuous, trace, closed_loop, replay):
        f()
    print(sorted(p.name for p in OUT.glob("*.png")))
