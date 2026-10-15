"""Diagrams + summary charts for the 2026-10-13..15 notebook entry (lab-notebook/images/2026-10-15_*.png).

  /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/notebook_figs_2026-10-15.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def box(ax, x, y, w, h, title, body="", new=False, fc="#f3f2ee"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ORANGE if new else "#b9b7b0", lw=2.0 if new else 1.0))
    ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top", fontsize=9.5, weight="bold", color=INK)
    if body:
        ax.text(x + w / 2, y + h - 0.52, body, ha="center", va="top", fontsize=7.6, color=INK2, linespacing=1.35)


def arrow(ax, p, q, text="", color=INK2, rad=0.0, fs=7.5, off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", fontsize=fs, color=INK2)


def canvas(w, h, xl, yl):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set(xlim=(0, xl), ylim=(0, yl))
    ax.axis("off")
    return fig, ax


def pipeline():
    fig, ax = canvas(13, 5.4, 13, 5.4)
    box(ax, 0.2, 3.0, 2.3, 2.1, "Injector", "Impact-T, 12 knobs\n(laser t, gun ampl/phase,\nd2 delay, drive frac, ...)\n-> L0AFEND", new=True)
    box(ax, 2.9, 3.0, 2.3, 2.1, "Linac", "Bmad, 23 + 2 (L0B) knobs\nCSR + wakes\n-> XTCAVF beams")
    box(ax, 5.6, 3.0, 2.4, 2.1, "Campaigns", "xtcav23  5.5k (golden inj)\ninjector_full  3027\ninjector_evalset  456\n(+ BLEN-monitor beams)", new=True)
    box(ax, 8.4, 3.0, 2.2, 2.1, "Flow surrogate", "37 knobs -> drive +\nwitness 6D clouds\n+ T heads\ninjector fine-tune,\ncentroid loss", new=True)
    box(ax, 10.9, 3.0, 1.9, 2.1, "Screen", "Cheetah XTCAVF->DTOTR\n+V / -V images\n(-V own energy axis)\nvalley gate, shared +V\ncut, crop-edge check", new=True)
    box(ax, 8.4, 0.3, 4.4, 2.0, "Delta-controller env (v4)", "actions: d ln(BC11 BLEN setpt), dL2 (MKB)\nhidden gains, drift, readback noise\nshot noise / floor / outliers / scale error\nhidden injector context, reachable goals", new=True)
    box(ax, 4.6, 0.3, 3.3, 2.0, "BPTT policy", "S1 cloud (truth)\nS2 screen (+-V emulation)\n4 + 3 seeds")
    box(ax, 0.2, 0.3, 3.9, 2.0, "Validation", "Bmad closed loop (v3: 6-7 um)\nreal E300 DAQ replay of the\nonline image features\nonline EPICS loop (todo)")
    for a, b in (((2.5, 4.05), (2.9, 4.05)), ((5.2, 4.05), (5.6, 4.05)), ((8.0, 4.05), (8.4, 4.05)),
                 ((10.6, 4.05), (10.9, 4.05))):
        arrow(ax, a, b)
    arrow(ax, (11.85, 3.0), (11.85, 2.3))
    arrow(ax, (8.4, 1.3), (7.9, 1.3))
    arrow(ax, (4.6, 1.3), (4.1, 1.3))
    ax.text(0.2, 5.3, "Pipeline (orange = new or changed 2026-10-13..15)", fontsize=11, weight="bold", va="top")
    fig.savefig(OUT / "2026-10-15_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def pm_measurement():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, sgn, name in ((axes[0], 1, "+V image"), (axes[1], -1, "-V image")):
        ax.add_patch(Ellipse((sgn * 0.9, 0.35), 2.0, 0.45, angle=sgn * -12, fc=BLUE, alpha=0.75, ec="none"))
        ax.add_patch(Ellipse((sgn * -1.2, 1.35), 0.7, 0.25, angle=sgn * -8, fc=ORANGE, alpha=0.85, ec="none"))
        ax.axhline(0.95, color=INK, ls="--", lw=1.2)
        ax.text(-2.4, 1.0, "shared +V cut", fontsize=7.5, color=INK)
        ax.text(sgn * 0.9, -0.15, "drive", ha="center", fontsize=8, color=BLUE)
        ax.text(sgn * -1.2, 1.62, "witness", ha="center", fontsize=8, color=ORANGE)
        ax.annotate("", (sgn * 0.9, -0.45), (sgn * -1.2, -0.45), arrowprops=dict(arrowstyle="<->", color=INK2))
        ax.text(-0.15, -0.72, f"offset{'+' if sgn > 0 else '-'} = x_d - x_w", ha="center", fontsize=7.5, color=INK2)
        ax.set(xlim=(-2.6, 2.6), ylim=(-0.9, 2.0), title=name, xlabel="x on screen  (+-S z: time)", ylabel="y  (energy)")
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].text(0, 1.85, "spacing = (offset+ - offset-) / 2S", ha="center", fontsize=8.5, weight="bold")
    ax = axes[2]
    y = np.linspace(0, 1, 400)
    base = 1.2 * np.exp(-((y - 0.75) / 0.05) ** 2) + 0.9 * np.exp(-((y - 0.4) / 0.16) ** 2) + 0.25 * np.exp(-((y - 0.25) / 0.05) ** 2)
    yc = 0.55
    warp = lambda s, b: np.interp((y - yc - b) / s + yc, y, base)
    ax.plot(y, warp(1.0, 0.0), color=INK2, lw=1.6, label="TCAV off")
    ax.plot(y, warp(0.97, 0.02), color=BLUE, lw=2, label="+V:  d' = d + k x")
    ax.plot(y, warp(1.04, -0.03), color=ORANGE, lw=2, label="-V:  d' = d - k x")
    ax.axvline(0.635, color=INK, ls="--", lw=1)
    ax.set(xlabel="y  (energy)", ylabel="projection", yticks=[], ylim=(0, 1.75),
           title="TCAV energy kick dE ~ +-eVk x (Panofsky-Wenzel)")
    ax.text(0.0, 1.72, "x = x0 + D d  ->  shift +-k x0, stretch 1 +- kD\nreal: -V vs +V shift ~4-5 px, stretch ~2-5%\nCheetah reproduces it (k = eVk/E ~ 0.6 /m)",
            fontsize=7.5, color=INK2, va="top")
    ax.annotate("valley moves\nbetween signs", (0.65, 0.05), (0.84, 0.55), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
    ax.legend(fontsize=7.5, loc="center left", bbox_to_anchor=(0.0, 0.55), frameon=False)
    fig.suptitle("XTCAV two-image measurement and why -V needs the +V cut (schematic)", fontsize=11, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_pm_measurement_pw.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def campaign():
    rng = np.random.default_rng(3)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    n_pts, gold_lin = 22, 0.5
    test = set(rng.choice(np.arange(1, n_pts), 3, replace=False))
    for i in range(1, n_pts):
        yy = i
        c = RED if i in test else BLUE
        ax.scatter([gold_lin], [yy], marker="s", s=28, color=VIOLET, zorder=3)
        lin = np.r_[rng.uniform(0, 1, 6), rng.uniform(0.38, 0.62, 4)]
        ax.scatter(lin, np.full(10, yy), s=10, color=c, alpha=0.85, zorder=2)
        npair = 2 if i in test else (1 if rng.random() < 0.6 else 0)
        for x in rng.choice(lin, npair, replace=False):
            ax.plot([x, x], [yy, 0], color=ORANGE, lw=0.8, alpha=0.7, zorder=1)
            ax.scatter([x], [0], s=14, color=ORANGE, zorder=3)
    ax.scatter(rng.uniform(0, 1, 120), np.full(120, -1.2), s=5, color=INK2, alpha=0.6)
    ax.axvspan(0.38, 0.62, color=AQUA, alpha=0.10)
    ax.axhline(0, color=GRID, lw=6, zorder=0)
    ax.set_yticks([-1.2, 0, n_pts / 2], ["old xtcav23\n(golden h5 beam)", "golden injector", "injector points\n(LHS, schematic)"])
    ax.set_xticks([0, gold_lin, 1], ["box edge", "golden linac", "box edge"])
    ax.set(xlabel="linac knobs (23 + 2)", ylim=(-2, n_pts + 0.5), xlim=(-0.03, 1.03),
           title="injector_full design: 250 LHS injector points x (1 crossed + 10 nested) + 275 pairs + 25 test points")
    h = [plt.Line2D([], [], marker="s", ls="", color=VIOLET, label="crossed: golden linac (injector response, 251)"),
         plt.Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="nested linac draws (2500; 40% controller region)"),
         plt.Line2D([], [], marker="o", ls="", color=RED, ms=4, label="test injector points (25, held out)"),
         plt.Line2D([], [], color=ORANGE, label="pair: same linac, injector = golden (275)"),
         plt.Rectangle((0, 0), 1, 1, color=AQUA, alpha=0.2, label="controller region"),
         plt.Line2D([], [], marker="o", ls="", color=INK2, ms=3, label="old campaigns (5.5k, golden injector)")]
    ax.legend(handles=h, fontsize=7.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_campaign_design.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def control_loop():
    fig, ax = canvas(13, 4.6, 13, 4.6)
    box(ax, 0.2, 1.5, 2.2, 1.7, "Policy (BPTT)", "obs: features, errors,\ntrend, cum. dL1 readback,\n8-step history", fc="#eef3fb")
    box(ax, 3.2, 2.7, 2.7, 1.55, "BC11 BLEN loop (on)", "action d ln(setpoint)\ndL1 = -g1 d ln,  g1 ~ U(1.5, 4)\n|cum| <= 0.75; loop drift", new=True)
    box(ax, 3.2, 0.75, 2.7, 1.55, "L2_PHASE.MKB (rel.)", "BC14 loop off, energy FB on\ndL2 = g2 a,  g2 ~ U(0.35, 1.1)\n|cum| <= 3.5 deg", new=True)
    box(ax, 6.6, 1.5, 2.4, 1.7, "Beam (surrogate)", "hidden injector context\nhidden phase drift\n0.02 / 0.05 deg/step", fc="#f3f2ee")
    box(ax, 9.6, 1.5, 3.2, 1.7, "XTCAV +-V screen", "M = 30 images/step\nshot noise 15 um, floor 6 um\n1% outliers, scale U(0.8, 1.2)\nvalid: peaks, |s| > 60 um, edge", new=True)
    arrow(ax, (2.4, 2.75), (3.2, 3.3)); arrow(ax, (2.4, 1.95), (3.2, 1.45))
    arrow(ax, (5.9, 3.3), (6.6, 2.75), "dL1", off=(0.15, 0.12)); arrow(ax, (5.9, 1.45), (6.6, 1.95), "dL2", off=(0.15, -0.3))
    arrow(ax, (9.0, 2.35), (9.6, 2.35))
    # measurement feedback along the bottom, readback along the top (straight segments)
    ax.plot([11.2, 11.2, 1.3], [1.5, 0.3, 0.3], color=INK2, lw=1.2)
    arrow(ax, (1.3, 0.3), (1.3, 1.5))
    ax.text(6.3, 0.38, "measured spacing, drive sz, charge fraction, deployed validity", ha="center", fontsize=7.8, color=INK2)
    ax.plot([4.55, 4.55, 1.3], [4.25, 4.45, 4.45], color=INK2, lw=1.2)
    arrow(ax, (1.3, 4.45), (1.3, 3.2))
    ax.text(2.9, 4.5, "FAST_PACT dL1 readback", ha="center", fontsize=7.8, color=INK2)
    ax.text(0.2, 4.6, "Control loop in env v4 (orange = machine actuator / measurement models)", fontsize=11, weight="bold", va="bottom")
    fig.savefig(OUT / "2026-10-15_control_loop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def rl_summary():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw={"width_ratios": [1.15, 1]})
    groups = [("v3 cloud (easier env)", [2.6], INK2), ("S1 cloud, blen_dlog", [5.0, 4.6, 5.0, 4.9], BLUE),
              ("S1 cloud, adapter", [5.0, 4.9], AQUA), ("S2 screen v5, 300 it", [8.7, 9.2, 8.3], ORANGE),
              ("S2 screen v5, 600 it", [7.6, 7.3, 7.2], ORANGE)]
    ax = axes[0]
    for i, (name, v, c) in enumerate(groups[::-1]):
        ax.scatter(v, np.full(len(v), i), s=40, color=c, zorder=3)
        ax.text(max(v) + 0.25, i, f"{np.mean(v):.1f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(groups)), [g[0] for g in groups[::-1]])
    ax.axvline(10, color=GRID, lw=1)
    ax.set(xlabel="true median spacing error [um] (dots = seeds)", xlim=(0, 11),
           title="Final spacing error, corrected eval")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax = axes[1]
    rob = [("nominal", 5.0), ("noise x2", 6.5), ("drift x2", 6.0), ("gains widened", 5.4), ("scale +-30%", 5.1),
           ("64-step episodes", 5.1)]
    ax.barh(range(len(rob)), [r[1] for r in rob][::-1], color=BLUE, height=0.55)
    ax.set_yticks(range(len(rob)), [r[0] for r in rob][::-1])
    for i, (_, v) in enumerate(rob[::-1]):
        ax.text(v + 0.1, i, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set(xlabel="true median spacing error [um]", xlim=(0, 8), title="S1 blen_dlog s0: eval-only robustness")
    ax.grid(axis="x", color=GRID, lw=0.6)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_rl_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def surrogate_summary():
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    rows = [("old flow (widened)", 12.6), ("injector fine-tune (flow)", 8.2), ("gradient boosting", 8.5),
            ("poly-2 ridge", 9.5), ("MLP ensemble (3)", 3.5)]
    cols = [INK2, BLUE, "#b9b7b0", "#b9b7b0", AQUA]
    ax.barh(range(len(rows)), [r[1] for r in rows][::-1], color=cols[::-1], height=0.55)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
    for i, (_, v) in enumerate(rows[::-1]):
        ax.text(v + 0.15, i, f"{v:.1f}", va="center", fontsize=8, color=INK2)
    ax.set(xlabel="held-out absolute spacing MAE [um]", xlim=(0, 14), title="Same data, different models")
    ax = axes[1]
    labels = ["main effect\n(golden linac)", "injector x linac\n(pairs)"]
    vals = {"fine-tuned flow": [0.95, 0.69], "MLP ensemble": [np.nan, 0.90], "GP / ridge (best simple)": [0.99, 0.76]}
    x = np.arange(2)
    for k, (name, v) in enumerate(vals.items()):
        ax.bar(x + (k - 1) * 0.26, v, 0.24, color=[BLUE, AQUA, "#b9b7b0"][k], label=name)
    ax.axhline(0.8, color=INK, ls=":", lw=1)
    ax.text(1.42, 0.81, "gate 0.8", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set(ylabel="held-out R2 of the injector effect", ylim=(0, 1.05), title="Main effect learned; interaction underfit")
    ax.legend(fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_surrogate_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    for f in (pipeline, pm_measurement, campaign, control_loop, rl_summary, surrogate_summary):
        f()
