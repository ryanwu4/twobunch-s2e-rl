"""Figures for the 2026-10-17 notebook entry: RL delta controller with BLEN observations (surrogate evals) and
closed-loop Bmad validation. Writes lab-notebook/images/2026-10-17_*.png. Read-only on results/ and logs/."""
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# one fixed color per policy family, used in every figure
POL = {
    "blen": ("blen_dlog (flow A)", "#2a78d6"),
    "l1v2": ("direct-L1 v2, no BLEN obs (flow A)", "#eb6834"),
    "l1blenA": ("direct-L1 + BLEN obs (flow A)", "#1baf7a"),
    "l1blenB": ("direct-L1 + BLEN obs (B2-cov)", "#6250d6"),
}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK})
CL = Path("results/closed_loop_bmad_2026-10-17")
CL_TAGS = [("S2blen_flowA", "blen"), ("S2l1blen_flowA", "l1blenA"), ("S2l1blen_covB2", "l1blenB")]
RNG = np.random.default_rng(0)
SHORT = {"blen": "blen_dlog\n(flow A)", "l1blenA": "direct-L1\n+BLEN\n(flow A)", "l1blenB": "direct-L1\n+BLEN\n(B2-cov)"}


# ---------------------------------------------------------------- 1. surrogate evals
def _parse_summary(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"(\S+) \| (.+?) \| .*final true ([\d.]+) um .*in10 meas [\d.]+ true ([\d.]+)", line)
        if m:
            out[(m.group(1), m.group(2))] = (float(m.group(3)), float(m.group(4)))
    return out


def _parse_table(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\| (S[12] [^|]+?) \| ([\d.]+) / (\d+)%", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), float(m.group(3)) / 100)
    return out


def surrogate_eval():
    cov = _parse_summary("results/cross_eval_2026-10-17/covB2/summary.txt")
    l1b = _parse_summary("logs/2026-10-16/l1blen/summary.txt")
    tab = _parse_table("results/cross_eval_2026-10-16/flowA/TABLE.md")
    # (policy family, eval plant, value) ; all on injector-varied (DAQ) context, 512 episodes
    data = {"S2": [], "S1": []}
    for s, scr in (("S2", "screen_v6"), ("S1", "cloud")):
        d = data[s]
        for sd in ("s0", "s1"):
            d.append(("blen", "A", tab[f"{s} {'v6 ' if s == 'S2' else ''}blen injflow {sd}"]))
        d.append(("blen", "B", cov[(f"injflow_flowA_{scr}_blen_{'s0' if s == 'S2' else 's1'}", "covB2 inj")]))
        d.append(("l1v2", "A", tab[f"{s} {'v6 ' if s == 'S2' else ''}l1direct v2 injflow s0"]))
        d.append(("l1blenA", "A", l1b[(f"injflow_flowA_{scr}_l1blen", "own")]))
        d.append(("l1blenA", "B", cov[(f"injflow_flowA_{scr}_l1blen_s0", "covB2 inj")]))
        for sd in ("s0", "s1"):
            d.append(("l1blenB", "B", cov[(f"injflow_covB2_{scr}_l1blen_{sd}", "covB2 inj")]))
    order = ["blen", "l1v2", "l1blenA", "l1blenB"]
    xt = ["blen_dlog\n(flow A)", "direct-L1 v2\nno BLEN (flow A)", "direct-L1\n+ BLEN (flow A)", "direct-L1\n+ BLEN (B2-cov)"]
    mk = {"A": "o", "B": "s"}
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 6.6), sharex=True)
    for j, s in enumerate(("S2", "S1")):
        for i, (k, ylab) in enumerate(((0, "median true spacing error [µm]"), (1, "fraction within 10 µm (true)"))):
            ax = axes[i, j]
            for pol, plant, v in data[s]:
                x = order.index(pol) + (-0.12 if plant == "A" else 0.12) + RNG.uniform(-0.04, 0.04)
                c = POL[pol][1]
                ax.scatter(x, v[k], s=46, marker=mk[plant], color=c if plant == "A" else SURF, edgecolor=c,
                           lw=1.6, zorder=3)
            ref = [v[k] for p, _, v in data[s] if p == "blen"]
            ax.axhspan(min(ref), max(ref), color=POL["blen"][1], alpha=0.08, lw=0, zorder=1)
            if i == 0 and s == "S2":
                ax.axhline(10, color=INK2, ls=":", lw=1)
                ax.text(3.45, 10, "tol 10 µm", color=INK2, fontsize=7, va="bottom", ha="right")
            ax.set_ylabel(ylab)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_xlim(-0.5, 3.5)
        axes[0, j].set_title(("S2: XTCAV-screen obs (deployable)" if s == "S2" else "S1: true-beam obs (ceiling)"),
                             fontsize=9.5, loc="left")
        axes[1, j].set_xticks(range(4), xt, fontsize=8)
    axes[0, 0].set_ylim(6, 14)
    axes[0, 1].set_ylim(3.5, 7.5)
    axes[1, 0].set_ylim(0.38, 0.65)
    axes[1, 1].set_ylim(0.62, 0.88)
    h = [plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7, label="eval plant: flow A"),
         plt.Line2D([], [], marker="s", ls="", mfc=SURF, mec=INK2, mew=1.6, ms=7, label="eval plant: B2-cov"),
         plt.Rectangle((0, 0), 1, 1, color=POL["blen"][1], alpha=0.15, label="blen_dlog range")]
    axes[0, 1].legend(handles=h, fontsize=7.5, frameon=False, loc="upper right")
    fig.suptitle("Surrogate eval (512 episodes, DAQ injector context): BLEN observations bring direct-L1 back to "
                 "blen_dlog parity (S2 12.8 → 9.0 µm)", fontsize=10, x=0.01, ha="left", weight="bold")
    fig.text(0.01, 0.005, "Points = seeds / eval runs; repeat evals of one policy scatter by ~±0.4 µm. Filled = flow A "
             "plant, hollow = B2-cov plant.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(OUT / "2026-10-17_rl_surrogate_eval.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- closed loop helpers
def _cl(tag, mode, plant):
    d = CL / f"{tag}_{mode}_{plant}"
    s = json.loads((d / "summary.json").read_text())["targets"]
    return d, s


def _boot_median(x, n=4000):
    b = np.median(RNG.choice(x, size=(n, len(x)), replace=True), axis=1)
    return np.percentile(b, [5, 95])


def _wilson(k, n, z=1.645):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


# ---------------------------------------------------------------- 2. closed loop summary
def closed_loop():
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2))
    labels, x0 = [], 0
    for mode in ("golden", "inj"):
        for tag, pol in CL_TAGS:
            c = POL[pol][1]
            for dx, plant in ((-0.18, "bmad"), (0.18, "flow")):
                _, s = _cl(tag, mode, plant)
                e = np.array(s["bunch_spacing"]["per_episode_final_true"]) * 1e6
                x = x0 + dx
                filled = plant == "bmad"
                # per-episode dots
                axes[0].scatter(x + RNG.uniform(-0.07, 0.07, len(e)), e, s=9, color=c, alpha=0.35, lw=0, zorder=2)
                med = np.median(e)
                lo, hi = _boot_median(e)
                axes[0].errorbar(x, med, yerr=[[med - lo], [hi - med]], fmt="o" if filled else "D", ms=7,
                                 color=c, mfc=c if filled else SURF, mew=1.6, capsize=3, lw=1.4, zorder=4)
                k = int(np.sum(e < 10))
                f = k / len(e)
                lo, hi = _wilson(k, len(e))
                axes[1].errorbar(x, f, yerr=[[f - lo], [hi - f]], fmt="o" if filled else "D", ms=7, color=c,
                                 mfc=c if filled else SURF, mew=1.6, capsize=3, lw=1.4, zorder=4)
            labels.append(SHORT[pol])
            x0 += 1
    for ax in axes:
        ax.set_xticks(range(6), labels, fontsize=7.5)
        ax.axvline(2.5, color=GRID, lw=1)
        for xc, t in ((1, "golden injector"), (4, "varied injector (DAQ)")):
            ax.text(xc, -0.27, t, transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=INK)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_xlim(-0.6, 5.6)
    axes[0].set_yscale("log")
    axes[0].axhline(10, color=INK2, ls=":", lw=1)
    axes[0].set_ylabel("final true |spacing − goal| [µm]")
    axes[0].set_title("Median final spacing error (90% bootstrap CI) + per-episode", fontsize=9.5, loc="left")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("fraction within 10 µm (true)")
    axes[1].set_title("Fraction within 10 µm (90% Wilson CI)", fontsize=9.5, loc="left")
    h = [plt.Line2D([], [], marker="o", ls="", color=INK2, ms=7, label="Bmad (closed loop)"),
         plt.Line2D([], [], marker="D", ls="", mfc=SURF, mec=INK2, mew=1.6, ms=7, label="flow surrogate (same episodes)")]
    axes[1].legend(handles=h, fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle("Closed-loop S2 (screen-obs) policies in Bmad ≈ or better than the paired surrogate run in 5/6 cases\n"
                 "(n = 24 episodes each, 32 steps, 2e4 macroparticles)", fontsize=10, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_closed_loop_bmad.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- 3. trajectories
def traj():
    tag, mode, pol = "S2l1blen_covB2", "inj", "l1blenB"
    c = POL[pol][1]
    t = {p: torch.load(CL / f"{tag}_{mode}_{p}" / "traj.pt", weights_only=False) for p in ("bmad", "flow")}

    def errs(tt):
        sp = (tt["bunch_spacing_cloud"] - tt["spacing_goal_cloud"]).abs().numpy() * 1e6
        sz = ((tt["drive_sigma_z_cloud"] - tt["drive_sigma_z_goal_cloud"]).abs() / tt["drive_sigma_z_goal_cloud"]).numpy()
        return sp, sz

    b, f = errs(t["bmad"]), errs(t["flow"])
    steps = np.arange(b[0].shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.2))
    for ax, i, tol, ylab in ((axes[0], 0, 10, "true |spacing − goal| [µm]"),
                             (axes[1], 1, 0.10, "drive σz relative error |σz − goal| / goal")):
        ax.plot(steps, b[i], color=c, alpha=0.22, lw=0.8)
        ax.plot(steps, np.median(b[i], axis=1), color=c, lw=2.4, label="Bmad median (24 episodes)")
        ax.plot(steps, np.median(f[i], axis=1), color=INK2, lw=1.8, ls="--", label="flow median (paired episodes)")
        ax.axhline(tol, color=INK2, ls=":", lw=1)
        ax.text(steps[-1] - 0.3, tol, f"tol {'10 µm' if i == 0 else '10%'}", color=INK2, fontsize=7.5, va="bottom",
                ha="right", bbox=dict(fc=SURF, ec="none", pad=1), zorder=5)
        ax.set_yscale("log")
        ax.set_xlabel("controller step")
        ax.set_ylabel(ylab)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlim(0, steps[-1])
    axes[0].plot([], [], color=c, alpha=0.4, lw=0.8, label="Bmad episodes")
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_title("Bunch spacing", fontsize=9.5, loc="left")
    axes[1].set_title("Drive bunch length", fontsize=9.5, loc="left")
    mb, mf = np.median(b[0][-1]), np.median(f[0][-1])
    fig.suptitle(f"Closed-loop Bmad, S2 direct-L1 + BLEN (B2-cov), varied injector: median final spacing error "
                 f"{mb:.1f} µm (flow {mf:.1f} µm)", fontsize=10, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_closed_loop_traj.png", dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------- 4. paired scatter
def paired():
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.4), sharex=True, sharey=True)
    lim = (0.3, 400)
    for r, mode in enumerate(("golden", "inj")):
        for j, (tag, pol) in enumerate(CL_TAGS):
            ax = axes[r, j]
            c = POL[pol][1]
            eb = np.array(_cl(tag, mode, "bmad")[1]["bunch_spacing"]["per_episode_final_true"]) * 1e6
            ef = np.array(_cl(tag, mode, "flow")[1]["bunch_spacing"]["per_episode_final_true"]) * 1e6
            ax.fill_between([lim[0], 10], lim[0], 10, color=GRID, alpha=0.5, lw=0, zorder=0)
            ax.plot(lim, lim, color=INK2, lw=1, zorder=1)
            ax.scatter(ef, eb, s=34, color=c, edgecolor=SURF, lw=1, zorder=3)
            nb = int(np.sum(eb < ef))
            ax.text(0.04, 0.96, f"Bmad better: {nb}/24\nmedian Bmad {np.median(eb):.1f} / flow {np.median(ef):.1f} µm",
                    transform=ax.transAxes, fontsize=7.5, va="top", color=INK)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect("equal")
            ax.grid(color=GRID, lw=0.5)
            ax.set_title(f"{POL[pol][0]} · {'golden' if mode == 'golden' else 'varied'} injector", fontsize=8.5,
                         loc="left", color=INK)
            if r == 1:
                ax.set_xlabel("flow surrogate final error [µm]")
            if j == 0:
                ax.set_ylabel("Bmad final error [µm]")
    fig.suptitle("Paired episodes (same injector point, start knobs, goal setpoint): final true spacing error.\n"
                 "Grey = both within 10 µm; below the 1:1 line = Bmad better", fontsize=9.5, x=0.01, ha="left", weight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_closed_loop_paired.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    surrogate_eval()
    closed_loop()
    traj()
    paired()
