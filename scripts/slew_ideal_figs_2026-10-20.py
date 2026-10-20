"""Noise-free Bmad slew eval figures (2026-10-20): per run in results/3_rl_eval/slew_ideal_bmad_2026-10-19/*_bmad, true
spacing vs step for 12 episodes (goals dashed, +-10 um band), and a per-setpoint summary across runs (final error median,
within 10 um, reached), Bmad vs the paired surrogate run. Outputs <dir>/slew_traces_<run>.png and slew_summary.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

D = Path("results/3_rl_eval/slew_ideal_bmad_2026-10-19")
runs = sorted(p.name[:-5] for p in D.glob("*_bmad") if (p / "summary.json").exists())
summ = {}
for r in runs:
    t = torch.load(D / f"{r}_bmad" / "traj.pt")
    tr, g = t["bunch_spacing_cloud"].numpy() * 1e6, t["spacing_goal_cloud"].numpy() * 1e6
    T, K = tr.shape
    fig, axs = plt.subplots(3, 4, figsize=(14, 8), sharex=True)
    for ax, b in zip(axs.flat, np.linspace(0, K - 1, 12).astype(int)):
        ax.fill_between(np.arange(T), g[:, b] - 10, g[:, b] + 10, step="mid", color="0.85", lw=0)
        ax.step(np.arange(T), g[:, b], where="mid", color="k", ls="--", lw=1)
        ax.plot(np.arange(T), tr[:, b], color="#2a6fdb", lw=1.6)
        ax.set_title(f"episode {b}", fontsize=9)
    for ax in axs[-1]:
        ax.set_xlabel("step")
    for ax in axs[:, 0]:
        ax.set_ylabel("true spacing [µm]")
    fig.suptitle(f"{r}: noise-free Bmad, 3 setpoints x 16 steps (dashed = setpoint, grey = ±10 µm)", fontsize=11)
    fig.tight_layout()
    fig.savefig(D / f"slew_traces_{r}.png", dpi=120)
    plt.close(fig)
    for plant in ("bmad", "flow"):
        f = D / f"{r}_{plant}" / "summary.json"
        if f.exists():
            summ[(r, plant)] = json.loads(f.read_text())["segments"]["bunch_spacing"]["segments"]
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
w = 0.8 / max(1, 2 * len(runs))
for i, r in enumerate(runs):
    for j, plant in enumerate(("bmad", "flow")):
        if (r, plant) not in summ:
            continue
        s = summ[(r, plant)]
        x = np.arange(len(s)) + (2 * i + j - len(runs)) * w
        kw = dict(width=w, label=f"{r} ({plant})", alpha=1.0 if plant == "bmad" else 0.45)
        axs[0].bar(x, [q["final_median"] * 1e6 for q in s], **kw)
        axs[1].bar(x, [q["final_within_tol"] for q in s], **kw)
        axs[2].bar(x, [q["reached_frac"] for q in s], **kw)
for ax, t in zip(axs, ("final |error| median [µm]", "final within 10 µm", "reached (any step within 10 µm)")):
    ax.set_xticks(range(3))
    ax.set_xticklabels(["setpoint 1", "setpoint 2", "setpoint 3"])
    ax.set_title(t, fontsize=10)
axs[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(D / "slew_summary.png", dpi=120)
print("runs:", runs)
