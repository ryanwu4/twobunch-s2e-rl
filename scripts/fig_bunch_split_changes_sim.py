"""Figures for the overnight 2026-10-15 split/validity changes, sim part (slac-rl env): which simulated
screen readings the crop-edge check (S2 v5) affects, over noise-free readings of the S2 v5 env (random goal
points, hidden context / optics / TCAV voltage varied).

  PYTHONPATH=src python scripts/fig_bunch_split_changes_sim.py --out results/bunch_split_changes_2026-10-15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from twobunch_s2e_rl.rl._train_utils import build_env_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3072)
    a = ap.parse_args()
    out = Path(a.out)
    cfg = yaml.safe_load(Path("configs/rl/delta_xtcav23_screen_v5_blen.yaml").read_text())
    de = cfg["params"]["diff_env"]
    de["v4"]["goal_valid_retries"] = 0                       # raw goal points (no redraw)
    de["screen"]["edge_frac_max"] = None                     # old rule from the env; new rule applied below
    env = build_env_fn(cfg, de["flow_ckpt"])(num_envs=a.n, device="cuda", seed=21, episode_length=32, no_grad=True)
    env.reset()
    with torch.no_grad():
        o = env._plant(env._goal_ctrl, noise=False)
    g = lambda k: o[k].float().cpu().numpy()
    true, meas = 1e6 * g("bunch_spacing_cloud"), 1e6 * g("bunch_spacing_true")
    edge, clip, mis = g("screen_edge_frac"), g("screen_clipped"), g("screen_misassign")
    dep_old = g("screen_valid_deployed") > 0.5               # two peaks + |spacing| > 60 um
    dep_new = dep_old & (edge < 0.01)                         # + crop-edge check (S2 v5)
    truth_new = (g("screen_valid") > 0.5) & (clip < 0.1)      # + clipped charge counts (S2 v5)
    bias = meas - true
    ctrl = env._goal_ctrl.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    ax = axes[0]
    cat = [("valid (old and new rule)", dep_old & dep_new, "tab:blue"),
           ("valid under old rule, flagged by crop-edge check (new)", dep_old & ~dep_new, "tab:orange"),
           ("invalid under both (no valley or |spacing| < 60 um)", ~dep_old, "0.7")]
    for name, m, c in cat:
        ax.scatter(true[m], np.clip(bias[m], -700, 200), s=5, c=c, alpha=0.6, label=f"{name}: {m.mean():.0%}")
    ax.axhline(0, c="k", lw=0.8)
    ax.set(xlabel="true spacing [um]", ylabel="measured - true [um] (clipped at -700)",
           title="sim screen readings (noise-free), deployed validity")
    ax.legend(fontsize=7, loc="lower left")
    ax = axes[1]
    sc = ax.scatter(true, 100 * edge, c=np.clip(clip, 0, 0.5), s=5, cmap="viridis")
    ax.axhline(1.0, c="tab:orange", ls="--", label="edge threshold 1%")
    fig.colorbar(sc, ax=ax, label="charge clipped out of the crop window (truth)")
    ax.set(xlabel="true spacing [um]", ylabel="charge within 0.3 mm of crop edge [%]",
           title="crop-edge fraction vs spacing (4.6 mm half-width auto-crop)")
    ax.legend(fontsize=7)
    ax = axes[2]
    for name, m, c in cat[:2]:
        ax.scatter(ctrl[m, 0], ctrl[m, 1], s=6, c=c, alpha=0.6, label=name)
    ax.set(xlabel="L1 phase (sim) [deg]", ylabel="L2 phase (sim) [deg]",
           title="where the flagged readings are (start box)")
    ax.legend(fontsize=7)
    fig.suptitle("Sim (S2 v5 env): the crop-edge check removes clipped large-spacing readings that the old deployed "
                 "rule called valid", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig7_sim_crop_edge.png", dpi=130)
    plt.close(fig)

    def stats(m):
        b = np.abs(bias[m])
        return f"n={m.sum()} ({m.mean():.0%}) |meas-true| median {np.median(b):.1f} p90 {np.percentile(b, 90):.1f} um"
    lines = ["sim S2 v5 env, noise-free goal-point readings (no goal redraw), n=%d" % a.n,
             "old deployed-valid: " + stats(dep_old),
             "new deployed-valid: " + stats(dep_new),
             "flagged by the edge check only: " + stats(dep_old & ~dep_new),
             f"truth-valid (misassign < 10%, clipped < 10%): {truth_new.mean():.0%}; clipped > 10% among old-valid: "
             f"{(clip[dep_old] > 0.1).mean():.1%}, among new-valid: {(clip[dep_new] > 0.1).mean():.1%}"]
    (out / "sim_crop_edge_stats.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
