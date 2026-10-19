"""Presentation GIF (2026-10-19): one continuous-operation episode of the delta controller tracking setpoint slews on
noisy screen readings. Policy zs_screen_noblen_jit_s0 (both loops off, spacing-head flow) in its own env under the
nominal-jitter eval condition (configs/rl/eval/jitter_nominal_2026-10-18.yaml: DAQ-calibrated per-image jitter, camera
noise, beam-side phase wander, 2.5% outlier steps) with the setpoint radius doubled (4 / 5 deg) so new setpoints are
real slews. 64 episodes are rolled out; the shown one is the episode with >= 3 setpoints and >= 1 slew > 80 um whose
median settled error is closest to the median over those episodes (representative, not the best).
Panels: spacing vs step (true, readings, goal +-10 um), the beam's longitudinal phase space at XTCAVF (surrogate
sample at the true controls), the L1/L2 path. Output results/5_reports/slew_gif_2026-10-19/slew.gif (+ a final-frame png).

  PYTHONPATH=src python scripts/slew_gif_2026-10-19.py [--device cuda:0]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from twobunch_s2e_rl.artifacts import policy_flow_ckpt, resolve_ckpt  # noqa: E402
from twobunch_s2e_rl.rl._train_utils import build_env_fn  # noqa: E402
from twobunch_s2e_rl.rl.eval_delta import _deep_merge  # noqa: E402
from twobunch_s2e_rl.rl.rollout import act, load_policy  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

LOGDIR = "logs/2026-10-18/night/zs_screen_noblen_jit_s0"
NOM = "configs/rl/eval/jitter_nominal_2026-10-18.yaml"
OUT = Path("results/5_reports/slew_gif_2026-10-19")
K, STEPS = 64, 96
STEP_S = 20.0                                    # one controller step ~ 20 s (DAQ realism)
C_TRUE, C_MEAS, C_GOAL, C_DRV, C_WIT = "#4dabf7", "#ffa94d", "#e9ecef", "#4dabf7", "#ffa94d"
BG, FG, MUTED = "#000000", "#e9ecef", "#868e96"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--pick", choices=("median", "good"), default="median",
                    help="median: settled error closest to the median; good: every setpoint reached, 25th percentile")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)                          # flow sampling draws from the global RNG
    cfg = yaml.safe_load((Path(LOGDIR) / "cfg.yaml").read_text())
    de = _deep_merge(cfg["params"]["diff_env"], yaml.safe_load(Path(NOM).read_text()))
    de["v4"]["goal_radius"] = {"L1PhaseSet": 4.0, "L2PhaseSet": 5.0}
    cfg["params"]["diff_env"] = de
    ck = policy_flow_ckpt(LOGDIR)
    env = build_env_fn(cfg, ck)(num_envs=K, device=a.device, seed=2468, episode_length=10 ** 6, no_grad=True)
    actor, rms = load_policy(str(Path(LOGDIR) / "final_policy.pt"), a.device)
    env.reset()
    gk = next(g for key, g, _ in env._goal_terms if key == "bunch_spacing")
    rec = {k: [] for k in ("true", "meas", "goal", "valid", "ctrl", "goal_ctrl")}
    obs = env._compute_obs()
    for _ in range(STEPS):
        obs, _, _, info = env.step(act(actor, rms, obs))
        ach = info["achieved"]
        rec["true"].append(ach["bunch_spacing_cloud"].cpu().numpy())
        rec["meas"].append(ach["bunch_spacing_meas"].cpu().numpy())
        rec["goal"].append(ach[f"{gk}_cloud"].cpu().numpy())
        rec["valid"].append(ach["screen_valid_deployed_true"].cpu().numpy())
        rec["ctrl"].append(ach["true_controls"].cpu().numpy())
        rec["goal_ctrl"].append(env._goal_ctrl.cpu().numpy())
    R = {k: np.stack(v) * (1e6 if k in ("true", "meas", "goal") else 1) for k, v in rec.items()}

    # pick a representative episode
    cands, scores = [], []
    for b in range(K):
        g = R["goal"][:, b]
        chg = np.flatnonzero(np.abs(np.diff(g)) > 1e-3) + 1
        starts = np.r_[0, chg, STEPS]
        jumps = np.abs(np.diff(g))[chg - 1] if len(chg) else np.array([])
        if len(starts) - 1 < 3 or not (jumps > 80).any():
            continue
        segs = [(s0, s1) for s0, s1 in zip(starts[:-1], starts[1:]) if s1 - s0 >= 12]
        settled = [np.median(np.abs(R["true"][s1 - 4:s1, b] - g[s1 - 4:s1])) for s0, s1 in segs]
        reached = all((np.abs(R["true"][s0:s1, b] - g[s0:s1]) <= 10).any() for s0, s1 in segs)
        if settled and (a.pick == "median" or reached):
            cands.append(b)
            scores.append(np.median(settled))
    scores = np.array(scores)
    ref = np.median(scores) if a.pick == "median" else np.percentile(scores, 25)
    b = cands[int(np.argmin(np.abs(scores - ref)))]
    print(f"{len(cands)} candidate episodes; showing env {b} (median settled error {scores[cands.index(b)]:.1f} um, "
          f"population median {np.median(scores):.1f} um)", flush=True)

    # beam LPS per step (surrogate sample at the true controls of the shown episode)
    flow = TwoBunchFlow.load_from_checkpoint(resolve_ckpt(ck), map_location=a.device).eval()
    ks, ci = env._ks, env._ci
    ctx = env._context[b:b + 1]
    lps = []
    with torch.no_grad():
        for t in range(STEPS):
            kn = ks.normalize(ctx.index_copy(1, ci, torch.as_tensor(R["ctrl"][t, b:b + 1], device=ctx.device)))
            torch.manual_seed(t)
            d, w = flow.sample_bunch(kn, 0, 3000)[0].cpu().numpy(), flow.sample_bunch(kn, 1, 1500)[0].cpu().numpy()
            z0, p0 = d[:, 2].mean(), d[:, 5].mean()
            lps.append(((d[:, 2] - z0) * 1e6, (d[:, 5] / p0 - 1) * 100, (w[:, 2] - z0) * 1e6, (w[:, 5] / p0 - 1) * 100))
    allz = np.concatenate([np.r_[x[0], x[2]] for x in lps])
    alle = np.concatenate([np.r_[x[1], x[3]] for x in lps])
    zl, el = np.percentile(allz, [0.5, 99.5]), np.percentile(alle, [0.5, 99.5])

    tru, mea, goal, val = R["true"][:, b], R["meas"][:, b], R["goal"][:, b], R["valid"][:, b] > 0.5
    ctrl, gctl = R["ctrl"][:, b], R["goal_ctrl"][:, b]
    tmin = np.arange(1, STEPS + 1) * STEP_S / 60
    plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
                         "axes.edgecolor": MUTED, "axes.labelcolor": FG, "text.color": FG,
                         "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": FG, "ytick.labelcolor": FG,
                         "legend.labelcolor": FG, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(13, 6.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.75, 1], hspace=0.42, wspace=0.24, left=0.07, right=0.98, top=0.86, bottom=0.11)
    ax, axl, axc = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    lo, hi = min(tru.min(), goal.min()) - 45, max(tru.max(), goal.max()) + 45      # outlier readings clipped to the edge

    def draw(t):
        for x in (ax, axl, axc):
            x.cla()
        s = slice(0, t + 1)
        ax.fill_between(tmin[s], goal[s] - 10, goal[s] + 10, step="mid", color=C_GOAL, alpha=0.14, lw=0, label="setpoint ± 10 µm")
        ax.step(tmin[s], goal[s], where="mid", color=C_GOAL, lw=1.6, ls="--", label="setpoint")
        ax.plot(tmin[s], tru[s], color=C_TRUE, lw=2.2, label="true spacing")
        v = val[s]
        on = v & (mea >= lo)[s] & (mea <= hi)[s]
        off = v & ~((mea >= lo) & (mea <= hi))[s]
        ax.scatter(tmin[s][on], mea[s][on], s=24, color=C_MEAS, edgecolor="none", zorder=3, label="screen reading")
        ax.scatter(tmin[s][off], np.where(mea[s][off] < lo, lo + 4, hi - 4), s=34, marker="v", color=C_MEAS,
                   facecolor="none", lw=1.2, zorder=3, label="off-scale reading")
        ax.scatter(tmin[s][~v], np.clip(mea[s][~v], lo, hi), s=36, marker="x", color=MUTED, zorder=3, label="invalid reading")
        ax.set_xlim(0, tmin[-1] + 0.3)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("time [min]")
        ax.set_ylabel("drive–witness spacing [µm]")
        ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), fontsize=11, frameon=False, ncol=3,
                  handletextpad=0.5, columnspacing=1.4)
        zd, ed, zw, ew = lps[t]
        axl.scatter(zd, ed, s=1.2, color=C_DRV, alpha=0.35, edgecolor="none", rasterized=True)
        axl.scatter(zw, ew, s=1.2, color=C_WIT, alpha=0.5, edgecolor="none", rasterized=True)
        axl.text(0.03, 0.92, "drive", color=C_DRV, transform=axl.transAxes, fontsize=11)
        axl.text(0.03, 0.80, "witness", color=C_WIT, transform=axl.transAxes, fontsize=11)
        axl.set_xlim(*zl)
        axl.set_ylim(*el)
        axl.set_xlabel("z − drive centroid [µm]")
        axl.set_ylabel("ΔE/E [%]")
        axc.plot(ctrl[s, 0], ctrl[s, 1], color=MUTED, lw=1.2)
        axc.scatter(ctrl[t, 0], ctrl[t, 1], s=50, color=C_TRUE, zorder=3)
        axc.scatter(gctl[t, 0], gctl[t, 1], s=140, marker="*", color=C_GOAL, zorder=3)
        axc.set_xlim(ctrl[:, 0].min() - 0.6, ctrl[:, 0].max() + 0.6)
        axc.set_ylim(ctrl[:, 1].min() - 0.6, ctrl[:, 1].max() + 0.6)
        axc.xaxis.set_major_locator(plt.MaxNLocator(5))
        axc.yaxis.set_major_locator(plt.MaxNLocator(5))
        axc.set_xlabel("L1 phase [deg]")
        axc.set_ylabel("L2 phase [deg]")

    anim = FuncAnimation(fig, draw, frames=STEPS, interval=150)
    anim.save(OUT / f"slew_{a.pick}.gif", writer=PillowWriter(fps=7), dpi=90)
    draw(STEPS - 1)
    fig.savefig(OUT / f"slew_{a.pick}_final.png", dpi=130)
    print("saved", OUT / f"slew_{a.pick}.gif", flush=True)


if __name__ == "__main__":
    main()
