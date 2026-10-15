"""Evaluate a trained delta controller on its own (surrogate) env with all hidden randomization on.

Rolls the deterministic policy for one full episode in `--n-episodes` parallel envs (random true
starts, context jitter, gain errors, achievable goals) and reports, per goal-conditioned target:
the error to goal at the start (= doing nothing) and at the end, measured and true, how many steps
the policy needs to get within tolerance, survival, and how the final error depends on the hidden
gain error (robustness). Writes eval.json + figures to results/rl/<run>/delta_eval/ (new dir).

  python -m twobunch_s2e_rl.rl.eval_delta --logdir logs/overnight_2026-10-14/delta_xtcav23_cloud_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from ..artifacts import policy_flow_ckpt, write_run_json
from ..datagen.paths import repo_root
from ._train_utils import build_env_fn
from .rollout import load_policy, rollout

# tolerance per goal metric for "reached": spacing 10 um, drive sigma_z 10 % (relative)
TOL = {"bunch_spacing": ("abs", 10e-6), "drive_sigma_z": ("rel", 0.10)}


def _err(x, g, key):
    kind, _ = TOL.get(key, ("rel", 0.1))
    return np.abs(x - g) if kind == "abs" else np.abs(x - g) / np.abs(g)


def _apply_overrides(de: dict, sets) -> dict:
    """`--set measurement.scale_error_halfwidth=0.3` style overrides of the run's diff_env (eval
    under conditions the policy was not trained on, e.g. larger misspecification)."""
    for item in sets or ():
        path, val = item.split("=", 1)
        node = de
        *parents, leaf = path.split(".")
        for k in parents:
            node = node.setdefault(k, {})
        node[leaf] = yaml.safe_load(val)
    return de


def evaluate(logdir, policy="best", n_episodes=512, device="cuda", seed=12345, out=None,
             sets=(), tag=""):
    device = device if torch.cuda.is_available() else "cpu"
    logdir = Path(logdir)
    cfg = yaml.safe_load((logdir / "cfg.yaml").read_text())
    de = _apply_overrides(cfg["params"]["diff_env"], sets)
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(
        num_envs=n_episodes, device=device, seed=seed,
        episode_length=int(de["episode_length"]), no_grad=True)
    actor, obs_rms = load_policy(str(logdir / f"{policy}_policy.pt"), device)
    # hidden per-episode draws, read BEFORE the rollout: it runs the full episode_length, so the env
    # auto-resets on the last step and afterwards holds the NEXT episode's draws
    gain = getattr(env, "_g", env._gain).detach().cpu().numpy()      # (B, n_ctrl); v4: actuator gain
    scale = env._scale.detach().cpu().numpy()
    traj = rollout(env, actor, obs_rms)
    npy = {k: v.cpu().numpy() for k, v in traj.items()}
    report = {"logdir": str(logdir), "policy": policy, "n_episodes": n_episodes, "seed": seed,
              "overrides": list(sets), "controls": env.controls, "targets": {}}
    for key, gkey, _ in env._goal_terms:
        # measured feature (what the policy sees; the reward view may hold the truth instead, e.g. v4
        # reward_on: true) vs the measured goal
        meas, goal = npy.get(f"{key}_meas", npy[key]), npy[gkey]          # (T, B)
        if f"{key}_cloud" in npy:                     # screen mode: true beam vs true goal point
            true, goal_true = npy[f"{key}_cloud"], npy[f"{gkey}_cloud"]
        else:
            true = npy[f"{key}_true"]
            goal_true = npy[f"{gkey}_cloud"] if f"{gkey}_cloud" in npy else (
                goal / scale if key in env._scale_keys else goal)
        e_meas = _err(meas, goal, key)
        e_true = _err(true, goal_true, key)
        tol = TOL.get(key, ("rel", 0.1))[1]
        within = e_meas <= tol
        # first step after which the error stays within tolerance to the end
        settled = np.array([next((t for t in range(len(w)) if w[t:].all()), -1) for w in within.T])
        report["targets"][key] = {
            "tolerance": TOL.get(key, ("rel", 0.1)),
            "err_start_median": float(np.median(e_meas[0])),
            "err_final_median": float(np.median(e_meas[-1])),
            "err_final_p90": float(np.percentile(e_meas[-1], 90)),
            "err_final_true_median": float(np.median(e_true[-1])),
            "frac_within_tol_final": float(within[-1].mean()),
            "frac_settled": float((settled >= 0).mean()),
            "steps_to_settle_median": (float(np.median(settled[settled >= 0]))
                                       if (settled >= 0).any() else None),
        }
    report["survival"] = {k: float(np.median(npy[k][-1])) for k in ("T_drive", "T_witness") if k in npy}
    if "screen_valid" in npy:
        report["screen_valid_frac"] = {"start": float(npy["screen_valid"][0].mean()),
                                       "final": float(npy["screen_valid"][-1].mean())}
    report["reward_final_median"] = float(np.median(npy["reward"][-1]))

    name = "delta_eval" + (f"_{tag}" if tag else "")
    out = Path(out) if out else repo_root() / "results" / "rl" / logdir.name / name
    write_run_json(out, "delta_eval", {"policy": policy, "n_episodes": n_episodes, "seed": seed,
                                       "overrides": list(sets)},
                   parents=[str(logdir)], flow_ckpt=policy_flow_ckpt(logdir))
    (out / "eval.json").write_text(json.dumps(report, indent=2))

    keys = [k for k, _, _ in env._goal_terms]
    fig, axes = plt.subplots(2, len(keys), figsize=(5 * len(keys), 7.5), squeeze=False)
    for j, key in enumerate(keys):
        gkey = env._goal_terms[j][1]
        e = _err(npy[key], npy[gkey], key)
        sc = 1e6 if TOL.get(key, ("rel",))[0] == "abs" else 100.0
        unit = "um" if sc == 1e6 else "%"
        t = np.arange(1, e.shape[0] + 1)
        ax = axes[0, j]
        ax.fill_between(t, np.percentile(e, 10, 1) * sc, np.percentile(e, 90, 1) * sc, alpha=0.3)
        ax.plot(t, np.median(e, 1) * sc, lw=2)
        ax.axhline(TOL.get(key, ("rel", 0.1))[1] * sc, ls="--", c="k", lw=1)
        ax.set(xlabel="step", ylabel=f"|{key} - goal| [{unit}]", yscale="log",
               title=f"{key}: median, p10-p90")
        ax = axes[1, j]
        for c, name in enumerate(env.controls):
            ax.scatter(gain[:, c], e[-1] * sc, s=6, alpha=0.5, label=f"gain {name}")
        ax.set(xlabel="hidden control gain", ylabel=f"final error [{unit}]", yscale="log")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "error_vs_step.png", dpi=130)
    plt.close(fig)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--n-episodes", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a diff_env setting for this eval (dotted keys), repeatable")
    ap.add_argument("--tag", default="", help="output subdir suffix: delta_eval_<tag>")
    args = ap.parse_args()
    rep = evaluate(args.logdir, args.policy, args.n_episodes, args.device, args.seed, args.out,
                   sets=args.set, tag=args.tag)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
