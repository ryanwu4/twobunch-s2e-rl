"""Evaluate a trained two-bunch MBRL policy by deterministic rollout on held-out random starts.

Loads the policy (best_policy.pt by default; SHAC saves a 5-tuple, BPTT a 2-tuple -- the obs
normalizer is found by type), rebuilds the env from the run's saved cfg.yaml, rolls full
episodes, and reports the ACHIEVED physical quantities at the end of tuning (from info["achieved"],
not the inverted reward): bunch spacing vs 200 um, per-bunch survival vs 90%, per-bunch emittance,
and the fraction of knobs pinned at a box edge. Eval defaults to rf_drift_std=0 (clean machine);
pass --rf-drift-std to probe robustness.

With --goal-sweep (a goal-conditioned policy), it also renders -- from surrogate-sampled clouds at
each goal's representative operating point -- a 6D corner plot (both bunches overlaid) and a combined
z-pz LPS, stitched across the sweep into corner_surr_sweep.gif / lps_surr_sweep.gif under
results/rl/ (mirrors eval_bmad's plots; --no-plot to skip).

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.rl.eval --logdir logs/shac \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

from .diff_env import TwoBunchFlowEnv
from .diffrl.utils import RunningMeanStd
from ._eval_plots import render_sweep_plots
from ._train_utils import build_reward_spec, resolve_ckpt, _knob_box_bounds
from ..datagen.paths import rl_dir


def _load_policy(path: str, device: str):
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, (list, tuple)):
        loaded = [loaded]
    actor = loaded[0].to(device).eval()
    obs_rms = next((x for x in loaded if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


def _assert_obs_dim(env, actor) -> None:
    """Guard against loading a goal-conditioned checkpoint into a non-goal env (or vice-versa):
    the actor's first Linear sets the expected obs dim. A mismatch otherwise dies in a CUDA matmul."""
    nin = next((m.in_features for m in actor.modules() if isinstance(m, nn.Linear)), None)
    if nin is not None and nin != env.num_obs:
        raise ValueError(
            f"obs-dim mismatch: env.num_obs={env.num_obs} but the loaded actor expects {nin} inputs. "
            f"A goal-conditioned policy needs a goal-conditioned env (spacing_goal_{{lo,hi}}_m), and a "
            f"fixed-target policy needs a non-goal env.")


def _pct(a, q):
    return float(np.percentile(a, q))


@torch.no_grad()
def evaluate(logdir, flow_ckpt, policy="best", num_rollouts=256, rf_drift_std=0.0,
             device="cuda", seed=12345, goal_um=None, make_frame=False, n_cloud=3000):
    """If `goal_um` is set, every env is pinned to that target spacing (a degenerate [g,g] range)
    and the error is reported vs that goal; else a goal-conditioned policy keeps its training range
    (random per-episode goal) and a fixed-target policy reports error vs 200 um as before."""
    device = device if torch.cuda.is_available() else "cpu"
    logdir = Path(logdir)
    with open(logdir / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    goal_conditioned = "spacing_goal" in spec.obs_keys
    if goal_um is not None:
        env_goal = dict(spacing_goal_lo=goal_um * 1e-6, spacing_goal_hi=goal_um * 1e-6)
    elif goal_conditioned:                       # keep the training range (random goal per episode)
        env_goal = dict(spacing_goal_lo=de.get("spacing_goal_lo_m"),
                        spacing_goal_hi=de.get("spacing_goal_hi_m"))
    else:
        env_goal = {}
    box_lo, box_hi = _knob_box_bounds(de)   # MUST match training: same knob box or the policy is off-frame
    env = TwoBunchFlowEnv(
        num_rollouts, device=device, seed=seed,
        episode_length=int(de.get("episode_length", 64)), stochastic_init=True, no_grad=True,
        flow_ckpt=resolve_ckpt(flow_ckpt), reward_spec=spec,
        n_particles=int(de.get("n_particles", 2048)),
        action_scale=float(de.get("action_scale", 0.05)),
        rf_drift_std=float(rf_drift_std),
        knob_box_lo=box_lo, knob_box_hi=box_hi,
        **env_goal,
    )
    actor, obs_rms = _load_policy(str(logdir / f"{policy}_policy.pt"), device)
    _assert_obs_dim(env, actor)

    obs = env.reset()
    achieved, knobs_pre = None, None
    for _ in range(env.episode_length):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        a = torch.tanh(actor(o, deterministic=True))
        obs, rew, done, info = env.step(a)
        achieved = info["achieved"]
        knobs_pre = info["obs_before_reset"][:, :env.n_knob]

    def arr(k):
        return achieved[k].detach().cpu().numpy()

    spacing_um = arr("bunch_spacing") * 1e6
    Td, Tw = arr("T_drive"), arr("T_witness")
    both90 = (Td >= 0.9) & (Tw >= 0.9)
    kn = knobs_pre.detach().cpu().numpy()
    pin = float(((kn < 1e-3) | (kn > 1 - 1e-3)).mean())

    # sweep frame: surrogate clouds at the REPRESENTATIVE operating point (env whose achieved spacing
    # is closest to the median) -- a single clean corner/LPS per goal, mirroring the Bmad eval.
    frame = None
    if make_frame:
        rep = int(np.argmin(np.abs(spacing_um - np.median(spacing_um))))
        kr = knobs_pre[rep:rep + 1]
        gstr = f"goal {goal_um:.0f} um" if goal_um is not None else logdir.name
        slug = (f"surr_goal{int(round(goal_um))}um" if goal_um is not None
                else "surr")
        frame = {"goal_um": goal_um, "slug": slug,
                 "title": f"{gstr} (surrogate)  |  spacing {spacing_um[rep]:.0f} um  |  T_w {Tw[rep]:.2f}",
                 "drive": env._flow.sample_bunch(kr, 0, n_cloud)[0].detach().cpu().numpy(),
                 "witness": env._flow.sample_bunch(kr, 1, n_cloud)[0].detach().cpu().numpy()}

    # error vs the goal when pinned, else vs the legacy 200 um set point (kept for old non-goal runs)
    ref_um = float(goal_um) if goal_um is not None else 200.0
    err_key = "abs_err_to_goal_median" if goal_um is not None else "abs_err_to_200_median"
    spacing_report = {"median": _pct(spacing_um, 50), "p10": _pct(spacing_um, 10),
                      "p90": _pct(spacing_um, 90), err_key: float(np.median(np.abs(spacing_um - ref_um)))}
    if goal_um is not None:
        spacing_report["goal_um"] = float(goal_um)

    report = {
        "logdir": str(logdir), "policy": policy, "num_rollouts": int(num_rollouts),
        "rf_drift_std": float(rf_drift_std), "goal_um": (float(goal_um) if goal_um is not None else None),
        "spacing_um": spacing_report,
        "survival": {"T_drive_median": float(np.median(Td)), "T_witness_median": float(np.median(Tw)),
                     "frac_both_over_90": float(both90.mean())},
        "emittance_um_rad_median": {k: float(np.median(arr(k) * 1e6)) for k in
                                    ("drive_norm_emit_x", "drive_norm_emit_y",
                                     "witness_norm_emit_x", "witness_norm_emit_y")},
        "knob_median": [float(np.median(kn[:, j])) for j in range(env.n_knob)],   # diagnostic for sweeps
        "knob_boundary_pin_frac": pin,
        "reward_mean": float(np.mean(arr("reward"))),
    }
    print(json.dumps(report, indent=2))
    suffix = f"_goal{int(round(goal_um))}um" if goal_um is not None else ""
    with open(logdir / f"eval_{policy}{suffix}.json", "w") as f:
        json.dump(report, f, indent=2)
    return report, frame


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--num-rollouts", type=int, default=256)
    ap.add_argument("--rf-drift-std", type=float, default=0.0)
    ap.add_argument("--goal-um", type=float, default=None,
                    help="pin every env to this target spacing (um) and report error vs it")
    ap.add_argument("--goal-sweep", nargs="?", const="100,150,200,250,300", default=None,
                    help="evaluate goal-tracking at a comma-separated list of target spacings (um); "
                         "bare flag uses 100,150,200,250,300")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no-plot", action="store_true",
                    help="skip the per-goal corner/LPS PNGs and the across-sweep GIFs")
    ap.add_argument("--gif-fps", type=float, default=2.0,
                    help="frames/sec for the corner + LPS sweep GIFs (one frame per goal)")
    args = ap.parse_args()

    common = dict(policy=args.policy, num_rollouts=args.num_rollouts, rf_drift_std=args.rf_drift_std,
                  device=args.device, seed=args.seed, make_frame=not args.no_plot)
    artdir = rl_dir(Path(args.logdir).name) / "goal_sweep"
    name = "surr"
    if args.goal_sweep is not None:
        goals = [float(x) for x in args.goal_sweep.split(",") if x.strip()]
        results = [evaluate(args.logdir, args.flow_ckpt, goal_um=g, **common) for g in goals]
        rows = [r for r, _ in results]
        frames = [f for _, f in results if f is not None]
        # achieved-vs-goal tracking curve is the gating metric; knob_median trends are diagnostic only
        # (the optimum is non-unique -> different knobs can give the same spacing).
        print(f"\n{'goal_um':>8s} {'achieved':>9s} {'abs_err':>8s} {'T_d':>5s} {'T_w':>5s} {'both>90':>8s}")
        sweep = []
        for g, r in zip(goals, rows):
            s, sv = r["spacing_um"], r["survival"]
            print(f"{g:8.0f} {s['median']:9.1f} {s['abs_err_to_goal_median']:8.1f} "
                  f"{sv['T_drive_median']:5.2f} {sv['T_witness_median']:5.2f} {sv['frac_both_over_90']:8.2f}")
            sweep.append({"goal_um": g, "achieved_um": s["median"],
                          "abs_err_to_goal_um": s["abs_err_to_goal_median"],
                          "survival": sv, "knob_median": r["knob_median"]})
        out = Path(args.logdir) / f"eval_{args.policy}_goal_sweep.json"
        with open(out, "w") as f:
            json.dump({"logdir": args.logdir, "policy": args.policy, "sweep": sweep}, f, indent=2)
        print(f"\nWrote {out}")
        if frames:
            artdir.mkdir(parents=True, exist_ok=True)
            render_sweep_plots(frames, artdir, name, fps=args.gif_fps)
    else:
        _, frame = evaluate(args.logdir, args.flow_ckpt, goal_um=args.goal_um, **common)
        if frame is not None:
            artdir.mkdir(parents=True, exist_ok=True)
            render_sweep_plots([frame], artdir, name, fps=args.gif_fps)


if __name__ == "__main__":
    main()
