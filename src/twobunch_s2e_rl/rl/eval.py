"""Evaluate a trained two-bunch MBRL policy by deterministic rollout on held-out random starts.

Loads the policy (best_policy.pt by default; SHAC saves a 5-tuple, BPTT a 2-tuple -- the obs
normalizer is found by type), rebuilds the env from the run's saved cfg.yaml, rolls full
episodes, and reports the ACHIEVED physical quantities at the end of tuning (from info["achieved"],
not the inverted reward): bunch spacing vs 200 um, per-bunch survival vs 90%, per-bunch emittance,
and the fraction of knobs pinned at a box edge. Eval defaults to rf_drift_std=0 (clean machine);
pass --rf-drift-std to probe robustness.

  PYTHONPATH=$PWD/src python -m twobunch_s2e_rl.rl.eval --logdir logs/shac \
      --flow-ckpt "trained/twobunch_flow_v4/checkpoints/best-*.ckpt"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .diff_env import N_KNOB, TwoBunchFlowEnv
from .diffrl.utils import RunningMeanStd
from ._train_utils import build_reward_spec, resolve_ckpt


def _load_policy(path: str, device: str):
    loaded = torch.load(path, map_location=device, weights_only=False)
    if not isinstance(loaded, (list, tuple)):
        loaded = [loaded]
    actor = loaded[0].to(device).eval()
    obs_rms = next((x for x in loaded if isinstance(x, RunningMeanStd)), None)
    if obs_rms is not None:
        obs_rms = obs_rms.to(device)
    return actor, obs_rms


def _pct(a, q):
    return float(np.percentile(a, q))


@torch.no_grad()
def evaluate(logdir, flow_ckpt, policy="best", num_rollouts=256, rf_drift_std=0.0,
             device="cuda", seed=12345):
    device = device if torch.cuda.is_available() else "cpu"
    logdir = Path(logdir)
    with open(logdir / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    env = TwoBunchFlowEnv(
        num_rollouts, device=device, seed=seed,
        episode_length=int(de.get("episode_length", 64)), stochastic_init=True, no_grad=True,
        flow_ckpt=resolve_ckpt(flow_ckpt), reward_spec=spec,
        n_particles=int(de.get("n_particles", 2048)),
        action_scale=float(de.get("action_scale", 0.05)),
        rf_drift_std=float(rf_drift_std),
    )
    actor, obs_rms = _load_policy(str(logdir / f"{policy}_policy.pt"), device)

    obs = env.reset()
    achieved, knobs_pre = None, None
    for _ in range(env.episode_length):
        o = obs_rms.normalize(obs) if obs_rms is not None else obs
        a = torch.tanh(actor(o, deterministic=True))
        obs, rew, done, info = env.step(a)
        achieved = info["achieved"]
        knobs_pre = info["obs_before_reset"][:, :N_KNOB]

    def arr(k):
        return achieved[k].detach().cpu().numpy()

    spacing_um = arr("bunch_spacing") * 1e6
    Td, Tw = arr("T_drive"), arr("T_witness")
    both90 = (Td >= 0.9) & (Tw >= 0.9)
    kn = knobs_pre.detach().cpu().numpy()
    pin = float(((kn < 1e-3) | (kn > 1 - 1e-3)).mean())

    report = {
        "logdir": str(logdir), "policy": policy, "num_rollouts": int(num_rollouts),
        "rf_drift_std": float(rf_drift_std),
        "spacing_um": {"median": _pct(spacing_um, 50), "p10": _pct(spacing_um, 10),
                       "p90": _pct(spacing_um, 90), "abs_err_to_200_median": float(np.median(np.abs(spacing_um - 200.0)))},
        "survival": {"T_drive_median": float(np.median(Td)), "T_witness_median": float(np.median(Tw)),
                     "frac_both_over_90": float(both90.mean())},
        "emittance_um_rad_median": {k: float(np.median(arr(k) * 1e6)) for k in
                                    ("drive_norm_emit_x", "drive_norm_emit_y",
                                     "witness_norm_emit_x", "witness_norm_emit_y")},
        "knob_boundary_pin_frac": pin,
        "reward_mean": float(np.mean(arr("reward"))),
    }
    print(json.dumps(report, indent=2))
    with open(logdir / f"eval_{policy}.json", "w") as f:
        json.dump(report, f, indent=2)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--num-rollouts", type=int, default=256)
    ap.add_argument("--rf-drift-std", type=float, default=0.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    evaluate(args.logdir, args.flow_ckpt, policy=args.policy, num_rollouts=args.num_rollouts,
             rf_drift_std=args.rf_drift_std, device=args.device, seed=args.seed)


if __name__ == "__main__":
    main()
