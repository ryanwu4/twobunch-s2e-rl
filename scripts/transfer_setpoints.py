"""Transfer final knob setpoints from a trained goal-conditioned controller, for Bmad validation.

For each requested target bunch separation, this runs the trained BPTT policy deterministically on
the (frozen) surrogate env, picks the best on-target rollout (argmax reward), and exports the 26
knob setpoints in PHYSICAL units -- denormalized in the surrogate's own union frame
(processed/*_norm.json knob_low/high), NOT the 8-knob bmad_env bounds -- so they can be fed straight
into FACET2-S2E setLattice for a Bmad cross-check. It also writes the surrogate-predicted metrics at
that setpoint (the numbers Bmad should reproduce) and a surrogate corner/LPS plot per target.

The physical setpoints are a dict {knob_name: value} keyed by the sweep param names (L1PhaseSet, ...,
Q5FFkG, ..., S1EL_xOffset, ...) -- i.e. setLattice(baseline + setpoints) applies them directly.

NOTE: the closed-loop Bmad *run* of a 26-knob policy is a separate step -- rl/bmad_env.py is still
8-knob (hardcoded original8 BOUNDS). This script does the setpoint TRANSFER + surrogate corner plots;
the Bmad tracking of these setpoints is done downstream (setLattice + trackBeam in the FACET2-S2E env).

Usage:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD/src python scripts/transfer_setpoints.py \
    --logdir logs/bptt_gc_combined \
    --flow-ckpt trained/twobunch_combined_ft/checkpoints/best-epoch=493-val_loss=0.5126.ckpt \
    --targets-um 150,250
Outputs: results/rl/bptt_gc_combined/setpoints/setpoints_goal<g>um.json, setpoints_summary.csv,
         corner_setpoint_goal<g>um.png, lps_setpoint_goal<g>um.png
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_s2e_rl.datagen.paths import repo_root
from twobunch_s2e_rl.rl.diff_env import TwoBunchFlowEnv
from twobunch_s2e_rl.rl.eval import _load_policy, _assert_obs_dim
from twobunch_s2e_rl.rl._train_utils import build_reward_spec, resolve_ckpt, _knob_box_bounds
from twobunch_s2e_rl.rl._eval_plots import render_sweep_plots

# achieved-metric key -> (report name, scale to display unit)
_METRICS = [
    ("bunch_spacing", "spacing_um", 1e6),
    ("T_drive", "T_drive", 1.0),
    ("T_witness", "T_witness", 1.0),
    ("drive_norm_emit_x", "drive_emit_x_um_rad", 1e6),
    ("drive_norm_emit_y", "drive_emit_y_um_rad", 1e6),
    ("witness_norm_emit_x", "witness_emit_x_um_rad", 1e6),
    ("witness_norm_emit_y", "witness_emit_y_um_rad", 1e6),
    ("transverse_offset", "transverse_offset_um", 1e6),
    ("angular_misalignment", "angular_misalignment_urad", 1e6),
    ("drive_sigma_z", "drive_sigma_z_um", 1e6),        # bunch length (matches validate_bmad std-of-z)
    ("witness_sigma_z", "witness_sigma_z_um", 1e6),
]


def _norm_frame(campaign_h5: str):
    """(knob_keys, low, high) physical union frame from the surrogate's norm.json."""
    nj = str(repo_root() / campaign_h5).replace(".h5", "_norm.json")
    with open(nj) as f:
        n = json.load(f)
    return nj, list(n["knob_keys"]), np.asarray(n["knob_low"], float), np.asarray(n["knob_high"], float)


@torch.no_grad()
def transfer(logdir, flow_ckpt, targets_um, policy="best", num_rollouts=256, seed=12345,
             n_cloud=3000, out="results/rl/bptt_gc_combined/setpoints", device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    logdir = Path(logdir)
    with open(logdir / "cfg.yaml") as f:
        cfg = yaml.safe_load(f)
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    if "spacing_goal" not in spec.obs_keys:
        raise SystemExit("policy is not goal-conditioned (no spacing_goal in obs) -- nothing to target")
    nj, knob_keys, lo, hi = _norm_frame(de["campaign_h5"])
    actor, obs_rms = None, None
    outdir = repo_root() / out
    outdir.mkdir(parents=True, exist_ok=True)

    box_lo, box_hi = _knob_box_bounds(de)   # MUST match training's knob box, else setpoints are off-frame
    rows, frames = [], []
    for g_um in targets_um:
        env = TwoBunchFlowEnv(
            num_rollouts, device=device, seed=seed,
            episode_length=int(de.get("episode_length", 64)), stochastic_init=True, no_grad=True,
            flow_ckpt=resolve_ckpt(flow_ckpt), reward_spec=spec,
            n_particles=int(de.get("n_particles", 512)), action_scale=float(de.get("action_scale", 0.05)),
            knob_box_lo=box_lo, knob_box_hi=box_hi,
            spacing_goal_lo=g_um * 1e-6, spacing_goal_hi=g_um * 1e-6)   # pin every env to this goal
        if actor is None:
            actor, obs_rms = _load_policy(str(logdir / f"{policy}_policy.pt"), device)
            _assert_obs_dim(env, actor)

        obs = env.reset()
        achieved, knobs_pre = None, None
        for _ in range(env.episode_length):
            o = obs_rms.normalize(obs) if obs_rms is not None else obs
            a = torch.tanh(actor(o, deterministic=True))
            obs, _, _, info = env.step(a)
            achieved = info["achieved"]
            knobs_pre = info["obs_before_reset"][:, :env.n_knob]

        reward = achieved["reward"].detach().cpu().numpy()
        rep = int(np.argmax(reward))                       # best composite on-target rollout
        knob_norm = knobs_pre[rep].detach().cpu().numpy()  # (n_knob,) in [0,1]
        phys = lo + knob_norm * (hi - lo)
        setpoints = {k: float(v) for k, v in zip(knob_keys, phys)}
        metrics = {name: float(achieved[key][rep].item() * sc) for key, name, sc in _METRICS}
        pin = float(((knob_norm < 1e-3) | (knob_norm > 1 - 1e-3)).mean())

        rec = {
            "target_um": float(g_um),
            "achieved_spacing_um": metrics["spacing_um"],
            "abs_err_um": abs(metrics["spacing_um"] - g_um),
            "surrogate_metrics": metrics,
            "knob_setpoints_physical": setpoints,
            "knob_setpoints_normalized": {k: float(v) for k, v in zip(knob_keys, knob_norm)},
            "knob_boundary_pin_frac": pin,
            "provenance": {"logdir": str(logdir), "policy": policy, "flow_ckpt": str(flow_ckpt),
                           "norm_json": nj, "campaign_h5": de["campaign_h5"],
                           "selected": "argmax_reward", "num_rollouts": int(num_rollouts)},
        }
        jpath = outdir / f"setpoints_goal{int(round(g_um))}um.json"
        with open(jpath, "w") as f:
            json.dump(rec, f, indent=2)
        rows.append(rec)

        slug = f"setpoint_goal{int(round(g_um))}um"
        kr = knobs_pre[rep:rep + 1]
        frames.append({
            "goal_um": g_um, "slug": slug,
            "title": f"goal {g_um:.0f} um (surrogate setpoint)  |  spacing "
                     f"{metrics['spacing_um']:.0f} um  |  T_w {metrics['T_witness']:.2f}",
            "drive": env._flow.sample_bunch(kr, 0, n_cloud)[0].detach().cpu().numpy(),
            "witness": env._flow.sample_bunch(kr, 1, n_cloud)[0].detach().cpu().numpy()})
        print(f"[goal {g_um:.0f} um] achieved {metrics['spacing_um']:.1f} um "
              f"(err {rec['abs_err_um']:.1f}), T_d/T_w {metrics['T_drive']:.2f}/{metrics['T_witness']:.2f}, "
              f"offset {metrics['transverse_offset_um']:.1f} um, pin {pin:.0%}  -> {jpath.name}")

    # combined CSV: one row per (target, knob) physical setpoint + a metrics block
    import csv
    with open(outdir / "setpoints_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target_um", "knob", "physical_value"])
        for rec in rows:
            for k, v in rec["knob_setpoints_physical"].items():
                w.writerow([rec["target_um"], k, v])
    render_sweep_plots(frames, outdir, "setpoint", fps=1.5)
    print(f"\nwrote {len(rows)} setpoint JSONs + setpoints_summary.csv + corner/LPS plots to {outdir}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logdir", default="logs/bptt_gc_combined")
    ap.add_argument("--flow-ckpt", required=True)
    ap.add_argument("--policy", default="best", choices=["best", "final"])
    ap.add_argument("--targets-um", default="150,250", help="comma-separated target spacings (um)")
    ap.add_argument("--num-rollouts", type=int, default=256)
    ap.add_argument("--n-cloud", type=int, default=3000)
    ap.add_argument("--out", default="results/rl/bptt_gc_combined/setpoints")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    targets = [float(x) for x in args.targets_um.split(",") if x.strip()]
    transfer(args.logdir, args.flow_ckpt, targets, policy=args.policy,
             num_rollouts=args.num_rollouts, seed=args.seed, n_cloud=args.n_cloud, out=args.out)


if __name__ == "__main__":
    main()
