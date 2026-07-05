"""Shared wiring for the SHAC/BPTT entry points (env_fn, reward spec, cfg overrides, CSV hook).

Mirrors photoinjector-rl-clean's flow_surrogate/train_{shac,bptt}.py, minus the moving-shape /
curriculum machinery (out of scope). The reward spec is built once from the campaign h5 (cached
to json) and bound into the env_fn so SHAC/BPTT can construct the env with their standard kwargs.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from functools import partial

from .diff_env import TwoBunchFlowEnv, rf_drift_std_vector
from .reward import reward_spec_from_campaign


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cfg", required=True, type=str)
    p.add_argument("--flow-ckpt", required=True, type=str, help="TwoBunchFlow checkpoint (glob ok).")
    p.add_argument("--campaign-h5", default=None, type=str, help="override cfg diff_env.campaign_h5")
    p.add_argument("--logdir", default=None, type=str)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--max-epochs", default=None, type=int)
    p.add_argument("--num-actors", default=None, type=int)
    p.add_argument("--steps-num", default=None, type=int)
    p.add_argument("--n-particles", default=None, type=int)
    p.add_argument("--action-scale", default=None, type=float)
    p.add_argument("--rf-drift-std", default=None, type=float)
    p.add_argument("--spacing-goal-lo-m", default=None, type=float,
                   help="enable goal-conditioned RL: lower bound of the per-episode target spacing (m)")
    p.add_argument("--spacing-goal-hi-m", default=None, type=float,
                   help="upper bound of the per-episode target spacing (m); both bounds enable goal mode")
    p.add_argument("--checkpoint", default=None, type=str, help="policy .pt for --play")
    p.add_argument("--play", action="store_true")


def override(cfg: dict, args: argparse.Namespace) -> dict:
    g, c, de = cfg["params"]["general"], cfg["params"]["config"], cfg["params"]["diff_env"]
    if args.seed is not None:
        g["seed"] = args.seed
    if args.device is not None:
        g["device"] = args.device
    if args.logdir is not None:
        g["logdir"] = args.logdir
    if args.max_epochs is not None:
        c["max_epochs"] = args.max_epochs
    if args.num_actors is not None:
        c["num_actors"] = args.num_actors
    if args.steps_num is not None:
        c["steps_num"] = args.steps_num
    if args.n_particles is not None:
        de["n_particles"] = args.n_particles
    if args.action_scale is not None:
        de["action_scale"] = args.action_scale
    if args.rf_drift_std is not None:
        de["rf_drift_std"] = args.rf_drift_std
        de["rf_drift"] = {"enabled": False}   # explicit scalar override wins over a config block
    if args.spacing_goal_lo_m is not None:
        de["spacing_goal_lo_m"] = args.spacing_goal_lo_m
    if args.spacing_goal_hi_m is not None:
        de["spacing_goal_hi_m"] = args.spacing_goal_hi_m
    if args.campaign_h5 is not None:
        de["campaign_h5"] = args.campaign_h5
    if args.play:
        g["train"] = False
        g["checkpoint"] = args.checkpoint
    return cfg


def resolve_ckpt(path: str) -> str:
    return sorted(glob.glob(path))[-1] if "*" in path else path


def _spacing_goal_range(de: dict):
    """(lo, hi) target-spacing range for goal-conditioned RL: from the objectives block's
    bunch_spacing `goal` if present, else the flat spacing_goal_{lo,hi}_m keys (back-compat)."""
    for e in de.get("objectives") or []:
        if e.get("key") == "bunch_spacing" and e.get("goal") is not None:
            lo, hi = e["goal"]
            return float(lo), float(hi)
    return de.get("spacing_goal_lo_m"), de.get("spacing_goal_hi_m")


def build_reward_spec(de: dict):
    cache = de.get("reward_norms_json")
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    # config-driven objectives: a `objectives` list fully specifies the reward terms + obs keys
    # (goal-conditioning is expressed by a `goal:[lo,hi]` on the bunch_spacing target entry).
    objectives = de.get("objectives")
    if objectives is not None:
        return reward_spec_from_campaign(
            de["campaign_h5"], cache_json=cache, objectives=objectives,
            floor_pct=de.get("floor_pct", 10.0),
            drive_floor_pct=de.get("drive_floor_pct"),
            witness_floor_pct=de.get("witness_floor_pct"),
            surv_T_min=de.get("surv_T_min", 0.9),
            surv_margin=de.get("surv_margin", 0.05),
            emit_below_weight=de.get("emit_below_weight", 1.0),
            emit_gate_band=de.get("emit_gate_band", 0.2),
            boundary_margin=de.get("boundary_margin", 0.05))
    # legacy flat-weights path: a spacing_goal_{lo,hi}_m range enables a per-env target-spacing obs +
    # the reward's spacing term reading that per-env goal. Norms are goal-independent (cache reused).
    goal_conditioned = de.get("spacing_goal_lo_m") is not None and de.get("spacing_goal_hi_m") is not None
    return reward_spec_from_campaign(
        de["campaign_h5"],
        cache_json=cache,
        spacing_goal_key="spacing_goal" if goal_conditioned else None,
        floor_pct=de.get("floor_pct", 10.0),
        drive_floor_pct=de.get("drive_floor_pct"),
        witness_floor_pct=de.get("witness_floor_pct"),
        spacing_target_m=de.get("spacing_target_m", 2.0e-4),
        surv_T_min=de.get("surv_T_min", 0.9),
        surv_margin=de.get("surv_margin", 0.05),
        emit_mode=de.get("emit_mode", "minimize_floor"),
        emit_below_weight=de.get("emit_below_weight", 1.0),
        emit_gate_band=de.get("emit_gate_band", 0.2),
        w_spacing=de.get("w_spacing", 1.0),
        w_emit=de.get("w_emit", 1.0),
        w_emit_witness=de.get("w_emit_witness", None),
        w_surv=de.get("w_surv", 1.0),
        w_ood=de.get("w_ood", 0.5),
        boundary_margin=de.get("boundary_margin", 0.05),
        w_collinearity=de.get("w_collinearity", 0.0),
    )


def build_env_fn(cfg: dict, flow_ckpt: str):
    """Bind the flow + reward spec into TwoBunchFlowEnv so the trainer can call it with the
    standard (num_envs, device, seed, episode_length, stochastic_init, ...) kwargs."""
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    goal_lo, goal_hi = _spacing_goal_range(de)
    # latent RF drift: an `rf_drift` block (physical units) builds a per-knob normalized-std
    # vector; otherwise fall back to the scalar `rf_drift_std` (CLI-overridable, 0 = off).
    rf = de.get("rf_drift", {})
    if rf.get("enabled", False):
        drift = rf_drift_std_vector(rf.get("phase_drift_deg", 0.5), rf.get("amp_drift_frac", 0.005))
    else:
        drift = de.get("rf_drift_std", 0.0)
    return partial(
        TwoBunchFlowEnv,
        flow_ckpt=resolve_ckpt(flow_ckpt),
        reward_spec=spec,
        n_particles=de.get("n_particles", 512),
        action_scale=de.get("action_scale", 0.05),
        rf_drift_std=drift,
        common_random_numbers=de.get("common_random_numbers", False),
        spacing_goal_lo=goal_lo,
        spacing_goal_hi=goal_hi,
    )


def attach_csv_hook(algo, logdir: str) -> None:
    os.makedirs(logdir, exist_ok=True)
    f = open(os.path.join(logdir, "learning_curve.csv"), "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "mean_episode_loss", "wall_time"])

    def hook(step: int, mean_loss: float, wall: float) -> None:
        writer.writerow([step, mean_loss, wall])
        f.flush()

    algo.step_metrics_hook = hook
