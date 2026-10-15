"""Shared wiring for the SHAC/BPTT entry points (env_fn, reward spec, cfg overrides, CSV hook).

Mirrors photoinjector-rl-clean's flow_surrogate/train_{shac,bptt}.py, minus the moving-shape /
curriculum machinery (out of scope). The reward spec is built once from the campaign h5 (cached
to json) and bound into the env_fn so SHAC/BPTT can construct the env with their standard kwargs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import warnings
from functools import partial
from pathlib import Path

import numpy as np

from ..artifacts import flow_knob_space, resolve_ckpt  # resolve_ckpt re-exported for eval/scripts
from ..surrogate.model import TwoBunchFlow
from .diff_env import TwoBunchFlowEnv, rf_drift_std_vector
from .plants import flow_plant, xtcav_screen_plant
from .reward import reward_spec_from_campaign
from ..datagen.paths import repo_root
from ..datagen.sweep_params import resolve_sweep_set


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--cfg", required=True, type=str)
    p.add_argument("--flow-ckpt", default=None, type=str,
                   help="TwoBunchFlow checkpoint (glob ok); default: cfg diff_env.flow_ckpt")
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




def goal_specs(de: dict, spec) -> dict:
    """{goal_key: (lo, hi, space)} the env samples per episode: from the objectives block's
    `goal:` entries (any target), else the flat spacing_goal_{lo,hi}_m keys (original form).
    The single place eval / eval_bmad / transfer_setpoints read goal ranges from."""
    specs = dict(spec.goal_specs())
    lo, hi = de.get("spacing_goal_lo_m"), de.get("spacing_goal_hi_m")
    if not specs and lo is not None and hi is not None:
        specs = {"spacing_goal": (float(lo), float(hi), "linear")}
    return specs


def pin_goal(specs: dict, goal_key: str, value: float) -> dict:
    """Copy of `specs` with one goal pinned to a fixed value (degenerate [v, v] range)."""
    if goal_key not in specs:
        raise ValueError(f"policy has no {goal_key!r} goal (goals: {list(specs)})")
    out = dict(specs)
    out[goal_key] = (float(value), float(value), specs[goal_key][2])
    return out


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
            observe_keys=tuple(de.get("observe", ())),
            overwrite_cache=bool(de.get("reward_norms_overwrite", False)),
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
        overwrite_cache=bool(de.get("reward_norms_overwrite", False)),
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


def _knob_box_bounds(de: dict):
    """Resolve a `knob_box: <sweep_set>` config into per-knob [lo,hi] in the surrogate's normalized
    frame: each knob is clamped to that set's bounds (e.g. `tightbox` -> transverse knobs confined
    near golden, longitudinal unchanged), keeping the policy on-distribution. Returns (None, None)
    if unset. The frame is the surrogate's own knob_low/high (from its `_norm.json`)."""
    name = de.get("knob_box")
    if not name:
        return None, None
    nj = str(repo_root() / de["campaign_h5"]).replace(".h5", "_norm.json")
    with open(nj) as f:
        norm = json.load(f)
    keys = norm["knob_keys"]
    nlo, nhi = np.array(norm["knob_low"], float), np.array(norm["knob_high"], float)
    span = np.where(nhi > nlo, nhi - nlo, 1.0)
    tk, tlo, thi, _ = resolve_sweep_set(name)
    klo, khi = np.zeros(len(keys)), np.ones(len(keys))
    for i, k in enumerate(keys):
        if k in tk:
            j = tk.index(k)
            klo[i] = (tlo[j] - nlo[i]) / span[i]
            khi[i] = (thi[j] - nlo[i]) / span[i]
    return np.clip(klo, 0, 1).tolist(), np.clip(khi, 0, 1).tolist()


def build_env_fn(cfg: dict, flow_ckpt: str):
    """Bind the flow + reward spec into TwoBunchFlowEnv (or DeltaControlEnv when
    diff_env.name says so) so the trainer can call it with the standard
    (num_envs, device, seed, episode_length, stochastic_init, ...) kwargs."""
    de = cfg["params"]["diff_env"]
    if de.get("name") == "DeltaControlEnv":
        return build_delta_env_fn(cfg, flow_ckpt)
    spec = build_reward_spec(de)
    box_lo, box_hi = _knob_box_bounds(de)
    # latent RF drift: an `rf_drift` block (physical units) builds a per-knob normalized-std
    # vector; otherwise fall back to the scalar `rf_drift_std` (CLI-overridable, 0 = off).
    rf = de.get("rf_drift", {})
    if rf.get("enabled", False):
        # in the flow's OWN frame (train-time bounds), not today's knob tables
        frame = flow_knob_space(TwoBunchFlow.load_from_checkpoint(resolve_ckpt(flow_ckpt),
                                                                  map_location="cpu"))
        drift = rf_drift_std_vector(rf.get("phase_drift_deg", 0.5), rf.get("amp_drift_frac", 0.005),
                                    knob_space=frame)
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
        goals=goal_specs(de, spec),
        knob_box_lo=box_lo,
        knob_box_hi=box_hi,
    )


def flow_norm_json(ckpt, campaign_h5) -> str:
    """The processed norm the flow was TRAINED on (train.py writes <out>/norm.json next to
    <out>/checkpoints/): full charges etc. must come from it, not from the reward campaign. Older
    checkpoints have none -> the reward campaign's norm (they were trained on it)."""
    here = Path(ckpt).resolve().parent
    for own in (here / "norm.json", here.parent / "norm.json"):     # <out>/checkpoints/x.ckpt or <dir>/x.ckpt
        if own.exists():
            return str(own)
    return str(campaign_h5).replace(".h5", "_norm.json")


def build_delta_env_fn(cfg: dict, flow_ckpt: str):
    """DeltaControlEnv over a plant chosen by diff_env.plant:
      - "flow": the flow's own observables at its treaty point (cloud-mode ground truth)
      - "xtcav_screen": flow at XTCAVF + Cheetah TCAV +/-V to DTOTR, particle-space screen
        features (screen mode; ground-truth twins kept as <key>_cloud)
    The plant's knob frame and baseline come from the flow (stored frame) + the knob table of
    its sweep set (baseline values only)."""
    from .delta_env import DeltaControlEnv
    de = cfg["params"]["diff_env"]
    spec = build_reward_spec(de)
    ckpt = resolve_ckpt(flow_ckpt)
    plant = de.get("plant", "flow")
    if plant not in ("flow", "xtcav_screen"):
        raise ValueError(f"unknown delta-env plant {plant!r}")
    frame = flow_knob_space(TwoBunchFlow.load_from_checkpoint(ckpt, map_location="cpu"))
    baseline = dict(resolve_sweep_set(de["baseline_set"])[3])
    baseline.update(de.get("baseline_override", {}))
    n_particles = int(de.get("n_particles", 512))

    def make(num_envs, device="cuda:0", seed=0, episode_length=32, no_grad=False, **_):
        flow = TwoBunchFlow.load_from_checkpoint(ckpt, map_location=device).to(device).eval()
        for p in flow.parameters():
            p.requires_grad_(False)
        ms = de.get("measurement", {})
        if plant == "flow":
            observe_fn = flow_plant(flow, n_particles)
        else:
            sc = de.get("screen", {})
            if "split" not in sc or "threshold" not in sc:
                warnings.warn("screen config without explicit `split` / `threshold`: using the v1 "
                              "defaults (zlobe, midpoint = TRUE-centroid threshold, a truth leak). "
                              "Set both explicitly in new configs.", stacklevel=2)
            observe_fn = xtcav_screen_plant(
                flow, n_particles, spec_path=str(repo_root() / sc.get("spec", "configs/segment/segment_baseline.json")),
                norm_json=flow_norm_json(ckpt, repo_root() / de["campaign_h5"]), device=device,
                tdc_voltage_V=float(sc.get("tdc_voltage_V", 22e6)), split=sc.get("split", "zlobe"),
                resolution_m=float(sc.get("extra_resolution_m", 0.0)),
                threshold=sc.get("threshold", "midpoint"), row_correct=bool(sc.get("row_correct", False)),
                quad_jitter=sc.get("quad_jitter"))
        v4 = de.get("v4")
        cls = DeltaControlEnv
        extra = {}
        if v4:
            from .delta_env_v4 import DeltaControlEnvV4
            cls = DeltaControlEnvV4
            extra = dict(actuators=v4.get("actuators", {}), drift_per_step=v4.get("drift_per_step"),
                         step_noise_floor=ms.get("step_noise_floor"),
                         outlier_prob=float(ms.get("outlier_prob", 0.0)),
                         outlier_scale=float(ms.get("outlier_scale", 5.0)),
                         reward_on=v4.get("reward_on", "true"), action_cost=float(v4.get("action_cost", 0.0)),
                         offcontext_goal_frac=float(v4.get("offcontext_goal_frac", 0.0)),
                         history=int(v4.get("history", 1)))
        return cls(
            num_envs, **extra, device=device, seed=seed, episode_length=episode_length, no_grad=no_grad,
            observe_fn=observe_fn, fallback_keys=tuple(ms.get("fallback_keys", ())),
            feature_offset_sigma=ms.get("feature_offset_sigma"),
            knob_space=frame, baseline=baseline, controls=de["controls"], reward_spec=spec,
            action_step=de["action_step"], start_box={k: tuple(v) for k, v in de["start_box"].items()},
            context_sigma=de.get("context_sigma"), context_dist=de.get("context_dist"),
            offset_sigma=de.get("offset_sigma"),
            gain_halfwidth=float(de.get("gain_halfwidth", 0.2)),
            scale_error_keys=tuple(ms.get("scale_error_keys", ())),
            scale_error_halfwidth=float(ms.get("scale_error_halfwidth", 0.0)),
            shot_noise=ms.get("shot_noise"), shots_per_step=int(ms.get("shots_per_step", 50)))
    return make


def attach_csv_hook(algo, logdir: str) -> None:
    os.makedirs(logdir, exist_ok=True)
    f = open(os.path.join(logdir, "learning_curve.csv"), "w", newline="")
    writer = csv.writer(f)
    writer.writerow(["step", "mean_episode_loss", "wall_time"])

    def hook(step: int, mean_loss: float, wall: float) -> None:
        writer.writerow([step, mean_loss, wall])
        f.flush()

    algo.step_metrics_hook = hook
