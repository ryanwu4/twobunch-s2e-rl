"""Continuous-operation eval of a delta controller: long episodes with setpoint changes (the env's `goal_change`, or
one set with --set v4.goal_change=[lo,hi]), metrics per setpoint segment on the TRUE beam vs the TRUE goal:

  settled error      the median |spacing - goal| over the last `tail` steps before each setpoint change (or episode end)
  time within 10 um  fraction of steps within tolerance, counting only steps >= `settle` after a change
  steps to reach     first step after a change with the error within tolerance (segments that never reach: inf)
  sigma_z            the same settled metric, relative error

  python -m twobunch_s2e_rl.rl.eval_continuous --logdir <run> [--set v4.drift_per_step.L1PhaseSet=0.07 ...] --out <dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from .eval_delta import _apply_overrides, _deep_merge

TOL_SP, TOL_SZ = 10e-6, 0.10


def run(logdir, out, sets=(), k=256, steps=128, policy="final", device="cuda:0", seed=97531, settle=8, tail=4,
        de_yaml=None, flow=None):
    from ..artifacts import policy_flow_ckpt
    from ._train_utils import build_env_fn
    from .rollout import load_policy, rollout
    cfg = yaml.safe_load((Path(logdir) / "cfg.yaml").read_text())
    if de_yaml:
        cfg["params"]["diff_env"] = _deep_merge(cfg["params"]["diff_env"], yaml.safe_load(Path(de_yaml).read_text()))
    de = _apply_overrides(cfg["params"]["diff_env"], sets)
    if not (de.get("v4") or {}).get("goal_change"):
        raise SystemExit("needs v4.goal_change (set it with --set v4.goal_change=[16,48])")
    env = build_env_fn(cfg, flow or policy_flow_ckpt(logdir))(num_envs=k, device=device, seed=seed, episode_length=10 ** 6,
                                                       no_grad=True)
    actor, rms = load_policy(str(Path(logdir) / f"{policy}_policy.pt"), device)
    env.reset()
    tr = rollout(env, actor, rms, n_steps=steps)
    g_sp = next(g for key, g, _ in env._goal_terms if key == "bunch_spacing")
    g_sz = next(g for key, g, _ in env._goal_terms if key == "drive_sigma_z")
    true = lambda key: (tr[f"{key}_cloud"] if f"{key}_cloud" in tr else tr[f"{key}_true"]).cpu().numpy()  # noqa: E731
    sp, sz = true("bunch_spacing"), true("drive_sigma_z")
    gsp, gsz = tr[f"{g_sp}_cloud"].cpu().numpy(), tr[f"{g_sz}_cloud"].cpu().numpy()
    e_sp, e_sz = np.abs(sp - gsp), np.abs(sz - gsz) / np.abs(gsz)
    # segment boundaries: the goal (cloud) value changes between consecutive recorded steps
    chg = np.vstack([np.ones((1, k), bool), np.abs(np.diff(gsp, axis=0)) > 1e-9])
    # spacing-only setpoints (env spacing_only_frac): sigma_z unconstrained there -> excluded from the sigma_z metrics
    free = tr["goal_free"].cpu().numpy() > 0.5 if "goal_free" in tr else np.zeros_like(chg)
    if "goal_free" in tr:
        chg = chg | np.vstack([np.zeros((1, k), bool), np.diff(free.astype(int), axis=0) != 0])
    settled_sp, settled_sz, reach, inside, seg_free, seg_goal = [], [], [], [], [], []
    for b in range(k):
        starts = list(np.flatnonzero(chg[:, b])) + [steps]
        for s0, s1 in zip(starts[:-1], starts[1:]):
            if s1 - s0 < settle + tail:
                continue
            settled_sp.append(np.median(e_sp[s1 - tail:s1, b]))
            settled_sz.append(np.median(e_sz[s1 - tail:s1, b]))
            ok = np.flatnonzero(e_sp[s0:s1, b] <= TOL_SP)
            reach.append(float(ok[0]) if len(ok) else np.inf)
            inside.append((e_sp[s0 + settle:s1, b] <= TOL_SP).mean())
            seg_free.append(bool(free[s1 - 1, b]))
            seg_goal.append(float(gsp[s1 - 1, b]))
    settled_sp, settled_sz, reach, seg_free = map(np.asarray, (settled_sp, settled_sz, reach, seg_free))
    sz_seg = settled_sz[~seg_free] if seg_free.any() else settled_sz
    summ = {"logdir": str(logdir), "sets": list(sets), "de_yaml": de_yaml, "flow": flow, "k": k, "steps": steps, "n_segments": int(len(settled_sp)),
            "settled_spacing_median_um": float(np.median(settled_sp) * 1e6),
            "settled_spacing_p90_um": float(np.percentile(settled_sp, 90) * 1e6),
            "settled_within_10um": float((settled_sp <= TOL_SP).mean()),
            "time_within_10um_after_settle": float(np.mean(inside)),
            "steps_to_reach_median": float(np.median(reach)) if np.isfinite(np.median(reach)) else None,
            "frac_segments_reached": float(np.isfinite(reach).mean()),
            "settled_sigma_z_rel_median": float(np.median(sz_seg)),
            "settled_sigma_z_within_10pct": float((sz_seg <= TOL_SZ).mean())}
    if seg_free.any():
        for tag, sel in (("free", seg_free), ("constrained", ~seg_free)):
            summ[f"settled_spacing_median_um_{tag}"] = float(np.median(settled_sp[sel]) * 1e6)
            summ[f"settled_within_10um_{tag}"] = float((settled_sp[sel] <= TOL_SP).mean())
            summ[f"n_segments_{tag}"] = int(sel.sum())
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    np.savez(out / "errors.npz", e_sp=e_sp, e_sz=e_sz, chg=chg, seg_settled_sp=settled_sp, seg_settled_sz=settled_sz,
             seg_goal_sp=np.asarray(seg_goal), seg_reach=reach, seg_free=seg_free)
    print(f"[continuous] segments {summ['n_segments']}: settled spacing {summ['settled_spacing_median_um']:.1f} um "
          f"(p90 {summ['settled_spacing_p90_um']:.1f}), settled within 10 um {summ['settled_within_10um']:.2f}, "
          f"time within 10 um {summ['time_within_10um_after_settle']:.2f}, reached {summ['frac_segments_reached']:.2f} "
          f"(median {summ['steps_to_reach_median']} steps); sz {summ['settled_sigma_z_rel_median']:.3f} "
          f"/ {summ['settled_sigma_z_within_10pct']:.2f}", flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--de-yaml", default=None, help="diff_env fragment deep-merged into the policy's config")
    ap.add_argument("--flow", default=None, help="evaluate on this flow checkpoint instead of the policy's own")
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--policy", default="final")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.logdir, a.out, sets=a.set, k=a.k, steps=a.steps, policy=a.policy, device=a.device, de_yaml=a.de_yaml, flow=a.flow)


if __name__ == "__main__":
    main()
