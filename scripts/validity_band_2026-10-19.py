"""Validity-band checks (2026-10-19): does the surrogate's particle-count-dependent screen validity near full compression
matter for the controller?

  1. training exposure: in the training env (zs_screen_noblen_jit / zs_screen_bc11loop_jit configs) the raw goal
     candidates (start + U(+-goal_radius)), their noise-free screen validity (N = 1024, as goals are screened) and the
     accepted goals, by |true spacing| band; plus the steps of a policy rollout (true spacing, measured validity).
     Reference validity by band: Bmad- vs surrogate-rendered test rows (results/screen_failure_detect_2026-10-18).
  2. error by goal band: continuous evals (results/validity_band_2026-10-19/eval/*/errors.npz, nominal jitter) and the
     closed-loop Bmad runs (results/closed_loop_bmad_2026-10-1{8,9}).
  3. where the DAQ scans' readings and replay goals sit (results/online_replay_v6_median_cut_2026-10-15).
Outputs results/validity_band_2026-10-19/summary.json (+ printed tables).

  PYTHONPATH=src python scripts/validity_band_2026-10-19.py [--device cuda:0]
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_s2e_rl.artifacts import policy_flow_ckpt
from twobunch_s2e_rl.rl._train_utils import build_env_fn
from twobunch_s2e_rl.rl.rollout import load_policy, rollout

OUT = Path("results/validity_band_2026-10-19")
EDGES = [0, 60, 100, 150, 250, 1e9]                       # |spacing| bands [um]
LABELS = ["<60", "60-100", "100-150", "150-250", ">250"]
RUNS = {"noblen": "logs/2026-10-18/night/zs_screen_noblen_jit_s0", "bc11loop": "logs/2026-10-18/night/zs_screen_bc11loop_jit_s1"}


def band(sp_m):
    return np.clip(np.digitize(np.abs(np.asarray(sp_m)) * 1e6, EDGES) - 1, 0, len(LABELS) - 1)


def by_band(sp_m, val):
    b, v = band(sp_m), np.asarray(val, float)
    return {LABELS[i]: [float(np.nanmean(v[b == i])) if (b == i).any() else None, int((b == i).sum())] for i in range(len(LABELS))}


def exposure(name, logdir, device, n=4096, reps=4):
    cfg = yaml.safe_load((Path(logdir) / "cfg.yaml").read_text())
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(num_envs=n, device=device, seed=4242, episode_length=10 ** 6,
                                                      no_grad=True)
    cand_sp, cand_ok, goal_sp = [], [], []
    with torch.no_grad():
        for _ in range(reps):
            env.reset()
            B, k = env.num_envs, len(env.controls)
            c = torch.minimum(torch.maximum(env._start + (2 * env._rand(B, k) - 1) * env._goal_radius, env._start_lo),
                              env._start_hi)
            o = env._plant(c, noise=False)
            cand_sp.append(o["bunch_spacing_cloud"].cpu().numpy())
            cand_ok.append(o["screen_valid_deployed"].cpu().numpy())
            g = next(g for key, g, _ in env._goal_terms if key == "bunch_spacing")
            goal_sp.append(env._goals_cloud[g].cpu().numpy())
    cand_sp, cand_ok, goal_sp = map(np.concatenate, (cand_sp, cand_ok, goal_sp))
    # policy rollout in the training env (training noise / jitter), 512 envs x 128 steps
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(num_envs=512, device=device, seed=777, episode_length=10 ** 6,
                                                      no_grad=True)
    actor, rms = load_policy(str(Path(logdir) / "final_policy.pt"), device)
    env.reset()
    tr = rollout(env, actor, rms, n_steps=128)
    st_sp = tr["bunch_spacing_cloud"].cpu().numpy().ravel()
    st_ok = tr["screen_valid_deployed_true"].cpu().numpy().ravel() if "screen_valid_deployed_true" in tr else None
    gb = band(goal_sp)
    return {"candidates_by_band": {LABELS[i]: int((band(cand_sp) == i).sum()) for i in range(len(LABELS))},
            "candidate_valid_rate_by_band": by_band(cand_sp, cand_ok),
            "accepted_goal_frac_by_band": {LABELS[i]: float((gb == i).mean()) for i in range(len(LABELS))},
            "candidate_frac_by_band": {LABELS[i]: float((band(cand_sp) == i).mean()) for i in range(len(LABELS))},
            "rollout_step_frac_by_band": {LABELS[i]: float((band(st_sp) == i).mean()) for i in range(len(LABELS))},
            "rollout_valid_rate_by_band": by_band(st_sp, st_ok) if st_ok is not None else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    summ = {"bands_um": LABELS}
    # reference validity by band, test rows (4096 particles)
    r = np.load("results/screen_failure_detect_2026-10-18/rows.npz")
    summ["test_rows_valid_rate_by_band"] = {s: by_band(r["Bmad|bunch_spacing_cloud"], r[f"{s}|screen_valid_deployed"] > 0.5)
                                            for s in ("Bmad", "spacing head")}
    # 1. training exposure
    summ["exposure"] = {name: exposure(name, d, a.device) for name, d in RUNS.items()}
    # 2a. continuous evals by goal band
    ev = {}
    for f in sorted(glob.glob(str(OUT / "eval" / "*" / "errors.npz"))):
        z = np.load(f)
        if "seg_goal_sp" not in z:
            continue
        s, g = z["seg_settled_sp"], z["seg_goal_sp"]
        ev[Path(f).parent.name] = {"settled_median_um": {k: [None if v[0] is None else v[0] * 1e6, v[1]]
                                                         for k, v in _med_by_band(g, s).items()},
                                   "within10_by_band": by_band(g, s <= 10e-6)}
    summ["continuous_by_goal_band"] = ev
    # 2b. closed-loop Bmad by goal band (all runs on disk)
    cl = {}
    for f in sorted(glob.glob("results/closed_loop_bmad_2026-10-1[89]/*_bmad/traj.pt")):
        t = torch.load(f)
        gk = "spacing_goal_cloud" if "spacing_goal_cloud" in t else None
        if gk is None:
            continue
        e = (t["bunch_spacing_cloud"][-1] - t[gk][-1]).abs().numpy()
        cl[Path(f).parent.name] = {"within10_by_band": by_band(t[gk][-1].numpy(), e <= 10e-6),
                                   "median_um_by_band": {k: [None if v[0] is None else v[0] * 1e6, v[1]]
                                                         for k, v in _med_by_band(t[gk][-1].numpy(), e).items()}}
    summ["closed_loop_bmad_by_goal_band"] = cl
    # 3. DAQ scans: readings / validity by band, replay goals by band
    daq = {}
    for scan in ("TEST_15668", "E300_15673", "E300_15671"):
        st = list(json.loads(Path(f"results/online_replay_v6_median_cut_2026-10-15/{scan}.json").read_text())["steps"].values())
        sp = np.array([s.get("spacing_um", np.nan) for s in st]) * 1e-6
        ok = np.array([s["valid"] for s in st]) > 0.5
        daq[scan] = {"readings_by_band": {LABELS[i]: int((band(sp) == i).sum()) for i in range(len(LABELS))},
                     "valid_rate_by_band": by_band(sp, ok),
                     "valid_spacing_range_um": [float(np.nanmin(np.abs(sp[ok])) * 1e6), float(np.nanmax(np.abs(sp[ok])) * 1e6)] if ok.any() else None}
    for f in sorted(glob.glob("results/eval_night_2026-10-18/*_replay_*/traj.npz")):
        z = np.load(f)
        g = z["goal_spacing"][0] if z["goal_spacing"].ndim > 1 else z["goal_spacing"]
        daq.setdefault("replay_goals", {})[Path(f).parent.name] = {
            "goal_frac_by_band": {LABELS[i]: float((band(g) == i).mean()) for i in range(len(LABELS))},
            "in10_by_band": by_band(g, np.abs(z["sp_err"]) <= 10e-6)}
    summ["daq"] = daq
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def _med_by_band(g, v):
    b = band(g)
    return {LABELS[i]: [float(np.median(v[b == i])) if (b == i).any() else None, int((b == i).sum())] for i in range(len(LABELS))}


if __name__ == "__main__":
    main()
