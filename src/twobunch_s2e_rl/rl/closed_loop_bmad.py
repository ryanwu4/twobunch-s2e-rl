"""Closed-loop Bmad check of a delta-controller policy: the policy acts on Bmad-tracked beams for a whole episode.

The policy's OWN env (its cfg.yaml, all hidden randomization: gains, offsets, drift, optics, streak scale, shot noise,
BLEN obs) runs unchanged; only the plant's beam source is swapped (plants: `observe.set_cloud_source`) from the flow
to Bmad tracking to XTCAVF (`_bmad_xtcav_server`, bmad-qpad-dev, a pool of single-thread workers), so the screen /
measurement pipeline is the same code the policy was trained on.

  phase 1 (surrogate): reset K envs (no auto-reset, no off-context goals); set the injector context
      golden: every env at the golden injector (its cached Impact-T beam)
      inj:    env i at the i-th DAQ-distribution ('local') injector point of the coverage campaign
              (data/injector_cov_2026-10-16, cached Impact-T beams; context = that point's inj12 values)
      and redraw the goal control points under it (the surrogate picks goals the screen can read).
  phase 2 (Bmad): the goal readings at the goal points and the start readings are Bmad readings (an operator
      records a goal on the machine), then `episode_length` policy steps on Bmad.
  --plant flow: the same episodes (same seeds, contexts, goal points) on the surrogate, for a paired comparison.

Tao seed fixed per episode (common random numbers across its steps); each injector beam downsampled ONCE (seeded) to
--num-macro. Outputs in --out: traj.pt (per-step achieved dict, actions, true controls), summary.json (final true
error vs the true goal, as eval_delta), calls.jsonl (per Bmad call wall time / failures).

  PYTHONPATH=src python -m twobunch_s2e_rl.rl.closed_loop_bmad --logdir <run> --mode golden --k 24 --workers 24 \
      --num-macro 20000 --out results/rl/<run>/closed_loop_bmad_golden
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from ..datagen.paths import repo_root
from ..datagen.sweep_params import INJECTOR_PARAMS, injector_golden, resolve_sweep_set
from ..surrogate.properties import inter_bunch, per_bunch
from ._bmad_xtcav_server import READY, RESULT

BMAD_PYTHON = "/home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python"
COV_CFG = "logs/injector_cov_2026-10-16/launch_config.yaml"
TOL = {"bunch_spacing": ("abs", 10e-6), "drive_sigma_z": ("rel", 0.10)}


class BmadServer:
    """slac-rl-side client of rl/_bmad_xtcav_server (JSON lines; a reader thread drains stdout, see bmad_bridge)."""

    def __init__(self, campaign_cfg, num_macro, workers, scratch, ready_timeout=1800.0):
        Path(scratch).mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONPATH": str(repo_root() / "src"), "OMP_NUM_THREADS": "1",
               "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        self._err = open(Path(scratch) / "server.err", "a")
        self.proc = subprocess.Popen(
            [BMAD_PYTHON, "-u", "-m", "twobunch_s2e_rl.rl._bmad_xtcav_server", "--campaign-cfg", str(campaign_cfg),
             "--num-macro", str(num_macro), "--workers", str(workers), "--scratch", str(scratch)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._err, text=True, bufsize=1, env=env,
            cwd=str(repo_root()))
        self._q: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()
        self._wait(lambda s: s.startswith(READY), ready_timeout, "server start")

    def _pump(self):
        for line in self.proc.stdout:
            self._q.put(line.rstrip("\n"))
        self._q.put(None)

    def _wait(self, pred, timeout, what):
        t_end = time.time() + timeout
        while True:
            try:
                line = self._q.get(timeout=max(1.0, t_end - time.time()))
            except queue.Empty:
                raise TimeoutError(f"Bmad server: no answer within {timeout:.0f}s ({what})") from None
            if line is None:
                raise RuntimeError(f"Bmad server exited ({what}); see server.err")
            if pred(line):
                return line

    def request(self, req, timeout=7200.0):
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        line = self._wait(lambda s: s.startswith(RESULT), timeout, req["cmd"])
        out = json.loads(line[len(RESULT):])
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(f"Bmad server {req['cmd']}: {out['error']}")
        return out

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
            self.proc.stdin.flush()
            self.proc.wait(timeout=120)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _stack(pg):
    return np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], axis=1).astype(np.float32)


class BmadSource:
    """fn(knobs_norm, n) -> (observables, drive (B, n, 6), witness (B, n, 6)) from Bmad, the flow-source contract."""

    def __init__(self, server, knob_space, beams, seeds, out_dir, device, log=None):
        self.server, self.ks, self.beams, self.seeds = server, knob_space, list(beams), list(seeds)
        self.lin = [k for k in knob_space.keys if k not in INJECTOR_PARAMS]
        self.out, self.device, self.log = Path(out_dir), device, log
        self.out.mkdir(parents=True, exist_ok=True)
        self.calls, self._prev = 0, []
        self.rng = np.random.default_rng(0)

    def __call__(self, knobs_norm, n):
        from pmd_beamphysics import ParticleGroup
        phys = self.ks.denormalize(knobs_norm.detach()).cpu().numpy().astype(float)
        B = phys.shape[0]
        tasks = [{"id": i, "knobs": {k: float(phys[i, self.ks.index(k)]) for k in self.lin}, "beam": self.beams[i],
                  "seed": int(self.seeds[i]), "out": str(self.out / f"c{self.calls:03d}_e{i:02d}")} for i in range(B)]
        t0 = time.time()
        res = {r["id"]: r for r in self.server.request({"cmd": "track", "tasks": tasks})}
        drive = np.zeros((B, n, 6), np.float32)
        witness = np.zeros((B, n, 6), np.float32)
        Td, Tw, ps = np.zeros(B), np.zeros(B), np.zeros(B)
        fails = []
        for i in range(B):
            r = res[i]
            if not r.get("ok"):
                fails.append(i)
                continue
            P = ParticleGroup(r["h5"])
            w = np.unique(P.weight)
            dpg = P[P.weight == w[-1]]
            wpg = P[P.weight == w[0]] if len(w) >= 2 else None
            d = _stack(dpg)
            drive[i] = d[self.rng.choice(len(d), n, replace=len(d) < n)]
            Td[i] = r["n_live_drive"] / max(r["n_in_drive"], 1)
            if wpg is not None and len(wpg.x) > 0:
                wv = _stack(wpg)
                witness[i] = wv[self.rng.choice(len(wv), n, replace=len(wv) < n)]
                Tw[i] = r["n_live_witness"] / max(r["n_in_witness"], 1)
                ps[i] = float(len(wpg.x) >= n)
            else:
                witness[i] = drive[i]                 # placeholder; T_witness = 0 drives the hinge / screen validity
        for i in fails:                               # lost row: the previous good row of that env (else golden-like zeros)
            drive[i] = self._prev[0][i] if self._prev else drive[i]
            witness[i] = self._prev[1][i] if self._prev else witness[i]
        if self.log is not None:
            self.log.write(json.dumps({"call": self.calls, "wall_s": time.time() - t0, "fails": fails,
                                       "wall_max_s": max((r.get("wall_s", 0) for r in res.values()), default=0),
                                       "errors": {i: res[i].get("error", "")[-300:] for i in fails}}) + "\n")
            self.log.flush()
        self._prev = [drive, witness]                 # XTCAVF beams stay on disk (calls x episodes, ~1.5 MB each)
        self.calls += 1
        dev = self.device
        dt, wt = torch.as_tensor(drive, device=dev), torch.as_tensor(witness, device=dev)
        obs = {}
        for name, parts in (("drive", dt), ("witness", wt)):
            for k, v in per_bunch(parts).items():
                obs[f"{name}_{k}"] = v
        obs.update(inter_bunch(dt, wt))
        f = dict(device=dev, dtype=torch.float32)
        obs.update(p_surv=torch.as_tensor(ps, **f), T_drive=torch.as_tensor(Td, **f), T_witness=torch.as_tensor(Tw, **f))
        return obs, dt, wt


def _local_points(k):
    rows = json.loads((repo_root() / "data/injector_cov_2026-10-16/manifest.json").read_text())
    pts = {}
    for r in rows:
        if r["injector_point"].startswith("lhsloc") and r["injector_point"] not in pts:
            pts[r["injector_point"]] = r["injector"]
    tags = sorted(pts, key=lambda t: int(t[len("lhsloc"):]))[:k]
    if len(tags) < k:
        raise SystemExit(f"only {len(tags)} local injector points")
    return [(t, pts[t]) for t in tags]


def _summary(env, traj):
    out = {}
    for key, g, _ in env._goal_terms:
        kind, tol = TOL.get(key, ("rel", 0.1))
        true = traj[f"{key}_cloud"] if f"{key}_cloud" in traj else traj[f"{key}_true"]
        goal = traj[f"{g}_cloud"]
        meas, gm = traj[key], traj[g]
        e = (true - goal).abs() if kind == "abs" else (true - goal).abs() / goal.abs()
        em = (meas - gm).abs() if kind == "abs" else (meas - gm).abs() / gm.abs()
        ef, e0 = e[-1].cpu().numpy(), e[0].cpu().numpy()
        out[key] = {"err_first_true_median": float(np.median(e0)), "err_final_true_median": float(np.median(ef)),
                    "err_final_true_p90": float(np.percentile(ef, 90)),
                    "frac_within_tol_final_true": float((ef <= tol).mean()),
                    "frac_within_tol_final_meas": float((em[-1].cpu().numpy() <= tol).mean()),
                    "per_episode_final_true": ef.tolist(), "tolerance": [kind, tol]}
    for k in ("T_drive", "T_witness"):
        if k in traj:
            out[k] = float(traj[k][-1].mean())
    return out


def run(logdir, mode, k, num_macro, workers, out, plant="bmad", device="cuda:0", policy="final", seed=777,
        campaign_cfg=COV_CFG, steps=None):
    from ..artifacts import policy_flow_ckpt
    from ._train_utils import build_env_fn
    from .delta_env import DeltaControlEnv
    from .rollout import load_policy, rollout
    out = Path(out)
    if (out / "summary.json").exists():
        raise SystemExit(f"{out} done already")
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((Path(logdir) / "cfg.yaml").read_text())
    de = cfg["params"]["diff_env"]
    n_steps = int(steps or de["episode_length"])
    if de.get("v4"):
        de["v4"]["offcontext_goal_frac"] = 0.0
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(num_envs=k, device=device, seed=seed,
                                                       episode_length=10 ** 6, no_grad=True)
    actor, rms = load_policy(str(Path(logdir) / f"{policy}_policy.pt"), device)
    env.reset()
    # injector context
    inj_keys = [key for key in env._ks.keys if key in INJECTOR_PARAMS]
    golden = injector_golden(resolve_sweep_set("inj12")[0])
    if mode == "golden":
        points = [("golden", golden)] * k
    elif mode == "inj":
        points = _local_points(k)
    else:
        raise SystemExit(f"mode {mode!r}")
    for i, (_, p) in enumerate(points):
        for key in inj_keys:
            env._context[i, env._ks.index(key)] = float(p[key])
    env._goal_context = env._context.clone()
    allm = torch.ones(k, dtype=torch.bool, device=env.device)
    if getattr(env, "_goal_radius", None) is not None:
        env._reachable_goals(allm)                    # goal points under the set context (surrogate validity)
    meta = {"logdir": str(logdir), "mode": mode, "k": k, "plant": plant, "num_macro": num_macro, "policy": policy,
            "seed": seed, "injector_points": [t for t, _ in points], "n_steps": n_steps, "inj_keys_in_frame": inj_keys,
            "start_true": env.true_controls().tolist(), "goal_ctrl": env._goal_ctrl.tolist()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    server = None
    if plant == "bmad":
        scratch = out / "bmad"
        server = BmadServer(repo_root() / campaign_cfg, num_macro, workers, scratch)
        keys = sorted({t for t, _ in points})
        paths = server.request({"cmd": "prepare", "points": [{"key": t, "injector": dict(points[[q for q, _ in points].index(t)][1])}
                                                              for t in keys]})
        log = open(out / "calls.jsonl", "a")
        src = BmadSource(server, env._ks, [paths[t] for t, _ in points], [1000 + i for i in range(k)], scratch / "beams",
                         env.device, log=log)
        env._observe_fn.set_cloud_source(src)
        # Bmad goal readings at the goal points, then the start reading
        with torch.no_grad():
            g_obs = env._plant(env._goal_ctrl, noise=False)
            g_view = DeltaControlEnv._reward_view(env, g_obs)
            for key, g, _ in env._goal_terms:
                env._goals[g] = g_view[key]
                env._goals_cloud[g] = g_obs.get(f"{key}_cloud", g_obs.get(f"{key}_true"))
            _, extra, err, achieved = env._evaluate(env._cmd.detach())
        env._last_extra, env._err, env._err_prev, env._last_achieved = extra, err, err, achieved
    try:
        traj = rollout(env, actor, rms, n_steps=n_steps)
        traj["true_controls_end"] = env.true_controls().clone()
    finally:
        if server is not None:
            server.close()
    torch.save({k_: v.cpu() for k_, v in traj.items()}, out / "traj.pt")
    summ = {"meta": {k_: v for k_, v in meta.items() if k_ not in ("start_true", "goal_ctrl")},
            "targets": _summary(env, traj)}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    t = summ["targets"]
    print(f"[{mode}/{plant}] spacing final true median {t['bunch_spacing']['err_final_true_median'] * 1e6:.1f} um, "
          f"p90 {t['bunch_spacing']['err_final_true_p90'] * 1e6:.1f}, within 10 um "
          f"{t['bunch_spacing']['frac_within_tol_final_true']:.2f}; sigma_z rel "
          f"{t['drive_sigma_z']['err_final_true_median']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--mode", choices=("golden", "inj"), required=True)
    ap.add_argument("--plant", choices=("bmad", "flow"), default="bmad")
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--num-macro", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--policy", default="final")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=None, help="override the episode length (smoke tests)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.logdir, a.mode, a.k, a.num_macro, a.workers, a.out, plant=a.plant, device=a.device, policy=a.policy,
        steps=a.steps)


if __name__ == "__main__":
    main()
