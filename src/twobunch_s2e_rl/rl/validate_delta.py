"""Open-loop Bmad check of a delta controller: does the beam it steers to on the surrogate match
the goal in real physics?

  export  (slac-rl):       roll the policy on its own env (all hidden randomization on); for each of
                           K episodes write the final TRUE knob settings (context + true controls,
                           physical), the goals, and the surrogate's prediction there.
  track   (bmad-qpad-dev): track every exported setting through FACET2-S2E (xtcav23 campaign
                           settings) to XTCAVF, save the beam, compute truth metrics with the same
                           kernels as the surrogate (sim.bmad.cloud_metrics).
  compare (either env):    surrogate vs Bmad vs goal per target -> compare.json.

Output dir: results/rl/<run>/bmad_check/ (new; refuses to overwrite).

  python -m twobunch_s2e_rl.rl.validate_delta export --logdir logs/overnight_2026-10-14/delta_xtcav23_cloud_s0
  python -m twobunch_s2e_rl.rl.validate_delta track  --logdir ... --num-macro 20000 --workers 12
  python -m twobunch_s2e_rl.rl.validate_delta compare --logdir ...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import yaml

from ..datagen.paths import repo_root

# campaign whose fixed settings the xtcav23 surrogate was trained under (until campaign.json exists)
CAMPAIGN_CFG = "configs/datagen/xtcav23_full.yaml"
TARGETS = ("bunch_spacing", "drive_sigma_z", "energy_difference", "drive_charge_fraction")


def _out(logdir, tag: str = "") -> Path:
    name = "bmad_check" + (f"_{tag}" if tag else "")
    return repo_root() / "results" / "rl" / Path(logdir).name / name


def export(logdir, k=24, device="cuda", seed=777, policy="best", tag=""):
    import torch
    from ..artifacts import policy_flow_ckpt, write_run_json
    from ._train_utils import build_env_fn
    from .rollout import load_policy, rollout

    device = device if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load((Path(logdir) / "cfg.yaml").read_text())
    de = cfg["params"]["diff_env"]
    env = build_env_fn(cfg, policy_flow_ckpt(logdir))(num_envs=k, device=device, seed=seed,
                                                       episode_length=int(de["episode_length"]),
                                                       no_grad=True)
    actor, rms = load_policy(str(Path(logdir) / f"{policy}_policy.pt"), device)
    # stop one step short of the episode end: the final step would reset the env and redraw the
    # hidden context, which the exported knob settings must come from
    traj = rollout(env, actor, rms, n_steps=int(de["episode_length"]) - 1)
    knobs = env._context.clone()                                       # (K, n_knob) physical
    knobs[:, env._ci] = traj["true_controls"][-1]                      # true controls at the end
    rows = []
    for i in range(k):
        rec = {"i": i, "knobs": {key: float(knobs[i, j]) for j, key in enumerate(env._ks.keys)},
               "goals": {}, "surrogate": {}}
        for key, gkey, _ in env._goal_terms:
            rec["goals"][key] = {"measured": float(traj[gkey][-1, i]),
                                 "cloud": float(traj[f"{gkey}_cloud"][-1, i])}
        for key in TARGETS + ("T_drive", "T_witness"):
            src = f"{key}_cloud" if f"{key}_cloud" in traj else (key if key in traj else None)
            if src is not None:
                rec["surrogate"][key] = float(traj[src][-1, i])
        rows.append(rec)
    out = _out(logdir, tag)
    write_run_json(out, "bmad_check", {"k": k, "seed": seed, "policy": policy, "tag": tag},
                   parents=[str(logdir)], flow_ckpt=policy_flow_ckpt(logdir))
    (out / "setpoints.json").write_text(json.dumps(rows, indent=2))
    print(f"wrote {out / 'setpoints.json'} ({k} setpoints)")


# ---- track (bmad-qpad-dev) ---------------------------------------------------------------------
_W = {}


def _init(camp, n_macro, out_dir):
    os.environ["OMP_NUM_THREADS"] = "1"
    import FACET2_S2E as qs
    from ..datagen.paths import facet2_root
    from ..sim.bmad import make_tao
    root = facet2_root()
    base = qs.loadConfig("/" + camp["baseline_config"], str(root))
    _W.update(qs=qs, root=root, base=base, camp=camp, n=n_macro, out=out_dir,
              tao=make_tao(qs, root, base, csr=camp["csrTF"], wakes=camp.get("transverseWakes", False),
                           n_particles=n_macro, scratch=Path(out_dir) / "scratch"))


def _track_one(row):
    from ..sim.bmad import cloud_metrics, merged_settings, track
    res_path = Path(_W["out"]) / f"bmad_{row['i']:03d}.json"
    if res_path.exists():
        return row["i"], "skip"
    settings = merged_settings(_W["base"], _W["camp"].get("fixed_overrides", {}), row["knobs"])
    prefix = str(Path(_W["out"]) / f"bmad_{row['i']:03d}")
    try:
        rec = track(_W["qs"], _W["tao"], _W["root"], settings, ["XTCAVF"], n_in=_W["n"],
                    save_beams_at=("XTCAVF",), out_prefix=prefix)
        rec["truth"] = cloud_metrics(prefix + "_XTCAVF.h5")
        rec["ok"] = True
    except Exception as e:  # noqa: BLE001 - recorded
        rec = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    res_path.write_text(json.dumps(rec, indent=2, default=float))
    return row["i"], "ok" if rec["ok"] else "fail"


def track_all(logdir, num_macro=20000, workers=12, tag=""):
    import multiprocessing
    out = _out(logdir, tag)
    rows = json.loads((out / "setpoints.json").read_text())
    camp = yaml.safe_load((repo_root() / CAMPAIGN_CFG).read_text())
    with multiprocessing.Pool(workers, initializer=_init, initargs=(camp, num_macro, str(out))) as pool:
        for i, status in pool.imap_unordered(_track_one, rows):
            print(f"  setpoint {i}: {status}", flush=True)


# ---- compare -----------------------------------------------------------------------------------
def compare(logdir, tag=""):
    out = _out(logdir, tag)
    rows = json.loads((out / "setpoints.json").read_text())
    table = {k: {"surrogate": [], "bmad": [], "goal_cloud": []} for k in TARGETS}
    n_ok = 0
    for r in rows:
        p = out / f"bmad_{r['i']:03d}.json"
        if not p.exists():
            continue
        b = json.loads(p.read_text())
        if not b.get("ok") or b["truth"].get("witness_lost"):
            continue
        n_ok += 1
        for k in TARGETS:
            if k in r["surrogate"] and k in b["truth"]:
                table[k]["surrogate"].append(r["surrogate"][k])
                table[k]["bmad"].append(b["truth"][k])
                table[k]["goal_cloud"].append(r["goals"].get(k, {}).get("cloud", np.nan))
    summary = {"n_tracked_ok": n_ok, "n_setpoints": len(rows), "targets": {}}
    for k, t in table.items():
        if not t["bmad"]:
            continue
        s, b, g = (np.array(t[x], float) for x in ("surrogate", "bmad", "goal_cloud"))
        rel = lambda a, c: np.abs(a - c) / np.abs(c).clip(1e-30)
        entry = {"surrogate_vs_bmad_rel_median": float(np.median(rel(s, b))),
                 "surrogate_vs_bmad_rel_p90": float(np.percentile(rel(s, b), 90))}
        if np.isfinite(g).any():
            m = np.isfinite(g)
            entry.update(bmad_vs_goal_abs_median=float(np.median(np.abs(b[m] - g[m]))),
                         bmad_vs_goal_rel_median=float(np.median(rel(b[m], g[m]))),
                         surrogate_vs_goal_abs_median=float(np.median(np.abs(s[m] - g[m]))))
        summary["targets"][k] = entry
    (out / "compare.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["export", "track", "compare"])
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-macro", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="", help="output subdir suffix: bmad_check_<tag>")
    a = ap.parse_args()
    if a.cmd == "export":
        export(a.logdir, a.k, a.device, a.seed, tag=a.tag)
    elif a.cmd == "track":
        track_all(a.logdir, a.num_macro, a.workers, tag=a.tag)
    else:
        compare(a.logdir, tag=a.tag)


if __name__ == "__main__":
    main()
