"""Gradient check (research, 2026-10-18 night, step C): d(spacing)/d(L1, L2) and d(drive sigma_z)/d(L1, L2) in Bmad vs
the surrogate (B2-cov, B2-cov + spacing head) -- what the BPTT controller trains on.

Points: 3 injector states (golden, a held-out DAQ-like point, a held-out full-box point) x 4 (L1, L2) points in the
controller start box, all other linac knobs at the baseline set. At each point: central finite differences at
h = 0.3 and 0.6 deg on L1 and on L2 (9 Bmad runs per point, same input beam and Tao seed = common random numbers).
Features per run: the beam (cloud, all live particles) spacing / drive sigma_z, and the screen reading (same plant as the
env: golden optics, 22 MV, no jitter / noise; all live particles, one row at a time).
Surrogate: autograd at the point and the same central differences, each averaged over --seeds sampling seeds
(common random numbers within a seed), --n particles per bunch.

  phase bmad (CPU, bmad-qpad-dev server; ~30-60 min at 96 workers / 50k macroparticles):
      PYTHONPATH=src python scripts/gradient_check_2026-10-18.py bmad --workers 96
  phase surr (CPU or GPU) + comparison:
      PYTHONPATH=src python scripts/gradient_check_2026-10-18.py surr
Outputs results/gradient_check_2026-10-18/{bmad.npz, surr.npz, summary.json}.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_s2e_rl.datagen.sweep_params import INJECTOR_PARAMS, injector_golden, resolve_sweep_set
from twobunch_s2e_rl.rl._train_utils import build_env_fn
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch

OUT = Path("results/gradient_check_2026-10-18")
CFG = "configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml"
FLOWS = {"B2-cov": "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",
         "spacing head": "trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt"}
INJ = ["golden", "lhsloc102", "lhs123"]                 # held-out (test-split) injector points of the coverage campaign
LIN = [(-22.5, -39.0), (-22.5, -35.0), (-20.0, -39.0), (-20.0, -35.0)]
HS = (0.3, 0.6)
CTRL = ("L1PhaseSet", "L2PhaseSet")
FEATS = ("bunch_spacing_cloud", "drive_sigma_z_cloud", "bunch_spacing", "drive_sigma_z", "screen_valid_deployed")


def make_env(flow_ckpt, n):
    cfg = yaml.safe_load(Path(CFG).read_text())
    de = cfg["params"]["diff_env"]
    de["flow_ckpt"], de["n_particles"] = flow_ckpt, n
    de["screen"]["arrival_jitter_s"] = 0.0
    de["screen"]["optics_choices"] = [{}]
    de["screen"]["quad_jitter"] = {}
    de["screen"]["tdc_voltage_range"] = None
    return build_env_fn(cfg, flow_ckpt)(num_envs=1, device="cpu", no_grad=True, episode_length=32)


def injector_points():
    rows = json.loads(Path("data/injector_cov_2026-10-16/manifest.json").read_text())
    pts = {}
    for r in rows:
        pts.setdefault(r["injector_point"], r["injector"])
    pts["golden"] = injector_golden(resolve_sweep_set("inj12")[0])
    return {t: pts[t] for t in INJ}


def design(env):
    """(rows of physical knob vectors, labels): per point the centre, then +-h on L1, +-h on L2 for each h."""
    ks, base = env._ks, env._base.detach().cpu().numpy().reshape(-1).astype(float)
    pts = injector_points()
    X, lab = [], []
    for it, t in enumerate(INJ):
        for il, (l1, l2) in enumerate(LIN):
            x0 = base.copy()
            for key in ks.keys:
                if key in INJECTOR_PARAMS and key in pts[t]:
                    x0[ks.index(key)] = float(pts[t][key])
            x0[ks.index("L1PhaseSet")], x0[ks.index("L2PhaseSet")] = l1, l2
            for c, h, s in [(None, 0.0, 0)] + [(c, h, s) for h in HS for c in CTRL for s in (1, -1)]:
                x = x0.copy()
                if c is not None:
                    x[ks.index(c)] += s * h
                X.append(x)
                lab.append((it, il, -1 if c is None else CTRL.index(c), h, s))
    return np.array(X), np.array(lab, float)


def fd(F, lab):
    """Central differences per (point, control, h) -> (n_inj, n_lin, 2, len(HS)) per feature."""
    out = np.full((len(INJ), len(LIN), 2, len(HS)), np.nan)
    for it in range(len(INJ)):
        for il in range(len(LIN)):
            for c in range(2):
                for ih, h in enumerate(HS):
                    sel = lambda s: np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] == c)  # noqa: E731
                                                   & (lab[:, 3] == h) & (lab[:, 4] == s))[0]
                    out[it, il, c, ih] = (F[sel(1)] - F[sel(-1)]) / (2 * h)
    return out


def phase_bmad(a):
    from twobunch_s2e_rl.datagen.paths import repo_root
    from twobunch_s2e_rl.rl.closed_loop_bmad import COV_CFG, BmadServer
    from pmd_beamphysics import ParticleGroup
    env = make_env(FLOWS["B2-cov"], 4096)
    X, lab = design(env)
    pts = injector_points()
    scratch = OUT / "bmad"
    server = BmadServer(repo_root() / COV_CFG, a.num_macro, a.workers, scratch)
    try:
        paths = server.request({"cmd": "prepare", "points": [{"key": t, "injector": dict(pts[t])} for t in INJ]})
        lin = [k for k in env._ks.keys if k not in INJECTOR_PARAMS]
        tasks = [{"id": i, "knobs": {k: float(X[i, env._ks.index(k)]) for k in lin}, "beam": paths[INJ[int(lab[i, 0])]],
                  "seed": 1000, "out": str(scratch / "beams" / f"r{i:03d}")} for i in range(len(X))]
        (scratch / "beams").mkdir(parents=True, exist_ok=True)
        res = {r["id"]: r for r in server.request({"cmd": "track", "tasks": tasks}, timeout=6 * 3600)}
    finally:
        server.close()
    plant = env._observe_fn
    cur = {}

    def src(knobs_norm, n):
        dt, wt = cur["d"], cur["w"]
        obs = {}
        for name, parts in (("drive", dt), ("witness", wt)):
            for k, v in per_bunch(parts).items():
                obs[f"{name}_{k}"] = v
        obs.update(inter_bunch(dt, wt))
        obs.update(p_surv=torch.ones(1), T_drive=torch.tensor([cur["Td"]]), T_witness=torch.tensor([cur["Tw"]]))
        return obs, dt, wt
    plant.set_cloud_source(src)
    F = {k: np.full(len(X), np.nan) for k in FEATS}
    for i in range(len(X)):
        r = res[i]
        if not r.get("ok"):
            print(f"row {i} failed: {r.get('error', '')[-300:]}", file=sys.stderr)
            continue
        P = ParticleGroup(r["h5"])
        w = np.unique(P.weight)
        st = lambda pg: torch.as_tensor(np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], 1), dtype=torch.float32)  # noqa: E731
        cur.update(d=st(P[P.weight == w[-1]]).unsqueeze(0), w=st(P[P.weight == w[0]]).unsqueeze(0),
                   Td=r["n_live_drive"] / max(r["n_in_drive"], 1), Tw=r["n_live_witness"] / max(r["n_in_witness"], 1))
        with torch.no_grad():
            o = plant(env._ks.normalize(torch.as_tensor(X[i:i + 1], dtype=torch.float32)))
        for k in FEATS:
            F[k][i] = float(o[k][0])
    np.savez(OUT / "bmad.npz", X=X, lab=lab, **F)
    print("bmad phase done", flush=True)


def phase_surr(a):
    B = np.load(OUT / "bmad.npz")
    X, lab = B["X"], B["lab"]
    centre = lab[:, 2] < 0
    summ = {"points": {"injector": INJ, "linac_L1_L2": LIN, "h_deg": list(HS)}, "units": "um/deg (spacing), 1/deg (rel sigma_z)"}
    bm = {k: fd(B[k], lab) for k in FEATS[:4]}
    sv = {}
    for name, ck in FLOWS.items():
        env = make_env(ck, a.n)
        plant, ks = env._observe_fn, env._ks
        Xc = torch.as_tensor(X[centre], dtype=torch.float32)
        G = {k: [] for k in FEATS[:4]}
        Fs = {k: [] for k in FEATS[:4]}
        for s in range(a.seeds):
            kn = ks.normalize(Xc).clone().requires_grad_(True)
            torch.manual_seed(100 + s)
            o = plant(kn)
            g_s = {}
            for k in FEATS[:4]:
                g, = torch.autograd.grad(o[k].sum(), kn, retain_graph=True)
                g_s[k] = np.stack([g[:, ks.index(c)].numpy() / float(ks.high[ks.index(c)] - ks.low[ks.index(c)])
                                   for c in CTRL], -1)        # (n_points, 2) per physical degree
            for k in FEATS[:4]:
                G[k].append(g_s[k])
            with torch.no_grad():
                torch.manual_seed(100 + s)       # same seed for every row: common random numbers across +-h
                o_all = plant(ks.normalize(torch.as_tensor(X, dtype=torch.float32)))
            for k in FEATS[:4]:
                Fs[k].append(o_all[k].numpy().astype(float))
        sv[name] = {"ag": {k: np.mean(G[k], 0) for k in G}, "fd": {k: np.mean([fd(f, lab) for f in Fs[k]], 0) for k in Fs},
                    "ag_seed_sd": {k: np.std(G[k], 0) for k in G}}
        print(f"{name}: surrogate done", flush=True)
    np.savez(OUT / "surr.npz", **{f"bmad|{k}": v for k, v in bm.items()},
             **{f"{n}|{t}|{k}": v for n, d in sv.items() for t in ("ag", "fd") for k, v in d[t].items()})

    def scale(k):
        return 1e6 if "spacing" in k else None
    for k in FEATS[:4]:
        b = bm[k][..., 0]                                     # Bmad FD at h = 0.3: (inj, lin, ctrl)
        if "sigma_z" in k:                                    # relative: d ln sigma_z / d deg
            c0 = B[k][centre].reshape(len(INJ), len(LIN))[..., None]
            b = b / c0
        else:
            b = b * 1e6
        ent = {"bmad_fd03_median_abs": np.median(np.abs(b), (0, 1)).tolist(),
               "bmad_fd03_vs_fd06_rel_diff_median": (float(np.nanmedian(np.abs(bm[k][..., 0] - bm[k][..., 1])
                                                                     / np.abs(bm[k][..., 1]).clip(1e-30)))
                                                     if len(HS) > 1 else None)}
        for name in FLOWS:
            for t in ("ag", "fd"):
                v = sv[name][t][k]
                v = v.reshape(len(INJ), len(LIN), 2) if t == "ag" else v[..., 0]
                if "sigma_z" in k:
                    v = v / c0
                else:
                    v = v * 1e6
                ok = np.isfinite(b) & np.isfinite(v)
                ratio = v[ok] / b[ok]
                ent[f"{name}|{t}"] = {"ratio_median": float(np.median(ratio)),
                                      "ratio_p10_p90": np.percentile(ratio, [10, 90]).tolist(),
                                      "sign_agree": float(np.mean(np.sign(v[ok]) == np.sign(b[ok]))),
                                      "corr": float(np.corrcoef(v[ok], b[ok])[0, 1]),
                                      "per_ctrl_ratio_median": [float(np.median((v[..., c] / b[..., c])[np.isfinite(b[..., c])]))
                                                                for c in range(2)]}
        summ[k] = ent
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=("bmad", "surr"))
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--num-macro", type=int, default=50000)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--smoke", action="store_true", help="1 point, h = 0.3 only, into <out>/smoke")
    a = ap.parse_args()
    if a.smoke:
        global OUT, INJ, LIN, HS
        OUT, INJ, LIN, HS = OUT / "smoke", INJ[1:2], LIN[:1], HS[:1]
    OUT.mkdir(parents=True, exist_ok=True)
    (phase_bmad if a.phase == "bmad" else phase_surr)(a)


if __name__ == "__main__":
    main()
