"""sigma_z vs particle count (2026-10-19): does sampling shot noise bias / scatter the sigma_z the controller sees?

Six settings = centre points of the gradient check (golden injector x 4 linac points, held-out lhs123 x 2), where the
Bmad XTCAVF beams (50k macroparticles) are on disk (results/gradient_check_2026-10-18/bmad/beams). Per setting and
particle count N per bunch, R repeats of:
  surrogate   flow samples (spacing-head flow; B2-cov has the same sigma_z) -> beam sigma_z (std z) + screen sigma_z
  Bmad sub    a random N-subsample of the Bmad beam -> the same features (the sampling noise on the TRUE distribution)
Screen = the env plant (golden optics, 22 MV, no jitter / measurement noise). Reference = the full Bmad beam.
Outputs results/sigmaz_convergence_2026-10-19/{rows.npz, summary.json}.

  PYTHONPATH=src python scripts/sigmaz_convergence_2026-10-19.py [--reps 32] [--device cuda:0]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch

spec = importlib.util.spec_from_file_location("gc", Path(__file__).with_name("gradient_check_2026-10-18.py"))
gc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gc)

OUT = Path("results/sigmaz_convergence_2026-10-19")
NS = (128, 256, 512, 1024, 2048, 4096, 8192, 20000)
SETTINGS = [(0, 0), (0, 1), (0, 2), (0, 3), (2, 1), (2, 3)]      # (injector idx, linac idx) into gc.INJ / gc.LIN
FEATS = ("drive_sigma_z_cloud", "witness_sigma_z", "drive_sigma_z", "bunch_spacing", "screen_valid_deployed")


def bmad_beam(i):
    from pmd_beamphysics import ParticleGroup
    P = ParticleGroup(str(gc.OUT / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5"))
    w = np.unique(P.weight)
    st = lambda pg: torch.as_tensor(np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], 1), dtype=torch.float32)  # noqa: E731
    return st(P[P.weight == w[-1]]), st(P[P.weight == w[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    B = np.load(gc.OUT / "bmad.npz")
    X, lab = B["X"], B["lab"]
    rows = [int(np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0]) for it, il in SETTINGS]
    env = gc.make_env(gc.FLOWS["spacing head"], 1024)
    plant, ks = env._observe_fn, env._ks
    beams = [bmad_beam(i) for i in rows]
    T = {i: (B["drive_sigma_z_cloud"][i], ) for i in rows}
    cur = {}

    def src(knobs_norm, n):
        if cur["mode"] == "flow":
            return gc_flow_obs(knobs_norm, cur["n"])
        dt, wt = cur["d"], cur["w"]
        obs = {}
        for name, parts in (("drive", dt), ("witness", wt)):
            for k, v in per_bunch(parts).items():
                obs[f"{name}_{k}"] = v
        obs.update(inter_bunch(dt, wt))
        one = torch.ones(dt.shape[0])
        obs.update(p_surv=one, T_drive=one, T_witness=one)
        return obs, dt, wt

    from twobunch_s2e_rl.rl.plants import _flow_observables
    flow = env._observe_fn.flow if hasattr(env._observe_fn, "flow") else None
    if flow is None:
        from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
        from twobunch_s2e_rl.artifacts import resolve_ckpt
        flow = TwoBunchFlow.load_from_checkpoint(resolve_ckpt(gc.FLOWS["spacing head"]), map_location="cpu").eval()
    gc_flow_obs = lambda kn, n: _flow_observables(flow, kn, n)          # noqa: E731
    plant.set_cloud_source(src)
    rng = np.random.default_rng(0)
    res = {m: {k: np.full((len(rows), len(NS), a.reps), np.nan) for k in FEATS} for m in ("surrogate", "bmad_sub")}
    full = {k: np.full(len(rows), np.nan) for k in FEATS}
    for si, r in enumerate(rows):
        kn1 = ks.normalize(torch.as_tensor(X[r:r + 1], dtype=torch.float32))
        d, w = beams[si]
        cur.update(mode="bmad", d=d.unsqueeze(0), w=w.unsqueeze(0))
        with torch.no_grad():
            o = plant(kn1)
        for k in FEATS:
            full[k][si] = float(o[k][0])
        for ni, n in enumerate(NS):
            kn = kn1.expand(a.reps, -1).contiguous()
            torch.manual_seed(1000 + 17 * si + ni)
            cur.update(mode="flow", n=n)
            with torch.no_grad():
                o = plant(kn)
            for k in FEATS:
                res["surrogate"][k][si, ni] = o[k].numpy()
            idd = np.stack([rng.choice(len(d), n, replace=len(d) < n) for _ in range(a.reps)])
            idw = np.stack([rng.choice(len(w), n, replace=len(w) < n) for _ in range(a.reps)])
            cur.update(mode="bmad", d=d[torch.as_tensor(idd)], w=w[torch.as_tensor(idw)])
            with torch.no_grad():
                o = plant(kn)
            for k in FEATS:
                res["bmad_sub"][k][si, ni] = o[k].numpy()
        print(f"setting {si} done (Bmad drive sigma_z {full['drive_sigma_z_cloud'][si] * 1e6:.1f} um, "
              f"{len(d)} / {len(w)} particles)", flush=True)
    np.savez(OUT / "rows.npz", NS=np.array(NS), rows=np.array(rows), settings=np.array(SETTINGS),
             **{f"full|{k}": v for k, v in full.items()},
             **{f"{m}|{k}": v for m, d_ in res.items() for k, v in d_.items()})
    summ = {"NS": list(NS), "reps": a.reps, "settings": [{"injector": gc.INJ[it], "L1_L2": gc.LIN[il]} for it, il in SETTINGS]}
    for k in ("drive_sigma_z_cloud", "witness_sigma_z", "drive_sigma_z"):
        ent = {}
        for m in res:
            v = res[m][k]
            if k == "drive_sigma_z":                       # screen: only valid readings
                v = np.where(res[m]["screen_valid_deployed"] > 0.5, v, np.nan)
            mean, sd = np.nanmean(v, -1), np.nanstd(v, -1)
            ent[m] = {"mean_rel_to_bmad_full": (mean / full[k][:, None]).round(4).tolist(),
                      "sd_rel": (sd / full[k][:, None]).round(4).tolist()}
        ent["bmad_full_um"] = (full[k] * 1e6).round(2).tolist()
        summ[k] = ent
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
