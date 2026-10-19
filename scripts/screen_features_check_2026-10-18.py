"""Screen-feature check (review 2026-10-18 issue 6): the XTCAV (DTOTR) features the controller actually uses, rendered
from BMAD beams vs from the surrogate (B2-cov, B2-cov + spacing head) at identical knobs, on the held-out TEST rows.
Same plant as the env (configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml `screen:`: Cheetah XTCAVF ->
DTOTR at +-V, energy split, valley threshold, row correction, ROI / edge check, deployed validity), with golden
optics, the nominal 22 MV TCAV voltage, no arrival jitter and no measurement noise; 4096 particles per bunch for all
sources (Bmad subsampled, seeded). Outputs results/2_surrogate_screen/screen_features_check_2026-10-18/rows.npz + summary.json and
figures in lab-notebook/images/2026-10-18_screen_*.png. CPU.

  PYTHONPATH=src python scripts/screen_features_check_2026-10-18.py [--max-rows N]
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from twobunch_s2e_rl.rl._train_utils import build_env_fn
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

T = "processed/xtcav23e_inj_cov_2026-10-17.h5"
CFG = "configs/rl/realistic_2026-10-17/real_covB2_screen_noblen.yaml"
FLOWS = {"B2-cov": "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",
         "spacing head": str(next(Path("trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints").glob("*.ckpt")))}
OUT = Path("results/2_surrogate_screen/screen_features_check_2026-10-18")
KEYS = ("bunch_spacing", "drive_sigma_z", "drive_charge_fraction", "screen_valid_deployed", "screen_misassign",
        "bunch_spacing_cloud", "drive_sigma_z_cloud", "drive_charge_fraction_cloud")
N = 4096


def plant_for(flow_ckpt):
    cfg = yaml.safe_load(Path(CFG).read_text())
    de = cfg["params"]["diff_env"]
    de["flow_ckpt"] = flow_ckpt
    de["n_particles"] = N
    de["screen"]["arrival_jitter_s"] = 0.0
    env = build_env_fn(cfg, flow_ckpt)(num_envs=1, device="cpu", no_grad=True, episode_length=32)
    return env._observe_fn


def bmad_source(rows_ref, ds, paths):
    from pmd_beamphysics import ParticleGroup
    rng = np.random.default_rng(0)

    def stack(pg):
        return np.stack([pg.x, pg.y, pg.z, pg.px, pg.py, pg.pz], axis=1).astype(np.float32)

    def fn(knobs_norm, n):
        rows = rows_ref["rows"]
        B = len(rows)
        d = np.zeros((B, n, 6), np.float32)
        w = np.zeros((B, n, 6), np.float32)
        for i, r in enumerate(rows):
            P = ParticleGroup(paths[r])
            u = np.unique(P.weight)
            dd, ww = stack(P[P.weight == u[-1]]), stack(P[P.weight == u[0]])
            d[i] = dd[rng.choice(len(dd), n, replace=len(dd) < n)]
            w[i] = ww[rng.choice(len(ww), n, replace=len(ww) < n)]
        dt, wt = torch.as_tensor(d), torch.as_tensor(w)
        obs = {}
        for name, parts in (("drive", dt), ("witness", wt)):
            for k, v in per_bunch(parts).items():
                obs[f"{name}_{k}"] = v
        obs.update(inter_bunch(dt, wt))
        rr = torch.as_tensor(rows)
        obs.update(p_surv=torch.ones(B), T_drive=ds.drive_frac[rr].float(), T_witness=ds.witness_frac[rr].float())
        return obs, dt, wt
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    ck = torch.load(FLOWS["B2-cov"], map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=T, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    ds = dm.full
    with h5py.File(T) as h:
        names = h.attrs["campaigns"].split(",")
        camp, idx = h["campaign"][...], h["idx"][...]
    both = (ds.drive_density & ds.witness_density).numpy().astype(bool)
    rows = np.array([i for i in dm.test.indices if both[i]])
    if a.max_rows:
        rows = rows[np.random.default_rng(0).permutation(len(rows))[:a.max_rows]]
    paths = {int(r): f"data/{names[camp[r]]}/sample_{idx[r]:05d}_XTCAVF.h5" for r in rows}
    res = {}
    for src in ("Bmad", *FLOWS):
        plant = plant_for(FLOWS["B2-cov" if src == "Bmad" else src])
        ref = {"rows": []}
        if src == "Bmad":
            plant.set_cloud_source(bmad_source(ref, ds, paths))
        acc = {k: [] for k in KEYS}
        for b in range(0, len(rows), a.batch):
            rr = [int(x) for x in rows[b:b + a.batch]]
            ref["rows"] = rr
            with torch.no_grad():
                o = plant(ds.knobs[rr].float())
            for k in KEYS:
                acc[k].append(o[k].detach().cpu().numpy().astype(float))
        res[src] = {k: np.concatenate(v) for k, v in acc.items()}
        print(f"{src}: {len(rows)} rows done", flush=True)
    kn = ds.knobs[rows].numpy()
    np.savez(OUT / "rows.npz", rows=rows, knobs_norm=kn,
             **{f"{s}|{k}": v for s, d in res.items() for k, v in d.items()})
    summ = {"n_rows": int(len(rows))}
    B = res["Bmad"]
    for src in FLOWS:
        S = res[src]
        ok = (B["screen_valid_deployed"] > 0.5) & (S["screen_valid_deployed"] > 0.5)
        e = (S["bunch_spacing"] - B["bunch_spacing"])[ok] * 1e6
        ez = (S["drive_sigma_z"] / B["drive_sigma_z"] - 1)[ok]
        summ[src] = {"valid_agree": float(np.mean((B["screen_valid_deployed"] > 0.5) == (S["screen_valid_deployed"] > 0.5))),
                     "both_valid": int(ok.sum()),
                     "spacing_screen_mae_um": float(np.mean(np.abs(e))), "spacing_screen_bias_um": float(np.median(e)),
                     "spacing_screen_p90_um": float(np.percentile(np.abs(e), 90)),
                     "sigz_screen_mean_abs_rel": float(np.mean(np.abs(ez))), "sigz_screen_bias_rel": float(np.median(ez)),
                     "qfrac_screen_mae": float(np.mean(np.abs(S["drive_charge_fraction"] - B["drive_charge_fraction"])[ok]))}
    for src, S in res.items():
        ok = S["screen_valid_deployed"] > 0.5
        summ[f"{src} screen-vs-own-cloud"] = {
            "valid_frac": float(ok.mean()),
            "spacing_meas_err_median_um": float(np.median((S["bunch_spacing"] - S["bunch_spacing_cloud"])[ok]) * 1e6),
            "spacing_meas_err_mad_um": float(np.median(np.abs(S["bunch_spacing"] - S["bunch_spacing_cloud"])[ok]) * 1e6),
            "sigz_meas_err_median_rel": float(np.median((S["drive_sigma_z"] / S["drive_sigma_z_cloud"] - 1)[ok]))}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
