"""Crop window vs screen readings (2026-10-19): how the screen ROI (the deployed auto-crop, centred on each image's
charge centroid; env default x +-4.6 mm, y +-2.44 mm, edge check 1% within 0.3 mm of the x edge) limits validity and
accuracy, for Bmad- vs surrogate-rendered test rows (same plant / rows as scripts/screen_failure_detect_2026-10-18.py:
golden optics, 22 MV, no noise, 4096 particles; spacing-head flow). Scales the ROI by 1, 1.5, 2 and removes it.
Outputs results/2_surrogate_screen/crop_window_2026-10-19/{rows.npz, summary.json}.

  PYTHONPATH=src python scripts/crop_window_check_2026-10-19.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from twobunch_s2e_rl.rl._train_utils import build_env_fn

spec = importlib.util.spec_from_file_location("sfc", Path(__file__).with_name("screen_features_check_2026-10-18.py"))
sfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfc)
spec2 = importlib.util.spec_from_file_location("sfd", Path(__file__).with_name("screen_failure_detect_2026-10-18.py"))
sfd = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(sfd)

OUT = Path("results/2_surrogate_screen/crop_window_2026-10-19")
SCALES = (1.0, 1.5, 2.0, None)
EDGES, LABELS = [0, 60, 100, 150, 250, 1e9], ["<60", "60-100", "100-150", "150-250", ">250"]
KEYS = ("bunch_spacing", "bunch_spacing_cloud", "screen_valid_deployed", "screen_misassign", "drive_sigma_z",
        "drive_sigma_z_cloud")


def plant_for(flow_ckpt, scale):
    cfg = yaml.safe_load(Path(sfc.CFG).read_text())
    de = cfg["params"]["diff_env"]
    de["flow_ckpt"], de["n_particles"] = flow_ckpt, sfc.N
    sc = de["screen"]
    sc["arrival_jitter_s"] = 0.0
    if scale is None:
        sc["roi"] = None
        sc["edge_frac_max"] = None
    else:
        sc["roi"] = {k: float(v) * scale for k, v in sc["roi"].items()}
    return build_env_fn(cfg, flow_ckpt)(num_envs=1, device="cpu", no_grad=True, episode_length=32)._observe_fn


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    # rows / Bmad paths exactly as the screen checks
    import h5py
    from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
    from twobunch_s2e_rl.surrogate.train import SCALER_KEYS
    ck = torch.load(sfc.FLOWS["B2-cov"], map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=sfc.T, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    ds = dm.full
    with h5py.File(sfc.T) as h:
        names = h.attrs["campaigns"].split(",")
        camp, idx = h["campaign"][...], h["idx"][...]
    both = (ds.drive_density & ds.witness_density).numpy().astype(bool)
    rows = np.array([i for i in dm.test.indices if both[i]])
    paths = {int(r): f"data/{names[camp[r]]}/sample_{idx[r]:05d}_XTCAVF.h5" for r in rows}
    res = {}
    for scale in SCALES:
        tag = "none" if scale is None else f"x{scale:g}"
        for src, ckpt in (("Bmad", sfc.FLOWS["B2-cov"]), ("surrogate", sfc.FLOWS["spacing head"])):
            plant = plant_for(ckpt, scale)
            ref = {"rows": []}
            if src == "Bmad":
                plant.set_cloud_source(sfc.bmad_source(ref, ds, paths))
            acc = {k: [] for k in KEYS}
            torch.manual_seed(0)
            for b in range(0, len(rows), 24):
                rr = [int(x) for x in rows[b:b + 24]]
                ref["rows"] = rr
                with torch.no_grad():
                    o = plant(ds.knobs[rr].float())
                for k in KEYS:
                    acc[k].append(o[k].detach().cpu().numpy().astype(float))
            res[(tag, src)] = {k: np.concatenate(v) for k, v in acc.items()}
            print(f"{tag} {src} done", flush=True)
    np.savez(OUT / "rows.npz", rows=rows, **{f"{t}|{s}|{k}": v for (t, s), d in res.items() for k, v in d.items()})
    truth = res[("x1", "Bmad")]["bunch_spacing_cloud"]
    band = np.clip(np.digitize(np.abs(truth) * 1e6, EDGES) - 1, 0, len(LABELS) - 1)
    summ = {"n_rows": int(len(rows)), "bands_um": LABELS}
    for scale in SCALES:
        tag = "none" if scale is None else f"x{scale:g}"
        B, S = res[(tag, "Bmad")], res[(tag, "surrogate")]
        vb, vs = B["screen_valid_deployed"] > 0.5, S["screen_valid_deployed"] > 0.5
        eb = np.abs(B["bunch_spacing"] - B["bunch_spacing_cloud"]) * 1e6       # Bmad reading vs true beam
        es = np.abs(S["bunch_spacing"] - S["bunch_spacing_cloud"]) * 1e6
        d = np.abs(S["bunch_spacing"] - B["bunch_spacing"]) * 1e6               # surrogate vs Bmad reading
        ent = {}
        for i, lab in enumerate(LABELS):
            m = band == i
            ok = m & vb & vs
            ent[lab] = {"n": int(m.sum()), "valid_bmad": float(vb[m].mean()), "valid_surr": float(vs[m].mean()),
                        "valid_agree": float((vb[m] == vs[m]).mean()),
                        "bmad_screen_vs_beam_median_um": float(np.median(eb[m & vb])) if (m & vb).any() else None,
                        "bmad_screen_vs_beam_within10": float((eb[m & vb] <= 10).mean()) if (m & vb).any() else None,
                        "surr_screen_vs_beam_within10": float((es[m & vs] <= 10).mean()) if (m & vs).any() else None,
                        "surr_vs_bmad_median_um": float(np.median(d[ok])) if ok.any() else None,
                        "surr_vs_bmad_within10": float((d[ok] <= 10).mean()) if ok.any() else None}
        ok = vb & vs
        ent["all"] = {"valid_bmad": float(vb.mean()), "valid_surr": float(vs.mean()), "valid_agree": float((vb == vs).mean()),
                      "bmad_screen_vs_beam_within10": float((eb[vb] <= 10).mean()),
                      "surr_vs_bmad_within10": float((d[ok] <= 10).mean()), "surr_vs_bmad_gt30": float((d[ok] > 30).mean())}
        summ[tag] = ent
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
