"""Can the deployed screen analysis flag its own split failures? (2026-10-18 night, step A.) Re-renders the TEST rows
of scripts/screen_features_check_2026-10-18.py (same plant, golden optics, 22 MV, no noise, 4096 particles) from BMAD
beams and from the spacing-head surrogate, with the truth-free diagnostics on: `screen_cut_sens` (half the spacing
change for a +-2 px energy-cut shift) and the measured drive charge fraction (vs the injected drive fraction, which
the machine knows from the laser). Failure = |screen spacing - own cloud spacing| > 30 um or misassign > 0.15.
Reports per-diagnostic AUC for Bmad failures, and P(fail | diagnostic bin) for Bmad vs surrogate -- the calibration
for the training-time failure injection. Outputs results/screen_failure_detect_2026-10-18/{rows.npz,summary.json}.

  PYTHONPATH=src python scripts/screen_failure_detect_2026-10-18.py [--max-rows N]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from twobunch_s2e_rl.rl._train_utils import build_env_fn
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

spec = importlib.util.spec_from_file_location("sfc", Path(__file__).with_name("screen_features_check_2026-10-18.py"))
sfc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sfc)

OUT = Path("results/screen_failure_detect_2026-10-18")
KEYS = sfc.KEYS + ("screen_cut_sens", "energy_difference")
CUT_PX = 2.0
FAIL_UM, FAIL_MIS = 30.0, 0.15


def plant_for(flow_ckpt):
    cfg = yaml.safe_load(Path(sfc.CFG).read_text())
    de = cfg["params"]["diff_env"]
    de["flow_ckpt"] = flow_ckpt
    de["n_particles"] = sfc.N
    de["screen"]["arrival_jitter_s"] = 0.0
    de["screen"]["cut_sens_px"] = CUT_PX
    env = build_env_fn(cfg, flow_ckpt)(num_envs=1, device="cpu", no_grad=True, episode_length=32)
    return env._observe_fn


def auc(score, label):
    """P(score of a failure > score of a non-failure), ties = 1/2."""
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    s = np.concatenate([pos, neg])
    r = np.argsort(np.argsort(s, kind="mergesort"), kind="mergesort").astype(float) + 1
    for v in np.unique(s):          # average ranks of ties
        m = s == v
        r[m] = r[m].mean()
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
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
    if a.max_rows:
        rows = rows[np.random.default_rng(0).permutation(len(rows))[:a.max_rows]]
    paths = {int(r): f"data/{names[camp[r]]}/sample_{idx[r]:05d}_XTCAVF.h5" for r in rows}
    srcs = {"Bmad": sfc.FLOWS["B2-cov"], "spacing head": sfc.FLOWS["spacing head"]}
    res = {}
    for src, ckpt in srcs.items():
        plant = plant_for(ckpt)
        ref = {"rows": []}
        if src == "Bmad":
            plant.set_cloud_source(sfc.bmad_source(ref, ds, paths))
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
    np.savez(OUT / "rows.npz", rows=rows, knobs_norm=ds.knobs[rows].numpy(),
             **{f"{s}|{k}": v for s, d in res.items() for k, v in d.items()})

    summ = {"n_rows": int(len(rows)), "cut_sens_px": CUT_PX, "fail_def": f"|screen-cloud spacing| > {FAIL_UM} um or misassign > {FAIL_MIS}"}
    edges = [0, 2, 5, 10, 20, 1e9]                                   # cut_sens bins [um]
    for src, S in res.items():
        ok = S["screen_valid_deployed"] > 0.5
        err = np.abs(S["bunch_spacing"] - S["bunch_spacing_cloud"]) * 1e6
        fail = ((err > FAIL_UM) | (S["screen_misassign"] > FAIL_MIS))[ok]
        cs = S["screen_cut_sens"][ok] * 1e6
        dq = np.abs(S["drive_charge_fraction"] - S["drive_charge_fraction_cloud"])[ok]
        combo = np.maximum(cs / 10.0, dq / 0.1)
        b = np.digitize(cs, edges) - 1
        summ[src] = {"n_valid": int(ok.sum()), "fail_rate": float(fail.mean()),
                     "auc_cut_sens": auc(cs, fail), "auc_charge_frac_dev": auc(dq, fail), "auc_combo": auc(combo, fail),
                     "p_fail_by_cut_sens_um": {f"{edges[i]}-{edges[i + 1]}": [float(fail[b == i].mean()) if (b == i).any() else None,
                                                                              int((b == i).sum())] for i in range(len(edges) - 1)},
                     "flag_rate_combo_gt1": float((combo > 1).mean()),
                     "recall_combo_gt1": float((combo > 1)[fail].mean()) if fail.any() else None}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
