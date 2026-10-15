"""Gates for the injector-aware surrogate before RL uses it (round-2 ML review, Q3a), on the processed
table with the split train.py used (--split group --orig-split ...):

  old-val regression   spacing MAE and drive sigma_z relative error on the OLD flow's validation rows,
                       candidate vs the widened starting checkpoint (= today's flow, exactly):
                       pass if spacing MAE <= base + 0.3 um and sigma_z rel error <= base + 0.5%
  paired delta (test)  held-out pre-registered test pairs (injector point vs injector = golden at the
                       same linac knobs): R^2 >= 0.8, slope (fit dpred ~ dtrue) in [0.85, 1.15],
                       p90 |dpred - dtrue| <= 2.5 um
  T calibration        mean |T_pred - T_true| over test rows with T_true in [0.95, 1] <= 0.02
Common random numbers for every prediction. Writes <out>/gates.json.

  python scripts/surrogate_gates.py --processed processed/xtcav23e_inj_2026-10-15.h5 \
      --ckpt 'trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt' \
      --base trained/xtcav23e_inj12_widened_2026-10-15/widened.ckpt --out results/surrogate_gates/ft_s0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from twobunch_s2e_rl.artifacts import resolve_ckpt
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch


@torch.no_grad()
def predict(model, knobs, n=2048, seed=0, dev="cuda"):
    g = torch.Generator().manual_seed(seed)
    zd, zw = torch.randn(n, model.latent_dim, generator=g).to(dev), torch.randn(n, model.latent_dim, generator=g).to(dev)
    out = {"bunch_spacing": [], "drive_sigma_z": [], "T_drive": [], "T_witness": []}
    for b in range(0, len(knobs), 64):
        o = model.observables(knobs[b:b + 64].to(dev), n=n, z_drive=zd, z_witness=zw)
        for k in out:
            out[k].append(o[k].double().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def truth(ds, rows, scal):
    d = ds.drive[rows].double() * scal["drive_std"] + scal["drive_mean"]
    w = ds.witness[rows].double() * scal["witness_std"] + scal["witness_mean"]
    return {"bunch_spacing": inter_bunch(d, w)["bunch_spacing"].numpy(),
            "drive_sigma_z": per_bunch(d)["sigma_z"].numpy(),
            "T_drive": ds.drive_frac[rows].numpy(), "T_witness": ds.witness_frac[rows].numpy()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--processed", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--orig-split", default="processed/xtcav23_combined.h5=xtcav23_full,xtcav23_pilot")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cand = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(a.ckpt)), map_location=dev).eval()
    base = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(a.base)), map_location=dev).eval()
    h5, camps = a.orig_split.split("=", 1)
    scal = {k: getattr(cand, k).cpu().double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    dm = TwoBunchFlowDataModule(processed_h5=a.processed, split="group", orig_split=orig_split_keys(h5, camps.split(",")),
                                scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    ok = ds.drive_density.numpy() & ds.witness_density.numpy()
    orig = orig_split_keys(h5, camps.split(","))
    oldval = [i for i in dm.val.indices if orig.get(ds.row_keys[i]) == "val" and ok[i]]
    res = {"n_oldval": len(oldval)}
    t = truth(ds, oldval, scal) if oldval else None
    for name, m in ((("candidate", cand), ("base", base)) if oldval else ()):
        p = predict(m, ds.knobs[oldval], dev=dev)
        res[f"oldval_{name}"] = {"spacing_mae_um": float(1e6 * np.mean(np.abs(p["bunch_spacing"] - t["bunch_spacing"]))),
                                 "sigma_z_rel": float(np.mean(np.abs(p["drive_sigma_z"] / t["drive_sigma_z"] - 1)))}
    if oldval:
        c, b = res["oldval_candidate"], res["oldval_base"]
        res["gate_oldval"] = bool(c["spacing_mae_um"] <= b["spacing_mae_um"] + 0.3
                                  and c["sigma_z_rel"] <= b["sigma_z_rel"] + 0.005)
    # held-out test pairs
    with h5py.File(a.processed) as h:
        key, ref = h["pair_key"][...], h["pair_ref"][...]
    test = np.asarray(dm.test.indices)
    by = {}
    for i in test:
        if key[i] >= 0 and ok[i]:
            by.setdefault((ds.row_campaign[i], key[i]), {})["ref" if ref[i] else "pt"] = i
    pairs = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2]
    if pairs:
        pt, rf = [x[0] for x in pairs], [x[1] for x in pairs]
        tp, tr = truth(ds, pt, scal), truth(ds, rf, scal)
        pp, pr = predict(cand, ds.knobs[pt], dev=dev), predict(cand, ds.knobs[rf], dev=dev)
        dt = 1e6 * (tp["bunch_spacing"] - tr["bunch_spacing"])
        dp = 1e6 * (pp["bunch_spacing"] - pr["bunch_spacing"])
        r2 = 1 - np.sum((dp - dt) ** 2) / max(np.sum((dt - dt.mean()) ** 2), 1e-30)
        slope = float(np.polyfit(dt, dp, 1)[0]) if len(dt) > 2 else float("nan")
        p90 = float(np.percentile(np.abs(dp - dt), 90))
        res["test_pairs"] = {"n": len(pairs), "R2": float(r2), "slope": slope, "p90_abs_err_um": p90,
                             "signal_rms_um": float(np.sqrt(np.mean(dt ** 2)))}
        res["gate_paired_delta"] = bool(r2 >= 0.8 and 0.85 <= slope <= 1.15 and p90 <= 2.5)
    tt = [i for i in test if ok[i]]
    if tt:
        t2, p2 = truth(ds, tt, scal), predict(cand, ds.knobs[tt], dev=dev)
        err = []
        for k in ("T_drive", "T_witness"):
            m = (t2[k] >= 0.95) & (t2[k] <= 1.0)
            err += list(np.abs(p2[k][m] - t2[k][m]))
        res["T_calibration_mae"] = float(np.mean(err)) if err else None
        res["gate_T"] = bool(err and np.mean(err) <= 0.02)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "gates.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
