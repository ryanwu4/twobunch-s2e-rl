"""Where does the flow's pair error come from? Absolute spacing error of the injector-point row (e_pt) and of its
golden-injector reference row at the same linac knobs (e_ref), their correlation (the cancellation in the
difference), and the flow's absolute error on the OLD xtcav23 rows (golden injector, 1e5) by train / val role.

  PYTHONPATH=src python scripts/injector_pair_error_decomp.py --out results/injector_flow_analysis_2026-10-15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_injector_flow import ORIG, PROC  # noqa: E402
from surrogate_gates import predict, truth  # noqa: E402

from twobunch_s2e_rl.artifacts import resolve_ckpt  # noqa: E402
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda"
    m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt("trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt")),
                                          map_location=dev).eval()
    scal = {k: getattr(m, k).cpu().double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    dm = TwoBunchFlowDataModule(processed_h5=PROC, split="group", orig_split=orig_split_keys(*ORIG),
                                scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    role = np.full(len(ds), "train", dtype=object)
    role[np.asarray(dm.val.indices)] = "val"
    role[np.asarray(dm.test.indices)] = "test"
    with h5py.File(PROC) as h:
        camp, key, ref = h["campaign"][...], h["pair_key"][...], h["pair_ref"][...]
    ok = ds.drive_density.numpy() & ds.witness_density.numpy()
    rows = np.where(ok)[0]
    t = 1e6 * truth(ds, list(rows), scal)["bunch_spacing"]
    p = 1e6 * predict(m, ds.knobs[list(rows)], n=2048, dev=dev)["bunch_spacing"]
    e = dict(zip(rows, p - t))
    res = {}
    for c, name in ((0, "xtcav23_full (old, golden inj)"), (1, "xtcav23_pilot (old)"), (2, "injector_full")):
        for r_ in ("train", "val", "test"):
            sel = [r for r in rows if camp[r] == c and role[r] == r_]
            if sel:
                v = np.array([e[r] for r in sel])
                res[f"{name}/{r_}"] = {"n": len(sel), "mae_um": float(np.abs(v).mean()), "bias_um": float(v.mean())}
    by = {}
    for r in rows:
        if key[r] >= 0 and camp[r] == 2:
            by.setdefault(int(key[r]), {})["ref" if ref[r] else "pt"] = r
    P = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2]
    ep, er = np.array([e[x] for x, _ in P]), np.array([e[y] for _, y in P])
    res["pairs"] = {"n": len(P), "rms_e_pt": float(np.sqrt(np.mean(ep ** 2))), "rms_e_ref": float(np.sqrt(np.mean(er ** 2))),
                    "corr_e_pt_e_ref": float(np.corrcoef(ep, er)[0, 1]), "rms_delta_error": float(np.sqrt(np.mean((ep - er) ** 2)))}
    (Path(a.out) / "pair_error_decomp.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].scatter(er, ep, s=10, alpha=0.7)
    lim = np.array([-40, 40]); ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel="flow error on the golden-injector reference row [um]", ylabel="flow error on the injector-point row [um]",
              title=f"pair errors: corr {res['pairs']['corr_e_pt_e_ref']:.2f} (difference cancels only the shared part)")
    names = [k for k in res if k != "pairs"]
    ax[1].barh(names, [res[k]["mae_um"] for k in names])
    ax[1].set(xlabel="absolute spacing MAE [um]", title="flow (ft_s0 best) spacing error by campaign / role")
    fig.tight_layout()
    fig.savefig(Path(a.out) / "fig7_pair_error_decomp.png", dpi=120)


if __name__ == "__main__":
    main()
