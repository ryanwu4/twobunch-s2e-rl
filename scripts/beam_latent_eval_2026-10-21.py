"""Beam-latent surrogate checks (2026-10-21), beyond scripts/flow_sharpness_eval_2026-10-20.py (which samples s from the prior):
  1. posterior sharpness: encode each of the 12 gradient-check Bmad beams (a P-particle subsample, P = the training table's)
     into s ~ q(s | cloud, knobs), sample 40k particles given that s, and measure thinness (surrogate / Bmad, 1 = as sharp
     as Bmad) -- can the conditional flow be sharp at all, given the right latent?
  2. prior realism on the 688 test rows: 16 prior draws per row -> per-draw spacing and drive sigma_z; their spread across
     draws, and Bmad's z-score (Bmad - mean of draws) / std of draws (calibrated: ~68% within 1, ~95% within 2).
Outputs results/2_surrogate_screen/flow_sharpness_2026-10-20/<tag>_latent.json (+ a line in summary_latent.txt).

  PYTHONPATH=src python scripts/beam_latent_eval_2026-10-21.py --tag <tag> --ckpt <ckpt> --P 1024
"""
import argparse
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/2_surrogate_screen/flow_sharpness_2026-10-20"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--P", type=int, default=1024)
    ap.add_argument("--draws", type=int, default=16)
    a = ap.parse_args()
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
    fe = _load("fe", REPO / "scripts/flow_sharpness_eval_2026-10-20.py")
    nf, gc = fe._load("nf", REPO / "scripts/notebook_figs_2026-10-20.py"), fe._load("gc", REPO / "scripts/gradient_check_2026-10-18.py")
    m = TwoBunchFlow.load_from_checkpoint(a.ckpt, map_location="cpu").eval()
    env = gc.make_env(gc.FLOWS["B2-cov"], 1024)
    B = np.load(fe.G / "bmad.npz")
    X, lab = B["X"], B["lab"]
    rng = np.random.default_rng(0)
    res = {"tag": a.tag, "ckpt": a.ckpt}

    # 1. posterior sharpness
    ratios = {"drive pz|z": [], "drive z|pz": [], "drive slice dE": [], "witness pz|z": []}
    for it, il in itertools.product(range(3), range(4)):
        i = int(np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0])
        c, q, isd = nf._bmad_cloud(fe.G / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5")
        c = c.numpy()
        kn = env._ks.normalize(torch.as_tensor(X[i:i + 1], dtype=torch.float32))
        outs = {}
        for k, sel in ((0, isd), (1, ~isd)):
            bm = c[sel]
            mean, std = m._scaler(k)
            sub = bm[rng.choice(len(bm), a.P, replace=len(bm) < a.P)]
            ps = torch.as_tensor((sub - mean.numpy()) / std.numpy(), dtype=torch.float32)[None]
            with torch.no_grad():
                h = m._encode(kn, k)
                mq, _ = m.latent_posterior_params(m._curve(ps, kn, k, -1.0), kn, k, h)
                g = torch.Generator().manual_seed(0)
                outs[k] = (bm, m.sample_bunch(kn, k, 40000, generator=g, s=mq)[0].numpy())
        (bd, sd), (bw, sw) = outs[0], outs[1]
        ratios["drive pz|z"].append(fe.thin(sd[:, 2], sd[:, 5]) / fe.thin(bd[:, 2], bd[:, 5]))
        ratios["drive z|pz"].append(fe.thin(sd[:, 5], sd[:, 2]) / fe.thin(bd[:, 5], bd[:, 2]))
        ratios["drive slice dE"].append(fe.slice_spread(sd[:, 2], sd[:, 5] / sd[:, 5].mean() - 1)
                                        / fe.slice_spread(bd[:, 2], bd[:, 5] / bd[:, 5].mean() - 1))
        ratios["witness pz|z"].append(fe.thin(sw[:, 2], sw[:, 5]) / fe.thin(bw[:, 2], bw[:, 5]))
    res["posterior_thinness_median"] = {k: float(np.median(v)) for k, v in ratios.items()}
    res["posterior_thinness_compressed"] = {k: float(np.median([v[j] for j in (1, 5, 9)])) for k, v in ratios.items()}

    # 2. prior realism on the test rows
    k_ = np.load(fe.SF / "rows.npz")
    kn = torch.as_tensor(k_["knobs_norm"], dtype=torch.float32)
    sp = np.zeros((a.draws, len(kn)))
    sz = np.zeros((a.draws, len(kn)))
    for dr in range(a.draws):
        for b0 in range(0, len(kn), 64):
            g = torch.Generator().manual_seed(1000 * dr + b0)
            with torch.no_grad():
                dd = m.sample_bunch(kn[b0:b0 + 64], 0, 2048, generator=g)
                ww = m.sample_bunch(kn[b0:b0 + 64], 1, 2048, generator=g)
            sp[dr, b0:b0 + 64] = inter_bunch(dd, ww)["bunch_spacing"].numpy()
            sz[dr, b0:b0 + 64] = per_bunch(dd)["sigma_z"].numpy()
    tb, tz = k_["Bmad|bunch_spacing_cloud"], k_["Bmad|drive_sigma_z_cloud"]
    out = {}
    for name, v, t in (("spacing", sp, tb), ("drive_sigma_z", sz, tz)):
        mu, s = v.mean(0), v.std(0).clip(1e-12)
        zsc = (t - mu) / s
        out[name] = {"draw_spread_median": float(np.median(s) * 1e6), "mean_abs_err_median": float(np.median(np.abs(mu - t)) * 1e6),
                     "within_1sigma": float(np.mean(np.abs(zsc) <= 1)), "within_2sigma": float(np.mean(np.abs(zsc) <= 2))}
    res["prior"] = out
    (OUT / f"{a.tag}_latent.json").write_text(json.dumps(res, indent=1))
    line = (f"{a.tag:14s} posterior thin: " + " ".join(f"{kk} {v:.1f}" for kk, v in res["posterior_thinness_median"].items())
            + f" (compressed pz|z {res['posterior_thinness_compressed']['drive pz|z']:.1f}) | prior: "
            + " ".join(f"{kk}: spread {v['draw_spread_median']:.1f} um, |err| {v['mean_abs_err_median']:.1f} um, "
                       f"1s {v['within_1sigma']:.2f} 2s {v['within_2sigma']:.2f}" for kk, v in out.items()))
    with open(OUT / "summary_latent.txt", "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
