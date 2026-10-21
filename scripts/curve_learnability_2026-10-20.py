"""Can the beam's backbone be predicted from the knobs? (2026-10-20 diagnostic for the LPS-sharpness work.)

Per row and bunch (standardized frame of the B2-cov scaler): energy frame c = median pz, w = robust std pz;
u = tanh(((pz - c) / w) / 3); ridge least-squares fit of z and x on Chebyshev T0..T12(u) (the "oracle" backbone).
A dedicated MLP (37 knobs -> frame + coefficients, standardized targets) is trained on the train split; on train and
held-out rows we report the residual spread of z around the predicted backbone, as a fraction of the row's z spread:
  oracle  = around the row's own LSQ fit (the floor the basis allows)
  mlp     = around the MLP's prediction (its own frame and coefficients)
  mean    = around the average backbone (no knob dependence); none = 1.
The beam's intrinsic thinness (z given pz) is ~0.02-0.03 of the z spread (results/2_surrogate_screen/flow_sharpness_2026-10-20).

  PYTHONPATH=src python scripts/curve_learnability_2026-10-20.py [--table processed/xtcav23e_inj_cov_2026-10-17.h5]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

CK = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
OUT = Path("results/2_surrogate_screen/curve_learnability_2026-10-20")
ORDER = 12


def basis(u, order=ORDER):
    T = [torch.ones_like(u), u]
    while len(T) < order + 1:
        T.append(2 * u * T[-1] - T[-2])
    return torch.stack(T[:order + 1], -1)


def rs(v, dim=-1):
    med = v.median(dim, keepdim=True).values
    return 1.4826 * (v - med).abs().median(dim).values


def frame_and_fit(parts):
    """parts (M, P, 6) standardized -> c, w (M,), coef (M, ORDER+1, 2) for [z, x], oracle residual fraction (M,)."""
    pz = parts[..., 5]
    c = pz.median(1).values
    w = rs(pz).clamp_min(1e-4)
    u = torch.tanh((pz - c[:, None]) / w[:, None] / 3.0)
    A = basis(u).double()
    Y = parts[..., [2, 0]].double()
    G = A.transpose(1, 2) @ A + 1e-6 * A.shape[1] * torch.eye(ORDER + 1, dtype=A.dtype)
    coef = torch.linalg.solve(G, A.transpose(1, 2) @ Y).float()
    res = parts[..., 2] - torch.einsum("mpo,mo->mp", A.float(), coef[..., 0])
    return c, w, coef, rs(res) / rs(parts[..., 2])


def resid_frac(parts, c, w, coef):
    u = torch.tanh((parts[..., 5] - c[:, None]) / w[:, None] / 3.0)
    res = parts[..., 2] - torch.einsum("mpo,mo->mp", basis(u), coef[..., 0])
    return rs(res) / rs(parts[..., 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="processed/xtcav23e_inj_cov_2026-10-17.h5")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=a.table, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    ds = dm.full
    summ = {"table": a.table, "order": ORDER}
    for k, name in ((0, "drive"), (1, "witness")):
        dens = (ds.drive_density if k == 0 else ds.witness_density).numpy().astype(bool)
        parts_all = ds.drive if k == 0 else ds.witness
        idx = {s: np.array([i for i in getattr(dm, s).indices if dens[i]]) for s in ("train", "val", "test")}
        fits = {}
        for s, ix in idx.items():
            out = [frame_and_fit(parts_all[ix[b:b + 256]].float()) for b in range(0, len(ix), 256)]
            fits[s] = [torch.cat([o[j] for o in out]) for j in range(4)]
        # targets: [c, log w, coef z (13), coef x (13)] standardized on the train split
        def tgt(f):
            return torch.cat([f[0][:, None], f[1].log()[:, None], f[2][..., 0], f[2][..., 1]], 1)
        Ytr = tgt(fits["train"])
        mu, sd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
        X = {s: ds.knobs[ix].float() for s, ix in idx.items()}
        dev = a.device
        net = nn.Sequential(nn.Linear(X["train"].shape[1], a.hidden), nn.SiLU(), nn.Linear(a.hidden, a.hidden), nn.SiLU(),
                            nn.Linear(a.hidden, a.hidden), nn.SiLU(), nn.Linear(a.hidden, Ytr.shape[1])).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
        xt, yt = X["train"].to(dev), ((Ytr - mu) / sd).to(dev)
        for ep in range(a.epochs):
            perm = torch.randperm(len(xt), device=dev)
            for b in range(0, len(xt), 512):
                j = perm[b:b + 512]
                loss = ((net(xt[j]) - yt[j]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        net.eval()
        ent = {}
        for s, ix in idx.items():
            with torch.no_grad():
                p = (net(X[s].to(dev)).cpu() * sd + mu)
            c, w = p[:, 0], p[:, 1].exp()
            coef = torch.stack([p[:, 2:2 + ORDER + 1], p[:, 2 + ORDER + 1:]], -1)
            mean_coef = Ytr.mean(0)
            cm, wm = mean_coef[0].expand(len(ix)), mean_coef[1].exp().expand(len(ix))
            coefm = torch.stack([mean_coef[2:2 + ORDER + 1], mean_coef[2 + ORDER + 1:]], -1).expand(len(ix), -1, -1)
            mlp, mean, orc = [], [], []
            for b in range(0, len(ix), 256):
                pp = parts_all[ix[b:b + 256]].float()
                sl = slice(b, b + 256)
                mlp.append(resid_frac(pp, c[sl], w[sl], coef[sl]))
                mean.append(resid_frac(pp, cm[sl], wm[sl], coefm[sl]))
            mlp, mean = torch.cat(mlp), torch.cat(mean)
            orc = fits[s][3]
            ent[s] = {"n": int(len(ix)), "oracle_median": float(orc.median()), "mlp_median": float(mlp.median()),
                      "mlp_p90": float(mlp.quantile(0.9)), "mean_curve_median": float(mean.median())}
        summ[name] = ent
        print(name, json.dumps(ent), flush=True)
    (OUT / f"summary_{Path(a.table).stem}.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
