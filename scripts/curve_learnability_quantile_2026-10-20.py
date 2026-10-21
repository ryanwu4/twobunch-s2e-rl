"""Curve learnability, quantile representation (2026-10-20): the backbone of each row as the median z (and x) in 32
equal-charge pz bins plus the bin pz medians (64 + 32 numbers), predicted from the knobs by an MLP with a curve-space loss;
applied by linear interpolation in pz. Same metric as scripts/curve_learnability_2026-10-20.py: residual spread of z around
the backbone / the row's z spread (oracle = the row's own binned curve).

  PYTHONPATH=src python scripts/curve_learnability_quantile_2026-10-20.py [--table ...]
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

spec = importlib.util.spec_from_file_location("cl", Path(__file__).with_name("curve_learnability_2026-10-20.py"))
cl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cl)
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule  # noqa: E402
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS  # noqa: E402

NB = 32


def binned(parts):
    """(M, P, 6) -> pz bin medians (M, NB), z / x bin medians (M, NB, 2) over equal-count pz bins."""
    pz = parts[..., 5]
    o = pz.argsort(1)
    srt = torch.gather(parts, 1, o[..., None].expand_as(parts))
    M, P = pz.shape
    chunks = srt[:, : (P // NB) * NB].reshape(M, NB, P // NB, 6)
    return chunks[..., 5].median(2).values, torch.stack([chunks[..., 2].median(2).values, chunks[..., 0].median(2).values], -1)


def interp(pz, qp, qz):
    """piecewise-linear z(pz) through (qp, qz) per row (qp sorted), constant beyond the ends."""
    idx = torch.searchsorted(qp.contiguous(), pz.contiguous()).clamp(1, NB - 1)
    x0, x1 = torch.gather(qp, 1, idx - 1), torch.gather(qp, 1, idx)
    y0, y1 = torch.gather(qz, 1, idx - 1), torch.gather(qz, 1, idx)
    t = ((pz - x0) / (x1 - x0).clamp_min(1e-9)).clamp(0, 1)
    return y0 + t * (y1 - y0)


def frac(parts, qp, qz):
    r = parts[..., 2] - interp(parts[..., 5], qp, qz)
    return cl.rs(r) / cl.rs(parts[..., 2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="processed/xtcav23e_inj_cov_2026-10-17.h5")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    ck = torch.load(cl.CK, map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=a.table, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    ds = dm.full
    summ = {"table": a.table, "bins": NB}
    for k, name in ((0, "drive"), (1, "witness")):
        dens = (ds.drive_density if k == 0 else ds.witness_density).numpy().astype(bool)
        parts_all = ds.drive if k == 0 else ds.witness
        idx = {s: np.array([i for i in getattr(dm, s).indices if dens[i]]) for s in ("train", "val", "test")}
        Q = {}
        for s, ix in idx.items():
            out = [binned(parts_all[ix[b:b + 256]].float()) for b in range(0, len(ix), 256)]
            Q[s] = (torch.cat([o[0] for o in out]), torch.cat([o[1] for o in out]))
        # targets: pz bin medians (as increments, log-positive -> monotone), z bin medians
        def tgt(q):
            qp, qz = q
            inc = torch.diff(qp, dim=1).clamp_min(1e-6).log()
            return torch.cat([qp[:, :1], inc, qz[..., 0]], 1)
        Ytr = tgt(Q["train"])
        mu, sd = Ytr.mean(0), Ytr.std(0).clamp_min(1e-6)
        dev = a.device
        net = nn.Sequential(nn.Linear(ds.knobs.shape[1], a.hidden), nn.SiLU(), nn.Linear(a.hidden, a.hidden), nn.SiLU(),
                            nn.Linear(a.hidden, a.hidden), nn.SiLU(), nn.Linear(a.hidden, Ytr.shape[1])).to(dev)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs, eta_min=1e-6)
        xt, yt = ds.knobs[idx["train"]].float().to(dev), ((Ytr - mu) / sd).to(dev)
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
                p = net(ds.knobs[ix].float().to(dev)).cpu() * sd + mu
            qp = torch.cat([p[:, :1], p[:, :1] + torch.cumsum(p[:, 1:NB].exp(), 1)], 1)
            qz = p[:, NB:]
            ml, orc = [], []
            for b in range(0, len(ix), 256):
                pp = parts_all[ix[b:b + 256]].float()
                sl = slice(b, b + 256)
                ml.append(frac(pp, qp[sl], qz[sl]))
                orc.append(frac(pp, Q[s][0][sl], Q[s][1][sl][..., 0]))
            ml, orc = torch.cat(ml), torch.cat(orc)
            ent[s] = {"n": int(len(ix)), "oracle_median": float(orc.median()), "mlp_median": float(ml.median()),
                      "mlp_p90": float(ml.quantile(0.9))}
        summ[name] = ent
        print(name, json.dumps(ent), flush=True)
    out = Path("results/2_surrogate_screen/curve_learnability_2026-10-20")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"summary_quantile_{Path(a.table).stem}.json").write_text(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
