"""Pair metrics (injector effect on spacing: point minus golden reference at identical linac knobs) on TEST and VAL
separately, with a CLUSTER bootstrap over injector points (pairs at one point are correlated; review 2026-10-18 issue
2). B2-cov (sampled clouds, frozen base sample, n = 2048 as the scorecard) vs B2-cov + spacing head (deterministic).
Val is the selection set of the spacing head (early stopping) -> labelled as such. CPU."""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

T = "processed/xtcav23e_inj_cov_2026-10-17.h5"
M = {"B2-cov": "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",
     "B2-cov + spacing head": next(Path("trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints").glob("*.ckpt"))}
OUT = Path("results/2_surrogate_screen/pair_cluster_bootstrap_2026-10-18"); OUT.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)
ck = torch.load(M["B2-cov"], map_location="cpu", weights_only=False)
scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
roles = json.loads(Path("processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
dm = TwoBunchFlowDataModule(processed_h5=T, split="group", orig_split=roles, scaler=scaler); dm.setup()
ds = dm.full
role = np.full(len(ds), "train", object); role[dm.val.indices] = "val"; role[dm.test.indices] = "test"
with h5py.File(T) as h:
    key, pref, grp = h["pair_key"][...], h["pair_ref"][...], h["group"][...]
both = (ds.drive_density & ds.witness_density).numpy().astype(bool)
md, sd = np.array(scaler["drive_mean"]), np.array(scaler["drive_std"])
mw, sw = np.array(scaler["witness_mean"]), np.array(scaler["witness_std"])
sp_true = (ds.drive.mean(1)[:, 2].numpy() * sd[2] + md[2]) - (ds.witness.mean(1)[:, 2].numpy() * sw[2] + mw[2])
by = {}
for i in np.flatnonzero((key >= 0) & both):
    by.setdefault(int(key[i]), {})["ref" if pref[i] else "pt"] = i
pairs = np.array([(v["pt"], v["ref"]) for v in by.values() if len(v) == 2])
zd, zw = torch.randn(2048, 6), torch.randn(2048, 6)
res = {}
for name, path in M.items():
    m = TwoBunchFlow.load_from_checkpoint(path, map_location="cpu").eval()
    pred = {}
    with torch.no_grad():
        for r in ("val", "test"):
            pp = pairs[role[pairs[:, 0]] == r]
            rows = np.unique(pp.ravel())
            sp = []
            for b in range(0, len(rows), 32):
                kn = ds.knobs[rows[b:b + 32]]
                sp.append(m.observables(kn, n=2048, z_drive=zd, z_witness=zw)["bunch_spacing"].numpy())
            pred.update(dict(zip(rows.tolist(), np.concatenate(sp).tolist())))
    for r in ("val", "test"):
        pp = pairs[role[pairs[:, 0]] == r]
        dt = sp_true[pp[:, 0]] - sp_true[pp[:, 1]]
        dp = np.array([pred[a] - pred[b] for a, b in pp])
        g = grp[pp[:, 0]]
        ug = np.unique(g)
        def stats(idx):
            e = dp[idx] - dt[idx]
            return (1 - np.var(e) / np.var(dt[idx]), np.polyfit(dt[idx], dp[idx], 1)[0], np.percentile(np.abs(e), 90) * 1e6)
        point = stats(np.arange(len(pp)))
        rng = np.random.default_rng(1)
        bs = []
        members = {u: np.flatnonzero(g == u) for u in ug}
        for _ in range(2000):
            pick = rng.choice(ug, len(ug), replace=True)
            bs.append(stats(np.concatenate([members[u] for u in pick])))
        bs = np.array(bs)
        lo, hi = np.percentile(bs, 5, 0), np.percentile(bs, 95, 0)
        res[f"{name} | {r}"] = {"n_pairs": int(len(pp)), "n_points": int(len(ug)), "R2": point[0], "slope": point[1], "p90_um": point[2],
                                "ci90_cluster": {"R2": [lo[0], hi[0]], "slope": [lo[1], hi[1]], "p90_um": [lo[2], hi[2]]}}
        print(f"{name:24s} {r:4s} n={len(pp)} pairs / {len(ug)} points: R2 {point[0]:.2f} [{lo[0]:.2f},{hi[0]:.2f}]  slope {point[1]:.2f} [{lo[1]:.2f},{hi[1]:.2f}]  p90 {point[2]:.1f} [{lo[2]:.1f},{hi[2]:.1f}] um", flush=True)
(OUT / "pairs_cluster_bootstrap.json").write_text(json.dumps(res, indent=1, default=float))
