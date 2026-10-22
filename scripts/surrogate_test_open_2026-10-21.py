"""One-time TEST-split opening for the phase-C finalists (2026-10-21), full-particle truth, with a cluster bootstrap over
injector points (90% CIs). Same sampling as scripts/surrogate_val_select_2026-10-21.py (common random numbers, 16384
particles), on the test rows of the 4096-particle table: spacing MAE, drive log-sigma_z median (all / < 30 um),
pair delta rms, pair R2 and slope. Writes results/2_surrogate_screen/test_open_2026-10-21/test_open.json.

  PYTHONPATH=src python scripts/surrogate_test_open_2026-10-21.py
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/2_surrogate_screen/test_open_2026-10-21"
TABLE = "processed/xtcav23e_inj_cov_P4096_2026-10-20.h5"
B2 = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
M = {"B2-cov": B2,
     "covB2zs": str(next((REPO / "trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints").glob("*.ckpt")).relative_to(REPO)),
     "C_B2_ema": "trained/xtcav23e_inj_covB2_C_B2_2026-10-21/checkpoints/final_ema.ckpt",
     "C_B3_ema": "trained/xtcav23e_inj_covB2_C_B3_2026-10-21/checkpoints/final_ema.ckpt"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = torch.load(str(REPO / B2), map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads((REPO / "processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=str(REPO / TABLE), split="group", orig_split=roles, scaler=scaler,
                                truth_h5=str(REPO / TABLE).replace(".h5", "_truth.h5"))
    dm.setup()
    ds = dm.full
    ok = (ds.drive_density & ds.witness_density).numpy().astype(bool) & np.isfinite(ds.truth_spacing.numpy())
    test = np.array([i for i in dm.test.indices if ok[i]])
    with h5py.File(REPO / TABLE) as h:
        key, ref, grp = h["pair_key"][...], h["pair_ref"][...], h["group"][...]
    pos = {int(i): j for j, i in enumerate(test)}
    by = {}
    for i in test:
        if key[i] >= 0:
            by.setdefault(int(key[i]), {})["ref" if ref[i] else "pt"] = int(i)
    pairs = np.array([(d["pt"], d["ref"]) for d in by.values() if len(d) == 2])
    sp_t, sz_t = ds.truth_spacing[test].numpy(), ds.truth_drive_sigma_z[test].double().numpy()
    dt = np.array([sp_t[pos[p]] - sp_t[pos[r]] for p, r in pairs])
    g_rows, g_pairs = grp[test], grp[pairs[:, 0]]
    rng = np.random.default_rng(1)
    boots = [rng.choice(np.unique(g_rows), len(np.unique(g_rows)), replace=True) for _ in range(2000)]
    res = {"n_test": int(len(test)), "n_pairs": int(len(pairs)), "n_points": int(len(np.unique(g_pairs)))}
    for name, path in M.items():
        m = TwoBunchFlow.load_from_checkpoint(str(REPO / path), map_location="cpu").eval()
        sp, sz = [], []
        kn = ds.knobs[test].float()
        with torch.no_grad():
            for b in range(0, len(test), 32):
                gen = torch.Generator().manual_seed(1000 + b)
                d = m.sample_bunch(kn[b:b + 32], 0, 16384, generator=gen)
                w = m.sample_bunch(kn[b:b + 32], 1, 16384, generator=gen)
                sp.append(inter_bunch(d, w)["bunch_spacing"].double().numpy())
                sz.append(per_bunch(d)["sigma_z"].double().numpy())
        sp_p, sz_p = np.concatenate(sp), np.concatenate(sz)
        dp = np.array([sp_p[pos[p]] - sp_p[pos[r]] for p, r in pairs])
        lsz = np.abs(np.log(sz_p / sz_t))

        def stats(rsel, psel):
            e = dp[psel] - dt[psel]
            return np.array([np.mean(np.abs(sp_p[rsel] - sp_t[rsel])) * 1e6, np.median(lsz[rsel]),
                             np.median(lsz[rsel & (sz_t < 30e-6)]), np.sqrt(np.mean(e ** 2)) * 1e6,
                             1 - np.var(e) / np.var(dt[psel]), np.polyfit(dt[psel], dp[psel], 1)[0]])
        names = ["spacing_mae_um", "logsz_median", "logsz_median_lt30", "pair_delta_rms_um", "pair_R2", "pair_slope"]
        pt = stats(np.ones(len(test), bool), np.ones(len(pairs), bool))
        bs = []
        for pick in boots:
            cnt_r = np.zeros(len(test)); cnt_p = np.zeros(len(pairs))
            for u, c in zip(*np.unique(pick, return_counts=True)):
                cnt_r[g_rows == u] = c; cnt_p[g_pairs == u] = c
            rs, ps = np.repeat(np.arange(len(test)), cnt_r.astype(int)), np.repeat(np.arange(len(pairs)), cnt_p.astype(int))
            if len(ps) < 3:
                continue
            e = dp[ps] - dt[ps]
            bs.append([np.mean(np.abs(sp_p[rs] - sp_t[rs])) * 1e6, np.median(lsz[rs]),
                       np.median(lsz[rs][sz_t[rs] < 30e-6]), np.sqrt(np.mean(e ** 2)) * 1e6,
                       1 - np.var(e) / np.var(dt[ps]), np.polyfit(dt[ps], dp[ps], 1)[0]])
        bs = np.array(bs)
        lo, hi = np.percentile(bs, 5, 0), np.percentile(bs, 95, 0)
        res[name] = {k: {"point": float(pt[j]), "ci90": [float(lo[j]), float(hi[j])]} for j, k in enumerate(names)}
        print(f"{name:10s} " + " | ".join(f"{k} {pt[j]:.3f} [{lo[j]:.3f},{hi[j]:.3f}]" for j, k in enumerate(names)), flush=True)
        # paired difference vs B2-cov on pair delta rms / spacing MAE would need per-model stored preds; keep preds
        res[name]["_pred"] = {"sp": sp_p.tolist(), "sz": sz_p.tolist()}
    (OUT / "test_open.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
