"""Model selection on the VALIDATION split only (2026-10-21, reviewer protocol), with full-particle truth:
  spacing MAE, injector-pair delta rms (val pairs), drive log-sigma_z error by sigma_z band (< 30, 30-60, > 60 um),
  per-row sliced-Wasserstein-2 on (z, pz) in the row's own data-standardized frame (drive; vs the 4096-particle table
  subsample, 4096 samples, 64 directions; the floor = two independent 4096-subsamples of Bmad is reported as 'sw_floor'),
  and (--sweeps) sigma_z(L1) on the val-role compression sweeps (processed/compress_sweeps table + truth).
Common random numbers: the same sampling seeds for every model. Test rows are never touched.
Appends one line per model to results/2_surrogate_screen/val_select_2026-10-21/summary.txt (+ <tag>.json).

  PYTHONPATH=src python scripts/surrogate_val_select_2026-10-21.py --tag <tag> --ckpt <ckpt> [--sweeps <table>] [--device cpu]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/2_surrogate_screen/val_select_2026-10-21"
TABLE = "processed/xtcav23e_inj_cov_P4096_2026-10-20.h5"
TRUTH = "processed/xtcav23e_inj_cov_P4096_2026-10-20_truth.h5"
B2 = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"


def sw2(a, b, n_dir=64, seed=0):
    """(M, P, 2) data, (M, P, 2) model in the data's per-row standardized frame -> (M,) sliced W2^2."""
    mu, sd = a.mean(1, keepdim=True), a.std(1, keepdim=True).clamp_min(1e-9)
    a, b = (a - mu) / sd, (b - mu) / sd
    th = torch.linspace(0, np.pi, n_dir + 1)[:-1]
    d = torch.stack([torch.cos(th), torch.sin(th)], 0).to(a.dtype)
    pa, pb = (a @ d).sort(1).values, (b @ d).sort(1).values
    return ((pa - pb) ** 2).mean((1, 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sweeps", default=None, help="processed table of the compression sweeps (with _truth.h5 beside it)")
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = a.device
    m = TwoBunchFlow.load_from_checkpoint(str(REPO / a.ckpt), map_location=dev).eval()
    ck = torch.load(str(REPO / B2), map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}            # one frame for the data side
    roles = json.loads((REPO / "processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=str(REPO / TABLE), split="group", orig_split=roles, scaler=scaler,
                                truth_h5=str(REPO / TRUTH))
    dm.setup()
    ds = dm.full
    ok = (ds.drive_density & ds.witness_density).numpy().astype(bool) & np.isfinite(ds.truth_spacing.numpy())
    val = np.array([i for i in dm.val.indices if ok[i]])
    gen = lambda s: torch.Generator(device=dev).manual_seed(s)  # noqa: E731

    @torch.no_grad()
    def predict(rows, n):
        sp, sz, dclouds = [], [], []
        kn = ds.knobs[rows].float().to(dev)
        for b in range(0, len(rows), 32):
            g = gen(1000 + b)
            d = m.sample_bunch(kn[b:b + 32], 0, n, generator=g)
            w = m.sample_bunch(kn[b:b + 32], 1, n, generator=g)
            sp.append(inter_bunch(d, w)["bunch_spacing"].double().cpu())
            sz.append(per_bunch(d)["sigma_z"].double().cpu())
        return torch.cat(sp).numpy(), torch.cat(sz).numpy()

    sp_p, sz_p = predict(val, a.n)
    sp_t, sz_t = ds.truth_spacing[val].numpy(), ds.truth_drive_sigma_z[val].double().numpy()
    res = {"tag": a.tag, "ckpt": a.ckpt, "n_val": int(len(val))}
    res["spacing_mae_um"] = float(np.mean(np.abs(sp_p - sp_t)) * 1e6)
    lsz = np.abs(np.log(sz_p / sz_t))
    res["logsz_median"] = float(np.median(lsz))
    for lo, hi in ((0, 30e-6), (30e-6, 60e-6), (60e-6, 1)):
        sel = (sz_t >= lo) & (sz_t < hi)
        res[f"logsz_median_{int(lo * 1e6)}_{int(hi * 1e6) if hi < 1 else 'inf'}"] = float(np.median(lsz[sel])) if sel.any() else None
    # pairs among val rows
    import h5py
    with h5py.File(REPO / TABLE) as h:
        key, ref = h["pair_key"][...], h["pair_ref"][...]
    pos = {int(i): j for j, i in enumerate(val)}
    by = {}
    for i in val:
        if key[i] >= 0:
            by.setdefault(int(key[i]), {})["ref" if ref[i] else "pt"] = int(i)
    pairs = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2]
    if pairs:
        dp = np.array([sp_p[pos[p]] - sp_p[pos[r]] for p, r in pairs])
        dt = np.array([sp_t[pos[p]] - sp_t[pos[r]] for p, r in pairs])
        res["pair_delta_rms_um"] = float(np.sqrt(np.mean((dp - dt) ** 2)) * 1e6)
        res["n_pairs"] = len(pairs)
    # per-row SW2 on the drive (z, pz) vs the 4096-particle table subsample, and the two-subsample floor
    sub = val[np.random.default_rng(0).permutation(len(val))[:300]]
    with torch.no_grad():
        kn = ds.knobs[sub].float().to(dev)
        sws, floor = [], []
        for b in range(0, len(sub), 32):
            data = ds.drive[sub[b:b + 32]].double()[..., [2, 5]]
            P = data.shape[1]
            mod = m.sample_bunch(kn[b:b + 32], 0, P, physical=False, generator=gen(5000 + b)).double().cpu()
            # the model's standardized frame may differ from the data's (warm starts keep B2-cov's): to physical, then data frame
            mm, msd = m._scaler(0)
            phys = mod * msd.double().cpu() + mm.double().cpu()
            dmean, dstd = torch.tensor(scaler["drive_mean"]).double(), torch.tensor(scaler["drive_std"]).double()
            mod = ((phys - dmean) / dstd)[..., [2, 5]]
            sws.append(sw2(data, mod))
            half = P // 2
            floor.append(sw2(data[:, :half], data[:, half:]))
        res["sw2_drive_median"] = float(torch.cat(sws).median())
        res["sw_floor_half_subsample"] = float(torch.cat(floor).median())
    if a.sweeps:
        # val-role compression sweeps (merged table + its truth): sigma_z(L1) along each L1 scan
        sdm = TwoBunchFlowDataModule(processed_h5=str(REPO / a.sweeps), split="group", orig_split=roles, scaler=scaler,
                                     truth_h5=str(REPO / a.sweeps).replace(".h5", "_truth.h5"))
        sdm.setup()
        sds = sdm.full
        with h5py.File(REPO / a.sweeps) as h:
            names = h.attrs["campaigns"].split(",")
            ci = names.index("compress_sweeps_2026-10-21")
            camp = h["campaign"][...]
        okk = (sds.drive_density & sds.witness_density).numpy().astype(bool) & np.isfinite(sds.truth_drive_sigma_z.numpy())
        srows = np.array([i for i in sdm.val.indices if camp[i] == ci and okk[i]])
        if len(srows):
            kn_all = sds.knobs[srows].float()
            # sweep id: all knobs but L1 equal
            keys = list(m.hparams.knob_keys)
            j1 = keys.index("L1PhaseSet")
            other = np.round(np.delete(kn_all.numpy(), j1, 1), 6)
            _, sid = np.unique(other, axis=0, return_inverse=True)
            with torch.no_grad():
                szp, spp = [], []
                for b in range(0, len(srows), 32):
                    g = gen(9000 + b)
                    d = m.sample_bunch(kn_all[b:b + 32].to(dev), 0, a.n, generator=g)
                    w = m.sample_bunch(kn_all[b:b + 32].to(dev), 1, a.n, generator=g)
                    szp.append(per_bunch(d)["sigma_z"].double().cpu()); spp.append(inter_bunch(d, w)["bunch_spacing"].double().cpu())
            szp, spp = torch.cat(szp).numpy(), torch.cat(spp).numpy()
            szt, spt = sds.truth_drive_sigma_z[srows].double().numpy(), sds.truth_spacing[srows].numpy()
            l1 = kn_all[:, j1].numpy()
            dloc, dmin = [], []
            for s_ in np.unique(sid):
                sel = sid == s_
                if sel.sum() < 5:
                    continue
                o = np.argsort(l1[sel])
                lp, tp, tt = l1[sel][o], szp[sel][o], szt[sel][o]
                lo_, hi_ = float(m.knob_low[j1]), float(m.knob_high[j1])
                dloc.append(abs(lp[np.argmin(tp)] - lp[np.argmin(tt)]) * (hi_ - lo_))
                dmin.append(abs(np.log(tp.min() / tt.min())))
            res["sweeps"] = {"n_rows": int(len(srows)), "n_scans": len(dloc),
                             "logsz_median": float(np.median(np.abs(np.log(szp / szt)))),
                             "logsz_median_lt30": float(np.median(np.abs(np.log(szp / szt))[szt < 30e-6])) if (szt < 30e-6).any() else None,
                             "sz_min_location_err_deg_median": float(np.median(dloc)) if dloc else None,
                             "sz_min_value_logerr_median": float(np.median(dmin)) if dmin else None,
                             "spacing_mae_um": float(np.mean(np.abs(spp - spt)) * 1e6)}
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    line = (f"{a.tag:22s} val n={res['n_val']}: spacing MAE {res['spacing_mae_um']:.2f} um | pair d rms "
            f"{res.get('pair_delta_rms_um', float('nan')):.2f} um (n={res.get('n_pairs', 0)}) | log sz median {res['logsz_median']:.3f} "
            f"(<30 {res['logsz_median_0_30']}, 30-60 {res['logsz_median_30_60']}, >60 {res['logsz_median_60_inf']}) | "
            f"SW2 drive (z,pz) {res['sw2_drive_median']:.4f} (floor {res['sw_floor_half_subsample']:.4f})"
            + (f" | sweeps: log sz {res['sweeps']['logsz_median']:.3f} (<30 {res['sweeps']['logsz_median_lt30']}), "
               f"sz-min loc err {res['sweeps']['sz_min_location_err_deg_median']} deg, min value log err {res['sweeps']['sz_min_value_logerr_median']}, "
               f"spacing MAE {res['sweeps']['spacing_mae_um']:.2f} um" if "sweeps" in res else ""))
    with open(OUT / "summary.txt", "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
