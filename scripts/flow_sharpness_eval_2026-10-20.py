"""Score flow checkpoints for LPS sharpness (2026-10-20): per checkpoint,
  1. thinness at the 12 gradient-check Bmad beams (results/2_surrogate_screen/gradient_check_2026-10-18): conditional
     spread of b given a (median over slices of a, / projected robust spread of b) for the thin pairs pz|z, z|pz,
     x|pz, z|x (drive) and pz|z (witness), and the drive slice energy spread; reported as surrogate / Bmad ratios
     (1 = as sharp as Bmad);
  2. knob response at those points: d spacing / dL1,L2 and d ln sigma_z / dL1 by central differences (+-0.3 deg,
     common random numbers, 20k particles) vs Bmad's (same points, same h);
  3. beam accuracy on the 688 held-out test rows (Bmad truth from results/2_surrogate_screen/screen_features_check_2026-10-18):
     spacing MAE, drive / witness sigma_z median |rel| error.
  --screen: also the screen-level spacing agreement with Bmad by band (wide window; ~10 min / checkpoint, CPU).
Outputs results/2_surrogate_screen/flow_sharpness_2026-10-20/<tag>.json and summary.txt.

  PYTHONPATH=src python scripts/flow_sharpness_eval_2026-10-20.py --tag B2-cov --ckpt <ckpt> [--screen]
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
G = REPO / "results/2_surrogate_screen/gradient_check_2026-10-18"
SF = REPO / "results/2_surrogate_screen/screen_features_check_2026-10-18"
PAIRS = {"drive pz|z": (0, 2, 5), "drive z|pz": (0, 5, 2), "drive x|pz": (0, 5, 0), "drive z|x": (0, 0, 2),
         "witness pz|z": (1, 2, 5)}


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rs(v):
    return 1.4826 * np.median(np.abs(v - np.median(v)))


def thin(a, b, nb=60):
    lo, hi = np.quantile(a, [0.02, 0.98])
    e = np.linspace(lo, hi, nb + 1)
    idx = np.digitize(a, e)
    s = [rs(b[idx == i]) for i in range(1, nb + 1) if (idx == i).sum() >= 40]
    return float(np.median(s) / rs(b)) if s else np.nan


def slice_spread(z, d, dz=2e-6):
    lo, hi = np.quantile(z, [0.1, 0.9])
    e = np.arange(lo, hi + dz, dz)
    idx = np.digitize(z, e)
    s = [rs(d[idx == i]) for i in range(1, len(e)) if (idx == i).sum() >= 30]
    return float(np.median(s)) if s else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--screen", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
    nf = _load("nf", REPO / "scripts/notebook_figs_2026-10-20.py")
    gc = _load("gc", REPO / "scripts/gradient_check_2026-10-18.py")
    m = TwoBunchFlow.load_from_checkpoint(a.ckpt, map_location=a.device).eval()
    env = gc.make_env(gc.FLOWS["B2-cov"], 1024)                  # knob frame only
    B = np.load(G / "bmad.npz")
    X, lab = B["X"], B["lab"]
    res = {"tag": a.tag, "ckpt": a.ckpt}

    def sample(kn, n, seed):
        g = torch.Generator(device=a.device).manual_seed(seed)
        with torch.no_grad():
            return (m.sample_bunch(kn, 0, n, generator=g)[0].cpu().numpy(),
                    m.sample_bunch(kn, 1, n, generator=g)[0].cpu().numpy())

    # 1. thinness at the 12 centre points
    ratios = {k: [] for k in PAIRS}
    ratios["drive slice dE"] = []
    for it, il in itertools.product(range(3), range(4)):
        i = int(np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0])
        c, q, isd = nf._bmad_cloud(G / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5")
        c = c.numpy()
        kn = env._ks.normalize(torch.as_tensor(X[i:i + 1], dtype=torch.float32)).to(a.device)
        d, w = sample(kn, 40000, 0)
        for key, (bunch, ia, ib) in PAIRS.items():
            bm = c[isd] if bunch == 0 else c[~isd]
            sm = d if bunch == 0 else w
            ratios[key].append(thin(sm[:, ia], sm[:, ib]) / thin(bm[:, ia], bm[:, ib]))
        p0, ps = c[isd, 5].mean(), d[:, 5].mean()
        ratios["drive slice dE"].append(slice_spread(d[:, 2], d[:, 5] / ps - 1) / slice_spread(c[isd, 2], c[isd, 5] / p0 - 1))
    comp = [4 * it + 1 for it in range(3)]                       # the compressed linac point per injector state
    res["thinness_ratio_median"] = {k: float(np.median(v)) for k, v in ratios.items()}
    res["thinness_ratio_compressed"] = {k: float(np.median([v[j] for j in comp])) for k, v in ratios.items()}

    # 2. knob response (central differences, CRN) vs Bmad
    def feats(i, seed=7, n=20000):
        kn = env._ks.normalize(torch.as_tensor(X[i:i + 1], dtype=torch.float32)).to(a.device)
        d, w = sample(kn, n, seed)
        dt, wt = torch.as_tensor(d)[None], torch.as_tensor(w)[None]
        o = per_bunch(dt)
        return float(inter_bunch(dt, wt)["bunch_spacing"][0]), float(o["sigma_z"][0])
    sl = {"spacing_L1": [], "spacing_L2": [], "lnsz_L1": [], "lnsz_L1_compressed": []}
    for it, il in itertools.product(range(3), range(4)):
        rows = {}
        for c_, s_ in ((0, 1), (0, -1), (1, 1), (1, -1)):
            rows[(c_, s_)] = int(np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] == c_) & (lab[:, 3] == 0.3)
                                                & (lab[:, 4] == s_))[0])
        ic = int(np.flatnonzero((lab[:, 0] == it) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0])
        f = {kk: feats(v) for kk, v in rows.items()}
        sz0 = feats(ic)[1]
        bf = {kk: (B["bunch_spacing_cloud"][v], B["drive_sigma_z_cloud"][v]) for kk, v in rows.items()}
        bsz0 = B["drive_sigma_z_cloud"][ic]
        for c_, key in ((0, "spacing_L1"), (1, "spacing_L2")):
            s_ = (f[(c_, 1)][0] - f[(c_, -1)][0]) / 0.6
            b_ = (bf[(c_, 1)][0] - bf[(c_, -1)][0]) / 0.6
            sl[key].append(s_ / b_)
        s_ = (f[(0, 1)][1] - f[(0, -1)][1]) / 0.6 / sz0
        b_ = (bf[(0, 1)][1] - bf[(0, -1)][1]) / 0.6 / bsz0
        (sl["lnsz_L1_compressed"] if il == 1 else sl["lnsz_L1"]).append(s_ / b_)
    res["slope_ratio_median"] = {k: float(np.median(v)) for k, v in sl.items()}
    res["slope_ratio_all"] = {k: [round(float(x), 3) for x in v] for k, v in sl.items()}

    # 3. beam accuracy on the test rows
    k = np.load(SF / "rows.npz")
    kn = torch.as_tensor(k["knobs_norm"], dtype=torch.float32).to(a.device)
    sp, szd, szw = [], [], []
    for b0 in range(0, len(kn), 64):
        g = torch.Generator(device=a.device).manual_seed(b0)
        with torch.no_grad():
            dd = m.sample_bunch(kn[b0:b0 + 64], 0, 4096, generator=g)
            ww = m.sample_bunch(kn[b0:b0 + 64], 1, 4096, generator=g)
        sp.append(inter_bunch(dd, ww)["bunch_spacing"].cpu().numpy())
        szd.append(per_bunch(dd)["sigma_z"].cpu().numpy())
        szw.append(per_bunch(ww)["sigma_z"].cpu().numpy())
    sp, szd = np.concatenate(sp), np.concatenate(szd)
    tb, tz = k["Bmad|bunch_spacing_cloud"], k["Bmad|drive_sigma_z_cloud"]
    res["test_spacing_mae_um"] = float(np.mean(np.abs(sp - tb)) * 1e6)
    res["test_drive_sigz_rel_median"] = float(np.median(np.abs(szd / tz - 1)))
    comp_rows = tz < 30e-6
    res["test_drive_sigz_rel_median_sigz_lt30"] = float(np.median(np.abs(szd[comp_rows] / tz[comp_rows] - 1)))

    # 4. screen agreement (optional)
    if a.screen:
        cw = _load("cw", REPO / "scripts/crop_window_check_2026-10-19.py")
        sfc = cw.sfc
        import h5py
        from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
        from twobunch_s2e_rl.surrogate.train import SCALER_KEYS
        rows = k["rows"]
        r = np.load(REPO / "results/2_surrogate_screen/crop_window_2026-10-19/rows.npz")
        Bsp, Bv = r["none|Bmad|bunch_spacing"], r["none|Bmad|screen_valid_deployed"] > 0.5
        plant = cw.plant_for(a.ckpt, None)
        acc, val = [], []
        torch.manual_seed(0)
        for b0 in range(0, len(rows), 24):
            with torch.no_grad():
                o = plant(torch.as_tensor(k["knobs_norm"][b0:b0 + 24], dtype=torch.float32))
            acc.append(o["bunch_spacing"].numpy())
            val.append(o["screen_valid_deployed"].numpy() > 0.5)
        S, V = np.concatenate(acc), np.concatenate(val)
        t = np.abs(r["x1|Bmad|bunch_spacing_cloud"]) * 1e6
        res["screen"] = {}
        for lo, hi in ((60, 150), (150, 1e9)):
            sel = (t >= lo) & (t < hi)
            ok = sel & Bv & V
            res["screen"][f"{lo}-{hi}"] = {"rel_median": float(np.median(np.abs(S[ok] / Bsp[ok] - 1))),
                                           "within10pct": float(np.mean(np.abs(S[ok] / Bsp[ok] - 1) <= 0.1)),
                                           "valid_agree": float(np.mean(Bv[sel] == V[sel]))}
    (OUT / f"{a.tag}.json").write_text(json.dumps(res, indent=1))
    line = (f"{a.tag:10s} thin(median, surr/Bmad): " + " ".join(f"{kk} {v:.1f}" for kk, v in res["thinness_ratio_median"].items())
            + f" | compressed pz|z {res['thinness_ratio_compressed']['drive pz|z']:.1f}"
            + " | slopes: " + " ".join(f"{kk} {v:.2f}" for kk, v in res["slope_ratio_median"].items())
            + f" | test spacing MAE {res['test_spacing_mae_um']:.1f} um, sigz rel {res['test_drive_sigz_rel_median']:.3f}"
            + f" (<30 um {res['test_drive_sigz_rel_median_sigz_lt30']:.3f})")
    if "screen" in res:
        line += " | screen " + " ".join(f"{kk}: {v['rel_median'] * 100:.1f}% / in10% {v['within10pct']:.2f} / agree {v['valid_agree']:.2f}"
                                       for kk, v in res["screen"].items())
    with open(OUT / "summary.txt", "a") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


if __name__ == "__main__":
    main()
