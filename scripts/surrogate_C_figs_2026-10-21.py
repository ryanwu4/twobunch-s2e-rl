"""Example figures for the 10-21 surrogate result (B2-cov vs covB2zs vs C_B3_ema):
  spacing_pairs.png   test-split injector effect on spacing: predicted vs Bmad pair delta (point minus golden reference at
                      identical linac knobs), and per-row spacing error; predictions from results/.../test_open.json
  thinness_lps.png    drive + witness LPS at XTCAVF at a normal (~220 um) and a compressed (~70 um) gradient-check point:
                      Bmad vs B2-cov vs C_B3_ema (same seed, 30k particles), plus the drive slice energy spread along z
                      (robust std of dE/E in 2 um slices) -- the "thinness" ratio is surrogate / Bmad of its median
Output results/2_surrogate_screen/test_open_2026-10-21/figs/. CPU.

  PYTHONPATH=src python scripts/surrogate_C_figs_2026-10-21.py
"""
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/2_surrogate_screen/test_open_2026-10-21/figs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nf = _load("nf", REPO / "scripts/notebook_figs_2026-10-20.py")
gc = _load("gc", REPO / "scripts/gradient_check_2026-10-18.py")
fe = _load("fe", REPO / "scripts/flow_sharpness_eval_2026-10-20.py")
to = _load("to", REPO / "scripts/surrogate_test_open_2026-10-21.py")
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402
from twobunch_s2e_rl.surrogate.train import SCALER_KEYS  # noqa: E402

COL = {"Bmad": "k", "B2-cov": "#8c8c8c", "covB2zs": "#b07aa1", "C_B3_ema": nf.GREEN}
LABEL = {"B2-cov": "B2-cov (old surrogate)", "covB2zs": "covB2zs (incumbent controller's)", "C_B3_ema": "C_B3_ema (new)"}


def spacing_fig():
    res = json.loads((REPO / "results/2_surrogate_screen/test_open_2026-10-21/test_open.json").read_text())
    ck = torch.load(str(REPO / to.B2), map_location="cpu", weights_only=False)
    scaler = {k: ck["state_dict"][k].tolist() for k in SCALER_KEYS}
    roles = json.loads((REPO / "processed/role_map_ftc_s0_2026-10-16.json").read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=str(REPO / to.TABLE), split="group", orig_split=roles, scaler=scaler,
                                truth_h5=str(REPO / to.TABLE).replace(".h5", "_truth.h5"))
    dm.setup()
    ds = dm.full
    ok = (ds.drive_density & ds.witness_density).numpy().astype(bool) & np.isfinite(ds.truth_spacing.numpy())
    test = np.array([i for i in dm.test.indices if ok[i]])
    assert len(test) == res["n_test"]
    with h5py.File(REPO / to.TABLE) as h:
        key, ref = h["pair_key"][...], h["pair_ref"][...]
    pos = {int(i): j for j, i in enumerate(test)}
    by = {}
    for i in test:
        if key[i] >= 0:
            by.setdefault(int(key[i]), {})["ref" if ref[i] else "pt"] = int(i)
    pairs = [(pos[d["pt"]], pos[d["ref"]]) for d in by.values() if len(d) == 2]
    p, r = np.array(pairs).T
    sp_t = ds.truth_spacing[test].numpy() * 1e6
    dt = sp_t[p] - sp_t[r]
    names = ["B2-cov", "covB2zs", "C_B3_ema"]
    fig, axs = plt.subplots(1, 4, figsize=(16, 4.1))
    lim = np.abs(dt).max() * 1.1
    for a, n in zip(axs, names):
        sp = np.array(res[n]["_pred"]["sp"]) * 1e6
        dp = sp[p] - sp[r]
        st = res[n]
        a.plot([-lim, lim], [-lim, lim], color="0.75", lw=1, zorder=0)
        a.scatter(dt, dp, s=10, color=COL[n], alpha=0.75, edgecolors="none")
        a.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xlabel="Bmad pair Δ spacing [µm]",
              ylabel="surrogate pair Δ spacing [µm]", title=LABEL[n])
        a.text(0.03, 0.97, f"rms error {st['pair_delta_rms_um']['point']:.2f} µm\nslope {st['pair_slope']['point']:.2f}, "
               f"R² {st['pair_R2']['point']:.2f}", transform=a.transAxes, va="top", fontsize=8)
    a = axs[3]
    bins = np.linspace(-12, 12, 49)
    for n in names:
        e = np.array(res[n]["_pred"]["sp"]) * 1e6 - sp_t
        a.hist(e, bins=bins, histtype="step", lw=1.6, color=COL[n],
               label=f"{n}: MAE {res[n]['spacing_mae_um']['point']:.2f} µm")
    a.set(xlabel="spacing error, surrogate − Bmad [µm]", ylabel="test rows", title="per-row spacing error")
    a.legend(frameon=False, fontsize=7.5)
    fig.suptitle(f"Test split ({res['n_test']} rows, {res['n_pairs']} injector pairs over {res['n_points']} points): pair Δ = "
                 "spacing at an injector point − spacing at the golden injector, same linac knobs", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "spacing_pairs.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def slice_profile(z, d, dz=2.0):
    lo, hi = np.quantile(z, [0.02, 0.98])
    e = np.arange(lo, hi + dz, dz)
    idx = np.digitize(z, e)
    c, s = [], []
    for i in range(1, len(e)):
        if (idx == i).sum() >= 30:
            c.append(0.5 * (e[i - 1] + e[i])); s.append(fe.rs(d[idx == i]))
    return np.array(c), np.array(s)


def thinness_fig():
    env = gc.make_env(gc.FLOWS["B2-cov"], 1024)
    flows = {n: TwoBunchFlow.load_from_checkpoint(str(REPO / to.M[n]), map_location="cpu").eval() for n in ("B2-cov", "C_B3_ema")}
    B = np.load(fe.G / "bmad.npz")
    lab = B["lab"]
    N = 30000
    fig, axs = plt.subplots(2, 4, figsize=(16, 7.4))
    for row, (pname, il) in enumerate((("normal (~220 µm)", 0), ("compressed (~70 µm)", 1))):
        i = int(np.flatnonzero((lab[:, 0] == 0) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0])
        cb, qb, isd = nf._bmad_cloud(fe.G / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5")
        kn = env._ks.normalize(torch.as_tensor(B["X"][i:i + 1], dtype=torch.float32))
        clouds = {"Bmad": (cb, isd)}
        for n, m in flows.items():
            g = torch.Generator().manual_seed(3)
            with torch.no_grad():
                d, w = m.sample_bunch(kn, 0, N, generator=g)[0].double(), m.sample_bunch(kn, 1, N, generator=g)[0].double()
            clouds[n] = (torch.cat([d, w]), np.r_[np.ones(N, bool), np.zeros(N, bool)])
        prof = {}
        for c, (n, (cl, ii)) in enumerate(clouds.items()):
            a = axs[row, c]
            z0 = nf._lps(a, cl, ii, alpha=0.3, s=0.5, label=(row == 0 and c == 0))
            x = cl.numpy()
            zd, pd = x[ii, 2] * 1e6 - z0, (x[ii, 5] / x[ii, 5].mean() - 1) * 100
            prof[n] = slice_profile(zd, pd)
            ratio = "" if n == "Bmad" else (f", slice dE/E {np.median(prof[n][1]) / np.median(prof['Bmad'][1]):.1f}× Bmad")
            a.set_title(f"{LABEL.get(n, 'Bmad')}\n{pname}: drive σz {zd.std():.1f} µm{ratio}", fontsize=8.5)
            a.set(xlabel="z − drive centroid [µm]", ylabel="ΔE/E [%]")
            if c:
                a.set_xlim(axs[row, 0].get_xlim()); a.set_ylim(axs[row, 0].get_ylim())
        a = axs[row, 3]
        for n, (zc, s) in prof.items():
            a.plot(zc, s, color=COL[n], lw=1.6 if n == "Bmad" else 1.3, label=LABEL.get(n, "Bmad"))
        a.set(yscale="log", xlabel="z − drive centroid [µm]", ylabel="drive slice ΔE/E spread (robust σ) [%]",
              title=f"drive slice energy spread along z\n{pname}")
        if row == 0:
            a.legend(frameon=False, fontsize=7.5)
    axs[0, 0].legend(frameon=False, fontsize=7, markerscale=12, loc="upper left")
    fig.suptitle("Thinness: LPS at XTCAVF, Bmad vs surrogate (one draw, 30k particles each); right: how thin each z-slice is "
                 "(lower = sharper; Bmad is the target)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "thinness_lps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    spacing_fig()
    thinness_fig()
    print("done")
