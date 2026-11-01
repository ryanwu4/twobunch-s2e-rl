"""Sample figures for the 10-21 surrogate candidates: Bmad vs B2-cov vs lat8_P4096 (beam latent, prior draw) vs bb32x
(backbone shear, prior draw; and its "own backbone" sample = posterior) at a normal (golden, L1 -22.5 / L2 -39, ~220 um)
and the compressed gradient-check point (golden, L1 -22.5 / L2 -35, ~70 um):
  samples_lps.png      drive (blue) / witness (orange) z vs dE/E
  samples_screens.png  +V DTOTR screens of the same beams (Bmad charges)
  samples_draws.png    4 independent draws per new model at the compressed point (current profiles + LPS)
Output results/2_surrogate_screen/flow_sharpness_2026-10-20/figs/. CPU.

  PYTHONPATH=src python scripts/surrogate_samples_figs_2026-10-21.py
"""
import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results/2_surrogate_screen/flow_sharpness_2026-10-20/figs"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


nf = _load("nf", REPO / "scripts/notebook_figs_2026-10-20.py")
gc = _load("gc", REPO / "scripts/gradient_check_2026-10-18.py")
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

MODELS = {"B2-cov (incumbent)": "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt",
          "lat8_P4096 (beam latent)": "trained/xtcav23e_inj_covB2_lat8_P4096_2026-10-21/checkpoints/final.ckpt",
          "bb32x (backbone, prior)": "trained/xtcav23e_inj_covB2_bb32x_P1024_2026-10-21/checkpoints/final.ckpt"}
G = REPO / "results/2_surrogate_screen/gradient_check_2026-10-18"
N = 30000


def point(il):
    B = np.load(G / "bmad.npz")
    lab = B["lab"]
    i = int(np.flatnonzero((lab[:, 0] == 0) & (lab[:, 1] == il) & (lab[:, 2] < 0))[0])
    return i, B["X"][i:i + 1]


def main():
    plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    env = gc.make_env(gc.FLOWS["B2-cov"], 1024)
    flows = {k: TwoBunchFlow.load_from_checkpoint(str(REPO / v), map_location="cpu").eval() for k, v in MODELS.items()}
    plus_image = nf._sim_tools()
    pts = {"normal (~220 µm)": 0, "compressed (~70 µm)": 1}
    beams = {}
    for pname, il in pts.items():
        i, X = point(il)
        cb, qb, isd = nf._bmad_cloud(G / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5")
        kn = env._ks.normalize(torch.as_tensor(X, dtype=torch.float32))
        row = {"Bmad": (cb, isd)}
        for name, m in flows.items():
            g = torch.Generator().manual_seed(3)
            with torch.no_grad():
                d, w = m.sample_bunch(kn, 0, N, generator=g)[0].double(), m.sample_bunch(kn, 1, N, generator=g)[0].double()
            row[name] = (torch.cat([d, w]), np.r_[np.ones(N, bool), np.zeros(N, bool)])
        m = flows["bb32x (backbone, prior)"]
        cc = {}
        for k, sel in ((0, isd), (1, ~isd)):
            mean, std = m._scaler(k)
            sub = cb.numpy()[sel][np.random.default_rng(0).choice(int(sel.sum()), 1024, replace=False)]
            ps = torch.as_tensor((sub - mean.numpy()) / std.numpy(), dtype=torch.float32)[None]
            with torch.no_grad():
                cc[k] = m.bb_encode(m.bb_features(ps), k)
        g = torch.Generator().manual_seed(3)
        with torch.no_grad():
            d = m.sample_bunch(kn, 0, N, generator=g, c=cc[0])[0].double()
            w = m.sample_bunch(kn, 1, N, generator=g, c=cc[1])[0].double()
        row["bb32x (own backbone)"] = (torch.cat([d, w]), np.r_[np.ones(N, bool), np.zeros(N, bool)])
        beams[pname] = (row, qb, isd, kn)
    cols = list(next(iter(beams.values()))[0])

    # 1. LPS
    fig, axs = plt.subplots(2, len(cols), figsize=(3.3 * len(cols), 6.4))
    for r, (pname, (row, qb, isd, kn)) in enumerate(beams.items()):
        for c, name in enumerate(cols):
            cl, ii = row[name]
            a = axs[r, c]
            nf._lps(a, cl, ii, alpha=0.3, s=0.5, label=(r == 0 and c == 0))
            zz = cl[:, 2].numpy() * 1e6
            a.set_title(f"{name}\n{pname}: drive σz {zz[ii].std():.1f} µm", fontsize=8)
            a.set(xlabel="z − drive centroid [µm]", ylabel="ΔE/E [%]")
            if r == 1:
                a.set(xlim=(-110, 60))
        for c in range(1, len(cols)):
            axs[r, c].set_xlim(axs[r, 0].get_xlim())
            axs[r, c].set_ylim(axs[r, 0].get_ylim())
    axs[0, 0].legend(frameon=False, fontsize=7, markerscale=12, loc="upper left")
    fig.suptitle("Longitudinal phase space at XTCAVF: Bmad vs surrogate samples (one draw each; 'own backbone' = bb32x given the "
                 "Bmad beam's backbone)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "samples_lps.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 2. screens
    fig, axs = plt.subplots(2, len(cols), figsize=(3.3 * len(cols), 5.8))
    for r, (pname, (row, qb, isd, kn)) in enumerate(beams.items()):
        qd, qw = float(qb[isd].sum()), float(qb[~isd].sum())
        for c, name in enumerate(cols):
            cl, ii = row[name]
            q = qb if name == "Bmad" else torch.tensor(np.where(ii, qd / N, qw / N))
            H, xe, ye = plus_image(cl, q)
            a = axs[r, c]
            a.imshow(H, origin="lower", aspect="auto", cmap="magma", extent=[xe[0], xe[-1], ye[0], ye[-1]], vmax=np.percentile(H, 99.8))
            a.set(xlim=(-5, 5), ylim=(-2.5, 2.5), xlabel="x (streak) [mm]", ylabel="y (energy) [mm]")
            a.set_title(f"{name}\n{pname}", fontsize=8)
    fig.suptitle("+V DTOTR screens of the same beams (golden optics, 22 MV, no noise)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "samples_screens.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # 3. draws at the compressed point
    row, qb, isd, kn = beams["compressed (~70 µm)"]
    fig, axs = plt.subplots(2, 3, figsize=(13, 7))
    cb = row["Bmad"][0].numpy()
    zb = cb[:, 2] * 1e6
    zb0 = zb[isd].mean()
    for j, name in enumerate(["lat8_P4096 (beam latent)", "bb32x (backbone, prior)"]):
        m = flows[name]
        a_cur, a_lps = axs[0, j], axs[1, j]
        hb, e = np.histogram(zb - zb0, bins=np.arange(-120, 60, 1.5), weights=qb.numpy())
        a_cur.plot(0.5 * (e[1:] + e[:-1]), hb / (e[1] - e[0]) / 1e-6 * 299792458 / 1e3, color="k", lw=1.6, label="Bmad")
        for dr in range(4):
            g = torch.Generator().manual_seed(100 + dr)
            with torch.no_grad():
                d, w = m.sample_bunch(kn, 0, N, generator=g)[0].numpy(), m.sample_bunch(kn, 1, N, generator=g)[0].numpy()
            z = np.r_[d[:, 2], w[:, 2]] * 1e6
            z0 = d[:, 2].mean() * 1e6
            qq = np.r_[np.full(N, float(qb[isd].sum()) / N), np.full(N, float(qb[~isd].sum()) / N)]
            h, _ = np.histogram(z - z0, bins=e, weights=qq)
            sp = (d[:, 2].mean() - w[:, 2].mean()) * 1e6
            a_cur.plot(0.5 * (e[1:] + e[:-1]), h / (e[1] - e[0]) / 1e-6 * 299792458 / 1e3, lw=1.0, label=f"draw {dr + 1}: spacing {sp:.0f} µm")
            if dr < 2:
                a_lps.scatter(d[::3, 2] * 1e6 - z0, (d[::3, 5] / d[:, 5].mean() - 1) * 100, s=0.4, alpha=0.3, rasterized=True,
                              label=f"draw {dr + 1}")
        a_lps.scatter(zb[isd][::3] - zb0, (cb[isd][::3, 5] / cb[isd, 5].mean() - 1) * 100, s=0.4, color="k", alpha=0.4,
                      rasterized=True, label="Bmad")
        a_cur.set(xlabel="z − drive centroid [µm]", ylabel="current [kA]", title=f"{name}: 4 draws at the compressed point")
        a_cur.legend(frameon=False, fontsize=7)
        a_lps.set(xlim=(-110, 60), xlabel="z − drive centroid [µm]", ylabel="ΔE/E [%]", title="drive LPS, 2 draws vs Bmad")
        a_lps.legend(frameon=False, fontsize=7, markerscale=10)
    bsp = (zb[isd].mean() - zb[~isd].mean())
    a = axs[0, 2]
    a.plot(0.5 * (e[1:] + e[:-1]), hb / (e[1] - e[0]) / 1e-6 * 299792458 / 1e3, color="k", lw=1.6, label=f"Bmad (spacing {bsp:.0f} µm)")
    m = flows["B2-cov (incumbent)"]
    g = torch.Generator().manual_seed(100)
    with torch.no_grad():
        d, w = m.sample_bunch(kn, 0, N, generator=g)[0].numpy(), m.sample_bunch(kn, 1, N, generator=g)[0].numpy()
    z = np.r_[d[:, 2], w[:, 2]] * 1e6
    h, _ = np.histogram(z - d[:, 2].mean() * 1e6, bins=e, weights=np.r_[np.full(N, float(qb[isd].sum()) / N), np.full(N, float(qb[~isd].sum()) / N)])
    a.plot(0.5 * (e[1:] + e[:-1]), h / (e[1] - e[0]) / 1e-6 * 299792458 / 1e3, color="0.5", lw=1.2, label="B2-cov")
    a.set(xlabel="z − drive centroid [µm]", ylabel="current [kA]", title="reference: Bmad vs B2-cov")
    a.legend(frameon=False, fontsize=7)
    axs[1, 2].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "samples_draws.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("done")


if __name__ == "__main__":
    main()
