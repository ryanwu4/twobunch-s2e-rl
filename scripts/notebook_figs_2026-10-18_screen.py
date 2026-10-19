"""Screen-feature figures for the 2026-10-18 surrogate notebook entry, from results/2_surrogate_screen/screen_features_check_2026-10-18
(scripts/screen_features_check_2026-10-18.py): Bmad-rendered vs surrogate-rendered XTCAV features on the held-out test
rows, plus example +V images (Bmad vs B2-cov + spacing head) rendered with the same segment / binning. CPU.

  PYTHONPATH=src python scripts/notebook_figs_2026-10-18_screen.py
"""
import importlib.util
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("sim_screens", REPO / "scripts/notebook_sim_screens_2026-10-15.py")
sim_screens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim_screens)
render = sim_screens.render
from twobunch_s2e_rl.online.features import band_stats, step_cut, step_features  # noqa: E402

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
RES = REPO / "results/2_surrogate_screen/screen_features_check_2026-10-18"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK})
z = np.load(RES / "rows.npz")
SRC = {s: {k.split("|")[1]: z[k] for k in z.files if k.startswith(s + "|")} for s in ("Bmad", "B2-cov", "spacing head")}
B = SRC["Bmad"]


def parity():
    S = SRC["spacing head"]
    ok = (B["screen_valid_deployed"] > 0.5) & (S["screen_valid_deployed"] > 0.5)
    mis = np.maximum(B["screen_misassign"], S["screen_misassign"])[ok]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    b, m = B["bunch_spacing"][ok] * 1e6, S["bunch_spacing"][ok] * 1e6
    sc = ax[0].scatter(b, m, c=mis, cmap="viridis_r", vmin=0, vmax=0.5, s=12, edgecolors="none")
    lim = [min(b.min(), m.min()) - 10, max(b.max(), m.max()) + 10]
    ax[0].plot(lim, lim, color=INK2, lw=0.8)
    ax[0].set(xlim=lim, ylim=lim, xlabel="Bmad beam -> screen spacing [µm]", ylabel="surrogate -> screen spacing [µm]",
              title=f"measured spacing ({ok.sum()} test rows valid in both)")
    cb = fig.colorbar(sc, ax=ax[0]); cb.set_label("misassigned fraction (worse of the two)")
    bz, mz = B["drive_sigma_z"][ok] * 1e6, S["drive_sigma_z"][ok] * 1e6
    ax[1].scatter(bz, mz, s=10, color=BLUE, alpha=0.6, edgecolors="none")
    lz = [0, max(bz.max(), mz.max()) * 1.05]
    ax[1].plot(lz, lz, color=INK2, lw=0.8)
    ez = np.median(np.abs(mz / bz - 1))
    ax[1].set(xlim=lz, ylim=lz, xlabel="Bmad -> screen drive σz [µm]", ylabel="surrogate -> screen drive σz [µm]",
              title=f"measured drive σz (median |rel| {ez:.1%})")
    for src, c in (("B2-cov", ORANGE), ("spacing head", BLUE)):
        S2 = SRC[src]
        ok2 = (B["screen_valid_deployed"] > 0.5) & (S2["screen_valid_deployed"] > 0.5)
        e = np.sort(np.abs(S2["bunch_spacing"] - B["bunch_spacing"])[ok2] * 1e6)
        ec = np.sort(np.abs(S2["bunch_spacing_cloud"] - B["bunch_spacing_cloud"])[ok2] * 1e6)
        ax[2].plot(e, np.linspace(0, 1, len(e)), color=c, lw=1.8, label=f"{src}: screen ({np.mean(e < 10):.0%} < 10 µm)")
        ax[2].plot(ec, np.linspace(0, 1, len(ec)), color=c, lw=1.0, ls="--", label=f"{src}: cloud truth ({np.mean(ec < 10):.0%})")
    ax[2].axvline(10, color=INK2, ls=":", lw=0.8)
    ax[2].set(xscale="log", xlim=(0.1, 400), ylim=(0, 1), xlabel="|surrogate − Bmad| spacing [µm]", ylabel="fraction of rows",
              title="error CDF: on the screen vs in the beam")
    ax[2].legend(frameon=False, fontsize=7.5, loc="lower right")
    for a in ax:
        a.grid(color=GRID, lw=0.5)
    fig.suptitle("XTCAV features rendered from Bmad beams vs from the surrogate, same knobs, same screen analysis "
                 "(held-out test rows, golden optics, no noise)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-18_screen_parity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def failure_modes():
    S = SRC["spacing head"]
    ok = (B["screen_valid_deployed"] > 0.5) & (S["screen_valid_deployed"] > 0.5)
    e = np.abs(S["bunch_spacing"] - B["bunch_spacing"])[ok] * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].scatter(B["screen_misassign"][ok], e, s=10, color=INK2, alpha=0.6, edgecolors="none", label="vs Bmad's misassignment")
    ax[0].scatter(S["screen_misassign"][ok], e, s=10, color=BLUE, alpha=0.5, edgecolors="none", label="vs the surrogate's")
    ax[0].axhline(10, color=INK2, ls=":", lw=0.8)
    ax[0].set(yscale="log", ylim=(0.1, 500), xlabel="fraction of particles assigned to the wrong bunch by the energy split",
              ylabel="|surrogate − Bmad| screen spacing [µm]", title="large disagreements = the split fails in one of the two")
    ax[0].legend(frameon=False, fontsize=8)
    t = B["bunch_spacing_cloud"] * 1e6
    bins = np.array([-200, 0, 60, 100, 150, 200, 300, 700])
    xc = 0.5 * (bins[1:] + bins[:-1])
    vb = [np.mean(B["screen_valid_deployed"][(t >= a) & (t < b)] > 0.5) for a, b in zip(bins[:-1], bins[1:])]
    vs = [np.mean(S["screen_valid_deployed"][(t >= a) & (t < b)] > 0.5) for a, b in zip(bins[:-1], bins[1:])]
    ag = [np.mean((B["screen_valid_deployed"] > 0.5)[(t >= a) & (t < b)] == (S["screen_valid_deployed"] > 0.5)[(t >= a) & (t < b)])
          for a, b in zip(bins[:-1], bins[1:])]
    ax[1].plot(xc, vb, "o-", color=INK2, label="Bmad: valid fraction")
    ax[1].plot(xc, vs, "o-", color=BLUE, label="surrogate: valid fraction")
    ax[1].plot(xc, ag, "s--", color=RED, label="validity agrees")
    ax[1].axvline(60, color=INK2, ls=":", lw=0.8)
    ax[1].set(ylim=(0, 1.02), xlabel="true spacing [µm] (Bmad cloud)", ylabel="fraction",
              title="screen validity vs spacing (60 µm deployed floor)")
    ax[1].legend(frameon=False, fontsize=8)
    for a in ax:
        a.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-18_screen_failure_modes.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def images(V=22e6):
    from pmd_beamphysics import ParticleGroup
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    from xtcav_segment.convert import cloud_to_beam
    from xtcav_segment.features import calibrate_streak
    from xtcav_segment.segment import back_drift, build_segment_from_spec, element_by_name, load_spec, set_tdc
    S = SRC["spacing head"]
    ok = (B["screen_valid_deployed"] > 0.5) & (S["screen_valid_deployed"] > 0.5)
    e = np.abs(S["bunch_spacing"] - B["bunch_spacing"]) * 1e6
    rows = z["rows"]
    idx_ok = np.flatnonzero(ok)
    good = idx_ok[np.argmin(np.abs(e[idx_ok] - np.median(e[idx_ok])) + 1e3 * (B["bunch_spacing_cloud"][idx_ok] < 100e-6))]
    bad = idx_ok[np.argmax(np.where(e[idx_ok] < 400, e[idx_ok], 0))]
    with h5py.File("processed/xtcav23e_inj_cov_2026-10-17.h5") as h:
        names = h.attrs["campaigns"].split(","); camp, idx = h["campaign"][...], h["idx"][...]
    m = TwoBunchFlow.load_from_checkpoint(str(next(Path("trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints").glob("*.ckpt"))),
                                          map_location="cpu").eval()
    spec = load_spec(REPO / "configs/segment/segment_baseline.json")
    e_ref, l_tdc = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
    seg, _ = build_segment_from_spec(spec, device=torch.device("cpu"), dtype=torch.float64)
    tdc = element_by_name(seg, "XTCAVF")
    set_tdc(seg, V, 0.0)
    tdc.voltage = torch.tensor([[V], [-V]], dtype=torch.float64)

    def screen(cloud, q, isd):
        with torch.no_grad():
            out = seg.track(back_drift(cloud_to_beam(cloud[None], e_ref, q[None], device=torch.device("cpu")), l_tdc))
        xp, xm = out.x[0, 0].numpy(), out.x[1, 0].numpy()
        yp, ym = out.y[0, 0].numpy(), out.y[1, 0].numpy()
        S_ = float(calibrate_streak(torch.tensor(0.5 * (xp - xm))[None], cloud[:, 2][None]))
        ip, xc, yc, _ = render(xp, yp, q.numpy())
        im, _, _, _ = render(xm, ym, q.numpy())
        cut, okc = step_cut([ip])
        meas = None
        if okc:
            bp, bm = band_stats(ip, xc, cut_row=cut), band_stats(im, xc, cut_row=cut)
            if bp and bm:
                meas = step_features([bp], [bm], streak=S_, edge_frac_max=0.01)
        return ip, xc, yc, (cut if okc else None), meas

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for r, (j, lab) in enumerate(((good, "typical row (median disagreement)"), (bad, "largest disagreement"))):
        row = int(rows[j])
        P = ParticleGroup(f"data/{names[camp[row]]}/sample_{idx[row]:05d}_XTCAVF.h5")
        u = np.unique(P.weight); isd = P.weight == u[-1]
        cb = torch.tensor(np.stack([P.x, P.y, P.z, P.px, P.py, P.pz], 1), dtype=torch.float64)
        qb = torch.tensor(np.asarray(P.weight, float), dtype=torch.float64)
        kn = torch.tensor(z["knobs_norm"][j:j + 1], dtype=torch.float32)
        with torch.no_grad():
            d = m.sample_bunch(kn, 0, 20000)[0].double(); w = m.sample_bunch(kn, 1, 20000)[0].double()
        qd, qw = float(qb[isd].sum()), float(qb[~isd].sum())
        cm = torch.cat([d, w]); qm = torch.cat([torch.full((20000,), qd / 20000), torch.full((20000,), qw / 20000)]).double()
        for c, (cloud, q, src) in enumerate(((cb, qb, "Bmad"), (cm, qm, "surrogate (spacing head)"))):
            ip, xc, yc, cut, meas = screen(cloud, q, None)
            a = axes[r, c]
            a.imshow(ip, origin="lower", aspect="auto", cmap="magma", extent=[xc[0] * 1e3, xc[-1] * 1e3, yc[0] * 1e3, yc[-1] * 1e3])
            if cut is not None:
                a.axhline(yc[min(int(cut), len(yc) - 1)] * 1e3, color="#9be3c6", ls="--", lw=1)
            true = (B if c == 0 else S)["bunch_spacing_cloud"][j] * 1e6
            mtxt = "no reading" if meas is None else f"screen {meas['bunch_spacing'] * 1e6:.0f} µm ({'valid' if meas['valid'] else 'INVALID'})"
            mis = (B if c == 0 else S)["screen_misassign"][j]
            a.set_title(f"{src} — {lab}", fontsize=9)
            a.text(0.02, 0.04, f"beam spacing {true:.0f} µm\n{mtxt}\nmisassigned {mis:.0%}", transform=a.transAxes,
                   color="white", fontsize=8, va="bottom")
            a.set(xlabel="x (streak) [mm]", ylabel="y (energy) [mm]")
    fig.suptitle("+V XTCAV images (golden optics, 22 MV); dashed = the energy cut that splits drive / witness",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-18_screen_images.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parity()
    failure_modes()
    images()
    print("done")
