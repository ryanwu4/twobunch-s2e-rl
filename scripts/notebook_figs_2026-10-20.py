"""Figures for the 2026-10-20 notebook entry (two-day progress 10-19 -> 20): real and simulated XTCAV screens with the
crop windows, Bmad beams along noise-free closed-loop slews, beams at the compressed gradient-check point, TCAV flip
settling. Writes lab-notebook/images/2026-10-20_*.png. CPU.

  PYTHONPATH=src python scripts/notebook_figs_2026-10-20.py [fig ...]
"""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
XT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, RED, GREEN = "#2a78d6", "#eb6834", "#e34948", "#2b8a3e"
AUTO = (4.6, 2.44)            # deployed auto-crop half-widths [mm] (150 x 80 px)
HAND = (6.86, 4.58)           # deployed hand crop as half-widths [mm] (450 x 300 px), the new sim window
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK})


def window(ax, cx, cy, hw, color, ls, label):
    ax.add_patch(Rectangle((cx - hw[0], cy - hw[1]), 2 * hw[0], 2 * hw[1], fill=False, ec=color, ls=ls, lw=1.3, label=label))


def real_crop(scan="E300_15673", steps=(1, 12, 24)):
    """Median +V DTOTR2 images (daq2npz, COM-centred) at three scan steps, deployed auto-crop window overlaid."""
    z = np.load(XT / "daq2npz/out" / f"{scan}.npz", allow_pickle=True)
    st, tc, gi = z["step"], z["tcav_class"], z["good_image"]
    v6 = json.loads((REPO / "results/4_daq_realism/online_replay_v6_median_cut_2026-10-15" / f"{scan}.json").read_text())["steps"]
    v7 = json.loads((REPO / "results/4_daq_realism/online_replay_v7_noauto_2026-10-19" / f"{scan}.json").read_text())["steps"]
    fig, axs = plt.subplots(1, len(steps), figsize=(4.6 * len(steps), 4.1))
    for ax, s in zip(axs, steps):
        m = np.flatnonzero((st == s) & (tc == 1) & gi)
        img = np.nanmedian(z["states_projections_2d"][m, 0], axis=0).T          # [iy, ix]
        xe, ye = z["states_projection_x_edges"][m[0], 0], z["states_projection_y_edges"][m[0], 0]
        cx, cy = 0.5 * (xe[0] + xe[-1]), 0.5 * (ye[0] + ye[-1])
        ax.imshow(img, origin="lower", aspect="auto", cmap="magma", extent=[xe[0] - cx, xe[-1] - cx, ye[0] - cy, ye[-1] - cy],
                  vmax=np.percentile(img, 99.8))
        window(ax, 0, 0, AUTO, "#7fdbff", "--", "deployed auto-crop (±4.6 × ±2.44 mm)")
        a, b = v6[str(s)], v7[str(s)]
        txt = (f"auto-crop:     {a['spacing_um']:.0f} µm, {'valid' if a['valid'] else 'INVALID'}"
               f"{' (edge)' if a['edge_frac'] > 0.01 else ''}\n"
               f"no auto-crop: {b['spacing_um']:.0f} µm, {'valid' if b['valid'] else 'INVALID'}")
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, color="white", fontsize=8, va="bottom", family="monospace")
        ax.set(xlim=(-7.8, 7.8), ylim=(-3.9, 3.9), xlabel="x (streak) − centroid [mm]", ylabel="y (energy) − centroid [mm]",
               title=f"step {s}, L2 offset {a['scan_val']:.2f}° ({len(m)} +V images, median)")
    axs[0].legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor="white")
    fig.suptitle(f"Real DTOTR2 images ({scan}): the deployed auto-crop (dashed) clips the witness at large spacing (left). "
                 "Plot area = the stored DAQ window (±7.8 × ±3.9 mm).", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-20_real_crop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sim_tools():
    spec = importlib.util.spec_from_file_location("sim_screens", REPO / "scripts/notebook_sim_screens_2026-10-15.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from xtcav_segment.convert import cloud_to_beam
    from xtcav_segment.features import calibrate_streak
    from xtcav_segment.segment import back_drift, build_segment_from_spec, element_by_name, load_spec, set_tdc
    sp = load_spec(REPO / "configs/segment/segment_baseline.json")
    e_ref, l_tdc = float(sp["meta"]["ref_energy_eV"]), float(sp["meta"]["tdc_length_m"])
    seg, _ = build_segment_from_spec(sp, device=torch.device("cpu"), dtype=torch.float64)
    tdc = element_by_name(seg, "XTCAVF")
    set_tdc(seg, 22e6, 0.0)
    tdc.voltage = torch.tensor([[22e6], [-22e6]], dtype=torch.float64)

    def plus_image(cloud, q, half=(7.8, 3.9), nx=256, ny=256, sign=+1):
        """+V (sign +1) or -V (sign -1) DTOTR image of a (N, 6) cloud with per-particle charge q: COM-centred [mm]."""
        with torch.no_grad():
            out = seg.track(back_drift(cloud_to_beam(cloud[None].double(), e_ref, q[None].double(), device=torch.device("cpu")), l_tdc))
        k = 0 if sign > 0 else 1
        x, y = out.x[k, 0].numpy() * 1e3, out.y[k, 0].numpy() * 1e3
        w = q.numpy()
        x, y = x - np.average(x, weights=w), y - np.average(y, weights=w)
        H, xe, ye = np.histogram2d(x, y, bins=[nx, ny], range=[[-half[0], half[0]], [-half[1], half[1]]], weights=w)
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(H.T, 1.0), xe, ye
    return plus_image


def _bmad_cloud(path):
    from pmd_beamphysics import ParticleGroup
    P = ParticleGroup(str(path))
    u = np.unique(P.weight)
    c = torch.tensor(np.stack([P.x, P.y, P.z, P.px, P.py, P.pz], 1), dtype=torch.float64)
    return c, torch.tensor(np.asarray(P.weight, float)), P.weight == u[-1]


def sim_crop():
    """Bmad vs surrogate +V screen at a > 250 um test row that the deployed auto-crop makes invalid; both windows drawn."""
    import h5py
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    plus_image = _sim_tools()
    r = np.load(REPO / "results/2_surrogate_screen/crop_window_2026-10-19/rows.npz")
    k = np.load(REPO / "results/2_surrogate_screen/screen_features_check_2026-10-18/rows.npz")
    assert np.array_equal(r["rows"], k["rows"])
    t = r["x1|Bmad|bunch_spacing_cloud"] * 1e6
    cand = np.flatnonzero((t > 260) & (t < 340) & (r["x1|Bmad|screen_valid_deployed"] < 0.5)
                          & (r["none|Bmad|screen_valid_deployed"] > 0.5) & (r["none|surrogate|screen_valid_deployed"] > 0.5))
    j = cand[np.argmin(np.abs(t[cand] - np.median(t[cand])))]
    row = int(r["rows"][j])
    with h5py.File(REPO / "processed/xtcav23e_inj_cov_2026-10-17.h5") as h:
        names = h.attrs["campaigns"].split(","); camp, idx = h["campaign"][...], h["idx"][...]
    cb, qb, isd = _bmad_cloud(REPO / f"data/{names[camp[row]]}/sample_{idx[row]:05d}_XTCAVF.h5")
    m = TwoBunchFlow.load_from_checkpoint(str(next((REPO / "trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints").glob("*.ckpt"))),
                                          map_location="cpu").eval()
    kn = torch.tensor(k["knobs_norm"][j:j + 1], dtype=torch.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        d, w = m.sample_bunch(kn, 0, 30000)[0].double(), m.sample_bunch(kn, 1, 30000)[0].double()
    qd, qw = float(qb[isd].sum()), float(qb[~isd].sum())
    cm, qm = torch.cat([d, w]), torch.cat([torch.full((30000,), qd / 30000), torch.full((30000,), qw / 30000)]).double()
    fig, axs2 = plt.subplots(2, 2, figsize=(11, 8.2))
    for ax, ax_m, (c, q, src, key) in zip(axs2[0], axs2[1], ((cb, qb, "Bmad", "Bmad"), (cm, qm, "surrogate (spacing head)", "surrogate"))):
        Hm, xem, yem = plus_image(c, q, sign=-1)
        ax_m.imshow(Hm, origin="lower", aspect="auto", cmap="magma", extent=[xem[0], xem[-1], yem[0], yem[-1]], vmax=np.percentile(Hm, 99.8))
        window(ax_m, 0, 0, AUTO, "#7fdbff", "--", "deployed auto-crop")
        window(ax_m, 0, 0, HAND, "#9be3c6", ":", "hand-crop-sized window (new)")
        ax_m.set(xlabel="x (streak) − centroid [mm]", ylabel="y (energy) − centroid [mm]", title=f"{src}: −V screen")
        H, xe, ye = plus_image(c, q)
        ax.imshow(H, origin="lower", aspect="auto", cmap="magma", extent=[xe[0], xe[-1], ye[0], ye[-1]], vmax=np.percentile(H, 99.8))
        window(ax, 0, 0, AUTO, "#7fdbff", "--", "deployed auto-crop")
        window(ax, 0, 0, HAND, "#9be3c6", ":", "hand-crop-sized window (new)")
        f = lambda tag, kk: r[f"{tag}|{key}|{kk}"][j]  # noqa: E731
        txt = (f"beam spacing {f('x1', 'bunch_spacing_cloud') * 1e6:.0f} µm\n"
               f"auto-crop:   {f('x1', 'bunch_spacing') * 1e6:.0f} µm, {'valid' if f('x1', 'screen_valid_deployed') > .5 else 'INVALID'}\n"
               f"wide window: {f('none', 'bunch_spacing') * 1e6:.0f} µm, {'valid' if f('none', 'screen_valid_deployed') > .5 else 'INVALID'}")
        ax.text(0.02, 0.03, txt, transform=ax.transAxes, color="white", fontsize=8, va="bottom", family="monospace")
        ax.set(xlabel="x (streak) − centroid [mm]", ylabel="y (energy) − centroid [mm]", title=f"{src}: +V screen, test row {row}")
    axs2[0, 0].legend(frameon=False, fontsize=7.5, loc="upper left", labelcolor="white")
    fig.suptitle("Simulated DTOTR ±V images of a held-out test simulation at large spacing (golden optics, 22 MV, no noise); "
                 "the −V image runs into the auto-crop edge",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-20_sim_crop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _lps(ax, c, isd, z0=None, alpha=0.25, s=0.4, colors=(BLUE, ORANGE), label=True):
    c = c.numpy()
    zz, pz = c[:, 2] * 1e6, c[:, 5]
    z0 = zz[isd].mean() if z0 is None else z0
    p0 = pz[isd].mean()
    rng = np.random.default_rng(0)
    for sel, col, lab in ((isd, colors[0], "drive"), (~isd, colors[1], "witness")):
        ii = np.flatnonzero(sel)
        ii = ii[rng.permutation(len(ii))[:6000]]
        ax.scatter(zz[ii] - z0, (pz[ii] / p0 - 1) * 100, s=s, color=col, alpha=alpha, edgecolors="none", rasterized=True,
                   label=lab if label else None)
    return z0


def slew_beams(run="nc_zs_screen_noblen_jit_s0"):
    """One noise-free closed-loop Bmad episode: true spacing vs step, and the Bmad beam (z-E) + +V screen at the end of
    each setpoint, with the goal beam (Bmad at the goal phases) in grey."""
    plus_image = _sim_tools()
    D = REPO / "results/3_rl_eval/slew_ideal_bmad_2026-10-19" / f"{run}_bmad"
    t = torch.load(D / "traj.pt")
    tr, g = t["bunch_spacing_cloud"].numpy() * 1e6, t["spacing_goal_cloud"].numpy() * 1e6
    sz, gsz = t["drive_sigma_z_cloud"].numpy() * 1e6, t["drive_sigma_z_goal_cloud"].numpy() * 1e6
    S = json.loads((D / "summary.json").read_text())["segments"]["bunch_spacing"]["segments"]
    fin = np.array([s["final"] for s in S]) * 1e6                      # (3, K)
    jumps = np.abs(np.diff(g[[0, 16, 32]], axis=0)).min(0)
    score = np.median(fin, 0)
    ok = np.flatnonzero(jumps > 50)
    b = int(ok[np.argmin(np.abs(score[ok] - np.median(score[ok])))])
    print(f"episode {b}: goals {np.round(g[[0, 16, 32], b])}, final errors {np.round(fin[:, b], 1)}")
    beams = D / "bmad" / "beams"
    ends = {0: 19, 1: 36, 2: 53}                                         # call index of each segment's last step
    fig = plt.figure(figsize=(14, 8.2))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.05, 1, 1], hspace=0.45, wspace=0.3)
    ax = fig.add_subplot(gs[0, :])
    T = np.arange(tr.shape[0])
    ax.fill_between(T, g[:, b] - 10, g[:, b] + 10, step="mid", color=GRID, lw=0)
    ax.step(T, g[:, b], where="mid", color=INK, ls="--", lw=1, label="setpoint (± 10 µm)")
    ax.plot(T, tr[:, b], color=BLUE, lw=1.8, label="Bmad beam spacing (true)")
    ax2 = ax.twinx()
    ax2.plot(T, sz[:, b], color=ORANGE, lw=1.2, alpha=0.8, label="drive σz (true)")
    ax2.step(T, gsz[:, b], where="mid", color=ORANGE, ls=":", lw=1, label="σz setpoint")
    ax2.set_ylabel("drive σz [µm]", color=ORANGE)
    ax2.spines["right"].set_visible(True)
    ax.set(xlabel="controller step (≈ 20 s each)", ylabel="spacing [µm]",
           title=f"noise-free closed loop on Bmad: policy {run}, episode {b} (golden injector, ideal machine)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right", ncol=4)
    for j in range(3):
        cg, qg, ig = _bmad_cloud(beams / f"c{j:03d}_e{b:02d}_XTCAVF.h5")
        ce, qe, ie = _bmad_cloud(beams / f"c{ends[j]:03d}_e{b:02d}_XTCAVF.h5")
        a = fig.add_subplot(gs[1, j])
        z0 = _lps(a, cg, ig, alpha=0.12, colors=("0.55", "0.75"), label=False)
        _lps(a, ce, ie, z0=None, label=(j == 0))
        a.set(xlabel="z − drive centroid [µm]", ylabel="ΔE/E [%]",
              title=f"setpoint {j + 1}: {g[16 * j + 15, b]:.0f} µm → beam {tr[16 * j + 15, b]:.0f} µm")
        if j == 0:
            a.legend(frameon=False, fontsize=7, markerscale=12, loc="upper left")
        H, xe, ye = plus_image(ce, qe)
        a2 = fig.add_subplot(gs[2, j])
        a2.imshow(H, origin="lower", aspect="auto", cmap="magma", extent=[xe[0], xe[-1], ye[0], ye[-1]], vmax=np.percentile(H, 99.8))
        window(a2, 0, 0, HAND, "#9be3c6", ":", "screen window")
        a2.set(xlabel="x (streak) [mm]", ylabel="y (energy) [mm]", title=f"its +V screen (end of setpoint {j + 1})")
    a = fig.add_subplot(gs[1:, 3])
    for j in range(3):
        ce, qe, ie = _bmad_cloud(beams / f"c{ends[j]:03d}_e{b:02d}_XTCAVF.h5")
        zz = ce[:, 2].numpy() * 1e6
        zz = zz - zz[ie].mean()
        h, e = np.histogram(zz, bins=np.arange(-450, 150, 3.0), weights=qe.numpy())      # charge [C] per 3 um
        I = h / ((e[1] - e[0]) * 1e-6) * 299792458 / 1e3                                   # current [kA]
        a.plot(0.5 * (e[1:] + e[:-1]), I, lw=1.2, label=f"setpoint {j + 1}: {g[16 * j + 15, b]:.0f} µm")
    a.set(xlabel="z − drive centroid [µm]", ylabel="current [kA]", title="Bmad current profiles at the three setpoints")
    a.legend(frameon=False, fontsize=8)
    fig.suptitle("Grey = Bmad beam at the setpoint's phases (what the operator asked for); colour = the beam the controller reached",
                 fontsize=10, x=0.01, ha="left")
    fig.savefig(OUT / "2026-10-20_slew_beams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def compressed_beams():
    """Bmad vs surrogate beams at the compressed gradient-check point (golden injector, L1 -22.5 / L2 -35) and L1 +-0.3
    deg: where the surrogate's sigma_z response to L1 is ~5x too weak."""
    from twobunch_s2e_rl.rl._train_utils import build_env_fn
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    G = REPO / "results/2_surrogate_screen/gradient_check_2026-10-18"
    B = np.load(G / "bmad.npz")
    X, lab = B["X"], B["lab"]
    sel = {"L1 − 0.3°": int(np.flatnonzero((lab[:, 0] == 0) & (lab[:, 1] == 1) & (lab[:, 2] == 0) & (lab[:, 3] == 0.3) & (lab[:, 4] == -1))[0]),
           "centre": int(np.flatnonzero((lab[:, 0] == 0) & (lab[:, 1] == 1) & (lab[:, 2] < 0))[0]),
           "L1 + 0.3°": int(np.flatnonzero((lab[:, 0] == 0) & (lab[:, 1] == 1) & (lab[:, 2] == 0) & (lab[:, 3] == 0.3) & (lab[:, 4] == 1))[0])}
    spec = importlib.util.spec_from_file_location("gc", REPO / "scripts/gradient_check_2026-10-18.py")
    gc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gc)
    env = gc.make_env(gc.FLOWS["spacing head"], 1024)
    m = TwoBunchFlow.load_from_checkpoint(gc.FLOWS["spacing head"], map_location="cpu").eval()
    fig, axs = plt.subplots(2, 4, figsize=(15, 6.6), gridspec_kw={"width_ratios": [1, 1, 1, 0.9]})
    prof = {}
    for c, (name, i) in enumerate(sel.items()):
        cb, qb, isd = _bmad_cloud(G / "bmad" / "beams" / f"r{i:03d}_XTCAVF.h5")
        kn = env._ks.normalize(torch.as_tensor(X[i:i + 1], dtype=torch.float32))
        torch.manual_seed(1)
        with torch.no_grad():
            d, w = m.sample_bunch(kn, 0, 30000)[0].double(), m.sample_bunch(kn, 1, 30000)[0].double()
        cs = torch.cat([d, w]); iss = np.r_[np.ones(30000, bool), np.zeros(30000, bool)]
        for r, (cl, ii, src) in enumerate(((cb, isd, "Bmad"), (cs, iss, "surrogate"))):
            a = axs[r, c]
            _lps(a, cl, ii, label=(c == 0))
            zz = cl[:, 2].numpy() * 1e6
            szd = zz[ii].std()
            a.set(xlim=(-90, 60), ylim=(-2.2, 1.6), xlabel="z − drive centroid [µm]", ylabel="ΔE/E [%]",
                  title=f"{src}, {name}: drive σz {szd:.1f} µm")
            prof.setdefault(src, []).append((name, zz[ii] - zz[ii].mean()))
            if c == 0:
                a.legend(frameon=False, fontsize=7, markerscale=12, loc="upper left")
    for r, src in enumerate(("Bmad", "surrogate")):
        a = axs[r, 3]
        for (name, zz), col in zip(prof[src], (GREEN, INK, RED)):
            h, e = np.histogram(zz, bins=120, range=(-60, 60), density=True)
            a.plot(0.5 * (e[1:] + e[:-1]), h, color=col, lw=1.3, label=name)
        a.set(xlabel="z − drive centroid [µm]", ylabel="drive density", title=f"{src}: drive profile vs L1")
        a.legend(frameon=False, fontsize=7)
    fig.suptitle("Compressed point (golden injector, L1 −22.5° / L2 −35°, spacing ≈ 70 µm): Bmad's drive σz moves with L1, the "
                 "surrogate's barely does", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-20_compressed_beams.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def tcav_flip():
    """Real DAQ (all runs, 177 +-V flips; scripts/xtcav_tcav_flip_lag_2026-10-19.py): after the TCAV phase readback locks
    within 1 deg of +-90, median |deviation| / settled robust sigma of the readback and of the DTOTR2 centroid."""
    med = {"TCAV phase readback": [1.7, 1.0, 1.0, 0.7, 0.8, 1.0, 0.8, 1.0, 0.9, 0.8, 0.9, 0.8, 0.9, 0.9, 1.0, 0.6],
           "screen centroid x (streak)": [1.6, 1.3, 1.2, 1.2, 1.1, 1.3, 1.1, 1.1, 1.0, 0.9, 1.0, 0.9, 0.8, 0.9, 0.8, 0.7],
           "screen centroid y (energy)": [1.4, 1.4, 1.1, 1.4, 1.1, 1.2, 1.4, 1.0, 1.1, 1.1, 1.0, 0.9, 0.7, 0.9, 0.8, 0.7]}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for (n, v), col in zip(med.items(), (INK, ORANGE, BLUE)):
        ax.plot(np.arange(len(v)) / 10, v, "o-", color=col, ms=3, label=n)
    ax.axhline(0.67, color=INK2, ls=":", lw=0.8)
    ax.text(1.5, 0.69, "settled scatter", color=INK2, fontsize=8, ha="right")
    ax.set(xlabel="time after the TCAV phase readback locks (10 Hz shots) [s]", ylabel="median |deviation| / settled σ",
           title="real DAQ, 177 ±V flips: the image settles ~1 s after the readback")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-20_tcav_flip.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def copies():
    for src, dst in (("results/2_surrogate_screen/gradient_check_2026-10-18/gradient_parity.png", "2026-10-20_gradient_parity.png"),
                     ("results/3_rl_eval/slew_ideal_bmad_2026-10-19/slew_traces_nc_zs_screen_noblen_jit_s0.png", "2026-10-20_slew_traces.png"),
                     ("results/3_rl_eval/slew_ideal_bmad_2026-10-19/slew_summary.png", "2026-10-20_slew_summary.png")):
        shutil.copy(REPO / src, OUT / dst)


if __name__ == "__main__":
    figs = sys.argv[1:] or ["real_crop", "sim_crop", "slew_beams", "compressed_beams", "tcav_flip", "copies"]
    for f in figs:
        globals()[f]()
        print(f, "done", flush=True)
