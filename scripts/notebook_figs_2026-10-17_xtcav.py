"""Figures for the 2026-10-17 notebook entry: simulated (Bmad closed-loop) and REAL XTCAV images, the L2-only DAQ replay
of the screen-mode controllers on the real scans, and the real BC14 BLEN response vs the env's BLEN model.
Writes lab-notebook/images/2026-10-17_*.png.

  PYTHONPATH=src OMP_NUM_THREADS=4 /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/notebook_figs_2026-10-17_xtcav.py [fig ...]

Image chain reused verbatim: notebook_sim_screens_2026-10-15.render (deployed auto-crop, 30.5 um bins, 1 px blur) +
online/features (shared +V valley cut, +-V half-difference) for the sim; fig_bunch_split_changes_real.step_images
(deployed preprocessing, median-of-per-image +V cut = v6) for the real DAQ.
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
_spec = importlib.util.spec_from_file_location("sim_screens", REPO / "scripts/notebook_sim_screens_2026-10-15.py")
sim_screens = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim_screens)
render = sim_screens.render

from twobunch_s2e_rl.online.features import band_stats, step_cut, step_features  # noqa: E402

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
RES = REPO / "results"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CUT = "#9be3c6"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 8.5,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "text.color": INK})
PX = 30.5e-6
REPLAY = RES / "online_replay_v6_median_cut_2026-10-15"
DQR = RES / "daq_replay_2026-10-17"
SCANS = ("TEST_15668", "E300_15673")
SCAN_C = {"TEST_15668": BLUE, "E300_15673": ORANGE}


def scan_steps(scan):
    st = json.loads((REPLAY / f"{scan}.json").read_text())["steps"]
    keys = sorted(st, key=lambda k: st[k]["scan_val"])
    return keys, st, np.array([st[k]["scan_val"] for k in keys]), np.array([st[k]["spacing_um"] for k in keys]), \
        np.array([st[k]["valid"] > 0.5 for k in keys])


def valid_window(v, x):
    return float(x[v].min()), float(x[v].max())


# ------------------------------------------------------------------ 1. Bmad closed-loop XTCAV images
def bmad_screens(run="S2l1blen_covB2_inj_bmad", V=22e6):
    from pmd_beamphysics import ParticleGroup
    from twobunch_s2e_rl.surrogate.properties import inter_bunch
    from xtcav_segment.convert import cloud_to_beam
    from xtcav_segment.features import calibrate_streak
    from xtcav_segment.segment import back_drift, build_segment_from_spec, element_by_name, load_spec, set_tdc

    d = RES / "closed_loop_bmad_2026-10-17" / run
    summ = json.loads((d / "summary.json").read_text())
    err = np.array(summ["targets"]["bunch_spacing"]["per_episode_final_true"]) * 1e6
    traj = torch.load(d / "traj.pt", weights_only=False)
    med_ep = int(np.argmin(np.abs(err - np.median(err))))
    worst_ep = int(np.argmax(err))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = load_spec(REPO / "configs/segment/segment_baseline.json")          # golden optics
    e_ref, l_tdc = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
    seg, _ = build_segment_from_spec(spec, device=torch.device(dev), dtype=torch.float64)
    tdc = element_by_name(seg, "XTCAVF")
    set_tdc(seg, V, 0.0)
    tdc.voltage = torch.tensor([[V], [-V]], device=dev, dtype=torch.float64)
    calls = [(1, "start"), (17, "step 16"), (33, "final (step 32)"), (0, "goal reading")]
    rows = []
    for ep in (med_ep, worst_ep):
        panels = []
        for c, lab in calls:
            P = ParticleGroup(str(d / "bmad/beams" / f"c{c:03d}_e{ep:02d}_XTCAVF.h5"))
            w = np.unique(P.weight)
            isd = P.weight == w[-1]
            cloud = torch.tensor(np.stack([P.x, P.y, P.z, P.px, P.py, P.pz], 1), dtype=torch.float64, device=dev)[None]
            q = torch.tensor(np.asarray(P.weight, float), device=dev, dtype=torch.float64)[None]
            with torch.no_grad():
                out = seg.track(back_drift(cloud_to_beam(cloud, e_ref, q, device=torch.device(dev)), l_tdc))
            xp, xm = out.x[0, 0].cpu().numpy(), out.x[1, 0].cpu().numpy()
            yp, ym = out.y[0, 0].cpu().numpy(), out.y[1, 0].cpu().numpy()
            z = cloud[0, :, 2].cpu().numpy()
            S = float(calibrate_streak(torch.tensor(0.5 * (xp - xm))[None], torch.tensor(z)[None]))
            qn = q[0].cpu().numpy()
            true_sp = float(inter_bunch(cloud[:, isd], cloud[:, ~isd])["bunch_spacing"][0]) * 1e6 if (~isd).any() else np.nan
            ip, xc, yc, _ = render(xp, yp, qn)
            im, _, _, _ = render(xm, ym, qn)
            cut, ok = step_cut([ip])
            meas = None
            if ok:
                bp, bm = band_stats(ip, xc, cut_row=cut), band_stats(im, xc, cut_row=cut)
                if bp and bm:
                    meas = step_features([bp], [bm], streak=S, edge_frac_max=0.01)
            panels.append(dict(call=c, lab=lab, ip=ip, xc=xc, yc=yc, cut=cut if ok else None, meas=meas, true=true_sp,
                               qw_frac=float(qn[~isd].sum() / qn.sum())))
            print(f"ep {ep} call {c}: true {true_sp:.1f} um, img", None if meas is None else
                  (round(1e6 * meas["bunch_spacing"], 1), meas["valid"]), flush=True)
        tsp = 1e6 * traj["bunch_spacing_cloud"][:, ep].cpu().numpy()
        msp = 1e6 * traj["bunch_spacing_meas"][:, ep].cpu().numpy()
        gsp = 1e6 * traj["spacing_goal"][:, ep].cpu().numpy()
        gcl = 1e6 * traj["spacing_goal_cloud"][:, ep].cpu().numpy()
        rows.append(dict(ep=ep, err=err[ep], panels=panels, tsp=tsp, msp=msp, gsp=gsp, gcl=gcl,
                         valid=traj["screen_valid"][:, ep].cpu().numpy(),
                         clip=traj["screen_clipped_true"][:, ep].cpu().numpy(),
                         mis=traj["screen_misassign_true"][:, ep].cpu().numpy()))
    return rows, err


def fig_bmad(rows, err):
    fig, axes = plt.subplots(len(rows), 5, figsize=(16, 3.35 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1, 1, 1, 1.15]}, squeeze=False)
    for i, r in enumerate(rows):
        for j, p in enumerate(r["panels"]):
            ax = axes[i, j]
            ext = [1e3 * p["xc"][0], 1e3 * p["xc"][-1], 1e3 * p["yc"][0], 1e3 * p["yc"][-1]]
            ax.imshow(p["ip"], origin="lower", extent=ext, aspect="auto", cmap="magma")
            if p["cut"] is not None:
                ax.axhline(1e3 * p["yc"][p["cut"]], color=CUT, ls="--", lw=1.2)
            m = p["meas"]
            txt = ("no valley: invalid" if m is None else
                   f"image {1e6 * m['bunch_spacing']:.0f} µm{'' if m['valid'] else ' (INVALID'}"
                   f"{'' if m['valid'] else (', crop edge %.0f%%)' % (100 * m['edge_frac']) if m['edge_frac'] > 0.01 else ')')}")
            ax.text(0.03, 0.96, f"+V  {p['lab']}", transform=ax.transAxes, color="white", fontsize=8.5,
                    weight="bold", va="top")
            ax.text(0.03, 0.05, f"{txt}\ntrue {p['true']:.0f} µm", transform=ax.transAxes, color="white", fontsize=8,
                    va="bottom")
            ax.set_xlabel("x (streak) [mm]")
            if j == 0:
                ax.set_ylabel("y (energy) [mm]")
            ax.spines[:].set_visible(False)
        ax = axes[i, 4]
        s = np.arange(1, len(r["tsp"]) + 1)
        ax.axhspan(r["gcl"][-1] - 10, r["gcl"][-1] + 10, color=GRID, lw=0, zorder=0)
        ax.plot(s, r["tsp"], color=BLUE, lw=2, label="true (Bmad cloud)")
        ok = r["valid"] > 0.5
        ax.plot(s[ok], r["msp"][ok], "o", color=ORANGE, ms=3.5, label="screen reading (env, noisy)")
        if (~ok).any():
            ax.plot(s[~ok], r["msp"][~ok], "x", color=INK2, ms=4, label="invalid reading")
        ax.axhline(r["gsp"][-1], color=INK, ls="--", lw=1, label="goal reading")
        ax.set(xlabel="step", ylabel="bunch spacing [µm]", xlim=(0, 33))

        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_title(f"grey = goal truth ±10 µm; dashed = goal reading\nreading − truth {r['gsp'][-1] - r['gcl'][-1]:+.0f} µm\nenv screen: clipped ≤{100 * r['clip'].max():.0f}%, misassigned ≤{100 * r['mis'].max():.0f}%",
                     fontsize=8, loc="left")
        if i == 0:
            ax.legend(fontsize=7, frameon=False, loc="best")
        kind = "median episode" if i == 0 else "worst episode"
        axes[i, 0].set_title(f"{kind}: e{r['ep']:02d}, final true error {r['err']:.1f} µm", fontsize=9.5, loc="left",
                             weight="bold")
    fig.suptitle(f"Bmad closed loop (S2 direct-L1 + BLEN, covB2, injector variation, 24 eps; median final error "
                 f"{np.median(err):.1f} µm): simulated DTOTR2 +V images, golden optics, 22 MV; dashed = shared +V energy cut",
                 fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_bmad_xtcav_episode.png", dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------ 2. real XTCAV scan images
def fig_real(scan="TEST_15668", steps=(1, 7, 12, 17, 22)):
    from fig_bunch_split_changes_real import step_images
    keys, st, x, sp, v = scan_steps(scan)
    imgs = step_images(scan, list(steps))
    fig = plt.figure(figsize=(16, 6.6))
    gs = fig.add_gridspec(2, len(steps), height_ratios=[1, 0.85], hspace=0.45, wspace=0.12)
    for j, s in enumerate(steps):
        ax = fig.add_subplot(gs[0, j])
        D = imgs[s]
        P = D["plus"]
        med = np.median(P, axis=0)
        k, ok = step_cut(P, method="median")
        xm = 1e3 * (D["x_m"] - D["x_m"].mean())
        ny = med.shape[0]
        y_mm = 1e3 * PX * (np.arange(ny) - (ny - 1) / 2)
        ax.imshow(med, origin="lower", aspect="auto", cmap="magma", extent=[xm[0], xm[-1], y_mm[0], y_mm[-1]])
        if ok:
            ax.axhline(y_mm[k], color=CUT, ls="--", lw=1.2)
        r = st[str(s)]
        ax.text(0.03, 0.96, f"+V  step {s}, mkb {r['scan_val']:.2f}°", transform=ax.transAxes, color="white",
                fontsize=8.5, weight="bold", va="top")
        ax.text(0.03, 0.05, f"spacing {r['spacing_um']:.0f} µm\n{'valid' if r['valid'] else 'INVALID (<60 µm)'}"
                f" · median of {len(P)}", transform=ax.transAxes, color="white", fontsize=8, va="bottom")
        ax.set_xlabel("x (streak) [mm]")
        if j == 0:
            ax.set_ylabel("y (energy) [mm]")
        else:
            ax.set_yticklabels([])
        ax.spines[:].set_visible(False)
    ax = fig.add_subplot(gs[1, :])
    lo, hi = valid_window(v, x)
    ax.axvspan(lo, hi, color=GRID, lw=0, zorder=0)
    ax.axhspan(-60, 60, color="#f1f0ed", lw=0, zorder=0)
    ax.plot(x, sp, color=INK2, lw=1, zorder=1)
    ax.plot(x[v], sp[v], "o", color=BLUE, ms=5, label="valid step", zorder=3)
    ax.plot(x[~v], sp[~v], "o", mfc=SURF, mec=INK2, ms=5, label="invalid step", zorder=3)
    for s in steps:
        r = st[str(s)]
        ax.annotate(f"{s}", (r["scan_val"], r["spacing_um"]), xytext=(0, 8), textcoords="offset points", ha="center",
                    fontsize=8, weight="bold", color=INK)
    ax.text(lo + 0.05, ax.get_ylim()[0] if False else -45, f"valid window {lo:g}–{hi:g}°", fontsize=8, color=INK2)
    ax.text(x[-1], 57, "|spacing| < 60 µm: deployed-invalid", fontsize=7.5, color=INK2, ha="right", va="top")
    ax.set(xlabel="L2 multiknob (scan value) [deg]", ylabel="step spacing [µm]", ylim=(sp.min() - 20, sp.max() + 30))
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    ax.set_title(f"{scan}: step spacing (v6 online replay); numbered = imaged steps", fontsize=9, loc="left")
    fig.suptitle(f"Real DTOTR2 XTCAV, {scan} L2 scan: median +V images (deployed preprocessing), dashed = shared +V cut "
                 "(median of per-image valleys)", fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.savefig(OUT / "2026-10-17_real_xtcav_scan.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 3. DAQ replay
POL = [("delta_xtcav23_screen_v6_blen_s0", "xtcav23 blen_dlog s0"), ("injflow_flowA_screen_v6_blen_s0", "injflow A blen_dlog s0"),
       ("injflow_flowA_screen_v6_l1blen_s0", "injflow A L1+BLEN s0"), ("injflow_covB2_screen_v6_l1blen_s0", "covB2 L1+BLEN s0"),
       ("injflow_covB2_screen_v6_l1blen_s1", "covB2 L1+BLEN s1")]


def fig_replay():
    show = [("injflow_flowA_screen_v6_blen_s0", "blen_dlog L1 (injflow A s0)"),
            ("injflow_covB2_screen_v6_l1blen_s1", "direct L1 + BLEN obs (covB2 s1)")]
    ep_c = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(2, 5, width_ratios=[0.55, 1, 1, 0.5, 1.05], hspace=0.38, wspace=0.18)
    for i, scan in enumerate(SCANS):
        _, _, x, sp, v = scan_steps(scan)
        lo, hi = valid_window(v, x)
        ax0 = fig.add_subplot(gs[i, 0])
        ax0.axhspan(lo, hi, color=GRID, lw=0, zorder=0)
        ax0.plot(sp, x, color=INK2, lw=1)
        ax0.plot(sp[v], x[v], "o", color=INK, ms=3.5)
        ax0.plot(sp[~v], x[~v], "o", mfc=SURF, mec=INK2, ms=3.5)
        ymin = min(-0.4, *(np.load(DQR / f"{r}_{scan}" / "traj.npz")["l2_path"].min() - 0.3 for r, _ in show))
        ax0.set(xlabel="scan spacing [µm]", ylabel="L2 multiknob [deg]", ylim=(ymin, 8.4))
        ax0.set_title(f"{scan}\nscan; grey = valid", fontsize=9, loc="left", weight="bold")
        ax0.grid(axis="x", color=GRID, lw=0.6)
        for j, (run, lab) in enumerate(show):
            ax = fig.add_subplot(gs[i, 1 + j], sharey=ax0)
            z = np.load(DQR / f"{run}_{scan}" / "traj.npz")
            path = np.vstack([z["start"][None], z["l2_path"]])
            # five episodes spanning the goal range (sorted by goal)
            idx = np.argsort(z["goal"])[np.linspace(4, len(z["goal"]) - 5, 5).astype(int)]
            ax.axhspan(lo, hi, color=GRID, lw=0, zorder=0)
            t = np.arange(path.shape[0])
            for c, e in zip(ep_c, idx):
                ax.plot(t, path[:, e], color=c, lw=1.6)
                ax.plot([t[-1] - 1, t[-1] + 2], [z["goal"][e]] * 2, color=c, lw=3, solid_capstyle="butt")
                ax.plot(0, z["start"][e], "o", color=c, ms=4)
            err = z["sp_err"] * 1e6
            ax.set(xlabel="step", xlim=(-0.5, 34.5))
            ax.tick_params(labelleft=False)
            ax.grid(axis="y", color=GRID, lw=0.6)
            ax.set_title(f"{lab}\nall 64 eps: in-10 µm {np.mean(err <= 10):.0%}, median err {np.median(err):.1f} µm",
                         fontsize=8.5, loc="left")
    ax = fig.add_subplot(gs[:, 4])
    rows = [(f"{lab}", DQR / f"{run}_{{}}") for run, lab in POL] + \
           [(f"{lab}\n(env BC14 model)", DQR / "blen_model" / f"{run}_{{}}") for run, lab in POL if "l1blen" in run]
    y = np.arange(len(rows))[::-1].astype(float)
    hgt = 0.38
    for k, scan in enumerate(SCANS):
        vals = [json.loads((Path(str(p).format(scan)) / "summary.json").read_text())["frac_spacing_within_10um_meas"]
                for _, p in rows]
        yy = y + (hgt / 2 + 0.02) * (1 if k == 0 else -1)
        ax.barh(yy, vals, height=hgt, color=SCAN_C[scan], label=scan)
        for yv, val in zip(yy, vals):
            ax.text(val + 0.01, yv, f"{val:.2f}", va="center", fontsize=7.5, color=INK2)
    ax.axhline(y[len(POL) - 1] - 0.5, color=INK2, lw=0.6, ls=":")
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set(xlim=(0, 1.05), xlabel="fraction of 64 episodes within 10 µm of goal (measured)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("final measured spacing within 10 µm\n(below dotted: real BC14 BLEN replaced by the env model)",
                 fontsize=9, loc="left", weight="bold")
    fig.suptitle("L2-only replay of S2 screen controllers on real DAQ scans (L1 frozen; readings = scan step images, "
                 "bootstrapped). Paths: 5 episodes spanning the goal range, dot = start, bar = goal", fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.savefig(OUT / "2026-10-17_daq_replay.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ 4. BC14 BLEN real vs model
def fig_bc14():
    b = json.loads((DQR / "blen14_steps.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.6), sharey=True)
    for ax, scan in zip(axes, SCANS):
        _, _, xs, _, v = scan_steps(scan)
        lo, hi = valid_window(v, xs)
        x = np.asarray(b[scan]["scan_val"])
        L = np.log(np.asarray(b[scan]["blen14_median"]) / b[scan]["blen14_median"][0])
        d = np.linspace(x[0], x[-1], 200) - x[0]
        P = 0.199 * d + 0.005 * d ** 2
        ax.axvspan(lo, hi, color=GRID, lw=0, zorder=0, label="valid screen window")
        ax.fill_between(d + x[0], 0.7 * P, 1.5 * P, color=AQUA, alpha=0.25, lw=0, label="env model, slope scale 0.7–1.5")
        ax.plot(d + x[0], P, color=AQUA, lw=1.5, label="env model, scale 1")
        # the model as the replay evaluates it: quadratic centred at L2 = -37 (mkb 4), referenced to the scan start
        Pc = lambda m: 0.199 * (m - 4) + 0.005 * (m - 4) ** 2  # noqa: E731
        ax.plot(d + x[0], Pc(d + x[0]) - Pc(x[0]), color=AQUA, lw=1, ls="--", label="env model, centred at L2 = −37°")
        ax.plot(x, L, "o-", color=SCAN_C[scan], lw=2, ms=4, label=f"real BC14 BLEN ({scan})", zorder=4)
        ax.set(xlabel="L2 multiknob [deg]")
        ax.set_title(f"{scan}: ln(BLEN_LI14 / BLEN at scan start)", fontsize=9.5, loc="left", weight="bold")
        ax.grid(color=GRID, lw=0.6)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    axes[0].set_ylabel("ln BLEN ratio")
    fig.suptitle("Real BC14 BLEN vs the env's BLEN model (ln S = 0.199 d2 + 0.005 d2², d2 from scan start): "
                 "TEST_15668 is non-monotonic inside the valid window", fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-17_bc14_real_vs_model.png", dpi=100)
    plt.close(fig)


def main():
    which = set(sys.argv[1:]) or {"bmad", "real", "replay", "bc14"}
    if "bc14" in which:
        fig_bc14()
    if "replay" in which:
        fig_replay()
    if "real" in which:
        fig_real()
    if "bmad" in which:
        fig_bmad(*bmad_screens())


if __name__ == "__main__":
    main()
