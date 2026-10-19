"""Figures for the 2026-10-18 notebook entry on the accuracy of the XTCAVF two-bunch flow surrogate (37 knobs):
B2-cov vs B2-cov + spacing head vs Bmad truth on the pre-registered held-out TEST rows of the cov table.
Writes lab-notebook/images/2026-10-18_surrogate_*.png. Read-only on data/, processed/, trained/, results/.

  PYTHONPATH=src OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= python scripts/notebook_figs_2026-10-18_surrogate.py [--cache DIR]

Observables (parity / error maps / pairs) follow scripts/surrogate_scorecard.py exactly: table truth clouds
(1024 particles, checkpoint scaler), CRN prediction (seed 0, n = 2048), properties.per_bunch / inter_bunch for both.
Corner / LPS panels use the raw Bmad XTCAVF beams (all 50k macroparticles) vs 20k-particle model draws.
All sampling on CPU (the GPUs are busy with RL training).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.ndimage import gaussian_filter

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MUTED = "#8a8984"
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
# one colour per source, in every figure
C_TRUTH, C_COV, C_ZS = INK2, ORANGE, BLUE
L_TRUTH, L_COV, L_ZS = "Bmad", "B2-cov", "B2-cov + spacing head"
# drive vs witness: solid / filled = drive, dashed / outline = witness (same source colour)
LS_D, LS_W = "-", "--"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
GREY_RAMP = ["#e4e3df", "#c8c7c1", "#a8a79f", "#8a8984", "#6b6a66", "#52514e", "#3a3936", "#1f1f1d"]
CM_ZS = LinearSegmentedColormap.from_list("zs", [SURF] + BLUE_RAMP)
CM_TRUTH = LinearSegmentedColormap.from_list("truth", [SURF] + GREY_RAMP)
CM_DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#6da7ec", "#f0efec", "#ec8a89", "#b8302f"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 9,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
                     "axes.titlecolor": INK, "text.color": INK})

TABLE = "processed/xtcav23e_inj_cov_2026-10-17.h5"
ROLE_MAP = "processed/role_map_ftc_s0_2026-10-16.json"
CK_COV = "trained/xtcav23e_inj_covB2_2026-10-17/checkpoints/best-epoch=034-val_paired_delta_rms_um=5.4293.ckpt"
CK_ZS = "trained/xtcav23e_inj_covB2zs_2026-10-18/checkpoints/best-step=09500-val_paired_delta_rms_um=4.6991.ckpt"
KEEP = ("bunch_spacing", "drive_sigma_z", "witness_sigma_z", "energy_difference", "witness_norm_emit_x",
        "drive_mean_energy", "witness_mean_energy", "T_drive", "T_witness")
C_LIGHT = 299792458.0
MC2 = 510998.95


def suptitle(fig, s, y=0.995):
    fig.suptitle(s, fontsize=10.5, x=0.01, y=y, ha="left", weight="bold", color=INK)


def r2(p, t):
    return float(1 - np.sum((p - t) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30))


# ---------------------------------------------------------------------------------------------------- data
def load_models():
    import torch
    from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 8)))
    return {k: TwoBunchFlow.load_from_checkpoint(p, map_location="cpu").eval() for k, p in (("cov", CK_COV), ("zs", CK_ZS))}


def datamodule(model):
    from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule
    from twobunch_s2e_rl.surrogate.train import SCALER_KEYS
    scaler = {k: getattr(model, k).tolist() for k in SCALER_KEYS}   # = the checkpoint's state-dict scaler
    roles = json.loads(Path(ROLE_MAP).read_text())["roles"]
    dm = TwoBunchFlowDataModule(processed_h5=TABLE, split="group", orig_split=roles, scaler=scaler)
    dm.setup()
    return dm


def compute(cache: Path, models):
    """Truth + CRN predictions on the TEST rows and on the references of the TEST pairs (as the scorecard)."""
    import torch
    from twobunch_s2e_rl.surrogate.properties import inter_bunch, per_bunch
    f = cache / "obs.npz"
    if f.exists():
        return dict(np.load(f, allow_pickle=True))
    dm = datamodule(models["zs"])
    ds = dm.full
    with h5py.File(TABLE) as h:
        t = {k: h[k][...] for k in ("campaign", "idx", "pair_key", "pair_ref", "drive_q_in_nC", "witness_q_in_nC")}
        names = str(h.attrs["campaigns"]).split(",")
    test = np.asarray(dm.test.indices)
    dd, wd = ds.drive_density.numpy(), ds.witness_density.numpy()
    both = dd & wd
    is_inj = np.isin(t["campaign"], [j for j, n in enumerate(names) if n.startswith("injector")])
    by = {}
    for r in np.flatnonzero(is_inj & both):
        if t["pair_key"][r] >= 0:
            by.setdefault(int(t["pair_key"][r]), {})["ref" if t["pair_ref"][r] else "pt"] = r
    tset = set(test.tolist())
    pairs = np.array([(v["pt"], v["ref"]) for v in by.values() if len(v) == 2 and v["pt"] in tset])
    rows = np.unique(np.r_[test, pairs.ravel()])
    m0 = models["zs"]
    sd, sw = (m0.drive_mean.double(), m0.drive_std.double()), (m0.witness_mean.double(), m0.witness_std.double())
    d = ds.drive[rows].double() * sd[1] + sd[0]
    w = ds.witness[rows].double() * sw[1] + sw[0]
    T = {}
    for nm, parts in (("drive", d), ("witness", w)):
        for kk, vv in per_bunch(parts).items():
            T[f"{nm}_{kk}"] = vv.numpy()
    T.update({k: v.numpy() for k, v in inter_bunch(d, w).items()})
    T["T_drive"], T["T_witness"] = ds.drive_frac[rows].numpy(), ds.witness_frac[rows].numpy()
    out = {f"true_{k}": T[k] for k in KEEP}
    for name, m in models.items():
        g = torch.Generator().manual_seed(0)
        zd, zw = torch.randn(2048, m.latent_dim, generator=g), torch.randn(2048, m.latent_dim, generator=g)
        acc = {}
        with torch.no_grad():
            for b in range(0, len(rows), 64):
                o = m.observables(ds.knobs[rows[b:b + 64]], n=2048, z_drive=zd, z_witness=zw)
                for k in KEEP:
                    acc.setdefault(k, []).append(o[k].double().numpy())
        for k in KEEP:
            out[f"{name}_{k}"] = np.concatenate(acc[k])
        print("predicted", name, flush=True)
    lo, hi = np.array(dm.norm["knob_low"]), np.array(dm.norm["knob_high"])
    out.update(rows=rows, test=test, pairs=pairs, knobs_norm=ds.knobs[rows].numpy(),
               knobs_phys=lo + ds.knobs[rows].numpy() * (hi - lo), knob_keys=np.array(dm.norm["knob_keys"]),
               dd=dd[rows], wd=wd[rows], viable=ds.witness_viable[rows].numpy(),
               campaign=np.array([names[c] for c in t["campaign"][rows]]), idx=t["idx"][rows],
               q_in_d=t["drive_q_in_nC"][rows], q_in_w=t["witness_q_in_nC"][rows])
    cache.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return dict(np.load(f, allow_pickle=True))


def raw_bunches(campaign, idx):
    """Raw Bmad XTCAVF beam split by weight (drive = higher). Returns {k: (coords (n,6), weights [C])}."""
    from beamphysics import ParticleGroup
    P = ParticleGroup(f"data/{campaign}/sample_{int(idx):05d}_XTCAVF.h5")
    P = P[P.status == 1]
    w = np.unique(P.weight)
    out = {}
    for k, wk in ((0, w[-1]), (1, w[0])):
        s = P[P.weight == wk]
        out[k] = (np.stack([s.x, s.y, s.z, s.px, s.py, s.pz], 1), s.weight)
    return out


def model_bunches(model, knobs_norm, q_in, n=20000, seed=1):
    import torch
    g = torch.Generator().manual_seed(seed)
    kn = torch.tensor(knobs_norm[None], dtype=torch.float32)
    with torch.no_grad():
        _, td, tw = model.feasibility(kn)
        out = {}
        for k, T in ((0, td), (1, tw)):
            c = model.sample_bunch(kn, k, n, generator=g)[0].double().numpy()
            out[k] = (c, np.full(n, float(T) * q_in[k] * 1e-9 / n))
    return out


def phase_coords(b):
    """x [um], x' [urad], y [um], y' [urad], z [um], delta [%], centred on the source's DRIVE centroid / mean pz."""
    c0, w0 = b[0]
    ref = np.average(c0, axis=0, weights=w0)
    xp0, yp0 = np.average(c0[:, 3] / c0[:, 5], weights=w0), np.average(c0[:, 4] / c0[:, 5], weights=w0)
    out = {}
    for k, (c, w) in b.items():
        out[k] = (np.stack([(c[:, 0] - ref[0]) * 1e6, (c[:, 3] / c[:, 5] - xp0) * 1e6, (c[:, 1] - ref[1]) * 1e6,
                            (c[:, 4] / c[:, 5] - yp0) * 1e6, (c[:, 2] - ref[2]) * 1e6, (c[:, 5] / ref[5] - 1) * 100], 1), w)
    return out


# ---------------------------------------------------------------------------------------------------- selection
def healthy(D):
    """held-out TEST rows with both bunches dense and the witness essentially fully transmitted (T_w >= 0.95)."""
    tset = set(D["test"].tolist())
    return np.array([i for i, r in enumerate(D["rows"]) if r in tset and D["dd"][i] and D["wd"][i]
                     and D["true_T_witness"][i] >= 0.95])


def pick_rows(D):
    h = healthy(D)
    sp = D["true_bunch_spacing"][h]
    err = np.abs(D["zs_bunch_spacing"][h] - sp)
    # representative: spacing within the central 40%, and BOTH the spacing error and the drive sigma_z relative
    # error closest (in rank) to their medians over the healthy test rows
    ez = np.abs(D["zs_drive_sigma_z"][h] / D["true_drive_sigma_z"][h] - 1)
    rk = lambda v: np.argsort(np.argsort(v)) / (len(v) - 1)      # noqa: E731
    score = np.maximum(np.abs(rk(err) - 0.5), np.abs(rk(ez) - 0.5))
    score[(sp <= np.percentile(sp, 30)) | (sp >= np.percentile(sp, 70))] = np.inf
    rep = h[np.argmin(score)]
    # contrasting: the most strongly compressed witness among the healthy test rows
    comp = h[np.argmin(D["true_witness_sigma_z"][h])]
    # LPS: nearest to the 5 / 35 / 65 / 95 % spacing quantiles (no selection on error)
    lps = [h[np.argmin(np.abs(sp - np.percentile(sp, q)))] for q in (5, 35, 65, 95)]
    return rep, comp, lps


# ---------------------------------------------------------------------------------------------------- figures
def hpd_levels(H, fracs):
    s = np.sort(H.ravel())[::-1]
    c = np.cumsum(s) / s.sum()
    return sorted(s[min(np.searchsorted(c, f), len(s) - 1)] for f in fracs)


def fig_corner(D, models, i, fname, title):
    names = ["x [µm]", "x′ [µrad]", "y [µm]", "y′ [µrad]", "z [µm]", "δ [%]"]
    q_in = (D["q_in_d"][i], D["q_in_w"][i])
    tb = phase_coords(raw_bunches(D["campaign"][i], D["idx"][i]))
    mb = phase_coords(model_bunches(models["zs"], D["knobs_norm"][i], q_in))
    allc = np.concatenate([tb[0][0], tb[1][0]])
    lim = []
    for j in range(6):
        lo, hi = np.percentile(allc[:, j], [0.7, 99.3])
        pad = 0.12 * (hi - lo)
        lim.append((lo - pad, hi + pad))
    fig, axes = plt.subplots(6, 6, figsize=(12.5, 12.5))
    nb = 70
    for r in range(6):
        for c in range(6):
            ax = axes[r, c]
            if c > r:
                ax.axis("off")
                continue
            if r == c:
                bins = np.linspace(*lim[c], 90)
                for src, b, col in (("t", tb, C_TRUTH), ("m", mb, C_ZS)):
                    for k, ls in ((0, LS_D), (1, LS_W)):
                        x, w = b[k][0][:, c], b[k][1]
                        h, e = np.histogram(x, bins, weights=w)
                        h = h / np.diff(e) * 1e9          # nC per unit
                        if src == "t":
                            if k == 0:
                                ax.fill_between(e[:-1], h, step="post", color=col, alpha=0.22, lw=0)
                            ax.step(e[:-1], h, where="post", color=col, lw=1.0, ls=ls)
                        else:
                            ax.step(e[:-1], h, where="post", color=col, lw=1.3, ls=ls)
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
                ax.set_xlim(lim[c])
            else:
                for src, b, col in (("t", tb, C_TRUTH), ("m", mb, C_ZS)):
                    for k, ls in ((0, LS_D), (1, LS_W)):
                        xy = b[k][0]
                        H, xe, ye = np.histogram2d(xy[:, c], xy[:, r], nb, range=[lim[c], lim[r]])
                        H = gaussian_filter(H, 1.3)
                        if H.sum() <= 0:
                            continue
                        lv = hpd_levels(H, (0.9, 0.5))
                        if len(set(lv)) < 2:
                            continue
                        xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
                        if src == "t" and k == 0:
                            ax.contourf(xc, yc, H.T, levels=[lv[0], lv[1], H.max() * 1.01],
                                        colors=["#e4e3df", "#c8c7c1"], zorder=1)
                        ax.contour(xc, yc, H.T, levels=lv, colors=[col], linestyles=ls,
                                   linewidths=0.9 if src == "t" else 1.15, zorder=3 if src == "m" else 2)
                ax.set_xlim(lim[c])
                ax.set_ylim(lim[r])
            if r == 5:
                ax.set_xlabel(names[c], fontsize=9)
            else:
                ax.set_xticklabels([])
            if c == 0 and r > 0:
                ax.set_ylabel(names[r], fontsize=9)
            elif c != r:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7.5)
    h_ = [Patch(facecolor="#c8c7c1", edgecolor=C_TRUTH, label=f"{L_TRUTH} drive (50 / 90% contours)"),
          Line2D([], [], color=C_TRUTH, ls=LS_W, label=f"{L_TRUTH} witness"),
          Line2D([], [], color=C_ZS, ls=LS_D, lw=1.3, label=f"{L_ZS} drive"),
          Line2D([], [], color=C_ZS, ls=LS_W, lw=1.3, label=f"{L_ZS} witness")]
    fig.legend(handles=h_, loc="upper right", bbox_to_anchor=(0.97, 0.93), fontsize=9.5)
    ts, ps = D["true_bunch_spacing"][i] * 1e6, D["zs_bunch_spacing"][i] * 1e6
    tz, pz = D["true_drive_sigma_z"][i] * 1e6, D["zs_drive_sigma_z"][i] * 1e6
    twz, pwz = D["true_witness_sigma_z"][i] * 1e6, D["zs_witness_sigma_z"][i] * 1e6
    kp = D["knobs_phys"][i]
    kk = list(D["knob_keys"])
    info = (f"held-out TEST row: {D['campaign'][i]} / sample {int(D['idx'][i])}\n"
            f"L1 {kp[kk.index('L1PhaseSet')]:.2f}°, L2 {kp[kk.index('L2PhaseSet')]:.2f}°, "
            f"drive fraction {kp[kk.index('drive_fraction')]:.2f}, gun phase {kp[kk.index('GUNF:theta0_deg')]:.1f}°\n\n"
            f"                      Bmad     model\n"
            f"spacing [µm]       {ts:7.1f}   {ps:7.1f}\n"
            f"drive σz [µm]      {tz:7.1f}   {pz:7.1f}\n"
            f"witness σz [µm]    {twz:7.1f}   {pwz:7.1f}\n\n"
            f"Bmad: all {int(raw_n(tb))} macroparticles; model: 20k / bunch.\n"
            f"Each source centred on its own drive centroid\n(δ relative to its drive mean pz); diagonal = charge density.")
    fig.text(0.56, 0.78, info, fontsize=9, family="monospace", va="top", color=INK2)
    suptitle(fig, title, y=0.985)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.06, top=0.955, wspace=0.08, hspace=0.08)
    fig.savefig(OUT / fname, dpi=140)
    plt.close(fig)


def raw_n(b):
    return sum(len(b[k][1]) for k in b)


def fig_lps(D, models, lps):
    fig, axes = plt.subplots(3, 4, figsize=(16, 10.2), gridspec_kw={"height_ratios": [0.75, 1, 1]})
    for j, i in enumerate(lps):
        q_in = (D["q_in_d"][i], D["q_in_w"][i])
        tb = phase_coords(raw_bunches(D["campaign"][i], D["idx"][i]))
        mb = phase_coords(model_bunches(models["zs"], D["knobs_norm"][i], q_in, n=50000))
        allz = np.concatenate([tb[0][0][:, 4], tb[1][0][:, 4]])
        alld = np.concatenate([tb[0][0][:, 5], tb[1][0][:, 5]])
        zl = np.percentile(allz, [0.3, 99.7]); zl = zl + np.array([-1, 1]) * 0.1 * np.ptp(zl)
        dl = np.percentile(alld, [0.3, 99.7]); dl = dl + np.array([-1, 1]) * 0.15 * np.ptp(dl)
        bins = np.linspace(*zl, 160)
        ax = axes[0, j]
        for b, col, lw in ((tb, C_TRUTH, 1.2), (mb, C_ZS, 1.2)):
            for k, ls in ((0, LS_D), (1, LS_W)):
                h, e = np.histogram(b[k][0][:, 4], bins, weights=b[k][1])
                I = h / (np.diff(e) * 1e-6) * C_LIGHT / 1e3       # kA
                ax.step(e[:-1], I, where="post", color=col, ls=ls, lw=lw)
        ax.set_xlim(zl)
        ax.set_ylim(bottom=0)
        ax.set_xticklabels([])
        ax.grid(axis="y", color=GRID, lw=0.6)
        if j == 0:
            ax.set_ylabel("current I(z) [kA]")
        ts, ps = D["true_bunch_spacing"][i] * 1e6, D["zs_bunch_spacing"][i] * 1e6
        tz, pz = D["true_drive_sigma_z"][i] * 1e6, D["zs_drive_sigma_z"][i] * 1e6
        ax.set_title(f"spacing  Bmad {ts:.0f} / model {ps:.0f} µm\ndrive σz  Bmad {tz:.1f} / model {pz:.1f} µm",
                     fontsize=9, loc="left")
        for r, (b, cm, lab) in enumerate(((tb, CM_TRUTH, L_TRUTH), (mb, CM_ZS, L_ZS)), start=1):
            ax = axes[r, j]
            z = np.concatenate([b[0][0][:, 4], b[1][0][:, 4]])
            dd = np.concatenate([b[0][0][:, 5], b[1][0][:, 5]])
            ww = np.concatenate([b[0][1], b[1][1]])
            H, xe, ye = np.histogram2d(z, dd, [160, 140], range=[zl, dl], weights=ww)
            H = gaussian_filter(H, 0.7)
            Hm = np.ma.masked_less_equal(H, H.max() * 2e-3)
            ax.pcolormesh(xe, ye, np.log10(Hm.T), cmap=cm, shading="auto", rasterized=True,
                          vmin=np.log10(H.max() * 2e-3), vmax=np.log10(H.max()))
            for k, lab_k in ((0, "D"), (1, "W")):
                cz = np.average(b[k][0][:, 4], weights=b[k][1])
                cd = np.average(b[k][0][:, 5], weights=b[k][1])
                ax.annotate(lab_k, (cz, cd), xytext=(0, 14), textcoords="offset points", ha="center",
                            fontsize=9, weight="bold", color=INK,
                            bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=0.8))
            ax.set_xlim(zl)
            ax.set_ylim(dl)
            ax.text(0.02, 0.96, lab, transform=ax.transAxes, fontsize=8.5, va="top", color=INK2,
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))
            if j == 0:
                ax.set_ylabel("δ = pz / p0,drive − 1  [%]")
            if r == 2:
                ax.set_xlabel("z − ⟨z_drive⟩ [µm]   (head →)")
            else:
                ax.set_xticklabels([])
        axes[1, j].text(0.98, 0.04, f"TEST row {D['campaign'][i].replace('injector_', 'inj_')} #{int(D['idx'][i])}",
                        transform=axes[1, j].transAxes, ha="right", va="bottom", fontsize=7.5, color=MUTED)
    h_ = [Line2D([], [], color=C_TRUTH, ls=LS_D, label=f"{L_TRUTH} drive"),
          Line2D([], [], color=C_TRUTH, ls=LS_W, label=f"{L_TRUTH} witness"),
          Line2D([], [], color=C_ZS, ls=LS_D, label=f"{L_ZS} drive"),
          Line2D([], [], color=C_ZS, ls=LS_W, label=f"{L_ZS} witness")]
    fig.legend(handles=h_, loc="upper left", ncol=4, bbox_to_anchor=(0.005, 0.965), fontsize=9)
    suptitle(fig, "Longitudinal phase space at XTCAVF, held-out TEST rows at the 5 / 35 / 65 / 95% spacing quantiles "
                  "(log density; model charge = predicted T × injected charge)", y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.06, top=0.865, wspace=0.13, hspace=0.07)
    fig.savefig(OUT / "2026-10-18_surrogate_lps.png", dpi=125)
    plt.close(fig)


def fig_parity(D):
    tset = set(D["test"].tolist())
    te = np.array([r in tset for r in D["rows"]])
    both = te & D["dd"] & D["wd"]
    panels = [("bunch_spacing", "bunch spacing [µm]", 1e6, both),
              ("drive_sigma_z", "drive σz [µm]", 1e6, te & D["dd"]),
              ("witness_sigma_z", "witness σz [µm]", 1e6, te & D["wd"]),
              ("energy_difference", "energy difference E_d − E_w [MeV]", 1e-6, both),
              ("T_witness", "witness transmission T_w", 1.0, te & D["viable"]),
              ("witness_norm_emit_x", "witness ε_n,x [µm·rad]", 1e6, te & D["wd"])]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 9.4))
    for ax, (k, lab, s, m) in zip(axes.ravel(), panels):
        t = D[f"true_{k}"][m] * s
        pz, pc = D[f"zs_{k}"][m] * s, D[f"cov_{k}"][m] * s
        same = np.allclose(pz, pc, rtol=1e-6, atol=1e-9 * max(np.abs(t).max(), 1))
        lo, hi = np.percentile(np.r_[t, pz], [0.2, 99.8])
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad
        if k == "T_witness":
            lo, hi = min(lo, 0.5), 1.02
        ax.plot([lo, hi], [lo, hi], color=MUTED, lw=0.8, ls=":", zorder=1)
        if not same:
            ax.scatter(t, pc, s=9, color=C_COV, alpha=0.45, lw=0, zorder=2)
        ax.scatter(t, pz, s=11, color=C_ZS, alpha=0.75, lw=0, zorder=3)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=f"Bmad {lab}", ylabel=f"model {lab}")
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        unit = lab[lab.find("[") + 1:lab.find("]")] if "[" in lab else ""
        fmt = (lambda v: f"{v:.3f}") if k == "T_witness" else (lambda v: f"{v:.2f}")
        txt = f"{L_ZS}: R² {r2(pz, t):.3f}, MAE {fmt(np.mean(np.abs(pz - t)))} {unit}"
        if not same:
            txt += f"\n{L_COV}: R² {r2(pc, t):.3f}, MAE {fmt(np.mean(np.abs(pc - t)))} {unit}"
        else:
            txt += f"\n(identical for {L_COV}: the head moves only z centroids)"
        ax.text(0.03, 0.97, txt + f"\nn = {m.sum()} test rows", transform=ax.transAxes, va="top", fontsize=8,
                color=INK2, bbox=dict(boxstyle="round,pad=0.3", fc=SURF, ec="none", alpha=0.9))
        ax.set_title(lab.split(" [")[0], fontsize=9.5, loc="left")
    h_ = [Line2D([], [], marker="o", ls="", color=C_ZS, label=L_ZS),
          Line2D([], [], marker="o", ls="", color=C_COV, alpha=0.6, label=L_COV)]
    fig.legend(handles=h_, loc="upper right", ncol=2, bbox_to_anchor=(0.995, 0.995), fontsize=9)
    suptitle(fig, "Parity on the held-out TEST rows (cov table, CRN n = 2048): controller observables")
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.07, top=0.92, wspace=0.28, hspace=0.3)
    fig.savefig(OUT / "2026-10-18_surrogate_parity.png", dpi=125)
    plt.close(fig)


def binned(ax, x, y, col, nb=8):
    e = np.percentile(x, np.linspace(0, 100, nb + 1))
    c, md, p10, p90 = [], [], [], []
    for a, b in zip(e[:-1], e[1:]):
        m = (x >= a) & (x <= b)
        if m.sum() < 5:
            continue
        c.append(np.median(x[m])); md.append(np.median(y[m]))
        p10.append(np.percentile(y[m], 10)); p90.append(np.percentile(y[m], 90))
    ax.fill_between(c, p10, p90, color=col, alpha=0.15, lw=0)
    ax.plot(c, md, color=col, lw=2)


def fig_error_maps(D):
    tset = set(D["test"].tolist())
    te = np.array([r in tset for r in D["rows"]])
    m = te & D["dd"] & D["wd"]
    kk = list(D["knob_keys"])
    kp = D["knobs_phys"][m]
    L1, L2 = kp[:, kk.index("L1PhaseSet")], kp[:, kk.index("L2PhaseSet")]
    df, gp = kp[:, kk.index("drive_fraction")], kp[:, kk.index("GUNF:theta0_deg")]
    es = (D["zs_bunch_spacing"][m] - D["true_bunch_spacing"][m]) * 1e6
    ez = (D["zs_drive_sigma_z"][m] / D["true_drive_sigma_z"][m] - 1) * 100
    tsz = D["true_drive_sigma_z"][m] * 1e6
    tsp = D["true_bunch_spacing"][m] * 1e6
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.4), gridspec_kw={"width_ratios": [1.25, 1, 1, 1]})
    for r, (e, lab, unit) in enumerate(((es, "spacing error (model − Bmad)", "µm"), (ez, "drive σz error (model / Bmad − 1)", "%"))):
        v = np.percentile(np.abs(e), 95)
        ax = axes[r, 0]
        o = np.argsort(np.abs(e))
        sc = ax.scatter(L1[o], L2[o], c=e[o], cmap=CM_DIV, norm=TwoSlopeNorm(0, -v, v), s=22, lw=0.3, edgecolor=SURF)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, extend="both")
        cb.set_label(f"{lab} [{unit}]", fontsize=8.5)
        cb.outline.set_visible(False)
        ax.set(xlabel="L1PhaseSet [°]", ylabel="L2PhaseSet [°]")
        ax.set_title(f"{lab}, by linac phases", fontsize=9.5, loc="left")
        for c, (x, xl) in enumerate(((df, "drive fraction"), (gp, "gun phase GUNF:theta0 [°]"),
                                     (tsp if r == 0 else tsz, "Bmad spacing [µm]" if r == 0 else "Bmad drive σz [µm]")), start=1):
            ax = axes[r, c]
            ax.axhline(0, color=MUTED, lw=0.8, ls=":")
            ax.scatter(x, e, s=8, color=C_ZS, alpha=0.35, lw=0)
            binned(ax, x, e, C_ZS)
            ax.set(xlabel=xl, ylabel=f"{lab.split(' (')[0]} [{unit}]")
            lim = np.percentile(np.abs(e), 99.5) * 1.1
            ax.set_ylim(-lim, lim)
            ax.grid(color=GRID, lw=0.5)
    axes[0, 1].text(0.02, 0.97, "line: binned median, band: p10–p90", transform=axes[0, 1].transAxes, fontsize=8,
                    va="top", color=INK2)
    suptitle(fig, f"Where {L_ZS} is worse: signed error on {m.sum()} held-out TEST rows vs operating point "
                  f"(spacing MAE {np.mean(np.abs(es)):.1f} µm, drive σz mean |rel| {np.mean(np.abs(ez)):.1f}%)")
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.08, top=0.92, wspace=0.3, hspace=0.32)
    fig.savefig(OUT / "2026-10-18_surrogate_error_maps.png", dpi=110)
    plt.close(fig)
    return dict(es=es, ez=ez, L1=L1, L2=L2, tsz=tsz)


def fig_pairs(D):
    pos = {r: i for i, r in enumerate(D["rows"])}
    P = D["pairs"]
    a, b = np.array([pos[x] for x in P[:, 0]]), np.array([pos[x] for x in P[:, 1]])
    t = (D["true_bunch_spacing"][a] - D["true_bunch_spacing"][b]) * 1e6
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    lim = np.percentile(np.abs(t), 99.5) * 1.25
    res = {}
    for ax, (nm, lab, col) in zip(axes[:2], (("cov", L_COV, C_COV), ("zs", L_ZS, C_ZS))):
        p = (D[f"{nm}_bunch_spacing"][a] - D[f"{nm}_bunch_spacing"][b]) * 1e6
        sl, ic = np.polyfit(t, p, 1)
        res[nm] = (r2(p, t), sl, np.percentile(np.abs(p - t), 90), p)
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=":")
        ax.plot([-lim, lim], [sl * -lim + ic, sl * lim + ic], color=col, lw=1.2, alpha=0.8)
        ax.scatter(t, p, s=16, color=col, alpha=0.75, lw=0)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="Bmad Δspacing (point − golden injector) [µm]",
               ylabel="model Δspacing [µm]")
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.set_title(lab, fontsize=10, loc="left")
        ax.text(0.03, 0.97, f"R² {res[nm][0]:.3f}\nslope {sl:.3f}  (gate 0.85–1.15)\np90 |err| {res[nm][2]:.1f} µm"
                            f"\nn = {len(t)} test pairs", transform=ax.transAxes, va="top", fontsize=8.5, color=INK2)
    ax = axes[2]
    ax.axhline(0, color=MUTED, lw=0.8, ls=":")
    for nm, col, lab in (("cov", C_COV, L_COV), ("zs", C_ZS, L_ZS)):
        e = res[nm][3] - t
        ax.scatter(t, e, s=12, color=col, alpha=0.5 if nm == "zs" else 0.4, lw=0)
        binned(ax, t, e, col, nb=7)
    ax.set(xlabel="Bmad Δspacing [µm]", ylabel="model − Bmad Δspacing [µm]")
    ax.grid(color=GRID, lw=0.5)
    ax.set_title("residual vs true injector effect (line: binned median)", fontsize=10, loc="left")
    h_ = [Line2D([], [], marker="o", ls="", color=C_COV, label=L_COV), Line2D([], [], marker="o", ls="", color=C_ZS, label=L_ZS)]
    ax.legend(handles=h_, loc="upper right", fontsize=8.5)
    suptitle(fig, "Injector effect at fixed linac knobs: Δspacing (injector point − golden reference) on the held-out "
                  "TEST pairs — the spacing head removes most of the slope shrinkage", y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.86, wspace=0.25)
    fig.savefig(OUT / "2026-10-18_surrogate_pairs.png", dpi=120)
    plt.close(fig)
    return {k: v[:3] for k, v in res.items()}


def fig_gates():
    S = "results/surrogate_scorecard_2026-10-1"
    ld = lambda p: json.loads(Path(S + p).read_text())["models"]     # noqa: E731
    b16, t17, t18, c17, c18 = (ld("6/B/scorecard.json"), ld("7/topup_table/scorecard.json"), ld("8/zs_topup_table/scorecard.json"),
                               ld("7/cov_table/scorecard.json"), ld("8/zs_cov_table/scorecard.json"))
    lin = ["ftc_s0", "A", "B2", "B2-cov", "+ spacing\nhead"]
    topup = [b16["ftc_s0 best"], b16["A best"], b16["B2 (top-up, w_pair 3) best"], t17["B2-cov best"], t18["B2-cov + spacing head"]]
    cov = [None, None, c17["B2 best"], c17["B2-cov best"], c18["B2-cov + spacing head"]]
    col = [MUTED, MUTED, MUTED, C_COV, C_ZS]
    panels = [("paired_test", "R2", "test-pair Δspacing R²", (0.8, None)),
              ("paired_test", "slope", "test-pair Δspacing slope", (0.85, 1.15)),
              ("paired_test", "p90_um", "test-pair Δspacing p90 |err| [µm]", (None, 8.0)),
              ("gate_oldval", "spacing_mae_um", "old-val absolute spacing MAE [µm]", (None, 9.136 + 0.3)),
              ("gate_oldval", "sigma_z_rel", "old-val drive σz mean |rel err|", (None, 0.1069 + 0.005)),
              ("T_calibration_mae", None, "T calibration MAE (T_true ≥ 0.95)", (None, 0.02))]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x = np.arange(len(lin))
    for ax, (sec, key, title, (glo, ghi)) in zip(axes.ravel(), panels):
        vals = []
        for off, tab, mk in ((-0.12, topup, "o"), (0.12, cov, "s")):
            for j, m in enumerate(tab):
                if m is None:
                    continue
                v = m[sec] if key is None else m[sec][key]
                vals.append(v)
                if sec == "paired_test":
                    lo, hi = m[sec]["ci90"][key]
                    vals += [lo, hi]
                    ax.errorbar(x[j] + off, v, yerr=[[v - lo], [hi - v]], fmt=mk, color=col[j], ms=6, capsize=2.5,
                                lw=1.2, mfc=col[j] if mk == "o" else SURF, mew=1.4)
                else:
                    ax.plot(x[j] + off, v, mk, color=col[j], ms=6, mfc=col[j] if mk == "o" else SURF, mew=1.4)
        y0, y1 = min(vals), max(vals)
        g = [v for v in (glo, ghi) if v is not None]
        y0, y1 = min([y0] + g), max([y1] + g)
        pad = 0.12 * (y1 - y0)
        y0, y1 = y0 - pad, y1 + pad
        # gate band = the PASS region (light neutral fill), gate edges dotted
        ax.axhspan(glo if glo is not None else y0 - 10, ghi if ghi is not None else y1 + 10, color="#f0efec", zorder=0)
        for gv in g:
            ax.axhline(gv, color=INK2, ls=":", lw=1)
        ax.set_ylim(y0, y1)
        ax.set_xticks(x, lin, fontsize=8.5)
        ax.set_xlim(-0.5, len(lin) - 0.5)
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.5)
    h_ = [Line2D([], [], marker="o", ls="", color=INK2, label="top-up table (117 test pairs; ftc_s0/A/B2 from 10-16/B, "
                                                               "B2-cov 10-17/topup_table, head 10-18/zs_topup_table)"),
          Line2D([], [], marker="s", ls="", color=INK2, mfc=SURF, mew=1.4,
                 label="cov table (167 test pairs; B2/B2-cov 10-17/cov_table, head 10-18/zs_cov_table)"),
          Patch(facecolor="#f0efec", edgecolor=INK2, ls=":", label="gate pass region (90% bootstrap CI on pair metrics)")]
    fig.legend(handles=h_, loc="lower center", ncol=1, fontsize=8.5, bbox_to_anchor=(0.5, 0.0))
    suptitle(fig, "Gate summary across the flow lineage: the spacing head clears slope and p90 on held-out pairs; "
                  "absolute spacing MAE costs 0.7 µm (still far inside its gate)")
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.17, top=0.91, wspace=0.22, hspace=0.38)
    fig.savefig(OUT / "2026-10-18_surrogate_gates.png", dpi=125)
    plt.close(fig)


def fig_jitter():
    rows = json.loads(Path("results/4_daq_realism/jitter_mc_2026-10-18/bmad/bmad_rows.json").read_text())
    summ = json.loads(Path("results/4_daq_realism/jitter_mc_2026-10-18/bmad/bmad_summary.json").read_text())
    pts = sorted({r["point"] for r in rows})
    mk = ["o", "s", "^"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for p, m_ in zip(pts, mk):
        R = [r for r in rows if r["point"] == p and r["kind"] == "jit" and r["ok"]]
        lab = f"L1 {R[0]['L1']:.1f}° / L2 {R[0]['L2']:.1f}°"
        sb = np.array([r["spacing_cloud"] for r in R]) * 1e6
        sf = np.array([r["flow_spacing_cloud"] for r in R]) * 1e6
        zb = np.array([r["sigz_cloud"] for r in R]) * 1e6
        zf = np.array([r["flow_sigz_cloud"] for r in R]) * 1e6
        s = next(x for x in summ if x["L1"] == R[0]["L1"] and x["L2"] == R[0]["L2"])
        axes[0].scatter(sf - sf.mean(), sb - sb.mean(), marker=m_, s=20, facecolor="none", edgecolor=INK2, lw=0.9,
                        label=f"{lab}: r = {s['per_shot_corr_spacing']:.3f}, slope {s['slope_bmad_on_flow_spacing']:.2f}")
        axes[1].scatter((zf - zf.mean()) / zf.mean() * 100, (zb - zb.mean()) / zb.mean() * 100, marker=m_, s=20,
                        facecolor="none", edgecolor=INK2, lw=0.9, label=f"{lab}: r = {s['per_shot_corr_sigz']:.3f}")
    for ax, t, xl, yl in ((axes[0], "per-shot spacing response", f"{L_COV} spacing − mean [µm]", "Bmad spacing − mean [µm]"),
                          (axes[1], "per-shot drive σz response", f"{L_COV} drive σz − mean [%]", "Bmad drive σz − mean [%]")):
        lim = max(np.abs(ax.get_xlim()).max(), np.abs(ax.get_ylim()).max())
        ax.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8, ls=":", zorder=0)
        ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel=xl, ylabel=yl)
        ax.set_aspect("equal")
        ax.grid(color=GRID, lw=0.5)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_title(t, fontsize=10, loc="left")
    ax = axes[2]
    xs = np.arange(len(summ))
    ax.bar(xs - 0.2, [s["bmad_spacing_std_um"] for s in summ], 0.38, color=C_TRUTH, label="Bmad (common Tao seed)")
    ax.bar(xs + 0.2, [s["flow_spacing_std_um"] for s in summ], 0.38, color=C_COV, label=f"{L_COV} (cloud)")
    for j, s in enumerate(summ):
        ax.text(j - 0.2, s["bmad_spacing_std_um"] + 0.4, f"{s['bmad_spacing_std_um']:.1f}", ha="center", fontsize=8, color=INK2)
        ax.text(j + 0.2, s["flow_spacing_std_um"] + 0.4, f"{s['flow_spacing_std_um']:.1f}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(xs, [f"L1 {s['L1']:.1f}° / L2 {s['L2']:.1f}°" for s in summ], fontsize=8.5)
    ax.set_ylabel("spacing std over 48 jittered shots [µm]")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title("jitter-induced spacing spread (Tao-seed noise 0.004 µm)", fontsize=10, loc="left")
    ax.set_ylim(0, max(s["bmad_spacing_std_um"] for s in summ) * 1.25)
    suptitle(fig, f"Jitter MC cross-check (2024u linac jitter, golden injector, identical draws): Bmad vs {L_COV} per shot",
             y=0.99)
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.12, top=0.87, wspace=0.25)
    fig.savefig(OUT / "2026-10-18_surrogate_jitter_bmad.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.environ.get("FIG_CACHE", "/tmp/notebook_figs_2026-10-18_cache"))
    ap.add_argument("--only", default=None, help="comma list: corner,lps,parity,errors,pairs,gates,jitter")
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None
    want = lambda s: only is None or s in only     # noqa: E731
    if want("gates"):
        fig_gates()
    if want("jitter"):
        fig_jitter()
    if not any(want(s) for s in ("corner", "lps", "parity", "errors", "pairs")):
        return
    models = load_models()
    D = compute(Path(a.cache), models)
    rep, comp, lps = pick_rows(D)
    sel = {"representative": rep, "compressed": comp, "lps": lps}
    print("selected:", {k: [(str(D["campaign"][i]), int(D["idx"][i])) for i in (v if isinstance(v, list) else [v])]
                        for k, v in sel.items()})
    if want("parity"):
        fig_parity(D)
    if want("errors"):
        fig_error_maps(D)
    if want("pairs"):
        print("pairs:", fig_pairs(D))
    if want("corner"):
        fig_corner(D, models, rep, "2026-10-18_surrogate_corner.png",
                   f"6-D phase space at XTCAVF, one representative held-out TEST row: Bmad vs {L_ZS}")
        fig_corner(D, models, comp, "2026-10-18_surrogate_corner_compressed.png",
                   f"6-D phase space at XTCAVF, the most compressed-witness held-out TEST row: Bmad vs {L_ZS}")
    if want("lps"):
        fig_lps(D, models, lps)


if __name__ == "__main__":
    main()
