"""Simulated DTOTR2 XTCAV readings for the 2026-10-15 notebook entry: flow (xtcav23, golden context) ->
Cheetah XTCAVF->DTOTR at +V / -V -> images binned at the DTOTR pixel (30.5 um) in the deployed auto-crop
window (+-4.6 mm x, +-2.44 mm y about each image's charge centroid), then the online analysis
(online/features.py: shared +V valley cut, +-V half-difference, crop-edge check).

  PYTHONPATH=src /home/rwu4/miniconda3/envs/slac-rl/bin/python scripts/notebook_sim_screens_2026-10-15.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy import ndimage

from twobunch_s2e_rl.artifacts import resolve_ckpt
from twobunch_s2e_rl.datagen.paths import repo_root
from twobunch_s2e_rl.online.features import band_stats, step_cut, step_features
from twobunch_s2e_rl.rl._train_utils import build_env_fn, flow_norm_json
from twobunch_s2e_rl.rl.plants import injected_charges_nC
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow
from twobunch_s2e_rl.surrogate.properties import inter_bunch
from xtcav_segment.convert import cloud_to_beam
from xtcav_segment.features import calibrate_streak
from xtcav_segment.segment import back_drift, build_segment_from_spec, element_by_name, load_spec, set_tdc

OUT = Path("/home/rwu4/photoinjector-rl/lab-notebook/images")
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 8,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
PX, HX, HY = 30.5e-6, 4.6e-3, 2.44e-3
CASES = [("clean split", -21.0, -36.5, 26e6), ("deep compression (L1 -23.5)", -23.5, -37.5, 26e6),
         ("large spacing, high streak: clipped", -19.5, -40.0, 30e6), ("near full compression", -22.0, -33.5, 26e6)]


def render(x, y, q):
    """Auto-cropped image (rows = y) about the charge centroid; returns img, x centres [m], y centres [m]."""
    cx, cy = np.average(x, weights=q), np.average(y, weights=q)
    xe = np.arange(-HX, HX + PX / 2, PX)
    ye = np.arange(-HY, HY + PX / 2, PX)
    img, _, _ = np.histogram2d(y - cy, x - cx, [ye, xe], weights=q)
    img = ndimage.gaussian_filter(img, 1.0)              # camera / optics blur (~1 px)
    return img, 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), (cx, cy)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = yaml.safe_load(open(repo_root() / "configs/rl/delta_xtcav23_screen_v6_blen.yaml"))
    de = cfg["params"]["diff_env"]
    env = build_env_fn(cfg, de["flow_ckpt"])(num_envs=1, device=dev, seed=0, episode_length=32, no_grad=True)
    flow = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(de["flow_ckpt"])), map_location=dev).eval()
    import json
    norm = json.loads(open(flow_norm_json(de["flow_ckpt"], repo_root() / de["campaign_h5"])).read())
    q_inj = injected_charges_nC(norm, flow.hparams.knob_keys or [], flow.knob_low.cpu(), flow.knob_high.cpu())
    spec = load_spec(repo_root() / de["screen"]["spec"])
    e_ref, l_tdc = float(spec["meta"]["ref_energy_eV"]), float(spec["meta"]["tdc_length_m"])
    seg, _ = build_segment_from_spec(spec, device=torch.device(dev), dtype=torch.float64)
    tdc = element_by_name(seg, "XTCAVF")
    torch.manual_seed(7)
    rows = []
    with torch.no_grad():
        for name, l1, l2, V in CASES:
            set_tdc(seg, V, 0.0)
            tdc.voltage = torch.tensor([[V], [-V]], device=dev, dtype=torch.float64)
            knobs = env._base.clone().reshape(1, -1)
            knobs[0, env._ci] = torch.tensor([l1, l2], device=knobs.device, dtype=knobs.dtype)
            kn = env._ks.normalize(knobs).to(dev).float()
            n = 20000
            d, w = flow.sample_bunch(kn, 0, n), flow.sample_bunch(kn, 1, n)
            _, td, tw = flow.feasibility(kn)
            qd, qw = q_inj(kn)
            qd, qw = float(qd * td), float(qw * tw)
            cloud = torch.cat([d, w], -2).double()
            qpp = torch.cat([torch.full((1, n), qd * 1e-9 / n), torch.full((1, n), qw * 1e-9 / n)], -1).to(dev).double()
            out = seg.track(back_drift(cloud_to_beam(cloud, e_ref, qpp, device=torch.device(dev)), l_tdc))
            xp, xm = out.x[0, 0].cpu().numpy(), out.x[1, 0].cpu().numpy()
            yp, ym = out.y[0, 0].cpu().numpy(), out.y[1, 0].cpu().numpy()
            z = cloud[0, :, 2].cpu().numpy()
            S = float(calibrate_streak(torch.tensor(0.5 * (xp - xm))[None], torch.tensor(z)[None]))
            q = qpp[0].cpu().numpy()
            truth_w = np.r_[np.zeros(n, bool), np.ones(n, bool)]
            true_sp = float(inter_bunch(d.double(), w.double())["bunch_spacing"][0]) * 1e6
            ip, xc, yc, comp = render(xp, yp, q)
            im, _, _, comm = render(xm, ym, q)
            im_old, _, _, _ = render(xm, yp, q)            # old sim: -V image on the +V energy axis
            cut, ok = step_cut([ip])
            meas = None
            if ok:
                bp, bm = band_stats(ip, xc, cut_row=cut), band_stats(im, xc, cut_row=cut)
                if bp and bm:
                    meas = step_features([bp], [bm], streak=S, edge_frac_max=0.01)
            # misassignment of the +V cut: particles on the wrong side (witness = the lower-charge side)
            if ok:
                ycut = comp[1] + yc[cut] + PX / 2
                above = yp > ycut
                wit_side = above if q[above].sum() < q[~above].sum() else ~above
                mis = float((q * (wit_side != truth_w)).sum() / q.sum())
            else:
                ycut, mis = None, float("nan")
            rows.append(dict(name=name, l1=l1, l2=l2, V=V, ip=ip, im=im, im_old=im_old, xc=xc, yc=yc, cut=cut if ok else None,
                             meas=meas, true=true_sp, mis=mis, xp=xp, yp=yp, comp=comp, truth_w=truth_w, ycut=ycut, q=q))
            print(name, f"true {true_sp:.1f} um", "meas", None if meas is None else round(1e6 * meas["bunch_spacing"], 1),
                  "valid", None if meas is None else meas["valid"], "edge", None if meas is None else round(meas["edge_frac"], 3),
                  f"misassign {mis:.3f}", flush=True)

    ext = lambda r: [1e3 * r["xc"][0], 1e3 * r["xc"][-1], 1e3 * r["yc"][0], 1e3 * r["yc"][-1]]
    fig, axes = plt.subplots(len(rows), 4, figsize=(15, 3.0 * len(rows)), gridspec_kw={"width_ratios": [1, 1, 1, 0.8]})
    for i, r in enumerate(rows):
        m = r["meas"]
        verdict = ("no valley: invalid" if m is None else
                   f"meas {1e6 * m['bunch_spacing']:.0f} um, {'valid' if m['valid'] else 'INVALID'}"
                   f"{' (edge %.0f%%)' % (100 * m['edge_frac']) if m['edge_frac'] > 0.01 else ''}")
        for j, (img, lab) in enumerate(((r["ip"], "+V"), (r["im"], "-V"))):
            ax = axes[i, j]
            ax.imshow(img, origin="lower", extent=ext(r), aspect="auto", cmap="magma")
            if r["cut"] is not None:
                ax.axhline(1e3 * r["yc"][r["cut"]], color="#9be3c6", ls="--", lw=1.3)
            ax.set(xlabel="x (streak) [mm]", ylabel="y (energy) [mm]")
            ax.text(0.02, 0.95, lab, transform=ax.transAxes, color="white", fontsize=10, weight="bold", va="top")
        axes[i, 0].set_title(f"{r['name']}  (L1 {r['l1']}, L2 {r['l2']}, {r['V'] / 1e6:.0f} MV)", fontsize=9, loc="left",
                             weight="bold")
        ax = axes[i, 2]
        sel = np.random.default_rng(0).choice(len(r["xp"]), 8000, replace=False)
        cx, cy = r["comp"]
        for wmask, c, lab in ((~r["truth_w"], BLUE, "drive (truth)"), (r["truth_w"], ORANGE, "witness (truth)")):
            s = sel[wmask[sel]]
            ax.scatter(1e3 * (r["xp"][s] - cx), 1e3 * (r["yp"][s] - cy), s=0.4, c=c, alpha=0.5, label=lab, rasterized=True)
        ax.add_patch(plt.Rectangle((-HX * 1e3, -HY * 1e3), 2 * HX * 1e3, 2 * HY * 1e3, fill=False, ec=INK2, lw=1, ls=":"))
        if r["ycut"] is not None:
            ax.axhline(1e3 * (r["ycut"] - cy), color=INK, ls="--", lw=1)
        ax.set(xlim=(-6.5, 6.5), ylim=(-2.7, 2.7), xlabel="x (streak) [mm]", ylabel="y [mm]")
        ax.set_title(f"+V particles by truth; {100 * r['mis']:.0f}% misassigned by the cut" if np.isfinite(r["mis"])
                     else "+V particles by truth", fontsize=8.5, loc="left")
        if i == 0:
            ax.legend(markerscale=12, fontsize=7, frameon=False, loc="upper right")
        ax = axes[i, 3]
        pr = lambda img: img.sum(1) / img.sum()
        ax.plot(1e3 * r["yc"], pr(r["ip"]), color=BLUE, lw=1.5, label="+V")
        ax.plot(1e3 * r["yc"], pr(r["im"]), color=ORANGE, lw=1.5, label="-V (own energy axis)")
        ax.plot(1e3 * r["yc"], pr(r["im_old"]), color=INK2, lw=1, ls=":", label="-V on +V axis (old sim)")
        if r["cut"] is not None:
            ax.axvline(1e3 * r["yc"][r["cut"]], color=INK, ls="--", lw=1)
        ax.set(xlabel="y (energy) [mm]", yticks=[])
        ax.set_title(f"true {r['true']:.0f} um\n{verdict}", fontsize=8.5, loc="left")
        ax.set_xlim(-2.44, 2.44)
        if i == 0:
            ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Simulated DTOTR2 readings (flow -> Cheetah, deployed auto-crop, online analysis); dashed = shared +V energy cut",
                 fontsize=10.5, weight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "2026-10-15_sim_screen_cases.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
