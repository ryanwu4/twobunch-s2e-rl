"""Figures for the overnight 2026-10-15 bunch-split changes, real E300 DAQ part (bmad-qpad-dev env).

Loads the deployed-preprocessed DTOTR2 images of selected steps (same chain as scripts/xtcav_online_replay.py)
and plots, per step: the +V and -V median images with the per-image valley cuts (old, v1/v2) and the shared
+V step cut (new, v3, decision D2), and the +V / -V y projections. Also reads the saved replay JSONs for the
scan-curve, per-image-offset, dropped-step and crop-edge panels.

  PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MPLBACKEND=Agg /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/fig_bunch_split_changes_real.py --out results/bunch_split_changes_2026-10-15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xtcav_real_features as rf  # noqa: E402

from twobunch_s2e_rl.online.features import band_stats, step_cut, valley_threshold  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results"
V = {"v6": "online_replay_v6_median_cut_2026-10-15", "v1": "online_replay_2026-10-15", "v2": "online_replay_v2_2026-10-15", "v3": "online_replay_v3_2026-10-15",
     "v4": "online_replay_v4_2026-10-15"}
DATASETS = ["E300_15673", "TEST_15668", "E300_15671"]


def load(v, d):
    return json.loads((RES / V[v] / f"{d}.json").read_text())


def step_images(dataset, steps, per_sign=60):
    """{step: {"plus": [img], "minus": [img], "x_m": x}} with the deployed preprocessing."""
    exp, num = dataset.rsplit("_", 1)
    ds = rf.DATASET(str(rf.XTCAV_ROOT), exp, num)
    cls = rf.xtcav.classify_tcav_shots(ds)
    stp = rf.xtcav.get_shot_steps(ds)
    t0 = json.loads((rf.XTCAV_ROOT / "sim2real" / "results" / "tier0_real_features" / dataset / "summary.json").read_text())
    a = rf.xtcav.XTCAVAnalysis(ds, cam="DTOTR2", d_nom=rf.D_NOM, chargeCal=1.0, hand_crop=rf.HAND_CROP,
                               auto_crop=rf.AUTO_CROP, hot_pix_threshold=rf.HOT_PIX_THRESHOLD,
                               noise_threshold=rf.NOISE_THRESHOLD, median_filter=rf.MEDIAN_FILTER)
    a.set_tcav_calibration(t0["calibration"]["cal_um_per_deg_used"])
    refs = rf.xtcav.build_step_distortion_references(a, cls, steps_per_shot=stp)
    out = {}
    for s in steps:
        d = {"plus": [], "minus": [], "x_m": None, "S": float(a.cal_scaled)}
        for tag, grp in (("plus", cls["ix_plus"]), ("minus", cls["ix_neg"])):
            for i in [int(i) for i in grp if stp[int(i)] == s][:per_sign]:
                try:
                    _, img, _ = rf.corrected_shot(a, i, rf.xtcav.select_reference_for_shot(refs, i, stp))
                except Exception:
                    continue
                if d["x_m"] is None:
                    d["x_m"], shape = 1e-3 * np.asarray(img.x, float), img.data.shape
                if img.data.shape == shape:
                    d[tag].append(np.clip(img.data.astype(float), 0, None))
        out[s] = d
    return out


def fig_scan_curves(out):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, d in zip(axes, DATASETS):
        for v, kw in (("v1", dict(c="0.6", ls=":", marker=".", label="v1: per-image cut, median of centroids")),
                      ("v2", dict(c="tab:red", ls="--", marker="x", label="v2: per-image cut, median of offsets")),
                      ("v3", dict(c="tab:blue", ls="-", marker="o", ms=4, label="v3: shared cut = valley of summed +V")),
                      ("v6", dict(c="tab:green", ls="-", lw=2.2, marker="D", ms=3.5,
                                  label="v6 (adopted): shared cut = median of per-image +V cuts"))):
            st = load(v, d)["steps"]
            x = [st[k]["scan_val"] for k in st]
            y = [st[k]["spacing_um"] for k in st]
            ax.plot(x, y, **kw)
        v2, v3 = load("v2", d)["steps"], load("v3", d)["steps"]
        drop = [k for k in v2 if k not in v3]
        if drop:
            ax.scatter([v2[k]["scan_val"] for k in drop], [v2[k]["spacing_um"] for k in drop], s=90, facecolors="none",
                       edgecolors="k", zorder=5, label="dropped in v3 (no valley in summed +V)")
        v4 = load("v4", d)["steps"]
        edge = [k for k in v4 if v4[k]["edge_frac"] > 0.01]
        if edge:
            ax.scatter([v4[k]["scan_val"] for k in edge], [v4[k]["spacing_um"] for k in edge], marker="s", s=70,
                       facecolors="none", edgecolors="tab:orange", lw=1.5, zorder=6, label="v4: crop-edge flag (>1%)")
        ax.axhspan(-60, 60, color="0.9", zorder=0)
        ax.set(title=d, xlabel="L2 scan value (scanVals)", ylabel="step spacing [um]")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].text(0.02, 0.03, "grey band: |spacing| < 60 um (deployed-invalid)", transform=axes[1].transAxes, fontsize=7)
    fig.suptitle("Real L2 scans: step spacing before / after the bunch-split changes", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "fig1_real_scan_curves.png", dpi=130)
    plt.close(fig)


def fig_offsets(out, d="E300_15673", steps=("4", "9", "13", "16")):
    v2, v3 = load("v2", d)["steps"], load("v3", d)["steps"]
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 6.2), sharey="row")
    for j, s in enumerate(steps):
        for i, (v, name) in enumerate(((v2, "v2 per-image cut"), (v3, "v3 shared +V cut"))):
            ax = axes[i, j]
            op, om = np.array(v[s]["offsets_plus_um"]), np.array(v[s]["offsets_minus_um"])
            bins = np.linspace(-400, 400, 81)
            ax.hist(op, bins, color="tab:red", alpha=0.6, label="+V images")
            ax.hist(om, bins, color="tab:blue", alpha=0.6, label="-V images")
            sp = 0.5 * (np.median(op) - np.median(om))
            ax.axvline(np.median(op), c="tab:red", lw=1.5)
            ax.axvline(np.median(om), c="tab:blue", lw=1.5)
            ax.set_title(f"step {s} (L2 {v[s]['scan_val']}), {name}\nspacing = (med+ - med-)/2 = {sp:.0f} um", fontsize=8)
            ax.set_xlabel("per-image band offset (x_d - x_w)/S [um of z]")
    axes[0, 0].set_ylabel("images")
    axes[1, 0].set_ylabel("images")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"{d}: per-image drive-witness offsets. +V and -V should be mirror images (opposite sign); with "
                 "per-image cuts the -V images split at another dip and give the same sign", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "fig3_offsets_per_image.png", dpi=130)
    plt.close(fig)


def fig_cut_examples(out, imgs, d, steps, fname, title):
    fig, axes = plt.subplots(len(steps), 3, figsize=(15, 3.3 * len(steps)), squeeze=False)
    for r, s in enumerate(steps):
        D = imgs[s]
        P, M = D["plus"], D["minus"]
        k_step, ok = step_cut(P, method="sum")
        k_med, ok_med = step_cut(P, method="median")
        cuts_p = [valley_threshold(i.sum(axis=1))[0] for i in P]
        cuts_m = [valley_threshold(i.sum(axis=1))[0] for i in M]
        for c, (L, cuts, nm) in enumerate(((P, cuts_p, "+V"), (M, cuts_m, "-V"))):
            ax = axes[r, c]
            if not L:
                continue
            med = np.median(L, axis=0)
            ax.imshow(med, aspect="auto", origin="lower", cmap="magma")
            for k in cuts:
                if k is not None:
                    ax.axhline(k, color="cyan", lw=0.4, alpha=0.35)
            if ok:
                ax.axhline(k_step, color="lime", lw=2, ls="--")
            if ok_med:
                ax.axhline(k_med, color="white", lw=1.8, ls=":")
            n_fail = sum(k is None for k in cuts)
            ax.set_title(f"step {s} {nm} (median of {len(L)}){'; %d images: no valley' % n_fail if n_fail else ''}",
                         fontsize=9)
            ax.set_xlabel("x pixel (streak axis = time)")
            ax.set_ylabel("y pixel (energy)")
        ax = axes[r, 2]
        sm = lambda L: ndimage.gaussian_filter1d(np.median([i.sum(axis=1) for i in L], axis=0), 2)
        if P:
            ax.plot(sm(P), c="tab:red", label="+V y projection")
        if M:
            ax.plot(sm(M), c="tab:blue", label="-V y projection")
        if ok:
            ax.axvline(k_step, color="lime", lw=2, ls="--", label="shared cut: summed +V valley (v3)")
        if ok_med:
            ax.axvline(k_med, color="k", lw=1.8, ls=":", label="shared cut: median per-image +V (v6, adopted)")
        km = [k for k in cuts_m if k is not None]
        if km:
            ax.axvline(np.median(km), color="tab:blue", ls=":", label="median per-image -V cut (old)")
        ax.set(xlabel="y pixel (energy)", ylabel="intensity")
        ax.legend(fontsize=7)
    fig.suptitle(title + "\nimage lines: cyan = per-image valley cuts (v1/v2), green dashed = summed-+V step cut (v3), "
                 "white dotted = median per-image +V step cut (v6, adopted)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out / fname, dpi=120)
    plt.close(fig)


def fig_edge(out, imgs, steps):
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(5 * (len(steps) + 1), 4))
    for ax, s in zip(axes, steps):
        P = imgs[s]["plus"]
        med = np.median(P, axis=0)
        xm = imgs[s]["x_m"]
        k = max(1, int(round(0.3e-3 / abs(xm[1] - xm[0]))))
        b = band_stats(med, xm, cut_row=step_cut(P)[0])
        ax.imshow(med, aspect="auto", origin="lower", cmap="magma")
        ax.axvspan(-0.5, k - 0.5, color="tab:orange", alpha=0.35)
        ax.axvspan(med.shape[1] - k - 0.5, med.shape[1] - 0.5, color="tab:orange", alpha=0.35)
        ax.set_title(f"E300_15673 step {s} +V (median)\nintensity within 0.3 mm of crop edge: {100 * b['edge_frac']:.1f}%",
                     fontsize=8)
        ax.set_xlabel("x pixel (auto-crop window)")
    ax = axes[-1]
    for d, mk in zip(DATASETS, "os^"):
        st = load("v4", d)["steps"]
        ax.scatter([abs(st[k]["spacing_um"]) for k in st], [100 * st[k]["edge_frac"] for k in st], marker=mk, label=d, s=18)
    ax.axhline(1.0, c="k", ls="--", lw=1, label="threshold 1%")
    ax.set(xlabel="|step spacing| [um]", ylabel="edge intensity [%]", title="crop-edge fraction, all real steps")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig5_real_crop_edge.png", dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_scan_curves(out)
    fig_offsets(out)
    im73 = step_images("E300_15673", [1, 4, 9, 13])
    fig_cut_examples(out, im73, "E300_15673", [4, 9, 13], "fig2_cut_examples_E300_15673.png",
                     "E300_15673: clean step 4 vs affected steps 9, 13. The -V energy profile differs from +V (the "
                     "TCAV energy kick flips with V), so per-image -V valleys land on another dip")
    im71 = step_images("E300_15671", [1, 7, 12])
    fig_cut_examples(out, im71, "E300_15671", [1, 7, 12], "fig4_dropped_steps_E300_15671.png",
                     "E300_15671: step 1 kept; steps 7 and 12 (near full compression) have no valley in the summed +V "
                     "projection -> dropped in v3 (previously individual images still produced readings)")
    fig_edge(out, im73, [1, 4])
    imT = step_images("TEST_15668", [24, 25, 29, 30])
    fig_cut_examples(out, imT, "TEST_15668", [24, 25, 29, 30], "fig6_regression_TEST_15668.png",
                     "TEST_15668 near the spacing sign change: the summed-+V cut (v3) jumps to another dip at steps 25 / 29 "
                     "(spacing +73 / +61 um between ~0 and -50 um neighbours); the median of per-image +V cuts (v6) does not")
    with_offsets_note(out)


def with_offsets_note(out):
    v2, v3 = load("v2", "TEST_15668")["steps"], load("v3", "TEST_15668")["steps"]
    rows = [(k, v2[k]["spacing_um"], v3.get(k, {}).get("spacing_um"), np.median(v2[k]["cut_rows_plus"]),
             np.median(v3[k]["cut_rows_plus"]) if k in v3 else None) for k in ("22", "23", "24", "25", "26", "27", "28", "29", "30")]
    (out / "TEST_15668_late_steps.txt").write_text(
        "step  v2_spacing  v3_spacing  v2_median_plus_cut_row  v3_step_cut_row\n"
        + "\n".join(f"{k:>4} {a:10.1f} {b if b is None else round(b, 1)!s:>10} {c:8.1f} {d!s:>8}" for k, a, b, c, d in rows) + "\n")


if __name__ == "__main__":
    main()
