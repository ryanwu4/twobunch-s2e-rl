"""Why do the +V and -V DTOTR2 images have different energy (y) projections? Diagnostics on real E300 DAQ data
(bmad-qpad-dev env). Images: deployed preprocessing WITHOUT the auto-crop (hand crop + noise filters + per-step
off-shot row correction), so pixel rows are absolute and comparable between images.

Per step:
  1. systematic vs noise: median +V, -V and TCAV-off y projections; the +V/-V difference in units of its
     image-to-image standard error
  2. x->y coupling from shot-to-shot jitter: within each sign, regress image y centroid on x centroid. Arrival
     jitter moves the whole streak in x; if y_screen = eta*delta + theta*x (rotation / x-y coupling) both signs
     show the same slope theta; a pure energy effect gives none
  3. time-mirrored centrelines: y_c vs z for +V (z = +(x - <x>)/S) and -V (z = -(x - <x>)/S); coupling predicts
     y+(z) - y-(z) = 2 theta S z + const

  PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 MPLBACKEND=Agg /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python \
      scripts/xtcav_pm_asymmetry.py --out results/bunch_split_changes_2026-10-15/pm_asymmetry
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

from twobunch_s2e_rl.online.features import valley_threshold  # noqa: E402

CASES = {"E300_15673": [4, 9, 13, 16], "TEST_15668": [10, 25]}


def load_dataset(dataset, steps, per_sign=60):
    exp, num = dataset.rsplit("_", 1)
    ds = rf.DATASET(str(rf.XTCAV_ROOT), exp, num)
    cls = rf.xtcav.classify_tcav_shots(ds)
    stp = rf.xtcav.get_shot_steps(ds)
    t0 = json.loads((rf.XTCAV_ROOT / "sim2real" / "results" / "tier0_real_features" / dataset / "summary.json").read_text())
    a = rf.xtcav.XTCAVAnalysis(ds, cam="DTOTR2", d_nom=rf.D_NOM, chargeCal=1.0, hand_crop=rf.HAND_CROP,
                               auto_crop=None, hot_pix_threshold=rf.HOT_PIX_THRESHOLD,
                               noise_threshold=rf.NOISE_THRESHOLD, median_filter=rf.MEDIAN_FILTER)
    a.set_tcav_calibration(t0["calibration"]["cal_um_per_deg_used"])
    refs = rf.xtcav.build_step_distortion_references(a, cls, steps_per_shot=stp)
    px_m = None
    out = {}
    for s in steps:
        d = {}
        for tag, grp in (("plus", cls["ix_plus"]), ("minus", cls["ix_neg"]), ("off", cls["ix_off"])):
            L = []
            for i in [int(i) for i in grp if stp[int(i)] == s][:per_sign]:
                try:
                    _, img, _ = rf.corrected_shot(a, i, rf.xtcav.select_reference_for_shot(refs, i, stp))
                except Exception:
                    continue
                if px_m is None:
                    px_m = abs(float(img.x[1] - img.x[0])) * 1e-3
                L.append(np.clip(img.data.astype(float), 0, None))
            d[tag] = L
        out[s] = d
    return out, float(a.cal_scaled), px_m


def com(img):
    q = img.sum()
    yy, xx = np.indices(img.shape)
    return float((img * xx).sum() / q), float((img * yy).sum() / q)


def centreline(img, sign, S_px):
    """(z [px of streak / S], y centroid per column) for columns with enough intensity."""
    col = img.sum(axis=0)
    ok = col > 0.02 * col.max()
    yy = np.arange(img.shape[0])
    yc = (img * yy[:, None]).sum(axis=0) / np.maximum(col, 1e-30)
    x0, _ = com(img)
    z = sign * (np.arange(img.shape[1]) - x0) / S_px
    return z[ok], yc[ok]


def analyse(dataset, data, S, px_m):
    S_px = S                                  # screen px per px of z (both in pixels -> S itself)
    res = {}
    fig, axes = plt.subplots(len(data), 4, figsize=(19, 3.6 * len(data)), squeeze=False)
    for r, (s, d) in enumerate(sorted(data.items())):
        P, M, O = d["plus"], d["minus"], d["off"]
        proj = {k: np.array([i.sum(axis=1) for i in L]) for k, L in (("plus", P), ("minus", M), ("off", O)) if L}
        norm = {k: v / v.sum(axis=1, keepdims=True) for k, v in proj.items()}
        med = {k: np.median(v, axis=0) for k, v in norm.items()}
        se = {k: 1.2533 * np.std(v, axis=0) / np.sqrt(len(v)) for k, v in norm.items()}
        diff = med["plus"] - med["minus"]
        z = diff / np.sqrt(se["plus"] ** 2 + se["minus"] ** 2 + 1e-30)
        # 2. jitter coupling
        cp, cm = np.array([com(i) for i in P]), np.array([com(i) for i in M])
        fitp, fitm = np.polyfit(cp[:, 0], cp[:, 1], 1), np.polyfit(cm[:, 0], cm[:, 1], 1)
        co = np.array([com(i) for i in O]) if O else None
        # 3. centrelines (median images)
        zp, yp = centreline(np.median(P, axis=0), +1, S_px)
        zm, ym = centreline(np.median(M, axis=0), -1, S_px)
        grid = np.linspace(max(zp.min(), zm.min()), min(zp.max(), zm.max()), 60)
        dy = np.interp(grid, *sorted_xy(zp, yp)) - np.interp(grid, *sorted_xy(zm, ym))
        lin = np.polyfit(grid, dy, 1)
        cuts = {k: [valley_threshold(i.sum(axis=1))[0] for i in L] for k, L in (("plus", P), ("minus", M))}
        aff = {k: affine_fit(med["off"], med[k]) for k in ("plus", "minus")} if "off" in med else {}
        res[s] = {
            "n": {k: len(v) for k, v in d.items()},
            "max_abs_proj_diff_in_se": float(np.max(np.abs(z))),
            "frac_rows_beyond_5se": float(np.mean(np.abs(z) > 5)),
            "ycom_px": {"plus": float(cp[:, 1].mean()), "minus": float(cm[:, 1].mean()),
                        "off": float(co[:, 1].mean()) if co is not None else None},
            "xcom_px": {"plus": float(cp[:, 0].mean()), "minus": float(cm[:, 0].mean()),
                        "off": float(co[:, 0].mean()) if co is not None else None},
            "jitter_slope_dy_dx": {"plus": float(fitp[0]), "minus": float(fitm[0])},
            "jitter_x_rms_px": {"plus": float(cp[:, 0].std()), "minus": float(cm[:, 0].std())},
            "centreline_dy_vs_z": {"slope_px_per_px_z": float(lin[0]), "offset_px": float(lin[1]),
                                    "rms_residual_px": float(np.std(dy - np.polyval(lin, grid))),
                                    "z_range_px": [float(grid[0]), float(grid[-1])]},
            "cut_rows": {k: [int(c) for c in v if c is not None] for k, v in cuts.items()},
            "affine_from_off": aff,
        }
        ax = axes[r, 0]
        for k, c in (("plus", "tab:red"), ("minus", "tab:blue"), ("off", "k")):
            if k in med:
                ax.plot(ndimage.gaussian_filter1d(med[k], 1), c=c, label=f"{k} (n={len(norm[k])})")
                ax.fill_between(np.arange(len(med[k])), med[k] - 2 * se[k], med[k] + 2 * se[k], color=c, alpha=0.2)
        ax.set(title=f"{dataset} step {s}: median y projections (+-2 SE)", xlabel="y row (absolute)", ylabel="norm. intensity")
        ax.legend(fontsize=7)
        ax = axes[r, 1]
        ax.plot(z, c="tab:purple")
        ax.axhline(5, c="k", ls=":"); ax.axhline(-5, c="k", ls=":")
        ax.set(title="(+V - -V) / standard error", xlabel="y row", ylabel="significance")
        ax = axes[r, 2]
        ax.scatter(cp[:, 0], cp[:, 1], s=8, c="tab:red", label=f"+V dy/dx = {fitp[0]:.3f}")
        ax.scatter(cm[:, 0], cm[:, 1], s=8, c="tab:blue", label=f"-V dy/dx = {fitm[0]:.3f}")
        if co is not None:
            ax.scatter(co[:, 0], co[:, 1], s=8, c="k", label="TCAV off")
        ax.set(title="per-image centroid (shot-to-shot jitter)", xlabel="x COM [px]", ylabel="y COM [px]")
        ax.legend(fontsize=7)
        ax = axes[r, 3]
        ax.plot(zp, yp, c="tab:red", label="+V centreline")
        ax.plot(zm, ym, c="tab:blue", label="-V centreline (x mirrored)")
        ax.plot(grid, dy + np.mean(yp), c="tab:purple", ls="--", label=f"difference (+mean); slope {lin[0]:.3f} px/px")
        ax.set(title="centreline y vs time (z in streak px / S)", xlabel="z [px of z]", ylabel="y [px]")
        ax.legend(fontsize=7)
    fig.tight_layout()
    return fig, res


def affine_fit(ref, tgt, smooth=1.5):
    """Fit tgt(y) ~ ref((y - y0) / a + y0) / a (a stretch a about the ref centroid y0, plus a shift b): returns
    a, b [px] and the residual L1 fraction before (a=1, b=0) and after the fit."""
    from scipy.optimize import minimize
    r, t = ndimage.gaussian_filter1d(ref, smooth), ndimage.gaussian_filter1d(tgt, smooth)
    r, t = r / r.sum(), t / t.sum()
    y = np.arange(len(r), dtype=float)
    y0 = float((r * y).sum())
    model = lambda p: np.interp((y - y0 - p[1]) / p[0] + y0, y, r, left=0, right=0) / p[0]
    loss = lambda p: np.abs(model(p) - t).sum()
    best = min((minimize(loss, [a0, b0], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
                for a0 in (0.95, 1.0, 1.05) for b0 in (-3.0, 0.0, 3.0)), key=lambda m: m.fun)
    return {"stretch_a": float(best.x[0]), "shift_b_px": float(best.x[1]),
            "resid_L1_before": float(loss([1.0, 0.0])), "resid_L1_after": float(best.fun)}


def sorted_xy(x, y):
    o = np.argsort(x)
    return x[o], y[o]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for dataset, steps in CASES.items():
        data, S, px_m = load_dataset(dataset, steps)
        fig, res = analyse(dataset, data, S, px_m)
        fig.savefig(out / f"pm_asymmetry_{dataset}.png", dpi=110)
        plt.close(fig)
        allres[dataset] = {"S_screen_per_z": S, "pixel_m": px_m, "steps": res}
    (out / "pm_asymmetry.json").write_text(json.dumps(allres, indent=1, default=float))
    for d, v in allres.items():
        for s, r in v["steps"].items():
            print(d, s, json.dumps({k: r[k] for k in ("n", "max_abs_proj_diff_in_se", "frac_rows_beyond_5se",
                                                       "jitter_slope_dy_dx", "affine_from_off")}))


if __name__ == "__main__":
    main()
