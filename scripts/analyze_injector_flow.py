"""Is the injector-aware flow data-limited? Failure modes of the fine-tuned flow on injector effects.

Data (processed/xtcav23e_inj_2026-10-15.h5, injector_full rows): 250 LHS injector points (25 pre-registered test
points) + golden; per point 1 'crossed' row (golden linac) + 10 'nested' linac draws; 275 'paired_golden' rows
re-run nested linac settings at injector = golden. Two injector-effect (Delta) sets, truth = Bmad:
  cross: point's crossed row - golden's crossed row (injector response at the golden linac point; 250)
  pairs: nested row - its paired_golden row (same linac knobs; 275)
Split by injector point, as the flow saw it (group split): train / val / test (= holdout).

Tests:
  1. flow (CRN predictions) R^2 / slope per split, base (widened old flow) vs fine-tunes: does the flow fit even
     its TRAINING injector points? (underfit -> loss/capacity, not data)
  2. simple regressors on the same rows (ridge on 37 knobs, poly-2 ridge, gradient boosting, and on the cross
     set a GP on the 12 injector knobs), learning curves vs number of training injector points: does a model
     that only sees these data predict held-out points, and is it still improving at full size? (data-limited)
  3. failure modes: attenuation (pred vs true), per-knob sensitivities, residual vs region / spacing, T calibration

  PYTHONPATH=src python scripts/analyze_injector_flow.py --out results/injector_flow_analysis_2026-10-15
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from surrogate_gates import predict, truth  # noqa: E402

from twobunch_s2e_rl.artifacts import resolve_ckpt  # noqa: E402
from twobunch_s2e_rl.surrogate.dataset import TwoBunchFlowDataModule, orig_split_keys  # noqa: E402
from twobunch_s2e_rl.surrogate.model import TwoBunchFlow  # noqa: E402

PROC = "processed/xtcav23e_inj_2026-10-15.h5"
ORIG = ("processed/xtcav23_combined.h5", ["xtcav23_full", "xtcav23_pilot"])
MODELS = {"base (old flow, widened)": "trained/xtcav23e_inj12_widened_2026-10-15/widened.ckpt",
          "ft_s0 best (ep 55)": "trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/best-*.ckpt",
          "ft_s0 last (ep 149)": "trained/xtcav23e_inj_ft_s0_2026-10-15/checkpoints/last.ckpt",
          "ft_s1 best": "trained/xtcav23e_inj_ft_s1_2026-10-15/checkpoints/best-*.ckpt"}
INJ = ["laser_timing_ps", "GUNF:rf_field_scale", "GUNF:theta0_deg", "distgen:t_dist:dists:d2:avg_t:value",
       "drive_fraction", "distgen:total_charge:value", "L0AF_phase_offset_deg", "SOL10111:solenoid_field_scale",
       "distgen:r_dist:sigma_xy:value", "distgen:r_dist:truncation_radius:value", "distgen:start:MTE:value",
       "sigma_t_ps"]
SHORT = ["laser t", "gun ampl", "gun phase", "delay d2", "drive frac", "charge", "L0AF ph", "solenoid",
         "sigma_xy", "trunc r", "MTE", "sigma_t"]


def r2(p, t):
    return float(1 - np.sum((p - t) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-30))


def slope(p, t):
    return float(np.polyfit(t, p, 1)[0]) if len(t) > 2 else float("nan")


def metrics(p, t):
    return {"n": int(len(t)), "R2": r2(p, t), "slope": slope(p, t), "rmse": float(np.sqrt(np.mean((p - t) ** 2))),
            "signal_rms": float(np.sqrt(np.mean((t - t.mean()) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2048)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    first = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(MODELS["ft_s0 best (ep 55)"])), map_location=dev).eval()
    scal = {k: getattr(first, k).cpu().double() for k in ("drive_mean", "drive_std", "witness_mean", "witness_std")}
    orig = orig_split_keys(ORIG[0], ORIG[1])
    dm = TwoBunchFlowDataModule(processed_h5=PROC, split="group", orig_split=orig, scaler={k: v.tolist() for k, v in scal.items()})
    dm.setup()
    ds = dm.full
    role = np.full(len(ds), "train", dtype=object)
    role[np.asarray(dm.val.indices)] = "val"
    role[np.asarray(dm.test.indices)] = "test"
    with h5py.File(PROC) as h:
        camp, idx = h["campaign"][...], h["idx"][...]
        key, ref = h["pair_key"][...], h["pair_ref"][...]
    ok = ds.drive_density.numpy() & ds.witness_density.numpy()
    inj_rows = np.where((camp == 2) & ok)[0]

    # row metadata from the campaign sample jsons
    meta = {}
    for f in glob.glob("data/injector_full/sample_*.json"):
        s = json.loads(Path(f).read_text())
        meta[int(s["idx"])] = s
    blk = {r: meta[int(idx[r])]["block"] for r in inj_rows}
    pt = {r: meta[int(idx[r])]["injector_point"] for r in inj_rows}
    region = {r: meta[int(idx[r])]["linac_region"] for r in inj_rows}
    injx = np.array([[meta[int(idx[r])]["knobs"][k] for k in INJ] for r in inj_rows])
    lo, hi = injx.min(0), injx.max(0)
    inj_norm = {r: (injx[i] - lo) / np.maximum(hi - lo, 1e-30) for i, r in enumerate(inj_rows)}

    # truth for all injector-campaign rows (+ old val rows for the absolute comparison)
    T = truth(ds, list(inj_rows), scal)
    tru = {k: dict(zip(inj_rows, v)) for k, v in T.items()}

    # delta sets
    g_cross = [r for r in inj_rows if pt[r] == "golden" and blk[r] == "crossed"][0]
    cross = [(r, g_cross) for r in inj_rows if blk[r] == "crossed" and pt[r] != "golden"]
    by = {}
    for r in inj_rows:
        if key[r] >= 0:
            by.setdefault(int(key[r]), {})["ref" if ref[r] else "pt"] = r
    pairs = [(d["pt"], d["ref"]) for d in by.values() if len(d) == 2]
    sets = {"cross": cross, "pairs": pairs}
    rl = lambda r: role[r]

    # 1. flow predictions
    preds, res = {}, {"flow": {}}
    for name, ck in MODELS.items():
        m = TwoBunchFlow.load_from_checkpoint(str(resolve_ckpt(ck)), map_location=dev).eval()
        P = predict(m, ds.knobs[list(inj_rows)], n=a.n, dev=dev)
        preds[name] = {k: dict(zip(inj_rows, v)) for k, v in P.items()}
        res["flow"][name] = {}
        for sname, S in sets.items():
            for split in ("train", "val", "test"):
                sel = [(p_, q_) for p_, q_ in S if rl(p_) == split]
                if not sel:
                    continue
                t = 1e6 * np.array([tru["bunch_spacing"][p_] - tru["bunch_spacing"][q_] for p_, q_ in sel])
                p = 1e6 * np.array([preds[name]["bunch_spacing"][p_] - preds[name]["bunch_spacing"][q_] for p_, q_ in sel])
                res["flow"][name][f"{sname}/{split}"] = metrics(p, t)
        del m
        torch.cuda.empty_cache()

    # absolute spacing error by role
    for name in MODELS:
        for split in ("train", "val", "test"):
            rr = [r for r in inj_rows if rl(r) == split]
            e = 1e6 * np.abs(np.array([preds[name]["bunch_spacing"][r] - tru["bunch_spacing"][r] for r in rr]))
            res["flow"][name][f"abs_spacing_mae/{split}"] = float(e.mean())

    # 2. simple regressors on the same data, learning curves vs number of training injector points
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    X_all = ds.knobs.numpy().astype(float)
    y_all = np.array([tru["bunch_spacing"][r] for r in inj_rows]) * 1e6
    Xr = X_all[inj_rows]
    train_pts = sorted({pt[r] for r in inj_rows if rl(r) == "train" and pt[r] != "golden"})
    eval_pairs = [(p_, q_) for p_, q_ in pairs if rl(p_) in ("val", "test")]
    eval_cross = [(p_, q_) for p_, q_ in cross if rl(p_) in ("val", "test")]
    rng = np.random.default_rng(0)

    def fit_abs(model, keep_pts):
        rows = [i for i, r in enumerate(inj_rows) if rl(r) == "train" and (pt[r] in keep_pts or pt[r] == "golden")]
        model.fit(Xr[rows], y_all[rows])
        pos = {r: i for i, r in enumerate(inj_rows)}
        f = lambda S: (np.array([model.predict(Xr[[pos[p_]]])[0] - model.predict(Xr[[pos[q_]]])[0] for p_, q_ in S]),
                       np.array([y_all[pos[p_]] - y_all[pos[q_]] for p_, q_ in S]))
        return f(eval_pairs), f(eval_cross)

    regs = {"ridge (linear, 37 knobs)": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 15))),
            "poly-2 ridge (37 knobs)": lambda: make_pipeline(PolynomialFeatures(2), StandardScaler(),
                                                              RidgeCV(alphas=np.logspace(-3, 4, 15))),
            "gradient boosting": lambda: HistGradientBoostingRegressor(max_iter=600, learning_rate=0.05,
                                                                      min_samples_leaf=10, random_state=0)}
    fracs = [0.15, 0.3, 0.5, 0.75, 1.0]
    lc = {}
    for rname, mk in regs.items():
        lc[rname] = {"pairs": [], "cross": []}
        for f in fracs:
            rp, rc = [], []
            for rep in range(5 if f < 1 else 1):
                keep = set(rng.choice(train_pts, int(round(f * len(train_pts))), replace=False)) if f < 1 else set(train_pts)
                (pp, tp), (pc, tc) = fit_abs(mk(), keep)
                rp.append(r2(pp, tp)); rc.append(r2(pc, tc))
            lc[rname]["pairs"].append((int(round(f * len(train_pts))), float(np.mean(rp)), float(np.std(rp))))
            lc[rname]["cross"].append((int(round(f * len(train_pts))), float(np.mean(rc)), float(np.std(rc))))
        print(rname, lc[rname], flush=True)

    # GP on the cross set (12 injector knobs only; injector response at the golden linac point)
    cr_train = [(p_, q_) for p_, q_ in cross if rl(p_) == "train"]
    Xc = lambda S: np.array([inj_norm[p_] for p_, _ in S])
    yc = lambda S: 1e6 * np.array([tru["bunch_spacing"][p_] - tru["bunch_spacing"][q_] for p_, q_ in S])
    gp_lc = []
    for f in fracs:
        vals = []
        for rep in range(5 if f < 1 else 1):
            sel = [cr_train[i] for i in (rng.choice(len(cr_train), int(round(f * len(cr_train))), replace=False)
                                         if f < 1 else range(len(cr_train)))]
            k = ConstantKernel(100.0) * RBF(length_scale=np.ones(12), length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(1.0)
            gp = GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=rep).fit(Xc(sel), yc(sel))
            vals.append(r2(gp.predict(Xc(eval_cross)), yc(eval_cross)))
        gp_lc.append((len(sel), float(np.mean(vals)), float(np.std(vals))))
    lc["GP (12 injector knobs, cross set)"] = {"cross": gp_lc}
    gp_full = gp
    print("GP", gp_lc, gp_full.kernel_, flush=True)
    res["learning_curves"] = lc
    res["gp_kernel"] = str(gp_full.kernel_)

    # per-knob linear sensitivities on the cross set (all points): true vs flow
    from sklearn.linear_model import LinearRegression
    Xall_c = Xc(cross)
    sens = {}
    for name in ("base (old flow, widened)", "ft_s0 best (ep 55)"):
        yp = 1e6 * np.array([preds[name]["bunch_spacing"][p_] - preds[name]["bunch_spacing"][q_] for p_, q_ in cross])
        sens[name] = LinearRegression().fit(Xall_c, yp).coef_.tolist()
    lt = LinearRegression().fit(Xall_c, yc(cross))
    sens["truth (Bmad)"] = lt.coef_.tolist()
    res["linear_sensitivity_um_per_full_range"] = dict(zip(["knobs"], [SHORT])) | sens
    res["truth_linear_R2_cross"] = float(lt.score(Xall_c, yc(cross)))
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))

    # ---------------- figures ----------------
    cols = {"train": "tab:blue", "val": "tab:orange", "test": "tab:red"}
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    for j, name in enumerate(["base (old flow, widened)", "ft_s0 best (ep 55)", "ft_s0 last (ep 149)"]):
        for i, sname in enumerate(("pairs", "cross")):
            ax = axes[i, j]
            for split in ("train", "val", "test"):
                sel = [(p_, q_) for p_, q_ in sets[sname] if rl(p_) == split]
                t = 1e6 * np.array([tru["bunch_spacing"][p_] - tru["bunch_spacing"][q_] for p_, q_ in sel])
                p = 1e6 * np.array([preds[name]["bunch_spacing"][p_] - preds[name]["bunch_spacing"][q_] for p_, q_ in sel])
                m = res["flow"][name][f"{sname}/{split}"]
                ax.scatter(t, p, s=10, c=cols[split], alpha=0.7,
                           label=f"{split} n={m['n']}: R2 {m['R2']:.2f}, slope {m['slope']:.2f}")
            lim = np.array(ax.get_xlim())
            ax.plot(lim, lim, "k--", lw=1)
            ax.set(title=f"{name}: {sname}", xlabel="true injector effect on spacing (Bmad) [um]",
                   ylabel="flow-predicted effect [um]")
            ax.legend(fontsize=7)
    fig.suptitle("Injector effect on bunch spacing: flow vs Bmad. pairs = same linac knobs, injector point vs golden; "
                 "cross = injector response at the golden linac point", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "fig1_flow_pred_vs_true.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, sname in zip(axes, ("pairs", "cross")):
        for rname, v in lc.items():
            if sname in v:
                n_, mu, sd = map(np.array, zip(*v[sname]))
                ax.errorbar(n_, mu, yerr=sd, marker="o", capsize=3, label=rname)
        for name, mk in (("ft_s0 best (ep 55)", "*"), ("ft_s1 best", "P")):
            vv = [res["flow"][name][f"{sname}/{s}"] for s in ("val", "test") if f"{sname}/{s}" in res["flow"][name]]
            t_all = sum(x["n"] for x in vv)
            ax.scatter([len(train_pts)], [sum(x["R2"] * x["n"] for x in vv) / t_all], marker=mk, s=160, c="k", zorder=5,
                       label=f"flow {name} (val+test)")
        ax.axhline(0.8, c="0.5", ls=":", label="gate R2 0.8")
        ax.set(xlabel="training injector points", ylabel="held-out R2 of the injector effect", ylim=(-0.2, 1.05),
               title=f"{sname}: learning curves (val + test injector points)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_learning_curves.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.27
    xs = np.arange(len(SHORT))
    for k_, (name, c) in enumerate((("truth (Bmad)", "k"), ("ft_s0 best (ep 55)", "tab:green"),
                                    ("base (old flow, widened)", "0.6"))):
        ax.bar(xs + (k_ - 1) * w, sens[name], w, color=c, label=name)
    ax.set_xticks(xs, SHORT, rotation=30)
    ax.axhline(0, c="k", lw=0.6)
    ax.set(ylabel="d spacing / d knob [um per full box range]",
           title=f"linear injector sensitivities at the golden linac point (cross set; linear fit of truth R2 "
                 f"{res['truth_linear_R2_cross']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig3_per_knob_sensitivity.png", dpi=120)
    plt.close(fig)

    # residual vs region / spacing magnitude (pairs, all splits), and T calibration
    name = "ft_s0 best (ep 55)"
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    t = 1e6 * np.array([tru["bunch_spacing"][p_] - tru["bunch_spacing"][q_] for p_, q_ in pairs])
    p = 1e6 * np.array([preds[name]["bunch_spacing"][p_] - preds[name]["bunch_spacing"][q_] for p_, q_ in pairs])
    sp = 1e6 * np.array([tru["bunch_spacing"][q_] for _, q_ in pairs])
    reg = np.array([region[p_] for p_, _ in pairs])
    for rg, c in (("controller", "tab:purple"), ("full", "tab:cyan")):
        m = reg == rg
        if m.any():
            axes[0].scatter(sp[m], p[m] - t[m], s=10, c=c, alpha=0.7,
                            label=f"{rg} region: rmse {np.sqrt(np.mean((p[m]-t[m])**2)):.1f} um, n={m.sum()}")
    axes[0].axhline(0, c="k", lw=0.6)
    axes[0].set(xlabel="spacing at the reference row [um]", ylabel="predicted - true effect [um]",
                title=f"{name}: pair residual vs spacing / linac region")
    axes[0].legend(fontsize=7)
    tt = np.array([tru["T_witness"][r] for r in inj_rows])
    tp = np.array([preds[name]["T_witness"][r] for r in inj_rows])
    td, tdp = np.array([tru["T_drive"][r] for r in inj_rows]), np.array([preds[name]["T_drive"][r] for r in inj_rows])
    axes[1].scatter(tt, tp, s=4, alpha=0.4, label="witness")
    axes[1].scatter(td, tdp, s=4, alpha=0.4, label="drive")
    axes[1].plot([0.8, 1.0], [0.8, 1.0], "k--", lw=1)
    axes[1].set(xlim=(0.8, 1.005), ylim=(0.8, 1.005), xlabel="true transmission (per-sample label)",
                ylabel="predicted", title="transmission calibration (injector rows)")
    axes[1].legend(fontsize=7)
    axes[2].hist(tt, bins=60, alpha=0.6, label="true witness")
    axes[2].hist(tp, bins=60, alpha=0.6, label="pred witness")
    axes[2].set(xlabel="T_witness", title="transmission distributions", xlim=(0.8, 1.01))
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig4_residuals_and_T.png", dpi=120)
    plt.close(fig)
    print(json.dumps({k: v for k, v in res["flow"].items()}, indent=1))


if __name__ == "__main__":
    main()
