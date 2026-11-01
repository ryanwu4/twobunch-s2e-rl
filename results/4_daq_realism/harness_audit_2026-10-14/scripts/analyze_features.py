"""Harness audit part 2: XTCAV feature noise, averaging, drift, shot-class usability, calibration,
per-bunch charge. Inputs: data/reext_*.npz (this audit) + Tier 0 summary.json (calibration).
Writes data/features_results.json + figs/fig3_noise_vs_window.png + figs/fig4_calibration.png."""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AUD, BASE, D_NOM_MM, E300, load_mat, load_reext, robust_std

warnings.filterwarnings("ignore")
T0 = BASE / "sim2real" / "results" / "tier0_real_features"
ENV = {"spacing_um": 15.0, "dsig_um": 2.0, "fd": 0.03, "M": 50}
RES = {}
allan_store = {}


def classes(R):
    A, P = R["amp"], R["phase"]
    good = (np.abs(P - 90) < 1) | (np.abs(P + 90) < 1)
    Am = np.nanmean(A[good & (A > 10)])
    c = np.full(len(A), "a", dtype="<U1")
    ampok = np.abs(A - Am) < 0.25
    c[ampok] = "x"
    c[ampok & (np.abs(P - 90) < 1)] = "+"
    c[ampok & (np.abs(P + 90) < 1)] = "-"
    return c


def blocks_of(idx_sorted, signs):
    """Split a time-ordered list of valid shots into same-sign blocks."""
    out, cur = [], [idx_sorted[0]]
    for a, b in zip(idx_sorted[:-1], idx_sorted[1:]):
        if signs[b] == signs[a] and (b - a) < 25:
            cur.append(b)
        else:
            out.append(cur)
            cur = [b]
    out.append(cur)
    return out


def allan(x, ms):
    res = {}
    for m in ms:
        n = len(x) // m
        if n < 2:
            continue
        y = x[: n * m].reshape(n, m).mean(1)
        res[m] = (0.5 * np.mean(np.diff(y) ** 2), n - 1)
    return res


for r in E300:
    R = load_reext(r)
    D = load_mat(r)
    ci, t = D["common_index"], D["t"]
    R["t"] = t[ci[R["ind"].astype(int)] - 1]
    c = classes(R)
    ok = R["ok"] == 1
    sgn = np.where(c == "+", 1.0, np.where(c == "-", -1.0, np.nan))
    sgn_any = np.sign(R["phase"])
    sep = R["z_lo"] - R["z_hi"]
    sz = R["sz_lo"]
    fd = R["Q_lo"] / (R["Q_lo"] + R["Q_hi"])
    dE = (R["y_lo"] - R["y_hi"]) / D_NOM_MM * R["E_bend"] * 1e3     # MeV
    feats = {"sep": sep, "dsig": sz, "fd": fd, "dE": dE}
    valid = ok & np.isfinite(sgn)
    steps = np.unique(R["step"]).astype(int)
    per_step = []
    for k in steps:
        e = {"step": int(k)}
        for s, tag in ((1, "p"), (-1, "n")):
            m = valid & (R["step"] == k) & (sgn == s)
            e[f"n_{tag}"] = int(m.sum())
            for f, v in feats.items():
                e[f"{f}_mean_{tag}"] = float(np.nanmean(v[m])) if m.any() else np.nan
                e[f"{f}_std_{tag}"] = float(np.nanstd(v[m], ddof=1)) if m.sum() > 2 else np.nan
                e[f"{f}_rstd_{tag}"] = float(robust_std(v[m]))
            e[f"var_mean_{tag}"] = float(np.nanmean(sz[m] ** 2)) if m.any() else np.nan
        e["spacing"] = 0.5 * (e["sep_mean_p"] - e["sep_mean_n"])
        e["spacing_se"] = 0.5 * np.sqrt(e["sep_std_p"] ** 2 / max(e["n_p"], 1) + e["sep_std_n"] ** 2 / max(e["n_n"], 1))
        e["spacing_se_robust"] = 0.5 * np.sqrt(e["sep_rstd_p"] ** 2 / max(e["n_p"], 1) + e["sep_rstd_n"] ** 2 / max(e["n_n"], 1))
        e["pair_sigma"] = 0.5 * np.sqrt(e["sep_std_p"] ** 2 + e["sep_std_n"] ** 2)
        e["pair_sigma_robust"] = 0.5 * np.sqrt(e["sep_rstd_p"] ** 2 + e["sep_rstd_n"] ** 2)
        e["dsig_halfvar"] = float(np.sqrt(0.5 * (e["var_mean_p"] + e["var_mean_n"])))
        mm = valid & (R["step"] == k)
        e["fd_all_mean"] = float(np.nanmean(fd[mm]))
        e["fd_all_std"] = float(np.nanstd(fd[mm], ddof=1))
        e["fd_all_rstd"] = float(robust_std(fd[mm]))
        e["dsig_all_std"] = float(np.nanstd(sz[mm], ddof=1))
        e["dsig_all_rstd"] = float(robust_std(sz[mm]))
        e["dE_all_rstd"] = float(robust_std(dE[mm]))
        e["dE_all_mean"] = float(np.nanmean(dE[mm]))
        e["n_valid"] = int(mm.sum())
        e["n_on_any"] = int(np.sum((R["step"] == k)))
        per_step.append(e)
    PS = {k: np.array([e[k] for e in per_step], float) for k in per_step[0]}
    well = np.abs(PS["spacing"]) > 80
    summ = {"n_steps": len(steps), "n_well_separated_steps": int(well.sum()),
            "valid_images_per_step_median": float(np.median(PS["n_valid"])),
            "valid_frac_of_on_images": float(np.sum(valid) / len(valid)),
            "band_split_ok_frac": float(np.mean(ok)),
            "spacing_range_um": [float(np.nanmin(np.abs(PS["spacing"]))), float(np.nanmax(np.abs(PS["spacing"])))]}
    for key in ("sep_std_p", "sep_std_n", "sep_rstd_p", "sep_rstd_n", "pair_sigma", "pair_sigma_robust",
                "spacing_se", "spacing_se_robust", "dsig_all_std", "dsig_all_rstd", "fd_all_std", "fd_all_rstd",
                "dE_all_rstd"):
        summ[key] = {"median_all": float(np.nanmedian(PS[key])), "median_well": float(np.nanmedian(PS[key][well])),
                     "p25_p75_all": [float(np.nanpercentile(PS[key], 25)), float(np.nanpercentile(PS[key], 75))]}
    summ["drive_sigma_z_um_range"] = [float(np.nanmin(PS["dsig_halfvar"])), float(np.nanmax(PS["dsig_halfvar"]))]
    summ["fd_range"] = [float(np.nanmin(PS["fd_all_mean"])), float(np.nanmax(PS["fd_all_mean"]))]
    summ["dE_MeV_range"] = [float(np.nanmin(PS["dE_all_mean"])), float(np.nanmax(PS["dE_all_mean"]))]
    # ---- adjacent-step consistency: residual from neighbour mean vs predicted SE
    s, se = np.abs(PS["spacing"]), PS["spacing_se"]
    rr, pv_ = [], []
    for j in range(1, len(s) - 1):
        if np.all(np.isfinite([s[j - 1], s[j], s[j + 1], se[j - 1], se[j], se[j + 1]])):
            rr.append(s[j] - 0.5 * (s[j - 1] + s[j + 1]))
            pv_.append(se[j] ** 2 + 0.25 * (se[j - 1] ** 2 + se[j + 1] ** 2))
    rr, pv_ = np.array(rr), np.array(pv_)
    summ["adjacent_step_resid_rms_um"] = float(np.sqrt(np.mean(rr ** 2)))
    summ["adjacent_step_resid_robust_um"] = float(robust_std(rr))
    summ["adjacent_step_pred_rms_um"] = float(np.sqrt(np.mean(pv_)))
    summ["adjacent_step_chi2_per_dof"] = float(np.mean(rr ** 2 / pv_))
    summ["adjacent_step_chi2_per_dof_median_based"] = float(np.median(rr ** 2 / pv_) / 0.455)
    # ---- within-step, same-sign block-to-block drift (~7.5 s apart) and autocorrelation / Allan
    zs = {f: [] for f in feats}
    ds = {f: [] for f in feats}
    dts = []
    ac = {f: {1: [], 2: [], 3: []} for f in feats}
    al = {f: {} for f in feats}
    for k in steps:
        idx = np.where(valid & (R["step"] == k))[0]
        if len(idx) < 4:
            continue
        idx = idx[np.argsort(R["t"][idx])]
        bl = blocks_of(list(idx), sgn)
        for f, v in feats.items():
            for b in bl:
                x = v[b]
                x = x[np.isfinite(x)]
                if len(x) >= 10:
                    x0 = x - x.mean()
                    for L in (1, 2, 3):
                        ac[f][L].append((np.sum(x0[:-L] * x0[L:]), np.sum(x0 ** 2) * (len(x0) - L) / len(x0)))
                    for m_, (v2, n2) in allan(x0, [1, 2, 3, 4, 6, 8, 12]).items():
                        al[f].setdefault(m_, []).append((v2, n2, np.var(x, ddof=1)))
            for s_ in (1, -1):
                bb = [b for b in bl if sgn[b[0]] == s_ and len(b) >= 8]
                for b1, b2 in zip(bb[:-1], bb[1:]):
                    x1, x2 = v[b1], v[b2]
                    x1, x2 = x1[np.isfinite(x1)], x2[np.isfinite(x2)]
                    d = x2.mean() - x1.mean()
                    se2 = x1.var(ddof=1) / len(x1) + x2.var(ddof=1) / len(x2)
                    zs[f].append(d / np.sqrt(se2))
                    ds[f].append(d)
                    if f == "sep":
                        dts.append(R["t"][b2].mean() - R["t"][b1].mean())
    summ["block_drift"] = {f: {"n_pairs": len(zs[f]), "z_rms": float(np.sqrt(np.mean(np.square(zs[f])))) if zs[f] else None,
                               "z_robust": float(robust_std(zs[f])) if zs[f] else None,
                               "abs_diff_median": float(np.median(np.abs(ds[f]))) if ds[f] else None}
                           for f in feats}
    summ["block_drift_dt_s_median"] = float(np.median(dts)) if dts else None
    summ["autocorr"] = {f: {L: float(np.sum([a for a, _ in v]) / np.sum([b for _, b in v])) if v else None
                            for L, v in ac[f].items()} for f in feats}
    # Allan ratio: sigma_A(m) / (sigma_1 / sqrt(m)) pooled
    summ["allan_ratio_to_white"] = {}
    for f in feats:
        sig1 = np.sqrt(np.sum([a * n for a, n, _ in al[f].get(1, [])]) / max(1, np.sum([n for _, n, _ in al[f].get(1, [])])))
        out = {}
        for m_, lst in sorted(al[f].items()):
            sA = np.sqrt(np.sum([a * n for a, n, _ in lst]) / np.sum([n for _, n, _ in lst]))
            out[m_] = float(sA / (sig1 / np.sqrt(m_)))
        summ["allan_ratio_to_white"][f] = out
        allan_store.setdefault(f, {})[r] = out
    # ---- usability of phase-/amplitude-settling shots: bias vs same-step same-sign valid shots
    use = {}
    for cl in ("x", "a"):
        b_sep, b_fd, b_sz, n_ = [], [], [], 0
        for k in steps:
            for s_ in (1, -1):
                mv = valid & (R["step"] == k) & (sgn == s_)
                mc = ok & (c == cl) & (R["step"] == k) & (sgn_any == s_)
                if mv.sum() < 5 or mc.sum() < 3:
                    continue
                sdv = np.nanstd(sep[mv])
                b_sep.append((np.nanmedian(sep[mc]) - np.nanmedian(sep[mv])) / max(sdv, 1e-9))
                b_fd.append(np.nanmedian(fd[mc]) - np.nanmedian(fd[mv]))
                b_sz.append(np.nanmedian(sz[mc]) - np.nanmedian(sz[mv]))
                n_ += int(mc.sum())
        use[cl] = {"n_images": n_, "sep_bias_in_shot_sigma_median": float(np.median(b_sep)) if b_sep else None,
                   "sep_bias_in_shot_sigma_abs_median": float(np.median(np.abs(b_sep))) if b_sep else None,
                   "fd_bias_median": float(np.median(b_fd)) if b_fd else None,
                   "dsig_bias_um_median": float(np.median(b_sz)) if b_sz else None,
                   "phase_dev_deg_median": float(np.nanmedian(np.abs(np.abs(R["phase"][ok & (c == cl)]) - 90))),
                   "amp_dev_rel_median": float(np.nanmedian(np.abs(R["amp"][ok & (c == cl)] / np.nanmean(R["amp"][valid]) - 1)))}
    summ["settling_shot_usability"] = use
    # ---- per-bunch charge: screen fraction x LI20 toroid; screen intensity vs toroid
    q3163 = D["pv"].get("TORO_LI20_3163_TMIT")
    q2452 = D["pv"].get("TORO_LI20_2452_TMIT")
    qin = D["pv"].get("TORO_IN10_431_TMIT")
    sidx = ci[R["ind"].astype(int)] - 1
    scr = R["Q_lo"] + R["Q_hi"]
    pq = {}
    for name, q in (("TORO_LI20_3163", q3163), ("TORO_LI20_2452", q2452)):
        if q is None:
            continue
        qq = q[sidx]
        m = valid & np.isfinite(qq) & (qq > 1e8) & np.isfinite(scr)
        pq[f"corr_screen_intensity_vs_{name}"] = float(np.corrcoef(scr[m], qq[m])[0, 1])
    # witness/drive charge per shot: Qd = Q_tor * fd; transmission proxy T_d = Qd/(Qin*fd0)
    if q2452 is not None and qin is not None:
        Tt = (q2452 / qin)[sidx]
        rel = []
        for k in steps:
            m = valid & (R["step"] == k) & np.isfinite(Tt)
            if m.sum() < 10:
                continue
            Td = Tt[m] * fd[m]
            Tw = Tt[m] * (1 - fd[m])
            rel.append((robust_std(Tt[m]) / np.median(Tt[m]), robust_std(Td) / np.median(Td), robust_std(Tw) / np.median(Tw)))
        rel = np.array(rel)
        pq["T_total_shot_rel_rms_median"] = float(np.median(rel[:, 0]))
        pq["Q_drive_LI20_shot_rel_rms_median"] = float(np.median(rel[:, 1]))
        pq["Q_witness_LI20_shot_rel_rms_median"] = float(np.median(rel[:, 2]))
    summ["per_bunch_charge"] = pq
    RES[r] = {"summary": summ, "per_step": per_step}

# ---- calibration stability from Tier 0 jitter regressions
cal = {}
for r in E300:
    S = json.load(open(T0 / r / "summary.json"))["calibration"]
    vals, errs, stp, tag = [], [], [], []
    for k, e in S["per_step"].items():
        for sg in ("plus", "neg"):
            if f"slope_{sg}_um_per_deg" in e:
                vals.append(abs(e[f"slope_{sg}_um_per_deg"]))
                errs.append(e[f"slope_{sg}_err"])
                stp.append(int(k))
                tag.append(sg)
    v, er = np.array(vals), np.array(errs)
    good = (er < 0.3 * v) & (v > 300)
    w = 1 / er[good] ** 2
    mu = np.sum(w * v[good]) / np.sum(w)
    chi2 = np.sum(w * (v[good] - mu) ** 2) / max(1, good.sum() - 1)
    # first vs second half of the dataset (time stability)
    st_ = np.array(stp)
    h1 = good & (st_ <= 16)
    h2 = good & (st_ > 16)
    def wm(mask):
        ww = 1 / er[mask] ** 2
        return float(np.sum(ww * v[mask]) / np.sum(ww)), float(1 / np.sqrt(np.sum(ww)))
    cal[r] = {"median_all_um_per_deg": float(np.median(v)), "n_regr": int(len(v)), "n_good": int(good.sum()),
              "wmean_good": float(mu), "wmean_se": float(1 / np.sqrt(np.sum(w))), "chi2_per_dof": float(chi2),
              "per_regression_rel_scatter_robust": float(robust_std(v[good]) / np.median(v[good])),
              "median_rel_err_per_regression": float(np.median(er[good] / v[good])),
              "first_half": wm(h1), "second_half": wm(h2), "per_step": list(zip(stp, vals, errs, tag))}
RES["calibration"] = cal
json.dump(RES, open(AUD / "data" / "features_results.json", "w"), indent=1, default=float)

# ---- figure: Allan ratio vs window
fig, axs = plt.subplots(1, 3, figsize=(13, 3.8), dpi=130)
for ax, f, lab in zip(axs, ("sep", "dsig", "fd"), ("signed band separation", "drive sigma_z (single image)", "drive charge fraction")):
    for r, col in zip(E300, ("C0", "C1", "C2")):
        o = allan_store[f][r]
        ms = sorted(o)
        ax.plot(ms, [o[m] for m in ms], "o-", color=col, label=r)
    ax.axhline(1, color="k", ls="--", lw=0.8, label="white noise")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("averaging window m [valid same-sign images, ~0.1-0.2 s each]")
    ax.set_ylabel("Allan dev / (sigma_1 / sqrt(m))")
    ax.set_title(lab, fontsize=9)
    ax.set_ylim(0, 3)
    ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(AUD / "figs" / "fig3_noise_vs_window.png")
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
for r, col in zip(E300, ("C0", "C1", "C2")):
    ps = cal[r]["per_step"]
    for sg, mk in (("plus", "o"), ("neg", "s")):
        q = [(s, v, e) for s, v, e, t in ps if t == sg and e < 0.3 * v and v > 300]
        if q:
            s_, v_, e_ = zip(*q)
            axs[0].errorbar(np.array(s_) + (0.15 if sg == "neg" else 0), v_, e_, fmt=mk, color=col, ms=3, lw=0.6,
                            label=f"{r} {sg}")
    axs[0].axhline(cal[r]["wmean_good"], color=col, lw=0.8)
axs[0].set_xlabel("scan step")
axs[0].set_ylabel("|d centroid / d TCAV phase| [um/deg]")
axs[0].set_title("streak calibration from natural TCAV phase jitter (Tier 0 regressions)", fontsize=9)
axs[0].legend(fontsize=5.5, ncol=2)
for r, col in zip(E300, ("C0", "C1", "C2")):
    PS = RES[r]["per_step"]
    s_ = [e["step"] for e in PS]
    axs[1].errorbar(s_, np.abs([e["spacing"] for e in PS]), [e["spacing_se"] for e in PS], fmt="o-", ms=3, lw=0.8,
                    color=col, label=f"{r}: half-difference spacing (all valid images in step)")
axs[1].set_xlabel("scan step (L2 +0.25 deg / step)")
axs[1].set_ylabel("|spacing| [um] (+/- 1 SE)")
axs[1].legend(fontsize=6)
fig.tight_layout()
fig.savefig(AUD / "figs" / "fig4_calibration_and_spacing.png")
for r in E300:
    print("==", r)
    print(json.dumps(RES[r]["summary"], indent=0, default=float)[:6000])
print(json.dumps({r: {k: v for k, v in cal[r].items() if k != "per_step"} for r in cal}, indent=0))
