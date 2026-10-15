"""Harness audit part 1: timing, TCAV shot pattern, settling of BSA/nonBSA scalars, actuator
readbacks, feedback activity, charge monitors. Writes data/scalars_results.json + figures."""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AUD, E300, E331, load_mat, robust_std, tcav_class

warnings.filterwarnings("ignore")
RES = {}
FIG = AUD / "figs"


def step_tail(a, st, k, frac=0.5):
    m = np.where(st == k)[0]
    m = m[int(len(m) * (1 - frac)):]
    return np.nanmedian(a[m])


def settle_beta(a, st, nbin=5, nmax=None):
    """Pooled fraction of the previous step's offset remaining vs shot index within step.
    beta(n) = sum (x_k(n)-T_k)(T_{k-1}-T_k) / sum (T_{k-1}-T_k)^2, T = median of the last half."""
    ks = np.unique(st)
    T = {k: step_tail(a, st, k) for k in ks}
    nmax = nmax or int(np.median([np.sum(st == k) for k in ks]))
    edges = np.arange(0, nmax + 1, nbin)
    num, den, cnt = np.zeros(len(edges) - 1), np.zeros(len(edges) - 1), np.zeros(len(edges) - 1)
    resid = [[] for _ in range(len(edges) - 1)]
    for k in ks[1:]:
        m = np.where(st == k)[0]
        dk = T[k - 1] - T[k] if (k - 1) in T else np.nan
        if not np.isfinite(dk):
            continue
        for b in range(len(edges) - 1):
            idx = m[edges[b]:edges[b + 1]]
            x = a[idx] - T[k]
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            num[b] += np.sum(x * dk)
            den[b] += len(x) * dk ** 2
            cnt[b] += len(x)
            resid[b].extend(list(x))
    beta = num / np.where(den > 0, den, np.nan)
    # SE from residual scatter
    se = np.array([np.std(r) * np.sqrt(c) / np.sqrt(d) if d > 0 and len(r) > 2 else np.nan
                   for r, c, d in zip(resid, cnt, den)])
    return 0.5 * (edges[1:] + edges[:-1]), beta, se, np.array([T[k] for k in ks])


def first_change_index(a, st, k):
    m = np.where(st == k)[0]
    x = a[m]
    ok = np.isfinite(x)
    if ok.sum() < 2:
        return None
    x0 = x[ok][0]
    ch = np.where(ok & (np.abs(x - x0) > 1e-9))[0]
    return int(ch[0]) if len(ch) else -1


timing = {}
for r in E300 + E331:
    try:
        D = load_mat(r)
    except Exception as e:  # noqa
        timing[r] = {"error": str(e)}
        continue
    t, st = D["t"], D["step"]
    ks = np.unique(st)
    if len(ks) < 2:
        continue
    durs = np.array([t[st == k].max() - t[st == k].min() for k in ks])
    nshot = np.array([np.sum(st == k) for k in ks])
    gaps = np.array([t[st == k + 1].min() - t[st == k].max() for k in ks[:-1] if np.any(st == k + 1)])
    A, P = D["pv"]["TCAV_LI20_2400_A"], D["pv"]["TCAV_LI20_2400_P"]
    c, Am = tcav_class(A, P)
    fr = {ch: float(np.mean(c == ch)) for ch in "+-xao"}
    timing[r] = {
        "comment": str(D["params"]["comment"]), "n_steps": int(len(ks)), "n_shot_param": int(D["params"]["n_shot"]),
        "shot_dt_s_median": float(np.median(np.diff(t))), "shots_per_step_median": float(np.median(nshot)),
        "step_data_s_median": float(np.median(durs)), "gap_s_median": float(np.median(gaps)),
        "gap_s_min": float(gaps.min()), "gap_s_max": float(gaps.max()),
        "gap_s_p90": float(np.percentile(gaps, 90)),
        "total_min": float((t.max() - t.min()) / 60), "Waits": float(D["params"]["Waits"]),
        "tcav_frac": fr, "tcav_amp_on_MV": float(Am),
        "scan_range": [float(np.min(D["params"]["scanVals"])), float(np.max(D["params"]["scanVals"]))],
        "dropped_pid_frac": float(np.mean(np.diff(D["pid"]) != 36)),
    }
    if r in E300:
        cnt = np.array([[np.sum((st == k) & (c == ch)) for ch in "+-xao"] for k in ks])
        timing[r]["per_step_counts_median"] = dict(zip("+-xao", np.median(cnt, 0).tolist()))
        timing[r]["per_step_counts_min"] = dict(zip("+-xao", cnt.min(0).tolist()))
        timing[r]["steps_with_0_off"] = int(np.sum(cnt[:, 4] == 0))
        timing[r]["steps_with_lt2_off"] = int(np.sum(cnt[:, 4] < 2))
        # sign-block structure: durations of contiguous same-sign valid runs and of transitions
        blocks, trans = [], []
        for k in ks:
            seq = c[st == k]
            cur, ln, tl = None, 0, 0
            for ch in seq:
                if ch in "+-":
                    if ch == cur:
                        ln += 1
                    else:
                        if cur is not None and ln:
                            blocks.append(ln)
                        cur, ln = ch, 1
                        if tl:
                            trans.append(tl)
                        tl = 0
                elif ch in "xao":
                    tl += 1
        timing[r]["valid_block_len_median"] = float(np.median(blocks))
        timing[r]["transition_len_median"] = float(np.median([x for x in trans if x > 3])) if trans else None
        # TCAV amplitude / phase stability on valid shots
        v = (c == "+") | (c == "-")
        Ast = np.array([np.nanmedian(A[(st == k) & v]) for k in ks])
        timing[r]["tcav_amp_shot_rel_rms"] = float(robust_std(A[v]) / Am)
        timing[r]["tcav_amp_step_median_rel_span"] = float((Ast.max() - Ast.min()) / Am)
        timing[r]["tcav_amp_step_median_rel_std"] = float(np.std(Ast) / Am)
        timing[r]["tcav_phase_shot_rms_deg_plus"] = float(robust_std(P[c == "+"]))
        timing[r]["tcav_phase_shot_rms_deg_neg"] = float(robust_std(P[c == "-"]))
RES["timing"] = timing

# ---------------------------------------------------------------- settling + actuators (per run)
SETTLE_PVS = ["BLEN_LI20_3014_BRAW", "BLEN_LI11_359_BRAW", "BLEN_LI14_888_BRAW", "KLYS_LI11_11_FB_FAST_PACT",
              "KLYS_LI11_21_FB_FAST_PACT", "BPMS_LI14_801_X", "BPMS_LI14_715_X", "BPMS_LI14_701_X",
              "BPMS_LI11_333_X", "BPMS_LI20_2445_X", "BPMS_IN10_731_X", "TORO_LI20_2452_TMIT"]
settle, act, fb, charge = {}, {}, {}, {}
for r in E300 + E331:
    if r not in timing or "error" in timing[r]:
        continue
    D = load_mat(r)
    st, pv = D["step"], D["pv"]
    sv = np.asarray(D["params"]["scanVals"], float)
    ks = np.unique(st)
    settle[r] = {}
    for name in SETTLE_PVS:
        if name not in pv or np.mean(np.isfinite(pv[name])) < 0.8:
            continue
        a = pv[name].copy()
        if "PACT" in name:   # robust: per-shot readback has outliers
            med = np.nanmedian(a)
            a[np.abs(a - med) > 5] = np.nan
        nb = 5 if r in E300 else 2
        n, beta, se, T = settle_beta(a, st, nbin=nb)
        dT = np.diff(T)
        shot_rms = np.nanmedian([robust_std(a[st == k]) for k in ks])
        settle[r][name] = {"n": n.tolist(), "beta": beta.tolist(), "se": se.tolist(),
                           "step_delta_rms": float(np.nanstd(dT)), "shot_rms": float(shot_rms),
                           "delta_over_noise": float(np.nanstd(dT) / shot_rms) if shot_rms > 0 else None,
                           "step_tails": T.tolist()}
    # ---- actuator: SBST PDES vs PHAS (nonBSA, ~1 Hz updates)
    act[r] = {}
    for sec in ("LI12", "LI13", "LI14"):
        pd, ph = pv.get(f"{sec}_SBST_1_PDES"), pv.get(f"{sec}_SBST_1_PHAS")
        if pd is None or ph is None:
            continue
        PD = np.array([step_tail(pd, st, k) for k in ks])
        PH = np.array([step_tail(ph, st, k) for k in ks])
        ok = np.isfinite(PD) & np.isfinite(PH)
        g, off = np.polyfit(PD[ok], PH[ok], 1)
        res = PH[ok] - (g * PD[ok] + off)
        dPD, dPH = np.diff(PD), np.diff(PH)
        okd = np.isfinite(dPD) & np.isfinite(dPH) & (np.abs(dPD) > 1e-6)
        stepgain = dPH[okd] / dPD[okd]
        # readback latency: first in-step change of PHAS after a PDES step
        lat = [first_change_index(ph, st, k) for k in ks[1:]]
        lat = [x for x in lat if x is not None]
        # is PHAS at shot 0 of step k still the previous step's value?
        stale = []
        for k in ks[1:]:
            m = np.where(st == k)[0]
            x0 = ph[m][np.isfinite(ph[m])]
            if len(x0) == 0:
                continue
            prevT = PH[np.where(ks == k - 1)[0][0]]
            curT = PH[np.where(ks == k)[0][0]]
            if abs(curT - prevT) > 0.05:
                stale.append(abs(x0[0] - prevT) < abs(x0[0] - curT))
        act[r][sec] = {"gain_fit": float(g), "offset_deg": float(np.median(PH[ok] - PD[ok])),
                       "resid_rms_deg": float(np.std(res)), "stepgain_median": float(np.median(stepgain)),
                       "stepgain_p10_p90": [float(np.percentile(stepgain, 10)), float(np.percentile(stepgain, 90))],
                       "stepgain_rel_std": float(robust_std(stepgain)),
                       "pdes_step_deg": float(np.median(np.abs(dPD[okd]))),
                       "readback_first_change_idx_median": float(np.median([x for x in lat if x >= 0])) if any(x >= 0 for x in lat) else None,
                       "frac_steps_readback_stale_at_start": float(np.mean(stale)) if stale else None,
                       "n_unique_vals_per_step_median": float(np.median([len(np.unique(ph[(st == k) & np.isfinite(ph)])) for k in ks]))}
    # ---- feedbacks: L1 klystron fast phase vs L2 scan; BC energy BPMs; BC11 BLEN
    fb[r] = {}
    for name in ("KLYS_LI11_11_FB_FAST_PACT", "KLYS_LI11_21_FB_FAST_PACT", "KLYS_LI10_41_FB_FAST_PACT",
                 "KLYS_LI10_31_FB_FAST_PACT", "BLEN_LI11_359_BRAW", "BLEN_LI14_888_BRAW", "BLEN_LI20_3014_BRAW",
                 "BPMS_LI11_333_X", "BPMS_LI14_801_X", "BPMS_LI14_715_X", "BPMS_LI20_2445_X", "BPMS_IN10_731_X",
                 "LI15_SBST_1_PHAS", "LI16_SBST_1_PHAS", "LI17_SBST_1_PHAS", "LI18_SBST_1_PHAS", "LI19_SBST_1_PHAS",
                 "KLYS_LI11_11_SSSB_PDES", "KLYS_LI11_21_SSSB_PDES", "LI11_SBST_1_PHAS"):
        if name not in pv:
            continue
        a = pv[name].copy()
        if "PACT" in name:
            a[np.abs(a - np.nanmedian(a)) > 5] = np.nan
        valid = float(np.mean(np.isfinite(a)))
        M = np.array([np.nanmedian(a[st == k]) for k in ks])
        okm = np.isfinite(M)
        e = {"valid_frac": valid}
        if okm.sum() > 5:
            x = sv[ks[okm] - 1]
            sl = np.polyfit(x, M[okm], 1)[0]
            e.update({"slope_per_deg_L2": float(sl), "span": float(M[okm].max() - M[okm].min()),
                      "first": float(M[okm][0]), "last": float(M[okm][-1]),
                      "corr_with_scan": float(np.corrcoef(x, M[okm])[0, 1]),
                      "shot_rms": float(np.nanmedian([robust_std(a[st == k]) for k in ks]))})
        fb[r][name] = e
    # ---- charge monitors
    charge[r] = {}
    toros = [k for k in pv if k.startswith("TORO") or (k.startswith("BPMS") and k.endswith("TMIT"))]
    ref = pv.get("TORO_IN10_431_TMIT")
    for name in toros:
        a = pv[name]
        ok = np.isfinite(a) & (a > 1e8)
        if ok.mean() < 0.5:
            charge[r][name] = {"valid_frac": float(ok.mean())}
            continue
        q = a * 1.602176634e-19 * 1e9
        e = {"valid_frac": float(ok.mean()), "median_nC": float(np.median(q[ok])),
             "rel_rms_robust": float(robust_std(q[ok]) / np.median(q[ok])),
             "rel_rms_plain": float(np.std(q[ok]) / np.median(q[ok]))}
        if ref is not None and name != "TORO_IN10_431_TMIT":
            rr = a / ref
            okr = ok & np.isfinite(rr)
            Tm = np.array([np.nanmedian(rr[(st == k) & okr]) for k in ks])
            e.update({"ratio_to_IN10_431_median": float(np.median(rr[okr])),
                      "ratio_shot_rel_rms_robust": float(np.nanmedian([robust_std(rr[(st == k) & okr]) for k in ks]) / np.median(rr[okr])),
                      "ratio_step_median_span": float(np.nanmax(Tm) - np.nanmin(Tm)),
                      "ratio_step_median_min": float(np.nanmin(Tm)), "ratio_step_median_max": float(np.nanmax(Tm))})
        charge[r][name] = e

RES["settle"], RES["actuator"], RES["feedback"], RES["charge"] = settle, act, fb, charge
json.dump(RES, open(AUD / "data" / "scalars_results.json", "w"), indent=1, default=float)

# ---------------------------------------------------------------- figures
# F1: step-response beta(n) for E300 and E331
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), dpi=130)
for ax, runs, title, dt in ((axs[0], E300, "E300 L2 scans (150 shots/step, 0.25 deg)", 0.1),
                            (axs[1], E331, "E331 L2 scans (20 shots/step, 0.25 deg)", 0.1)):
    for name, col in (("BLEN_LI20_3014_BRAW", "C0"), ("KLYS_LI11_11_FB_FAST_PACT", "C1"),
                      ("BPMS_LI14_715_X", "C2"), ("BPMS_LI14_801_X", "C3")):
        bs, ns = [], None
        for r in runs:
            e = settle.get(r, {}).get(name)
            if e is None or (e["delta_over_noise"] or 0) < 0.05:
                continue
            bs.append(e["beta"])
            ns = e["n"]
        if not bs:
            continue
        L = min(len(b) for b in bs)
        B = np.array([b[:L] for b in bs], float)
        ax.plot(np.array(ns[:L]) * dt, np.nanmedian(B, 0), "o-", color=col, ms=3,
                label=f"{name} (median of {len(bs)} runs)")
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(1, color="gray", lw=0.8, ls="--")
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlabel("time since first recorded shot of step [s]")
    ax.set_ylabel("fraction of previous-step offset remaining")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=6.5)
fig.suptitle("Settling at step start: 1 = still at previous step's value, 0 = settled (recording starts after a ~6-7 s gap)", fontsize=9)
fig.tight_layout()
fig.savefig(FIG / "fig2_step_response.png")
plt.close(fig)
print(json.dumps(timing, indent=0, default=float)[:3000])
