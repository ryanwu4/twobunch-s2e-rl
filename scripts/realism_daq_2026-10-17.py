"""DAQ realism constraints for the delta-RL env (2026-10-17). Read-only on the DAQ .mat files.

Run: OMP_NUM_THREADS=1 /home/rwu4/miniconda3/envs/slac-rl/bin/python results/realism_2026-10-17/realism.py
Writes summary.json, recommended_params.json, fig*.png in this directory.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "harness_audit_2026-10-14" / "scripts"))
from common import E300, E331, load_mat, robust_std  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUNS = E300 + E331
K_NOM = -0.22          # d lnS11 / d phi (per deg), shot-level median; used only for the tiny residual-error term
T_CTRL = 20.0          # s per controller step (6.5 s settle + ~130 shots for 30 valid images/sign at 10 Hz)
M_ARX = 40


def ols(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    A = np.c_[np.ones_like(x), x]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    cov = r.var(ddof=2) * np.linalg.inv(A.T @ A)
    return b[1], np.sqrt(cov[1, 1]), b[0]


def structure_fit(t, x, tmax, ramp=True, nb=12):
    """D(tau) = <(x_i-x_j)^2> binned in tau (min); fit D = 2 s_w^2 + q tau (+ r^2 tau^2) with nonneg LSQ."""
    from scipy.optimize import nnls
    m = np.isfinite(x)
    t, x = t[m], x[m]
    i, j = np.triu_indices(len(t), 1)
    tau = np.abs(t[j] - t[i])
    dd = (x[j] - x[i]) ** 2
    k = tau <= tmax
    edges = np.unique(np.quantile(tau[k], np.linspace(0, 1, nb + 1)))
    tc, Dc = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s = k & (tau >= a) & (tau < b)
        if s.sum() > 5:
            tc.append(tau[s].mean())
            Dc.append(dd[s].mean())
    tc, Dc = np.array(tc), np.array(Dc)
    A = np.c_[np.full_like(tc, 2.0), tc] if not ramp else np.c_[np.full_like(tc, 2.0), tc, tc ** 2]
    coef, _ = nnls(A / Dc[:, None] ** 0.5, Dc / Dc ** 0.5)  # relative weighting
    return dict(tau=tc.tolist(), D=Dc.tolist(), sw2=coef[0], q=coef[1], r=np.sqrt(coef[2]) if ramp else 0.0)


# --------------------------------------------------------------------------------------------- load
R = {}
for r in RUNS:
    d = load_mat(r)
    pv, st, t = d["pv"], d["step"], d["t"]
    a, b = pv["KLYS_LI11_11_FB_FAST_PACT"], pv["KLYS_LI11_21_FB_FAST_PACT"]
    okc = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
    c = np.where(okc, 0.5 * (a + b), np.nan)
    S11 = pv["BLEN_LI11_359_BRAW"]
    S14 = pv["BLEN_LI14_888_BRAW"]
    Q = pv["TORO_LI11_360_TMIT"]
    Q = np.where(Q > 0, Q, np.nan)
    R[r] = dict(t=t, st=st, c=c, a=np.where(okc, a, np.nan), b=np.where(okc, b, np.nan),
                lS=np.log(S11), lS14=np.log(S14), lQ=np.log(Q), Q14=pv.get("TORO_LI14_890_TMIT"),
                lQ591=np.log(np.where(pv["TORO_IN10_591_TMIT"] > 0, pv["TORO_IN10_591_TMIT"], np.nan)),
                sb={s: (pv[f"{s}_SBST_1_PHAS"], pv[f"{s}_SBST_1_PDES"]) for s in ("LI12", "LI13", "LI14")},
                n_shot=int(d["params"]["n_shot"]), rate=str(d["params"]["rate"]),
                drop_c=float(np.mean(~okc)), drop_S=float(np.mean(~np.isfinite(S11))))
t0_day = {r: R[r]["t"][0] for r in RUNS}
S = {}

# ------------------------------------------------------------------------------------ 1. step timing
timing = {}
for r in RUNS:
    t, st = R[r]["t"], R[r]["st"]
    us = np.unique(st)
    t_first = np.array([t[st == s][0] for s in us])
    t_last = np.array([t[st == s][-1] for s in us])
    n = np.array([np.sum(st == s) for s in us])
    timing[r] = dict(period_s=float(np.median(np.diff(t_first))), period_p10_p90=np.percentile(np.diff(t_first), [10, 90]).round(2).tolist(),
                     span_s=float(np.median(t_last - t_first)), gap_s=float(np.median(t_first[1:] - t_last[:-1])),
                     shots_per_step=float(np.median(n)), shot_dt_s=float(np.median(np.diff(t))),
                     duration_min=float((t[-1] - t[0]) / 60), rate=R[r]["rate"])
S["timing"] = timing

# ---------------------------------------------------------------- per-step aggregates (L1 + charge)
def step_table(r):
    D = R[r]
    out = {k: [] for k in ("tm", "c", "e", "lQ", "lQ591", "lS14", "csd", "esd", "qsd", "n", "P")}
    lSset = np.nanmedian(D["lS"])
    for s in np.unique(D["st"]):
        m = D["st"] == s
        v = m & np.isfinite(D["c"]) & np.isfinite(D["lS"])
        out["tm"].append(np.mean(D["t"][m]))
        out["c"].append(np.nanmean(D["c"][m]))
        out["e"].append(np.nanmean(D["lS"][v]) - lSset if v.sum() else np.nan)
        out["lQ"].append(np.nanmean(D["lQ"][m]))
        out["lQ591"].append(np.nanmean(D["lQ591"][m]))
        out["lS14"].append(np.nanmedian(D["lS14"][m]))
        out["P"].append(np.nanmedian(D["sb"]["LI12"][1][m]))
        cc = D["c"][m]
        tt = D["t"][m]
        g = np.isfinite(cc)
        if g.sum() > 5:  # within-step fast jitter: residual about a linear trend inside the step
            p = np.polyfit(tt[g] - tt[g][0], cc[g], 1)
            out["csd"].append(np.std(cc[g] - np.polyval(p, tt[g] - tt[g][0])))
        else:
            out["csd"].append(np.nan)
        out["esd"].append(np.nanstd(D["lS"][v]))
        out["qsd"].append(np.nanstd(D["lQ"][m]))
        out["n"].append(v.sum())
    return {k: np.array(v) for k, v in out.items()}


T = {r: step_table(r) for r in RUNS}

def l2_offsets(r, sec):
    D = R[r]
    ph, pd = D["sb"][sec]
    t, st = D["t"], D["st"]
    us = np.unique(st)
    out_t, out_o, out_p = [], [], []
    prev_last = None
    for i, s in enumerate(us):
        idx = np.where(st == s)[0]
        tmove = t[np.where(st == us[i - 1])[0][-1]] if i > 0 else t[idx[0]] - 6
        p, q = ph[idx], pd[idx]
        g = np.isfinite(p) & np.isfinite(q)
        if g.sum() == 0:
            continue
        pi, ti, qi = p[g], t[idx][g], q[g]
        ch = np.r_[False, np.diff(pi) != 0]           # fresh update events inside the recorded window
        good = ch & (ti - tmove >= 3.0)
        if good.any():
            k = np.where(good)[0][-1]
            out_t.append(ti[k]); out_o.append(pi[k] - qi[k]); out_p.append(qi[k])
    return np.array(out_t), np.array(out_o), np.array(out_p)



# ---------------------------------- scan coupling: fit y = a_run + r*t + s*P over adjacent E331 run pairs
def pairfit(tA, yA, PA, tB, yB, PB):
    t = np.r_[tA, tB] / 60.0
    y, P = np.r_[yA, yB], np.r_[PA, PB]
    ia = np.r_[np.ones(len(tA)), np.zeros(len(tB))]
    m = np.isfinite(y) & np.isfinite(P)
    A = np.c_[ia, 1 - ia, t - t[m].mean(), P - P[m].mean()][m]
    cf, *_ = np.linalg.lstsq(A, y[m], rcond=None)
    res = y[m] - A @ cf
    cov = res.var(ddof=4) * np.linalg.inv(A.T @ A)
    return cf[2], cf[3], np.sqrt(cov[3, 3])


coup = {"PACT": []}
for ra, rb in zip(E331[:-1], E331[1:]):
    XA, XB = T[ra], T[rb]
    r_, s_, se_ = pairfit(XA["tm"], XA["c"] - XA["e"] / K_NOM, XA["P"], XB["tm"], XB["c"] - XB["e"] / K_NOM, XB["P"])
    coup["PACT"].append(dict(pair=f"{ra[-5:]}-{rb[-5:]}", ramp=r_, s=s_, s_se=se_))
S_PACT = float(np.median([c["s"] for c in coup["PACT"]]))
GAM = {}
for sec in ("LI12", "LI13", "LI14"):
    coup[sec] = []
    for ra, rb in zip(E331[:-1], E331[1:]):
        tA, oA, pA = l2_offsets(ra, sec)
        tB, oB, pB = l2_offsets(rb, sec)
        if len(tA) < 4 or len(tB) < 4:
            continue
        r_, s_, se_ = pairfit(tA, oA, pA, tB, oB, pB)
        coup[sec].append(dict(pair=f"{ra[-5:]}-{rb[-5:]}", ramp=r_, s=s_, s_se=se_))
    GAM[sec] = float(np.median([c["s"] for c in coup[sec]]))
S["scan_coupling"] = dict(pairs=coup, PACT_dpact_dPDES=S_PACT,
                          PACT_s_p16_p84=np.percentile([c["s"] for c in coup["PACT"]], [16, 84]).tolist(),
                          PACT_pair_ramp_median=float(np.median([c["ramp"] for c in coup["PACT"]])),
                          readback_gain_minus1=GAM)

# --------------------------------------------- 2. L1 beam-side drift (= BC11-loop slow correction)
l1 = {}
for r in RUNS:
    X = T[r]
    tmin = (X["tm"] - X["tm"][0]) / 60
    x = X["c"] - X["e"] / K_NOM                     # loop correction + not-yet-corrected residual error
    ramp_raw = ols(tmin, x)[0]
    x = x - S_PACT * (X["P"] - np.nanmean(X["P"]))   # remove the L2-scan-correlated part (fit on E331 fwd/bwd pairs)
    ramp, ramp_se, _ = ols(tmin, x)
    bq, bq_se, _ = ols(X["lQ"] - np.nanmean(X["lQ"]), x - np.nanmean(x))
    # charge-explained part: multiple regression x ~ t + lnQ
    m = np.isfinite(x) & np.isfinite(X["lQ"])
    A = np.c_[np.ones(m.sum()), tmin[m], X["lQ"][m] - np.nanmean(X["lQ"][m])]
    cf, *_ = np.linalg.lstsq(A, x[m], rcond=None)
    resid_tq = x[m] - A @ cf
    sf = structure_fit(tmin, x, tmax=0.6 * tmin[-1], ramp=True)
    xr = x - (ramp * tmin)
    sf0 = structure_fit(tmin, xr, tmax=0.6 * tmin[-1], ramp=False)
    l1[r] = dict(fast_jitter_deg=float(np.nanmedian(X["csd"])), ramp_raw_deg_per_min=float(ramp_raw),
                 span_deg=float(np.nanmax(x) - np.nanmin(x)),
                 ramp_deg_per_min=float(ramp), ramp_se=float(ramp_se),
                 dphi_dlnQ_deg=float(bq), dphi_dlnQ_se=float(bq_se),
                 ramp_given_Q=float(cf[1]), dphi_dlnQ_given_t=float(cf[2]),
                 rw_q_deg2_per_min_detrended=float(sf0["q"]), white_step_deg=float(np.sqrt(sf0["sw2"])),
                 rw_q_deg2_per_min_with_ramp=float(sf["q"]), ramp_from_SF=float(sf["r"]),
                 sf_tau=sf0["tau"], sf_D=sf0["D"],
                 pact_mean=float(np.nanmean(X["c"])), lS_set=float(np.nanmedian(R[r]["lS"])),
                 drop_frac_pact=R[r]["drop_c"], drop_frac_blen11=R[r]["drop_S"])

# long-lag structure function across consecutive E331 runs with the same BC11 setpoint
groups = []
cur = [E331[0]]
for r in E331[1:]:
    if abs(l1[r]["lS_set"] - l1[cur[-1]]["lS_set"]) < 0.05:
        cur.append(r)
    else:
        groups.append(cur)
        cur = [r]
groups.append(cur)
S["E331_setpoint_groups"] = [[g, float(np.exp(l1[g[0]]["lS_set"]))] for g in groups]
long_sf = {}
for g in groups:
    if len(g) < 2:
        continue
    tt = np.concatenate([T[r]["tm"] for r in g]) / 60
    xx = np.concatenate([T[r]["c"] - T[r]["e"] / K_NOM - S_PACT * T[r]["P"] for r in g])
    long_sf["+".join(x[-5:] for x in g)] = structure_fit(tt - tt[0], xx, tmax=tt[-1] - tt[0], ramp=True, nb=16)
S["L1_long_structure"] = {k: dict(q=v["q"], r=v["r"], sw2=v["sw2"], tau_max=v["tau"][-1], D_max=v["D"][-1]) for k, v in long_sf.items()}
S["L1"] = l1

# ---------------------------------------------------------------------- 3. BC11 loop dynamics (ARX)
loop = {}
for r in E300 + ["E331_16019"]:
    D = R[r]
    e = D["lS"] - np.nanmedian(D["lS"])
    rows, ys = [], []
    for s in np.unique(D["st"]):
        idx = np.where(D["st"] == s)[0]
        cc, ee = D["c"][idx], e[idx]
        for n in range(M_ARX, len(idx)):
            xc, xe = cc[n - M_ARX:n][::-1], ee[n - M_ARX:n][::-1]
            if np.isfinite(cc[n]) and np.all(np.isfinite(xc)) and np.all(np.isfinite(xe)):
                rows.append(np.r_[1.0, xc, xe])
                ys.append(cc[n])
    Xm, y = np.array(rows), np.array(ys)
    beta, *_ = np.linalg.lstsq(Xm, y, rcond=None)
    ac, be = beta[1:M_ARX + 1], beta[M_ARX + 1:]
    res = y - Xm @ beta
    cov = res.var() * np.linalg.inv(Xm.T @ Xm)
    # ramp-rate of c under a sustained unit ln-error (deg/s per unit ln), from the ARX step response 50..150 shots
    h = np.zeros(200)
    for n in range(200):
        h[n] = sum(ac[m] * h[n - 1 - m] for m in range(min(n, M_ARX))) + be[:min(n, M_ARX)].sum()
    rate = np.polyfit(np.arange(50, 150) / 10.0, h[50:150], 1)[0]
    # shot-level slopes: lnS on c (and lnQ), within-step demeaned
    cs, ls, qs = [], [], []
    for s in np.unique(D["st"]):
        m = (D["st"] == s) & np.isfinite(D["c"]) & np.isfinite(D["lS"]) & np.isfinite(D["lQ"])
        if m.sum() < 10:
            continue
        cs.append(D["c"][m] - D["c"][m].mean()); ls.append(D["lS"][m] - D["lS"][m].mean()); qs.append(D["lQ"][m] - D["lQ"][m].mean())
    cs, ls, qs = map(np.concatenate, (cs, ls, qs))
    kk = np.linalg.lstsq(np.c_[cs, qs], ls, rcond=None)[0][0]
    rho = np.corrcoef(cs, ls)[0, 1]
    tau = 1.0 / (abs(kk) * rate) if rate > 0 else np.nan
    # lag of peak response: first lag where cumulative b reaches 50%
    cb = np.cumsum(be)
    lag50 = int(np.argmax(cb >= 0.5 * cb[-1])) + 1
    loop[r] = dict(sum_b=float(be.sum()), sum_b_se=float(np.sqrt(cov[M_ARX + 1:, M_ARX + 1:].sum())), sum_a=float(ac.sum()),
                   c_rate_deg_per_s_per_ln=float(rate), k_shot=float(kk), rho_shot=float(rho), tau_closed_s=float(tau),
                   lag50_shots=lag50)
# BLEN control error per step (loop ON): step-mean ln S scatter vs shot-noise expectation
for r in RUNS:
    X = T[r]
    sm = X["e"]
    loop.setdefault(r, {})
    loop[r].update(blen_shot_cv=float(np.nanmedian(X["esd"])),
                   blen_step_mean_rms=float(robust_std(sm)),
                   blen_step_mean_expected=float(np.nanmedian(X["esd"] / np.sqrt(np.maximum(X["n"], 1)))))
S["loop"] = loop

# ------------------------------------------------------------- 4. gain g vs operating point
gain = {}
for r in RUNS:
    D = R[r]
    cs, ls, qs = [], [], []
    for s in np.unique(D["st"]):
        m = (D["st"] == s) & np.isfinite(D["c"]) & np.isfinite(D["lS"]) & np.isfinite(D["lQ"])
        if m.sum() < 8:
            continue
        cs.append(D["c"][m] - D["c"][m].mean()); ls.append(D["lS"][m] - D["lS"][m].mean()); qs.append(D["lQ"][m] - D["lQ"][m].mean())
    cs, ls, qs = map(np.concatenate, (cs, ls, qs))
    A = np.c_[cs, qs]
    cf, *_ = np.linalg.lstsq(A, ls, rcond=None)
    res = ls - A @ cf
    se = np.sqrt(res.var() * np.linalg.inv(A.T @ A)[0, 0])
    rho = np.corrcoef(cs, ls)[0, 1]
    # bootstrap over steps would be better; report OLS SE and the errors-in-variables bracket
    gain[r] = dict(S_set=float(np.exp(np.nanmedian(D["lS"]))), pact=float(np.nanmean(D["c"])), k=float(cf[0]), k_se=float(se),
                   dlnS_dlnQ=float(cf[1]), rho=float(rho), g_shot=float(-1 / cf[0]), g_lower_reverse=float(-rho ** 2 / cf[0]),
                   day="04-15" if r in E300 else "05-01")
# across-run: lnS_set vs PACT per day
xr = {}
for day, rr in (("04-15", E300), ("05-01", E331)):
    xx = np.array([gain[r]["pact"] for r in rr]); yy = np.log([gain[r]["S_set"] for r in rr])
    k, kse, _ = ols(xx, yy)
    xr[day] = dict(k=k, k_se=kse, g=-1 / k)
xx = np.array([gain[r]["pact"] for r in RUNS]); yy = np.log([gain[r]["S_set"] for r in RUNS])
dd = np.array([0.0 if r in E300 else 1.0 for r in RUNS])
A = np.c_[np.ones_like(xx), dd, xx]
cf, *_ = np.linalg.lstsq(A, yy, rcond=None)
xr["pooled_day_offset"] = dict(k=float(cf[2]), g=float(-1 / cf[2]))
gs = np.array([gain[r]["g_shot"] for r in RUNS]); lset = np.log([gain[r]["S_set"] for r in RUNS])
kg, kgse, g0 = ols(lset - np.log(10000), gs)
S["gain"] = dict(per_run=gain, across_run=xr, g_shot_vs_lnSset=dict(slope_per_ln=kg, se=kgse, g_at_10k=g0))

# ------------------------------------------------------------- 5. L2 readback offsets + BLEN14 beam side
l2 = {}
off_series = {}
for sec in ("LI12", "LI13", "LI14"):
    allD, alldt = [], []
    per = {}
    for r in RUNS:
        tt, oo, pp = l2_offsets(r, sec)
        oo = oo - GAM[sec] * (pp - np.mean(pp)) if len(pp) else oo   # remove readback-gain (PHAS/PDES != 1) term
        off_series[(r, sec)] = (tt, oo, pp)
        if len(tt) < 5:
            continue
        dlt = np.diff(oo); dtt = np.diff(tt)
        allD.append(dlt); alldt.append(dtt)
        tmin = (tt - tt[0]) / 60
        ramp, rse, _ = ols(tmin, oo)
        sf = structure_fit(tmin, oo - ramp * tmin, tmax=0.6 * tmin[-1], ramp=False, nb=8)
        per[r] = dict(n=len(tt), mean=float(np.mean(oo)), rms=float(np.std(oo)), ramp_deg_per_min=float(ramp),
                      rw_q=float(sf["q"]), white=float(np.sqrt(sf["sw2"])), n_jump=int(np.sum(np.abs(dlt) > 0.5)),
                      minutes=float(tmin[-1]))
    dl = np.concatenate(allD); dtt = np.concatenate(alldt)
    big = np.abs(dl) > 0.5
    small = dl[~big]
    l2[sec] = dict(per_run=per, n_pairs=int(len(dl)), median_dt_s=float(np.median(dtt)),
                   jump_frac_per_pair=float(big.mean()), jump_rate_per_min=float(big.sum() / (dtt.sum() / 60)),
                   jump_abs_pcts_10_50_90=(np.percentile(np.abs(dl[big]), [10, 50, 90]).round(2).tolist() if big.any() else []),
                   jump_max=float(np.abs(dl).max()), small_change_rms=float(np.std(small)),
                   small_change_rms_per_sqrtmin=float(np.std(small) / np.sqrt(np.median(dtt) / 60)),
                   quantum_deg=float(np.median(np.abs(dl[np.abs(dl) > 1e-3])) if np.any(np.abs(dl) > 1e-3) else np.nan))
S["L2_readback"] = l2

# BLEN14 beam-side: shift of the lnS14(PDES) curve between E331 runs with the same BC11 setpoint (A earlier, B later).
# Model: y_B(P) = f_A(P - D - rho*lag(P)) [+ v]; beam-side L2-equivalent phase change between the visits = -(D + rho*lag).
from scipy.signal import savgol_filter


def shift_fit(rA, rB, sat=11.4):
    XA, XB = T[rA], T[rB]
    okA = np.isfinite(XA["lS14"]) & (XA["lS14"] < sat)
    okB = np.isfinite(XB["lS14"]) & (XB["lS14"] < sat)
    o = np.argsort(XA["P"][okA])
    PA, yA, tA = XA["P"][okA][o], savgol_filter(XA["lS14"][okA][o], 5, 2), XA["tm"][okA][o]
    PB, yB, tB = XB["P"][okB], XB["lS14"][okB], XB["tm"][okB]
    lagB = (tB - np.interp(PB, PA, tA)) / 60

    def cost(D, rho, vert):
        q = PB - D - rho * lagB
        m = (q > PA[0]) & (q < PA[-1])
        if m.sum() < 12:
            return np.inf, 0, 0
        r = yB[m] - np.interp(q[m], PA, yA)
        v = np.median(r) if vert else 0.0
        return np.median((r - v) ** 2), v, m.sum()

    Ds = np.arange(-8, 8.001, 0.02)
    c0 = np.array([cost(D, 0, False)[0] for D in Ds])
    D0 = Ds[np.argmin(c0)]
    c1 = np.array([cost(D, 0, True)[0] for D in Ds])
    D1 = Ds[np.argmin(c1)]
    v1 = cost(D1, 0, True)[1]
    best = (np.inf, 0, 0)
    for rho in np.arange(-0.3, 0.3001, 0.01):
        for D in np.arange(-3, 3.001, 0.05):
            cc = cost(D, rho, False)[0]
            if cc < best[0]:
                best = (cc, D, rho)
    # bootstrap the constant shift over B points (resample steps)
    rng = np.random.default_rng(0)
    bs = []
    for _ in range(100):
        ii = rng.integers(0, len(PB), len(PB))
        q0 = PB[ii]
        cs = []
        for D in Ds[::2]:
            q = q0 - D
            m = (q > PA[0]) & (q < PA[-1])
            cs.append(np.median((yB[ii][m] - np.interp(q[m], PA, yA)) ** 2) if m.sum() >= 12 else np.inf)
        bs.append(Ds[::2][int(np.argmin(cs))])
    return dict(lag_mean_min=float(np.mean(lagB)), lag_range=[float(lagB.min()), float(lagB.max())],
                D_const=float(D0), D_const_bs_sd=float(np.std(bs)), D_with_vert=float(D1), vert_ln=float(v1),
                D0_ramp=float(best[1]), rho_deg_per_min=float(best[2]),
                beam_drift_deg=float(-D0), beam_rate_deg_per_min=float(-D0 / np.mean(lagB)),
                dlnQ=float(np.nanmean(XB["lQ"]) - np.nanmean(XA["lQ"])),
                dir=("f" if R[rA]["sb"]["LI12"][1][0] < R[rA]["sb"]["LI12"][1][-1] else "b") + ("f" if R[rB]["sb"]["LI12"][1][0] < R[rB]["sb"]["LI12"][1][-1] else "b"))


pair_tab = {}
for g in groups:
    for ia in range(len(g)):
        for ib in range(ia + 1, len(g)):
            pair_tab[f"{g[ia][-5:]}-{g[ib][-5:]}"] = shift_fit(g[ia], g[ib])
adj = [v for k, v in pair_tab.items() if int(k[-5:]) - int(k[:5]) <= 2]
S["L2_beam_BLEN14"] = dict(pairs=pair_tab,
                           adjacent_rate_median=float(np.median([v["beam_rate_deg_per_min"] for v in adj])),
                           adjacent_rate_range=[float(min(v["beam_rate_deg_per_min"] for v in adj)), float(max(v["beam_rate_deg_per_min"] for v in adj))],
                           note="D>0: run B needs higher PDES for the same BC14 BLEN -> beam-side L2-equivalent phase fell by D")

# BLEN14 within-run step-to-step residual (beam-side jump check)
jr = []
from scipy.signal import savgol_filter
for r in E331:
    pd = np.array([np.nanmedian(R[r]["sb"]["LI12"][1][R[r]["st"] == s]) for s in np.unique(R[r]["st"])])
    y = T[r]["lS14"]
    ok = np.isfinite(y)
    ys = savgol_filter(y[ok], 9, 2)
    sl = np.gradient(ys, pd[ok])
    res = (y[ok] - ys)
    m = np.abs(sl) > 0.15
    jr.append(np.c_[res[m] / sl[m], res[m]])
jr = np.concatenate(jr)
S["BLEN14_within_run_resid"] = dict(rms_equiv_deg=float(np.std(jr[:, 0])), robust_equiv_deg=float(robust_std(jr[:, 0])),
                                    frac_gt_0p5deg=float(np.mean(np.abs(jr[:, 0]) > 0.5)), rms_ln=float(np.std(jr[:, 1])))

# ------------------------------------------------------------------------------------ 6. charge
chg = {}
for r in RUNS:
    D, X = R[r], T[r]
    tmin = (X["tm"] - X["tm"][0]) / 60
    k_, kse_, _ = ols(tmin, X["lQ"])
    sf = structure_fit(tmin, X["lQ"] - k_ * tmin, tmax=0.6 * tmin[-1], ramp=False, nb=8)
    chg[r] = dict(Q_nC=float(np.nanmedian(np.exp(D["lQ"])) * 1.602e-19 * 1e9), shot_rms_pct=float(100 * np.nanmedian(X["qsd"])),
                  step_mean_rms_pct=float(100 * robust_std(np.diff(X["lQ"])) / np.sqrt(2)),
                  ramp_pct_per_min=float(100 * k_), span_pct=float(100 * (np.nanmax(X["lQ"]) - np.nanmin(X["lQ"]))),
                  rw_pct_per_sqrtmin=float(100 * np.sqrt(sf["q"])),
                  li14_890_valid=float(np.mean(np.isfinite(D["Q14"]) & (D["Q14"] > 0))) if D["Q14"] is not None else 0.0)
S["charge"] = chg

# ------------------------------------------------------------------------- aggregate + recommendations
tstep = np.array([timing[r]["period_s"] for r in E300])
ctrl_s = T_CTRL
mps = ctrl_s / 60.0
l1r = np.array([l1[r]["ramp_deg_per_min"] for r in RUNS])
l1q = np.array([l1[r]["rw_q_deg2_per_min_detrended"] for r in RUNS])
l1w = np.array([l1[r]["white_step_deg"] for r in RUNS])
taus = np.array([loop[r]["tau_closed_s"] for r in loop if "tau_closed_s" in loop[r]])
adjk = ["16019-16021", "16021-16022", "16023-16024", "16025-16027", "16028-16029", "16030-16032", "16032-16033", "16033-16034", "16034-16035"]
PT = S["L2_beam_BLEN14"]["pairs"]
lagA = np.array([v["lag_mean_min"] for v in PT.values()]); DA = np.array([v["beam_drift_deg"] for v in PT.values()])
rhoL2 = float(np.sum(DA * lagA) / np.sum(lagA ** 2))
la = np.array([PT[k]["lag_mean_min"] for k in adjk]); da = np.array([PT[k]["beam_drift_deg"] for k in adjk])
qL2 = float(np.mean((da - rhoL2 * la) ** 2 / la))
agg = dict(
    ctrl_step_s=ctrl_s, daq_step_s_E300=float(tstep.mean()),
    L1_ramp_deg_per_min=dict(all=l1r.round(3).tolist(), rms=float(np.sqrt(np.mean(l1r ** 2))), abs_median=float(np.median(np.abs(l1r))), abs_max=float(np.abs(l1r).max()),
                             E331_rms=float(np.sqrt(np.mean(np.array([l1[r]["ramp_deg_per_min"] for r in E331]) ** 2))),
                             E300=[l1[r]["ramp_deg_per_min"] for r in E300]),
    L1_rw_deg_per_sqrtmin=dict(median=float(np.sqrt(np.median(l1q))), p16=float(np.sqrt(np.percentile(l1q, 16))), p84=float(np.sqrt(np.percentile(l1q, 84))),
                               long_lag_groups=[float(np.sqrt(v["q"])) for v in S["L1_long_structure"].values()]),
    L1_step_white_deg_loop_on=dict(median=float(np.median(l1w)), range=[float(l1w.min()), float(l1w.max())]),
    L1_fast_jitter_shot_deg=[float(min(l1[r]["fast_jitter_deg"] for r in RUNS)), float(max(l1[r]["fast_jitter_deg"] for r in RUNS))],
    loop_tau_s=dict(values=taus.round(2).tolist(), range=[float(taus.min()), float(taus.max())]),
    L2_beam_ramp_deg_per_min=rhoL2, L2_beam_rw_resid_deg_per_sqrtmin=float(np.sqrt(qL2)),
)
S["aggregate"] = agg
step = lambda v: round(float(v) * mps, 4)
rstep = lambda v: round(float(v) * np.sqrt(mps), 4)
rec = {
    "_meta": dict(source="results/realism_2026-10-17/realism.py", ctrl_step_s=ctrl_s,
                  note="per-step = per controller step of ~20 s (6.5 s settle + ~130 shots at 10 Hz for 30 valid images per TCAV sign). deg = physical RF degrees."),
    "drift_per_step": {"L1PhaseSet": dict(current=0.025, daq=rstep(agg["L1_rw_deg_per_sqrtmin"]["median"]),
                                          daq_range=[rstep(agg["L1_rw_deg_per_sqrtmin"]["p16"]), rstep(agg["L1_rw_deg_per_sqrtmin"]["p84"])],
                                          recommended=0.035, stress=0.07, basis="BC11-loop slow correction (loop ON), RW part after per-run ramp + L2-scan coupling removed; loop-off upper bound"),
                       "L2PhaseSet": dict(current=0.05, daq_upper=rstep(agg["L2_beam_rw_resid_deg_per_sqrtmin"]), recommended=0.05, stress=0.1,
                                          basis="BLEN_LI14 revisit residual about the ramp (incl. measurement error) -> upper bound; readback RW ~0")},
    "drift_ramp": {"L1PhaseSet": dict(current="U(+-0.05)", daq_rms=step(agg["L1_ramp_deg_per_min"]["rms"]), daq_abs_max=step(agg["L1_ramp_deg_per_min"]["abs_max"]),
                                      recommended="U(+-0.05)", stress="U(+-0.1)"),
                   "L2PhaseSet": dict(current="none", daq=step(rhoL2), recommended="U(+-0.04)", stress="U(+-0.07)",
                                      basis="BLEN_LI14 curve shift between E331 runs, -0.094 deg/min sustained >1 h on 05-01 (one day; sign day-dependent)")},
    "L2_jumps": dict(current="none",
                     readback=dict(rate_per_ctrl_step_per_sector={k: round(l2[k]["jump_rate_per_min"] * mps, 4) for k in l2},
                                   size_deg="persistent +0.7..+0.8 (max +1.8) level shifts on one sector's SBST readback, always positive; none on LI14"),
                     recommended=dict(p_per_step=0.02, size_deg_on_L2PhaseSet="sign*U(0.2, 0.6) (one of ~3 sectors jumps 0.7-1.8 deg)", visible_on_readback=True,
                                      beam_side_check="BLEN_LI14 within-run residual >0.5 deg-equiv on %.1f%% of steps" % (100 * S["BLEN14_within_run_resid"]["frac_gt_0p5deg"])),
                     stress=dict(p_per_step=0.05, size_deg_on_L2PhaseSet="sign*U(0.2, 1.0)")),
    "blen_dlog": dict(gain_range=dict(current=[1.5, 4.0], recommended=[2.5, 5.0], stress=[1.5, 7.0], nominal=3.4),
                      gain_state_dependence="g = g0 * (1 + a*ln(S_set/10k)), a ~ U(0, 0.5) (shot-level: +0.3/ln on 05-01, ~0 on 04-15)",
                      loop_tau_s=agg["loop_tau_s"]["range"], settle="complete within the 6.5 s settle (residual <~5% of commanded change over the image window); model as instant",
                      readback_noise=dict(current=0.02, recommended=0.1, basis="loop-ON common-mode PACT step-mean white scatter %.2f-%.2f deg" % tuple(agg["L1_step_white_deg_loop_on"]["range"])),
                      readback_drift=dict(current=0.04, recommended=0.035, basis="= L1 RW (the loop's correction is the beam drift)"),
                      L2_to_L1_coupling_deg_per_deg=dict(value=S_PACT, p16_p84=S["scan_coupling"]["PACT_s_p16_p84"], note="loop-ON PACT moves +0.034 deg per deg of L2 MKB; mechanism unknown; optional")),
    "charge": dict(shot_rms_pct=[min(v["shot_rms_pct"] for v in chg.values()), max(v["shot_rms_pct"] for v in chg.values())],
                   step_mean_rms_pct=[min(v["step_mean_rms_pct"] for v in chg.values()), max(v["step_mean_rms_pct"] for v in chg.values())],
                   ramp_pct_per_min_abs_median=float(np.median([abs(v["ramp_pct_per_min"]) for v in chg.values()])),
                   recommended=dict(per_step_white_pct=0.6, ramp_pct_per_step="U(+-0.03)", dlnS11_dlnQ=[0.4, 2.5])),
}
(HERE / "recommended_params.json").write_text(json.dumps(_clean_rec(rec) if False else rec, indent=1, default=float))

# ---------------------------------------------------------------------------- write summary + figs
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, np.integer):
        return int(o)
    return o


(HERE / "summary.json").write_text(json.dumps(_clean(S), indent=1))

# Fig 1: PACT slow drift traces
fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for r in RUNS:
    X = T[r]
    ax = axs[0] if r in E300 else axs[1]
    tm = (X["tm"] - X["tm"][0]) / 60
    x = X["c"] - X["e"] / K_NOM - S_PACT * (X["P"] - np.nanmean(X["P"]))
    x = x - np.nanmean(x[:5])
    # 45 s moving average
    sm = np.array([np.nanmean(x[np.abs(X["tm"] - X["tm"][i]) <= 22.5]) for i in range(len(x))])
    ax.plot(tm, sm, lw=1.2, label=r[-5:])
for ax, ttl in zip(axs, ("2026-04-15 (E300, 150 shots/step)", "2026-05-01 (E331, 20-50 shots/step)")):
    ax.set_title(ttl, fontsize=9); ax.set_xlabel("time in run [min]"); ax.grid(alpha=.3)
    ax.legend(fontsize=6, ncol=2)
axs[0].set_ylabel("L1 loop correction - start [deg]")
fig.suptitle("Common-mode L1 FAST_PACT, 45 s avg, L2-scan-coupled part (%.3f deg/deg) removed" % S_PACT, fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "fig1_pact_slow_drift.png", dpi=130); plt.close(fig)

# Fig 2: L2 readback offsets + BLEN14 beam-side drift
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for k_, sec in enumerate(("LI12", "LI13", "LI14")):
    ax = axs[0] if sec != "LI14" else axs[1]
    for r in RUNS:
        tt, oo = off_series[(r, sec)][:2]
        if len(tt) == 0:
            continue
        if r in E300:
            continue
        tday = (tt - t0_day[E331[0]]) / 60
        dl = np.diff(oo)
        j = np.where(np.abs(dl) > 0.5)[0]
        ax.plot(tday, oo, ".-", ms=2, lw=.5, color=f"C{k_}", label=sec if r == E331[0] else None)
        ax.plot(tday[j + 1], oo[j + 1], "kx", ms=4)
axs[0].set_title("SBST PHAS-PDES (readback), LI12/LI13; x = jump >0.5 deg", fontsize=8)
axs[1].set_title("LI14 SBST PHAS-PDES (readback, energy-loop trims)", fontsize=8)
for ax in axs[:2]:
    ax.set_xlabel("min since E331_16019 start (2026-05-01)"); ax.set_ylabel("PHAS - PDES - gain term [deg]"); ax.grid(alpha=.3); ax.legend(fontsize=7)
ax = axs[2]
for k_, v in pair_tab.items():
    ax.errorbar(v["lag_mean_min"], v["beam_drift_deg"], yerr=v["D_const_bs_sd"], xerr=[[v["lag_mean_min"] - v["lag_range"][0]], [v["lag_range"][1] - v["lag_mean_min"]]],
                fmt="o", ms=4, capsize=2, alpha=.8)
    ax.annotate(k_.replace("E331_", ""), (v["lag_mean_min"], v["beam_drift_deg"]), fontsize=5)
tl = np.linspace(0, 50, 50)
ax.plot(tl, -0.094 * tl, "k--", lw=.8, label="fit through 0: -0.094 deg/min")
ax.set_xlabel("time between visits of the same PDES [min]"); ax.set_ylabel("beam-side L2-equivalent phase change [deg]")
ax.set_title("Beam side: shift of lnS14(PDES) between E331 runs (same BC11 setpoint)", fontsize=8)
ax.grid(alpha=.3); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(HERE / "fig2_l2_offsets_jumps.png", dpi=130); plt.close(fig)

# Fig 3: gain vs operating point
fig, axs = plt.subplots(1, 2, figsize=(10, 4))
for r in RUNS:
    G = gain[r]
    col = "C0" if G["day"] == "04-15" else "C1"
    axs[0].errorbar(G["S_set"] / 1e3, G["g_shot"], yerr=G["k_se"] / G["k"] ** 2, fmt="o", color=col, ms=4)
    axs[0].plot([G["S_set"] / 1e3] * 2, [G["g_lower_reverse"], G["g_shot"]], color=col, alpha=.3, lw=4)
    axs[1].plot(G["pact"], np.log(G["S_set"]), "o", color=col)
for day, col in (("04-15", "C0"), ("05-01", "C1")):
    axs[1].plot([], [], "o", color=col, label=f"{day}: across-run g={xr[day]['g']:.2f} deg/ln")
axs[0].axhspan(1.5, 4.0, color="gray", alpha=.15, label="env U(1.5, 4)")
axs[0].set_xlabel("BC11 BLEN setpoint [k counts]"); axs[0].set_ylabel("g = -1/(d lnS/d phi) [deg/ln]")
axs[0].set_title("shot-level g (dot = upper end; bar to reverse-regression lower end)", fontsize=8)
axs[0].legend(fontsize=7); axs[0].grid(alpha=.3)
axs[1].set_xlabel("mean common-mode PACT [deg]"); axs[1].set_ylabel("ln(BC11 setpoint)"); axs[1].legend(fontsize=7); axs[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(HERE / "fig3_bc11_gain_vs_op.png", dpi=130); plt.close(fig)
print(json.dumps(_clean({k: S[k] for k in ("E331_setpoint_groups", "L1_long_structure", "BLEN14_within_run_resid")}), indent=0))
print("done")
