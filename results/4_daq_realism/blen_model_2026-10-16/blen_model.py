"""First BLEN (coherent-radiation bunch-length monitor) signal model from the injector_full campaign beams.

S_loc(band) = Q^2 * int |F(f)|^2 R_band(f) df   (flat source spectrum, smooth band-pass R; arbitrary units nC^2 THz)
F = longitudinal form factor of the charge-weighted particle z (both bunches; per bunch kept for insight), with the
macroparticle incoherent floor sum(w^2)/sum(w)^2 subtracted. Read-only on data/; writes only into this directory.

  nice -n 10 env OMP_NUM_THREADS=1 /home/rwu4/miniconda3/envs/bmad-qpad-dev/bin/python blen_model.py [--extract]
"""
import argparse, glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("/home/rwu4/photoinjector-rl/twobunch-s2e-rl/data/injector_full")
OUT = Path(__file__).resolve().parent
CACHE = OUT / "blen_cache.npz"
LOCS = ["BL11356", "BL14888", "BL20_4"]
C = 299792458.0
F_THZ = np.geomspace(0.02, 40.0, 320)                     # frequency grid [THz]
BANDS = {"lo 0.1-1 THz": (0.1, 1.0), "mid 0.3-3 THz": (0.3, 3.0), "hi 1-10 THz": (1.0, 10.0)}
BAND_KEYS = list(BANDS)
DEF = "mid 0.3-3 THz"                                      # default band
L1_0, L2_0 = -20.2889213421, -35.5447801603                # golden / operating point
INJ = ["laser_timing_ps", "GUNF:rf_field_scale", "GUNF:theta0_deg", "distgen:t_dist:dists:d2:avg_t:value",
       "drive_fraction", "distgen:total_charge:value", "L0AF_phase_offset_deg", "SOL10111:solenoid_field_scale",
       "distgen:r_dist:sigma_xy:value", "distgen:r_dist:truncation_radius:value", "distgen:start:MTE:value", "sigma_t_ps"]
SHORT = {"laser_timing_ps": "laser_t", "GUNF:rf_field_scale": "gunE", "GUNF:theta0_deg": "gun_ph",
         "distgen:t_dist:dists:d2:avg_t:value": "pulse_sep", "drive_fraction": "drive_frac",
         "distgen:total_charge:value": "Q", "L0AF_phase_offset_deg": "L0A_ph", "SOL10111:solenoid_field_scale": "sol",
         "distgen:r_dist:sigma_xy:value": "laser_r", "distgen:r_dist:truncation_radius:value": "laser_trunc",
         "distgen:start:MTE:value": "MTE", "sigma_t_ps": "laser_sig_t", "L1PhaseSet": "L1", "L2PhaseSet": "L2",
         "L1EnergyOffset": "L1E", "L2EnergyOffset": "L2E", "L0BPhaseSet": "L0B_ph", "L0BEnergyOffset": "L0B_E"}
UP11 = ["L1PhaseSet", "L1EnergyOffset", "L0BPhaseSet", "L0BEnergyOffset"] + INJ          # upstream of BC11
UP14 = UP11 + ["L2PhaseSet", "L2EnergyOffset"]
SPECK = ["sigma_z_rms_total", "sigma_z_rms_drive", "sigma_z_rms_witness", "bunchSpacing", "transmission_total"]


def resp(f, lo, hi, n=4):
    return 1.0 / (1 + (lo / f) ** (2 * n)) / (1 + (f / hi) ** (2 * n))


def formfactor2(z, w, k, dz=0.25e-6):
    """|F(k)|^2 of the weighted z distribution (histogram DFT), incoherent floor subtracted; also returns F."""
    if len(w) < 10 or w.sum() <= 0:
        return np.zeros(len(k), complex), 0.0
    zc = z - np.average(z, weights=w)
    edges = np.arange(zc.min() - dz, zc.max() + 2 * dz, dz)
    h, _ = np.histogram(zc, edges, weights=w)
    m = h > 0
    zz = 0.5 * (edges[1:] + edges[:-1])[m]
    hq = h[m] / w.sum()
    F = np.empty(len(k), complex)
    for i in range(0, len(k), 40):
        F[i:i + 40] = np.exp(1j * np.outer(k[i:i + 40], zz)) @ hq
    F *= np.sinc(k * dz / (2 * np.pi))  # undo bin smoothing (np.sinc(x)=sin(pi x)/(pi x))
    floor = np.sum(w ** 2) / np.sum(w) ** 2
    return F, floor


def select_rows(seed=0):
    rows = []
    for f in sorted(glob.glob(str(DATA / "sample_*.json"))):
        d = json.load(open(f))
        if d.get("success"):
            rows.append((d["idx"], d["linac_region"], d["block"], d["injector_point"]))
    rng = np.random.default_rng(seed)
    gold = [r[0] for r in rows if r[3] == "golden"]
    shared = [r[0] for r in rows if r[1] == "shared" and r[3] != "golden"]
    ctrl = [r[0] for r in rows if r[1] == "controller" and r[2] == "nested"]
    full = [r[0] for r in rows if r[1] == "full" and r[2] == "nested"]
    pick = gold + list(rng.choice(shared, 150, replace=False)) + list(rng.choice(ctrl, 200, replace=False)) \
        + list(rng.choice(full, 50, replace=False))
    return sorted(set(int(i) for i in pick))


def extract():
    from pmd_beamphysics import ParticleGroup
    idx = select_rows()
    k = 2 * np.pi * F_THZ * 1e12 / C
    df = np.gradient(F_THZ)
    Rb = np.array([resp(F_THZ, *BANDS[b]) for b in BAND_KEYS])
    nb, nl = len(BAND_KEYS), len(LOCS)
    rec = {"idx": [], "region": [], "block": [], "inj": [], "knobs": [], "specs": [],
           "S_tot": [], "S_drv": [], "S_wit": [], "spec_tot": []}
    for n, i in enumerate(idx):
        d = json.load(open(DATA / f"sample_{i:05d}.json"))
        St, Sd, Sw, spt = np.zeros((nl, nb)), np.zeros((nl, nb)), np.zeros((nl, nb)), np.zeros((nl, len(k)))
        for j, loc in enumerate(LOCS):
            P = ParticleGroup(str(DATA / f"sample_{i:05d}_{loc}.h5"))
            live = P.status == 1
            z, w = P.z[live], P.weight[live]
            u = np.unique(w)
            if j == 0:
                wd = u.max()                     # drive = +0.1% weight tag (both bunches alive at BC11)
            drv = np.isclose(w, wd, rtol=2e-4, atol=0)
            Q, Qd, Qw = w.sum() * 1e9, w[drv].sum() * 1e9, w[~drv].sum() * 1e9   # nC
            Ft, fl = formfactor2(z, w, k)
            Fd, fld = formfactor2(z[drv], w[drv], k)
            Fw, flw = formfactor2(z[~drv], w[~drv], k)
            p_t = Q ** 2 * (np.abs(Ft) ** 2 - fl)
            p_d = Qd ** 2 * (np.abs(Fd) ** 2 - fld)
            p_w = Qw ** 2 * (np.abs(Fw) ** 2 - flw)
            spt[j] = p_t
            St[j], Sd[j], Sw[j] = Rb @ (p_t * df), Rb @ (p_d * df), Rb @ (p_w * df)
        rec["idx"].append(i); rec["region"].append(d["linac_region"]); rec["block"].append(d["block"])
        rec["inj"].append(d["injector_point"])
        rec["knobs"].append([d["knobs"][kk] for kk in UP14])
        rec["specs"].append([[d["specs"][loc].get(s, np.nan) for s in SPECK] for loc in LOCS + ["XTCAVF"]])
        rec["S_tot"].append(St); rec["S_drv"].append(Sd); rec["S_wit"].append(Sw); rec["spec_tot"].append(spt)
        if n % 50 == 0:
            print(n, len(idx), i, np.round(np.log(St[:, 1]), 2), flush=True)
    np.savez_compressed(CACHE, **{kk: np.array(v) for kk, v in rec.items()}, f_thz=F_THZ, bands=np.array(BAND_KEYS),
                        knob_names=np.array(UP14), spec_names=np.array(SPECK), locs=np.array(LOCS + ["XTCAVF"]))
    print("cached", CACHE, len(idx))


def lsq(X, y):
    A = np.c_[np.ones(len(y)), X]
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ beta
    return beta, r, 1 - r.var() / y.var()


def ksmooth(x, y, grid, h=0.35):
    """Gaussian-kernel local-linear fit: value and slope on grid."""
    v, s = np.full(len(grid), np.nan), np.full(len(grid), np.nan)
    for i, g in enumerate(grid):
        w = np.exp(-0.5 * ((x - g) / h) ** 2)
        if w.sum() < 3:
            continue
        A = np.c_[np.ones_like(x), x - g] * np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
        v[i], s[i] = beta
    return v, s


def daq_points():
    d = json.load(open(OUT.parent / "blen_units_2026-10-14" / "blen_steps.json"))
    days = {"04-15": ([], []), "05-01": ([], [])}
    for r, v in d.items():
        day = "05-01" if r.startswith("E331") else "04-15"
        days[day][0].append(np.nanmedian(np.array(v["L1"], float)))
        days[day][1].append(np.log(np.nanmedian(np.array(v["BC11"], float))))
    return {k: (np.array(a), np.array(b_)) for k, (a, b_) in days.items()}


def analyse_bc11_daq(D, out, col, dL1, dL2, gold, box, lnS, SP):
    f = D["f_thz"]; df = np.gradient(f); spec = D["spec_tot"][:, 0, :]
    L1 = col["L1PhaseSet"]
    grid = np.linspace(-27.3, -18.7, 87)
    xg = L1[gold]

    def band_lnS(lo, hi, rows=gold):
        return np.log(np.clip(spec[rows] @ (resp(f, lo, hi) * df), 1e-30, None))

    curves, slopes, resid = {}, {}, {}
    for bk in BAND_KEYS:
        v, s = ksmooth(xg, band_lnS(*BANDS[bk]), grid)
        curves[bk], slopes[bk] = v, s
        resid[bk] = float(np.std(band_lnS(*BANDS[bk]) - np.interp(xg, grid, v)))
    daq = daq_points()
    best = None
    scan = []
    for lo in np.geomspace(0.05, 1.5, 15):
        for hi in np.geomspace(0.3, 15, 15):
            if hi < 1.5 * lo:
                continue
            v, _ = ksmooth(xg, band_lnS(lo, hi), grid)
            sse, offs = 0.0, {}
            for day, (pl, pv) in daq.items():
                m = np.interp(pl, grid, v)
                off = np.mean(m - pv)
                sse += np.sum((m - off - pv) ** 2); offs[day] = off
            scan.append((lo, hi, sse))
            if best is None or sse < best[2]:
                best = (lo, hi, sse, v, offs)
    lo, hi, sse, vbest, offs = best
    lab = f"{lo:.2f}-{hi:.1f} THz"
    _, sbest = ksmooth(xg, band_lnS(lo, hi), grid)
    curves[lab], slopes[lab] = vbest, sbest
    resid[lab] = float(np.std(band_lnS(lo, hi) - np.interp(xg, grid, vbest)))
    n_daq = sum(len(p[0]) for p in daq.values())
    # DAQ local slopes (secants between run medians, per day)
    daq_sec = {}
    for day, (pl, pv) in daq.items():
        o = np.argsort(pl)
        daq_sec[day] = float(np.polyfit(pl, pv, 1)[0])
    def at(bk, x):
        return float(np.interp(x, grid, slopes[bk]))
    out["bc11_vs_L1_kernel"] = {
        "slope_dlnS_dL1_at": {bk: {str(x): at(bk, x) for x in [-19.0, -20.29, -22.0, -24.0, -25.5, -26.5]} for bk in slopes},
        "peak_L1_deg": {bk: float(grid[np.nanargmax(curves[bk])]) for bk in curves},
        "rms_scatter_about_kernel_curve_golden_ln": resid,
        "daq_linear_slope_per_day": daq_sec,
        "best_daq_band": {"lo_THz": float(lo), "hi_THz": float(hi), "rms_ln_misfit": float(np.sqrt(sse / (n_daq - 2))),
                          "n_daq_runs": int(n_daq)},
        "mid_band_daq_rms_misfit": None,
    }
    # misfit of the three fixed bands for reference
    for bk in BAND_KEYS:
        ss = 0.0
        for day, (pl, pv) in daq.items():
            m = np.interp(pl, grid, curves[bk]); ss += np.sum((m - np.mean(m - pv) - pv) ** 2)
        out["bc11_vs_L1_kernel"].setdefault("fixed_band_daq_rms_misfit", {})[bk] = float(np.sqrt(ss / (n_daq - 2)))
    del out["bc11_vs_L1_kernel"]["mid_band_daq_rms_misfit"]
    # env fit with the DAQ-fitted band in the controller box (cubic in dL1 + injector terms)
    lnSb = band_lnS(lo, hi, rows=np.ones(len(L1), bool))
    inj_feats = ["drive_fraction", "distgen:total_charge:value", "laser_timing_ps", "GUNF:theta0_deg",
                 "sigma_t_ps", "distgen:t_dist:dists:d2:avg_t:value", "GUNF:rf_field_scale"]
    Xi = np.column_stack([(col[q] - 0.5 * (col[q].min() + col[q].max())) / (0.5 * (col[q].max() - col[q].min()))
                          for q in inj_feats])
    feats = ["dL1", "dL1^2", "dL1^3", "L1E_MeV", "L0B_ph", "L0B_E_MeV"]
    base = np.c_[dL1, dL1 ** 2, dL1 ** 3, col["L1EnergyOffset"] / 1e6, col["L0BPhaseSet"] + 15.0,
                 col["L0BEnergyOffset"] / 1e6 + 2.259313]
    m = box
    beta, r, R2 = lsq(np.c_[base, Xi][m], lnSb[m])
    names = ["a"] + feats + [SHORT[q] + "(unit=half box)" for q in inj_feats]
    out["env_fit_BC11_daqband_ctrlbox"] = {"band": lab, "coef": {n_: float(v) for n_, v in zip(names, beta)},
                                           "R2": float(R2), "resid_rms_ln": float(r.std()), "n": int(m.sum())}
    # hidden-gain test: is ln S14 informative when the realised L2 differs from the nominal action?
    rng = np.random.default_rng(1)
    sp_x, szd = np.abs(SP[:, 3, 3]) * 1e6, np.log(SP[:, 3, 1])
    bmid = BAND_KEYS.index(DEF)
    res = {}
    for gl, gh, osd in [(1.0, 1.0, 0.0), (0.35, 1.1, 0.0), (0.35, 1.1, 1.5)]:
        gain = rng.uniform(gl, gh, len(L1)); offset = rng.normal(0, osd, len(L1))
        nom2 = (dL2 - offset) / gain
        A = np.c_[dL1, nom2, dL1 ** 2, nom2 ** 2, dL1 * nom2][m]
        s14 = np.c_[lnS[m, 1, bmid], lnS[m, 1, bmid] ** 2, lnS[m, 1, bmid] * dL1[m]]
        key = f"L2 gain U({gl},{gh}), offset sd {osd} deg"
        res[key] = {}
        for tn, t in [("spacing_um", sp_x[m]), ("ln_sigz_drive", szd[m])]:
            _, r0, R0 = lsq(A, t); _, r1, R1 = lsq(np.c_[A, s14], t)
            _, r2, R2_ = lsq(np.c_[A, s14, lnSb[m]], t)
            res[key][tn] = {"R2_actions": float(R0), "R2_+lnS14": float(R1), "R2_+lnS14+lnS11": float(R2_),
                            "rms_actions": float(r0.std()), "rms_+lnS14": float(r1.std())}
    out["xtcav_info_hidden_L2_gain"] = res
    # implied L2 resolution of the BC14 BLEN: scatter about a smooth (L1,L2) fit / slope
    return {"grid": grid, "curves": curves, "slopes": slopes, "best_label": lab,
            "daq": {day: (pl, pv, offs[day], "s" if day == "04-15" else "D") for day, (pl, pv) in daq.items()}}


def analyse():
    D = np.load(CACHE, allow_pickle=False)
    K = D["knobs"]; kn = list(D["knob_names"]); S = D["S_tot"]; Sd = D["S_drv"]; Sw = D["S_wit"]
    SP = D["specs"]; reg = D["region"]; inj = D["inj"]
    col = {n: K[:, i] for i, n in enumerate(kn)}
    dL1, dL2 = col["L1PhaseSet"] - L1_0, col["L2PhaseSet"] - L2_0
    b = BAND_KEYS.index(DEF)
    out = {"n_rows": int(len(K)), "n_controller": int((reg == "controller").sum()),
           "n_golden_inj": int((inj == "golden").sum())}
    lnS = np.log(np.clip(S, 1e-30, None))
    gold = inj == "golden"
    ctrl = reg == "controller"
    box = (col["L1PhaseSet"] >= -26.5) & (col["L1PhaseSet"] <= -18.5) & (col["L2PhaseSet"] >= -41) & (col["L2PhaseSet"] <= -33)
    box &= np.all(np.isfinite(SP[:, 3, :4]), axis=1)          # controller box, both bunches alive at XTCAVF
    out["n_ctrlbox"] = int(box.sum())

    # ---------- (2) BC11: local slope in L1, and relation to sigma_z ----------
    def zs(x):
        return (x - x.mean()) / x.std()
    other11 = ["L1EnergyOffset", "L0BPhaseSet", "L0BEnergyOffset"]
    slopes = {}
    for bi, bk in enumerate(BAND_KEYS):
        for name, m in [("golden_inj", gold), ("all", np.ones(len(K), bool))]:
            X = np.c_[dL1, dL1 ** 2] if name == "golden_inj" else np.c_[dL1, dL1 ** 2]
            X = np.c_[X, np.column_stack([col[o] / np.std(col[o]) for o in other11])] if name == "golden_inj" else \
                np.c_[X, np.column_stack([zs(col[o]) for o in other11 + INJ])]
            beta, r, R2 = lsq(X[m], lnS[m, 0, bi])
            slopes[f"{bk}|{name}"] = {"b_per_deg": float(beta[1]), "c_per_deg2": float(beta[2]), "R2": float(R2),
                                      "resid_rms_ln": float(r.std()), "gain_deg_per_ln": float(-1 / beta[1]),
                                      "slope_at_-24": float(beta[1] + 2 * beta[2] * (-24 - L1_0))}
    out["bc11_L1_fit"] = slopes
    # relation to sigma_z at BC11 (golden injector rows, default band)
    sz_t, sz_d, sz_w = SP[:, 0, 0], SP[:, 0, 1], SP[:, 0, 2]
    rel = {}
    for nm, s in [("sigma_z_total", sz_t), ("sigma_z_drive", sz_d), ("sigma_z_witness", sz_w)]:
        for g, m in [("golden_inj", gold), ("all", np.ones(len(K), bool))]:
            beta, r, R2 = lsq(np.log(s[m]), lnS[m, 0, b])
            rel[f"{nm}|{g}"] = {"dlnS_dln_sigma": float(beta[1]), "R2": float(R2)}
        rel[f"{nm}|dln_sigma_dL1_golden"] = float(np.polyfit(dL1[gold], np.log(s[gold]), 1)[0])
    out["bc11_vs_sigma"] = rel
    frac_w11 = Sw[:, 0, :] / (Sd[:, 0, :] + Sw[:, 0, :])
    frac_w14 = Sw[:, 1, :] / (Sd[:, 1, :] + Sw[:, 1, :])
    out["witness_share_of_incoherent_sum"] = {bk: {"BC11_median": float(np.median(frac_w11[:, i])),
                                                   "BC14_median": float(np.median(frac_w14[:, i]))}
                                              for i, bk in enumerate(BAND_KEYS)}
    # monotonicity of BC11 in L1 over the golden rows: running binned median
    o = np.argsort(dL1[gold])
    edges = np.linspace(dL1.min(), dL1.max(), 12)
    med = [np.median(lnS[gold, 0, b][(dL1[gold] >= a) & (dL1[gold] < c)]) for a, c in zip(edges[:-1], edges[1:])
           if ((dL1[gold] >= a) & (dL1[gold] < c)).sum() > 3]
    out["bc11_binned_median_monotone"] = bool(np.all(np.diff(med) < 0) or np.all(np.diff(med) > 0))

    # ---------- (1) partial dependence (standardized linear coefficients over all rows) ----------
    pd_ = {}
    for j, (loc, feats) in enumerate([("BL11356", UP11), ("BL14888", UP14)]):
        X = np.column_stack([zs(col[f]) for f in feats])
        beta, r, R2 = lsq(np.c_[X, zs(col["L1PhaseSet"]) ** 2, zs(col["L2PhaseSet"]) ** 2,
                                zs(col["L1PhaseSet"]) * zs(col["L2PhaseSet"])], lnS[:, j, b])
        pd_[loc] = {"coef_per_1sd": {SHORT[f]: float(c) for f, c in zip(feats, beta[1:len(feats) + 1])},
                    "R2_quadratic": float(R2), "resid_rms_ln": float(r.std())}
    out["partial_dependence_mid_band"] = pd_

    # ---------- (3) BC14: L2 sensitivity, info on XTCAV spacing / drive sigma_z ----------
    sp_x, szd_x = np.abs(SP[:, 3, 3]) * 1e6, SP[:, 3, 1] * 1e6   # um
    bc14 = {}
    for bi, bk in enumerate(BAND_KEYS):
        m = gold & box
        beta, r, R2 = lsq(np.c_[dL1[m], dL2[m], dL1[m] ** 2, dL2[m] ** 2, dL1[m] * dL2[m]], lnS[m, 1, bi])
        bc14[bk] = {"dlnS14_dL2_at_op": float(beta[2]), "dlnS14_dL1_at_op": float(beta[1]), "R2": float(R2),
                    "resid_rms_ln": float(r.std()),
                    "corr_lnS14_spacing_ctrlbox": float(np.corrcoef(lnS[box, 1, bi], sp_x[box])[0, 1]),
                    "corr_lnS14_szdrive_ctrlbox": float(np.corrcoef(lnS[box, 1, bi], np.log(szd_x[box]))[0, 1])}
    out["bc14"] = bc14
    # information value: predict XTCAV spacing / ln sigma_z_drive in the controller box from
    # (A) known actions only, (B) + ln S14, (C) + ln S11, (D) + both; quadratic in actions
    m = box
    act = np.c_[dL1, dL2, dL1 ** 2, dL2 ** 2, dL1 * dL2][m]
    info = {}
    for tgt_name, tgt in [("spacing_um", sp_x[m]), ("ln_sigz_drive", np.log(szd_x[m]))]:
        res = {}
        for nm, X in [("actions", act), ("+lnS14", np.c_[act, lnS[m, 1, b], lnS[m, 1, b] ** 2]),
                      ("+lnS11", np.c_[act, lnS[m, 0, b]]),
                      ("+lnS11+lnS14", np.c_[act, lnS[m, 0, b], lnS[m, 1, b], lnS[m, 1, b] ** 2]),
                      ("+lnS11+lnS14(all bands)", np.c_[act, lnS[m, 0, :], lnS[m, 1, :]])]:
            beta, r, R2 = lsq(X, tgt)
            res[nm] = {"R2": float(R2), "resid_rms": float(r.std())}
        res["target_std"] = float(tgt.std())
        info[tgt_name] = res
    out["xtcav_info_ctrlbox"] = info
    # within-injector-point deltas (log-ratio obs): pairs of rows sharing an injector point
    pairs = []
    for ip in np.unique(inj):
        ii = np.where((inj == ip) & box)[0]
        for a in range(len(ii)):
            for c_ in range(a + 1, len(ii)):
                pairs.append((ii[a], ii[c_]))
    pa = np.array(pairs)
    dlr = lnS[pa[:, 1], 1, b] - lnS[pa[:, 0], 1, b]
    dsp = sp_x[pa[:, 1]] - sp_x[pa[:, 0]]
    dsz = np.log(szd_x[pa[:, 1]]) - np.log(szd_x[pa[:, 0]])
    da = np.c_[dL1[pa[:, 1]] - dL1[pa[:, 0]], dL2[pa[:, 1]] - dL2[pa[:, 0]]]
    out["logratio_pairs"] = {"n_pairs": int(len(pa)),
                             "corr_dlnS14_dspacing": float(np.corrcoef(dlr, dsp)[0, 1]),
                             "corr_dlnS14_dlnsigz_drive": float(np.corrcoef(dlr, dsz)[0, 1]),
                             "corr_dL2_dspacing": float(np.corrcoef(da[:, 1], dsp)[0, 1])}

    # ---------- (4) compact parametric fits for the env ----------
    inj_feats = ["drive_fraction", "distgen:total_charge:value", "laser_timing_ps", "GUNF:theta0_deg",
                 "sigma_t_ps", "distgen:t_dist:dists:d2:avg_t:value"]
    fits = {}
    lo_hi = {f: (np.min(col[f]), np.max(col[f])) for f in kn}
    for loc_j, loc, feats in [(0, "BC11", ["dL1", "dL1^2", "L1E_MeV", "L0B_ph", "L0B_E_MeV"]),
                              (1, "BC14", ["dL1", "dL2", "dL1^2", "dL2^2", "dL1*dL2", "L1E_MeV", "L2E_MeV"])]:
        base = {"dL1": dL1, "dL2": dL2, "dL1^2": dL1 ** 2, "dL2^2": dL2 ** 2, "dL1*dL2": dL1 * dL2,
                "L1E_MeV": col["L1EnergyOffset"] / 1e6, "L2E_MeV": col["L2EnergyOffset"] / 1e6,
                "L0B_ph": col["L0BPhaseSet"] + 15.0, "L0B_E_MeV": col["L0BEnergyOffset"] / 1e6 + 2.259313}
        Xi = np.column_stack([(col[f] - np.mean(lo_hi[f])) / (0.5 * (lo_hi[f][1] - lo_hi[f][0])) for f in inj_feats])
        X = np.column_stack([base[f] for f in feats] + [Xi])
        m = box
        beta, r, R2 = lsq(X[m], lnS[m, loc_j, b])
        names = ["a"] + feats + [SHORT[f] + "(unit=half box)" for f in inj_feats]
        fits[loc] = {"coef": {n_: float(v) for n_, v in zip(names, beta)}, "R2": float(R2),
                     "resid_rms_ln": float(r.std()), "n": int(m.sum())}
    out["env_fit_mid_band_ctrlbox"] = fits
    (OUT / "blen_model_summary.json").write_text(json.dumps(out, indent=1))

    # ======================= figures =======================
    plt.rcParams.update({"font.size": 9})
    # Fig 1: spectra + band responses for the golden row(s) at BC11/BC14/BC20
    i0 = int(np.where(gold & (np.abs(dL1) < 0.3) & (np.abs(dL2) < 0.5))[0][0]) if (gold & (np.abs(dL1) < 0.3) & (np.abs(dL2) < 0.5)).any() else int(np.where(gold)[0][0])
    ilo = int(np.where(gold)[0][np.argmax(dL1[gold])]); ihi = int(np.where(gold)[0][np.argmin(dL1[gold])])
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, loc in enumerate(LOCS):
        for ii, lab, ls in [(i0, f"L1={col['L1PhaseSet'][i0]:.1f}°", "-"), (ilo, f"L1={col['L1PhaseSet'][ilo]:.1f}°", ":"),
                            (ihi, f"L1={col['L1PhaseSet'][ihi]:.1f}°", "--")]:
            ax[j].loglog(F_THZ, np.clip(D["spec_tot"][ii, j], 1e-6, None), ls, lw=1.2, label=lab)
        a2 = ax[j].twinx()
        for bk, cc in zip(BAND_KEYS, ["0.6", "0.35", "0.1"]):
            a2.semilogx(F_THZ, resp(F_THZ, *BANDS[bk]), color=cc, lw=0.8, alpha=0.6)
        a2.set_ylim(0, 4); a2.set_yticks([0, 1]); a2.set_ylabel("R(f) (3 bands)", fontsize=8)
        ax[j].set(xlabel="f [THz]", ylabel=r"$Q^2|F(f)|^2$ [nC$^2$]", title=f"{loc} (golden injector)", ylim=(1e-5, 5))
        ax[j].legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(OUT / "fig1_formfactor_spectra.png", dpi=130); plt.close(fig)

    # Fig 2: BC11 BLEN vs L1 (3 bands + DAQ-fitted band), local slope, vs sigma_z
    X2 = analyse_bc11_daq(D, out, col, dL1, dL2, gold, box, lnS, SP)
    (OUT / "blen_model_summary.json").write_text(json.dumps(out, indent=1))
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    g = X2["grid"]
    for bi, (bk, cc) in enumerate(zip(BAND_KEYS, ["C0", "C1", "C2"])):
        ref = X2["curves"][bk][np.argmin(np.abs(g - L1_0))]
        ax[0].plot(col["L1PhaseSet"][~gold], lnS[~gold, 0, bi] - ref, ".", color=cc, ms=2, alpha=0.2)
        ax[0].plot(col["L1PhaseSet"][gold], lnS[gold, 0, bi] - ref, "o", color=cc, ms=2.5, label=bk)
        ax[1].plot(g, -X2["slopes"][bk], color=cc, label=bk)
    bb = X2["best_label"]
    ref = X2["curves"][bb][np.argmin(np.abs(g - L1_0))]
    ax[0].plot(g, X2["curves"][bb] - ref, "k-", lw=1.5, label=f"DAQ-fit band {bb}")
    for day, (pl, pv, off, mk) in X2["daq"].items():
        ax[0].plot(pl, pv + off - ref, mk, color="r", mfc="none", ms=6, label=f"DAQ {day} setpoints (offset fit)")
    ax[1].plot(g, -X2["slopes"][bb], "k-", lw=1.5, label="DAQ-fit band")
    ax[0].axvline(L1_0, color="k", lw=0.6, ls=":")
    ax[0].set(xlabel="L1PhaseSet (sim) / L1 FAST_PACT (DAQ) [deg]", ylabel=r"$\ln S_{BC11}$ − ln S(op)",
              title="BC11 BLEN model vs L1 (o golden inj., · varied inj.)")
    ax[0].legend(fontsize=6.5)
    ax[1].axhspan(1 / 4.0, 1 / 1.5, color="0.85", label="env prior 1/g, g~U(1.5,4) deg/ln")
    ax[1].axhline(0.29, color="r", ls="--", lw=1, label="DAQ 04-15 (0.29/°)")
    ax[1].axhline(0.38, color="m", ls="--", lw=1, label="DAQ 05-01 (0.38/°)")
    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].axvspan(-26.5, -18.5, color="y", alpha=0.08)
    ax[1].set(xlabel="L1PhaseSet [deg]", ylabel=r"$-d\ln S_{BC11}/d\varphi_{L1}$ [1/deg]", ylim=(-0.6, 1.2),
              title="local slope (kernel fit, golden inj.; yellow = ctrl box)")
    ax[1].legend(fontsize=6.5)
    ax[2].loglog(sz_t[gold] * 1e6, S[gold, 0, b], "o", ms=3, label=r"$\sigma_{z}$ total")
    ax[2].loglog(sz_d[gold] * 1e6, S[gold, 0, b], "s", ms=3, label=r"$\sigma_{z}$ drive")
    ax[2].loglog(sz_w[gold] * 1e6, S[gold, 0, b], "^", ms=3, label=r"$\sigma_{z}$ witness")
    ax[2].set(xlabel=r"rms bunch length at BL11356 [$\mu$m]", ylabel=f"S_BC11 ({DEF}) [arb]", title="BC11 BLEN vs bunch length (golden inj.)")
    ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "fig2_bc11_vs_L1.png", dpi=130); plt.close(fig)

    # Fig 3: partial dependence bars (BC11, BC14)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for j, loc in enumerate(["BL11356", "BL14888"]):
        cc = pd_[loc]["coef_per_1sd"]
        names = sorted(cc, key=lambda n_: -abs(cc[n_]))
        ax[j].barh(names[::-1], [cc[n_] for n_ in names[::-1]], color=["C3" if cc[n_] < 0 else "C0" for n_ in names[::-1]])
        ax[j].set(xlabel=r"$\Delta\ln S$ per +1 SD of knob (linear term, all rows)",
                  title=f"{loc}  ({DEF}; quad fit R²={pd_[loc]['R2_quadratic']:.2f}, resid {pd_[loc]['resid_rms_ln']:.2f})")
    fig.tight_layout(); fig.savefig(OUT / "fig3_partial_dependence.png", dpi=130); plt.close(fig)

    # Fig 4: BC14 vs L2, and vs XTCAV spacing / drive sigma_z (controller box)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    for bi, (bk, cc) in enumerate(zip(BAND_KEYS, ["C0", "C1", "C2"])):
        m = gold & box
        ax[0].plot(col["L2PhaseSet"][m], lnS[m, 1, bi] - np.median(lnS[m, 1, bi]), "o", ms=3, color=cc, label=bk)
    ax[0].set(xlabel="L2PhaseSet [deg]", ylabel=r"$\ln S_{BC14}$ − median", title="BC14 BLEN vs L2 (golden inj., ctrl box; L1 varies)")
    ax[0].legend(fontsize=7)
    sc = ax[1].scatter(sp_x[box], lnS[box, 1, b], c=col["L2PhaseSet"][box], s=6, cmap="viridis")
    plt.colorbar(sc, ax=ax[1], label="L2 [deg]")
    ax[1].set(xlabel=r"XTCAVF bunch spacing [$\mu$m]", ylabel=r"$\ln S_{BC14}$ (mid band)",
              title=f"r = {bc14[DEF]['corr_lnS14_spacing_ctrlbox']:.2f} (controller box, all inj.)")
    sc = ax[2].scatter(szd_x[box], lnS[box, 1, b], c=col["L1PhaseSet"][box], s=6, cmap="plasma")
    plt.colorbar(sc, ax=ax[2], label="L1 [deg]")
    ax[2].set_xscale("log")
    ax[2].set(xlabel=r"XTCAVF drive $\sigma_z$ [$\mu$m]", ylabel=r"$\ln S_{BC14}$ (mid band)",
              title=f"r(ln σz) = {bc14[DEF]['corr_lnS14_szdrive_ctrlbox']:.2f}")
    fig.tight_layout(); fig.savefig(OUT / "fig4_bc14_vs_L2_xtcav.png", dpi=130); plt.close(fig)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    a = ap.parse_args()
    if a.extract or not CACHE.exists():
        extract()
    analyse()
