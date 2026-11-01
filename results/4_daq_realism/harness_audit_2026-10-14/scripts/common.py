"""Shared loaders for the harness audit (read-only on the DAQ .mat files and Tier 0 / re-extract outputs)."""
import glob
import json
from pathlib import Path

import numpy as np
import scipy.io

BASE = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
AUD = Path("/home/rwu4/photoinjector-rl/twobunch-s2e-rl/results/harness_audit_2026-10-14")
E300 = ["TEST_15668", "E300_15671", "E300_15673"]
E331 = ["E331_%d" % i for i in range(16019, 16036) if i not in (16020, 16026, 16031)]
D_NOM_MM = 59.31


def robust_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def load_mat(r):
    d = scipy.io.loadmat(BASE / r / f"{r}.mat", simplify_cells=True)["data_struct"]
    s, p = d["scalars"], d["pulseID"]
    n = len(np.atleast_1d(p["steps"]))
    out = {"t": np.asarray(p["SLAC_time"], float), "pid": np.asarray(p["scalar_PID"], float),
           "step": np.asarray(p["steps"], int), "params": d["params"],
           "common_index": np.asarray(s["common_index"], int), "pv": {}}
    for L, v in s.items():
        if not isinstance(v, dict):
            continue
        for pv, a in v.items():
            a = np.asarray(a, float).ravel()
            if len(a) != n:
                continue
            a = np.where(a == -10000.0, np.nan, a)
            if pv.startswith("BLEN"):
                a = np.where(a <= 0, np.nan, a)
            out["pv"][pv] = a
    return out


def tcav_class(A, P, amp_tol=0.25, ph_tol=1.0):
    """'+', '-', 'x' (amp ok, phase outside +-1 deg), 'a' (amp not settled), 'o' (off)."""
    on = A > 10
    Am = np.nanmean(A[on & ((np.abs(P - 90) < ph_tol) | (np.abs(P + 90) < ph_tol))])
    c = np.full(len(A), "o", dtype="<U1")
    c[on] = "a"
    ampok = on & (np.abs(A - Am) < amp_tol)
    c[ampok] = "x"
    c[ampok & (np.abs(P - 90) < ph_tol)] = "+"
    c[ampok & (np.abs(P + 90) < ph_tol)] = "-"
    c[(A <= 10) & (A > 5)] = "a"
    return c, Am


def load_reext(r):
    fs = sorted(glob.glob(str(AUD / "data" / f"reext_{r}_c*.npz")))
    parts = [np.load(f, allow_pickle=True) for f in fs]
    keys = [k for k in parts[0].files if k not in ("cal_scaled", "cal_um_per_deg", "n_off_per_step")]
    R = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    o = np.argsort(R["ind"])
    R = {k: v[o] for k, v in R.items()}
    R["cal_scaled"] = float(parts[0]["cal_scaled"])
    R["n_off_per_step"] = json.loads(str(parts[0]["n_off_per_step"]))
    return R
