"""Settling, v3: drift-robust lag regression. For each step k (needs k-1, k+1):
  d_k = mean(first window of step k) - mean(second half of step k)
  X_k = (S_{k-1} - S_{k+1}) / 2   (S = second-half mean; independent shots from d_k)
beta = cov(d, X) / var(X) = fraction of one step's change still missing in the first window
(0 = settled at first recorded shot; 1 = first window still at the previous step's value).
Bootstrap over steps (pooled over runs of a campaign). Also: event position statistics."""
import json
import warnings

import numpy as np

from common import AUD, E300, E331, load_mat

warnings.filterwarnings("ignore")
rng = np.random.default_rng(0)
PVS = ["BPMS_LI14_801_X", "BPMS_LI14_715_X", "BPMS_LI20_2445_X", "BLEN_LI20_3014_BRAW", "BLEN_LI11_359_BRAW",
       "BLEN_LI14_888_BRAW", "KLYS_LI11_11_FB_FAST_PACT", "KLYS_LI11_21_FB_FAST_PACT", "TORO_LI20_2452_TMIT",
       "BPMS_LI11_333_X"]
WIN = {"E300": [(0, 10), (0, 30), (10, 30), (30, 60)], "E331": [(0, 3), (0, 5), (5, 10)]}


def collect(runs, wins):
    res = {}
    for name in PVS:
        rows = {w: [] for w in wins}
        for r in runs:
            D = load_mat(r)
            st, pv = D["step"], D["pv"]
            if name not in pv or np.mean(np.isfinite(pv[name])) < 0.8:
                continue
            a = pv[name].copy()
            if "PACT" in name:
                a[np.abs(a - np.nanmedian(a)) > 5] = np.nan
            ks = np.unique(st)
            xs = {k: a[st == k] for k in ks}
            S = {k: np.nanmean(x[len(x) // 2:]) for k, x in xs.items()}
            for k in ks[1:-1]:
                X = 0.5 * (S[k - 1] - S[k + 1])
                for w in wins:
                    F = np.nanmean(xs[k][w[0]:w[1]])
                    if np.isfinite(F) and np.isfinite(X):
                        rows[w].append((F - S[k], X))
        out = {}
        for w, rw in rows.items():
            if len(rw) < 20:
                continue
            A = np.array(rw)
            def b(Ai):
                return np.cov(Ai[:, 0], Ai[:, 1])[0, 1] / np.var(Ai[:, 1])
            bs = [b(A[rng.integers(0, len(A), len(A))]) for _ in range(500)]
            out[f"{w[0]}-{w[1]}"] = {"beta": float(b(A)), "ci68": [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))],
                                     "n_steps": len(A)}
        res[name] = out
    return res


R = {"E300": collect(E300, WIN["E300"]), "E331": collect(E331, WIN["E331"])}
# event position: are >3 mm BC14 excursions concentrated at step start?
pos = {"E300": [], "E331": []}
for camp, runs in (("E300", E300), ("E331", E331)):
    for r in runs:
        D = load_mat(r)
        st, x = D["step"], D["pv"]["BPMS_LI14_801_X"]
        big = np.abs(x - np.nanmedian(x)) > 3.0
        for i in np.where(np.diff(np.r_[0, big.astype(int)]) == 1)[0]:
            idx = np.where(st == st[i])[0]
            pos[camp].append(int(i - idx[0]))
R["events_bc14_gt3mm"] = {c: {"n": len(p), "frac_in_first_2_shots": float(np.mean(np.array(p) < 2)) if p else None,
                             "positions": p} for c, p in pos.items()}
R["events_bc14_gt3mm"]["E331"]["uniform_expect_first2"] = 2 / 20.5
R["events_bc14_gt3mm"]["E300"]["uniform_expect_first2"] = 2 / 150.5
json.dump(R, open(AUD / "data" / "settle3.json", "w"), indent=1)
for c in ("E300", "E331"):
    print("==", c)
    for k, v in R[c].items():
        print("  %-28s " % k + "  ".join("%s: %.2f [%.2f,%.2f]" % (w, e["beta"], *e["ci68"]) for w, e in v.items()))
print(R["events_bc14_gt3mm"]["E300"]["n"], R["events_bc14_gt3mm"]["E300"]["frac_in_first_2_shots"],
      R["events_bc14_gt3mm"]["E331"]["n"], R["events_bc14_gt3mm"]["E331"]["frac_in_first_2_shots"])
