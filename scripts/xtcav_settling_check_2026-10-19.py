"""Settling inside the DAQ steps (2026-10-19): after an L2 scan-step change, do beam-side readbacks and the DTOTR2 image
settle at the same rate, and is anything still moving when the DAQ starts recording?

Data: XTCAV_analysis/daq2npz/out/<run>.npz (per-shot, 10 Hz, pulse-ID aligned). The stored L2 'readback' is the
setpoint (nominal + scan value), and the L2 klystrons have no phase readback in the BSA set, so L2's effect is seen
only through the beam: BLEN_LI14 (BC14 bunch length), BC14 / BC20 dispersive BPMs, BLEN_LI20, the DTOTR2 centroid
(x = streak / time axis, y = energy axis; split by TCAV sign). Also the L1 klystron phases (the BC11 loop) and the
TCAV phase / amplitude readbacks (which flip between +-90 deg inside each step).

Per boundary and channel: pre = median of the last 40 shots of the previous step, post = median of a LATE reference
window of the new step that is disjoint from the tested shots (E300, 150 shots/step: shots 75+; E331, 20 shots/step:
shots 14+); normalized response r_k = (x_k - pre) / (post - pre). Boundaries count only where |post - pre| > 6 x the
per-shot robust sigma / sqrt(n_post) (a clearly resolved step; the remaining selection bias acts equally on every
tested k, so early-vs-mid comparisons are fair: settled data give r_early = r_mid). The DTOTR2 centroid uses good images of one TCAV class (+1, -1, 0) at a time, so the TCAV flips do not enter.
Outputs XTCAV_analysis/sim2real/results/settling_2026-10-19/{summary.json, response.png}.

  python scripts/xtcav_settling_check_2026-10-19.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("/home/rwu4/photoinjector-rl/XTCAV_analysis")
OUT = ROOT / "sim2real/results/settling_2026-10-19"
RUNS = ["E300_15671", "E300_15673"] + [f"E331_{r}" for r in (16019, 16021, 16022, 16023, 16024, 16025, 16027, 16028,
                                                             16029, 16030, 16032, 16033, 16034, 16035)]
CH = ["BLEN_LI14_888_BRAW", "BPMS_LI14_801_X", "BLEN_LI20_3014_BRAW", "BPMS_LI20_2445_X", "BPMS_LI20_3156_X",
      "KLYS_LI11_11_FB_FAST_PACT", "KLYS_LI11_21_FB_FAST_PACT", "BLEN_LI11_359_BRAW"]
IMG = [("dtotr2_com_x_mm", c) for c in (1, -1, 0)] + [("dtotr2_com_y_mm", c) for c in (1, -1, 0)]
K = 40                                       # shots after the first recorded shot of a step


def rsig(x):
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) > 4 else np.nan


def responses(x, st, t, mask=None, post_from=75, kmax=60):
    """Per resolved boundary: (k offsets < post_from, normalized response, time gap to the first shot)."""
    out = []
    steps = np.unique(st)
    for s0, s1 in zip(steps[:-1], steps[1:]):
        a_all, b_all = np.flatnonzero(st == s0), np.flatnonzero(st == s1)
        b0 = b_all[0]
        k_all = b_all - b0
        a = a_all[mask[a_all]] if mask is not None else a_all
        okb = mask[b_all] if mask is not None else np.ones(len(b_all), bool)
        late = b_all[(k_all >= post_from) & okb]
        test = b_all[(k_all < min(post_from, kmax)) & okb]
        if len(a) < 8 or len(late) < 4 or len(test) < 3:
            continue
        pre_x = x[a[-40:]]
        pre, post = np.nanmedian(pre_x), np.nanmedian(x[late])
        sig = rsig(np.r_[pre_x - pre, x[late] - post])
        if not np.isfinite(sig) or abs(post - pre) < 6 * sig / np.sqrt(len(late)):
            continue
        out.append((test - b0, (x[test] - pre) / (post - pre), float(t[b0] - t[a_all[-1]])))
    return out


def summarize(resp, K):
    if not resp:
        return None
    R = np.full((len(resp), K), np.nan)
    for i, (k, r, _) in enumerate(resp):
        R[i, k] = r
    med = np.nanmedian(R, 0)
    rng = np.random.default_rng(0)
    bs = np.array([np.nanmedian(R[rng.integers(0, len(R), len(R))], 0) for _ in range(300)])
    e, m = slice(0, 3), slice(K // 2, K)
    diff = np.nanmedian(R[:, e], 1) - np.nanmedian(R[:, m], 1)          # per boundary: early minus mid
    bd = np.array([np.nanmedian(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(1000)])
    return {"n_boundaries": len(resp), "median_curve": med, "ci": np.nanpercentile(bs, [5, 95], 0),
            "early": float(np.nanmedian(R[:, e])), "mid": float(np.nanmedian(R[:, m])),
            "early_minus_mid": float(np.nanmedian(diff)), "early_minus_mid_ci90": np.nanpercentile(bd, [5, 95]).round(3).tolist(),
            "gap_s_median": float(np.median([g for *_, g in resp]))}


CAMPAIGNS = {"E300": dict(runs=RUNS[:2], post_from=75, K=60), "E331": dict(runs=RUNS[2:], post_from=14, K=14)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    keys = CH + [f"{a}|{c}" for a, c in IMG]
    summ, curves = {}, {}
    for camp, cfg in CAMPAIGNS.items():
        acc = {c: [] for c in keys}
        for run in cfg["runs"]:
            f = ROOT / "daq2npz/out" / f"{run}.npz"
            if not f.exists():
                continue
            z = np.load(f, allow_pickle=True)
            names = [str(n) for n in z["states_scalar_feature_names"]]
            S, st, t, tc, gi = z["states_scalar"], z["step"], z["slac_time"].astype(float), z["tcav_class"], z["good_image"]
            kw = dict(post_from=cfg["post_from"], kmax=cfg["K"])
            for c in CH:
                if c in names:
                    acc[c] += responses(S[:, names.index(c)].astype(float), st, t, **kw)
            for a, c in IMG:
                acc[f"{a}|{c}"] += responses(S[:, names.index(a)].astype(float), st, t, mask=(tc == c) & gi, **kw)
        for c, resp in acc.items():
            sm = summarize(resp, cfg["K"])
            key = f"{camp}|{c}"
            if sm is None:
                summ[key] = {"n_boundaries": 0}
                continue
            curves[key] = sm
            summ[key] = {k: v for k, v in sm.items() if k not in ("median_curve", "ci")}
    (OUT / "summary.json").write_text(json.dumps(summ, indent=1))
    for c, sm in summ.items():
        if sm["n_boundaries"]:
            print(f"{c:40s} n={sm['n_boundaries']:3d}  early {sm['early']:.2f}  mid {sm['mid']:.2f}  "
                  f"early-mid {sm['early_minus_mid']:+.2f} {sm['early_minus_mid_ci90']}  gap {sm['gap_s_median']:.1f}s")
    fig, axs = plt.subplots(2, 2, figsize=(14, 9), sharey=True)
    for i, camp in enumerate(CAMPAIGNS):
        K = CAMPAIGNS[camp]["K"]
        kk = np.arange(K) * 0.1
        for ax, ks_, title in ((axs[i, 0], CH, "beam-side BSA channels"), (axs[i, 1], [f"{a}|{c}" for a, c in IMG], "DTOTR2 centroid by TCAV class")):
            for c in ks_:
                key = f"{camp}|{c}"
                if key not in curves or curves[key]["n_boundaries"] < 8:
                    continue
                sm = curves[key]
                ax.plot(kk, sm["median_curve"], lw=1.5, label=f"{c} (n={sm['n_boundaries']})")
                ax.fill_between(kk, *sm["ci"], alpha=0.12)
            ax.axhline(1, color="k", lw=0.7)
            ax.axhline(0, color="k", lw=0.7, ls=":")
            ax.set_title(f"{camp}: {title}")
            ax.legend(fontsize=7)
            ax.set_ylim(-0.4, 1.6)
            ax.set_xlabel("time after the first recorded shot of the step [s]")
        axs[i, 0].set_ylabel("normalized response (0 = previous step, 1 = late in this step)")
    fig.tight_layout()
    fig.savefig(OUT / "response.png", dpi=120)


if __name__ == "__main__":
    main()
