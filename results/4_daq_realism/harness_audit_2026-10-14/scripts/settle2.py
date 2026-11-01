"""Settling, v2: excess deviation from the step's own settled value vs shot index within step.
D(n) = median over steps of |x_k(n) - T_k| / sigma_k, T_k / sigma_k = median / robust std of the last
half of step k. Stationary white noise gives D ~ 0.67 at every n. Also counts large excursions
(beam events) vs position within step, and times the relaxation. Writes data/settle2.json + fig2."""
import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import AUD, E300, E331, load_mat, robust_std

warnings.filterwarnings("ignore")
PVS = ["BPMS_LI14_801_X", "BPMS_LI20_2445_X", "BLEN_LI20_3014_BRAW", "BLEN_LI11_359_BRAW",
       "KLYS_LI11_11_FB_FAST_PACT", "TORO_LI20_2452_TMIT", "BPMS_LI11_333_X", "BPMS_IN10_731_X"]
out = {}
curves = {}
for r in E300 + E331:
    D = load_mat(r)
    st, pv = D["step"], D["pv"]
    ks = np.unique(st)
    nmax = int(np.median([np.sum(st == k) for k in ks]))
    out[r] = {}
    for name in PVS:
        if name not in pv or np.mean(np.isfinite(pv[name])) < 0.8:
            continue
        a = pv[name].copy()
        if "PACT" in name:
            a[np.abs(a - np.nanmedian(a)) > 5] = np.nan
        Z = np.full((len(ks), nmax), np.nan)
        for j, k in enumerate(ks):
            x = a[st == k][:nmax]
            tail = x[len(x) // 2:]
            T, s = np.nanmedian(tail), robust_std(tail)
            if not (s > 0):
                continue
            Z[j, :len(x)] = np.abs(x - T) / s
        Dn = np.nanmedian(Z, 0)
        big = np.nanmean(Z > 5, 0)            # fraction of steps with a >5 sigma deviation at shot n
        curves.setdefault(name, {})[r] = Dn
        nb = 10 if nmax >= 100 else 2
        first = np.nanmean(Z[:, :nb] > 5)
        rest = np.nanmean(Z[:, nmax // 2:] > 5)
        # settle index: first n after which the running D stays below 1.0 (from 3-shot medians)
        sm = np.array([np.nanmedian(Dn[max(0, i - 1):i + 2]) for i in range(nmax)])
        above = np.where(sm > 1.0)[0]
        settle_n = int(above.max() + 1) if len(above) and above.max() < nmax // 2 else (0 if not len(above) else None)
        out[r][name] = {"D_first5": float(np.nanmean(Dn[:5])), "D_5to10": float(np.nanmean(Dn[5:10])),
                        "D_10to20": float(np.nanmean(Dn[10:20])), "D_secondhalf": float(np.nanmean(Dn[nmax // 2:])),
                        "frac_gt5sig_first": float(first), "frac_gt5sig_secondhalf": float(rest),
                        "settle_shots_D_lt_1": settle_n}
# event rate: BC14 BPM excursions > 3 mm and LI20 toroid drops > 20 %, per hour of beam
ev = {}
for r in E300 + E331:
    D = load_mat(r)
    t, st, pv = D["t"], D["step"], D["pv"]
    x = pv.get("BPMS_LI14_801_X")
    q = pv.get("TORO_LI20_2452_TMIT")
    hrs = (len(t) * 0.1) / 3600
    big = np.abs(x - np.nanmedian(x)) > 3.0
    qd = q < 0.8 * np.nanmedian(q)
    # group into events (consecutive shots)
    def nevents(mask):
        m = mask.astype(int)
        return int(np.sum(np.diff(np.r_[0, m]) == 1))
    pos = []
    for i in np.where(np.diff(np.r_[0, big.astype(int)]) == 1)[0]:
        k = st[i]
        idx = np.where(st == k)[0]
        pos.append(int(i - idx[0]))
    ev[r] = {"beam_hours": hrs, "bc14_bpm_gt3mm_events": nevents(big), "toro_drop_gt20pct_events": nevents(qd),
             "shots_affected_frac": float(np.mean(big | qd)), "event_index_in_step": pos}
json.dump({"D": out, "events": ev}, open(AUD / "data" / "settle2.json", "w"), indent=1)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), dpi=130)
for ax, runs, ttl in ((axs[0], E300, "E300 (150 shots/step)"), (axs[1], E331, "E331 (20 shots/step)")):
    for name, col in (("BPMS_LI14_801_X", "C0"), ("BLEN_LI20_3014_BRAW", "C1"), ("BPMS_LI20_2445_X", "C2"),
                      ("KLYS_LI11_11_FB_FAST_PACT", "C3"), ("TORO_LI20_2452_TMIT", "C4"), ("BLEN_LI11_359_BRAW", "C5")):
        cs = [curves[name][r] for r in runs if r in curves.get(name, {})]
        if not cs:
            continue
        L = min(len(c) for c in cs)
        C = np.nanmedian(np.array([c[:L] for c in cs]), 0)
        ax.plot(np.arange(L) * 0.1, C, "-", color=col, lw=1.2, label=f"{name} ({len(cs)} runs)")
    ax.axhline(0.674, color="k", ls="--", lw=0.8, label="stationary white noise")
    ax.set_xlabel("time since first recorded shot of the step [s]  (preceded by a 6-7 s unrecorded gap)")
    ax.set_ylabel("median |x - settled| / shot sigma")
    ax.set_title(ttl, fontsize=9)
    ax.set_ylim(0, 3)
    ax.legend(fontsize=6.5)
axs[0].set_xlim(0, 8)
fig.tight_layout()
fig.savefig(AUD / "figs" / "fig2_settling.png")
for r in out:
    print(r, {k: (round(v["D_first5"], 2), round(v["D_5to10"], 2), round(v["D_10to20"], 2), round(v["D_secondhalf"], 2), v["settle_shots_D_lt_1"], round(v["frac_gt5sig_first"], 3), round(v["frac_gt5sig_secondhalf"], 3)) for k, v in out[r].items() if k in ("BPMS_LI14_801_X", "BLEN_LI20_3014_BRAW", "KLYS_LI11_11_FB_FAST_PACT")})
for r, e in ev.items():
    print(r, {k: v for k, v in e.items() if k != "event_index_in_step"}, e["event_index_in_step"][:12])
